#!/usr/bin/env python3
"""Data registry + instance builder for R2E-Gym rollout collection.

This is the importable DATA LAYER for the collection driver (collect.py). It turns
a registered dataset name into a list of rollout-ready instances (each with a
`docker_image`, `ip`, and the native fields R2E-Gym's docker.py needs).

ADDING A NEW DATASET is a one-liner: append a `register(DatasetSpec(...))` call
below. A spec says where the HF data lives, how to derive the docker image for a
row, and (optionally) any per-row preparation. Nothing else in the pipeline needs
to change — collect.py and collect_config.yaml pick it up by name automatically.

Registered datasets (public images; the private harbor mirror is NOT reachable
from a dev box):
  swegym    : xingyaoww/sweb.eval.x86_64.<iid __->_s_>   (SWE-Gym/SWE-Gym, 2438, 1 inst/image)
  swesmith  : jyangballin/swesmith.x86_64.* (row.image_name)  (SWE-bench/SWE-smith, ~59k; ~266 inst/image)
  swerebench: swerebench/sweb.eval.x86_64.* (nebius filtered) (nebius/SWE-rebench:filtered, 6542; 1 inst/image)

CLI (quick inspection):
  python build_dataset.py                 # list registered datasets
  python build_dataset.py swesmith 200    # counts + first 5 image groups
"""
import json
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

IP = "127.0.0.1"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to turn an HF dataset into rollout-ready instances.

    name      : registry key used on the CLI / in the config and as the output dir.
    hf_path   : HuggingFace dataset id.
    hf_split  : split to load.
    image_fn  : row(dict) -> docker_image str, or None to SKIP the row.
    prepare_fn: optional row(dict) -> row(dict) | None. Runs after image_fn sets
                docker_image; return None to skip. Use for dataset-specific work
                (e.g. swerebench's make_test_spec). Default: identity.
    """
    name: str
    hf_path: str
    hf_split: str
    image_fn: Callable[[dict], Optional[str]]
    prepare_fn: Optional[Callable[[dict], Optional[dict]]] = None


REGISTRY: "OrderedDict[str, DatasetSpec]" = OrderedDict()


def register(spec: DatasetSpec) -> None:
    REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Dataset registrations — ADD NEW DATASETS HERE (one register() call each).
# ---------------------------------------------------------------------------

# swegym: no image column; derive from instance_id (xingyaoww public mirror).
# Docker repo names MUST be lowercase, and the published images lowercase the
# instance_id (e.g. Project-MONAI__MONAI-6127 -> ...project-monai_s_monai-6127),
# so we .lower() the derived name. instance_id itself stays original-case (resume
# matches on it), only the docker_image is lowercased.
register(DatasetSpec(
    name="swegym",
    hf_path="SWE-Gym/SWE-Gym",
    hf_split="train",
    image_fn=lambda row: ("xingyaoww/sweb.eval.x86_64." + row["instance_id"].replace("__", "_s_")).lower(),
))

# swesmith: image is in the `image_name` column (jyangballin public images).
register(DatasetSpec(
    name="swesmith",
    hf_path="SWE-bench/SWE-smith",
    hf_split="train",
    image_fn=lambda row: row.get("image_name"),
))


def _swerebench_image(row: dict) -> Optional[str]:
    img = row.get("docker_image")
    return img if img and img.startswith("swerebench/") else None


def _swerebench_prepare(row: dict) -> Optional[dict]:
    # R2E-Gym's docker.py does json.loads(make_test_spec), skipping any GitHub fetch.
    from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec
    try:
        ts = make_test_spec(row)
        row["make_test_spec"] = json.dumps(dataclasses.asdict(ts))
        return row
    except Exception as e:
        print(f"  [swerebench] skip {row.get('instance_id')} (make_test_spec failed: {repr(e)[:80]})")
        return None


register(DatasetSpec(
    name="swerebench",
    hf_path="nebius/SWE-rebench",
    hf_split="filtered",
    image_fn=_swerebench_image,
    prepare_fn=_swerebench_prepare,
))


# Back-compat alias for callers that referenced the old tuple.
DATASETS = tuple(REGISTRY.keys())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load(name: str):
    from datasets import load_dataset
    spec = REGISTRY[name]
    return load_dataset(spec.hf_path, split=spec.hf_split)


# Disk cache for the built instance list. The build itself can be expensive
# (swerebench runs make_test_spec per row ~= 308 ms x 6542 rows ~= 34 min), and
# it re-runs on EVERY launch/resume — BEFORE resume can skip done instances. The
# cache makes a resume load in ~1s instead. Cache is keyed by a fingerprint
# (dataset + hf path/split + row count + BUILD_VERSION) so it auto-invalidates if
# the dataset changes or the build logic (image_fn/prepare_fn) is updated.
CACHE_DIR = __import__("pathlib").Path(__file__).resolve().parent / ".instance_cache"
BUILD_VERSION = 1   # bump when image_fn / prepare_fn logic changes (invalidates caches)


def _fingerprint(dataset: str, n_rows: int) -> dict:
    spec = REGISTRY[dataset]
    return {"dataset": dataset, "hf_path": spec.hf_path, "hf_split": spec.hf_split,
            "n_rows": n_rows, "build_version": BUILD_VERSION}


def _cache_path(dataset: str):
    return CACHE_DIR / f"{dataset}.instances.json"


def _load_cache(dataset: str, fp: dict):
    p = _cache_path(dataset)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            blob = json.load(f)
        if blob.get("_meta") == fp:
            return blob.get("instances")
    except Exception:
        return None
    return None   # stale (fingerprint mismatch) -> rebuild


def _save_cache(dataset: str, instances: list, fp: dict):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _cache_path(dataset)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"_meta": fp, "instances": instances}, f)
        tmp.replace(p)   # atomic
    except Exception:
        pass   # cache is an optimization; never fail the build over it


def build_instances(dataset: str, limit: "int | None" = None, start: int = 0,
                    use_cache: bool = True) -> "list[dict]":
    """Return rollout-ready instances for `dataset` (sliced [start:start+limit]).

    Each dict has at least: instance_id, docker_image, ip, problem_statement, plus
    the dataset-native fields R2E-Gym needs. Rows whose image_fn returns None (or
    whose prepare_fn returns None) are skipped.

    The FULL build (limit=None, start=0) is disk-cached (see CACHE_DIR): the first
    build writes the cache, later resumes load it in ~1s instead of re-running the
    (possibly minutes-long) per-row prepare_fn. Sliced builds (smoke tests with a
    limit/start) bypass the cache. Set use_cache=False to force a rebuild.
    """
    if dataset not in REGISTRY:
        raise ValueError(f"unknown dataset {dataset!r}; registered: {tuple(REGISTRY)}")
    spec = REGISTRY[dataset]
    d = _load(dataset)                       # cheap: HF reads its local parquet cache
    n = len(d)
    full = (limit is None and start == 0)
    fp = _fingerprint(dataset, n)
    if full and use_cache:
        cached = _load_cache(dataset, fp)
        if cached is not None:
            return cached
    end = n if limit is None else min(n, start + limit)
    out: "list[dict]" = []
    for i in range(start, end):
        row = dict(d[i])
        img = spec.image_fn(row)
        if not img:
            continue
        row["docker_image"] = img
        row["ip"] = IP
        if spec.prepare_fn is not None:
            row = spec.prepare_fn(row)
            if row is None:
                continue
        out.append(row)
    if full and use_cache:
        _save_cache(dataset, out, fp)
    return out


def group_by_image(instances: "list[dict]") -> "dict[str, list[dict]]":
    """Group instances by docker_image (order-preserving), so all tasks sharing an
    image are processed together and the image is pulled once / removed once."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for inst in instances:
        groups.setdefault(inst["docker_image"], []).append(inst)
    return groups


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("registered datasets:")
        for name, spec in REGISTRY.items():
            cp = _cache_path(name)
            tag = f"  [cached: {cp.stat().st_size//1024} KB]" if cp.exists() else ""
            print(f"  {name:12s} {spec.hf_path}:{spec.hf_split}{tag}")
        print("\nusage: build_dataset.py <dataset> [limit]   |   build_dataset.py warm <dataset>")
        sys.exit(0)
    # `warm`: pre-build + cache the FULL instance list (so the next collect resume is instant)
    if sys.argv[1] == "warm":
        ds = sys.argv[2]
        import time
        t = time.time()
        insts = build_instances(ds, use_cache=True)
        print(f"warmed {ds}: {len(insts)} instances cached in {time.time()-t:.1f}s -> {_cache_path(ds)}")
        sys.exit(0)
    ds = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    insts = build_instances(ds, limit=lim)
    groups = group_by_image(insts)
    print(f"{ds}: {len(insts)} instances, {len(groups)} unique images "
          f"(avg {len(insts)/max(len(groups),1):.1f} inst/image)")
    for img, g in list(groups.items())[:5]:
        print(f"  {img}  x{len(g)}  e.g. {g[0]['instance_id']}")
