#!/usr/bin/env python3
"""Build full-dataset R2E-Gym rollout instance lists (swegym / swesmith / swerebench).

This is the importable data layer for the full-collection driver (collect.py).
Unlike a per-image smoke check (which verifies every image manifest, fine for
1-3 instances), this returns ALL instances fast and lets the driver pull/verify
per image-group wave. Every instance is rollout-ready: has `docker_image`,
`ip=127.0.0.1`, and (for swerebench) a JSON-string `make_test_spec`.

Public images (the private harbor mirror is NOT reachable from a dev box):
  swegym    : xingyaoww/sweb.eval.x86_64.<iid with __ -> _s_>     (SWE-Gym/SWE-Gym, 2438, 1 inst/image)
  swesmith  : jyangballin/swesmith.x86_64.*  (row.image_name)     (SWE-bench/SWE-smith, ~59k; ~266 inst/image)
  swerebench: swerebench/sweb.eval.x86_64.*  (nebius filtered)    (nebius/SWE-rebench:filtered, 6542; 1 inst/image)

CLI (quick inspection):
  python build_dataset.py swesmith 200      # show counts + first 5 image groups
"""
import json
import dataclasses
from functools import lru_cache

IP = "127.0.0.1"

DATASETS = ("swegym", "swesmith", "swerebench")


@lru_cache(maxsize=4)
def _load(name: str):
    from datasets import load_dataset
    if name == "swegym":
        return load_dataset("SWE-Gym/SWE-Gym", split="train")
    if name == "swesmith":
        return load_dataset("SWE-bench/SWE-smith", split="train")
    if name == "swerebench":
        return load_dataset("nebius/SWE-rebench", split="filtered")
    raise ValueError(f"unknown dataset {name}")


def _swegym_image(iid: str) -> str:
    return "xingyaoww/sweb.eval.x86_64." + iid.replace("__", "_s_")


def build_instances(dataset: str, limit: int | None = None, start: int = 0) -> list[dict]:
    """Return rollout-ready instances for `dataset` (sliced [start:start+limit]).

    Each dict has at least: instance_id, docker_image, ip, problem_statement, and
    the dataset-native fields R2E-Gym's docker.py needs. swerebench rows also get
    a JSON-string `make_test_spec` (docker.py does json.loads on it, skipping any
    GitHub fetch).
    """
    assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
    d = _load(dataset)
    n = len(d)
    end = n if limit is None else min(n, start + limit)
    out: list[dict] = []

    if dataset == "swerebench":
        from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec
        for i in range(start, end):
            row = dict(d[i])
            img = row.get("docker_image")
            if not img or not img.startswith("swerebench/"):
                continue
            try:
                ts = make_test_spec(row)
                row["make_test_spec"] = json.dumps(dataclasses.asdict(ts))
            except Exception as e:
                print(f"  [swerebench] skip {row.get('instance_id')} (make_test_spec failed: {repr(e)[:80]})")
                continue
            row["docker_image"] = img
            row["ip"] = IP
            out.append(row)
        return out

    if dataset == "swesmith":
        for i in range(start, end):
            row = dict(d[i])
            img = row.get("image_name")
            if not img:
                continue
            row["docker_image"] = img
            row["ip"] = IP
            out.append(row)
        return out

    # swegym: no image field; derive it
    for i in range(start, end):
        row = dict(d[i])
        row["docker_image"] = _swegym_image(row["instance_id"])
        row["ip"] = IP
        out.append(row)
    return out


def group_by_image(instances: list[dict]) -> "dict[str, list[dict]]":
    """Group instances by docker_image (order-preserving), so all tasks sharing an
    image are processed together and the image is pulled once / removed once."""
    from collections import OrderedDict
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for inst in instances:
        groups.setdefault(inst["docker_image"], []).append(inst)
    return groups


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "swerebench"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    insts = build_instances(ds, limit=lim)
    groups = group_by_image(insts)
    print(f"{ds}: {len(insts)} instances, {len(groups)} unique images "
          f"(avg {len(insts)/max(len(groups),1):.1f} inst/image)")
    for img, g in list(groups.items())[:5]:
        print(f"  {img}  x{len(g)}  e.g. {g[0]['instance_id']}")
