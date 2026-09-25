#!/usr/bin/env python3
"""Unified, config-driven, resumable trajectory collection driver (STREAMING).

One entry for every dataset. A run is fully described by the dataset name plus a
small set of settings (model, max_steps, temperature, ...). Settings come from a
YAML config with per-dataset overrides; any field is overridable on the CLI.

  ENTRY:        ./collect.sh <dataset> [--flag value ...]
  or directly:  python collect.py <dataset> --config collect_config.yaml [...]

  PRECEDENCE:   DEFAULTS  <  config(global)  <  config.datasets.<name>  <  CLI flag

  DATA REGISTRY: datasets live in build_dataset.py's REGISTRY (add one there).
  SERVING URL:   taken from the config (no blob auto-detection); falls back to
                 reading `router_json` from blob only if `url` is null.

Output layout (settings are reflected in the path, so different settings on the
SAME dataset never collide):
  <out_root>/<dataset>/<model>_iter<max_steps>_t<temperature>[_nofc]/
      trajectories.jsonl        # one rollout+reward record per line (append/resume)
      trajectories_failed.jsonl # quarantined transient/degenerate (retried next run)
      config.json               # exact resolved settings for this run
      progress.json             # live counters
      run.log                   # OVERALL process log (driver-level: waves, progress, errors)
      logs/<iid>.log            # DETAILED per-instance log (subprocess stdout+stderr, kept)
      _stream/<iid>/            # transient working dir (removed once the record is merged)

SCHEDULER (streaming, no wave barrier): a ThreadPoolExecutor capped at
--max_workers runs rollouts continuously. Each rollout is its own 1-instance
runagent_multiple subprocess (rollout + INLINE reward, process-isolated). Images
are pulled ON DEMAND (deduped, capped at --pull_workers) and `docker rmi`'d by
reference count the moment an image's last instance finishes. Peak disk ~
(distinct in-flight images) x ~4GB.

RESUME: re-run the SAME command; instances already in trajectories.jsonl are
skipped (matched by instance_id). Transient failures are quarantined for retry.

Environment overrides:
  SWE_MASTER_PY   path to the swe-master conda python (default below)
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # SWE-Master/trajectory_collection
REPO = HERE.parent                              # SWE-Master
R2E = REPO / "R2E-Gym"
VENV_PY = os.environ.get(
    "SWE_MASTER_PY", "/home/v-murongma/miniconda3/envs/swe-master/bin/python")
BLOB_SAS = HERE / "blob_sas.sh"
DEFAULT_CONFIG = HERE / "collect_config.yaml"
DEFAULT_ROUTER_JSON = "sglang_workers/sing_sglang_router_qwen35_397b_8node_v6_router.json"

# Hardcoded last-resort defaults (used only if absent from both CLI and config).
DEFAULTS = {
    "url": None, "router_json": DEFAULT_ROUTER_JSON, "model": "Qwen/Qwen3.5-397B-A17B",
    "max_steps": 100, "max_tokens": 131072, "temperature": 0.6, "use_fn_calling": True,
    "max_workers": 6, "pull_workers": 8, "out_root": str(HERE / "collect_runs"),
    "k": None, "start": 0, "exp": None, "instance_timeout": 7200,
    "keep_images": False, "retry_failed": True, "orphan_cleanup": True,
}

sys.path.insert(0, str(HERE))
import build_dataset  # noqa: E402  (local module; resolves from HERE)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def log(msg: str, logfile=None):
    line = f"{now()} [collect] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def load_config(path) -> dict:
    """Load the YAML config (globals + per-dataset overrides). Missing file => {}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}


def resolve_settings(dataset: str, cli: dict, cfg: dict) -> dict:
    """Merge settings with precedence: DEFAULTS < cfg(global) < cfg.datasets.<ds> < CLI.

    Only CLI values that are not None override (argparse uses None for 'unset')."""
    dscfg = (cfg.get("datasets") or {}).get(dataset) or {}
    out = {}
    for key, dflt in DEFAULTS.items():
        if cli.get(key) is not None:
            val = cli[key]
        elif key in dscfg:
            val = dscfg[key]
        elif key in cfg:
            val = cfg[key]
        else:
            val = dflt
        out[key] = val
    for b in ("use_fn_calling", "keep_images", "retry_failed", "orphan_cleanup"):
        out[b] = to_bool(out[b])
    return out


def run_tag(model: str, max_steps: int, temperature: float, use_fn_calling: bool) -> str:
    """A filesystem-safe tag reflecting the MAIN settings, so different settings on
    the same dataset land in different dirs:  <model>_iter<max_steps>_t<temp>[_nofc]."""
    short = str(model).split("/")[-1]
    short = re.sub(r"[^A-Za-z0-9._-]", "-", short)
    temp = ("%g" % float(temperature))          # 0.6 -> "0.6", 1.0 -> "1"
    tag = f"{short}_iter{int(max_steps)}_t{temp}"
    if not use_fn_calling:
        tag += "_nofc"
    return tag


def resolve_url(cli_url: str | None, router_json: str, logfile=None) -> str:
    if cli_url:
        return cli_url.rstrip("/")
    out = subprocess.run(
        ["bash", str(BLOB_SAS), "cat", router_json],
        capture_output=True, text=True, timeout=60,
    ).stdout
    try:
        return json.loads(out)["url"].rstrip("/")
    except Exception:
        log(f"FATAL: could not resolve router URL from blob {router_json}", logfile)
        sys.exit(1)


def health_ok(url: str) -> bool:
    r = subprocess.run(
        ["curl", "-s", "-f", "--max-time", "20", f"{url}/v1/models"],
        capture_output=True,
    )
    return r.returncode == 0


def ensure_socat(logfile=None):
    """Expose local dockerd at 127.0.0.1:2375 (R2E-Gym hardcodes that). Evict squatters."""
    listening = subprocess.run(
        "ss -ltn 2>/dev/null | grep -q '127.0.0.1:2375'", shell=True
    ).returncode == 0
    has_socat = subprocess.run(
        "ss -ltnp 2>/dev/null | grep '127.0.0.1:2375 ' | grep -q socat", shell=True
    ).returncode == 0
    if listening and has_socat:
        return
    subprocess.run(
        "for pid in $(ss -ltnp 2>/dev/null | awk -F'pid=' '/127.0.0.1:2375 /{print $2}' "
        "| awk -F',' '{print $1}'); do kill \"$pid\" 2>/dev/null || true; done; sleep 1; "
        "nohup socat TCP-LISTEN:2375,bind=127.0.0.1,reuseaddr,fork "
        "UNIX-CONNECT:/var/run/docker.sock > /tmp/socat-2375.log 2>&1 & "
        "for _ in $(seq 1 10); do ss -ltnp 2>/dev/null | grep '127.0.0.1:2375 ' "
        "| grep -q socat && break; sleep 0.3; done",
        shell=True,
    )
    log("socat 127.0.0.1:2375 -> /var/run/docker.sock ready", logfile)


def patch_docker_tls():
    """Force plain-TCP docker: our socat at 127.0.0.1:2375 has no TLS. docker-py
    enables TLS if EITHER DOCKER_TLS_VERIFY or DOCKER_CERT_PATH is non-empty, so
    both must be blanked in R2E-Gym's docker.py (a non-empty cert path alone causes
    'Path to a certificate and key files must be provided' and every rollout fails)."""
    dp = str(R2E / "src/r2egym/agenthub/runtime/docker.py")
    subprocess.run(
        ["sed", "-i", "-E",
         r's|^DOCKER_TLS_VERIFY = ".*"|DOCKER_TLS_VERIFY = ""|; '
         r's|^DOCKER_CERT_PATH = "[^"]*"|DOCKER_CERT_PATH = ""|',
         dp],
        check=False,
    )


def patch_agent_fn_calling():
    """Ensure Qwen3.5 reasoning + native fn-calling work in R2E-Gym's agent.py.
    Two upstream-revert-prone fixes are re-applied idempotently before each run:
      (1) `qwen3.5` in the `support_fn_calling` allow-list — else tools aren't sent
          and the model loops emitting empty `<function=>` / 'forgot to use a function'.
      (2) the fn-calling parser must use `reasoning_parser` when `reasoning_content`
          is present (thinking models) — `custom_parser` only reads `.content` and
          DROPS the <think> reasoning, yielding tool calls with empty thought.
    Both were reverted by the same upstream commit more than once, so we self-heal."""
    ap = R2E / "src/r2egym/agenthub/agent/agent.py"
    txt = ap.read_text()
    changed = False
    # (1) allow-list
    if '"qwen3.5"' not in txt:
        anchor = '            or "qwen3-coder" in self.llm_name.lower()\n'
        if anchor in txt:
            txt = txt.replace(
                anchor,
                anchor + '            or "qwen3.5" in self.llm_name.lower()\n', 1)
            changed = True
    # (2) reasoning parser selection inside the fn-calling branch
    old_sel = ('            if self.use_fn_calling:\n'
               '                if "kimi" in self.llm_name:\n'
               '                    thought, action = self.reasoning_parser(response)\n'
               '                else:\n'
               '                    thought, action = self.custom_parser(response)\n')
    if old_sel in txt:
        new_sel = ('            if self.use_fn_calling:\n'
                   '                if getattr(response.choices[0].message, "reasoning_content", None):\n'
                   '                    thought, action = self.reasoning_parser(response)\n'
                   '                else:\n'
                   '                    thought, action = self.custom_parser(response)\n')
        txt = txt.replace(old_sel, new_sel, 1)
        changed = True
    if changed:
        ap.write_text(txt)


def ensure_git_ssh_rewrite():
    """swesmith's non-Python profiles (Go/Rust/C/C++/C#/PHP) clone the mirror repo
    LOCALLY via SSH (`git clone git@github.com:swesmith/<repo>.git`) to compute the
    test command. The dev box has no GitHub SSH key, so those clones fail with exit
    128 (`Permission denied (publickey)`) and the rollout produces no trajectory.
    The mirror repos are PUBLIC, so rewrite SSH->HTTPS globally (idempotent). Python
    profiles don't clone, which is why they were unaffected."""
    subprocess.run(
        ["git", "config", "--global",
         "url.https://github.com/.insteadOf", "git@github.com:"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def warn_if_no_github_token(dataset, logfile=None):
    """swesmith's non-Python profiles (Go/Rust/C/...) verify the mirror repo via the
    GitHub REST API (`GhApi(...).repos.get`) BEFORE cloning it. Unauthenticated, that
    API is capped at 60 req/hr per IP; with hundreds of Go instances (e.g. bleve=351)
    each launched in a FRESH subprocess, the cap is exhausted almost instantly. The
    resulting HTTP 403 is swallowed by a bare `except:` in swesmith and misreported as
    `Mirror clone repo must be created first` -> no trajectory. A GITHUB_TOKEN (no
    scopes needed; the mirrors are public) raises the limit to 5000/hr and fixes it.
    NOTE: the git SSH->HTTPS rewrite does NOT help here -- that's the REST API, not git."""
    if dataset != "swesmith":
        return
    if os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        return
    log("WARNING: GITHUB_TOKEN not set. swesmith's Go/non-Python reward path calls the "
        "GitHub API (60 req/hr unauthenticated); it WILL be rate-limited and those "
        "instances fail with 'Mirror clone repo must be created first'. Export a "
        "scope-less PAT (export GITHUB_TOKEN=...) and relaunch to collect them.", logfile)


def image_present(img: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", img],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def pull(img: str, logfile=None, timeout=1800) -> bool:
    if image_present(img):
        return True
    r = subprocess.run(["docker", "pull", img],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        log(f"  pull FAILED {img}: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}", logfile)
    return r.returncode == 0


def rmi(img: str, logfile=None):
    subprocess.run(["docker", "rmi", "-f", img],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pull_many(imgs, logfile=None, workers=8):
    """Pull a wave's images IN PARALLEL (docker pull is I/O-bound). Returns the
    set of images that are present afterwards. Serial pulling of ~32x4GB images
    left the GPUs idle for many minutes between waves; a thread pool overlaps the
    layer downloads so the run phase starts much sooner."""
    ok = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for img, good in zip(imgs, ex.map(lambda i: pull(i, logfile), imgs)):
            if good:
                ok.add(img)
    return ok


def cleanup_orphan_containers(logfile=None):
    """Remove leftover rollout sandbox containers (e.g. orphaned when a replica
    hung mid-rollout). R2E-Gym names them after the image (sweb.eval...); removing
    them frees image refs so the per-wave `docker rmi` actually reclaims disk."""
    ids = subprocess.run(
        "docker ps -aq --filter 'ancestor=' 2>/dev/null; "
        "docker ps -aq --filter 'name=sweb.eval' 2>/dev/null",
        shell=True, capture_output=True, text=True).stdout.split()
    ids = sorted(set(ids))
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"  cleaned {len(ids)} orphan rollout container(s)", logfile)


# Exit reasons that mean a TRANSIENT failure (server/LLM hiccup), not a real
# completion. Such trajectories are quarantined out of the main jsonl so their
# instances get RETRIED on the next run, instead of being treated as "done".
RETRYABLE_EXIT_REASONS = {"llm_query_error", "", None}


def _is_degenerate(rec) -> bool:
    """True if a trajectory is a fn-calling-bug / degraded-server victim that must be
    re-collected (not kept as 'done'). With thinking-mode ON and reasoning capture,
    a HEALTHY rollout records `<think>` reasoning in essentially every step; a broken
    one (tools not sent, or server returned no reasoning_content) has ~zero reasoning
    and typically loops to the step limit. So the robust signal is a near-zero
    reasoning rate over a non-trivial trajectory. (Also flags explicit empty
    `<function=>` / 'forgot to use a function' loops.)"""
    ts = rec.get("trajectory_steps") or []
    if len(ts) < 5:
        return False
    bad = thought = 0
    for s in ts:
        act = s.get("action") or ""
        obs = (s.get("observation") or "").lower()
        if "<function=>" in act or "forgot to use a function" in obs:
            bad += 1
        if (s.get("thought") or "").strip():
            thought += 1
    n = len(ts)
    return (thought / n) <= 0.5 or (bad / n) >= 0.3


def quarantine_failed(jsonl: Path, logfile=None) -> tuple[int, int]:
    """Move transient-failure trajectories out of the main jsonl into <exp>_failed.jsonl,
    so their instances are re-attempted on this run. Catches both (a) exit_reason in
    RETRYABLE_EXIT_REASONS (e.g. llm_query_error) and (b) degenerate fn-calling-bug
    victims (empty <function=> loops with no reasoning) regardless of exit_reason.

    The failed partials are kept (appended to the sidecar) for inspection, but no
    longer count as "done" — neither here nor in R2E-Gym's --use_existing filter
    (which reads the main jsonl). Returns (kept, moved). Atomic rewrite.
    """
    if not jsonl.exists():
        return 0, 0
    keep, fail = [], []
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            ln = line if line.endswith("\n") else line + "\n"
            try:
                rec = json.loads(line)
            except Exception:
                keep.append(ln)              # unparseable -> leave as-is (won't crash resume)
                continue
            retry = rec.get("exit_reason") in RETRYABLE_EXIT_REASONS or _is_degenerate(rec)
            (fail if retry else keep).append(ln)
    if not fail:
        return len(keep), 0
    failed_path = jsonl.with_name(jsonl.stem + "_failed.jsonl")
    with open(failed_path, "a") as f:
        f.writelines(fail)
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.writelines(keep)
    tmp.replace(jsonl)
    log(f"quarantined {len(fail)} failed/degenerate trajectories -> {failed_path.name} "
        f"(will retry); {len(keep)} good trajectories kept", logfile)
    return len(keep), len(fail)


def done_instance_ids(jsonl: Path) -> set:
    """instance_ids already in the jsonl (resume: skip these)."""
    done = set()
    if not jsonl.exists():
        return done
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                iid = (rec.get("ds") or {}).get("instance_id") or rec.get("instance_id")
                if iid:
                    done.add(iid)
            except Exception:
                pass
    return done


def _tail_error(log_path: Path) -> str:
    """Best-effort one-line error hint from a per-instance log (the last line that
    looks like an error, else the last non-empty line). Surfaced in the overall log
    so a failure is diagnosable without opening the file."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    pat = re.compile(r"Error|Exception|Traceback|FileNotFound|Timeout|refused|"
                     r"FATAL|Killed|No such|abort|HSA|HIP|OOM", re.I)
    for l in reversed(lines):
        s = l.strip()
        if s and pat.search(s):
            return s[:200]
    for l in reversed(lines):
        if l.strip():
            return l.strip()[:200]
    return ""


def run_one_instance(inst, run_dir: Path, exp: str, url: str, args, stream_dir: Path,
                     logs_dir: Path):
    """Run ONE instance end-to-end (rollout + inline reward) in an isolated
    runagent_multiple subprocess (1-item dataset, max_workers=1). The image is
    assumed already pulled. Returns (line, log_name, err):
      line     : the trajectory record (newline-terminated) or None if nothing was
                 produced (subprocess crash / empty) -> instance retried next run.
      log_name : filename of this instance's detailed log under run_dir/logs/.
      err      : one-line error hint when line is None (else "").

    Each task uses its OWN temp working dir + exp_name, so concurrent rollouts never
    share a jsonl; the parent merges the single record into the main jsonl (single
    writer). The DETAILED per-instance log (subprocess stdout+stderr, incl. errors)
    is written to run_dir/logs/<iid>.log and ALWAYS kept (success or failure); the
    temp working dir is removed once its one record is extracted. Process isolation
    per instance means one crashing/hung rollout cannot take down the driver."""
    iid = inst.get("instance_id") or "inst"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(iid))[:80]
    tmpd = stream_dir / safe
    tmpd.mkdir(parents=True, exist_ok=True)
    log_name = f"{safe}.log"
    log_path = logs_dir / log_name
    ds_file = tmpd / "ds.json"
    ds_file.write_text(json.dumps([inst]))
    sub_exp = f"s_{safe}"
    env = dict(os.environ)
    env["URL"] = url
    env["OPENAI_API_BASE"] = f"{url}/v1"
    env["OPENAI_API_KEY"] = "not-needed"
    env["DOCKER_TLS_VERIFY"] = ""
    # Fork-safety: R2E-Gym's runagent_multiple forks a ProcessPoolExecutor AFTER
    # importing litellm (httpx threads) + HF tokenizers (rayon threadpool). Forking
    # after those thread pools are live can deadlock the child on an inherited,
    # never-released lock (futex). Disable the parallelism that triggers it.
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("OMP_NUM_THREADS", "1")
    yaml = ("./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml"
            if args.use_fn_calling else
            "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml")
    cmd = [
        VENV_PY, "-m", "r2egym.agenthub.run.edit", "runagent_multiple",
        "--dataset", str(ds_file),
        "--split", "test",
        "--k", "1",
        "--start_idx", "0",
        "--traj_dir", str(tmpd),
        "--exp_name", sub_exp,
        "--llm_name", f"openai/{args.model}",
        "--temperature", str(args.temperature),
        "--use_fn_calling", str(args.use_fn_calling),
        "--backend", "docker",
        "--scaffold", "openhands",
        "--used_yaml", yaml,
        "--max_steps", str(args.max_steps),
        "--max_steps_absolute", str(args.max_steps),
        "--max_tokens", str(args.max_tokens),
        "--max_workers", "1",
        "--prepull_images", "False",          # image already pulled on-demand by the scheduler
        "--use_existing", "False",            # fresh temp dir; resume handled by the parent
        "--ip", "127.0.0.1",
        "--use_lsp", "False",
    ]
    timeout_s = int(getattr(args, "instance_timeout", 7200) or 7200)
    with open(log_path, "a") as lf:           # detailed per-instance log (kept always)
        lf.write(f"\n===== {now()} START {iid}  (exp={exp}, image={inst.get('docker_image')}) =====\n")
        lf.flush()
        # start_new_session=True puts the child in its OWN process group so a hang
        # can be killed as a GROUP (the runagent_multiple proc + its ProcessPool
        # grandchildren) without signalling the driver. A hung rollout therefore
        # frees its worker slot after timeout_s instead of blocking it forever
        # (the bug that starved the whole pool and stalled collection for hours).
        proc = subprocess.Popen(cmd, cwd=str(R2E), env=env, stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            lf.write(f"===== {now()} TIMEOUT {iid} after {timeout_s}s — killing process group =====\n")
            lf.flush()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            rc = -9
        lf.write(f"===== {now()} END {iid}  (subprocess rc={rc}) =====\n")
    line = None
    out = tmpd / f"{sub_exp}.jsonl"
    if out.exists():
        for raw in out.read_text().splitlines():
            if raw.strip():
                line = raw if raw.endswith("\n") else raw + "\n"
                break
    err = "" if line else _tail_error(log_path)
    # The detailed log lives under run_dir/logs/ and is kept regardless; the temp
    # working dir (ds.json + raw subprocess jsonl) is no longer needed once merged.
    shutil.rmtree(tmpd, ignore_errors=True)
    return line, log_name, err


def main():
    ap = argparse.ArgumentParser(
        description="Unified, config-driven, resumable trajectory collection (streaming).")
    ap.add_argument("dataset", choices=build_dataset.DATASETS,
                    help="which registered dataset to roll out (see build_dataset.py REGISTRY)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML config with serving + settings + per-dataset overrides")
    # All settings default to None => 'not set on CLI' => fall back to config/defaults.
    ap.add_argument("--url", default=None, help="router base URL (overrides config)")
    ap.add_argument("--router_json", default=None, help="blob router json (used only if url unset)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=None, help="max instances (default: all)")
    ap.add_argument("--start", type=int, default=None, help="dataset start offset")
    ap.add_argument("--max_steps", type=int, default=None, help="agent step cap ('max iter')")
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_workers", type=int, default=None,
                    help="max concurrent rollouts (streaming concurrency cap)")
    ap.add_argument("--pull_workers", type=int, default=None,
                    help="max concurrent on-demand docker pulls")
    ap.add_argument("--instance_timeout", type=int, default=None,
                    help="hard wall-clock cap (s) per rollout subprocess; on hit the "
                         "process group is killed so a hung rollout frees its slot "
                         "instead of starving the pool (default 7200)")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--exp", default=None,
                    help="override the auto run-dir name (default: <model>_iter<N>_t<temp>)")
    ap.add_argument("--use_fn_calling", type=to_bool, default=None,
                    metavar="BOOL", help="native function-calling (true/false)")
    ap.add_argument("--keep_images", type=to_bool, default=None, metavar="BOOL",
                    help="keep docker images after an image's instances finish (debug)")
    ap.add_argument("--orphan_cleanup", type=to_bool, default=None, metavar="BOOL",
                    help="preflight remove ALL sweb.eval containers; set false for a "
                         "2nd concurrent collection on the same host")
    ap.add_argument("--retry_failed", type=to_bool, default=None, metavar="BOOL",
                    help="quarantine transient/degenerate trajectories for retry")
    # Back-compat aliases for the old store_true flags.
    ap.add_argument("--no_orphan_cleanup", dest="orphan_cleanup", action="store_const",
                    const=False, help="alias for --orphan_cleanup false")
    ap.add_argument("--no_retry_failed", dest="retry_failed", action="store_const",
                    const=False, help="alias for --retry_failed false")
    ap.add_argument("--wave_images", type=int, default=None,
                    help="(deprecated; ignored — streaming bounds disk via --max_workers)")
    args = ap.parse_args()

    # Resolve effective settings: DEFAULTS < config(global) < config.datasets.<ds> < CLI.
    cfg = load_config(args.config)
    cli = {k: getattr(args, k, None) for k in DEFAULTS}
    s = resolve_settings(args.dataset, cli, cfg)
    for k, v in s.items():                     # expose resolved values as args.<k>
        setattr(args, k, v)

    # Output layout: <out_root>/<dataset>/<run_tag>/  (run_tag reflects main settings).
    # Anchor a RELATIVE out_root to the script dir so run_dir is ABSOLUTE — the
    # per-instance rollout subprocess runs with cwd=R2E-Gym and is handed these
    # paths (--dataset/--traj_dir), so they must not be cwd-relative.
    out_root = Path(s["out_root"])
    if not out_root.is_absolute():
        out_root = HERE / out_root
    tag = args.exp or run_tag(s["model"], s["max_steps"], s["temperature"], s["use_fn_calling"])
    run_dir = out_root / args.dataset / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    exp = tag
    jsonl = run_dir / "trajectories.jsonl"
    logfile = run_dir / "run.log"             # OVERALL process log (driver-level)
    logs_dir = run_dir / "logs"               # per-instance detailed logs (one file each)
    logs_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"dataset": args.dataset, **s, "run_tag": tag, "run_dir": str(run_dir),
         "config_file": args.config, "started_utc": now(), "host": os.uname().nodename},
        indent=2, default=str))

    log(f"=== COLLECT {args.dataset} -> {run_dir} ===", logfile)
    log(f"settings: model={s['model']} max_steps={s['max_steps']} temp={s['temperature']} "
        f"fn_calling={s['use_fn_calling']} max_workers={s['max_workers']} "
        f"pull_workers={s['pull_workers']} (streaming)", logfile)

    # --- preflight: server, socat, TLS patch ---
    url = resolve_url(s["url"], s["router_json"], logfile)
    if not health_ok(url):
        log(f"FATAL: router URL not healthy: {url}", logfile); sys.exit(1)
    log(f"router OK: {url}", logfile)
    ensure_socat(logfile)
    patch_docker_tls()
    patch_agent_fn_calling()
    ensure_git_ssh_rewrite()   # swesmith non-Python profiles SSH-clone github -> rewrite to HTTPS
    warn_if_no_github_token(args.dataset, logfile)   # swesmith Go reward path needs GH API auth (60->5000/hr)
    if subprocess.run(["curl", "-sf", "http://127.0.0.1:2375/version"],
                      stdout=subprocess.DEVNULL).returncode != 0:
        log("FATAL: docker TCP 127.0.0.1:2375 not reachable", logfile); sys.exit(1)
    if s["orphan_cleanup"]:
        cleanup_orphan_containers(logfile)   # clear leftovers from a previous interrupted run

    # --- build + group instances ---
    log("building instance list...", logfile)
    instances = build_dataset.build_instances(args.dataset, limit=args.k, start=args.start)
    groups = build_dataset.group_by_image(instances)
    log(f"{len(instances)} instances across {len(groups)} unique images", logfile)

    # --- resume: drop fully-done image groups + already-done instances ---
    # First quarantine transient failures (llm_query_error) so they are RETRIED
    # this run instead of counting as done (unless retry_failed is off).
    if args.retry_failed:
        quarantine_failed(jsonl, logfile)
    done = done_instance_ids(jsonl)
    if done:
        log(f"resume: {len(done)} instances already in {jsonl.name}", logfile)
    pending_groups = []  # list of (img, [instances not yet done])
    for img, g in groups.items():
        todo = [inst for inst in g if inst["instance_id"] not in done]
        if todo:
            pending_groups.append((img, todo))
    n_pending = sum(len(g) for _, g in pending_groups)
    log(f"pending: {n_pending} instances across {len(pending_groups)} images "
        f"(skipped {len(instances) - n_pending} done)", logfile)
    if not pending_groups:
        log("nothing to do — all instances already collected.", logfile)
        return

    # --- streaming scheduler: continuous, no wave barrier ---
    # Flatten pending instances (kept grouped by image so an image's instances run
    # close together and it can be offloaded as soon as its LAST instance finishes).
    # Concurrency is capped at --max_workers; each in-flight rollout holds exactly one
    # image, so peak disk ~ (distinct images among in-flight) x ~4GB — which is
    # <= max_workers for 1-image-per-instance datasets (swegym) and far less for
    # swesmith (~266 instances share one image). Images are pulled ON DEMAND (deduped,
    # capped at --pull_workers concurrent pulls) and `docker rmi`'d by reference count
    # the moment an image's last instance completes. No pre-pull-all, no wave barrier:
    # when one rollout finishes, its slot immediately starts the next pending instance.
    pending = [inst for _, g in pending_groups for inst in g]
    image_remaining = Counter(inst.get("docker_image") for inst in pending)
    total_all = len(instances)
    t0 = time.time()
    stream_dir = run_dir / "_stream"; stream_dir.mkdir(exist_ok=True)

    pulled, failed_pull, resident = set(), set(), set()
    pull_state_lock = threading.Lock()
    pull_sem = threading.Semaphore(max(1, args.pull_workers))
    locks_lock = threading.Lock()
    img_locks = {}

    def get_img_lock(img):
        with locks_lock:
            lk = img_locks.get(img)
            if lk is None:
                lk = img_locks[img] = threading.Lock()
            return lk

    def ensure_pulled(img):
        with get_img_lock(img):                 # serialize per-image so it is pulled once
            with pull_state_lock:
                if img in pulled:
                    return True
                if img in failed_pull:
                    return False
            with pull_sem:                      # cap concurrent pulls globally
                ok = pull(img, logfile)
            with pull_state_lock:
                (pulled if ok else failed_pull).add(img)
                if ok:
                    resident.add(img)
            return ok

    def offload_image(img):
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"ancestor={img}"],
                             capture_output=True, text=True).stdout.split()
        if ids:                                 # only THIS image's containers (safe vs others)
            subprocess.run(["docker", "rm", "-f", *ids],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rmi(img, logfile)
        with pull_state_lock:
            resident.discard(img)

    def worker(inst):
        img = inst.get("docker_image")
        if not ensure_pulled(img):
            return "drop", inst, img, None, None, "image pull failed"
        line, log_name, err = run_one_instance(inst, run_dir, exp, url, args,
                                                stream_dir, logs_dir)
        return ("ok" if line else "fail"), inst, img, line, log_name, err

    done_count = len(done)
    ok_n = drop_n = fail_n = 0
    last_prog = 0.0

    def write_progress():
        df = subprocess.run("df -h /datadisk 2>/dev/null | awk 'NR==2{print $4}'",
                            shell=True, capture_output=True, text=True).stdout.strip()
        with pull_state_lock:
            res = len(resident)
        (run_dir / "progress.json").write_text(json.dumps({
            "instances_done": done_count, "instances_total": total_all,
            "ok": ok_n, "dropped_pull": drop_n, "no_traj": fail_n,
            "max_workers": args.max_workers, "images_resident": res,
            "datadisk_free": df, "updated_utc": now(),
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }, indent=2))
        return df, res

    log(f"streaming: {len(pending)} instances across {len(image_remaining)} images "
        f"| max_workers={args.max_workers} pull_workers={args.pull_workers}", logfile)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(worker, inst): inst for inst in pending}
        for fut in as_completed(futs):
            log_name = err = None
            try:
                status, inst, img, line, log_name, err = fut.result()
            except Exception as e:                 # worker itself crashed (rare)
                inst = futs[fut]; img = inst.get("docker_image")
                status, line = "fail", None
                err = str(e)
                log(f"  worker crashed for {inst.get('instance_id')}: {e}", logfile)
            iid = inst.get("instance_id")
            if status == "ok" and line:
                with open(jsonl, "a") as f:        # main thread = single writer (no race)
                    f.write(line)
                ok_n += 1; done_count += 1
            elif status == "drop":
                drop_n += 1
                log(f"  dropped {iid} — image pull failed: {img}", logfile)
            else:
                fail_n += 1
                msg = f"  no trajectory for {iid} (retry next run)"
                if log_name:
                    msg += f" — see logs/{log_name}"
                if err:
                    msg += f" :: {err}"
                log(msg, logfile)
            # reference-count the image: offload as soon as its LAST instance finishes
            image_remaining[img] -= 1
            if image_remaining[img] <= 0 and not args.keep_images:
                offload_image(img)
            if time.time() - last_prog > 20:
                df, res = write_progress(); last_prog = time.time()
                log(f"  progress: {done_count}/{total_all} done "
                    f"(ok={ok_n} drop={drop_n} no_traj={fail_n}) | resident~={res} | "
                    f"/datadisk {df}", logfile)

    write_progress()
    log(f"=== COLLECT DONE: {done_count}/{total_all} instances "
        f"(ok={ok_n} drop={drop_n} no_traj={fail_n}) in {jsonl} ===", logfile)


if __name__ == "__main__":
    main()
