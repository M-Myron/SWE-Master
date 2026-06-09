#!/usr/bin/env python3
"""Full-dataset trajectory collection driver — IMAGE-GROUPED, RESUMABLE, DISK-BOUNDED.

Design (answers the 4 core requirements):

  1. UNIFORM ENTRY + CONFIG: one CLI. All knobs are flags with sane defaults; a run
     is fully described by (dataset, model, k, wave_images, max_workers). Re-runs are
     deterministic given the same flags. Data is stored under a single run dir.

  2. RESUME: instances already present in the run's <exp>.jsonl are skipped (we reuse
     R2E-Gym's `--use_existing True`, which filters by problem_statement; we ALSO skip
     at the wave level by instance_id so we don't even re-pull a fully-done image's
     group). Just re-run the SAME command; it picks up where it left off.

  3. DISK-BOUNDED IMAGE LIFECYCLE: instances are grouped by docker_image (swesmith has
     ~266 instances/image!). We process images in WAVES of `--wave_images` images at a
     time: pull the wave's images, run ALL their instances (one runagent_multiple call
     so they parallelize across `--max_workers`), then `docker rmi` exactly that wave's
     images before the next wave. Peak disk = wave_images x ~4GB, never the whole set.

  4. REWARD INLINE PER-INSTANCE: R2E-Gym's runagent computes the reward IN THE SAME
     container right after the rollout (env.runtime._calculate_reward before env.close()).
     There is NO separate eval pass and NO second image pull — so grouping + per-wave
     offload is safe: by the time we rmi an image, every rollout AND its reward for that
     image are already done and written.

Data store layout (under --out_root, default collect_runs/):
  <out_root>/<exp>/
    <exp>.jsonl              # one trajectory record per line (rollout + reward). APPEND.
    config.json             # the exact run config (reproducibility)
    progress.json           # waves done, instances done/total, last update (resume aid)
    waves/wave_XXXX.json     # the runagent_multiple input for each wave (audit)
    run.log                 # full driver + runagent log

Usage:
  python collect.py --dataset swerebench --k 200 --wave_images 8 --max_workers 6
  # resume: just run the same line again.

Environment overrides:
  SWE_MASTER_PY   path to the swe-master conda python (default below)
  URL             router base URL; if unset, resolved from blob via blob_sas.sh
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # SWE-Master/trajectory_collection
REPO = HERE.parent                              # SWE-Master
R2E = REPO / "R2E-Gym"
VENV_PY = os.environ.get(
    "SWE_MASTER_PY", "/home/v-murongma/miniconda3/envs/swe-master/bin/python")
BLOB_SAS = HERE / "blob_sas.sh"
DEFAULT_ROUTER_JSON = "sglang_workers/sing_sglang_router_qwen35_397b_8node_v6_router.json"

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


def run_wave(wave_input: Path, traj_dir: Path, exp: str, url: str, args, logfile):
    """One runagent_multiple call over a wave's instances (all images pre-pulled)."""
    env = dict(os.environ)
    env["URL"] = url
    env["OPENAI_API_BASE"] = f"{url}/v1"
    env["OPENAI_API_KEY"] = "not-needed"
    env["DOCKER_TLS_VERIFY"] = ""
    yaml = ("./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml"
            if args.use_fn_calling else
            "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml")
    n = json.loads(wave_input.read_text()).__len__()
    cmd = [
        VENV_PY, "-m", "r2egym.agenthub.run.edit", "runagent_multiple",
        "--dataset", str(wave_input),
        "--split", "test",
        "--k", str(n),
        "--start_idx", "0",
        "--traj_dir", str(traj_dir),
        "--exp_name", exp,
        "--llm_name", f"openai/{args.model}",
        "--temperature", str(args.temperature),
        "--use_fn_calling", str(args.use_fn_calling),
        "--backend", "docker",
        "--scaffold", "openhands",
        "--used_yaml", yaml,
        "--max_steps", str(args.max_steps),
        "--max_steps_absolute", str(args.max_steps),
        "--max_tokens", str(args.max_tokens),
        "--max_workers", str(args.max_workers),
        "--prepull_images", "False",          # we pull per wave ourselves
        "--use_existing", "True",             # RESUME: skip instances already in jsonl
        "--ip", "127.0.0.1",
        "--use_lsp", "False",
    ]
    with open(logfile, "a") as lf:
        proc = subprocess.run(cmd, cwd=str(R2E), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Image-grouped, resumable trajectory collection.")
    ap.add_argument("--dataset", required=True, choices=build_dataset.DATASETS)
    ap.add_argument("--k", type=int, default=None, help="max instances (default: all)")
    ap.add_argument("--start", type=int, default=0, help="dataset start offset")
    ap.add_argument("--wave_images", type=int, default=8,
                    help="images resident at once (peak disk ~ this x 4GB)")
    ap.add_argument("--max_workers", type=int, default=6, help="parallel rollouts")
    ap.add_argument("--pull_workers", type=int, default=8,
                    help="parallel docker pulls per wave (overlaps layer downloads)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    ap.add_argument("--url", default=None, help="router URL (default: read from blob)")
    ap.add_argument("--router_json", default=DEFAULT_ROUTER_JSON)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=131072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--use_fn_calling", default="True")
    ap.add_argument("--out_root", default=str(HERE / "collect_runs"))
    ap.add_argument("--exp", default=None, help="experiment/run name (default: <dataset>_full)")
    ap.add_argument("--keep_images", action="store_true",
                    help="do NOT docker rmi after each wave (debug)")
    ap.add_argument("--no_retry_failed", action="store_true",
                    help="do NOT quarantine+retry transient failures (llm_query_error); "
                         "treat them as done like terminal trajectories")
    args = ap.parse_args()

    exp = args.exp or f"{args.dataset}_full"
    run_dir = Path(args.out_root) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    waves_dir = run_dir / "waves"; waves_dir.mkdir(exist_ok=True)
    jsonl = run_dir / f"{exp}.jsonl"
    logfile = run_dir / "run.log"
    cfg = {**vars(args), "exp": exp, "started_utc": now(),
           "host": os.uname().nodename}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    log(f"=== COLLECT {args.dataset} exp={exp} k={args.k} wave_images={args.wave_images} "
        f"max_workers={args.max_workers} ===", logfile)

    # --- preflight: server, socat, TLS patch ---
    url = resolve_url(args.url, args.router_json, logfile)
    if not health_ok(url):
        log(f"FATAL: router URL not healthy: {url}", logfile); sys.exit(1)
    log(f"router OK: {url}", logfile)
    ensure_socat(logfile)
    patch_docker_tls()
    patch_agent_fn_calling()
    if subprocess.run(["curl", "-sf", "http://127.0.0.1:2375/version"],
                      stdout=subprocess.DEVNULL).returncode != 0:
        log("FATAL: docker TCP 127.0.0.1:2375 not reachable", logfile); sys.exit(1)

    # --- build + group instances ---
    log("building instance list...", logfile)
    instances = build_dataset.build_instances(args.dataset, limit=args.k, start=args.start)
    groups = build_dataset.group_by_image(instances)
    log(f"{len(instances)} instances across {len(groups)} unique images", logfile)

    # --- resume: drop fully-done image groups + already-done instances ---
    # First quarantine transient failures (llm_query_error) so they are RETRIED
    # this run instead of counting as done (unless --no_retry_failed).
    if not args.no_retry_failed:
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

    # --- wave loop ---
    total_done = len(done)
    total_all = len(instances)
    t0 = time.time()
    for wi in range(0, len(pending_groups), args.wave_images):
        wave = pending_groups[wi: wi + args.wave_images]
        wave_imgs = [img for img, _ in wave]
        wave_insts = [inst for _, g in wave for inst in g]
        wnum = wi // args.wave_images
        log(f"--- WAVE {wnum}: {len(wave_imgs)} images, {len(wave_insts)} instances ---", logfile)

        # 1) pull this wave's images IN PARALLEL (skip any that fail)
        present = pull_many(wave_imgs, logfile, workers=args.pull_workers)
        ok_imgs, ok_insts = [], []
        for img, g in wave:
            if img in present:
                ok_imgs.append(img); ok_insts.extend(g)
            else:
                log(f"  dropping image {img} ({len(g)} instances) — pull failed", logfile)
        if not ok_insts:
            log("  wave has no pullable images; skipping", logfile)
            continue

        # 2) write wave input + run (rollout + inline reward, parallel across workers)
        wave_input = waves_dir / f"wave_{wnum:04d}.json"
        wave_input.write_text(json.dumps(ok_insts, indent=2))
        rc = run_wave(wave_input, run_dir, exp, url, args, logfile)
        log(f"  wave {wnum} runagent_multiple rc={rc}", logfile)

        # 3) offload this wave's images (rollout + reward already done & written)
        if not args.keep_images:
            cleanup_orphan_containers(logfile)   # free image refs from any hung rollouts
            for img in ok_imgs:
                rmi(img, logfile)
            log(f"  offloaded {len(ok_imgs)} images (docker rmi)", logfile)

        # 4) progress
        total_done = len(done_instance_ids(jsonl))
        df = subprocess.run("df -h /datadisk 2>/dev/null | awk 'NR==2{print $4}'",
                            shell=True, capture_output=True, text=True).stdout.strip()
        (run_dir / "progress.json").write_text(json.dumps({
            "waves_done": wnum + 1,
            "instances_done": total_done, "instances_total": total_all,
            "datadisk_free": df, "updated_utc": now(),
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }, indent=2))
        log(f"  progress: {total_done}/{total_all} done | /datadisk free {df}", logfile)

    log(f"=== COLLECT DONE: {total_done}/{total_all} instances in {jsonl} ===", logfile)


if __name__ == "__main__":
    main()
