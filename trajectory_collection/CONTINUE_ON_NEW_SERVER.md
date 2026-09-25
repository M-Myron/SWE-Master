# Continue trajectory collection on a NEW server

**Read this first if you are an agent resuming this work on a different machine.**
This is a long-running, resumable trajectory-collection job. It was paused only to
move the directory to a new host. The backend LLM is served separately (on
Singularity) and **its URL does not change** — you do **not** need to re-serve anything.
Your job: recreate the local prerequisites, then relaunch the drivers. They resume
automatically from the already-collected data.

> Deep reference for how the driver works lives in [README.md](README.md). This file
> is the **transfer + resume runbook** only.

---

## 0. State at the time of the move  (2026-06-12 ~03:20 UTC)

| Dataset | Records collected | Approx total | Status when moved | Action on new box |
|---|---|---|---|---|
| **swegym** | 2404 | ~2438 (~98.5%) | **done** (driver stopped) | optional — only ~34 left |
| **swesmith** | 8771 | ~59,136 (~15%) | **running** | **resume** |
| **swerebench** | 3061 | ~6,538 (~47%) | **running** | **resume** |

All three use the **same** served model `Qwen/Qwen3.5-397B-A17B` (reasoning ON) at the
**same** endpoint (8-node SGLang router). Resume is by `instance_id`: instances already
present in `trajectories.jsonl` are skipped, so re-running a dataset only collects what
is missing. Re-running is always safe and idempotent.

---

## 1. What travels with the move (and what does NOT)

**Move the WHOLE `SWE-Master/` directory.** The driver requires `R2E-Gym/` to sit as a
**sibling** of `trajectory_collection/` (`collect.py` resolves it as `../R2E-Gym`). If you
move only `trajectory_collection/`, the driver will break. Keep this layout intact:

```
SWE-Master/
├── R2E-Gym/                                  # 6.6 GB  rollout harness (MUST travel)
├── swebench_fork_swegym-2.0.13-*.whl         # env wheels (MUST travel)
├── swebench_fork_swerebench-4.0.3-*.whl
├── swesmith-0.0.7-*.whl
└── trajectory_collection/
    ├── collect.py  collect.sh  build_dataset.py  collect_config.yaml
    ├── env/swe-master.pip-freeze.txt         # exact env snapshot (version reference)
    ├── collect_runs/                         # 27 GB  ← THE COLLECTED DATA + resume state. MUST travel.
    └── .instance_cache/                      # 206 MB ← saves ~30 min swerebench rebuild. Nice to have; auto-regenerates if absent.
```

**Do NOT bother copying:**
- Docker **images** — re-pulled on demand on the new box (large; re-download is fine).
- Docker **containers** — ephemeral rollout sandboxes; recreated per instance.
- The conda **env** — recreate it (see step 2); paths are absolute and won't move cleanly.

Total to transfer ≈ **34 GB** (mostly `collect_runs/`). Use `tar`/`rsync`.

### Before you copy (on the OLD box) — get a clean snapshot
The drivers append to `trajectories.jsonl` continuously. Resume tolerates a partial last
line, but a clean stop avoids losing in-flight instances:

```bash
cd SWE-Master/trajectory_collection
# 1) stop the driver(s) and their per-instance subprocess trees
for p in $(pgrep -f 'collect\.py (swesmith|swerebench|swegym)'); do
  pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null
done
sleep 5
pkill -KILL -f 'collect\.py (swesmith|swerebench|swegym)' 2>/dev/null
# 2) (optional) remove leftover rollout containers so the tar/move is clean
docker ps -aq --filter 'name=sweb.eval' --filter 'name=swesmith' | xargs -r docker rm -f
```

---

## 2. Recreate prerequisites on the NEW box (these are NOT in the directory)

### 2a. conda env `swe-master`
Python 3.12. The exact pins are in [env/swe-master.pip-freeze.txt](env/swe-master.pip-freeze.txt)
(use it as a version reference — the `-e git+...` / `file://` lines point at old paths, so
prefer the recipe below which installs from the moved location):

```bash
conda create -n swe-master python=3.12 -y
conda activate swe-master
cd /path/to/SWE-Master                 # the moved directory
pip install -e R2E-Gym                 # editable rollout harness (pulls litellm, docker, datasets, ...)
pip install swebench_fork_swegym-2.0.13-py3-none-any.whl \
            swebench_fork_swerebench-4.0.3-py3-none-any.whl \
            swesmith-0.0.7-py3-none-any.whl
pip install pyyaml                      # collect.py reads the YAML config
```
Key versions that must end up installed: `swebench==3.0.2`, `litellm==1.86.1`,
`docker==7.1.0`, `datasets==2.19.0`, `transformers==4.45.2`, `tiktoken==0.13.0`.

### 2b. Docker daemon with a big data-root
Rollout images are large and numerous. Put Docker's data-root on a disk with **≥ ~300 GB
free** (the old box used `/datadisk/docker`). Example `/etc/docker/daemon.json`:
```json
{ "data-root": "/datadisk/docker" }
```
Then `sudo systemctl restart docker` and verify `docker info | grep "Docker Root Dir"`.

### 2c. Docker Hub login (avoid pull rate limits)
The pulls hit Docker Hub heavily. Log in (a **Pro** account removes the anonymous
100-pulls/6h limit that otherwise causes `toomanyrequests` errors):
```bash
docker login
```

### 2c-bis. `GITHUB_TOKEN` (REQUIRED for swesmith)
swesmith's Go/non-Python reward path verifies the mirror repo via the **GitHub REST
API**, capped at **60 req/hr** unauthenticated. With hundreds of Go instances (e.g.
`bleve` = 351) each in a fresh subprocess, that cap is exhausted in seconds → HTTP 403
→ swesmith's bare `except:` misreports it as `Mirror clone repo must be created first`
→ no trajectory. (The git SSH→HTTPS rewrite does NOT help — that's the API, not git.)
Fix: export a **scope-less** classic PAT (the mirrors are public, so no scopes needed;
raises the limit to 5000/hr) **before launching the swesmith driver**:
```bash
export GITHUB_TOKEN=ghp_xxxxxxxx     # the per-instance subprocesses inherit this
```
The driver prints `WARNING: GITHUB_TOKEN not set` at preflight if it's missing.

### 2d. Everything else is auto-healed by `collect.py` preflight — nothing to do
On every launch the driver re-applies these automatically, so they self-heal on a fresh box:
- **socat TCP bridge** `127.0.0.1:2375 → /var/run/docker.sock` (R2E-Gym hardcodes that TCP endpoint).
- **git SSH→HTTPS rewrite** (`url."https://github.com/".insteadOf "git@github.com:"`) — needed by swesmith non-Python repos.
- **docker-TLS patch** + **qwen3.5 function-calling / reasoning-parser patch** inside `R2E-Gym`.

---

## 3. Verify before resuming

```bash
conda activate swe-master
cd /path/to/SWE-Master/trajectory_collection

# backend LLM reachable (URL is unchanged; should print 200)
curl -s -o /dev/null -w '%{http_code}\n' --max-time 15 \
  https://declaration-accidents-together-disabled.trycloudflare.com/health

# env sane
python -c "import r2egym, swebench, litellm, yaml; print('env OK', swebench.__version__)"

# docker sane
docker ps >/dev/null && echo "docker OK"

# what will be skipped on resume (already-collected counts)
for d in swegym swesmith swerebench; do
  f=$(ls collect_runs/$d/*/trajectories.jsonl 2>/dev/null | head -1)
  echo "$d: $([ -f "$f" ] && wc -l < "$f" || echo 0) records already collected"
done
```
If `/health` is **not** 200, see §5 "If the backend URL DID change".

---

## 4. Resume the collection

Run from `SWE-Master/trajectory_collection`. The config (`collect_config.yaml`) already
sets the URL, model, `max_workers=48`, `max_steps=100`, `temperature=0.6`, etc., so the
command is minimal. Resume = just re-run the same dataset.

> **IMPORTANT — concurrency:** when running **swesmith and swerebench at the same time,
> launch BOTH with `--no_orphan_cleanup`.** The startup orphan-cleanup does
> `docker rm -f` on every container whose name matches `sweb.eval`, which catches
> **swerebench** containers (named `swerebench-sweb.eval.x86_64.*`) and swegym containers —
> so an unguarded launch of one dataset will **kill the other dataset's in-flight
> containers**. On a freshly-moved box there are no orphans to clean anyway, so disabling
> it costs nothing.

```bash
conda activate swe-master
cd /path/to/SWE-Master/trajectory_collection

export GITHUB_TOKEN=ghp_xxxxxxxx       # REQUIRED for swesmith (see §2c-bis); scope-less PAT

# resume swesmith
nohup ./collect.sh swesmith --no_orphan_cleanup > /tmp/swesmith.out 2>&1 &

# resume swerebench
nohup ./collect.sh swerebench --no_orphan_cleanup > /tmp/swerebench.out 2>&1 &

# swegym is effectively done (2404/2438). Only if you want the last ~34:
# nohup ./collect.sh swegym --no_orphan_cleanup > /tmp/swegym.out 2>&1 &
```

If you run **only one** dataset at a time, you may omit the flag (cleanup is then harmless).

---

## 5. Gotchas & operations

- **Backend URL is unchanged.** It lives in `collect_config.yaml` (`url:`). The model is
  served on Singularity, independent of this box, so it works from anywhere.
  **If the backend URL DID change** (only if someone restarted the serving job and got a
  new Cloudflare URL): edit the `url:` at the top of `collect_config.yaml` (and any
  per-dataset override under `datasets:`), then relaunch. The served model id must stay
  `Qwen/Qwen3.5-397B-A17B`.

- **Host load.** Two datasets × `max_workers=48` is heavy. On a 24-core box this drove
  loadavg to ~48 and caused transient `Connection refused (Errno 111)` to
  `127.0.0.1:2375` (socat/dockerd saturation; failures just become retries). If loadavg
  is far above your core count or you see those errors, lower workers:
  `./collect.sh swesmith --no_orphan_cleanup --max_workers 32`.

- **Per-instance logs:** `collect_runs/<dataset>/<run_tag>/logs/<instance_id>.log`
  (kept always; each has `===== START/END/TIMEOUT =====` markers). Failed/transient
  trajectories are quarantined to `trajectories_failed.jsonl` and retried next run.

- **Hung rollouts self-recover:** `instance_timeout` (7200s, in config) SIGKILLs a stuck
  per-instance subprocess (deadlock protection), so the worker pool never starves.

- **Monitor:**
  ```bash
  tail -f /tmp/swesmith.out
  wc -l collect_runs/swesmith/*/trajectories.jsonl
  cat  collect_runs/swesmith/*/progress.json
  pgrep -af 'collect\.py'          # confirm drivers alive
  docker ps -q | wc -l             # active rollout sandboxes
  ```

- **swesmith Go repos (~1.2%, 7 repos / 737 instances)** need `GITHUB_TOKEN` set (§2c-bis)
  — without it they fail with `Mirror clone repo must be created first` (GitHub API rate
  limit). With the token they collect normally. A few may still be flaky under heavy
  concurrency (an internal `go mod tidy` / `*_test.go` scan); those just retry next run.
