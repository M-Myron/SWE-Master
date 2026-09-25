# SWE-Master Trajectory Collection — Server / Session Handoff

> **Purpose.** Everything a NEW server (and a NEW AI session) needs to continue the
> SFT/RL trajectory-collection effort. The code is pushed to GitHub; this file covers
> the parts that are **NOT in git** (credentials, conda envs, docker host, amlt project,
> local data) plus the exact env vars and commands to make it runnable.
>
> **This file is intentionally not committed** — it references credential *locations*
> and host-specific setup. Keep it private. It contains **no secrets** (no SAS token,
> no account keys); you must copy the real credential file separately (see §4.1).
>
> Snapshot date: **2026-06-09**. Original box: `GCRAZGDL1683` (GCR dev box, has MI300X-free
> local docker host). User: `v-murongma`.

---

## 0. TL;DR — what is running and what to do

- **Goal:** collect agent trajectories (rollout + inline reward) for **swegym / swesmith /
  swerebench** using a served **Qwen3.5-397B-A17B** (reasoning ON) policy model, for SFT/RL.
- **Two halves:**
  1. **Serving** (the LLM) runs on **Singularity via amlt** (MI300X nodes). The dev box only
     *submits* it and *reads* its published router URL from blob. **No GPU needed on the dev box.**
  2. **Collection** (the rollouts) runs **on the dev box** against that URL, driving R2E-Gym
     inside per-instance docker containers.
- **Live right now (original box):**
  - Serving job `sing_sglang_router_qwen35_397b_8node_v6` — 8 nodes, 7/8 replicas healthy,
    URL `https://gore-isaac-life-carnival.trycloudflare.com` (changes on tunnel restart — always
    re-read from blob). Backup `sing_sglang_router_qwen35_397b_2node_v1` also up.
  - Collection `swegym` at **196 / 2438** instances (resumable).
- **To continue on a new box:** do §3 (clone) → §4 (recreate the not-in-git pieces) → §5 (env
  vars) → §6 (bring-up commands). Collection is **resumable** — re-running the same command skips
  done instances.

---

## 1. The repository (this IS pushed)

- **Remote:** `https://github.com/M-Myron/SWE-Master.git`
- **Branch:** `main`  •  **HEAD:** `2a342a1` (origin/main is in sync)
- **Last 4 commits (the work being handed off):**
  | Commit | What |
  |---|---|
  | `2a342a1` | docs(copilot_memory): SGLang multinode serving + e2e rollout playbooks |
  | `6bf2600` | feat(trajectory_collection): the collection toolkit + serving configs (parameterized) |
  | `1f19a9e` | fix(r2e-gym): Qwen3.5 fn-calling allow-list + reasoning_content parser + docker TLS/cert + backoff |
  | `74c3461` | chore(gitignore): ignore `collect_runs/` + `singularity/job_logs/` |
- **Key paths inside the repo:**
  - `trajectory_collection/` — the collection driver (`collect.py`, `collect.sh`, `build_dataset.py`,
    `blob_sas.sh`) and its `README.md`.
  - `trajectory_collection/serving/` — the SGLang serving job (amlt YAMLs, supervisor `.sh`,
    Dockerfile, monitors) and its `README.md`. **Read these two READMEs first.**
  - `R2E-Gym/` — the agent harness (NOT a submodule; shares the SWE-Master `.git`). Patched
    `agent.py` + `docker.py` are committed.
  - `copilot_memory/` — operational playbooks (serving + e2e collection).

---

## 2. Architecture (how the two halves connect)

```
  dev box (this server)                         Singularity / amlt (MI300X)
  ─────────────────────                         ───────────────────────────
  collect.py  ──reads──►  blob: sglang_workers/<exp>_router.json  ◄──writes── rank-0 of serving job
      │                          {"url": "https://<slug>.trycloudflare.com"}
      │ OPENAI_API_BASE = <url>/v1
      ▼
  R2E-Gym rollout (per instance, in docker)  ──HTTP──►  Cloudflare tunnel ──► sglang-router ──► N replicas
      │   (reward computed inline, same container)
      ▼
  collect_runs/<exp>/<exp>.jsonl   (local, gitignored)
```

- Each MI300X node serves a **full TP=8 replica** of the 397B model; rank-0 runs an in-job
  `sglang-router` that fans out over all replicas behind **one** Cloudflare tunnel.
- The tunnel URL **changes on every restart** → the collector re-reads the blob router-json each run.
- **Served model id MUST be exactly** `Qwen/Qwen3.5-397B-A17B` (the collector/agent send this name).

---

## 3. Step 1 — clone

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/M-Myron/SWE-Master.git
cd SWE-Master
git checkout main          # should already be on main @ 2a342a1
```

---

## 4. Step 2 — recreate the pieces that are NOT in git

These do **not** come with the clone. Recreate all of them.

### 4.1 Credentials (`cred/` — entirely gitignored)

`cred/.gitignore` is `*` + `!.gitignore`, so **nothing** in `cred/` is pushed. You must recreate:

**(a) `cred/zhibinmain_murongma_sas.url`** — read/write SAS URL for the blob the serving job
publishes to and the collector reads from. **DO NOT paste the token into this handoff.** Copy the
real file from the old box over a secure channel (scp/rsync), OR regenerate a fresh SAS:

- Where: Azure Portal → Storage account **`zhibinmain`** → (container **`murongma`**) →
  *Shared access signature* → permissions read+list (+write if you also publish) → Generate.
- File format (single non-comment line is the URL; `chmod 600`, never commit):
  ```
  # comments allowed (ignored)…
  https://zhibinmain.blob.core.windows.net/murongma?<SAS-QUERY-STRING>
  ```
- **Expiry:** SAS tokens expire. If `blob_sas.sh` starts failing with auth errors, regenerate.

**(b) `cred/load_sas.sh`** — tiny helper (also gitignored). Recreate verbatim:
```bash
#!/usr/bin/env bash
# Print just the SAS URL (no comments).
F="${1:-zhibinmain_murongma_sas.url}"
D="$(dirname "${BASH_SOURCE[0]}")"
grep -v '^#' "$D/$F" | grep -v '^[[:space:]]*$' | head -1
```

**(c) `cred/.gitignore`** — recreate verbatim:
```
# Never commit this directory
*
!.gitignore
```

> `trajectory_collection/blob_sas.sh` (committed) reads the SAS from
> `cred/zhibinmain_murongma_sas.url` and **never echoes it**. Needs `azcopy` on PATH.

### 4.2 Conda environments (two of them)

**`swe-master`** — runs the R2E-Gym rollouts (Python 3.12):
```bash
conda create -y -n swe-master python=3.12 && conda activate swe-master
cd ~/code/SWE-Master
# swebench forks ship as wheels at the repo root:
pip install ./swebench_fork_swegym-2.0.13-py3-none-any.whl \
            ./swebench_fork_swerebench-4.0.3-py3-none-any.whl \
            ./swesmith-0.0.7-py3-none-any.whl
pip install -e R2E-Gym            # installs r2e-gym 0.1.0 + deps
# verify key pins (must roughly match):
#   r2e-gym 0.1.0 (editable) | swebench 3.0.2 | openai 2.38.0 | docker 7.1.0
#   datasets 2.19.0 | tenacity 9.1.4
```
> If `pip install -e R2E-Gym` misses anything, `RL_ENV.txt` / `SFT_ENV.txt` at the repo root are
> full pip freezes from the training envs you can cross-reference (they are supersets).

**`amlt10`** — submits/monitors the Singularity serving job (Python 3.8, **Amulet 10.14.0**):
```bash
conda create -y -n amlt10 python=3.8 && conda activate amlt10
# Amulet (amlt) is the MSR-internal experiment CLI — install from the internal feed:
pip install -U "amlt==10.14.0"     # use your internal/Singularity pip index
amlt --version                      # expect 10.14.0
```
> `amlt` is Microsoft-internal; install it per your team's Singularity onboarding (internal pip
> index). 10.14.0 is what the original box used.

### 4.3 Docker host (for the rollout containers)

The collector runs each instance inside its SWE-bench eval image via a **local docker daemon**.
R2E-Gym hardcodes the TCP endpoint `127.0.0.1:2375`, so a `socat` bridge is required.

- **Daemon:** rootless or root dockerd, **data-root on a big disk**. Original box: docker `29.4.3`,
  `root=/datadisk/docker`, `/datadisk` = 880 G total / ~575 G free.
- **Disk budget:** peak ≈ `wave_images × ~4 GB`. With `wave_images=48` that's ~190 GB transient;
  keep **≥ 300 GB free**.
- **socat bridge** (R2E-Gym expects `tcp://127.0.0.1:2375`):
  ```bash
  # one-shot; the collector also auto-starts this, but you can pre-create it:
  socat TCP-LISTEN:2375,bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:/var/run/docker.sock &
  ```
- The collector's preflight blanks `DOCKER_TLS_VERIFY` / `DOCKER_CERT_PATH` and verifies the TCP
  endpoint; it aborts with a clear `FATAL:` if docker is unreachable.

### 4.4 amlt project (needed to submit the serving job)

`amlt run` uses an **amlt project** for its own bookkeeping/storage (separate from the job-data
blob). On the old box the project is `sing_mi300_test`, with `.amltconfig`:
```json
{"project_name": "sing_mi300_test", "storage_account_name": "zhibinmain",
 "container_name": "amulet", "blob_storage_account_name": "zhibinmain",
 "registry_name": "projects", "version": "10.14.0"}
```
Recreate/activate a project pointing at storage you control (the `amulet` container on `zhibinmain`
above, or your own):
```bash
conda activate amlt10
az login                                   # Azure auth for amlt + blob + ACR
amlt project create sing_mi300_test zhibinmain amulet   # name, storage acct, container
#   (or: copy the old project dir; or `amlt project checkout sing_mi300_test`)
docker login msraairgroup.azurecr.io       # so amlt can pull the serving image (or use UAI)
```
> Note: the **job-data** storage (where the router URL is published) is parameterized **separately**
> in the YAML (`SING_BLOB_ACCOUNT=zhibinmain` / `SING_BLOB_CONTAINER=murongma`, see §5). The amlt
> *project* storage (`amulet`) is only amlt's internal store.

### 4.5 (Optional) carry over already-collected data

`collect_runs/` is gitignored (large jsonl dumps) and lives only on the old box
(`~/code/SWE-Master/trajectory_collection/collect_runs/`). Collection is resumable **by
instance_id against the local `<exp>.jsonl`**, so on a fresh box it simply re-collects. To keep the
196 swegym records already done, copy the dir over first:
```bash
rsync -avz OLDBOX:~/code/SWE-Master/trajectory_collection/collect_runs/ \
           ~/code/SWE-Master/trajectory_collection/collect_runs/
```

---

## 5. Environment variables

### 5.1 Serving job — tenant/infra identifiers (parameterized via amlt `env_defaults`)

The 3 serving YAMLs read these via an `env_defaults:` block. **They resolve to the defaults below
with no action needed** — submitting works out-of-the-box. Export an override only to retarget a
different Singularity tenant/workspace/storage.

| Env var | Default (current tenant) | What it sets |
|---|---|---|
| `SING_TARGET` | `omai-aue-vc` | Singularity target (MI300X, australiaeast) |
| `SING_WORKSPACE` | `msraairwsws` | AML workspace name |
| `SING_REGISTRY` | `msraairgroup.azurecr.io` | ACR for the serving image |
| `SING_BLOB_ACCOUNT` | `zhibinmain` | blob account for job data (router URL publish) |
| `SING_BLOB_CONTAINER` | `murongma` | blob container for job data |
| `SING_JOB_UAI` | `/subscriptions/762905fc-41fb-4bfb-8e41-478b86cb99ab/resourceGroups/system_yeyun/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msraairwsid` | user-assigned managed identity for the job |

```bash
# default tenant — nothing to set. To retarget, e.g.:
# export SING_TARGET=my-cluster SING_WORKSPACE=my-ws SING_BLOB_ACCOUNT=myacct ...
```
> These are **infra identifiers, not secrets**. The serving image is
> `msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot`.

### 5.2 Collection — runtime overrides (all optional; sane defaults baked in)

`collect.sh <dataset> [K] [WAVE_IMAGES] [MAX_WORKERS]` plus optional env:

| Env var | Default | Meaning |
|---|---|---|
| `URL` | *(read from blob)* | router base URL; if unset, resolved from the router-json on blob |
| `ROUTER_JSON` | `sglang_workers/sing_sglang_router_qwen35_397b_8node_v6_router.json` | blob path of the router-json to read |
| `TEMP` | `0.6` | sampling temperature |
| `MAX_STEPS` | `100` | max agent steps per instance |
| `USE_FN_CALLING` | `True` | function-calling mode (required for Qwen3.5 tool use) |
| `SWE_MASTER_PY` | `~/miniconda3/envs/swe-master/bin/python` | rollout interpreter |
| `OUT_ROOT` | `<repo>/trajectory_collection/collect_runs` | output root |
| `EXP` | `<dataset>_full` | run/experiment name (the resume key) |

> The collector sets the agent's `OPENAI_API_BASE=<url>/v1`, `OPENAI_API_KEY=not-needed`,
> `DOCKER_TLS_VERIFY=""` internally — you don't set those.

---

## 6. Step 3 — bring-up commands (new box → running collection)

```bash
conda activate amlt10
cd ~/code/SWE-Master/trajectory_collection/serving

# ── A. Is a serving job already up? (reuse it — cold start is ~13 min)
../blob_sas.sh cat sglang_workers/sing_sglang_router_qwen35_397b_8node_v6_router.json
#   -> {"url": "...trycloudflare.com", "replicas": N, ...}  => skip to step C.

# ── B. Otherwise submit one. IMPORTANT: run amlt in the FOREGROUND (see Gotchas).
#      8-node (max throughput) — pick the YAML that matches the SKU you want:
amlt run sing_sglang_serve_multinode_qwen35.yaml sing_sglang_router_qwen35_397b_8node_v6
#      or 1-node (free-GPU-friendly):
#   amlt run sing_sglang_serve_1node_qwen35.yaml  sing_sglang_router_qwen35_397b_1node_v1
#   answer the interactive "description" prompt when asked. Wait ~13 min for /health.

# ── C. Watch it come up (live dashboard of per-replica load/health):
./monitor_serving.sh --watch 6
#   or job detail: ./watch_sglang_router.sh

# ── D. Run collection (separate shell). Resumable: re-run the SAME line to continue.
conda activate swe-master
cd ~/code/SWE-Master/trajectory_collection
./collect.sh swegym  0 48 48      # ALL swegym (2438), 48 images/wave, 48 workers
#   ./collect.sh swesmith 0 8 48  # ALL swesmith (~266 inst/image -> small wave_images)
#   ./collect.sh swerebench 0 8 6 # ALL swerebench (6542)
#   smoke test first:  ./collect.sh swerebench 2 2 4

# ── E. Monitor collection progress:
cat collect_runs/swegym_full/progress.json
wc -l collect_runs/swegym_full/swegym_full.jsonl
tail -f collect_runs/swegym_full/run.log
```

### If the router URL points at a different exp than v6
Pass `ROUTER_JSON=sglang_workers/<your_exp>_router.json ./collect.sh ...`, or just
`URL=https://<slug>.trycloudflare.com ./collect.sh ...`.

---

## 7. Operating a running collection

- **Resume:** re-run the exact same `./collect.sh` line; done instances (by `instance_id` in
  `<exp>.jsonl`) are skipped, finished images aren't even re-pulled.
- **Quarantine/retry of bad records:** the driver auto-detects degenerate trajectories
  (thought-rate ≤ 50% or ≥ 30% empty-`<function=>`/"forgot" steps) and moves them to
  `<exp>_failed.jsonl` for re-collection. Pass `--no_retry_failed` to disable.
- **Self-healing patches:** `collect.py` re-applies the three R2E-Gym fixes (Qwen3.5 fn-calling
  allow-list, `reasoning_content` parser-select, docker TLS/cert blanking) at every preflight, so a
  stray `git checkout`/reset can't silently corrupt data quality.
- **Quality tell-tale for SFT:** per-trajectory **thought-rate**. Healthy (thinking-ON) ≈ 100% of
  steps have `<think>`; a broken run ≈ 0%. `run.log` is append-across-runs — slice from the last
  `router OK` line to judge the *current* run.

---

## 8. Gotchas / hard-won lessons (read before you debug)

1. **`amlt run` must run in the FOREGROUND.** `--yes` does **not** suppress the interactive
   "Provide a description for this experiment" prompt; with stdout redirected it **hangs forever**.
   Run it in a real terminal and answer the prompt.
2. **Router URL changes on every tunnel restart** — never hardcode it; the collector re-reads the
   blob router-json each run. If rollouts suddenly 404/refuse, re-read the router-json.
3. **Cold start ≈ 13 min** (≈10 min weights + ≈3 min CUDA-graph capture + aiter MoE JIT) before
   `/health` is 200. Don't panic early.
4. **Served model id must be exactly** `Qwen/Qwen3.5-397B-A17B`.
5. **Router load-balancing:** `BALANCE_ABS` must be **<< max_workers** or all requests pile on one
   prefix-owning replica and the detokenizer melts. Baked: 8 (8-node) / 16 (2-node).
6. **SAS expiry:** if `blob_sas.sh` fails auth, regenerate `cred/zhibinmain_murongma_sas.url` (§4.1).
7. **Disk:** rollout images are ~4 GB each; keep ≥300 GB free on the docker data-root. Waves rmi
   their images, but a too-large `wave_images` can still fill the disk.
8. **The serving image is patched for uid-9000 Singularity pods** (5 baked fixes). If you rebuild it,
   keep those fixes (see `serving/README.md` → "The Docker image"). A runtime `export` does **not**
   reach SGLang's TP-worker subprocesses — fixes must be in the image.
9. **R2E-Gym is not a submodule** — it shares the SWE-Master `.git`. Don't try to `git submodule`
   it; just edit in place (changes are tracked by the SWE-Master repo).

---

## 9. Live state snapshot (original box, 2026-06-09)

| Thing | Value |
|---|---|
| Serving (primary) | `sing_sglang_router_qwen35_397b_8node_v6` — 8 nodes, 7/8 replicas healthy |
| Router URL | `https://gore-isaac-life-carnival.trycloudflare.com` (will change on restart) |
| Serving (backup) | `sing_sglang_router_qwen35_397b_2node_v1` (idle) |
| Collection | `swegym` 196 / 2438 (`waves_done=2`), resumable |
| Docker | 29.4.3, root `/datadisk/docker`, `/datadisk` 880 G (~575 G free) |
| Conda | `swe-master` (py3.12, rollouts) • `amlt10` (py3.8, amlt 10.14.0) |

> These are point-in-time. On the new box you'll submit a fresh serving job and get a new URL.

---

## 10. Where to read more (in-repo)

- `trajectory_collection/README.md` — the collection driver, wave model, schema, every flag.
- `trajectory_collection/serving/README.md` — serving architecture, every knob, the 5 image
  fixes, scaling replicas, monitoring, troubleshooting.
- `copilot_memory/sglang_multinode_serving_guide.md` — serving bring-up runbook + failure modes.
- `copilot_memory/e2e_rollout_collection_playbook.md` — end-to-end collection runbook.

**First moves for a new AI session:** read this file, then the two READMEs above, then check whether
a serving job is already healthy (§6.A) before submitting a new one.
