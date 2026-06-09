# Serving — the rollout LLM (multi-node SGLang router on MI300X)

The trajectory collector ([../README.md](../README.md)) needs an OpenAI-compatible
endpoint serving the policy model. This directory holds everything to stand up that
endpoint: **N MI300X nodes each serve a full TP=8 replica of Qwen3.5-397B-A17B, and
rank-0 runs an in-job `sglang-router` that fans out over all replicas behind a single
Cloudflare tunnel** — giving the collector one URL with ~N× throughput.

> These jobs run on **Singularity (amlt)**, not on the dev box. The dev box only
> *submits* them and *reads* the published router URL from blob. The collector then
> reads that same URL (see [`../collect.py`](../collect.py) `resolve_url`).

---

## Table of contents
1. [Architecture](#architecture)
2. [Files in this directory](#files-in-this-directory)
3. [The Docker image (why it's patched)](#the-docker-image-why-its-patched)
4. [Submit a serving job](#submit-a-serving-job)
5. [Get the router URL (what the collector consumes)](#get-the-router-url-what-the-collector-consumes)
6. [Configuration — every knob](#configuration--every-knob)
7. [Scaling replicas](#scaling-replicas)
8. [Monitoring](#monitoring)
9. [The serve script (supervisor) explained](#the-serve-script-supervisor-explained)
10. [Hard-won fixes baked in](#hard-won-fixes-baked-in)
11. [Troubleshooting](#troubleshooting)

---

## Architecture

```mermaid
flowchart LR
    subgraph job["amlt job (Nx192G8-MI300X, 1 process/node)"]
      direction TB
      R0["rank 0\nSGLang replica :8000 (TP=8)\n+ sglang-router :30000\n+ cloudflared tunnel"]
      R1["rank 1\nSGLang replica :8000 (TP=8)"]
      R2["rank 2\nSGLang replica :8000 (TP=8)"]
      R1 -- "eth0 100.65.x.x" --> R0
      R2 -- "eth0 100.65.x.x" --> R0
    end
    R0 == "ONE Cloudflare tunnel" ==> CF["https://&lt;slug&gt;.trycloudflare.com"]
    CF --> BLOB[("blob: sglang_workers/&lt;exp&gt;_router.json\n{\"url\": ...}")]
    BLOB --> COL["collector on GCR\ncollect.py resolve_url"]
```

- Each node serves a **full** replica (the 397B model fits on one MI300X node at TP=8).
- **Node discovery via blob**: each node writes its `eth0` IP to
  `sglang_cluster/<exp>/node_<rank>.url`; rank-0 globs them, health-checks each
  (the connectivity proof), and starts the router over the reachable ones.
- Rank-0 publishes the tunnel URL to `sglang_workers/<exp>_router.json`. **The URL
  changes whenever the tunnel restarts**, so the collector always re-reads this file.

---

## Files in this directory

| File | Role |
|---|---|
| `sing_sglang_serve_multinode_qwen35.yaml` | **Main amlt job** (3 replicas). All serving knobs are `export`s in `command:`. |
| `sing_sglang_serve_1node_qwen35.yaml` | Single-node variant (1 replica) for when free GPU is limited. Identical flags, `sku: 1x`. |
| `sing_sglang_serve_multinode.sh` | The supervisor that runs on every node: launch replica → wait `/health` → (rank0) router + tunnel → auto-restart loop. |
| `Dockerfile.sglang_patched_nonroot` | Builds the non-root-ready ROCm SGLang image (5 baked fixes). |
| `monitor_serving.sh` | **Live dashboard** of per-replica load/health/throughput from the dev box (router tunnel + tiny metrics). Shows spread vs concentration. |
| `watch_sglang_router.sh` | Job-detail watcher for the multi-node router job (status, per-rank milestones, crash logs via `../blob_sas.sh`). |
| `watch_sglang_worker.sh` | Job-detail watcher for a single-node worker job (blob-based). |

Both YAMLs run the **same** `sing_sglang_serve_multinode.sh` (with `NODE_COUNT=1` the
single-node case just has rank-0 do everything).

---

## The Docker image (why it's patched)

Image: `msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot`
(bundles SGLang 0.5.11 + `sglang_router` + `qwen3_5.py`).

Singularity pods run as **uid 9000** with only `/tmp` writable, which breaks the stock
ROCm SGLang image in five places. `Dockerfile.sglang_patched_nonroot` bakes the fixes
so the serve script needs no per-issue workarounds (a runtime `export` does **not**
reach SGLang's TP-worker subprocesses — the fixes must be in the image):

| # | Problem (uid 9000) | Baked fix |
|---|---|---|
| 1 | aiter hardcodes `/tmp/aiter_configs` and writes a `.lock` at `import sglang`; baked root-owned → crash | `rm -rf /tmp/aiter_configs` (recreated user-owned at runtime) |
| 2 | CUDA-graph capture shells to `nvcc --version`; root-mode `0700` nvcc → uncaught `PermissionError` | `chmod 0755` every nvcc |
| 3 | `HOME=/` read-only; triton/inductor/HF caches default there | writable `HOME`+cache dirs under `/tmp` (ENV) |
| 4 | inductor path fragile under uid 9000 during capture | `ENV TORCH_COMPILE_DISABLE=1` (CUDA graphs stay on; they're SGLang's own runner) |
| 5 | aiter JIT builds MoE `.so` in `~/.aiter/jit` but imports from read-only pkg dir → `ModuleNotFoundError` after ~190s | `ENV AITER_JIT_DIR=/tmp/aiter_jit` + pre-seed it world-writable |

Rebuild only if you change the base image:
```bash
docker build -f Dockerfile.sglang_patched_nonroot \
  -t msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot .
docker push msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot
```

---

## Submit a serving job

From the dev box, in the conda env that has `amlt` (default `amlt10`):

```bash
conda activate amlt10
cd SWE-Master/trajectory_collection/serving

# 3-replica (recommended for collection)
amlt run sing_sglang_serve_multinode_qwen35.yaml sing_sglang_router_qwen35_397b_3node_v5

# OR single replica (free-GPU-friendly)
amlt run sing_sglang_serve_1node_qwen35.yaml sing_sglang_router_qwen35_397b_1node_v1
```

The job name (last arg) becomes `<exp>`; the router URL is published to
`sglang_workers/<exp>_router.json`. **Cold start is ~13 min** (~10 min weights load +
~3 min CUDA-graph capture + aiter MoE JIT) before `/health` is 200.

> **Tenant/infra identifiers are parameterized via env vars** (an `env_defaults:`
> block in each YAML). They resolve to the current defaults unless you export an
> override, so submitting works out-of-the-box. To retarget a different Singularity
> tenant/workspace, export any of: `SING_TARGET`, `SING_WORKSPACE`, `SING_REGISTRY`,
> `SING_BLOB_ACCOUNT`, `SING_BLOB_CONTAINER`, `SING_JOB_UAI` before `amlt run`.

---

## Get the router URL (what the collector consumes)

```bash
# from serving/ (blob_sas.sh is one level up)
bash ../blob_sas.sh cat sglang_workers/sing_sglang_router_qwen35_397b_3node_v5_router.json
# -> {"url": "https://<slug>.trycloudflare.com", ...}
```

The collector resolves this automatically (its `--router_json` default points at the
v5 file). If you submit under a different `<exp>`, pass it through:
```bash
ROUTER_JSON=sglang_workers/<exp>_router.json ./collect.sh swegym 0 48 48
```

Sanity-check the endpoint:
```bash
URL=$(bash ../blob_sas.sh cat sglang_workers/<exp>_router.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["url"])')
curl -s "$URL/v1/models" | head -c 300          # 200 + model list
```

---

## Configuration — every knob

All serving knobs are `export`s in the YAML's `command:` block (edit the YAML to
change them). The serve script reads them with `${VAR:-default}`.

| Env (in YAML) | Value | Meaning / why |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-397B-A17B` | served model id; **the collector's `--model` must match exactly** |
| `TP_SIZE` | `8` | tensor-parallel across the node's 8 GPUs |
| `MAX_LEN` | `131072` | context length (≥128K to preserve long thinking) |
| `MEM_FRAC` | `0.7` | static HBM fraction. **Critical**: 0.8 left only ~27 GB runtime → long-reasoning decode OOM (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`). 0.7 → ~47 GB headroom. |
| `TOOL_PARSER` | `qwen3_coder` | parses the XML `<function=…>` tool calls |
| `REASONING_PARSER` | `qwen3` | splits `<think>` into `reasoning_content` (thinking is ON) |
| `MAMBA_STRATEGY` | `no_buffer` | **required on AMD MI** for the hybrid Gated-DeltaNet (V2 buffer is NVIDIA-only) |
| `MOE_RUNNER_BACKEND` | `triton` | validated-good on MI300X; `auto`/aiter CK kernels can produce fluent-but-wrong output |
| `SGLANG_USE_AITER` | `1` | AMD Day-0 instruction; pairs with `--attention-backend triton` |
| `POLICY` | `cache_aware` | router LB policy (RadixAttention prefix-aware) |
| `MAX_RUNNING_REQ` | `48` | **per-replica** concurrent-request cap (`--max-running-requests`); protects the single-threaded detokenizer from melting under a burst |
| `BALANCE_ABS` | `16` | cache_aware: divert to least-loaded once a replica exceeds this many in-flight. **Must be < the collector's `max_workers`** (see below) |
| `BALANCE_REL` | `1.3` | cache_aware: divert when the busiest replica is >1.3× the least-loaded |
| `CACHE_THRESHOLD` | `0.5` | cache_aware: min prefix-match rate to still prefer the cache-warm replica (keeps per-rollout KV reuse below the load cap) |
| `PORT` / `ROUTER_PORT` | `8000` / `30000` | replica / router ports (internal) |

Supervisor knobs (env, with defaults in the script):

| Env | Default | Meaning |
|---|---|---|
| `MAX_RESTARTS` | `50` | max auto-restarts per component (sglang / router / tunnel) |
| `HEALTH_FAIL_THRESHOLD` | `12` | consecutive local `/health` failures (~3 min) → replica deemed HUNG → hard restart |
| `LOGSYNC_INTERVAL` | `60` | blob log-sync period (s); higher = lighter on slow blobfuse |
| `LOG_CAP_MB` | `200` | truncate `/tmp/sglang.log` past this (tmpfs = RAM) |
| `METRICS_INTERVAL` | `20` | tiny per-replica metrics-JSON publish period (for the monitor) |

### Routing & load balancing (read this if you scale `max_workers`)

Every R2E-Gym rollout sends the **same** long system-prompt + tool-schema prefix, so
plain `cache_aware` would pin **all** traffic to whichever replica cached that prefix
first — one replica melts at ~50 concurrent while the others (and the router node)
idle. That is exactly what happened in v5.

The fix is **not** to abandon cache locality (you want each rollout's *growing
conversation* to stick to one replica for KV reuse) — it is to make the load-balance
override actually fire. cache_aware only diverts off the cached replica when
`load > BALANCE_ABS` **and** `load > BALANCE_REL × min_load`. The upstream default
`BALANCE_ABS=64` is **larger than our 48-worker client pool**, so a replica's in-flight
count can never reach it → balancing never triggers → total concentration.

Rule of thumb: **keep `BALANCE_ABS` well below the collector's `max_workers`.** With
`max_workers=48` and `BALANCE_ABS=16`, each replica caps around ~16 in-flight before
the router spreads the rest → an even ~16/16/16 split that still reuses each
rollout's conversation prefix. `MAX_RUNNING_REQ=48` is a per-replica hard backstop for
the degraded case (e.g. two replicas restarting) — lower it toward ~24 if a single
replica ever melts.

> `power_of_two` (pick the lighter of two random replicas) is load-aware and tempting,
> but the bundled binding marks it **PD-mode only**, so we stay on tuned `cache_aware`
> for regular (non-disaggregated) serving.

---

## Scaling replicas

Throughput scales with replica count — just change the `sku`:

```yaml
jobs:
- name: sglang_router_qwen35_397b
  sku: 3x192G8-MI300X   # 1x / 2x / 3x / 4x / 8x ... = that many full replicas
```

`AZUREML_NODE_COUNT` flows through automatically; rank-0 discovers and fans out over
all replicas. No other change needed. The collector's parallelism (`max_workers`) can
then go higher — the practical ceiling is total server throughput across replicas.

---

## Monitoring

**Live dashboard (recommended)** — `monitor_serving.sh` shows whether traffic is
**spread** across replicas or **concentrated** on one (the failure mode we fixed). It
reads the router tunnel directly (`/get_loads`, `/workers`, `/health`) — fast, no blob:

```bash
cd SWE-Master/trajectory_collection/serving

./monitor_serving.sh                       # one-shot, fast (router tunnel only)
./monitor_serving.sh <exp> --watch 6       # refresh every 6s
./monitor_serving.sh <exp> --watch 6 --full  # also pull per-replica kv%/throughput from blob
```

The table shows per-replica `health`, in-flight `load` (router view), and a
`spread(max-min load)` line: **≈0 means evenly balanced; a large value means one
replica is hot** (the old concentration bug). With `--full` it adds running/queue/
kv-usage/throughput from each replica's tiny metrics JSON.

**Job-detail watchers** (status, per-rank milestones, node IPs, crash logs):

```bash
conda activate amlt10
./watch_sglang_router.sh sing_sglang_router_qwen35_397b_3node_v5   # multi-node router job
./watch_sglang_worker.sh <exp>                                    # single-node worker job
```

Healthy signs: per-rank status `READY`, `node_<rank>.url` present for each rank,
`router will fan out over N replica(s)`, a `{"url": ...}` in the router json, and an
even `load` spread in `monitor_serving.sh`.

### A note on blob I/O (it was a real aggravator)

The in-job `/mnt/murongma` blobfuse mount is **slow**. The old loop copied the entire
(chatty) `/tmp/sglang.log` to blob every 20 s; under 48-concurrent load that log grows
fast, and the repeated copy + blobfuse daemon CPU competed with rank-0's
single-threaded detokenizer (which also shares CPU with the router + tunnel) — a
secondary contributor to rank-0's stalls. It did **not** cause the rank1/rank2 melts
(those were genuine 50-concurrent overload from the routing bug). The serve script now
syncs less often (`LOGSYNC_INTERVAL=60`), caps the log smaller (`LOG_CAP_MB=200`), and
publishes only a **tiny** metrics JSON for monitoring — so live monitoring no longer
depends on copying big logs over blobfuse.

---

## The serve script (supervisor) explained

`sing_sglang_serve_multinode.sh` runs on **every** node and self-heals:

1. `launch_sglang` — start the TP=8 replica on `:8000` with all the flags above.
2. `wait_sglang_ready` — **gate on `/health`** (not `/v1/models`). `/v1/models` returns
   200 as soon as the HTTP server is up — *before* the ~190 s CUDA-graph + aiter-JIT
   warmup during which `/health` is 503. Gating on `/health` means "READY" == truly
   generate-ready, so the hang-detector never mistakes a warming replica for a hung one.
3. Every node publishes its `eth0` IP to `sglang_cluster/<exp>/node_<rank>.url`.
4. **rank 0 only**: `start_router` (over the reachable replicas) → `launch_tunnel`
   (cloudflared) → `publish_router_url` (write the `{"url":...}` json to blob).
5. **Supervisor loop**: independently auto-restart sglang / router / tunnel
   (`MAX_RESTARTS` each). A hung replica (local `/health` fails `HEALTH_FAIL_THRESHOLD`
   times while the process is alive) is hard-restarted. A tunnel restart re-greps the
   **new** `trycloudflare` URL and republishes the json (so "only the tunnel died"
   self-heals).
6. `start_log_sync` mirrors `/tmp/sglang.log` → `sglang_logs/<exp>_rank<rank>.sglang.log`
   on blob every ~20 s (amlt's own log view often 503s); `preserve_crash_log` keeps the
   pre-restart log.

`nice -n 10` on the router + cloudflared keeps rank-0's single-threaded detokenizer
winning CPU.

---

## Hard-won fixes baked in

These were diagnosed the hard way; the configs above already encode them. Don't revert
without re-checking:

| Issue | Symptom | Fix (where) |
|---|---|---|
| **GPU HBM OOM** | `HSA_STATUS_ERROR_OUT_OF_RESOURCES … Free mem : 0 MB`, SIGABRT during long decode | `MEM_FRAC=0.7` (YAML) |
| **fn-calling silently off** | every step "forgot to use a function call"; `Using fn calling: False` | model id allow-list in R2E-Gym `agent.py` includes `qwen3.5` (collector side) |
| **garbled MoE output** | fluent-but-unrelated text, `tool_calls=null` | `MOE_RUNNER_BACKEND=triton` (YAML) |
| **rank0 detokenizer hang** | `Health check failed … detokenizer … 20s`; replica alive but unhealthy | `nice` router/tunnel + removed `tail -f` + hang-detector (script) |
| **v4 kill-loop** | hang-detector killed a still-warming replica right after READY | `wait_sglang_ready` gates on `/health` (script) |
| **5 non-root crashes** | aiter lock / nvcc / cache / inductor / aiter-JIT `PermissionError`/`ModuleNotFoundError` | baked into the Dockerfile |

Full narrative: [`../../copilot_memory/sglang_multinode_serving_guide.md`](../../copilot_memory/sglang_multinode_serving_guide.md)
and [`../../copilot_memory/e2e_rollout_collection_playbook.md`](../../copilot_memory/e2e_rollout_collection_playbook.md).

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| collector `FATAL: router URL not healthy` | job still cold-starting (~13 min) or tunnel cycling | `./watch_sglang_router.sh <exp>`; wait for `/health` 200 + a `{"url":...}` json |
| router json present but `/v1/models` fails | tunnel restarted; URL changed | re-read the json (collector does this each run); the published URL is authoritative |
| `watch` shows replica loads then dies at capture | a non-root image fix missing (custom base) | rebuild from `Dockerfile.sglang_patched_nonroot` |
| OOM / SIGABRT mid-collection | `MEM_FRAC` too high, or a genuinely huge context | keep `MEM_FRAC=0.7`; supervisor auto-restarts and the collector resumes |
| only some replicas healthy | one node unreachable at router start | router fans out over reachable ones (partial degrade); restart the job for full N |
| `amlt status` errors about cryptography/x509 | cosmetic amlt warnings | ignored by the watch scripts' grep filters |
