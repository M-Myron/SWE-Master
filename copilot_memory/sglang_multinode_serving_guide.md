# Serving LLMs on Singularity MI300X with SGLang + SGLang Router (Multi-Node)

**Status: WORKING & VALIDATED (2026-06-07).** This guide captures everything needed to
serve a large model (validated with **Qwen3-Coder-480B-A35B-Instruct**, TP=8) on Azure
**Singularity MI300X** nodes using **SGLang**, optionally fan requests across **N nodes via
the SGLang Router** for ~N× throughput, and expose the endpoint to a remote rollout box
(GCR) over a **Cloudflare tunnel**.

It also documents every problem we hit and fixed along the way, so this is replicable.

> **For the full end-to-end pipeline** (serve → tunnel → **GCR agent rollout → trajectory
> collection**), start at the orchestration playbook:
> **[`e2e_rollout_collection_playbook.md`](./e2e_rollout_collection_playbook.md)**.
> This guide is the serving/image deep-dive it references.


> TL;DR — submit a working single replica:
> ```bash
> conda activate amlt10
> cd /home/v-murongma/code/sing_mi300_test/dind_test
> amlt run sing_sglang_serve.yaml sing_sglang_qwen3_coder_480b_vN -d "..." -y
> ```
> Submit an N-replica router cluster (≈N× throughput, one public URL):
> ```bash
> # edit sku: Nx192G8-MI300X in sing_sglang_serve_multinode.yaml
> amlt run sing_sglang_serve_multinode.yaml sing_sglang_router_480b_Nnode_vN -d "..." -y
> ```
> Read logs (amlt portal log is flaky — use blob):
> ```bash
> bash watch_sglang_router.sh <exp_name>      # multi-node
> bash watch_sglang_worker.sh <exp_name>      # single-node
> bash blob_sas.sh list sglang_logs           # raw blob listing
> ```

---

## 0. Coordinates / environment

| Thing | Value |
|---|---|
| Singularity target | `omai-aue-vc` (MI300X, australiaeast) |
| Workspace | `msraairwsws` |
| amlt project | `sing_mi300_test` (dir `/home/v-murongma/code/sing_mi300_test`) |
| amlt CLI | conda env `amlt10` (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate amlt10`) |
| SKU | `Nx192G8-MI300X` (N nodes × 8×MI300X-192G); `process_count_per_node: 1`, `mpi: False` |
| Managed identity (UAI) | `/subscriptions/762905fc-.../resourceGroups/system_yeyun/.../msraairwsid` |
| ACR (reachable from GCR + Singularity) | `msraairgroup.azurecr.io` |
| Blob storage | account `zhibinmain`, container `murongma`, mounted in-job at `/mnt/murongma` |
| Working image | `msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot` |
| Scripts dir | `/home/v-murongma/code/sing_mi300_test/dind_test/` |

Pod runtime facts that drive almost every fix below:
- Singularity pods run as **uid 9000** (`aiscuser`), **CapEff=0** (no root, no sudo).
- `HOME=/` is **read-only**; `/sgl-workspace/*` is root-owned read-only.
- **Only `/tmp` is writable**, plus the blob mount `/mnt/murongma` and `/scratch` (local fast disk).
- Outbound internet works (so Cloudflare tunnel + HF download work); inbound does not (hence the tunnel).

---

## 1. Architecture

### Single node (one full replica)
```
Rollout (GCR)  --HTTPS-->  Cloudflare quick-tunnel  -->  SGLang :8000 (TP=8, one MI300X node)
```

### Multi-node with in-job router (what you want for throughput)
```
                                   ONE amlt job, sku: Nx192G8-MI300X
                         ┌───────────────────────────────────────────────┐
Rollout (GCR) --HTTPS--> │ rank0: sglang-router :30000  --internal eth0--> sglang :8000 (replica 0, TP=8)
        (1 tunnel)       │                              --internal eth0--> sglang :8000 (replica 1, TP=8)
                         │                              --internal eth0--> sglang :8000 (replica N-1, TP=8)
                         └───────────────────────────────────────────────┘
```
- **Every node** serves a *full* TP=8 replica of the same model on `0.0.0.0:8000` (internal only).
- **Rank 0 additionally** runs `sglang_router`, which load-balances across all replicas over
  the fast internal **eth0** fabric, and opens the **single** Cloudflare tunnel to the router.
- The rollout points at the one router URL → ~N× throughput, only **1 tunnel** total.

Why the router needs **SGLang** workers (not vLLM): the router health-checks SGLang-native
endpoints (`/v1/models`, `/get_model_info`) and `cache_aware` policy uses RadixAttention
prefix reuse — that only exists on SGLang servers. (`--backend openai` is a single-endpoint
proxy with **no** load balancing, so it is not what we want.)

---

## 2. The image (the hard part) — non-root MI300X SGLang

### 2.1 Base images
- Official SGLang ROCm MI300X images exist: `lmsysorg/sglang:v0.5.12-rocm720-mi30x`
  (also `v0.5.11`, `v0.5.9`). ROCm 7.2 matches the cluster — **no need to build `sgl_kernel`
  from source.** Mirror to ACR: `docker pull … && docker tag … msraairgroup.azurecr.io/sglang:<tag> && docker push …`.
- We ended up basing the final image on a colleague's `msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched`
  (sglang 0.5.11). **That base is `User: root` and is NOT non-root-ready** — it fails the same
  way the raw lmsys image does. All non-root fixes must be layered on top (below).
- The router (`sglang_router`, pip pkg, a Rust LB) is **bundled inside these images** already
  (`python3 -c "import sglang_router"` works).

### 2.2 The 5 non-root fixes (all baked into the image)
`Dockerfile.sglang_patched_nonroot` (in the scripts dir). Build context can be any small dir.

```dockerfile
ARG BASE=msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched
FROM ${BASE}

# (1) aiter hardcodes its tuned-GEMM cache to /tmp/aiter_configs (no env override) and writes
#     a .lock there at `import sglang`. Base bakes that dir root-owned 0755 -> uid 9000 cannot
#     create the lock -> PermissionError. Remove it; aiter recreates it user-owned at runtime
#     (source CSVs stay under /sgl-workspace/aiter/aiter/configs, re-merged in ms, no GPU re-tune).
RUN rm -rf /tmp/aiter_configs

# (2) torch inductor (triggered during CUDA-graph capture) shells out to `nvcc --version` for a
#     debug-repro string; on ROCm cuda.is_available()==True so it tries nvcc. Image nvcc is
#     root-mode 0700, and torch only catches FileNotFoundError/CalledProcessError, NOT
#     PermissionError -> crash AFTER weights fully load. Make nvcc world-exec (no-op if absent).
RUN for n in $(bash -lc 'command -v nvcc' 2>/dev/null) /usr/local/cuda/bin/nvcc /opt/rocm/bin/nvcc; do \
      [ -e "$n" ] && chmod 0755 "$n" || true; done; true

# (3) HOME=/ is read-only for uid 9000; triton/inductor/HF caches default to $HOME/.cache.
#     Provide writable HOME + cache dirs under /tmp (the only writable mount). BAKED ENV so it
#     is inherited by every spawned TP-worker subprocess.
RUN mkdir -p /tmp/home/.cache /tmp/triton_cache /tmp/torchinductor_cache \
 && chmod -R 1777 /tmp/home /tmp/triton_cache /tmp/torchinductor_cache
ENV HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/home/.cache \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

# (4) SGLang @torch.compiles small helpers (e.g. get_masked_input_and_mask) during CUDA-graph
#     capture even WITHOUT --enable-torch-compile. The inductor path is fragile under uid 9000.
#     Disable torch.compile globally as BAKED ENV (TORCH_COMPILE_DISABLE, NOT TORCHDYNAMO_DISABLE
#     — only the former sets torch._dynamo.config.disable=True in this build). CUDA graphs are
#     SGLang's own CudaGraphRunner (independent of torch.compile) and STAY enabled.
ENV TORCH_COMPILE_DISABLE=1

# (5) aiter JIT-compiles fused-MoE CK kernels at RUNTIME. Build vs import dirs must agree
#     (aiter/jit/core.py): with the package dir read-only and AITER_JIT_DIR unset, it builds the
#     .so into ~/.aiter/jit but the import branch still does importlib.import_module("aiter.jit.<md>")
#     from the read-only package dir -> ModuleNotFoundError after a ~190s build (crash at capture).
#     Setting AITER_JIT_DIR makes BOTH branches use this writable dir (also inserted into sys.path).
ENV AITER_JIT_DIR=/tmp/aiter_jit
RUN cp -r /sgl-workspace/aiter/aiter/jit /tmp/aiter_jit && chmod -R 1777 /tmp/aiter_jit
```

Build + push (on the GCR box, docker works without sudo, ACR already authed):
```bash
cd /home/v-murongma/code/sing_mi300_test/dind_test
PNR=msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot
docker build -f Dockerfile.sglang_patched_nonroot -t "$PNR" .
docker push "$PNR"
```

### 2.3 Validate the image as uid 9000 WITHOUT a GPU
You can reproduce/verify the non-root fixes locally with `--user 9000:9000` (the lock + JIT-dir
checks don't need a GPU; a full `import sglang` does, because it needs `from aiter import dtypes`
which requires real MI300X):
```bash
docker run --rm --user 9000:9000 "$PNR" bash -lc '
echo "HOME=$HOME AITER_JIT_DIR=$AITER_JIT_DIR TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE"
python3 -c "from pathlib import Path; import os; p=Path(\"/tmp/aiter_configs/\"); p.mkdir(parents=True,exist_ok=True); lk=\"/tmp/aiter_configs/x.lock\"; fd=os.open(lk,os.O_CREAT|os.O_EXCL); os.close(fd); os.remove(lk); print(\"aiter lock OK owner\", os.stat(\"/tmp/aiter_configs\").st_uid)"
python3 -c "import aiter.jit.core as c,sys; d=c.get_user_jit_dir(); print(\"jit_dir\",d,\"on_syspath\",d in sys.path,\"writable\",os.access(d,os.W_OK))" 2>/dev/null
python3 -c "import torch._dynamo as d; print(\"dynamo.disable\", d.config.disable)"
'
# Expect: aiter lock OK owner 9000 ; jit_dir /tmp/aiter_jit on_syspath True writable True ; dynamo.disable True
```

---

## 3. The serve scripts & configs

All live in `/home/v-murongma/code/sing_mi300_test/dind_test/`:

| File | Purpose |
|---|---|
| `Dockerfile.sglang_patched_nonroot` | The 5 non-root fixes layered on the base image |
| `sing_sglang_serve.sh` / `.yaml` | **Single-node** SGLang serve + tunnel + worker registration |
| `sing_sglang_serve_multinode.sh` / `.yaml` | **Multi-node** serve + in-job router + single tunnel |
| `gcr_sglang_router.sh` | (alt topology) router running ON GCR, fanning over N single-node tunnels |
| `watch_sglang_worker.sh` | watch a single-node job (status + blob log) |
| `watch_sglang_router.sh` | watch a multi-node router job (per-rank status, IP rendezvous, connectivity proof, router URL) |
| `blob_sas.sh` | read job-written blobs from GCR via SAS (GCR `/mnt/murongma` mount is flaky) |

### 3.1 The critical SGLang launch flags (MI300X + Qwen3-Coder)
```bash
SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tp-size 8 \
  --context-length 131072 \
  --mem-fraction-static 0.90 \
  --attention-backend triton \          # REQUIRED on AMD Instinct
  --tool-call-parser qwen3_coder \       # SGLang's parser name (vLLM's was qwen3_xml)
  --moe-runner-backend triton \          # CRITICAL: 'auto' picks aiter CK MoE -> GARBAGE output
  --trust-remote-code \
  --watchdog-timeout 1200 \              # big model: weight load is slow
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}'
```
- `--moe-runner-backend triton` is **not optional** — see §6 landmine #6.
- The official cookbook for this model is baked in the image at
  `/sgl-workspace/sglang/.claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm/qwen3-coder-480b-a35b-instruct.yaml`.

### 3.1b Per-model flag differences (what changes when you swap models)
Our SGLang 0.5.11 image already ships the model code for both (`qwen3_moe.py`, `qwen3_5.py`),
so **no image rebuild** is needed to switch models — only the launch flags change. The serve
scripts expose each as an env var.

| Flag / env | Qwen3-Coder-480B-A35B-Instruct | **Qwen3.5-397B-A17B** |
|---|---|---|
| `MODEL` / `--model-path` | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | `Qwen/Qwen3.5-397B-A17B` |
| `MEM_FRAC` / `--mem-fraction-static` | `0.90` | **`0.7`** — 0.8 OOMs under long reasoning (see §6 #10) |
| `REASONING_PARSER` / `--reasoning-parser` | *(none)* | `qwen3` (separates `<think>` into `reasoning_content`) |
| `MAMBA_STRATEGY` / `--mamba-scheduler-strategy` | *(n/a)* | **`no_buffer`** — REQUIRED on AMD MI (hybrid Gated-DeltaNet; `extra_buffer`/V2 is NVIDIA-only) |
| `TOOL_PARSER` / `--tool-call-parser` | `qwen3_coder` | `qwen3_coder` (same) |
| `MOE_RUNNER_BACKEND` / `--moe-runner-backend` | `triton` | `triton` |
| `--attention-backend` | `triton` | `triton` |
| TP / nodes | `tp=8`, 1 node | `tp=8`, 1 node (MI300X 192GB BF16) |

**Qwen3.5 thinking is ON by default.** It will emit lots of reasoning tokens. The reasoning is
returned in `reasoning_content` (thanks to `--reasoning-parser qwen3`) and is useful for SFT
training data. For tool-calling / agent rollouts where you do NOT want thinking, pass per request:
`"chat_template_kwargs": {"enable_thinking": false}` (then `content` is the direct answer and
`tool_calls` are clean). Validated: thinking-on → `reasoning_content` populated; thinking-off →
clean `execute_bash` tool_call.

Qwen3.5 multi-node config: `sing_sglang_serve_multinode_qwen35.yaml` (sets all the above as env).


### 3.2 Multi-node yaml (the important bits)
`sing_sglang_serve_multinode.yaml`:
```yaml
target: { service: sing, name: omai-aue-vc, workspace_name: msraairwsws }
environment:
  image: sglang:v0.5.11-rocm700-mi30x-patched-nonroot
  registry: msraairgroup.azurecr.io
code: { local_dir: $CONFIG_DIR/, remote_dir: murongma/dind_test }
storage:
  data: { storage_account_name: zhibinmain, container_name: murongma, mount_dir: /mnt/murongma }
jobs:
- name: sglang_router_qwen3_coder_480b
  sku: 2x192G8-MI300X          # <-- N nodes = N replicas. Change to 4x / 8x to scale.
  sla_tier: Premium
  priority: High
  mpi: False
  process_count_per_node: 1
  identity: managed
  command:
    - export MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct"
    - export PORT=8000
    - export ROUTER_PORT=30000
    - export TP_SIZE=8
    - export MAX_LEN=131072
    - export MEM_FRAC=0.90
    - export TOOL_PARSER=qwen3_coder
    - export MOE_RUNNER_BACKEND=triton
    - export POLICY=cache_aware
    - export NODE_COUNT=$${AZUREML_NODE_COUNT:=1}   # NOTE: amlt needs $$ to emit one $
    - export NODE_RANK=$${NODE_RANK:=0}
    - export MASTER_ADDR=$${MASTER_ADDR:=localhost}
    - chmod +x sing_sglang_serve_multinode.sh
    - bash sing_sglang_serve_multinode.sh
  submit_args:
    env:
      {
        "NCCL_DEBUG": "INFO",
        "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/762905fc-41fb-4bfb-8e41-478b86cb99ab/resourceGroups/system_yeyun/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msraairwsid"
      }
  tags: [ "Project_Name:Sing_SGLang_Router_Qwen3_Coder_480B" ]
```

**Singularity multi-node convention** (learned from `code/cluster-demo/Singularity/demo.yaml`):
the cluster injects `AZUREML_NODE_COUNT`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`.
`sku: Nx<sku>` gives N nodes. In amlt yaml, reference them with **`$$`** escaping
(`$${AZUREML_NODE_COUNT:=1}`) so amlt passes a literal `$` to the shell.

### 3.3 How the multi-node script works (`sing_sglang_serve_multinode.sh`)
1. **Every node** starts a full TP=8 SGLang replica on `:8000`, waits for `/v1/models`.
2. Each node computes its **eth0 IP** (`ip -4 addr show eth0 | grep -oP 'inet \K[0-9.]+'`) and
   writes `http://<ip>:8000` to a shared blob rendezvous file
   `/mnt/murongma/sglang_cluster/<exp>/node_<rank>.url` (+ heartbeats it).
3. **Workers (rank ≥ 1)** then just keep serving (no tunnel).
4. **Rank 0** globs all `node_*.url`, **health-checks each** over the internal network (this is
   the intra-job connectivity proof — printed to the log), starts
   `python3 -m sglang_router.launch_router --worker-urls <live IPs> --policy cache_aware --port 30000`,
   opens **one** cloudflared tunnel to `:30000`, and writes the public URL to
   `/mnt/murongma/sglang_workers/<exp>_router.json` (the file the rollout reads).

This blob-rendezvous design means we do **not** depend on `MASTER_ADDR` knowing worker IPs;
rank 0 discovers and health-checks peers itself, and degrades gracefully if a peer is missing.

---

## 4. Submitting a job

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate amlt10
cd /home/v-murongma/code/sing_mi300_test/dind_test

# Single replica:
amlt run sing_sglang_serve.yaml sing_sglang_qwen3_coder_480b_v8 \
  -d "single-node SGLang Qwen3-Coder-480B" -y

# N replicas behind the in-job router (edit sku: Nx192G8-MI300X first):
amlt run sing_sglang_serve_multinode.yaml sing_sglang_router_480b_2node_v1 \
  -d "2-node SGLang router cluster" -y
```
Notes:
- Always pass `-d "<desc>"` and `-y`; without `-d` amlt prompts interactively in a way that is
  awkward to script.
- If you hit `User identity does not have jobs submit permission` it's a **stale Azure token** →
  `az login --use-device-code`, then resubmit (the cluster RBAC check uses a fresh token).
- amlt is `v10.14.0` in `amlt10`; the `cryptography`/py3.8 deprecation warnings are noise.
- Cancel a superseded run to free MI300X capacity (it's tight): `amlt cancel <exp> --yes`.

---

## 5. Checking logs — DO NOT rely on the amlt portal

**The amlt portal log (`amlt log`) frequently 503s or returns empty.** That is why the serve
scripts **stream the SGLang log to blob** every 20s + a status file. Read those instead.

Two gotchas:
- The GCR dev box's `/mnt/murongma` blobfuse mount is **flaky / often unmounted**. Use the SAS
  helper `blob_sas.sh` (reads the same blob via a SAS URL, no mount needed).
- The SAS cred file: `/home/v-murongma/code/SWE-Master/cred/zhibinmain_murongma_sas.url`.

```bash
cd /home/v-murongma/code/sing_mi300_test/dind_test

# One-shot status for a multi-node router job (status + IP rendezvous + connectivity + router URL):
bash watch_sglang_router.sh <exp_name>

# One-shot status for a single-node job:
bash watch_sglang_worker.sh <exp_name>

# Raw blob access:
bash blob_sas.sh list sglang_logs                       # list streamed logs + status files
bash blob_sas.sh list sglang_cluster/<exp>              # node IP rendezvous files
bash blob_sas.sh cat  sglang_logs/<file>.sglang.log     # full sglang log
bash blob_sas.sh cat  sglang_workers/<exp>_router.json  # the router public URL (rollout endpoint)
```

What "healthy" looks like in the rank-0 log:
```
LOCAL REPLICA READY at http://100.65.x.x:8000
discovered 2/2 replicas
  REACHABLE   http://100.65.28.164:8000          <-- intra-job connectivity proof
  REACHABLE   http://100.65.17.213:8000
router will fan out over 2 replica(s)
router healthy
[ROUTER URL]  https://<slug>.trycloudflare.com
... Capture cuda graph end ... The server is fired up and ready to roll!
```

> Gotcha: when grepping the log for crashes, **exclude** `custom_sigquit_handler=None` — it
> appears in the printed `server_args=ServerArgs(...)` dump and is a false positive for "sigquit".

---

## 6. Problems we solved (the landmine list)

These are in the order we hit them. Every one is now fixed in the image or scripts.

1. **aiter `/tmp/aiter_configs` lock PermissionError at `import sglang`.**
   Base image bakes that dir root-owned; `aiter/jit/core.py` hardcodes the path (no env
   override) and writes a `.lock` there. uid 9000 can't. → image fix #1 (`rm -rf /tmp/aiter_configs`).

2. **`HOME=/` read-only → triton/inductor/HF caches unwritable.** → image fix #3
   (writable `HOME=/tmp/home` + cache dirs, baked ENV).

3. **CUDA-graph capture dies with `PermissionError: 'nvcc'` AFTER weights fully load.**
   torch inductor's normal flow calls `nvcc --version` for a debug string; on ROCm it tries
   nvcc, which is root-0700, and torch doesn't catch `PermissionError`. → image fix #2
   (`chmod 0755 nvcc`) **and** #4 (`TORCH_COMPILE_DISABLE=1`).

4. **Runtime `export`s do NOT reach SGLang's spawned TP-worker subprocesses.** v3 set
   `TORCH_COMPILE_DISABLE` and a PATH nvcc-shim in the serve script and still failed identically
   after 23 min. **Lesson: every non-root fix must be baked into the image** (RUN to fix files +
   ENV inherited by children). Runtime `mkdir /tmp/aiter_configs` is also a no-op because the
   baked dir is root-owned (can't write the lock; can't `rm -rf` it as uid 9000).

5. **aiter MoE CK JIT build/import dir mismatch → `ModuleNotFoundError: aiter.jit.module_moe_ck2stages_…`**
   after a ~190s kernel build, at capture. `get_user_jit_dir()` builds the `.so` into
   `~/.aiter/jit` (because the package dir is read-only) but the import branch loads from the
   read-only package dir unless `AITER_JIT_DIR` is set. → image fix #5
   (`ENV AITER_JIT_DIR=/tmp/aiter_jit` + pre-seed the JIT tree, world-writable).

6. **Model serves but emits FLUENT-BUT-WRONG output on tool/long prompts** (plain short prompts
   were perfect). Not the chat template (the HF tokenizer renders the correct Qwen3-Coder
   `<tools>`/`<tool_call><function=>` format) and not SGLang's server (a hand-rendered correct
   prompt sent raw to `/v1/completions` also garbled). **Root cause: `--moe-runner-backend auto`
   selects aiter CK MoE kernels that are numerically wrong for some shapes on MI300X.**
   → **`--moe-runner-backend triton`** (the official cookbook setting). Fixed: tool calls now
   return clean `tool_calls`.

7. **`amlt log` 503 / empty.** → serve scripts stream the log to
   `/mnt/murongma/sglang_logs/<id>.sglang.log` + a status file every 20s. Read via `blob_sas.sh`.

8. **GCR `/mnt/murongma` mount flaky/absent.** → `blob_sas.sh` (SAS-based blob reads) +
   `watch_*` helpers use it.

9. **Stale Azure token → `jobs submit permission` error.** → `az login --use-device-code`.

10. **Replica SIGABRTs ~1h in with `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`
    (GPU HBM OOM), `scheduler_N crashed with exit code -6`.** Hit on BOTH the 3-node and 1-node
    Qwen3.5 jobs, mid-rollout, during a long **reasoning** decode (aiter MoE GEMM at `M:7565`
    tokens). The server returns degraded/garbage tokens for ~2 min, then the whole replica dies.
    Memory math (per MI300X, ~181 GB usable): weights = **92.78 GB/GPU**; `--mem-fraction-static
    0.8` → static pool 153.6 GB → only **~27 GB runtime headroom**, which long reasoning + untuned
    aiter GEMMs exhaust. → **`--mem-fraction-static 0.7`** (static 134 GB, KV pool ~41 GB, runtime
    headroom ~47 GB). Validated: clean 27-step reasoning rollout, no OOM. (Contributing: we delete
    `/tmp/aiter_configs` for the non-root fix, so GEMMs run untuned `torch solution:0` = more
    scratch memory. Reasoning-ON inflates sequence length, which is the real trigger.)

11. **One sglang crash tore down the WHOLE serving stack.** rank0 ended on `wait $SGL_PID`; any
    replica death fired the `cleanup` EXIT trap which killed the router + tunnel. No supervision.
    → serve script refactored into functions (`launch_sglang`, `wait_sglang_ready`, `start_router`,
    `launch_tunnel`, `publish_router_url`) + a **supervisor loop** (rank0 and workers) that
    auto-restarts sglang / router / tunnel independently (`MAX_RESTARTS=50` each). A dead tunnel is
    re-established and the NEW trycloudflare URL is **republished** to `<exp>_router.json` (so
    rollouts that re-read the file self-heal). `preserve_crash_log()` copies `/tmp/sglang.log` to
    blob before relaunch (the relaunch's `>` was truncating crash evidence). Also: `NCCL_DEBUG`
    `INFO`→`WARN` and a 1 GB cap on the tmpfs log to avoid a *host*-RAM OOM (a separate earlier
    `Killed` at ~60 min).

12. **rank0's SGLang detokenizer HANGS under load (`Health check failed. Server couldn't get a
    response from detokenizer for last 20 seconds`) — replica alive but unhealthy, router routes
    around it.** Seen in the K=10 batch: at peak concurrency (5 running reqs, 141K tokens) rank0's
    detokenizer missed its heartbeat at 02:54:17 and never recovered; **ranks 1 & 2 had ZERO such
    failures** over the same window and kept decoding. **Root cause = CPU/IO contention unique to
    rank0**, which alone runs `sglang_router` + `cloudflared` + the supervisor + log-sync (+ a
    former `tail -f`) *on top of* its full TP=8 replica. The single-threaded detokenizer gets
    starved. **Aggravator = aiter's untuned-GEMM logging flood:** 20,240 GEMM calls/run miss the
    tuned-config CSV (`/tmp/aiter_configs/bf16_tuned_gemm.csv` has 0 entries for our shapes — it's an
    *offline-tuning output*, not shipped pre-populated; dynamic batching also yields 632 distinct M
    values) and fall to `torch_gemm` = `F.linear`. **Precise mechanism (verified against
    `aiter/tuned_gemm.py`):** the fallback matmul itself runs on the **GPU** (rocBLAS, just untuned
    → slower decode), so it is NOT "CPU compute"; the CPU/IO cost is the **unconditional
    `logger.info()` printed on every miss** (the *found* branch is gated by `AITER_LOG_TUNED_CONFIG`,
    the *miss* branch has no guard) — ~20K Python-log lines/run written to the tmpfs `/tmp/sglang.log`,
    which steals host CPU + RAM-I/O from the detokenizer. **NOT** an OOM/crash (0 HSA / 0 SIGABRT).
    → Fixes (serve script): (a) **supervisor hang-detector** — HTTP `/health` with a consecutive-fail
    threshold (`HEALTH_FAIL_THRESHOLD=12` ≈ 3 min) hard-restarts a hung-but-alive replica (the plain
    `kill -0 $SGL_PID` death-check never fired); (b) `nice -n 10` the router + cloudflared so the
    detokenizer wins CPU; (c) removed the pointless `tail -f /tmp/cloudflared.log`; (d) tightened the
    router health check to 10s/8s so a hung replica is dropped + re-added fast. **The batch still
    collected 23/23 clean trajectories on the 2 healthy replicas** (graceful degradation worked).
    For maximum stability at scale, consider dedicating rank0 to **router-only** (don't serve a
    replica on it) — costs one GPU node but removes the contention entirely. To cut the GEMM-log
    flood itself: pre-tune GEMMs offline (`AITER_TUNE_GEMM=1`) for the serving shapes, or suppress
    aiter's logger — enabling `SGLANG_USE_AITER=1` (below) does NOT silence it (the tuned_gemm path
    runs regardless).
    **CRITICAL companion fix (v5):** the hang-detector checks `/health` (generate-ready), so the
    readiness gate `wait_sglang_ready` MUST also gate on **`/health`**, NOT `/v1/models`. `/v1/models`
    returns 200 the moment the HTTP server is up (right after weights load), but the replica then
    spends ~190s in CUDA-graph capture + aiter MoE JIT during which `/health` returns **503**. If you
    declare READY on `/v1/models`, the detector arms ~3 min too early and kills the still-warming
    replica → reload → warm → kill = **kill-loop** (v4 rank2: `/health` 503×8, `/health` 200 count=0,
    killed 1m46s after "READY"). Gating `wait_sglang_ready` on `/health` makes READY == truly
    generate-ready, so the detector only arms after `/health` is 200. Cold start to READY is then
    ~13 min (≈10 min load + ~3 min warmup/JIT) — expected, not a hang.

13. **`SGLANG_USE_AITER=1` was missing (only in a comment).** AMD's Day-0 Qwen3.5-on-Instinct
    article says to use `SGLANG_USE_AITER=1` **and** `--attention-backend triton` (both attn and
    Gated-DeltaNet use Triton kernels on ROCm). We had the triton backend but never `export`ed the
    aiter flag — now set in both yamls. It enables aiter's optimized kernels for the model's ops;
    it is **not** a fix for landmine #12 (the GEMM-log flood comes from aiter's `tuned_gemm`, which
    is active with or without this flag). Low-risk correctness/perf alignment with AMD guidance.

Other facts worth knowing:
- SGLang tool parser for Qwen3-Coder **and Qwen3.5** is **`qwen3_coder`** (vLLM's was `qwen3_xml`).
  The bundled HF chat template (no `--chat-template` override → `chat_template=None`) renders the
  matching `<tools>` / `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
  format, so **template ⟷ parser are exactly aligned** (verified against `tokenizer_config.json`).
- **R2E-Gym `support_fn_calling` allow-list did NOT match `qwen3.5`** (only `qwen3-coder`/`qwen3-max`).
  See §8 — this silently disables tool-calling and is the #1 rollout gotcha for this model.
- `--attention-backend triton` is required on AMD Instinct for this model.
- `sglang_router` is **bundled in the image** (no pip install needed in-job).

---

## 7. Validation evidence

### 7a. 2-node Qwen3-Coder-480B (`sing_sglang_router_480b_2node_v1`)
- Both nodes loaded a full TP=8 480B replica; published eth0 IPs `100.65.28.164` / `100.65.17.213`.
- Rank 0 health-checked **both** over internal eth0 → "router will fan out over 2 replica(s)".
- Router `/workers`: both `is_healthy: true`, tp_size=8.
- Through the router tunnel: `/v1/models` OK, plain `PONG` OK, tool call clean
  (`execute_bash {"command":"ls /testbed"}`).
- **Throughput**: 12 concurrent requests → all HTTP 200, ~1.0s each, **total wall 1.04s**
  (true parallel fan-out, not serialized).

### 7b. 3-node Qwen3.5-397B-A17B (`sing_sglang_router_qwen35_397b_3node_v1`)
- 3 nodes loaded a full TP=8 397B replica; published eth0 IPs `100.65.11.74` / `100.65.28.181` / `100.65.17.156`.
- Router `/workers`: all 3 `is_healthy: true`.
- Through the router tunnel: plain `PONG` OK; **tool call clean** with `enable_thinking:false`
  (`execute_bash {"command":"ls /testbed"}`); **thinking-on** populated `reasoning_content`.
- **Throughput**: 18 concurrent requests → **18/18 HTTP 200**, ~1.42s each, **total wall 1.45s**.
- One-time startup race: the router's own first sanity check can return
  `no_available_workers` (worker circuits not settled yet) — harmless, healthy within seconds.

Scaling: change `sku: Nx192G8-MI300X`. Same script. N replicas → ~N× throughput, still **one** tunnel/URL.


---

## 8. Wiring a rollout to the router

The router exposes a standard OpenAI-compatible API. Point the rollout at it:
```bash
ROUTER_URL=$(bash blob_sas.sh cat sglang_workers/<exp>_router.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["url"])')
export OPENAI_API_BASE="$ROUTER_URL/v1"
export OPENAI_API_KEY="not-needed"
# R2E-Gym: --llm_name openai/Qwen/Qwen3-Coder-480B-A35B-Instruct --use_fn_calling True --scaffold openhands
```
Notes for R2E-Gym specifically (from the GCR smoke work):
- The agent reads `URL` → `OPENAI_API_BASE="${URL}/v1"`, so pass `URL=$ROUTER_URL`.
- The model id sent as `model=` must equal the served `--model-path`
  (`Qwen/Qwen3-Coder-480B-A35B-Instruct`).

### 8.1 Rollout root causes (Qwen3.5 + R2E-Gym) — READ THIS BEFORE COLLECTING TRAJECTORIES

Three independent bugs each produced the **same** symptom — every step logging
`You forgot to use a function call` / mangled `<function=bash>` / Hermes-JSON
`{"name":"bash",...}` with made-up tool names, reward 0. They are easy to confuse; diagnose by
elimination in this order:

1. **fn-calling silently disabled (the deterministic one).** R2E-Gym gates tools behind an
   allow-list: `self.use_fn_calling = use_fn_calling AND support_fn_calling`, where
   `support_fn_calling` is a substring match in `agent.py`. It listed `qwen3-coder`/`qwen3-max`
   but **not `qwen3.5`**, so model id `openai/Qwen/Qwen3.5-397B-A17B` → `support_fn_calling=False`
   → `tools=None` sent → SGLang renders **no** tool schema → the model guesses a generic Hermes
   format with invented names (`bash`, `python`). **Tell-tale: the run log prints
   `Using fn calling: False`, and the model uses tool names that aren't yours.**
   → Fix: add `or "qwen3.5" in self.llm_name.lower()` to the `support_fn_calling` list
   (`R2E-Gym/src/r2egym/agenthub/agent/agent.py`, ~L425). After the fix the log says
   `Using fn calling: True` and step-0 emits a real `execute_bash`/`str_replace_editor` call.
   *Isolation test that misled us:* a raw curl/`litellm` call that passes `tools=[...]` explicitly
   bypasses this gate and returns clean tool calls — so single-emission tests pass while the agent
   fails. Always check `Using fn calling:` in the actual run log.

2. **Server GPU-OOM crash mid-rollout.** See §6 landmine #10. A reasoning rollout grows the
   sequence until an aiter MoE GEMM OOMs; the replica returns garbage for ~2 min then SIGABRTs.
   **Tell-tale: `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... 0 MB` in the sglang log; the rollout sees
   503 / `no_available_workers` around the same time.** → Fix: `--mem-fraction-static 0.7`.

3. **Reasoning dropped from trajectories.** With `--reasoning-parser qwen3`, thinking lands in
   `reasoning_content`, not `content`. R2E-Gym's `custom_parser` read only `content`, so the
   `thought` field came out empty even when tool calls were fine. → Fix (agent.py): when
   `reasoning_content` is present, route through `reasoning_parser` (fn path) and fold it into
   `thought` as `<think>…</think>` (non-fn path); also make the qwen parser-select case-insensitive
   (`self.llm_name.lower()`) so `Qwen` matches. After the fix every step has a populated `<think>`.

**Validated end-to-end (1-node Qwen3.5-397B, MEM_FRAC 0.7, fn-calling on):** swerebench instance
`fair-workflows__nanopub-140` → `exit_reason=agent`, **reward 1.0**, 27 steps, **27/27 real tool
calls** (`execute_bash`×17, `str_replace_editor`×9, `submit`×1), **27/27 steps with `<think>`**
(15 K reasoning chars), correct `output_patch` (3-arg click callback). The chat template + `qwen3_coder`
parser were correct the whole time — these were agent-integration + memory-headroom bugs, not a
template/parser mismatch.

---

## 9. Cold-start / cost notes (future optimizations)

- Weight load for 480B is ~11–14 min/node; each node downloads ~480 GB to its **own**
  ephemeral `/tmp/home/.cache` on first run (the multinode script does **not** set `HF_HOME`).
  To speed cold starts, set `HF_HOME=/scratch/hf_cache` (≈28 TB local fast disk) or point at a
  shared blob HF cache. (The single-node `sing_sglang_serve.sh` already uses `/scratch/hf_cache`.)
- `cache_aware` routing (RadixAttention prefix reuse) is ideal for SWE rollouts where the same
  repo context recurs across agent steps. Other policies: `round_robin`, `random`, `power_of_two`.
- One tunnel per job (the router's). If you run multiple router jobs, each gets its own
  `trycloudflare.com` URL + its own `<exp>_router.json`.

---

## 10. Quick reference (copy/paste)

```bash
# ---- env ----
source ~/miniconda3/etc/profile.d/conda.sh && conda activate amlt10
cd /home/v-murongma/code/sing_mi300_test/dind_test

# ---- (re)build the non-root image (only if base changed) ----
PNR=msraairgroup.azurecr.io/sglang:v0.5.11-rocm700-mi30x-patched-nonroot
docker build -f Dockerfile.sglang_patched_nonroot -t "$PNR" . && docker push "$PNR"

# ---- submit N-replica router cluster (edit sku: Nx192G8-MI300X) ----
amlt run sing_sglang_serve_multinode.yaml sing_sglang_router_480b_4node_v1 -d "4-node router" -y

# ---- watch ----
bash watch_sglang_router.sh sing_sglang_router_480b_4node_v1

# ---- get rollout URL ----
bash blob_sas.sh cat sglang_workers/sing_sglang_router_480b_4node_v1_router.json

# ---- smoke the router ----
URL=$(bash blob_sas.sh cat sglang_workers/sing_sglang_router_480b_4node_v1_router.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["url"])')
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Coder-480B-A35B-Instruct","messages":[{"role":"user","content":"PONG?"}],"max_tokens":8,"temperature":0}'

# ---- cancel ----
amlt cancel <exp_name> --yes
```
