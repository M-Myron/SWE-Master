# End-to-End Playbook: Sing(SGLang+Router) → Tunnel → GCR Agent Rollout → Trajectories

**Goal of this doc:** let a future session run the *entire* pipeline — submit a multi-node
SGLang serving job on Singularity MI300X, expose it to the GCR dev box via a Cloudflare tunnel,
then run the R2E-Gym agent on GCR to collect SFT/RL trajectories — **without re-hitting the
failures we already debugged**.

This is the orchestration playbook. For the deep *image* internals (the 5 non-root MI300X fixes,
how the serve script works, the landmine list) see the companion doc:
**[`sglang_multinode_serving_guide.md`](./sglang_multinode_serving_guide.md)** (referenced below as
**GUIDE**). This doc does NOT duplicate that; it links to the relevant section.

---

## 0. Mental model (what talks to what)

```
┌─ Singularity job (omai-aue-vc, MI300X) ───────────────────────┐
│  N nodes, each a FULL TP=8 replica of Qwen3.5-397B on :8000   │
│  rank0 also runs:  sglang_router :30000  →  cloudflared tunnel │
│  rank0 writes the public URL to blob: sglang_workers/<exp>_router.json
└───────────────────────────────────────────────────────────────┘
                         │  Cloudflare quick-tunnel (https://*.trycloudflare.com)
                         ▼
┌─ GCR dev box (this machine) ──────────────────────────────────┐
│  local dockerd (no sudo) runs the SWE issue container         │
│  socat exposes it at 127.0.0.1:2375 (R2E-Gym hardcodes :2375) │
│  R2E-Gym agent: OPENAI_API_BASE = <tunnel>/v1 , model = served name
│  → writes trajectory jsonl to dataset_smoke_runs/<exp>/       │
└───────────────────────────────────────────────────────────────┘
```

Two conda envs on GCR:
- **`amlt10`** — for `amlt` (submit/cancel/status). `conda activate amlt10`.
- **`swe-master`** — for the rollout (`/home/v-murongma/miniconda3/envs/swe-master/bin/python`).

Coordinates (target/workspace/UAI/blob): see **GUIDE §0**. Don't re-derive them.

---

## 1. Files (everything already exists — reuse, don't rewrite)

Serving (under `/home/v-murongma/code/sing_mi300_test/dind_test/`):
- `sing_sglang_serve_multinode.sh` — the serve+router+tunnel script **with supervisor/auto-restart**.
  Used by BOTH 1-node and multi-node (NODE_COUNT=1 → rank0 does everything).
- `sing_sglang_serve_1node_qwen35.yaml` — **1 replica** (free-GPU-friendly). `sku: 1x192G8-MI300X`.
- `sing_sglang_serve_multinode_qwen35.yaml` — **N replicas**. Edit `sku: Nx192G8-MI300X`.
- `blob_sas.sh` — blob reads via SAS (GCR mount is flaky). Subcommands: `list [prefix]`,
  `cat <blobpath>`, `get <prefix> <dst>`.
- `watch_sglang_router.sh <exp>` — one-shot status (job + per-rank + IPs + router URL).

Rollout (under `/home/v-murongma/code/sing_mi300_test/`):
- `r2egym_dataset_smoke.sh <swegym|swesmith|swerebench>` — full launcher: prep input → patch
  docker.py TLS → start socat 127.0.0.1:2375 → set URL → `runagent_multiple` → validate jsonl.
- `prep_dataset_smoke.py [all|<ds>]` — writes 1-instance inputs with **public** images to
  `dataset_smoke_inputs/<ds>_smoke.json` (verifies each image manifest first).

R2E-Gym source (patched): `/home/v-murongma/code/SWE-Master/R2E-Gym/` (env `swe-master`).

---

## 2. CRITICAL settings — get these wrong and it fails the way we already saw

These are non-negotiable for Qwen3.5-397B-A17B on MI300X. All are already baked into the yamls/scripts;
listed here so a future edit doesn't silently regress them.

| Setting | Value | Why (failure if wrong) |
|---|---|---|
| `--mem-fraction-static` | **`0.7`** | 0.8 → GPU HBM OOM `HSA_STATUS_ERROR_OUT_OF_RESOURCES … 0 MB` → SIGABRT mid-rollout. weights=92.8GB/GPU; 0.8 left only ~27GB runtime headroom, long reasoning exhausts it. |
| `--mamba-scheduler-strategy` | **`no_buffer`** | REQUIRED on AMD MI (hybrid Gated-DeltaNet). `extra_buffer`/V2 is NVIDIA-only. |
| `--moe-runner-backend` | **`triton`** | `auto`/aiter-CK MoE kernels are numerically wrong on some MI300X shapes → fluent-but-garbage. |
| `--attention-backend` | **`triton`** | required on AMD Instinct for this model. |
| `--reasoning-parser` | **`qwen3`** | separates `<think>` into `reasoning_content` (we keep it for training). |
| `--tool-call-parser` | **`qwen3_coder`** | matches the bundled chat template's `<function=…>` XML. **Do NOT** pass `--chat-template`. |
| image | `sglang:v0.5.11-rocm700-mi30x-patched-nonroot` | the non-root-fixed image (GUIDE §2). |
| `NCCL_DEBUG` | `WARN` | `INFO` floods tmpfs `/tmp/sglang.log` → host-RAM OOM `Killed` ~60min. |

R2E-Gym agent side (already patched in `R2E-Gym/src/r2egym/agenthub/agent/agent.py`):
- **`support_fn_calling` allow-list MUST include `qwen3.5`** (~L425). Otherwise the agent silently
  sets `tools=None`, the model never sees a tool schema, and EVERY step fails with "You forgot to
  use a function call" (it invents Hermes `{"name":"bash"}` calls). Tell-tale in run.log:
  `Using fn calling: False`. ← this was our #1 time-sink; see §5.
- Reasoning capture (~L346, ~L592, ~L602): `reasoning_content` is folded into the trajectory
  `thought` as `<think>…</think>` and the qwen parser-select is `.lower()`-cased.

---

## 3. RUNBOOK — serving (do this first)

```bash
source /home/v-murongma/miniconda3/etc/profile.d/conda.sh && conda activate amlt10
cd /home/v-murongma/code/sing_mi300_test/dind_test

# az token can go stale → 'jobs submit permission' error. Refresh if needed:
#   az login --use-device-code

# --- 1 replica (free-GPU-friendly; start here) ---
amlt run sing_sglang_serve_1node_qwen35.yaml sing_sglang_router_qwen35_397b_1node_v1 \
  -d "1-node Qwen3.5-397B serving (TP=8) + supervisor + router + tunnel" -y

# --- OR N replicas (throughput; edit sku: Nx192G8-MI300X inside the yaml first) ---
amlt run sing_sglang_serve_multinode_qwen35.yaml sing_sglang_router_qwen35_397b_3node_v1 \
  -d "3-node Qwen3.5-397B router (3 replicas) + 1 tunnel" -y
```

Wait for it to load (cold start ~14 min: weights are ~92.8GB/GPU). Watch via blob, NOT the amlt
portal (portal log 503s — GUIDE §5):

```bash
bash watch_sglang_router.sh sing_sglang_router_qwen35_397b_1node_v1
```

Ready when `sglang_workers/<exp>_router.json` exists and PONG works:

```bash
PY=/home/v-murongma/miniconda3/envs/swe-master/bin/python
URL=$(bash blob_sas.sh cat sglang_workers/sing_sglang_router_qwen35_397b_1node_v1_router.json \
      | $PY -c 'import sys,json;print(json.load(sys.stdin)["url"])')
echo "URL=$URL"
curl -s --max-time 25 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-397B-A17B","messages":[{"role":"user","content":"say PONG"}],
       "max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

**The tunnel URL changes on every (re)start** (trycloudflare hands out a new hostname). Always
re-read the router json before a rollout — never hard-code the URL. The supervisor auto-restarts
sglang/router/tunnel and republishes the new URL, so transient crashes self-heal (GUIDE §6 #11).

---

## 4. RUNBOOK — GCR rollout / trajectory collection

The launcher does prep + socat + run + validate. One instance per dataset:

```bash
cd /home/v-murongma/code/sing_mi300_test
PY=/home/v-murongma/miniconda3/envs/swe-master/bin/python
URL=$(bash dind_test/blob_sas.sh cat sglang_workers/sing_sglang_router_qwen35_397b_1node_v1_router.json \
      | $PY -c 'import sys,json;print(json.load(sys.stdin)["url"])')

# datasets: swerebench | swegym | swesmith   (fn-calling ON = the proven path)
URL="$URL" USE_FN_CALLING=True bash r2egym_dataset_smoke.sh swerebench
```

Env knobs: `URL` (else read from blob), `MODEL` (=`Qwen/Qwen3.5-397B-A17B`), `K` (#instances, def 1),
`MAX_STEPS` (def 100), `TEMP` (def 0.6), `USE_FN_CALLING` (def True).

**Run detached + don't interfere** (a stray `pkill ...<ds>` from a parallel command can kill the run
mid-reward → 0-byte jsonl):
```bash
nohup env URL="$URL" USE_FN_CALLING=True bash r2egym_dataset_smoke.sh swerebench \
      > /tmp/swerebench.out 2>&1 &
```

### Validate a trajectory (what "good" looks like)
```bash
JSONL=$(ls -t dataset_smoke_runs/swerebench_*/*.jsonl | head -1)
$PY - "$JSONL" <<'P'
import json,sys,re; from collections import Counter
r=json.loads(open(sys.argv[1]).readline()); ts=r.get("trajectory_steps") or []
real=[s for s in ts if '<function=' in (s.get('action') or '') and '<function=>' not in (s.get('action') or '')]
think=sum('<think>' in (s.get('thought') or '') for s in ts)
print("exit",r.get("exit_reason"),"reward",r.get("reward"),"steps",len(ts),
      "real",f"{len(real)}/{len(ts)}","think",f"{think}/{len(ts)}",
      "patch",len(r.get("output_patch") or ""))
print(Counter(re.findall(r'<function=(\w+)',' '.join(s.get('action') or '' for s in ts))))
P
```
Healthy = `Using fn calling: True` in run.log, real==steps (0 empty `<function=>`), think==steps,
patch non-empty, reward computed. (reward 0.0 is a legit failed *attempt* — still valid training data.)

### Public images per dataset (private harbor is NOT reachable from GCR)
`prep_dataset_smoke.py` already encodes these; reference if building larger inputs:
- **swegym**: `xingyaoww/sweb.eval.x86_64.<iid with `__`→`_s_`>` (docker.io public). SWE-Gym-Lite has
  NO image field; we add `docker_image`; test_spec is built pure-from-ds (no GitHub fetch). Source:
  HF `SWE-Gym/SWE-Gym-Lite` (~230 instances).
- **swesmith**: HF `SWE-bench/SWE-smith` row has `image_name = jyangballin/swesmith.x86_64.*` (~59k).
- **swerebench**: source HF **`nebius/SWE-rebench` split=`filtered`** (6,542 instances, **100% public**
  `swerebench/sweb.eval.x86_64.<X>` docker.io images). Each row → build `make_test_spec` with
  `swebench_fork_swerebench.harness.test_spec.test_spec.make_test_spec(row)` then `json.dumps(
  dataclasses.asdict(ts))` (docker.py does `json.loads(self.ds['make_test_spec'])`). **No private
  harbor and no local build needed** — the `harbor.weizhipin.com/arsenal-ai/swerebench/<X>` images in
  the bundled demo are just an internal mirror of the same public `swerebench/<X>` images. (SWE-Master
  itself sources swerebench from `nebius/SWE-rebench` — see `data_preparation/download_swe_datasets.sh`.)
  The OTHER 14,794 nebius rows (the non-`filtered` remainder) have `docker_image=null` and would need
  a local build via the fork's `build_instance_images()` — not needed, 6,542 is plenty.

---

## 5. The failures we already hit (DO NOT rediscover these)

All three below produced *similar-looking* breakage. Diagnose in this order.

1. **fn-calling silently disabled (deterministic, #1 time-sink).**
   Symptom: every step `You forgot to use a function call`; model emits `<tool_call>{"name":"bash"…}`
   with invented tool names. Cause: `Qwen/Qwen3.5-…` didn't match the `support_fn_calling` allow-list
   → `tools=None` sent → no tool schema rendered. **Check `Using fn calling:` in run.log.**
   Fix: `qwen3.5` is now in the allow-list (agent.py ~L425). *Trap that misled us:* a raw curl that
   passes `tools=[…]` bypasses the gate and looks fine — so isolated emission tests pass while the
   agent fails. Trust the run.log, not an ad-hoc curl.

2. **Server GPU-OOM crash mid-rollout.** Symptom: rollout suddenly gets 503 /
   `no_available_workers`; trajectories full of garbage for the steps around it. Cause:
   `HSA_STATUS_ERROR_OUT_OF_RESOURCES … 0 MB` during a long reasoning decode → replica SIGABRT.
   Fix: `--mem-fraction-static 0.7`. The supervisor auto-restarts (~3 min), but in-flight rollouts
   fail — so fix the headroom, don't rely on restart.

3. **Reasoning dropped from trajectories.** Symptom: tool calls fine but `thought` empty. Cause:
   `--reasoning-parser qwen3` puts CoT in `reasoning_content`, not `content`; agent read only
   `content`. Fix: agent folds `reasoning_content` into `<think>…</think>` (agent.py).

Plus the serving-stack lessons (full detail in **GUIDE §6**):
- One sglang crash used to tear down router+tunnel (rank0 `wait $SGL_PID` + EXIT trap). Now a
  **supervisor** restarts each component independently and republishes the new tunnel URL.
- Host-RAM OOM `Killed` ~60min from `NCCL_DEBUG=INFO` filling tmpfs log → `WARN` + 1GB log cap.
- All 5 non-root image landmines (aiter lock, HOME RO, nvcc, torch.compile, aiter JIT dir) — GUIDE §2.
- The "official" NVIDIA serve cmd (EAGLE + `--enable-flashinfer-allreduce-fusion`) is **not** for
  MI300X: FlashInfer is CUDA-only; EAGLE adds memory (worsens OOM). Our flags (§2) are the AMD set.

---

## 6. Scaling to large-scale collection

- **Throughput**: bump `sku: Nx192G8-MI300X` in the multinode yaml → N replicas behind ONE router/URL
  (~N× throughput). Validated: 3 replicas → 18 concurrent → 18/18 HTTP200 in 1.45s wall (GUIDE §7b).
- **Driver**: `runagent_multiple` already takes `--k`/`--start_idx`. For many instances, build a
  dataset JSON (list of instances with public `docker_image` + `ip:127.0.0.1`, like
  `prep_dataset_smoke.py` does) and raise `K`. Prepull images first (`docker pull`) to de-risk.
- Keep `MEM_FRAC=0.7`. Concurrency on a single replica is fine for light loads; for sustained
  parallel rollouts use more replicas so no single replica hits the OOM ceiling.
- Trajectories land in `dataset_smoke_runs/<exp>/<exp>.jsonl` (one record per instance). The schema
  (top-level + per-step keys) is documented in `/memories/repo/r2egym_rollout_collection.md`.

---

## 7. One-screen quick reference

```bash
# ---------- SERVE ----------
conda activate amlt10; cd /home/v-murongma/code/sing_mi300_test/dind_test
amlt run sing_sglang_serve_1node_qwen35.yaml <exp> -d "..." -y        # 1 replica
#   (edit sku: Nx192G8-MI300X in *_multinode_qwen35.yaml for N)
bash watch_sglang_router.sh <exp>                                     # status
PY=/home/v-murongma/miniconda3/envs/swe-master/bin/python
URL=$(bash blob_sas.sh cat sglang_workers/<exp>_router.json | $PY -c 'import sys,json;print(json.load(sys.stdin)["url"])')
curl -s "$URL/v1/models"                                             # ready?

# ---------- ROLLOUT (GCR) ----------
cd /home/v-murongma/code/sing_mi300_test
nohup env URL="$URL" USE_FN_CALLING=True bash r2egym_dataset_smoke.sh swerebench > /tmp/r.out 2>&1 &
grep -m1 "Using fn calling" dataset_smoke_runs/swerebench_*/run.log  # MUST be True

# ---------- CANCEL ----------
conda activate amlt10; amlt cancel <exp> -y
```

**If anything looks like "the model can't call tools": first `grep "Using fn calling" run.log`.**
**If the rollout dies with 503: check the sglang log for `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.**
