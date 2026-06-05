# SWE-Master on Singularity + GCR-as-Docker — Status & Plan

**Last updated:** 2026-05-28
**Author:** v-murongma@microsoft.com
**Status:** Infra path proven end-to-end. Ready for first R2E-Gym agent rollout (P1a smoke). SFT/RL pipelines unblocked pending data + LLM source.

This document is the single source of truth for *how* to run SWE-Master's SFT and RL on our cluster (Singularity MI300X + GCR dev box as Docker server). It also lists what is still TODO. For the longer infrastructure investigation that led to this plan, see `DIND_INVESTIGATION_REPORT.md` (in `~/blob/murong/code/sing_mi300_test/`).

---

## 1. The plan in one paragraph

We run the **full upstream SWE-Master hybrid-engine RL** (vLLM + FSDP colocated, Ray-internal weight sync → near-on-policy) inside **one Singularity MI300X job**. The `env.step` calls from the in-pod rollout actors go *out* through a **Cloudflare named tunnel** to the **GCR dev box**, which runs the real `dockerd` and hosts every per-issue container. SFT data synthesis uses the same agent loop (different LLM, different scheduler — see §5). No code changes to R2E-Gym (one tiny patch), no DinD on Singularity, no AML Serverless required.

```
+-------------------------------------+         Cloudflare named tunnel        +-------------------------------+
| Singularity MI300X job              |       (outbound HTTPS / QUIC)          | GCR dev box                   |
|   Ray cluster:                      |                                        |  socat :2376 -> docker.sock   |
|     - FSDP actor workers            |  cloudflared access tcp                |  cloudflared tunnel run       |
|     - vLLM rollout workers          |  --hostname docker.swerl-...uk         |    swerl-docker               |
|     - AsyncAgentExecutionEngine     |  --url tcp://127.0.0.1:2375            |  dockerd hosts swebench       |
|       per step:                     | <-------------------------------------->   per-issue containers       |
|         LLM (local vLLM)            |                                        |  pulls public images from     |
|         env.step -> docker.from_env |                                        |    docker.io/swebench/*       |
|           (tcp://127.0.0.1:2375)    |                                        |                               |
+-------------------------------------+                                        +-------------------------------+
```

Trainer + vLLM colocate on the same Ray cluster ⇒ weight sync is GPU→GPU NCCL/HBM (verl's hybrid engine), so RL is essentially on-policy. The only off-axis hop is `env.step`, which is on a worker thread already (~250 ms over the tunnel — acceptable since vLLM batching hides it across rollouts).

---

## 2. What is proven (with citations to working artifacts)

| # | Capability | Artifact (in `~/blob/murong/code/`) | Date |
|---|---|---|---|
| 1 | Singularity pods CANNOT run Docker (uid 9000, `CapEff=0`, no socket) | `singularity_scripts/config/dind_test/probe_docker.sh` + `dind_test.yaml` | 2026-05-26 |
| 2 | AML Serverless `docker-tools:34` env IS privileged (root, all caps) — can run dockerd | `singularity_scripts/config/dind_test/azureml_docker_probe.yaml` | 2026-05-27 |
| 3 | Singularity → Cloudflare quick tunnel works (outbound HTTPS allowed) | `singularity_scripts/config/dind_test/sing_inbound_probe.sh` | 2026-05-27 |
| 4 | vLLM on MI300X served via cloudflared quick tunnel, dev-box agent calls it | `singularity_scripts/config/dind_test/sing_vllm_tunnel.{yaml,sh}` + `rollout_dev.py` | 2026-05-27 |
| 5 | **Cloudflare named tunnel** on GCR dev box exposes local dockerd | `singularity_scripts/config/cloudflared/{config.yml,start_tunnel.sh,README.md}` | 2026-05-28 |
| 6 | **Singularity pod → cloudflared access tcp → GCR dockerd → `exec_run`** (the call R2E-Gym makes every step) | `singularity_scripts/config/dind_test/sing_remote_docker_probe.{yaml,sh}` | 2026-05-28 |

Combined, (5) + (6) settle the "can we do this at all?" question. **Yes.** ~250 ms per `exec_run`.

---

## 3. Production tunnel setup (current state)

### 3.1 Domain & tunnel

- Domain: **`swerl-docker-connection.uk`** (registered through Cloudflare, ~$10/yr).
- Cloudflare account: `Murongma@gmail.com`.
- Named tunnel: **`swerl-docker`** (UUID `affe8c0e-5830-48fd-a88f-895ba2c56b25`).
- DNS: `docker.swerl-docker-connection.uk` CNAME → tunnel.
- Tunnel ingress: `tcp://localhost:2376` (socat → unix dockerd socket).

### 3.2 Persisted artifacts (Blob — survive GCR machine death)

Path: `/home/v-murongma/blob/murong/code/singularity_scripts/config/cloudflared/`
- `cloudflared-linux-amd64` — the binary (~39 MB)
- `cert.pem` — Cloudflare account origin cert
- `tunnel-swerl-docker.json` — tunnel credentials
- `config.yml` — ingress mapping
- `service-token.env` — Access service token (currently unused; see §6.1)
- `start_tunnel.sh` — foreground starter (run in tmux on the GCR box)
- `README.md` — restore instructions

### 3.3 Start procedure on GCR box

```bash
# (one-time per GCR machine) restore artifacts
cp /home/v-murongma/blob/murong/code/singularity_scripts/config/cloudflared/cloudflared-linux-amd64 \
   ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
mkdir -p ~/.cloudflared
cp /home/v-murongma/blob/murong/code/singularity_scripts/config/cloudflared/{cert.pem,tunnel-swerl-docker.json,config.yml} \
   ~/.cloudflared/

# every session (or in tmux)
bash /home/v-murongma/blob/murong/code/singularity_scripts/config/cloudflared/start_tunnel.sh
```

When you see `tunnel live (HTTP 200)`, the path is up. Leave it running.

### 3.4 Client side (Singularity job)

```bash
# install cloudflared
curl -fSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /tmp/cloudflared

# open local TCP listener
nohup /tmp/cloudflared access tcp \
  --hostname docker.swerl-docker-connection.uk \
  --url     tcp://127.0.0.1:2375 \
  > /tmp/cf-access.log 2>&1 &

# now anything that speaks Docker REST API can hit tcp://127.0.0.1:2375
DOCKER_HOST=tcp://127.0.0.1:2375 docker version
```

(Cloudflare Access lockdown is currently disabled — the hostname is non-discoverable but unprotected. See §6.4.)

---

## 4. Code changes required in SWE-Master / R2E-Gym

### 4.1 Mandatory

**Patch 1 — disable forced TLS in DockerRuntime.** `R2E-Gym/src/r2egym/agenthub/runtime/docker.py:103` hardcodes `DOCKER_TLS_VERIFY = "1"`. Our cloudflared listener speaks plain TCP — the Docker SDK refuses without certs. Patch at job startup:

```bash
sed -i 's|^DOCKER_TLS_VERIFY = "1"|DOCKER_TLS_VERIFY = ""|' \
       src/r2egym/agenthub/runtime/docker.py
```

### 4.2 No patch needed — confirmed by reading the code

- **Remote Docker IP:** `--ip <X>` propagates `runagent_multiple → runagent → RepoEnv → DockerRuntime → tcp://<X>:2375`. Pass `--ip 127.0.0.1`.
- **LLM endpoint:** `agent.py` reads `OPENAI_API_BASE` + `OPENAI_API_KEY` when `--llm_name` contains `openai/` or `hosted_vllm/`. Just set env vars.
- **Hybrid engine:** `DeepSWE_RL/rllm/rllm/trainer/verl/agent_ppo_trainer.py` with `hybrid_engine=True` keeps Actor + Rollout in the same Ray WorkerGroup; weight sync is Ray-internal — orthogonal to where Docker runs.
- **env.step concurrency:** `agent_execution_engine.py:340-345` already runs `env.step` in a `ThreadPoolExecutor`, so per-step Docker latency (~250 ms via tunnel) doesn't stall vLLM batching.

---

## 5. Workflow per phase

### 5.1 SFT data synthesis (Stage 1a)

- Same agent code (`R2E-Gym/src/r2egym/agenthub/run/edit.py runagent_multiple`).
- Teacher LLM: **TODO §6.1** (TRAPI ideal but currently blocked; interim use self-hosted Qwen on GCR or Singularity).
- Where rollouts run: Singularity job (single MI300X, no training; rollout actors only) **OR** plain Python on the GCR box itself (since GCR has the dockerd locally).
- Per-instance writes a trajectory JSON to Blob.

### 5.2 SFT training (Stage 2)

- No Docker, no LLM-serving. Singularity MI300X, multi-node FSDP via `OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_32B_*.sh`.
- Reads filtered trajectory parquet from Blob; writes SFT checkpoint to Blob.

### 5.3 RL (Stage 3) — the on-policy hybrid engine

- **Single Singularity job** running upstream `agent_ppo_trainer.py` (NOT the `_pipeline` variant) with `hybrid_engine=True`.
- Inside the job: FSDP trainer + vLLM rollout workers share the same Ray cluster + same MI300X GPUs.
- Rollout actors call `env.step` → goes out via `cloudflared access tcp` to `docker.swerl-docker-connection.uk` → GCR dockerd.
- Weight sync after each PPO update happens Ray-internally — no Blob round-trip, no minute-scale staleness.

Resource budget (rough):
- SFT data synth: GCR box CPU + dockerd, or 1–4 Singularity jobs as rollout farm.
- SFT training: 2–4 nodes × 8 MI300X × 1–2 days.
- RL: 4–6 nodes × 8 MI300X × 1+ week.
- GCR docker pool: 1 box for now (caps concurrent rollouts at ~32–64). Add more later via the same tunnel pattern, picking from a pool registry in Blob.

---

## 6. What is NOT solved yet (TODO)

### 6.1 TRAPI access is not available

**Problem:** TRAPI is the natural teacher for SFT data (per `singularity_scripts/config/demo/hello_api.py`), but it is **not OpenAI-API-compatible** — it requires an Azure-AD bearer token (`api://trapi/.default` scope, `AzureCliCredential`/`ManagedIdentityCredential`), endpoint `https://trapi.research.microsoft.com/redmond/interactive`, deployment `gpt-5.2-chat_2025-12-11`, API version `2025-04-01-preview`. R2E-Gym's `agent.py` uses `litellm` with a static `OPENAI_API_KEY`.

**What we need:**
1. Stable TRAPI access from this account (RBAC / approval).
2. Either of two integration paths:
   - **a)** litellm azure-provider: set `--llm_name azure/gpt-5.2-chat_2025-12-11`, env vars `AZURE_API_BASE`, `AZURE_API_VERSION`, `AZURE_AD_TOKEN`. Token expires every 1 h → need a refresher (cron in pod or sidecar).
   - **b)** Tiny proxy on GCR/Azure App Service that accepts `Bearer <static-key>`, refreshes the Azure AD token internally, and forwards to TRAPI. Then R2E-Gym uses standard `openai/` mode pointing at the proxy.
3. Decide which: **(b)** is more robust for long RL runs but +1 day of setup.

Workaround until TRAPI is available: see §6.2.

### 6.2 Self-hosted teacher / serving stack (open-source models)

We need to demonstrate SFT-quality data synthesis without TRAPI. Two-stage validation:

#### 6.2.1 Local serving on GCR dev box (small model)
- Bring up vLLM locally on GCR with `Qwen3-Coder-32B-A3B-Instruct` (or similar small MoE).
- Verify it serves OpenAI-compat on `127.0.0.1:8000/v1`.
- Run R2E-Gym (`runagent_multiple`) **on GCR itself** against ONE public SWE-Bench-Verified instance using:
  - `--ip 127.0.0.1` (local dockerd, no tunnel)
  - `OPENAI_API_BASE=http://127.0.0.1:8000/v1`
  - `--llm_name openai/Qwen/Qwen3-Coder-32B-A3B-Instruct`
- Goal: confirm R2E-Gym + the model produce a sensible trajectory format end-to-end on a single machine. No tunneling involved yet.

#### 6.2.2 Singularity inference with GCR-as-Docker (large model)
- Serve a larger model (e.g. `Qwen3-Coder-480B-A35B-Instruct`) inside a Singularity job (4–8 × MI300X with TP/PP).
- Same Singularity job runs R2E-Gym agent loop, with `env.step` going via tunnel → GCR dockerd.
- Goal: prove that the agent quality is high enough on a real SWE-Bench-Verified instance to produce a "good" SFT trajectory. Then scale up.

### 6.3 RL pilot — small first, then scale up

- **Small RL pilot, local on GCR:** start from an SFT'd small model (or even base Qwen3-Coder-7B), run a few hybrid-engine GRPO steps locally with a tiny rollout batch. Validate that the verl `agent_ppo_trainer` works with our remote-docker pattern and produces a checkpoint.
- **Scale to Singularity:** launch the same script in a multi-MI300X Singularity job with the bigger SFT'd model. Verify hybrid-engine weight sync works in the multi-GPU setting and that GCR dockerd holds up under N concurrent rollouts (likely cap ~32 per GCR box; bring up a second GCR box as a Docker pool member when needed).

### 6.4 Optional but recommended (lower priority)

- **Cloudflare Access lockdown** (currently disabled): the hostname is not on any DNS index but the dockerd is unprotected. To fix, recreate the Access app with policy `Action: Service Auth, Include: Service Token = swerl-docker-client`. Service token already created and saved to `cloudflared/service-token.env`. Two earlier attempts hit dashboard UX issues — deferred.
- **GCR-side persistence** via cron `@reboot`: tunnel currently runs in `tmux` foreground (`start_tunnel.sh`) and dies on machine reboot.
- **Multi-GCR docker pool**: extend the same tunnel pattern to more GCR boxes (each with its own subdomain like `docker2.swerl-docker-connection.uk`). Add a small "pool registry" in Blob that lists healthy endpoints; rollout actors lease one per trajectory.

---

## 7. Phased rollout (concrete next steps)

| Phase | What | Effort | Output |
|---|---|---|---|
| **P0 (done)** | Cloudflare named tunnel from GCR; Singularity → tunnel → dockerd `exec_run` proven | done | this report |
| **P1a (next)** | Clone SWE-Master from public github, patch `docker.py`, run `runagent_multiple` on 1 SWE-Bench-Verified instance using the existing Qwen2.5-0.5B Singularity vLLM. Goal: validate the wiring end-to-end. Agent quality unimportant; just need a trajectory JSON. | 0.5 d | 1 trajectory file; failures in the agent loop OK |
| **P1b** | Bring up Qwen3-Coder-32B locally on GCR; run R2E-Gym locally (no tunneling). | 1 d | 1 high-quality trajectory file (proves model quality + format) |
| **P1c** | Bring up Qwen3-Coder-480B in Singularity; agent loop in Singularity job using GCR tunnel for Docker. | 1–2 d | 1 high-quality trajectory file via the production path |
| **P2** | Scale P1c rollout: parallel agents, 10–50 SWE-Bench-Verified instances. | 1 d | first real SFT data shard |
| **P3** | SFT-train Qwen3-Coder-32B on Singularity (existing `OpenRLHF_SFT` scripts). | 2–3 d walltime | first SFT checkpoint |
| **P4** | RL pilot: small model on GCR locally with verl hybrid engine + remote docker. | 2 d | 1 GRPO step works |
| **P5** | RL pilot scaled to Singularity multi-node. | 3–5 d | first multi-node GRPO step |
| **P6** | TRAPI integration (proxy or token refresher). | 1 d (after access granted) | TRAPI-based teacher available |
| **P7+** | Production hardening: Access lockdown, multi-GCR pool, restart-on-eviction. | ongoing | stable long RL runs |

---

## 8. Key files & locations

### On the GCR dev box
- `~/.cloudflared/` — tunnel cert + creds + config (mirrored from Blob).
- `~/.local/bin/cloudflared` — binary.
- `/var/run/docker.sock` — the actual dockerd serving for the world (via the tunnel).
- `tmux session running start_tunnel.sh` — the always-on plumbing.

### On Blob (`shuailu1/murong` mounted at `/home/v-murongma/blob/murong/`)
- `blob/murong/code/SWE-Master/` — convenience copy of this report (dev-box visible).
- `blob/murong/code/sing_mi300_test/DIND_INVESTIGATION_REPORT.md` — long investigation doc.
- `blob/murong/code/singularity_scripts/config/cloudflared/` — tunnel artifacts (see §3.2).
- `blob/murong/code/singularity_scripts/config/dind_test/` — all probes.

### On Blob cache (`blob/cache/...` — different mount, also dev-box visible)
- `blob/cache/murong/code/SWE-Master/SWE_MASTER_SING_GCR_REPORT.md` — primary copy of this report.
- `blob/cache/murong/code/SWE-Master/copilot_memory/session_notes.md` — short summary for future Copilot sessions.

### On Blob (`zhibinmain/murongma` mounted at `/mnt/murongma` in Singularity jobs)
- Used as the production data path for SFT trajectories, RL checkpoints, RL trajectories. Standardized layout in `DIND_INVESTIGATION_REPORT.md` §9.1.

---

## 9. Glossary / open IDs

- Singularity workspace: `msraairwsws` (RG `system_yeyun`, sub `762905fc-…`).
- Singularity primary target: `omai-aue-vc` (AMD MI300X, australiaeast).
- amlt project: `sing_mi300_test` (storage `zhibinmain/amulet`).
- Cloudflare tunnel name: `swerl-docker` (UUID `affe8c0e-5830-48fd-a88f-895ba2c56b25`).
- Cloudflare service token: `swerl-docker-client` (Client ID `a687c0cffa599774eb0770a6ba3d07d9.access`).
- Public hostname: `docker.swerl-docker-connection.uk`.
- Long-lived self-hosted teacher (tiny): Qwen2.5-0.5B-Instruct, currently still served at `https://heart-autos-graduate-velocity.trycloudflare.com` (ephemeral quick tunnel; ID may change).
