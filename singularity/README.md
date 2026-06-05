# SWE-Master — MI300X serve + remote-Docker rollout (fork notes)

This fork adapts [RUCAIBox/SWE-Master](https://github.com/RUCAIBox/SWE-Master)
to run on **Singularity MI300X** with a specific topology used for rollout / RL:

> **One job serves the policy LLM (vLLM, local) AND drives the R2E-Gym agent,
> while the issue containers run on a REMOTE Docker host reached over a
> Cloudflare named TCP tunnel.**

Upstream code and license are unchanged except for the small, documented patches
below. All credit for the framework belongs to the original authors.

## What was changed vs upstream

`R2E-Gym/src/r2egym/agenthub/runtime/docker.py`
- `DOCKER_TLS_VERIFY=""` / `DOCKER_CERT_PATH=""` — the remote dockerd is reached
  over a plain-TCP Cloudflare tunnel, so forced TLS must be off.
- `make_test_spec` is imported locally inside the `swebench_verified` branch to
  avoid an `UnboundLocalError` from name shadowing.
- `setup_env_swebench` now writes `/run_tests.sh` from `test_spec.eval_script`
  (public `swebench/sweb.eval.*` images don't ship it), so reward calc works.

`R2E-Gym/src/r2egym/agenthub/agent/agent.py`
- `qwen3-coder` added to the native-function-calling allow-list (the served
  Qwen3-Coder model returns OpenAI `tool_calls` via vLLM's `qwen3_xml` parser).

`R2E-Gym/src/r2egym/agenthub/utils/utils.py`
- `HfFolder` import made defensive (it was unused and removed in newer
  `huggingface_hub`; guarding it lets the module import under the serving image's
  hub version).

## Image

`docker/Dockerfile.mi300x` builds on the proven ROCm/vLLM base and adds:
- system tools: `socat`, `iproute2` (ss), `jq`, Docker **CLI** client,
  `cloudflared`, `uv`;
- an isolated agent venv at `/opt/venvs/swe-agent` with the validated dependency
  set (`transformers==4.45.2`, `huggingface_hub==0.27.0`, `docker==7.1.0`,
  R2E-Gym + the three `swebench_fork_*` / `swesmith` wheels). It is deliberately
  separate from the base system Python, which keeps the serving stack
  (`transformers 5.x`, `huggingface_hub 1.x`) for vLLM.

Build & push:
```bash
bash docker/build_and_push.sh build     # build only
bash docker/build_and_push.sh push      # build + push to ACR
```

> SFT (OpenRLHF) and RL (verl/rLLM) venvs are **not** baked yet — their upstream
> pins are CUDA-only and need MI300X to validate. They will be layered in a
> follow-up; the Dockerfile leaves a clear seam for them.

## In-job entrypoint

`singularity/run_serve_and_rollout.sh`:
1. clones this repo fresh (so patches are never hand-applied),
2. serves the model with vLLM locally (`--tool-call-parser qwen3_xml`),
3. opens a `cloudflared access tcp` listener to the remote dockerd
   (`$DOCKER_TUNNEL_HOST -> 127.0.0.1:2375`),
4. runs `runagent_multiple` with the LLM local and Docker remote,
5. publishes the trajectory, reward, and status JSON to `$PUBDIR`.

The remote side (the dev box hosting dockerd) must be running the matching
Cloudflare **named** tunnel — see `SINGULARITY_TO_GCR_CLOUDFLARE_TUNNEL.md` in
the workspace for the one-time setup.
