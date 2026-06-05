# SWE-Master — Working Notes

Persistent context for resuming work on `/home/v-murongma/code/SWE-Master`. Read this file first.

## Project at a glance

Open-source post-training pipeline for SWE agents (paper: *SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training*).
End-to-end: **trajectory synthesis → SFT → RL with verifiable rewards → test-time scaling**.

Base model: `Qwen2.5-Coder-32B-Instruct` (also a 4B variant). Released SFT/RL checkpoints on HF under `RUC-AIBOX/SWE-Master-*`.

## Top-level layout

| Dir | Role |
|---|---|
| `R2E-Gym/` | Inference / rollout framework (forked R2E-Gym). Contains the Agent, Docker runtime, tools, scaffolds, eval scripts. |
| `OpenRLHF_SFT/` | SFT pipeline (built on OpenRLHF). Data conversion + pretokenize + DeepSpeed training. |
| `DeepSWE_RL/rllm/` | RL pipeline (rLLM + VeRL v0.5). GRPO/PPO with Docker-based reward. |
| `data_preparation/` | Dataset download, difficulty grading, offline unit-test cache generation. |
| `data_examples/` | Reference data formats (inference traj JSONL, SFT, RL parquet). |
| `swebench_fork_*.whl`, `swesmith-*.whl` | Vendored harnesses for SWE-Gym / SWE-rebench / SWE-smith / SWE-bench. |
| `SFT_ENV.txt`, `RL_ENV.txt` | Extra dependency lists for training envs. |
| `ray_start_{head,work}.sh` | Multi-node Ray bootstrap. |

## Environments (3 separate uv venvs)

User currently has one conda env `swe-master` containing the inference deps.

1. **Inference env (`swe-master`, already installed):**
   ```bash
   cd /home/v-murongma/code/SWE-Master
   uv venv && source .venv/bin/activate
   uv sync && uv pip install -e .
   uv pip install ./swebench_fork_swegym-2.0.13-py3-none-any.whl
   uv pip install ./swebench_fork_swerebench-4.0.3-py3-none-any.whl
   uv pip install ./swesmith-0.0.7-py3-none-any.whl
   ```
2. **SFT env (separate):** `cd OpenRLHF_SFT && uv venv --python 3.11 && pip install -e OpenRLHF` + extras from `SFT_ENV.txt`.
3. **RL env (separate):** `cd DeepSWE_RL/rllm && uv venv --python 3.11 && uv pip install -e . && uv pip install -e ./verl` (VeRL v0.5) + same swebench wheels + extras from `RL_ENV.txt`.

> ⚠ Don't reuse a single env across stages; version pins conflict.

## Stage 1 — Rollout / Inference

**Architecture:** LLM (any OpenAI-compatible endpoint) ↔ host-side Agent loop ↔ Docker container per issue.

- Entry: `R2E-Gym/src/r2egym/agenthub/run/edit.py runagent_multiple`, wrapped by `R2E-Gym/rollout_foundation_model/run_deepseek_v3.sh`.
- LLM client: `litellm`/`openai`; uses `OPENAI_API_BASE`, `OPENAI_API_KEY`.
- Docker: one container per SWE issue (image identified per dataset entry: `docker_image`/`image_name`). `DOCKER_HOST=tcp://<ip>:2375`.
- Scaffolds: `openhands` (default in scripts), `r2egym`, `sweagent` — chosen via `--scaffold` + `--used_yaml`.
- Tools live host-side in `R2E-Gym/src/r2egym/agenthub/tools/` (`str_replace_editor`, `execute_bash`, `submit`, `search`, `finish`, optional `lsp_tool`). On `RepoEnv.__init__`, host copies them via Docker archive API into `/usr/local/bin/<name>` and chmod+x.
- Per step: `Action.to_bashcmd()` → `docker exec bash -c <cmd>` with `cwd=/testbed`, captures stdout+exit code → becomes next Observation.
- Reward: at `submit/finish`, runs the test spec (built by `make_test_spec(...)`) inside the container, parses with swebench parser → 0/1.
- LSP rollouts need `pyright`+`node`+`lsp_daemon` baked into the image; daemon launched via `nohup lsp_daemon &`, port written to `/var/tmp/lsp_port_session_abc.pid`.

**Key files:**
- `R2E-Gym/src/r2egym/agenthub/agent/agent.py` — `Agent`, `AgentArgs`, LLM call + history + summary memory.
- `R2E-Gym/src/r2egym/agenthub/environment/env.py` — `RepoEnv` (gym.Env wrapper).
- `R2E-Gym/src/r2egym/agenthub/runtime/docker.py` — `DockerRuntime` (container lifecycle, exec, copy). **Update `base_dir` here if changing test-spec cache path.**
- `R2E-Gym/rollout_foundation_model/readme.md` — full parameter reference.

**Run modes:**
```bash
bash ./R2E-Gym/rollout_foundation_model/run_deepseek_v3.sh                       # standard
bash ./R2E-Gym/rollout_foundation_model/lsp/run_deepseek_v3_lsp.sh               # with LSP tool
bash ./R2E-Gym/rollout_foundation_model/memory/run_deepseek_v3_memory.sh         # with context manager
bash ./R2E-Gym/rollout_trained_model/vllm_server_launch.sh                       # eval own checkpoint
bash ./R2E-Gym/rollout_trained_model/qwen25_coder_15k_lsp_with_gerenal_04.sh
```
Visualize trajectories: `python R2E-Gym/app/json2html_for_view-en.py <traj.jsonl> -o out.html`.

## Stage 2 — SFT (no Docker)

Consumes JSONL trajectories from rollout step (or HF). Pure next-token CE on assistant turns with loss masking.

1. Convert R2E traj → OpenRLHF multi-turn chat:
   `OpenRLHF_SFT/SFT_data_pre_process/r2e_to_openrlhf_format/0_covert_r2e_format_to_sft_foramt.py`
2. Format filter: `1_init_format_filter.py`. Optional BoN/difficulty filtering: `SFT_data_pre_process/bon_filter/`.
3. Pre-tokenize with loss masks: `OpenRLHF_SFT/scripts_swe_master/sft_data_pre_tokenize.py` (or `sft_data_pre_tokenize_toolcall.py` for native tool_calls). Then `mv scripts_swe_master/sft_dataset.py datasets/`.
4. Train: `bash OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_32B_new_remove_01_not_dedep.sh`
   - `deepspeed --module openrlhf.cli.train_sft`
   - `--max_len 81920`, `--packing_samples`, `--ring_attn_size 16`, ZeRO-3, bf16, `--multiturn`, `lr=5e-5`.

Output: HF checkpoint → becomes RL policy init.

## Stage 3 — RL (Docker + GPUs, hybrid engine)

Script: `DeepSWE_RL/rllm/examples/swe/swe_rl_debug_hls_try_arsenal_long_step_and_long_time.sh`
Entry: `python -m rllm.trainer.verl.train_agent_ppo` (or `train_agent_ppo_pipeline` for async-overlap mode).

Algorithm: GRPO (`algorithm.adv_estimator=loop`), no KL, `clip_ratio_high=0.28`, `loss_agg_mode=seq-mean-token-sum`, verifiable 0/1 reward from unit tests.

### Hybrid engine — training + rollout share the same GPUs

- `actor_rollout_ref.hybrid_engine=True`
- Rollout: vLLM async server, TP=4, `gpu_memory_utilization=0.6`, `chat_scheduler=CompletionsScheduler`.
- Training: FSDP (PyTorch Fully Sharded Data Parallel, ≈ ZeRO-3). Sharded params + grads + optim across all DP ranks. Plus Ulysses SP=8 for sequence parallelism on 100K-token trajectories. Param + optimizer offload to CPU.
- Weight sync after each step via `rollout_wg.update_rollout_actor_module(state_dict)` — on-device broadcast, no checkpoint roundtrip.

### Parallelism axes (vs Megatron)
| Megatron | Here | Where |
|---|---|---|
| DP | FSDP/ZeRO-3 | implicit, fills remaining ranks |
| TP | vLLM rollout TP=4 (training uses FSDP only) | `rollout.tensor_model_parallel_size=4` |
| CP/SP | Ulysses SP | `actor.ulysses_sequence_parallel_size=8` |
| PP | not enabled (VeRL FSDP backend) | — |

64 GPUs (8 nodes × 8 H800) in this script.

### How slow rollouts are hidden
1. **Massive async concurrency**: `AsyncAgentExecutionEngine` runs all `batch * n` trajectories as asyncio tasks with `asyncio.Semaphore`; Docker exec offloaded via `loop.run_in_executor`; vLLM serves continuous batched completions.
2. **Bounded tails**: `step_timeout=90`, `agent.trajectory_timeout=5400`, outer `wait_for(..., 7200)`. `agent.overlong_filter=True` to drop truncated.
3. **Pipelined trainer** (`agent_ppo_trainer_pipeline.py`): background thread streams completed groups (grouped by uid for GRPO) into a `replay_queue`; main loop trains mini-batches as groups arrive; weights synced to rollout only after the last mini-batch of the step. So vLLM is generating step *k+1* while optimizer updates step *k*. Tolerated off-policy via clip.
4. **Filter trivial/impossible groups**: groups with all-zero or all-one rewards masked from gradient (zero GRPO advantage anyway). Combined with offline difficulty filter.
5. Per-step: `train_batch_size=32`, `rollout.n=4` → 128 trajectories/step.

### Mandatory edits before launching RL
- Update both `sys.path.insert(...)` lines in `DeepSWE_RL/rllm/rllm/trainer/verl/train_agent_ppo.py` `train_agent` to point at your VeRL + R2E-Gym checkouts.
- Ensure swebench wheels are installed inside RL env (RepoEnv imports them).
- If offline (no HF access on training nodes): pre-build test-spec cache with `data_preparation/make_test_spec.py --base_dir <dir>` and update matching `base_dir` in `R2E-Gym/src/r2egym/agenthub/runtime/docker.py`. Then `data_preparation/make_test_spec_for_rl.py` to produce the parquet.

## Stage 0 — Data prep
```bash
bash data_preparation/download_swe_datasets.sh
python data_preparation/difficulty_score_add.py        # optional
python data_preparation/difficulty_score_filter.py     # optional
python data_preparation/make_test_spec.py --base_dir <cache> --data_file_path <data.json>   # for offline / RL
python data_preparation/make_test_spec_for_rl.py
```
Example data: `data_examples/inference_data/ood_data` + `..._unit_tests_cache`.

## Test-time scaling
Use external repo `SWE-World` (`RUC-AIBOX/SWE-World-SWR-32B-w-cot`) as verifier; runs N parallel rollouts → ranks candidate patches. See https://github.com/RUCAIBox/SWE-World/blob/main/swe_world/src/simulation/tts/README.md

## Recommended progression
1. Smoke-test inference on `data_examples/inference_data/ood_data` (1–2 instances, `max_workers=1`) — verify Docker + tool copy + reward path.
2. Full-scale rollout with teacher model to produce SFT JSONL.
3. SFT (separate env) → produces 32B SFT checkpoint.
4. Eval SFT model via `rollout_trained_model/`.
5. RL (separate env) seeded from SFT checkpoint.
6. TTS via SWE-World for final eval.

## Open / pending items
- User has only installed inference env so far. SFT and RL envs not yet created.
- No data downloaded yet (`download_swe_datasets.sh` not run).
- Docker host IP not configured (`--ip` in run scripts still points to authors' internal `10.106.35.101`).
- Need to set `OPENAI_API_KEY` / `OPENAI_API_BASE` before any rollout.

## Glossary quick refs
- **FSDP**: PyTorch Fully Sharded DP. Each rank holds 1/N of params/grads/optim; all-gather per layer in fwd/bwd, reduce-scatter for grads. Equivalent to DeepSpeed ZeRO-3.
- **Ulysses SP**: sequence parallelism via all-to-all of (heads ↔ tokens) around attention.
- **Hybrid engine**: rollout (vLLM) and training (FSDP actor) share the same GPUs and process group; toggled by memory offload + brief weight sync.
- **GRPO**: group-relative PPO; advantage = `R_i − mean(R_j, j≠i)` within a group of n rollouts for the same prompt. No critic.
- **RLVR**: RL with Verifiable Rewards — here, pass/fail of hidden unit tests.

## Useful paths
- Inference run script: `R2E-Gym/rollout_foundation_model/run_deepseek_v3.sh`
- Inference params doc: `R2E-Gym/rollout_foundation_model/readme.md`
- SFT script: `OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_32B_new_remove_01_not_dedep.sh`
- RL script: `DeepSWE_RL/rllm/examples/swe/swe_rl_debug_hls_try_arsenal_long_step_and_long_time.sh`
- RL trainer (sync): `DeepSWE_RL/rllm/rllm/trainer/verl/agent_ppo_trainer.py`
- RL trainer (pipelined): `DeepSWE_RL/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py`
- Async exec engine: `DeepSWE_RL/rllm/rllm/engine/agent_execution_engine.py` (current) / `agent_execution_engine_backup1231.py` (reference).
- SWE env adapter for RL: `DeepSWE_RL/rllm/rllm/environments/swe/swe_arsenal.py` (wraps `r2egym.RepoEnv`).
