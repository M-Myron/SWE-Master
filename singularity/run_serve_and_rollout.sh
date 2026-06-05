#!/usr/bin/env bash
# =============================================================================
# SWE-Master combined in-job entrypoint (Singularity MI300X).
#
# This is the "RL-shaped" topology: ONE Singularity job both
#   (1) SERVES the policy LLM with vLLM (local, on the job's 8 MI300X GPUs), and
#   (2) acts as the R2E-Gym CLIENT that drives a SWE rollout, while
#   (3) the Docker work (issue containers) runs on a REMOTE host (the GCR dev box)
#       reached over a Cloudflare *named* TCP tunnel.
#
# i.e. the LLM is local (127.0.0.1:$PORT), the Docker daemon is remote
# (127.0.0.1:$DOCKER_LOCAL_PORT -> cloudflared -> GCR dockerd).
#
# Everything dependency-wise is already baked into the image:
#   * system python  -> vLLM (ROCm) serving stack
#   * /opt/venvs/swe-agent -> isolated R2E-Gym agent venv (transformers 4.45.2 etc.)
#   * cloudflared / socat / docker CLI / jq / ss / uv
#
# The SWE-Master *source* is cloned fresh from GitHub at job time (REPO_URL), so
# code/patches are never re-applied by hand.
#
# ---- Required env (set in the amlt yaml) ------------------------------------
#   REPO_URL          git repo to clone (default: M-Myron/SWE-Master)
#   REPO_REF          branch/tag/sha to check out (default: main)
#   DOCKER_TUNNEL_HOST  cloudflare hostname of the remote dockerd (named tunnel)
#                       e.g. docker.swerl-docker-connection.uk
# ---- Optional env -----------------------------------------------------------
#   MODEL             served model (default Qwen/Qwen3-Coder-480B-A35B-Instruct)
#   PORT              vLLM port (default 8000)
#   TP_SIZE           tensor parallel (default 8)
#   MAX_LEN           context window (default 131072)
#   GPU_MEM_UTIL      vLLM gpu mem frac (default 0.90)
#   DOCKER_LOCAL_PORT local port the docker tunnel listens on (default 2375)
#   K                 rollouts per instance (default 1)
#   MAX_STEPS         agent step cap (default 30)
#   DATASET           path (in repo) to the inference dataset json
#   SCAFFOLD          openhands | r2egym | sweagent (default openhands)
#   PUBDIR            where to publish status/artifacts (default /mnt/murongma/swe_rl)
# =============================================================================
set -uo pipefail
hr() { echo; echo "==================== $* ===================="; }
die() { echo "FATAL: $*" >&2; sleep "${KEEPALIVE_ON_FAIL:-600}"; exit 1; }

# ---------------- config ----------------
REPO_URL="${REPO_URL:-https://github.com/M-Myron/SWE-Master.git}"
REPO_REF="${REPO_REF:-main}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_LEN="${MAX_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DOCKER_TUNNEL_HOST="${DOCKER_TUNNEL_HOST:-docker.swerl-docker-connection.uk}"
DOCKER_LOCAL_PORT="${DOCKER_LOCAL_PORT:-2375}"
K="${K:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
SCAFFOLD="${SCAFFOLD:-openhands}"
API_KEY="${API_KEY:-not-needed}"
PUBDIR="${PUBDIR:-/mnt/murongma/swe_rl}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WORKROOT="${WORKROOT:-/workspace/run_${STAMP}}"
AGENT_PY=/opt/venvs/swe-agent/bin/python
# Default dataset: the single-instance public-docker demo shipped in the repo.
DATASET="${DATASET:-data_examples/inference_data/swe_bench_verified/test_swe_bench_verified_full_ip_demo_pubdocker.json}"

mkdir -p "$PUBDIR" "$WORKROOT"
exec > >(tee -a "$WORKROOT/entrypoint.log") 2>&1

hr "Environment"
hostname; date -u
echo "MODEL=$MODEL PORT=$PORT TP=$TP_SIZE MAX_LEN=$MAX_LEN"
echo "REPO_URL=$REPO_URL REPO_REF=$REPO_REF"
echo "DOCKER_TUNNEL_HOST=$DOCKER_TUNNEL_HOST DOCKER_LOCAL_PORT=$DOCKER_LOCAL_PORT"
echo "WORKROOT=$WORKROOT PUBDIR=$PUBDIR"
command -v cloudflared docker socat ss jq >/dev/null || die "missing baked tools"
"$AGENT_PY" -c "import r2egym, litellm, docker; print('agent venv OK', r2egym.__file__)" \
  || die "agent venv broken"

# ---------------- clone source ----------------
hr "Clone SWE-Master source ($REPO_REF)"
SRC="$WORKROOT/SWE-Master"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SRC" \
  || git clone "$REPO_URL" "$SRC" || die "git clone failed"
cd "$SRC"
git --no-pager log --oneline -1 || true
# Re-point the editable agent install at this fresh checkout so our patched
# R2E-Gym source (qwen3-coder fn-calling, /run_tests.sh writer, TLS off) is used.
"$AGENT_PY" -m pip install -e "$SRC/R2E-Gym" --no-deps -q \
  && echo "re-pointed swe-agent venv at fresh R2E-Gym checkout" \
  || echo "WARN: editable re-point failed; baked R2E-Gym will be used"

# =============================================================================
# (1) SERVE vLLM locally
# =============================================================================
hr "Start vLLM ($MODEL, TP=$TP_SIZE)"
export HF_HOME="${HF_HOME:-/scratch/hf_cache}"; mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_ENABLE_HF_TRANSFER=0
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_LEN" --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "$WORKROOT/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vllm pid=$VLLM_PID log=$WORKROOT/vllm.log"

hr "Wait for vLLM /v1/models (up to 45 min)"
READY=""
for i in $(seq 1 270); do
  sleep 10
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { READY=yes; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -100 "$WORKROOT/vllm.log"; die "vLLM exited"; }
  (( i % 6 == 0 )) && { echo "  waiting $((i*10))s"; tail -2 "$WORKROOT/vllm.log" | sed 's/^/    [vllm] /'; }
done
[ -n "$READY" ] || { tail -150 "$WORKROOT/vllm.log"; die "vLLM not ready in 45m"; }
echo "vLLM ready."
LLM_BASE="http://127.0.0.1:$PORT/v1"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word PONG\"}],\"max_tokens\":8,\"temperature\":0}" \
  | sed 's/^/  /'; echo

# =============================================================================
# (2) OPEN the remote-Docker tunnel (cloudflared access tcp -> GCR dockerd)
# =============================================================================
hr "Open Docker tunnel to $DOCKER_TUNNEL_HOST -> 127.0.0.1:$DOCKER_LOCAL_PORT"
# Evict anything squatting the local port, then start the access listener.
for pid in $(ss -ltnp 2>/dev/null | awk -F'pid=' "/127.0.0.1:$DOCKER_LOCAL_PORT /{print \$2}" | awk -F',' '{print $1}'); do
  [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
done
nohup cloudflared access tcp \
  --hostname "$DOCKER_TUNNEL_HOST" \
  --url "tcp://127.0.0.1:$DOCKER_LOCAL_PORT" \
  > "$WORKROOT/cf-access.log" 2>&1 &
CF_PID=$!
echo "cloudflared access pid=$CF_PID log=$WORKROOT/cf-access.log"
for i in $(seq 1 60); do
  ss -ltn 2>/dev/null | grep -q "127.0.0.1:$DOCKER_LOCAL_PORT " && break
  sleep 1
done
ss -ltn 2>/dev/null | grep -q "127.0.0.1:$DOCKER_LOCAL_PORT " || { tail -60 "$WORKROOT/cf-access.log"; die "docker tunnel listener never came up"; }

export DOCKER_HOST="tcp://127.0.0.1:$DOCKER_LOCAL_PORT"
export DOCKER_TLS_VERIFY=""
hr "Verify remote dockerd"
for i in $(seq 1 20); do
  if docker version >/dev/null 2>&1; then break; fi
  sleep 2
done
docker version --format 'remote dockerd OK: server={{.Server.Version}}' \
  || { tail -60 "$WORKROOT/cf-access.log"; die "cannot reach remote dockerd via tunnel"; }

# =============================================================================
# (3) RUN the R2E-Gym rollout (LLM local, Docker remote)
# =============================================================================
hr "Run R2E-Gym rollout (scaffold=$SCAFFOLD, k=$K, max_steps=$MAX_STEPS)"
EXP_NAME="sing_swe_${STAMP}"
TRAJ_DIR="$WORKROOT/$EXP_NAME"; mkdir -p "$TRAJ_DIR"
export OPENAI_API_BASE="$LLM_BASE"
export OPENAI_API_KEY="$API_KEY"
cd "$SRC/R2E-Gym"
set +e
"$AGENT_PY" -m r2egym.agenthub.run.edit runagent_multiple \
    --dataset    "$SRC/$DATASET" \
    --split      test \
    --k          "$K" \
    --start_idx  0 \
    --traj_dir   "$TRAJ_DIR" \
    --exp_name   "$EXP_NAME" \
    --llm_name   "openai/$MODEL" \
    --temperature 0.0 \
    --use_fn_calling True \
    --backend    docker \
    --scaffold   "$SCAFFOLD" \
    --max_steps  "$MAX_STEPS" \
    --ip         "127.0.0.1" \
    --use_lsp    False \
    2>&1 | tee "$TRAJ_DIR/run.log"
RC=${PIPESTATUS[0]}
set -e

hr "Publish results"
REWARD=$("$AGENT_PY" - <<PY 2>/dev/null
import json,glob,os
js=sorted(glob.glob("$TRAJ_DIR/*.jsonl"))
r="NA"
if js:
    last=None
    for ln in open(js[0]):
        ln=ln.strip()
        if ln: last=json.loads(ln)
    if last and "reward" in last: r=last["reward"]
print(r)
PY
)
echo "exit_rc=$RC reward=$REWARD traj=$TRAJ_DIR"
cp -f "$TRAJ_DIR/run.log" "$PUBDIR/run_${STAMP}.log" 2>/dev/null || true
cat > "$PUBDIR/status_${STAMP}.json" <<EOF
{"stamp":"$STAMP","model":"$MODEL","scaffold":"$SCAFFOLD","k":$K,"max_steps":$MAX_STEPS,
 "exit_rc":$RC,"reward":"$REWARD","repo_ref":"$REPO_REF",
 "docker_tunnel_host":"$DOCKER_TUNNEL_HOST","amlt_job":"${AMLT_JOB_NAME:-unknown}"}
EOF
cp -rf "$TRAJ_DIR" "$PUBDIR/" 2>/dev/null || true
echo "published to $PUBDIR"

hr "Done (rc=$RC, reward=$REWARD). Keeping vLLM alive ${KEEPALIVE_AFTER:-300}s for inspection."
sleep "${KEEPALIVE_AFTER:-300}"
