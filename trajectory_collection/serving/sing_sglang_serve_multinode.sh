#!/usr/bin/env bash
# Multi-node SGLang serving + IN-JOB sglang-router (one amlt job, N MI300X nodes).
#
#   sku: Nx192G8-MI300X, process_count_per_node: 1
#
# Every node serves a FULL TP=8 replica of the model on 0.0.0.0:8000 (reachable
# only on the internal job network). Rank 0 ALSO runs sglang-router, which
# load-balances across all N replicas, and opens the SINGLE Cloudflare tunnel
# that exposes the router. The rollout (on GCR) points at that one router URL
# and gets ~N x single-node throughput.
#
# Why in-job router: the router talks to the N workers over the fast internal
# eth0 fabric, and we need only ONE public tunnel (vs one per worker).
#
# Node discovery (Singularity injects these, see cluster-demo/Singularity/demo.yaml):
#   AZUREML_NODE_COUNT  -> number of nodes
#   NODE_RANK           -> this node's rank (0..N-1)
#   MASTER_ADDR         -> rank-0 address (we don't strictly need it; we use blob)
# Each node writes its own eth0 IP to a shared blob dir keyed by experiment;
# rank 0 globs all of them, health-checks each (= the connectivity proof), then
# starts the router over exactly the reachable ones (graceful partial degrade).

set -uo pipefail
hr() { echo; echo "==================== $* ===================="; }

MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct}"
PORT="${PORT:-8000}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_LEN="${MAX_LEN:-131072}"
MEM_FRAC="${MEM_FRAC:-0.90}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"   # 'auto'/aiter CK gives garbage on 480B
MAMBA_STRATEGY="${MAMBA_STRATEGY:-}"                  # Qwen3.5 hybrid Gated-DeltaNet: 'no_buffer' REQUIRED on AMD MI
POLICY="${POLICY:-cache_aware}"                       # RadixAttention prefix routing — great for SWE rollouts
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-48}"              # per-replica concurrent-request cap (protects the single-threaded detokenizer)
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.5}"             # cache_aware: min prefix-match rate to prefer the already-cached replica
BALANCE_ABS="${BALANCE_ABS:-16}"                      # cache_aware: divert to least-loaded once a replica exceeds this many in-flight.
BALANCE_REL="${BALANCE_REL:-1.3}"                     #   CRITICAL: default 64 > our client worker count => balancing NEVER fired => all 48
                                                     #   reqs funneled onto ONE prefix-owning replica (rank0 starved to ~2, rank1/2 melted at ~50).
MAX_RESTARTS="${MAX_RESTARTS:-50}"                    # supervisor: max auto-restarts per component (sglang/router/tunnel)
HEALTH_FAIL_THRESHOLD="${HEALTH_FAIL_THRESHOLD:-12}"  # consecutive local /health failures (~3min @15s loop) => replica HUNG (e.g. SGLang detokenizer stall) => restart
LOGSYNC_INTERVAL="${LOGSYNC_INTERVAL:-60}"            # blob log-sync period (was 20s; lighter on slow blobfuse + tmpfs)
LOG_CAP_MB="${LOG_CAP_MB:-200}"                       # truncate /tmp/sglang.log past this MB (tmpfs=RAM; keep small)
METRICS_INTERVAL="${METRICS_INTERVAL:-20}"            # tiny per-replica metrics JSON publish period (for the dev-box monitor)

NODE_COUNT="${AZUREML_NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
EXP="${AMLT_EXPERIMENT_NAME:-localexp}"

PUBDIR="${PUBDIR:-/mnt/murongma/sglang_workers}"       # rollout reads the router URL from here
LOGDIR="${LOGDIR:-/mnt/murongma/sglang_logs}"
CLUSTERDIR="${CLUSTERDIR:-/mnt/murongma/sglang_cluster/$EXP}"   # per-job node IP rendezvous
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
HOSTTAG="$(hostname | tr -c 'A-Za-z0-9_.-' '_')"
NODE_TAG="${EXP}_rank${NODE_RANK}"
BLOB_LOG="$LOGDIR/${NODE_TAG}_${STAMP}.sglang.log"
BLOB_STATUS="$LOGDIR/${NODE_TAG}_${STAMP}.status.txt"
NODE_URL_FILE="$CLUSTERDIR/node_${NODE_RANK}.url"
ROUTER_WORKER_FILE="$PUBDIR/${EXP}_router.json"        # the single URL the rollout uses
mkdir -p "$PUBDIR" "$LOGDIR" "$CLUSTERDIR" 2>/dev/null || true

status() { echo "$(date -u +%H:%M:%SZ) [rank$NODE_RANK] $*" | tee -a "$BLOB_STATUS" 2>/dev/null; }
# Preserve a crashed sglang log to blob BEFORE relaunch overwrites /tmp/sglang.log
# (launch_sglang uses `>` which truncates). Lets us see HSA/HIP OOM causes after a restart.
preserve_crash_log() {
  [ -f /tmp/sglang.log ] || return 0
  local dst="$LOGDIR/${NODE_TAG}_$(date -u +%Y%m%dT%H%M%SZ).CRASH.sglang.log"
  cp -f /tmp/sglang.log "$dst" 2>/dev/null && status "preserved crash log -> $dst"
}
start_log_sync() {
  ( while true; do
      if [ -f /tmp/sglang.log ]; then
        cp -f /tmp/sglang.log "$BLOB_LOG" 2>/dev/null || true
        # /tmp is tmpfs (RAM-backed); a chatty run can grow this until it fills RAM
        # and the host OOM-killer SIGKILLs sglang. Cap in place (same inode, so the
        # fd stays valid). Lower cap + longer period keeps blobfuse + tmpfs pressure
        # off the rank0 detokenizer (blob writes are slow; see README monitoring note).
        sz=$(stat -c%s /tmp/sglang.log 2>/dev/null || echo 0)
        [ "$sz" -gt $((LOG_CAP_MB*1024*1024)) ] && truncate -s 0 /tmp/sglang.log 2>/dev/null || true
      fi
      sleep "$LOGSYNC_INTERVAL"
    done ) & LOGSYNC_PID=$!
}

# Publish a TINY per-replica metrics JSON (few hundred bytes) to blob every
# METRICS_INTERVAL by scraping the local replica's Prometheus /metrics. The dev-box
# monitor reads THIS for live pressure/throughput instead of pulling the big log —
# decoupling monitoring from heavy/slow blobfuse I/O.
METRICS_FILE="$LOGDIR/${NODE_TAG}.metrics.json"
publish_metrics() {
  ( while true; do
      m=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null)
      if [ -n "$m" ]; then
        run=$(printf '%s\n' "$m" | awk '/^sglang:num_running_reqs/{v=$2} END{print (v==""?"null":v)}')
        que=$(printf '%s\n' "$m" | awk '/^sglang:num_queue_reqs/{v=$2}   END{print (v==""?"null":v)}')
        tok=$(printf '%s\n' "$m" | awk '/^sglang:token_usage/{v=$2}      END{print (v==""?"null":v)}')
        gen=$(printf '%s\n' "$m" | awk '/^sglang:gen_throughput/{v=$2}   END{print (v==""?"null":v)}')
        printf '{"rank":%s,"ip":"%s","running":%s,"queue":%s,"token_usage":%s,"gen_throughput":%s,"ts":"%s"}\n' \
          "$NODE_RANK" "$MY_IP" "$run" "$que" "$tok" "$gen" "$(date -u +%H:%M:%SZ)" > "$METRICS_FILE.tmp" 2>/dev/null \
          && mv -f "$METRICS_FILE.tmp" "$METRICS_FILE" 2>/dev/null || true
      fi
      sleep "$METRICS_INTERVAL"
    done ) & METRICS_PID=$!
}
cleanup() {
  status "EXIT (final flush)"
  [ -f /tmp/sglang.log ] && cp -f /tmp/sglang.log "$BLOB_LOG" 2>/dev/null || true
  rm -f "$NODE_URL_FILE" 2>/dev/null || true
  [ "$NODE_RANK" = "0" ] && rm -f "$ROUTER_WORKER_FILE" 2>/dev/null || true
  for p in "${SGL_PID:-}" "${CF_PID:-}" "${ROUTER_PID:-}" "${LOGSYNC_PID:-}" "${METRICS_PID:-}" "${HEARTBEAT_PID:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

hr "Multi-node SGLang — rank $NODE_RANK / $NODE_COUNT"
hostname; date -u
MY_IP=$(ip -4 addr show eth0 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
[ -z "$MY_IP" ] && MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
status "START host=$HOSTTAG ip=$MY_IP node_count=$NODE_COUNT model=$MODEL moe=$MOE_RUNNER_BACKEND"
echo "MY_IP=$MY_IP NODE_COUNT=$NODE_COUNT NODE_RANK=$NODE_RANK MASTER_ADDR=${MASTER_ADDR:-?}"

# ---------------- start SGLang (full TP=8 replica on every node) ----------------
hr "Start SGLang replica on rank $NODE_RANK"
EXTRA=()
[ -n "$TOOL_PARSER" ]        && EXTRA+=(--tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ]   && EXTRA+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$MOE_RUNNER_BACKEND" ] && EXTRA+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
[ -n "$MAMBA_STRATEGY" ]     && EXTRA+=(--mamba-scheduler-strategy "$MAMBA_STRATEGY")

# (Re)launch the local SGLang replica. Sets SGL_PID. Called at startup and by the
# supervisor when the replica is killed (e.g. host-RAM OOM) so serving self-heals.
launch_sglang() {
  nohup python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 0.0.0.0 --port "$PORT" \
      --tp-size "$TP_SIZE" \
      --context-length "$MAX_LEN" \
      --mem-fraction-static "$MEM_FRAC" \
      --attention-backend triton \
      --trust-remote-code \
      --watchdog-timeout 1200 \
      --max-running-requests "$MAX_RUNNING_REQ" \
      --enable-metrics \
      --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
      "${EXTRA[@]}" \
      > /tmp/sglang.log 2>&1 &
  SGL_PID=$!
  status "sglang launched pid=$SGL_PID"
}

# Block until the local replica is GENERATE-READY, or it dies / times out.
# Gate on /health, NOT /v1/models: /v1/models returns 200 as soon as the HTTP server
# is up (right after weights load), but the replica is still doing CUDA-graph capture +
# the ~190s aiter MoE JIT compile, during which /health returns 503. Gating on /health
# means READY == can actually generate, and (critically) the hang-detector — which also
# checks /health — only arms AFTER /health is genuinely 200, so a still-warming replica
# is never mistaken for "hung" and killed (the v4 rank2 kill-loop bug).
# $1 = max seconds (default 2700 = 45m cold load + warmup). 0=ready, 1=timeout, 2=died.
wait_sglang_ready() {
  local secs="${1:-2700}" i=0
  while [ "$i" -lt "$secs" ]; do
    sleep 10; i=$((i+10))
    curl -s -f --max-time 15 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SGL_PID" 2>/dev/null; then status "SGLang DIED during readiness wait"; tail -120 /tmp/sglang.log; return 2; fi
    (( i % 60 == 0 )) && status "loading/warming up... (${i}s; waiting for /health 200)"
  done
  return 1
}

# True when the local replica's HTTP /health fails while the PROCESS is still alive
# (e.g. SGLang detokenizer subprocess hangs -> "Health check failed ... detokenizer";
# the router marks the replica unhealthy and routes around it, but `kill -0 $SGL_PID`
# still passes so the plain death-check never restarts it). Use a consecutive-failure
# threshold so transient slowness under load does not trigger a needless restart.
local_replica_unhealthy() {
  ! curl -s -f --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

# Hard-restart the local replica (kill even if the main pid is alive-but-hung).
hard_restart_sglang() {
  preserve_crash_log
  kill "$SGL_PID" 2>/dev/null || true
  sleep 5
  kill -9 "$SGL_PID" 2>/dev/null || true
  launch_sglang
  if wait_sglang_ready 2700; then
    echo "http://$MY_IP:$PORT" > "$NODE_URL_FILE"
    status "replica recovered after hard-restart; republished $NODE_URL_FILE"
  fi
}

launch_sglang
start_log_sync

hr "Wait for local SGLang to be GENERATE-READY (/health 200 after warmup)"
if ! wait_sglang_ready 2700; then
  status "SGLang not ready on cold start; aborting node so Singularity can reschedule."
  exit 1
fi
status "LOCAL REPLICA READY at http://$MY_IP:$PORT"
publish_metrics   # start the tiny per-replica metrics feed for the monitor

# ---------------- publish this node's internal URL + heartbeat ----------------
echo "http://$MY_IP:$PORT" > "$NODE_URL_FILE"
( while kill -0 "$SGL_PID" 2>/dev/null; do touch "$NODE_URL_FILE" 2>/dev/null || true; sleep 30; done ) &
status "published $NODE_URL_FILE = http://$MY_IP:$PORT"

# ===================================================================
# Workers (rank >= 1): just keep serving. Rank 0 runs the router below.
# ===================================================================
if [ "$NODE_RANK" != "0" ]; then
  hr "Worker rank $NODE_RANK serving internally; supervising sglang (auto-restart on crash OR hang)."
  WSGL_RESTARTS=0; WHEALTH_FAILS=0
  while true; do
    if ! kill -0 "$SGL_PID" 2>/dev/null; then
      WSGL_RESTARTS=$((WSGL_RESTARTS+1))
      status "WARN worker replica died (restart #$WSGL_RESTARTS/$MAX_RESTARTS) — relaunching"
      [ "$WSGL_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL worker replica restart limit"; exit 1; }
      preserve_crash_log
      launch_sglang
      if wait_sglang_ready 2700; then
        echo "http://$MY_IP:$PORT" > "$NODE_URL_FILE"
        status "worker replica back up; republished $NODE_URL_FILE"
      fi
      WHEALTH_FAILS=0
    elif local_replica_unhealthy; then
      WHEALTH_FAILS=$((WHEALTH_FAILS+1))
      if [ "$WHEALTH_FAILS" -ge "$HEALTH_FAIL_THRESHOLD" ]; then
        WSGL_RESTARTS=$((WSGL_RESTARTS+1))
        status "WARN worker replica HUNG (/health failed ${WHEALTH_FAILS}x; restart #$WSGL_RESTARTS/$MAX_RESTARTS) — hard-restarting"
        [ "$WSGL_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL worker replica restart limit"; exit 1; }
        hard_restart_sglang
        WHEALTH_FAILS=0
      fi
    else
      WHEALTH_FAILS=0
    fi
    sleep 15
  done
fi

# ===================================================================
# Rank 0: gather all node URLs, health-check (connectivity proof),
# start router, open the single tunnel, register router URL for rollout.
# ===================================================================
hr "[rank0] Gather $NODE_COUNT node URLs from $CLUSTERDIR"
WORKER_URLS=()
for t in $(seq 1 120); do          # up to ~20 min for all replicas to load
  WORKER_URLS=()
  for f in "$CLUSTERDIR"/node_*.url; do
    [ -e "$f" ] || continue
    u=$(cat "$f" 2>/dev/null)
    [ -n "$u" ] && WORKER_URLS+=("$u")
  done
  status "discovered ${#WORKER_URLS[@]}/$NODE_COUNT replicas"
  [ "${#WORKER_URLS[@]}" -ge "$NODE_COUNT" ] && break
  sleep 10
done

hr "[rank0] Health-check each replica over internal network (CONNECTIVITY PROOF)"
LIVE_URLS=()
for u in "${WORKER_URLS[@]}"; do
  if curl -s -f --max-time 10 "$u/v1/models" >/dev/null 2>&1; then
    status "  REACHABLE   $u"
    LIVE_URLS+=("$u")
  else
    status "  unreachable $u  (intra-job connectivity FAILED for this peer)"
  fi
done
if [ "${#LIVE_URLS[@]}" -eq 0 ]; then
  status "[rank0] no reachable replicas?! falling back to local only"
  LIVE_URLS=("http://127.0.0.1:$PORT")
fi
status "[rank0] router will fan out over ${#LIVE_URLS[@]} replica(s): ${LIVE_URLS[*]}"

hr "[rank0] Start sglang-router ($POLICY) on :$ROUTER_PORT"
# (Re)start the router over the currently-live replicas. Sets ROUTER_PID.
# `nice`d below the sglang server: on rank0 the router+tunnel share CPU with this
# node's full TP=8 replica, and the single-threaded SGLang detokenizer can starve
# ("couldn't get a response from detokenizer") — only rank0 hung in the K=10 run
# while serve-only workers stayed healthy. Lower priority lets the detokenizer win CPU.
# Health check tightened (10s interval / 8s timeout) so a hung replica is dropped and
# re-added fast once the supervisor hard-restarts it.
start_router() {
  nohup nice -n 10 python3 -m sglang_router.launch_router \
      --host 0.0.0.0 --port "$ROUTER_PORT" \
      --policy "$POLICY" \
      --worker-urls "${LIVE_URLS[@]}" \
      --cache-threshold "$CACHE_THRESHOLD" \
      --balance-abs-threshold "$BALANCE_ABS" \
      --balance-rel-threshold "$BALANCE_REL" \
      --health-check-interval-secs 10 \
      --health-check-timeout-secs 8 \
      > /tmp/router.log 2>&1 &
  ROUTER_PID=$!
  status "router pid=$ROUTER_PID (nice+10)"
}
start_router
RREADY=""
for i in $(seq 1 60); do
  sleep 2
  curl -s -f "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null 2>&1 && { RREADY=yes; break; }
  kill -0 $ROUTER_PID 2>/dev/null || { status "router DIED"; tail -60 /tmp/router.log; break; }
done
[ -n "${RREADY:-}" ] && status "router healthy" || status "router health unconfirmed (continuing)"
curl -s "http://127.0.0.1:$ROUTER_PORT/workers" 2>/dev/null | head -c 600; echo

# ---------------- cloudflared: ONE tunnel exposing the router ----------------
CFBIN=/tmp/cloudflared
if [ ! -x "$CFBIN" ]; then
  curl -fSL --connect-timeout 10 --max-time 120 -o "$CFBIN" \
       https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CFBIN"
fi
hr "[rank0] Start cloudflared tunnel -> router :$ROUTER_PORT"

# Write the single rollout-facing URL file. Re-called on every tunnel (re)start
# because a trycloudflare quick tunnel hands out a NEW hostname each connect.
publish_router_url() {
  cat > "$ROUTER_WORKER_FILE" <<EOF
{
  "url": "$URL",
  "kind": "sglang_router",
  "model": "$MODEL",
  "replicas": ${#LIVE_URLS[@]},
  "policy": "$POLICY",
  "tool_parser": "$TOOL_PARSER",
  "amlt_experiment": "$EXP",
  "node_count": $NODE_COUNT,
  "started_utc": "$STAMP",
  "tunnel_started_utc": "$(date -u +%Y%m%dT%H%M%SZ)"
}
EOF
  status "published router URL -> $ROUTER_WORKER_FILE : $URL"
}

# (Re)start cloudflared, capture the fresh URL, republish. Sets CF_PID + URL.
launch_tunnel() {
  : > /tmp/cloudflared.log                 # truncate so we never grep a stale URL
  nohup nice -n 10 "$CFBIN" tunnel --url "http://localhost:$ROUTER_PORT" --no-autoupdate >> /tmp/cloudflared.log 2>&1 &
  CF_PID=$!
  URL=""
  local i
  for i in $(seq 1 120); do
    sleep 2
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -1 || true)
    [ -n "$URL" ] && break
    kill -0 "$CF_PID" 2>/dev/null || { status "cloudflared exited early"; break; }
  done
  [ -z "$URL" ] && { status "no tunnel URL yet"; tail -40 /tmp/cloudflared.log; return 1; }
  publish_router_url
  return 0
}

launch_tunnel || status "initial tunnel attempt failed; supervisor will retry"

if [ -n "${URL:-}" ]; then
  hr "[rank0] ROUTER PUBLIC URL"
  echo "============================================================"
  echo "  [ROUTER URL]  $URL   (fans out over ${#LIVE_URLS[@]} replicas)"
  echo "  [ROLLOUT]     export OPENAI_API_BASE=$URL/v1 ; OPENAI_API_KEY=not-needed"
  echo "  [WORKER FILE] $ROUTER_WORKER_FILE"
  echo "============================================================"
  sleep 8
  hr "[rank0] Remote sanity through router tunnel"
  curl -s --max-time 30 -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word PONG\"}],\"max_tokens\":8,\"temperature\":0}" \
    "$URL/v1/chat/completions" | head -c 300; echo
  status "router serving via $URL"
fi

# heartbeat the router worker file independent of restarts
( while true; do touch "$ROUTER_WORKER_FILE" 2>/dev/null || true; sleep 30; done ) & HEARTBEAT_PID=$!
# (Removed `tail -f /tmp/cloudflared.log` — it added a continuous reader/CPU consumer
#  on rank0 for no benefit; cloudflared already streams to /tmp/cloudflared.log and the
#  supervisor logs URL changes itself.)

# ----------------------------------------------------------------------------
# [rank0] SUPERVISOR — keep the sglang replica + router + tunnel alive. A crash
# in any single component is restarted in place instead of tearing the whole
# job down (the old `wait $SGL_PID` let one sglang OOM kill the tunnel+router).
# The "only the tunnel failed" case is fully recoverable: cloudflared is
# relaunched and the NEW url is republished to the rollout worker file.
# ----------------------------------------------------------------------------
hr "[rank0] Supervisor online (auto-restart sglang/router/tunnel on crash OR hang; max $MAX_RESTARTS each)"
SGL_RESTARTS=0; ROUTER_RESTARTS=0; TUN_RESTARTS=0; HEALTH_FAILS=0
while true; do
  sleep 15

  # 1) local sglang replica (router routes to http://$MY_IP:$PORT among others)
  if ! kill -0 "$SGL_PID" 2>/dev/null; then
    SGL_RESTARTS=$((SGL_RESTARTS+1))
    status "WARN local sglang replica died (restart #$SGL_RESTARTS/$MAX_RESTARTS) — relaunching"
    [ "$SGL_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL sglang restart limit"; exit 1; }
    preserve_crash_log
    launch_sglang
    if wait_sglang_ready 2700; then
      echo "http://$MY_IP:$PORT" > "$NODE_URL_FILE"
      status "local replica recovered; router re-adds it on next health check"
    fi
    HEALTH_FAILS=0
    continue
  fi

  # 1b) local replica ALIVE BUT HUNG (e.g. detokenizer stall) — router marks it
  #     unhealthy and routes around it; restart it so we get the replica back.
  if local_replica_unhealthy; then
    HEALTH_FAILS=$((HEALTH_FAILS+1))
    if [ "$HEALTH_FAILS" -ge "$HEALTH_FAIL_THRESHOLD" ]; then
      SGL_RESTARTS=$((SGL_RESTARTS+1))
      status "WARN local replica HUNG (/health failed ${HEALTH_FAILS}x; restart #$SGL_RESTARTS/$MAX_RESTARTS) — hard-restarting"
      [ "$SGL_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL sglang restart limit"; exit 1; }
      hard_restart_sglang
      HEALTH_FAILS=0
    fi
    continue
  else
    HEALTH_FAILS=0
  fi

  # 2) sglang-router
  if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    ROUTER_RESTARTS=$((ROUTER_RESTARTS+1))
    status "WARN sglang-router died (restart #$ROUTER_RESTARTS/$MAX_RESTARTS) — relaunching"
    [ "$ROUTER_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL router restart limit"; exit 1; }
    start_router
    sleep 5
    continue
  fi

  # 3) cloudflared tunnel (the "only tunnel failed" case) — republish new URL
  if ! kill -0 "$CF_PID" 2>/dev/null; then
    TUN_RESTARTS=$((TUN_RESTARTS+1))
    status "WARN cloudflared tunnel died (restart #$TUN_RESTARTS/$MAX_RESTARTS) — re-establishing"
    [ "$TUN_RESTARTS" -gt "$MAX_RESTARTS" ] && { status "FATAL tunnel restart limit"; exit 1; }
    launch_tunnel && echo "  [ROUTER URL re-established] $URL"
    continue
  fi
done
