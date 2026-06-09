#!/usr/bin/env bash
# Live serving dashboard for the multi-node SGLang router — run from the dev box.
#
#   ./monitor_serving.sh [EXP] [--watch [SECS]] [--full]
#
# Default (fast, NO blob I/O): queries the router tunnel for per-replica in-flight
# load + health + router health. This alone shows whether traffic is SPREAD evenly
# across replicas or CONCENTRATED on one (the failure mode we fixed).
#
#   --full     also pull each replica's tiny metrics JSON from blob (running/queue/
#              kv-usage/throughput). A few small azcopy reads (slower).
#   --watch N  refresh every N seconds (default 6).
#
# Env: BLOB_SAS=path to blob_sas.sh (default ../blob_sas.sh); AMLT_ENV (default amlt10).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOB="${BLOB_SAS:-$HERE/../blob_sas.sh}"
EXP="sing_sglang_router_qwen35_397b_8node_v6"
WATCH=""; SECS=6; FULL=""
for a in "$@"; do
  case "$a" in
    --watch) WATCH=1 ;;
    --full)  FULL=1 ;;
    [0-9]*)  SECS="$a" ;;
    *)       EXP="$a" ;;
  esac
done

render() {
  local URL
  URL=$(bash "$BLOB" cat "sglang_workers/${EXP}_router.json" 2>/dev/null \
        | python3 -c 'import json,sys;print(json.load(sys.stdin).get("url",""))' 2>/dev/null)
  echo "================ serving monitor: $EXP  @ $(date -u +%H:%M:%SZ) ================"
  if [ -z "$URL" ]; then echo "  router URL not published yet (job cold-starting?)"; return; fi
  local rh
  rh=$(curl -s --max-time 10 "$URL/health" -o /dev/null -w "%{http_code}" 2>/dev/null)
  echo "router: $URL   /health=$rh"
  local loads workers
  loads=$(curl -s --max-time 12 "$URL/get_loads" 2>/dev/null)
  workers=$(curl -s --max-time 12 "$URL/workers" 2>/dev/null)
  EXP="$EXP" BLOB="$BLOB" FULL="$FULL" python3 - "$loads" "$workers" <<'PY'
import json, os, subprocess, sys
loads_raw, workers_raw = sys.argv[1], sys.argv[2]
exp, blob, full = os.environ["EXP"], os.environ["BLOB"], os.environ.get("FULL")
def jload(s, key):
    try: return json.loads(s).get(key, [])
    except Exception: return []
loads = {w.get("worker"): w.get("load") for w in jload(loads_raw, "workers")}
health = {w.get("url"): w.get("is_healthy") for w in jload(workers_raw, "workers")}
urls = list(health) or list(loads)
if not urls:
    print("  no workers registered yet"); sys.exit(0)

# Optional: enrich with the tiny per-replica metrics JSON from blob.
metrics = {}
if full:
    def cat(p):
        try: return subprocess.run(["bash", blob, "cat", p], capture_output=True,
                                    text=True, timeout=30).stdout
        except Exception: return ""
    for rank in range(8):
        txt = cat(f"sglang_logs/{exp}_rank{rank}.metrics.json")
        if not txt.strip(): continue
        try:
            m = json.loads(txt); metrics[m.get("ip")] = m
        except Exception: pass

ip_of = lambda u: u.split("//")[-1].split(":")[0]
hdr = f"{'replica':24} {'health':7} {'load':>5}"
if full: hdr += f" {'run':>4} {'queue':>5} {'kv%':>5} {'tok/s':>7}"
print(hdr)
tot = 0
for u in sorted(urls):
    ld = loads.get(u, "?"); tot += (ld if isinstance(ld, int) else 0)
    hl = health.get(u); hs = "UP" if hl else ("DOWN" if hl is False else "?")
    row = f"{u.replace('http://',''):24} {hs:7} {str(ld):>5}"
    if full:
        m = metrics.get(ip_of(u), {})
        def g(k):
            v = m.get(k); return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
        kv = m.get("token_usage"); kvs = "-" if kv is None else f"{float(kv)*100:.0f}"
        row += f" {g('running'):>4} {g('queue'):>5} {kvs:>5} {g('gen_throughput'):>7}"
    print(row)
print(f"{'TOTAL in-flight (router view)':24} {'':7} {tot:>5}")
n = len(urls)
if n and isinstance(tot, int) and tot:
    spread = max((loads.get(u,0) or 0) for u in urls) - min((loads.get(u,0) or 0) for u in urls)
    print(f"  spread(max-min load)={spread}  (≈0 = evenly balanced; large = concentrated on one replica)")
PY
}

if [ -n "$WATCH" ]; then
  while true; do clear; render; echo; echo "(--watch ${SECS}s; Ctrl-C to stop)"; sleep "$SECS"; done
else
  render
fi
