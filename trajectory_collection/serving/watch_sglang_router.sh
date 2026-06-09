#!/usr/bin/env bash
# Watch the multi-node in-job SGLang ROUTER job via blob SAS (GCR /mnt is flaky).
#
#   ./watch_sglang_router.sh [EXP]
#
# EXP defaults to the current 3-node Qwen3.5 router job. Reads everything the job
# published to blob: per-rank status, node-IP rendezvous, the router worker json
# (= the rollout URL), and rank-0 milestones (connectivity proof + router up).
#
# Env overrides:
#   AMLT_ENV   conda env with amlt (default amlt10)
#   BLOB_SAS   path to blob_sas.sh (default: ../blob_sas.sh next to this dir)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOB="${BLOB_SAS:-$HERE/../blob_sas.sh}"
source /home/v-murongma/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "${AMLT_ENV:-amlt10}" 2>/dev/null

EXP="${1:-sing_sglang_router_qwen35_397b_8node_v6}"

echo "==================== amlt status: $EXP ===================="
amlt status "$EXP" 2>/dev/null | grep -vE "Cryptography|x509|update to|instructions" \
  | grep -E ":0|:1|:2|sglang|queued|running|completed|failed|DURATION" | head -8

echo
echo "==================== per-rank status files (blob) ===================="
bash "$BLOB" list sglang_logs 2>/dev/null | grep "${EXP}.*status.txt" | while read -r line; do
  f=$(echo "$line" | awk '{print $1}' | tr -d ';')
  echo "--- $f ---"
  bash "$BLOB" cat "sglang_logs/$f" 2>/dev/null | tail -6
done

echo
echo "==================== cluster node IPs published (blob) ===================="
bash "$BLOB" list "sglang_cluster/$EXP" 2>/dev/null | tail -10

echo
echo "==================== router worker file (the rollout URL) ===================="
bash "$BLOB" cat "sglang_workers/${EXP}_router.json" 2>/dev/null || echo "  (router not up yet)"

echo
echo "==================== rank-0 log: connectivity proof + router milestones ===================="
R0=$(bash "$BLOB" list sglang_logs 2>/dev/null | grep "${EXP}_rank0.*sglang.log" | tail -1 | awk '{print $1}' | tr -d ';')
if [ -n "$R0" ]; then
  bash "$BLOB" cat "sglang_logs/$R0" 2>/dev/null > /tmp/mn_rank0.log
  echo "rank0 log: $R0 ($(wc -l < /tmp/mn_rank0.log) lines)"
  grep -iE "REACHABLE|unreachable|discovered|router will fan|router healthy|ROUTER URL|ROUTER PUBLIC|server is fired|Capture cuda graph end|PONG|connectivity" /tmp/mn_rank0.log | tail -20
else
  echo "  (rank0 log not present yet)"
fi
