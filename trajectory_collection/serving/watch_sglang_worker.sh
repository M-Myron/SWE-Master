#!/usr/bin/env bash
# Poll a SINGLE-NODE SGLang Singularity worker job via blob SAS: job status,
# blob status file, blob sglang log milestones, registered worker json, and a
# live probe of the newest worker URL.
#
#   ./watch_sglang_worker.sh [EXP]
#
# This is for the single-node serve flow (one replica + its own tunnel). For the
# multi-node in-job router flow use watch_sglang_router.sh instead.
#
# Env overrides:
#   AMLT_ENV   conda env with amlt (default amlt10)
#   BLOB_SAS   path to blob_sas.sh (default: ../blob_sas.sh next to this dir)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOB="${BLOB_SAS:-$HERE/../blob_sas.sh}"
source /home/v-murongma/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "${AMLT_ENV:-amlt10}" 2>/dev/null

EXP="${1:-sing_sglang_qwen3_coder_480b_v7}"

echo "==================== amlt status: $EXP ===================="
amlt status "$EXP" 2>/dev/null | grep -vE "Cryptography|x509|update to|instructions" | tail -12

echo
echo "==================== blob status file (newest for EXP) ===================="
S=$(bash "$BLOB" list sglang_logs 2>/dev/null | grep "${EXP}.*status.txt" | tail -1 | awk '{print $1}' | tr -d ';')
if [ -n "$S" ]; then
  echo "--- $S ---"; bash "$BLOB" cat "sglang_logs/$S" 2>/dev/null | tail -8
else
  echo "  (no status file yet)"
fi

echo
echo "==================== blob SGLang log: key milestones (newest) ===================="
NEWLOG=$(bash "$BLOB" list sglang_logs 2>/dev/null | grep "${EXP}.*sglang.log" | tail -1 | awk '{print $1}' | tr -d ';')
if [ -n "$NEWLOG" ]; then
  bash "$BLOB" cat "sglang_logs/$NEWLOG" 2>/dev/null > /tmp/sn_worker.log
  echo "--- $NEWLOG ($(wc -l < /tmp/sn_worker.log) lines) ---"
  grep -iE "Load weight end|KV Cache is allocated|Capture cuda graph|Capturing batches|PermissionError|nvcc|Scheduler hit|sigquit|Exception|Traceback|server is fired|Uvicorn running|The server is|ready" /tmp/sn_worker.log 2>/dev/null | tail -15
  echo "--- raw tail ---"; tail -5 /tmp/sn_worker.log 2>/dev/null
else
  echo "  (no blob sglang log yet — job still preparing/queued or pre-launch)"
fi

echo
echo "==================== registered worker json (blob) ===================="
W=$(bash "$BLOB" list sglang_workers 2>/dev/null | grep "${EXP}.*json" | tail -1 | awk '{print $1}' | tr -d ';')
if [ -n "$W" ]; then
  echo "--- sglang_workers/$W ---"
  bash "$BLOB" cat "sglang_workers/$W" 2>/dev/null
  URL=$(bash "$BLOB" cat "sglang_workers/$W" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('url',''))" 2>/dev/null)
  if [ -n "$URL" ]; then
    echo
    echo "==================== probe worker $URL ===================="
    echo "--- /v1/models ---";       curl -s --max-time 15 "$URL/v1/models" | head -c 300; echo
    echo "--- /get_model_info ---";  curl -s --max-time 15 "$URL/get_model_info" | head -c 300; echo
  fi
else
  echo "  (no worker registered yet — model still loading)"
fi
