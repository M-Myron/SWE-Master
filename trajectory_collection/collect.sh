#!/usr/bin/env bash
# Uniform entry point for full-dataset trajectory collection.
#
#   ./collect.sh <swegym|swesmith|swerebench> [K] [WAVE_IMAGES] [MAX_WORKERS]
#
# Positional args:
#   $1 DATASET       swegym | swesmith | swerebench            (required)
#   $2 K             max instances; 0 => ALL                   (default 0)
#   $3 WAVE_IMAGES   images resident at once (peak disk ~ N*4GB)(default 8)
#   $4 MAX_WORKERS   parallel rollouts inside each wave         (default 6)
#
# Examples:
#   ./collect.sh swerebench            # ALL swerebench (6542), waves of 8 images, 6 workers
#   ./collect.sh swesmith 500 8 48     # first 500 swesmith, 8 images/wave, 48 workers
#   ./collect.sh swegym 0 48 48        # ALL swegym (1 inst/image -> wave_images=workers)
#
# Env overrides (optional):
#   URL=...            router base URL (default: read from blob via blob_sas.sh)
#   TEMP=0.6           sampling temperature
#   MAX_STEPS=100      max agent steps per instance
#   USE_FN_CALLING=True
#   SWE_MASTER_PY=...  python interpreter (default: swe-master conda env)
#   OUT_ROOT=...       output root (default: <this dir>/collect_runs)
#   EXP=...            experiment name (default: <dataset>_full)
#   ROUTER_JSON=...    blob path of the router json
#
# Everything is resumable: re-run the SAME command and it skips instances already
# collected (matched by instance_id in <OUT_ROOT>/<EXP>/<EXP>.jsonl).
#
# Design notes:
#   - Reward is computed INLINE per instance (same container, right after the rollout)
#     by R2E-Gym — no separate eval pass, no second image pull.
#   - Instances are grouped by docker image and processed in WAVES; each wave's images
#     are pulled, all their instances run (rollout+reward), then `docker rmi`'d. Peak
#     disk ~ WAVE_IMAGES x ~4GB. swesmith has ~266 instances/image so grouping is a big win.
#   - The router URL is read from blob each run (it changes on tunnel restart). Override
#     with URL=... if needed.
set -euo pipefail

DATASET="${1:?usage: $0 <swegym|swesmith|swerebench> [K] [WAVE_IMAGES] [MAX_WORKERS]}"
K="${2:-0}"                      # 0 => all instances
WAVE_IMAGES="${3:-8}"
MAX_WORKERS="${4:-6}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SWE_MASTER_PY:-/home/v-murongma/miniconda3/envs/swe-master/bin/python}"

# K=0 means "all" -> omit --k so the driver collects the whole dataset.
KFLAG=()
[ "$K" != "0" ] && KFLAG=(--k "$K")

# Optional overrides -> CLI flags only when set.
URLFLAG=();    [ -n "${URL:-}" ]         && URLFLAG=(--url "$URL")
OUTFLAG=();    [ -n "${OUT_ROOT:-}" ]    && OUTFLAG=(--out_root "$OUT_ROOT")
EXPFLAG=();    [ -n "${EXP:-}" ]         && EXPFLAG=(--exp "$EXP")
RJFLAG=();     [ -n "${ROUTER_JSON:-}" ] && RJFLAG=(--router_json "$ROUTER_JSON")

exec "$PY" "$HERE/collect.py" \
  --dataset      "$DATASET" \
  "${KFLAG[@]}" \
  --wave_images  "$WAVE_IMAGES" \
  --max_workers  "$MAX_WORKERS" \
  --temperature  "${TEMP:-0.6}" \
  --max_steps    "${MAX_STEPS:-100}" \
  --use_fn_calling "${USE_FN_CALLING:-True}" \
  "${URLFLAG[@]}" "${OUTFLAG[@]}" "${EXPFLAG[@]}" "${RJFLAG[@]}"
