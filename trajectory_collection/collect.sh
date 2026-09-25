#!/usr/bin/env bash
# Unified entry point for trajectory collection.
#
#   ./collect.sh <dataset> [--flag value ...]
#
# <dataset> is any name registered in build_dataset.py (swegym | swesmith | swerebench | ...).
# All settings come from collect_config.yaml (serving URL, model, max_steps, temperature,
# max_workers, per-dataset overrides). ANY setting is overridable on the CLI.
#
# Examples:
#   ./collect.sh swegym                          # full swegym, config defaults
#   ./collect.sh swesmith --max_workers 32       # override concurrency
#   ./collect.sh swerebench --k 200              # first 200 only (smoke)
#   ./collect.sh swegym --url https://x.trycloudflare.com   # override endpoint
#   ./collect.sh swegym --config my_config.yaml  # use a different config file
#
# Output goes to:  <out_root>/<dataset>/<model>_iter<max_steps>_t<temp>[_nofc]/trajectories.jsonl
# Resumable: re-run the SAME command; already-collected instances are skipped.
#
# Env:
#   SWE_MASTER_PY   python interpreter (default: swe-master conda env)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SWE_MASTER_PY:-/home/v-murongma/miniconda3/envs/swe-master/bin/python}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <dataset> [--flag value ...]   (see collect_config.yaml)" >&2
  "$PY" "$HERE/build_dataset.py" >&2 || true
  exit 2
fi

exec "$PY" "$HERE/collect.py" "$@"
