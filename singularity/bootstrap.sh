#!/usr/bin/env bash
# Tiny bootstrap for the Singularity job.
#
# The image (msraairgroup.azurecr.io/swe-master-serve-agent) already contains all
# dependencies (vLLM ROCm serving + isolated /opt/venvs/swe-agent R2E-Gym venv +
# cloudflared/socat/docker-cli). This bootstrap simply pulls the latest source and
# hands off to the combined entrypoint, so the whole run is GitHub-driven and no
# code/patch is baked.
#
# Everything is configured by env vars (set in the amlt yaml). See
# singularity/run_serve_and_rollout.sh for the full list.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/M-Myron/SWE-Master.git}"
REPO_REF="${REPO_REF:-main}"
BOOT_DIR="${BOOT_DIR:-/workspace/swe_boot}"

echo "==================== bootstrap ===================="
echo "REPO_URL=$REPO_URL  REPO_REF=$REPO_REF"
hostname; date -u

rm -rf "$BOOT_DIR"; mkdir -p "$BOOT_DIR"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$BOOT_DIR/SWE-Master" \
  || git clone "$REPO_URL" "$BOOT_DIR/SWE-Master"

ENTRY="$BOOT_DIR/SWE-Master/singularity/run_serve_and_rollout.sh"
chmod +x "$ENTRY"
# Tell the entrypoint to reuse the already-cloned tree (skip a second clone).
export WORKROOT="${WORKROOT:-$BOOT_DIR/run}"
export PRECLONED_SRC="$BOOT_DIR/SWE-Master"
exec bash "$ENTRY"
