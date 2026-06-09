#!/usr/bin/env bash
# Read artifacts the Singularity serving job published to the blob container
# (the job's mount is not visible from a dev box, so we use the SAS URL instead).
#
# Usage:
#   bash blob_sas.sh list  [prefix]        # list blobs under prefix (default swe_rl/)
#   bash blob_sas.sh cat   <blobpath>      # print a single blob to stdout
#   bash blob_sas.sh get   <prefix> <dst>  # download a prefix/blob to a local dir
#
# The router URL the collector needs lives at e.g.:
#   sglang_workers/sing_sglang_router_qwen35_397b_3node_v5_router.json   (a {"url": ...} doc)
#
# The SAS token is read from the repo cred file and never echoed. Override the cred
# path with SASFILE=... ; needs `azcopy` on PATH.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # SWE-Master
SASFILE="${SASFILE:-$REPO_ROOT/cred/zhibinmain_murongma_sas.url}"
[ -f "$SASFILE" ] || { echo "SAS file not found: $SASFILE" >&2; exit 1; }
SAS_URL="$(grep -vE '^#|^[[:space:]]*$' "$SASFILE" | head -1)"
# Split into base (https://acct.blob.core.windows.net/container) and ?token
BASE="${SAS_URL%%\?*}"
TOKEN="${SAS_URL#*\?}"

# Build a URL for a given path under the container: BASE/path?TOKEN
url_for() { local p="${1#/}"; if [ -n "$p" ]; then echo "${BASE}/${p}?${TOKEN}"; else echo "${BASE}?${TOKEN}"; fi; }

cmd="${1:-list}"; shift || true
case "$cmd" in
  list)
    PREFIX="${1:-swe_rl/}"
    AZCOPY_LOG_LEVEL=ERROR azcopy list "$(url_for "$PREFIX")" 2>&1 | grep -vE "INFO:|Authentication|Log file" || true
    ;;
  cat)
    BLOB="${1:?usage: blob_sas.sh cat <blobpath>}"
    tmp="$(mktemp)"
    AZCOPY_LOG_LEVEL=ERROR azcopy copy "$(url_for "$BLOB")" "$tmp" --from-to BlobLocal --overwrite=true >/dev/null 2>&1
    cat "$tmp"; rm -f "$tmp"
    ;;
  get)
    PREFIX="${1:?usage: blob_sas.sh get <prefix> <dst>}"; DST="${2:?dst dir}"
    mkdir -p "$DST"
    AZCOPY_LOG_LEVEL=ERROR azcopy copy "$(url_for "$PREFIX")" "$DST" --recursive=true --overwrite=true 2>&1 | tail -8
    ;;
  *) echo "unknown cmd: $cmd" >&2; exit 2;;
esac
