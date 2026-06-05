#!/usr/bin/env bash
# Build (and optionally push) the SWE-Master serving+agent image for Singularity MI300X.
#
# Usage:
#   bash docker/build_and_push.sh build        # build only (default)
#   bash docker/build_and_push.sh push         # build + push to ACR
#   IMAGE_TAG=v2 bash docker/build_and_push.sh push
#
# The build context is the repo ROOT (the Dockerfile COPYs R2E-Gym + the three wheels).
# .dockerignore keeps .venv/.git/etc. out of the context.
set -euo pipefail

ACTION="${1:-build}"
REGISTRY="${REGISTRY:-msraairgroup.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-swe-master-serve-agent}"
IMAGE_TAG="${IMAGE_TAG:-rocm7.2-vllm0.18-agent-v1}"
FULL="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.mi300x"

echo "================================================================"
echo "  image      : ${FULL}"
echo "  context    : ${REPO_ROOT}"
echo "  dockerfile : ${DOCKERFILE}"
echo "  action     : ${ACTION}"
echo "================================================================"

cd "${REPO_ROOT}"

echo "[build] docker build ..."
docker build \
    -f "${DOCKERFILE}" \
    -t "${FULL}" \
    "${REPO_ROOT}"

echo "[build] done: ${FULL}"
docker images "${FULL}" --format '  size: {{.Size}}'

if [[ "${ACTION}" == "push" ]]; then
    echo "[push] az acr login -n ${REGISTRY%%.*} ..."
    az acr login -n "${REGISTRY%%.*}"
    echo "[push] docker push ${FULL} ..."
    docker push "${FULL}"
    echo "[push] done: ${FULL}"
else
    echo "[skip] not pushing (pass 'push' to push). To push later:"
    echo "       az acr login -n ${REGISTRY%%.*} && docker push ${FULL}"
fi
