#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

IMAGE_NAME=${IMAGE_NAME:-deepinfer:cu132-sm100-8b16149741a4}
MAX_JOBS=${MAX_JOBS:-4}
NVCC_THREADS=${NVCC_THREADS:-2}
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] && (( MAX_JOBS <= 32 )) || { echo 'MAX_JOBS must be 1..32' >&2; exit 2; }
[[ "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]] && (( NVCC_THREADS <= 8 )) || { echo 'NVCC_THREADS must be 1..8' >&2; exit 2; }
BASE_IMAGE=$(python3 -c 'import json; print(json.load(open("source.lock.json"))["base_image"])')
SOURCE_COMMIT=$(python3 -c 'import json; print(json.load(open("source.lock.json"))["upstream_commit"])')
exec docker build --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "DEEPINFER_COMMIT=$SOURCE_COMMIT" \
  --build-arg "MAX_JOBS=$MAX_JOBS" \
  --build-arg "NVCC_THREADS=$NVCC_THREADS" \
  --target runtime --tag "$IMAGE_NAME" "$@" .
