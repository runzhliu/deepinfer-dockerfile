#!/usr/bin/env bash
set -euo pipefail
case "${1:-doctor}" in
  doctor)
    if (( $# )); then shift; fi
    exec python /opt/container-tools/doctor.py "$@"
    ;;
  offline)
    shift
    cd /opt/deepinfer
    exec python examples/offline_inference.py "$@"
    ;;
  benchmark)
    shift
    cd /opt/deepinfer
    exec python examples/benchmark.py "$@"
    ;;
  *) exec "$@" ;;
esac
