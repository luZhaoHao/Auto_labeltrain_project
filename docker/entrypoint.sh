#!/usr/bin/env bash
# Auto-Tune Studio container entrypoint (F1.2-C).
#
# The delivery rules — persistent directories, mount contract, write
# permission, configuration bootstrap, runtime components and the GPU the
# formal delivery requires — live in auto_tune/delivery/preflight.py so the
# Windows delivery can enforce the same rules. This script only selects the
# container subset and hands over to the Studio Web UI.
#
# It never prints configuration contents or credentials.

set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/auto-tune}"

export AUTO_TUNE_APP_ROOT="${AUTO_TUNE_APP_ROOT:-${APP_ROOT}}"
export AUTO_TUNE_CONFIG_PATH="${AUTO_TUNE_CONFIG_PATH:-/data/config/config.yaml}"
export AUTO_TUNE_DATASETS_DIR="${AUTO_TUNE_DATASETS_DIR:-/data/datasets}"
export AUTO_TUNE_HOST="${AUTO_TUNE_HOST:-0.0.0.0}"
export AUTO_TUNE_PORT="${AUTO_TUNE_PORT:-8000}"

cd "${APP_ROOT}"

# --require-mounts: an unmounted directory would silently absorb the operator's
#   configuration and history into the container's writable layer.
# --require-gpu: the formal delivery requests the GPU and never falls back to
#   the CPU, so a missing device stops the start instead of degrading quietly.
# --bootstrap-config: the sanitized template is copied only when no
#   configuration exists; an operator's file is never overwritten.
python -m auto_tune.delivery.preflight \
    --require-mounts --require-gpu --bootstrap-config

echo "[entrypoint] starting Auto-Tune Studio on ${AUTO_TUNE_HOST}:${AUTO_TUNE_PORT}"

exec python -m auto_tune.main
