#!/usr/bin/env bash
# Auto-Tune Studio container entrypoint (F1.2-C).
#
# The container starts as root for exactly one reason: a bind-mount host
# directory that does not exist yet is created by the container engine owned by
# root, while the Studio itself runs as the unprivileged studio account. That one
# privileged step — the declared directories only, the Studio after dropping to
# 10001:10001, the shared delivery preflight after the drop, and finally an
# `exec` of the Studio so PID 1 is the Python process receiving SIGTERM — lives
# in auto_tune/delivery/container_entrypoint.py. The delivery rules themselves
# (persistent directories, mount contract, write permission, configuration
# bootstrap, runtime components and the GPU) stay in
# auto_tune/delivery/preflight.py so the Windows delivery enforces the same ones.
#
# This script only fixes the documented environment defaults and hands over. It
# never builds a command string and never prints configuration or credentials,
# and every path it exports is re-checked against the declared container layout
# before the module touches anything: an override that points outside it stops
# the start instead of being honoured.

set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/auto-tune}"

export AUTO_TUNE_APP_ROOT="${AUTO_TUNE_APP_ROOT:-${APP_ROOT}}"
export AUTO_TUNE_CONFIG_PATH="${AUTO_TUNE_CONFIG_PATH:-/data/config/config.yaml}"
export AUTO_TUNE_DATASETS_DIR="${AUTO_TUNE_DATASETS_DIR:-/data/datasets}"
export AUTO_TUNE_HOST="${AUTO_TUNE_HOST:-0.0.0.0}"
export AUTO_TUNE_PORT="${AUTO_TUNE_PORT:-8000}"
# API keys saved from the settings page are persisted into the mounted secrets
# directory, so they survive a restart, a recreation and an image update. The
# file is created by the first save; nothing here reads or prints it.
export AUTO_TUNE_CREDENTIALS_PATH="${AUTO_TUNE_CREDENTIALS_PATH:-/data/secrets/credentials.json}"
# The directories the folder pickers may browse. The container has no drives and
# the configuration ships no allowed root, so without this a bare `docker run`
# would offer an empty picker. Both are mounted directories: the dataset share
# the product only reads, and /opt/auto-tune/detect where every training run is
# written. The product accepts exactly these two and nothing else, and the
# start-up refuses a value that is not exactly this pair.
export AUTO_TUNE_INPUT_ALLOWED_ROOTS="${AUTO_TUNE_INPUT_ALLOWED_ROOTS:-/data/datasets;/opt/auto-tune/detect}"

# The base image leaves HOME=/root behind, but the Studio runs as the
# unprivileged studio account. Ultralytics resolves its settings directory
# through YOLO_CONFIG_DIR and only then through $HOME/.config, so a training run
# started after the drop failed on the first start with
# "PermissionError: /root/.config/Ultralytics". Both values are fixed here, and
# they are *assigned* rather than defaulted: a preset HOME or YOLO_CONFIG_DIR
# must not decide where a business process writes. The configuration directory
# is derived from the configuration path, which the start-up already refuses
# unless it is one of the declared container paths; a HOME or a config path that
# leaves that layout is not honoured anywhere downstream.
export AUTO_TUNE_CONFIG_DIR="${AUTO_TUNE_CONFIG_PATH%/*}"
export HOME="/home/studio"
export YOLO_CONFIG_DIR="${AUTO_TUNE_CONFIG_DIR}"

cd "${APP_ROOT}"

exec python -m auto_tune.delivery.container_entrypoint
