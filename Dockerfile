# Auto-Tune Studio — single Linux container (F1.2-B feasibility baseline).
#
# Runs the complete Studio Web UI (FastAPI + Jinja2 SPA) in one image and one
# container. No Engine mode, no external job API, no second image, no
# Web/Worker split. Persistent application directories are bind-mounted from
# the host by compose.yaml and are created by docker/entrypoint.sh.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTO_TUNE_HOST=0.0.0.0 \
    AUTO_TUNE_PORT=8000 \
    AUTO_TUNE_CONFIG_PATH=/data/config/config.yaml

# The runtime CUDA image ships no Python; these are the minimum Linux
# libraries OpenCV and Ultralytics need. curl serves the health probe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# The Studio runs as an unprivileged user; its writable directories are the
# mounted ones, created and owned here so a bare container still starts.
RUN useradd --create-home --uid 10001 --shell /bin/bash studio \
    && mkdir -p /opt/auto-tune/log /opt/auto-tune/detect /opt/auto-tune/runs \
        /opt/auto-tune/models/weights /data/config /data/datasets \
    && chown -R studio:studio /opt/auto-tune /data

WORKDIR /opt/auto-tune

# CUDA 12.1 wheels first. docker/requirements-runtime.txt deliberately omits
# torch/torchvision so pip cannot resolve a CPU build over these.
COPY --chown=studio:studio docker/requirements-runtime.txt ./docker/requirements-runtime.txt
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 \
    && python -m pip install --no-cache-dir \
        -r docker/requirements-runtime.txt

# Only lightweight source and the sanitized template are copied. Local
# artifacts are excluded by .dockerignore.
COPY --chown=studio:studio auto_tune ./auto_tune
COPY --chown=studio:studio docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

# Operational probe only: no datasets, models, credentials or training state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

USER studio

ENTRYPOINT ["/opt/auto-tune/docker/entrypoint.sh"]
