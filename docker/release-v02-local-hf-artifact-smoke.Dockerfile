# syntax=docker/dockerfile:1

ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache \
    HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
    TRANSFORMERS_CACHE=/hf-cache/transformers \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /opt/ndm

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY ndm ./ndm

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        -e ".[eval]" \
        flash-linear-attention==0.4.1 \
        transformers==4.57.3 \
        huggingface-hub==0.36.0 \
    && python - <<'PY'
import tiktoken
tiktoken.get_encoding("p50k_base")
PY

COPY scripts ./scripts

ENTRYPOINT ["python", "scripts/smoke_local_hf_artifact_generation.py"]
