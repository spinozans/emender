# syntax=docker/dockerfile:1

ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf-disabled \
    TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers

WORKDIR /opt/ndm

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY ndm ./ndm
COPY scripts ./scripts

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e ".[eval]" flash-linear-attention==0.4.1 \
    && python - <<'PY'
import tiktoken
tiktoken.get_encoding("p50k_base")
PY

ENTRYPOINT ["python", "scripts/smoke_local_checkpoint_generation.py"]
