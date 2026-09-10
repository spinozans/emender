# Candidate only: immutable image ID must pass native-tool qualification.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash tmux git ripgrep procps libmagic1 build-essential curl sudo \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt
COPY upstream /opt/openhands
ENV PYTHONPATH=/opt/openhands \
    PYTHONDONTWRITEBYTECODE=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    LOG_LEVEL=ERROR \
    HOME=/tmp/agent-home
RUN groupadd --gid 1000 agent && useradd --uid 1000 --gid 1000 --no-create-home agent \
    && mkdir -p /testbed /workspace && chown agent:agent /testbed /workspace
USER 1000:1000
WORKDIR /testbed
# No automatic server, network call, tool execution or model startup.
CMD ["python", "-c", "print('OpenHands source-native candidate; execution qualification required')"]
