# Candidate only: immutable image ID must pass native-tool qualification.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash tmux git ripgrep procps libmagic1 build-essential curl sudo \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt
COPY upstream /opt/openhands
# The acquired source deliberately has owner-only permissions. COPY changes its
# owner to root; expose this public upstream code to the nonroot runtime user.
RUN chmod -R a+rX /opt/openhands
ENV PYTHONPATH=/opt/openhands \
    PYTHONDONTWRITEBYTECODE=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    LOG_LEVEL=ERROR \
    HOME=/tmp/agent-home
RUN groupadd --gid 1000 agent && useradd --uid 1000 --gid 1000 --no-create-home agent \
    && mkdir -p /testbed /workspace && chown agent:agent /testbed /workspace
USER 1000:1000
WORKDIR /testbed
# Verify the real runtime modules are importable as the final nonroot user.
RUN python -c "import os; from pathlib import Path; Path(os.environ['HOME']).mkdir(parents=True, exist_ok=True); from openhands.runtime.action_execution_server import ActionExecutor; from openhands.agenthub.codeact_agent.function_calling import response_to_actions"
# No automatic server, network call, tool execution or model startup.
CMD ["python", "-c", "print('OpenHands source-native candidate; execution qualification required')"]
