#!/usr/bin/env python3
"""CPU-only Pi client gate for eight private-analysis tool round trips.

This uses a deterministic scripted engine, not useful model inference. It proves
that the installed Pi OpenAI-completions client preserves reasoning_content
through streaming, tool dispatch, transcript replay, and the E97 server grammar.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from http.server import HTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any

from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1
from ndm.e97_agent_server import AgentCompletionService, make_openai_handler
from ndm.e97_atomic import publish_bytes_no_replace

SCHEMA = "emender-e97-pi-private-analysis-roundtrip-probe-v1"


@dataclass(frozen=True)
class Cache:
    token_ids: tuple[int, ...]

    @property
    def state_bytes(self) -> int:
        return len(self.token_ids)


class ScriptedEngine:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids) -> str:
        return bytes(token_ids).decode("utf-8")

    def advance(self, token_ids, cache=None) -> Cache:
        prefix = () if cache is None else cache.token_ids
        return Cache(prefix + tuple(token_ids))

    def generate(self, cache, *, max_new_tokens: int, temperature: float, top_p: float):
        if not self.outputs:
            raise RuntimeError("Pi requested more than nine scripted turns")
        tokens = tuple(self.encode(self.outputs.pop(0)))
        if len(tokens) > max_new_tokens:
            raise RuntimeError("scripted turn exceeds service output bound")
        return list(tokens), Cache(cache.token_ids + tokens)


class RecordingService(AgentCompletionService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[dict[str, Any]] = []

    def prepare_completion(self, request, *, session_id):
        self.requests.append(json.loads(json.dumps(request, ensure_ascii=False)))
        return super().prepare_completion(request, session_id=session_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", default=shutil.which("pi"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not args.pi or not Path(args.pi).is_file():
        raise SystemExit("installed pi executable is required")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be positive")

    analyses = [
        f"synthetic rationale {index}: preserve λ, newline\\n, Action:, Arguments:, and Final:"
        for index in range(8)
    ]
    outputs = [
        "Analysis: " + json.dumps(reasoning, ensure_ascii=False, separators=(",", ":"))
        + f'\nAction: read\nArguments: {{"path":"file-{index}.txt","offset":1,"limit":1}}'
        for index, reasoning in enumerate(analyses)
    ]
    terminal_reasoning = "synthetic terminal rationale"
    outputs.append(
        "Analysis: " + json.dumps(terminal_reasoning, separators=(",", ":"))
        + "\nFinal: eight private-analysis turns preserved\n"
    )
    engine = ScriptedEngine(outputs)
    service = RecordingService(
        engine, model_id="e97-dense-agent-analysis-v1", max_output_tokens=512,
        max_sessions=2, private_analysis=True,
    )
    handler = make_openai_handler(service, api_key="local")
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="e97-pi-analysis-probe-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            config = root / "pi"
            workspace.mkdir()
            config.mkdir()
            for index in range(8):
                (workspace / f"file-{index}.txt").write_text(f"observation-{index}\n")
            models = {
                "providers": {
                    "emender-local-analysis": {
                        "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
                        "api": "openai-completions",
                        "apiKey": "local",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                            "supportsUsageInStreaming": False,
                            "supportsFinishReason": True,
                            "supportsStrictMode": False,
                            "sendSessionAffinityHeaders": True,
                            "sessionAffinityFormat": "openrouter",
                        },
                        "models": [{
                            "id": "e97-dense-agent-analysis-v1",
                            "name": "E97 scripted private-analysis probe",
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": 65_536,
                            "maxTokens": 512,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        }],
                    }
                }
            }
            (config / "models.json").write_text(json.dumps(models, indent=2) + "\n")
            environment = dict(os.environ)
            environment.update({
                "PI_CODING_AGENT_DIR": str(config),
                "PI_OFFLINE": "1",
                "NO_COLOR": "1",
            })
            command = [
                str(args.pi), "--mode", "json",
                "--provider", "emender-local-analysis",
                "--model", "e97-dense-agent-analysis-v1",
                "--no-session", "--no-skills", "--no-context-files",
                "--system-prompt", E97_PI_AGENT_ANALYSIS_SYSTEM_V1,
                "--approve",
                "Read file-0.txt through file-7.txt in order, then report completion.",
            ]
            completed = subprocess.run(
                command, cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=args.timeout_seconds, check=False,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    errors = []
    if completed.returncode != 0:
        errors.append(f"pi_exit:{completed.returncode}")
    if len(service.requests) != 9:
        errors.append(f"request_count:{len(service.requests)}")
    for request_index, request in enumerate(service.requests):
        prior_assistants = [
            message for message in request.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        expected = analyses[:request_index]
        actual = [message.get("reasoning_content") for message in prior_assistants]
        if actual != expected:
            errors.append(f"reasoning_roundtrip:{request_index}")
        if any("reasoning_content" not in message for message in prior_assistants):
            errors.append(f"reasoning_dropped:{request_index}")
    if engine.outputs:
        errors.append(f"unused_scripted_outputs:{len(engine.outputs)}")
    final_present = "eight private-analysis turns preserved" in completed.stdout
    if not final_present:
        errors.append("missing_final")

    result = {
        "schema": SCHEMA,
        "status": "passed" if not errors else "failed",
        "training_eligible": False,
        "claim": "CPU-only Pi/OpenAI protocol round trip; not useful model inference",
        "pi_executable": str(Path(args.pi).resolve()),
        "pi_sha256": hashlib.sha256(Path(args.pi).read_bytes()).hexdigest(),
        "request_count": len(service.requests),
        "tool_turns": 8,
        "reasoning_sha256s": [hashlib.sha256(item.encode()).hexdigest() for item in analyses],
        "terminal_reasoning_sha256": hashlib.sha256(terminal_reasoning.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "returncode": completed.returncode,
        "errors": errors,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if args.output is not None:
        publish_bytes_no_replace(args.output, payload, mode=0o600)
    print(payload.decode(), end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
