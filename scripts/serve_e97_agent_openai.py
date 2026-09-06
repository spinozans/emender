#!/usr/bin/env python3
"""Serve dense recurrent E97 through bounded OpenAI Chat Completions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from ndm.e97 import load_e97_checkpoint
from ndm.e97_acquisition_controller import READ_OBSERVE_TOOLS
from ndm.e97_agent_protocol import DENSE_AGENT_CLI_DIRECT_SYSTEM, DENSE_AGENT_CLI_SYSTEM, DENSE_AGENT_V1_SYSTEM, DENSE_AGENT_V2_SYSTEM, E97_PI_AGENT_SYSTEM_V2, E97_PI_CORE_SYSTEM
from ndm.e97_agent_server import (
    AgentCompletionService,
    TorchE97AgentEngine,
    run_openai_server,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--model-id", default="e97-dense-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--api-key", default=os.environ.get("EMENDER_AGENT_API_KEY"))
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--ingest-mode", choices=("tokenwise", "segment"), default="tokenwise")
    parser.add_argument(
        "--weight-mode",
        choices=("saved", "train"),
        default="saved",
        help="serve saved Schedule-Free x weights or reconstruct train/y weights",
    )
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument("--v1-canonical-system", action="store_true")
    system_group.add_argument("--v2-canonical-system", action="store_true")
    system_group.add_argument("--cli-canonical-system", action="store_true")
    system_group.add_argument("--cli-direct-canonical-system", action="store_true")
    system_group.add_argument("--pi-core-canonical-system", action="store_true")
    system_group.add_argument("--pi-agent-v2-canonical-system", action="store_true")
    parser.add_argument("--trace-generated-errors", action="store_true")
    parser.add_argument(
        "--external-controller",
        action="store_true",
        help="delegate repeat-cycle policy to the trusted dedicated acquisition controller",
    )
    parser.add_argument("--runtime-image", type=Path, help="required immutable sandbox/runtime image artifact for external-controller mode")
    parser.add_argument("--runtime-image-sha256", help="expected SHA-256 for --runtime-image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    system_prompt_override = (
        DENSE_AGENT_V1_SYSTEM if args.v1_canonical_system else
        DENSE_AGENT_V2_SYSTEM if args.v2_canonical_system else
        DENSE_AGENT_CLI_DIRECT_SYSTEM if args.cli_direct_canonical_system else
        DENSE_AGENT_CLI_SYSTEM if args.cli_canonical_system else
        E97_PI_CORE_SYSTEM if args.pi_core_canonical_system else
        E97_PI_AGENT_SYSTEM_V2 if args.pi_agent_v2_canonical_system else None
    )
    if args.external_controller and system_prompt_override is not None:
        parser.error("--external-controller cannot be combined with a server-side canonical system override")
    if args.external_controller and (args.runtime_image is None or not args.runtime_image.is_file()
                                    or not isinstance(args.runtime_image_sha256, str) or len(args.runtime_image_sha256) != 64):
        parser.error("--external-controller requires an existing --runtime-image and --runtime-image-sha256")
    if args.runtime_image is not None and args.runtime_image_sha256 and sha256_file(args.runtime_image) != args.runtime_image_sha256:
        parser.error("--runtime-image SHA-256 does not match --runtime-image-sha256")
    if args.device == "cuda":
        torch.cuda.set_device(0)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    dtype_name = "bfloat16" if args.device == "cuda" else "float32"
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=args.device,
        dtype=dtype,
        weight_mode=args.weight_mode,
        use_triton=args.device == "cuda",
        mmap=True,
    )
    engine = TorchE97AgentEngine(loaded, ingest_mode=args.ingest_mode, weight_mode=args.weight_mode,
                                 device=args.device, dtype=dtype_name, use_triton=args.device == "cuda")
    service = AgentCompletionService(
        engine,
        model_id=args.model_id,
        max_output_tokens=args.max_output_tokens,
        max_sessions=args.max_sessions,
        trace_generated_errors=args.trace_generated_errors,
        system_prompt_override=system_prompt_override,
        require_tool_call=args.v2_canonical_system or args.cli_canonical_system or args.cli_direct_canonical_system,
        external_controller=args.external_controller,
        checkpoint_sha256=sha256_file(loaded.checkpoint_path), checkpoint_path=str(loaded.checkpoint_path),
        args_json_path=str(args.args_json), args_json_sha256=sha256_file(args.args_json), config_sha256=hashlib.sha256(json.dumps(loaded.config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        weight_mode=args.weight_mode, tokenizer=str(loaded.config["tokenizer"]),
        device=args.device, dtype=dtype_name,
        use_triton=args.device == "cuda", ingest_mode=args.ingest_mode,
        runtime_image_path=str(args.runtime_image or args.checkpoint), runtime_image_sha256=args.runtime_image_sha256 or sha256_file(args.checkpoint),
        tool_schema_sha256=hashlib.sha256(json.dumps(READ_OBSERVE_TOOLS, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        controller_build_sha256=sha256_file(Path(__file__).resolve().parents[1] / "ndm" / "e97_acquisition_controller.py"),
    )
    print(
        f"serving model={args.model_id} checkpoint={loaded.checkpoint_path} "
        f"address={args.host}:{args.port} max_sessions={args.max_sessions} "
        f"ingest_mode={args.ingest_mode} external_controller={args.external_controller}",
        flush=True,
    )
    run_openai_server(
        service,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        max_body_bytes=args.max_body_bytes,
    )


if __name__ == "__main__":
    main()
