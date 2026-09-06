#!/usr/bin/env python3
"""Create or restore an E97 state artifact without transcript replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import tiktoken

from ndm.e97 import advance_e97_cache, generate_e97_from_cache, load_e97_checkpoint
from ndm.e97_agent_protocol import serialize_pi_messages
from ndm.e97_state import load_e97_state, save_e97_state


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "restore"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--system-prompt", default=(
        "You are a concise helpful assistant. Respond with exactly Final: followed by your answer."
    ))
    parser.add_argument("--user", default=(
        "Remember the opaque identifiers ORCHID-731 and BASALT-204. Acknowledge briefly."
    ))
    parser.add_argument("--weight-mode", choices=("saved", "train"), default="saved")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.set_device(0)
    started = time.monotonic()
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        weight_mode=args.weight_mode,
        use_triton=args.device == "cuda",
        mmap=True,
    )
    load_seconds = time.monotonic() - started
    tokenizer = str(loaded.tokenizer_name)
    prompt_sha = sha256_text(args.system_prompt)

    if args.mode == "create":
        prompt = serialize_pi_messages([
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.user},
        ])
        token_ids = tiktoken.get_encoding(tokenizer).encode(
            prompt, disallowed_special=()
        )
        ingest_started = time.monotonic()
        cache = advance_e97_cache(loaded, token_ids)
        ingest_seconds = time.monotonic() - ingest_started
        state_receipt = save_e97_state(
            args.state,
            cache,
            model_checkpoint_sha256=args.checkpoint_sha256,
            tokenizer=tokenizer,
            system_prompt_sha256=prompt_sha,
            metadata={"probe": "cross-process-greedy-parity-v1"},
        )
        replayed_tokens = len(token_ids)
        restore_seconds = None
    else:
        restore_started = time.monotonic()
        restored = load_e97_state(
            args.state,
            loaded=loaded,
            model_checkpoint_sha256=args.checkpoint_sha256,
            tokenizer=tokenizer,
            system_prompt_sha256=prompt_sha,
        )
        restore_seconds = time.monotonic() - restore_started
        cache = restored.cache
        state_receipt = {
            "artifact_sha256": restored.artifact_sha256,
            "artifact_bytes": restored.artifact_bytes,
            "state_bytes": cache.state_bytes,
            "token_count": cache.total_token_count,
        }
        ingest_seconds = 0.0
        replayed_tokens = 0

    generation_started = time.monotonic()
    generated, completed = generate_e97_from_cache(
        loaded,
        cache,
        max_new_tokens=args.max_new_tokens,
        temperature=0,
        top_k=0,
        top_p=1.0,
    )
    generation_seconds = time.monotonic() - generation_started
    decoded = tiktoken.get_encoding(tokenizer).decode(generated)

    if args.mode == "create":
        expected = {
            "generated_token_ids": generated,
            "generated_text": decoded,
            "source_token_count": cache.total_token_count,
            "source_token_lineage_sha256": cache.token_lineage_sha256,
        }
        atomic_json(args.expected, expected)
        parity = True
    else:
        expected = json.loads(args.expected.read_text())
        parity = (
            generated == expected["generated_token_ids"]
            and decoded == expected["generated_text"]
            and cache.total_token_count == expected["source_token_count"]
            and cache.token_lineage_sha256 == expected["source_token_lineage_sha256"]
        )
        if not parity:
            raise RuntimeError("restored state failed greedy generation parity")

    receipt = {
        "schema": "emender-e97-portable-state-probe-v1",
        "mode": args.mode,
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_sha256": args.checkpoint_sha256,
        "weight_mode": args.weight_mode,
        "tokenizer": tokenizer,
        "system_prompt_sha256": prompt_sha,
        "state": str(args.state),
        "state_artifact_sha256": state_receipt["artifact_sha256"],
        "state_artifact_bytes": state_receipt["artifact_bytes"],
        "recurrent_tensor_bytes": state_receipt["state_bytes"],
        "source_token_count": cache.total_token_count,
        "replayed_tokens": replayed_tokens,
        "load_seconds": load_seconds,
        "ingest_seconds": ingest_seconds,
        "restore_seconds": restore_seconds,
        "generation_seconds": generation_seconds,
        "generated_token_ids": generated,
        "generated_text": decoded,
        "completed_token_count": completed.total_token_count,
        "greedy_parity": parity,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
