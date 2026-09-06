#!/usr/bin/env python3
"""Restore one E97 state and validate isolated token-delta branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import tiktoken
import torch

from ndm.e97 import advance_e97_cache, generate_e97_from_cache, load_e97_checkpoint
from ndm.e97_state import clone_e97_cache, load_e97_state, save_e97_state


def equal_tree(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    return left is None and right is None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--system-prompt", default=(
        "You are a concise helpful assistant. Respond with exactly Final: followed by your answer."
    ))
    parser.add_argument("--weight-mode", choices=("saved", "train"), default="saved")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.set_device(0)
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        weight_mode=args.weight_mode,
        use_triton=args.device == "cuda",
        mmap=True,
    )
    tokenizer_name = str(loaded.tokenizer_name)
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    prompt_sha = hashlib.sha256(args.system_prompt.encode()).hexdigest()
    restored = load_e97_state(
        args.state,
        loaded=loaded,
        model_checkpoint_sha256=args.checkpoint_sha256,
        tokenizer=tokenizer_name,
        system_prompt_sha256=prompt_sha,
    )
    original = restored.cache
    snapshot = clone_e97_cache(original)
    branch_a = clone_e97_cache(original)
    branch_b = clone_e97_cache(original)

    delta_a = tokenizer.encode(
        "\n\nUser:\nFocus only on ORCHID-731.\n\nAssistant:\n",
        disallowed_special=(),
    )
    delta_b = tokenizer.encode(
        "\n\nUser:\nFocus only on BASALT-204.\n\nAssistant:\n",
        disallowed_special=(),
    )
    started = time.monotonic()
    branch_a = advance_e97_cache(loaded, delta_a, branch_a)
    branch_a_tokens, branch_a_completed = generate_e97_from_cache(
        loaded, branch_a, max_new_tokens=args.max_new_tokens,
        temperature=0, top_k=0, top_p=1.0,
    )
    branch_b = advance_e97_cache(loaded, delta_b, branch_b)
    branch_b_tokens, branch_b_completed = generate_e97_from_cache(
        loaded, branch_b, max_new_tokens=args.max_new_tokens,
        temperature=0, top_k=0, top_p=1.0,
    )
    branch_seconds = time.monotonic() - started

    unchanged = (
        original.total_token_count == snapshot.total_token_count
        and original.token_lineage_sha256 == snapshot.token_lineage_sha256
        and equal_tree(original.hidden, snapshot.hidden)
        and torch.equal(original.next_logits, snapshot.next_logits)
    )
    branches_distinct = (
        branch_a_completed.token_lineage_sha256 != branch_b_completed.token_lineage_sha256
        and not equal_tree(branch_a_completed.hidden, branch_b_completed.hidden)
    )

    expected = json.loads(args.expected.read_text())
    reference_tokens = expected["generated_token_ids"][:args.max_new_tokens]
    original_tokens, _ = generate_e97_from_cache(
        loaded, original, max_new_tokens=args.max_new_tokens,
        temperature=0, top_k=0, top_p=1.0,
    )
    original_parity = original_tokens == reference_tokens
    if not unchanged or not branches_distinct or not original_parity:
        raise RuntimeError("portable E97 branch-isolation probe failed")

    branch_a_path = args.state.with_name(args.state.stem + "-branch-a.e97state")
    branch_b_path = args.state.with_name(args.state.stem + "-branch-b.e97state")
    receipt_a = save_e97_state(
        branch_a_path, branch_a_completed,
        model_checkpoint_sha256=args.checkpoint_sha256,
        tokenizer=tokenizer_name,
        system_prompt_sha256=prompt_sha,
        metadata={"parent_artifact_sha256": restored.artifact_sha256, "branch": "a"},
    )
    receipt_b = save_e97_state(
        branch_b_path, branch_b_completed,
        model_checkpoint_sha256=args.checkpoint_sha256,
        tokenizer=tokenizer_name,
        system_prompt_sha256=prompt_sha,
        metadata={"parent_artifact_sha256": restored.artifact_sha256, "branch": "b"},
    )
    receipt = {
        "schema": "emender-e97-portable-state-fork-probe-v1",
        "parent_artifact_sha256": restored.artifact_sha256,
        "source_token_count": original.total_token_count,
        "replayed_tokens": 0,
        "branch_a_delta_tokens": len(delta_a),
        "branch_b_delta_tokens": len(delta_b),
        "branch_a_generated_token_ids": branch_a_tokens,
        "branch_b_generated_token_ids": branch_b_tokens,
        "branch_a_text": tokenizer.decode(branch_a_tokens),
        "branch_b_text": tokenizer.decode(branch_b_tokens),
        "branch_a_artifact": receipt_a,
        "branch_b_artifact": receipt_b,
        "original_state_unchanged": unchanged,
        "branches_distinct": branches_distinct,
        "original_greedy_parity_after_branches": original_parity,
        "branch_seconds": branch_seconds,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
