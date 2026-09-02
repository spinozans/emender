#!/usr/bin/env python3
"""Bitwise-compare resumable E97 Pi-SFT state while ignoring log-only loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_EQUAL_METADATA = (
    "schema",
    "sft_updates",
    "sft_total_tokens",
    "assistant_target_tokens",
    "weight_mode",
    "parent_checkpoint",
    "parent_checkpoint_sha256",
    "authority_manifest_sha256",
    "pack_manifest_sha256",
    "sampler_key",
    "sampler_cursor",
    "data_world_size",
    "context_size",
    "island_size",
    "diloco_k",
    "source_commit",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
    "optimizer_state_storage",
    "gradient_checkpoint_group_size",
    "empty_cache_min_record_tokens",
    "mlp_checkpoint_chunk_size",
    "merge_bucket_numel",
    "boundary_aware_packs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(left: Any, right: Any, path: str, counts: dict[str, int]) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise RuntimeError(f"{path}: tensor/non-tensor mismatch")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"{path}: tensor metadata mismatch: "
                f"{tuple(left.shape)}/{left.dtype} != {tuple(right.shape)}/{right.dtype}")
        if not torch.equal(left, right):
            raise RuntimeError(f"{path}: tensor values differ")
        counts["tensors"] += 1
        counts["tensor_elements"] += left.numel()
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise RuntimeError(f"{path}: mapping type mismatch")
        if left.keys() != right.keys():
            raise RuntimeError(f"{path}: mapping keys differ")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}", counts)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"{path}: sequence metadata differs")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            compare(left_value, right_value, f"{path}[{index}]", counts)
        return
    if left != right:
        raise RuntimeError(f"{path}: {left!r} != {right!r}")
    counts["scalars"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    for path in (args.left, args.right):
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")

    left = torch.load(args.left, map_location="cpu", mmap=True, weights_only=False)
    right = torch.load(args.right, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError("checkpoint roots must be mappings")
    for key in REQUIRED_EQUAL_METADATA:
        if key not in left or key not in right:
            raise RuntimeError(f"required metadata missing: {key}")
        if left[key] != right[key]:
            raise RuntimeError(f"metadata differs for {key}: {left[key]!r} != {right[key]!r}")

    counts = {"tensors": 0, "tensor_elements": 0, "scalars": 0}
    compare(left["model_state_dict"], right["model_state_dict"], "model_state_dict", counts)
    compare(
        left["optimizer_state_dict"], right["optimizer_state_dict"],
        "optimizer_state_dict", counts)
    print(json.dumps({
        "schema": "emender-e97-4b-pi-sft-exact-resume-parity-v1",
        "result": "bitwise-equal",
        "left": str(args.left.resolve()),
        "left_sha256": sha256(args.left),
        "right": str(args.right.resolve()),
        "right_sha256": sha256(args.right),
        **counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
