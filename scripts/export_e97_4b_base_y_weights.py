#!/usr/bin/env python3
"""Export the schedule-free y/train weights of the E97 4B base pre-training
checkpoint to a standalone .pt for GGUF conversion.

Why this exists: the base pre-training checkpoint (step_024448, 99.72B tokens,
sha 3ace0042...) is a Schedule-Free checkpoint saved with ``train_mode=False``:
its ``model_state_dict`` holds the x/averaged point, and the canonical
generation point — the one ndm.e97.load_e97_checkpoint documents as the
"generation default" and the one the entire SFT lineage (bridge -> v6)
was initialized from — is the y/train point recovered from the saved
optimizer state via the ScheduleFree ``train()`` swap (y = lerp(x -> z,
1 - beta1)).  The existing base GGUF built directly from the saved
model_state_dict (x-form) produces degenerate document NLL (~23.5 nats,
top1 ~0.003) through two independent implementations, so the probe baseline
must be rebuilt from the y-form weights.

CPU-only by construction: the model is loaded on CPU with use_triton=False
and the optimizer-state swap is arithmetic, never CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from ndm.e97 import load_e97_checkpoint

BASE_SHA256 = "3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568"
BETAS = (0.9, 0.95)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-sha256", default=BASE_SHA256)
    args = ap.parse_args()

    digest = sha256_of(args.checkpoint)
    if digest != args.expect_sha256:
        raise SystemExit(f"checkpoint sha mismatch: {digest} != {args.expect_sha256}")

    loaded = load_e97_checkpoint(
        args.checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
        weight_mode="train",
        use_triton=False,
        mmap=True,
    )
    if not loaded.schedulefree_train_weight_swap:
        raise SystemExit("expected the schedule-free train() weight swap to be applied")
    print("schedule-free y/train weight swap applied (weight_mode='train')", flush=True)

    sd = loaded.model.state_dict()
    if len(sd) != 237:
        raise SystemExit(f"expected 237 tensors, got {len(sd)}")
    expected = {
        "embedding.weight": (50281, 3840),
        "lm_head.weight": (50281, 3840),
        "norm.weight": (3840,),
        "layer_norms.0.weight": (3840,),
        "layers.0.mixer.qkv_proj.weight": (11520, 3840),
        "layers.17.mlp.w3.weight": (3840, 9600),
    }
    for key, shape in expected.items():
        got = tuple(sd[key].shape)
        if got != shape:
            raise SystemExit(f"{key}: {got} != {shape}")

    # Independent arithmetic cross-check on the first parameter (embedding):
    # y = (1 - beta1) * z + beta1 * x with the z stored in the optimizer state.
    ck = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    z0 = ck["optimizer_state_dict"]["state"][0]["z"].float()
    x0 = ck["model_state_dict"]["embedding.weight"].float()
    manual_y = (1.0 - BETAS[0]) * z0 + BETAS[0] * x0
    got_y = sd["embedding.weight"].float()
    diff = (manual_y - got_y).abs().max().item()
    scale = manual_y.abs().max().item()
    print(f"embedding y-swap cross-check: max|manual - loaded| = {diff:.6f} "
          f"(weight scale {scale:.3f}; bf16 rounding expected)", flush=True)
    if diff > 1e-2 * max(scale, 1.0):
        raise SystemExit("embedding y-swap cross-check failed")

    payload = {
        "schema": "emender-e97-4b-base-y-weights-export-v1",
        "model_state_dict": sd,
        "weight_form": "schedulefree-train-y",
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": digest,
        "note": (
            "y/train point recovered from the schedule-free optimizer state via "
            "ndm.e97.load_e97_checkpoint(weight_mode='train'); this is the "
            "generation default and the exact parent of the SFT lineage"
        ),
    }
    torch.save(payload, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB, "
          f"sha256 {sha256_of(args.output)})", flush=True)


if __name__ == "__main__":
    main()
