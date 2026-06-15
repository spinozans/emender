#!/usr/bin/env python3
"""Score an emender-mlp checkpoint on the fixed Pile p50k held-out tensor.

This is intentionally narrow: catchup-parallel-diloco only needs the
SCALE_PLAN §1 emender-mlp geometry, so the evaluator avoids guessing model
arguments from checkpoint metadata that train.py does not currently store.
"""
import argparse
import csv
import math
import re
from pathlib import Path

import torch

from ndm.models import LadderLM


def step_from_checkpoint(path: Path, checkpoint: dict) -> int:
    if "step" in checkpoint:
        return int(checkpoint["step"])
    match = re.search(r"checkpoint_step_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer step from {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    heldout = torch.load(args.heldout, map_location="cpu")
    chunks = heldout["chunks"]
    bytes_per_token = float(heldout["bytes_per_token"])

    device = torch.device("cuda")
    model = LadderLM(
        vocab_size=50281,
        dim=1792,
        depth=11,
        level="E97",
        expansion=1.0,
        n_state=32,
        n_heads=216,
        use_gate=True,
        gate_activation="silu",
        e88_raw_write=False,
        use_triton=True,
        mlp_ratio=2.262336203876648,
        mlp_multiple=64,
    ).to(device).bfloat16()

    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for offset in range(0, chunks.shape[0], args.batch_size):
            batch = chunks[offset : offset + args.batch_size].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(batch, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
            scored = batch.shape[0] * (batch.shape[1] - 1)
            total_nll += float(loss.item()) * scored
            total_tokens += scored

    ce = total_nll / max(total_tokens, 1)
    bpb = (ce / math.log(2.0)) / bytes_per_token
    if not math.isfinite(bpb):
        raise RuntimeError("non-finite held-out BPB")

    step = step_from_checkpoint(ckpt_path, checkpoint)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with out_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["tag", "step", "tokens", "heldout_bpb", "heldout_ce", "scored_tokens", "bytes_per_token", "checkpoint"])
        writer.writerow([
            args.tag,
            step,
            step * 5 * 4 * 2048,
            f"{bpb:.6f}",
            f"{ce:.6f}",
            total_tokens,
            f"{bytes_per_token:.6f}",
            str(ckpt_path),
        ])
    print(f"BPB_RESULT {args.tag} step={step} bpb={bpb:.6f} ce={ce:.6f} tokens={total_tokens}", flush=True)


if __name__ == "__main__":
    main()
