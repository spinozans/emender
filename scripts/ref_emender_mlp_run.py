#!/usr/bin/env python3
"""Launch the single-GPU emender-mlp reference run from docs/SCALE_PLAN.md.

This script is intentionally small orchestration around train.py. It constructs
the base command through scripts.cmaes_search_v2.build_train_command, then
overrides only the long-run recipe knobs required by SCALE_PLAN:
schedule-free constant LR, no warmup/cosine, Pile + p50k_base + ctx2048,
bf16 fused E97 Triton, and y-mode held-out BPB curve logging.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cmaes_search_v2 as C  # noqa: E402

DATA = "/home/erikg/elman/data/pile.txt"
OUT_ROOT = Path("/mnt/nvme1n1/erikg/ref_emender_mlp")
HELDOUT = OUT_ROOT / "heldout_pile_tail_p50k_2048_1m.pt"
TOK = "p50k_base"
CHUNK = 2048
N_CHUNKS = 512
TAIL_FRACTION = 0.90
HELDOUT_SEED = 7777
TARGET_TOKENS = 2_000_000_000
LR = 1.007e-3
MLP_RATIO = 2.262336203876648


def build_heldout_tensor(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        scored = int(payload.get("scored_tokens", payload["chunks"].shape[0] * CHUNK))
        if scored >= 1_000_000:
            print(f"[heldout] reusing {path} scored_tokens={scored}", flush=True)
            return
        raise RuntimeError(f"Existing held-out tensor is too small: {path} scored_tokens={scored}")

    enc = tiktoken.get_encoding(TOK)
    with open(DATA, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
        tail_start = int(size * TAIL_FRACTION)
        need_bytes = (CHUNK + 1) * 8
        max_start = size - need_bytes - 1
        rng = np.random.RandomState(HELDOUT_SEED)
        chunks = []
        attempts = 0
        while len(chunks) < N_CHUNKS:
            attempts += 1
            if attempts > N_CHUNKS * 100:
                raise RuntimeError("too many failed held-out extraction attempts")
            pos = rng.randint(tail_start, max_start)
            raw = bytes(mm[pos : pos + need_bytes])
            text = raw.decode("utf-8", errors="replace")
            toks = enc.encode(text, disallowed_special=())
            if len(toks) < CHUNK + 2:
                continue
            toks = toks[1 : CHUNK + 2]
            if len(toks) >= CHUNK + 1:
                chunks.append(toks[: CHUNK + 1])

    tensor = torch.tensor(chunks, dtype=torch.long)
    scored = tensor[:, 1:]
    total_bytes = sum(len(enc.decode(row).encode("utf-8")) for row in scored.tolist())
    total_scored = int(scored.numel())
    payload = {
        "chunks": tensor,
        "chunk_size": CHUNK,
        "n_chunks": N_CHUNKS,
        "tokenizer": TOK,
        "seed": HELDOUT_SEED,
        "tail_fraction": TAIL_FRACTION,
        "scored_tokens": total_scored,
        "total_utf8_bytes": total_bytes,
        "bytes_per_token": total_bytes / total_scored,
        "data": DATA,
        "file_size": size,
        "tail_start_byte": tail_start,
    }
    torch.save(payload, path)
    print(
        "[heldout] wrote "
        + json.dumps({k: v for k, v in payload.items() if k != "chunks"}, sort_keys=True),
        flush=True,
    )


def strip_arg(cmd: list[str], name: str) -> list[str]:
    out = []
    skip = False
    for item in cmd:
        if skip:
            skip = False
            continue
        if item == name:
            skip = True
            continue
        out.append(item)
    return out


def build_command(args: argparse.Namespace) -> tuple[list[str], int]:
    C.DATA_PATH = DATA
    C.CHUNK_SIZE = CHUNK
    C.TOKENIZER_NAME = TOK
    C.USE_TRITON_E88 = True

    params = {
        "dim": 1792,
        "n_heads": 216,
        "n_state": 32,
        "depth": 11,
        "batch_size": 4,
        "lr": LR,
    }
    cmd, est_params = C.build_train_command(params, "e97", 1.0, str(OUT_ROOT / "runs"))
    cmd = strip_arg(list(cmd), "--train_minutes")
    cmd = strip_arg(cmd, "--save_every")
    cmd = strip_arg(cmd, "--keep_checkpoints")
    cmd = strip_arg(cmd, "--lr")
    cmd = strip_arg(cmd, "--optimizer")

    tokens_per_step = params["batch_size"] * (CHUNK + 1)
    steps = int(math.ceil(args.target_tokens / tokens_per_step))
    curve_every = args.curve_every
    save_every = args.save_every
    curve_path = OUT_ROOT / "heldout_curve.csv"

    cmd += [
        "--lr",
        f"{LR:.10g}",
        "--optimizer",
        "schedulefree",
        "--steps",
        str(steps),
        "--save_every",
        str(save_every),
        "--keep_checkpoints",
        str(args.keep_checkpoints),
        "--log_every",
        str(args.log_every),
        "--heldout_tensor",
        str(HELDOUT),
        "--heldout_eval_mode",
        "y",
        "--heldout_curve_every",
        str(curve_every),
        "--heldout_curve_path",
        str(curve_path),
        "--final_heldout_eval",
        "--mlp_ratio",
        str(MLP_RATIO),
        "--mlp_multiple",
        "64",
    ]
    if args.extra:
        cmd += args.extra
    return cmd, est_params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--curve-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25000)
    parser.add_argument("--keep-checkpoints", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("CUDA_VISIBLE_DEVICES is unset; run via scripts/ref_emender_mlp_run.sh")
    visible = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    if len(visible) != 1:
        raise SystemExit(f"Expected exactly one leased GPU, got CUDA_VISIBLE_DEVICES={visible!r}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_heldout_tensor(HELDOUT)
    cmd, est_params = build_command(args)
    manifest = {
        "command": cmd,
        "estimated_params": est_params,
        "target_tokens": args.target_tokens,
        "tokens_per_step": 4 * (CHUNK + 1),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "recipe": {
            "optimizer": "schedulefree",
            "lr": LR,
            "warmup_steps": 0,
            "cosine": False,
            "data": DATA,
            "tokenizer": TOK,
            "chunk_size": CHUNK,
            "heldout_eval_mode": "y",
            "fused_triton": True,
            "single_gpu": True,
        },
    }
    (OUT_ROOT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[launch] " + " ".join(cmd), flush=True)
    print(f"[launch] estimated_params={est_params}", flush=True)
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HELDOUT_EVAL_BS", "8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(OUT_ROOT / "train.log", "a", buffering=1) as log:
        log.write("[launch] " + " ".join(cmd) + "\n")
        return subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    raise SystemExit(main())
