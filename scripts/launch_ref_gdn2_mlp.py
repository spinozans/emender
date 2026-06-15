#!/usr/bin/env python3
"""Launch the single-GPU gdn2-mlp long reference run.

This is intentionally a thin, auditable wrapper around
scripts.cmaes_search_v2.build_train_command.  The CMA builder provides the
train.py geometry flags; this script replaces the short CMA budget with the
constant-LR long-run recipe from docs/SCALE_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cmaes_search_v2 as cmaes


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/mnt/nvme1n1/erikg/ref_gdn2_mlp")
DEFAULT_HELDOUT_SOURCE = Path(
    "/mnt/nvme1n1/erikg/ref_emender_mlp/heldout_pile_tail_p50k_2048_1m.pt"
)

PARAMS = {
    "dim": 2176,
    "expansion": 1,
    "depth": 12,
    "n_heads": 30,
    "mlp_ratio": 3.258732449079677,
    "lr": 4.74e-4,
    "batch_size": 4,
}

DATA = "/home/erikg/elman/data/pile.txt"
TOKENIZER = "p50k_base"
CHUNK_SIZE = 2048
TARGET_TOKENS = 2_000_000_000
TOKENS_PER_STEP = PARAMS["batch_size"] * (CHUNK_SIZE + 1)
TARGET_STEPS = math.ceil(TARGET_TOKENS / TOKENS_PER_STEP)


def replace_arg(cmd: list[str], flag: str, value: str) -> None:
    if flag not in cmd:
        cmd.extend([flag, value])
        return
    idx = cmd.index(flag)
    cmd[idx + 1] = value


def remove_arg_with_value(cmd: list[str], flag: str) -> None:
    while flag in cmd:
        idx = cmd.index(flag)
        del cmd[idx:idx + 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--heldout-source", type=Path, default=DEFAULT_HELDOUT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.out
    runs_dir = out / "runs"
    out.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    heldout = out / "heldout_pile_tail_p50k_2048_1m.pt"
    if not heldout.exists():
        if not args.heldout_source.exists():
            raise FileNotFoundError(f"held-out tensor source is missing: {args.heldout_source}")
        shutil.copy2(args.heldout_source, heldout)

    cmaes.DATA_PATH = DATA
    cmaes.TOKENIZER_NAME = TOKENIZER
    cmaes.PARAM_VOCAB_SIZE = cmaes.resolve_vocab_size(TOKENIZER)
    cmaes.CHUNK_SIZE = CHUNK_SIZE

    cmd, estimated_params = cmaes.build_train_command(
        PARAMS,
        "gdn2-mlp",
        train_minutes=999999,
        output_dir=str(runs_dir),
    )

    remove_arg_with_value(cmd, "--train_minutes")
    replace_arg(cmd, "--lr", f"{PARAMS['lr']:.8g}")
    replace_arg(cmd, "--optimizer", "schedulefree")
    replace_arg(cmd, "--steps", str(TARGET_STEPS))
    replace_arg(cmd, "--save_every", "25000")
    replace_arg(cmd, "--keep_checkpoints", "12")
    replace_arg(cmd, "--log_every", "100")
    cmd.extend([
        "--heldout_tensor", str(heldout),
        "--heldout_eval_mode", "y",
        "--heldout_curve_every", "500",
        "--heldout_curve_path", str(out / "heldout_curve.csv"),
        "--final_heldout_eval",
    ])

    manifest = {
        "command": cmd,
        "estimated_params": estimated_params,
        "target_tokens": TARGET_TOKENS,
        "target_steps": TARGET_STEPS,
        "tokens_per_step": TOKENS_PER_STEP,
        "curve_every_steps": 500,
        "recipe": {
            "arm": "gdn2-mlp",
            "data": DATA,
            "tokenizer": TOKENIZER,
            "chunk_size": CHUNK_SIZE,
            "batch_size": PARAMS["batch_size"],
            "optimizer": "schedulefree",
            "lr": PARAMS["lr"],
            "warmup_steps": 0,
            "cosine": False,
            "decay": False,
            "bf16": True,
            "single_gpu": True,
            "heldout_eval_mode": "y",
            "fused_kernel": "external GDN-2 FLA chunked path",
        },
    }
    (out / "recipe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    env = os.environ.copy()
    env.setdefault("GDN2_PATH", "/home/erikg/GatedDeltaNet-2")
    env.setdefault("HELDOUT_EVAL_BS", "8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    wrapper = REPO_ROOT / "scripts" / "launch_detached_run.sh"
    launch_cmd = [
        str(wrapper),
        "--name",
        "ref-gdn2-mlp",
        "--gpus",
        "1",
        "--logdir",
        str(out),
        "--",
    ] + cmd

    proc = subprocess.Popen(
        launch_cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    pid = proc.stdout.readline().strip()
    if not pid:
        _, stderr = proc.communicate(timeout=10)
        raise RuntimeError(f"detached launcher did not print a PID: {stderr}")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # The detached child may keep a descriptor inherited through the lease
        # heartbeat. Once the PID has been printed, the durable pid/manifest files
        # are authoritative and the training process is independent of this helper.
        pass
    print(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
