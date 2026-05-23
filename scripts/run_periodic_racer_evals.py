#!/usr/bin/env python3
"""Periodically freeze racer checkpoints and run continuation evals.

This is a limited tracking harness for live convergence runs.  Each snapshot
hardlinks the newest named checkpoint for every configured racer into a stable
directory, copies the adjacent ``args.json``, and invokes
``scripts/racer_eval_suite.py`` on the frozen paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RACERS = {
    "E88": "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832",
    "FLA-GDN": "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832",
    "Mamba2": "/tmp/pile_convergence_3arch/ctx2k/mamba2_resume_ckpt/levelmamba2_1270M_20260511_233832",
    "M2RNN": "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023",
}


@dataclass(frozen=True)
class Racer:
    label: str
    ckpt_dir: Path


@dataclass(frozen=True)
class Ckpt:
    path: Path
    step: int
    loss: float | None


def parse_step_loss(path: Path) -> tuple[int, float | None]:
    m = re.search(r"checkpoint_step_(\d+)_loss_([-+0-9.eEnaifINF]+)\.pt$", path.name)
    if not m:
        raise ValueError(f"unexpected checkpoint name: {path}")
    step = int(m.group(1))
    try:
        loss = float(m.group(2))
    except ValueError:
        loss = None
    return step, loss


def newest_checkpoint(ckpt_dir: Path) -> Ckpt:
    paths = list(ckpt_dir.glob("checkpoint_step_*_loss_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint_step_*.pt files in {ckpt_dir}")
    best = max(paths, key=lambda p: parse_step_loss(p)[0])
    step, loss = parse_step_loss(best)
    return Ckpt(best.resolve(), step, loss)


def parse_racers(values: list[str]) -> list[Racer]:
    if not values:
        return [Racer(label, Path(path)) for label, path in DEFAULT_RACERS.items()]
    racers = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--racer expects LABEL=/checkpoint/dir, got {value!r}")
        label, path = value.split("=", 1)
        racers.append(Racer(label, Path(path)))
    return racers


def freeze_snapshot(racers: list[Racer], out_root: Path, snapshot_name: str) -> dict:
    snapshot_dir = out_root / snapshot_name
    models_dir = snapshot_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    items = []

    for racer in racers:
        ckpt = newest_checkpoint(racer.ckpt_dir)
        args_src = ckpt.path.parent / "args.json"
        if not args_src.exists():
            raise FileNotFoundError(f"missing args.json beside {ckpt.path}")

        model_dir = models_dir / racer.label
        model_dir.mkdir(parents=True, exist_ok=True)
        frozen_ckpt = model_dir / ckpt.path.name
        if not frozen_ckpt.exists():
            os.link(ckpt.path, frozen_ckpt)
        frozen_args = model_dir / "args.json"
        frozen_args.write_bytes(args_src.read_bytes())

        items.append(
            {
                "label": racer.label,
                "source_checkpoint": str(ckpt.path),
                "frozen_checkpoint": str(frozen_ckpt),
                "step": ckpt.step,
                "loss": ckpt.loss,
                "size_bytes": frozen_ckpt.stat().st_size,
            }
        )

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot": snapshot_name,
        "items": items,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_eval(manifest: dict, probes: str, device: str, batch_size: int, primary_score: str) -> None:
    snapshot_dir = Path(manifest["items"][0]["frozen_checkpoint"]).parents[2]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "racer_eval_suite.py"),
        "--probes",
        probes,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--primary_score",
        primary_score,
        "--out",
        str(snapshot_dir / "eval.json"),
        "--report",
        str(snapshot_dir / "eval.md"),
    ]
    for item in manifest["items"]:
        cmd.extend(["--checkpoint", item["frozen_checkpoint"], "--label", item["label"]])
    env = os.environ.copy()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def steps_tuple(manifest: dict) -> tuple[int, ...]:
    return tuple(int(item["step"]) for item in manifest["items"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--racer", action="append", default=[], help="LABEL=/path/to/checkpoint_dir. Repeatable.")
    p.add_argument("--out_root", default="/home/erikg/racer_eval_runs/ctx2k_periodic")
    p.add_argument("--probes", default="built-in")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--primary_score", choices=["avg_nll", "nll", "pmi_avg"], default="avg_nll")
    p.add_argument("--snapshots", type=int, default=4)
    p.add_argument("--interval_minutes", type=float, default=360.0)
    p.add_argument("--min_step_delta", type=int, default=0)
    p.add_argument("--skip_initial", action="store_true")
    p.add_argument("--poll_seconds", type=float, default=300.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    racers = parse_racers(args.racer)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    last_steps: tuple[int, ...] | None = None
    next_due = time.time() if not args.skip_initial else time.time() + args.interval_minutes * 60

    while completed < args.snapshots:
        now = time.time()
        if now < next_due:
            time.sleep(min(args.poll_seconds, next_due - now))
            continue

        snapshot_name = time.strftime("snapshot_%Y%m%d_%H%M%S", time.gmtime())
        manifest = freeze_snapshot(racers, out_root, snapshot_name)
        current_steps = steps_tuple(manifest)
        if last_steps is not None:
            deltas = [cur - prev for cur, prev in zip(current_steps, last_steps)]
            if any(delta < args.min_step_delta for delta in deltas):
                print(
                    f"{snapshot_name}: not enough new steps {deltas}; "
                    f"need >= {args.min_step_delta}; polling again",
                    flush=True,
                )
                next_due = time.time() + args.poll_seconds
                continue

        print(f"{snapshot_name}: running eval for steps {current_steps}", flush=True)
        run_eval(manifest, args.probes, args.device, args.batch_size, args.primary_score)
        completed += 1
        last_steps = current_steps
        next_due = time.time() + args.interval_minutes * 60

    print(f"completed {completed} snapshots under {out_root}", flush=True)


if __name__ == "__main__":
    main()
