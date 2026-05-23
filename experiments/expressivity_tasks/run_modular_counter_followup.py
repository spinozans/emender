"""Focused modular-counter follow-up for E88/NDM vs M2RNN.

This sweep targets the ambiguous result from the full 8M matched suite:
E88 solves modular_counter K=5 in most seeds, but tied M2RNN is slightly ahead
at 10K steps. The follow-up separates four effects:

* finite-training/grokking: longer K=5 runs;
* precision: E88/FLA with and without bf16 autocast;
* multiprogramming shape: E88 H32/N32 vs H64/N16;
* algorithmic generalization: harder K=20/K=50 and length extrapolation.

The runner is intentionally independent of run_separation_suite.py so it can be
queued behind a larger suite without modifying an already-running job queue.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]


VARIANTS = {
    "K5_T128_long": {
        "K": 5,
        "seq_len": 128,
        "steps": 30000,
        "eval_lengths": [128, 256, 512, 1024, 2048],
    },
    "K20_T256_hard": {
        "K": 20,
        "seq_len": 256,
        "steps": 20000,
        "eval_lengths": [256, 512, 1024, 2048],
    },
    "K50_T256_hard": {
        "K": 50,
        "seq_len": 256,
        "steps": 20000,
        "eval_lengths": [256, 512, 1024, 2048],
    },
}


MODELS = {
    "E88_H32N32_bf16": {
        "layer_pattern": ["E88"],
        "dim": 384,
        "n_heads": 32,
        "n_state": 32,
        "extra": [],
    },
    "E88_H32N32_fp32": {
        "layer_pattern": ["E88"],
        "dim": 384,
        "n_heads": 32,
        "n_state": 32,
        "extra": ["--disable_autocast"],
    },
    "E88_H64N16_fp32": {
        "layer_pattern": ["E88"],
        "dim": 384,
        "n_heads": 64,
        "n_state": 16,
        "extra": ["--disable_autocast"],
    },
    "FLA_H32N32_bf16": {
        "layer_pattern": ["fla-gdn"],
        "dim": 640,
        "n_heads": 32,
        "n_state": 32,
        "extra": [],
    },
    "FLA_H32N32_fp32": {
        "layer_pattern": ["fla-gdn"],
        "dim": 640,
        "n_heads": 32,
        "n_state": 32,
        "extra": ["--disable_autocast"],
    },
    "M2RNN_tied": {
        "layer_pattern": ["m2rnn"],
        "dim": 384,
        "n_heads": 32,
        "n_state": 32,
        "extra": [],
    },
    "M2RNN_paper": {
        "layer_pattern": ["m2rnn-paper"],
        "dim": 608,
        "n_heads": 32,
        "n_state": 32,
        "extra": [],
    },
}


@dataclass(frozen=True)
class Job:
    variant: str
    model: str
    seed: int

    @property
    def label(self) -> str:
        return f"modcount_{self.variant}__{self.model}__seed{self.seed}"


def build_command(job: Job, args: argparse.Namespace, out_dir: Path) -> list[str]:
    variant = VARIANTS[job.variant]
    model = MODELS[job.model]
    steps = max(1, int(round(variant["steps"] * args.steps_scale)))
    cmd = [
        "python",
        str(THIS / "train_hybrid.py"),
        "--task",
        "modular_counter",
        "--layer_pattern",
        *model["layer_pattern"],
        "--dim",
        str(model["dim"]),
        "--depth",
        str(args.depth),
        "--steps",
        str(steps),
        "--seq_len",
        str(variant["seq_len"]),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--optimizer",
        "schedulefree",
        "--n_heads",
        str(model["n_heads"]),
        "--n_state",
        str(model["n_state"]),
        "--expansion",
        "1.0",
        "--K",
        str(variant["K"]),
        "--seed",
        str(job.seed),
        "--label",
        job.label,
        "--output_dir",
        str(out_dir),
        "--eval_lengths",
        *[str(x) for x in variant["eval_lengths"]],
        "--eval_lengths_n_batches",
        str(args.eval_lengths_n_batches),
        *model["extra"],
    ]
    if args.use_triton_e88:
        cmd.append("--use_triton_e88")
    return cmd


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def run_job(job: Job, gpu: str, args: argparse.Namespace, out_dir: Path) -> dict:
    json_path = out_dir / f"{job.label}.json"
    log_path = out_dir / f"{job.label}.log"
    if json_path.exists() and not args.force:
        data = load_result(json_path)
        return {
            "variant": job.variant,
            "model": job.model,
            "seed": job.seed,
            "status": "skipped",
            "gpu": gpu,
            "json": str(json_path),
            "log": str(log_path),
            "final_acc": data.get("final_acc"),
            "final_loss": data.get("final_loss"),
            "length_extrap": data.get("length_extrap"),
            "params": data.get("params"),
            "elapsed_total_s": data.get("elapsed_total_s"),
        }

    cmd = build_command(job, args, out_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.time()
    with log_path.open("w") as log:
        log.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    result = {
        "variant": job.variant,
        "model": job.model,
        "seed": job.seed,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "gpu": gpu,
        "json": str(json_path),
        "log": str(log_path),
        "elapsed_wall_s": time.time() - start,
    }
    if proc.returncode == 0 and json_path.exists():
        data = load_result(json_path)
        result.update(
            final_acc=data.get("final_acc"),
            final_loss=data.get("final_loss"),
            length_extrap=data.get("length_extrap"),
            params=data.get("params"),
            elapsed_total_s=data.get("elapsed_total_s"),
        )
    return result


def worker(gpu: str, jobs: queue.Queue[Job], results: list[dict], lock: threading.Lock,
           args: argparse.Namespace, out_dir: Path) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        print(f"[gpu {gpu}] start {job.label}", flush=True)
        result = run_job(job, gpu, args, out_dir)
        with lock:
            results.append(result)
            write_summary(results, out_dir)
        if result["status"] in {"ok", "skipped"}:
            print(
                f"[gpu {gpu}] {result['status']:>7s} {job.label} "
                f"acc={result.get('final_acc')} loss={result.get('final_loss')}",
                flush=True,
            )
        else:
            print(f"[gpu {gpu}] FAILED {job.label} log={result['log']}", flush=True)
        jobs.task_done()


def write_summary(results: list[dict], out_dir: Path) -> None:
    with (out_dir / "modular_counter_followup_summary.json").open("w") as f:
        json.dump(results, f, indent=2)


def aggregate(results: list[dict]) -> str:
    ok = [
        r for r in results
        if r["status"] in {"ok", "skipped"} and r.get("final_acc") is not None
    ]
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in ok:
        by_key.setdefault((row["variant"], row["model"]), []).append(row)

    lines = ["", "=== Aggregate ==="]
    lines.append(f"{'variant':>16s}  {'model':>18s}  {'n':>2s}  {'mean_acc':>9s}  {'min':>7s}  {'max':>7s}")
    for (variant, model), rows in sorted(by_key.items()):
        accs = [float(r["final_acc"]) for r in rows]
        lines.append(
            f"{variant:>16s}  {model:>18s}  {len(accs):2d}  "
            f"{sum(accs) / len(accs):9.4f}  {min(accs):7.4f}  {max(accs):7.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,5,6,7")
    parser.add_argument("--output_dir", default="experiments/expressivity_tasks/results/modular_counter_followup_20260511")
    parser.add_argument("--variants", nargs="+", default=["K5_T128_long", "K20_T256_hard", "K50_T256_hard"],
                        choices=sorted(VARIANTS))
    parser.add_argument("--models", nargs="+",
                        default=[
                            "E88_H32N32_bf16",
                            "E88_H32N32_fp32",
                            "E88_H64N16_fp32",
                            "FLA_H32N32_fp32",
                            "M2RNN_tied",
                            "M2RNN_paper",
                        ],
                        choices=sorted(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps_scale", type=float, default=1.0)
    parser.add_argument("--eval_lengths_n_batches", type=int, default=8)
    parser.add_argument("--use_triton_e88", action="store_true",
                        help="Route E88 layers through the Triton fwd/bwd kernels.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue[Job] = queue.Queue()
    for variant in args.variants:
        for model in args.models:
            for seed in args.seeds:
                jobs.put(Job(variant=variant, model=model, seed=seed))

    total = jobs.qsize()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    print(f"Queued {total} modular-counter jobs on GPUs {','.join(gpus)}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, jobs, results, lock, args, out_dir), daemon=False)
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    write_summary(results, out_dir)
    print(aggregate(results), flush=True)


if __name__ == "__main__":
    main()
