"""Run the matched NDM/E88 expressivity suite across multiple GPUs.

The default suite covers the registered synthetic expressivity tasks. It can
also be restricted to the targeted formal-separation subset:
overwrite_recall, reset_recall, keyed_fsm_memory.

Default model presets are parameter-matched near 8M parameters under the
canonical expressivity regime: schedule-free AdamW, depth=4, H/N chosen to
preserve the many-head E88 setting where the earlier parity/FSM/mod-counter
results grokked.

Each child run writes the usual train_hybrid JSON plus a sibling .log file.
This runner writes an aggregate separation_summary.json at the end.
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


TASK_CONFIG = {
    "parity": {"steps": 10000, "seq_len": 128, "K": 2, "lr": 3e-4},
    "modular_counter": {"steps": 10000, "seq_len": 128, "K": 5, "lr": 3e-4},
    "fsm_tracking": {"steps": 10000, "seq_len": 256, "K": 4, "lr": 3e-4},
    "s3_permutation": {
        "steps": 10000,
        "seq_len": 128,
        "K": 3,
        "lr": 3e-4,
        "eval_lengths": [128, 256, 512, 1024],
        "eval_lengths_n_batches": 8,
    },
    "s5_permutation": {
        "steps": 20000,
        "seq_len": 128,
        "K": 5,
        "lr": 3e-4,
        "eval_lengths": [128, 256, 512, 1024],
        "eval_lengths_n_batches": 8,
    },
    "dyck": {"steps": 10000, "seq_len": 256, "K": 8, "lr": 3e-4},
    "dyck2": {"steps": 10000, "seq_len": 256, "K": 8, "lr": 3e-4},
    "selective_copy": {"steps": 10000, "seq_len": 256, "K": 8, "lr": 3e-4},
    "assoc_recall": {"steps": 10000, "seq_len": 64, "K": 8, "lr": 3e-4},
    "overwrite_recall": {"steps": 10000, "seq_len": 128, "K": 16, "lr": 3e-4},
    "reset_recall": {"steps": 10000, "seq_len": 128, "K": 16, "lr": 3e-4},
    "keyed_fsm_memory": {"steps": 10000, "seq_len": 128, "K": 8, "lr": 3e-4},
}

DEFAULT_TASK_ORDER = [
    "parity",
    "modular_counter",
    "fsm_tracking",
    "s3_permutation",
    "s5_permutation",
    "dyck",
    "dyck2",
    "selective_copy",
    "assoc_recall",
    "overwrite_recall",
    "reset_recall",
    "keyed_fsm_memory",
]


MODEL_CONFIG = {
    "E88_8M": {
        "layer_pattern": ["E88"],
        "dim": 384,
        "n_heads": 32,
        "n_state": 32,
        "kwargs": {},
    },
    "E88_H64N16_8M": {
        "layer_pattern": ["E88"],
        "dim": 384,
        "n_heads": 64,
        "n_state": 16,
        "kwargs": {},
    },
    "FLA_8M": {
        "layer_pattern": ["fla-gdn"],
        "dim": 640,
        "n_heads": 32,
        "n_state": 32,
        "kwargs": {},
    },
    "M2RNN_8M": {
        "layer_pattern": ["m2rnn"],
        "dim": 384,
        "n_heads": 32,
        "n_state": 32,
        "kwargs": {},
    },
    "M2RNN_paper_8M": {
        "layer_pattern": ["m2rnn-paper"],
        "dim": 608,
        "n_heads": 32,
        "n_state": 32,
        "kwargs": {},
    },
    "pure_E88": {"layer_pattern": ["E88"], "kwargs": {}},
    "pure_FLA": {"layer_pattern": ["fla-gdn"], "kwargs": {}},
    "pure_M2RNN": {"layer_pattern": ["m2rnn"], "kwargs": {}},
    "pure_M2RNN_paper": {"layer_pattern": ["m2rnn-paper"], "kwargs": {}},
    "hybrid_GDN_E88_single": {"layer_pattern": ["fla-gdn", "fla-gdn", "fla-gdn", "E88"], "kwargs": {}},
    "hybrid_GDN_M2RNN_single": {"layer_pattern": ["fla-gdn", "fla-gdn", "fla-gdn", "m2rnn-paper"], "kwargs": {}},
}


@dataclass(frozen=True)
class Job:
    task: str
    model: str
    seed: int

    @property
    def label(self) -> str:
        return f"sep_{self.task}__{self.model}__seed{self.seed}"


def build_command(job: Job, args: argparse.Namespace, out_dir: Path) -> list[str]:
    task_cfg = TASK_CONFIG[job.task]
    model_cfg = MODEL_CONFIG[job.model]
    steps = max(1, int(round(task_cfg["steps"] * args.steps_scale)))
    dim = model_cfg.get("dim", args.dim)
    n_heads = model_cfg.get("n_heads", args.n_heads)
    n_state = model_cfg.get("n_state", args.n_state)
    expansion = model_cfg.get("expansion", args.expansion)

    cmd = [
        "python",
        str(THIS / "train_hybrid.py"),
        "--task",
        job.task,
        "--layer_pattern",
        *model_cfg["layer_pattern"],
        "--dim",
        str(dim),
        "--depth",
        str(args.depth),
        "--steps",
        str(steps),
        "--seq_len",
        str(task_cfg["seq_len"]),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(task_cfg["lr"]),
        "--optimizer",
        args.optimizer,
        "--n_heads",
        str(n_heads),
        "--n_state",
        str(n_state),
        "--expansion",
        str(expansion),
        "--K",
        str(task_cfg["K"]),
        "--seed",
        str(job.seed),
        "--label",
        job.label,
        "--output_dir",
        str(out_dir),
    ]
    if task_cfg.get("eval_lengths"):
        cmd.append("--eval_lengths")
        cmd.extend(str(length) for length in task_cfg["eval_lengths"])
        cmd.extend([
            "--eval_lengths_n_batches",
            str(task_cfg.get("eval_lengths_n_batches", 8)),
        ])
    for key, value in model_cfg["kwargs"].items():
        cmd.extend([f"--{key}", str(value)])
    if args.use_triton_e88:
        cmd.append("--use_triton_e88")
    return cmd


def run_job(job: Job, gpu: str, args: argparse.Namespace, out_dir: Path) -> dict:
    json_path = out_dir / f"{job.label}.json"
    log_path = out_dir / f"{job.label}.log"
    if json_path.exists() and not args.force:
        with json_path.open() as f:
            data = json.load(f)
        return {
            "task": job.task,
            "model": job.model,
            "seed": job.seed,
            "status": "skipped",
            "gpu": gpu,
            "json": str(json_path),
            "log": str(log_path),
            "final_acc": data.get("final_acc"),
            "final_loss": data.get("final_loss"),
            "elapsed_total_s": data.get("elapsed_total_s"),
            "params": data.get("params"),
            "random_baseline": data.get("random_baseline_acc"),
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
    elapsed = time.time() - start

    result = {
        "task": job.task,
        "model": job.model,
        "seed": job.seed,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "gpu": gpu,
        "json": str(json_path),
        "log": str(log_path),
        "elapsed_wall_s": elapsed,
    }
    if proc.returncode == 0 and json_path.exists():
        with json_path.open() as f:
            data = json.load(f)
        result.update(
            final_acc=data.get("final_acc"),
            final_loss=data.get("final_loss"),
            elapsed_total_s=data.get("elapsed_total_s"),
            params=data.get("params"),
            random_baseline=data.get("random_baseline_acc"),
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
        if result["status"] in {"ok", "skipped"}:
            print(
                f"[gpu {gpu}] {result['status']:>7s} {job.label} "
                f"acc={result.get('final_acc')} loss={result.get('final_loss')}",
                flush=True,
            )
        else:
            print(f"[gpu {gpu}] FAILED {job.label} log={result['log']}", flush=True)
        jobs.task_done()


def aggregate(results: list[dict], out_dir: Path) -> None:
    ok = [r for r in results if r["status"] in {"ok", "skipped"} and r.get("final_acc") is not None]
    by_key: dict[tuple[str, str], list[dict]] = {}
    for item in ok:
        by_key.setdefault((item["task"], item["model"]), []).append(item)

    print("\n=== Aggregate ===")
    print(f"{'task':>18s}  {'model':>12s}  {'mean_acc':>9s}  {'min':>7s}  {'max':>7s}  {'baseline':>8s}")
    for (task, model), rows in sorted(by_key.items()):
        accs = [float(r["final_acc"]) for r in rows]
        baseline = rows[0].get("random_baseline")
        baseline_s = f"{baseline:.4f}" if baseline is not None else "?"
        print(
            f"{task:>18s}  {model:>12s}  "
            f"{sum(accs) / len(accs):>9.4f}  {min(accs):>7.4f}  {max(accs):>7.4f}  {baseline_s:>8s}"
        )

    with (out_dir / "separation_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {out_dir / 'separation_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,5,6,7")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASK_ORDER,
                        choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--models", nargs="+", default=["E88_8M", "FLA_8M", "M2RNN_8M", "M2RNN_paper_8M"],
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_state", type=int, default=32)
    parser.add_argument("--expansion", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", default="schedulefree", choices=["adamw", "schedulefree"])
    parser.add_argument("--steps_scale", type=float, default=1.0)
    parser.add_argument("--use_triton_e88", action="store_true",
                        help="Route E88 layers through the Triton fwd/bwd kernels.")
    parser.add_argument("--output_dir", default=str(THIS / "results" / "separation_suite"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    job_queue: queue.Queue[Job] = queue.Queue()
    for task in args.tasks:
        for model in args.models:
            for seed in args.seeds:
                job_queue.put(Job(task=task, model=model, seed=seed))

    results: list[dict] = []
    lock = threading.Lock()
    threads = []
    for gpu in [g.strip() for g in args.gpus.split(",") if g.strip()]:
        thread = threading.Thread(
            target=worker,
            args=(gpu, job_queue, results, lock, args, out_dir),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    aggregate(results, out_dir)


if __name__ == "__main__":
    main()
