#!/usr/bin/env python
"""S5 candidate-budget calibration launcher (task s5sym-calibrate).

Runs ~10 deliberately-spread configs across THREE 8M arms (E88, M2RNN, GDN/FLA)
on the S5 permutation task, seq_len=128, batch=32, seed=42, schedule-free AdamW,
4000 steps each with DENSE S5 eval logging every 100 steps.

Round-robins across GPUs 2,3,4,5 ONLY (never 0,1). Up to 4 concurrent jobs.
Real harness (train_hybrid.py), real S5 eval. No mocks.
"""
import os, sys, subprocess, time, threading, queue, shutil

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, "train_hybrid.py")
OUT_DIR = os.path.join(THIS, "results", "s5_calib_20260603")
GPUS = [2, 3, 4, 5]
STEPS = 4000
EVAL_INTERVAL = 100

COMMON = dict(task="s5_permutation", depth=4, n_heads=32, n_state=32,
              seq_len=128, batch_size=32, seed=42, optimizer="schedulefree")

# (label, layer_pattern, dim, lr, extra_args)
# E88 listed first (slowest) so the pool front-loads them across all 4 GPUs.
JOBS = [
    # --- E88 arm: dim384/H32/N32 ---
    ("E88_lr3e-4_tanh",   "E88",     384, 3e-4, ["--linear_state", "0"]),  # seed
    ("E88_lr9e-5_tanh",   "E88",     384, 9e-5, ["--linear_state", "0"]),  # lr x0.3
    ("E88_lr9e-4_tanh",   "E88",     384, 9e-4, ["--linear_state", "0"]),  # lr x3
    ("E88_lr3e-4_linear", "E88",     384, 3e-4, ["--linear_state", "1"]),  # BL-1 knob: linear vs tanh
    # --- M2RNN-CMA arm: dim384/H32/N32 ---
    ("M2RNN_lr3e-4",      "m2rnn",   384, 3e-4, []),  # seed
    ("M2RNN_lr9e-5",      "m2rnn",   384, 9e-5, []),  # lr x0.3
    ("M2RNN_lr9e-4",      "m2rnn",   384, 9e-4, []),  # lr x3
    # --- GDN/FLA arm: dim640/H32/N32 ---
    ("GDN_lr3e-4",        "fla-gdn", 640, 3e-4, []),  # seed
    ("GDN_lr9e-5",        "fla-gdn", 640, 9e-5, []),  # lr x0.3
    ("GDN_lr9e-4",        "fla-gdn", 640, 9e-4, []),  # lr x3
]


def gpu_mem_used(idx):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(idx)]).decode().strip()
    return int(out)


def safety_gate():
    busy = [(g, gpu_mem_used(g)) for g in GPUS]
    bad = [(g, m) for g, m in busy if m > 2048]
    if bad:
        print(f"GPU-SAFETY ABORT: GPUs busy (>2GB): {bad}", flush=True)
        sys.exit(2)
    print(f"GPU-safety gate PASS: {busy} (all <2GB on {GPUS})", flush=True)


def build_cmd(job, gpu):
    label, pattern, dim, lr, extra = job
    cmd = ["python", TRAIN,
           "--task", COMMON["task"],
           "--layer_pattern", pattern,
           "--dim", str(dim),
           "--depth", str(COMMON["depth"]),
           "--n_heads", str(COMMON["n_heads"]),
           "--n_state", str(COMMON["n_state"]),
           "--steps", str(STEPS),
           "--eval_interval", str(EVAL_INTERVAL),
           "--seq_len", str(COMMON["seq_len"]),
           "--batch_size", str(COMMON["batch_size"]),
           "--lr", str(lr),
           "--optimizer", COMMON["optimizer"],
           "--seed", str(COMMON["seed"]),
           "--label", label,
           "--output_dir", OUT_DIR] + extra
    return cmd


def worker(gpu, jobq):
    while True:
        try:
            job = jobq.get_nowait()
        except queue.Empty:
            return
        label = job[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log_path = os.path.join(OUT_DIR, f"{label}.train.log")
        t0 = time.time()
        print(f"[GPU{gpu}] START {label}", flush=True)
        with open(log_path, "w") as lf:
            rc = subprocess.call(build_cmd(job, gpu), env=env, stdout=lf,
                                 stderr=subprocess.STDOUT, cwd=os.path.dirname(THIS))
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[GPU{gpu}] DONE  {label}  {status}  {dt:.0f}s", flush=True)
        jobq.task_done()


def main():
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi missing", flush=True); sys.exit(2)
    safety_gate()
    os.makedirs(OUT_DIR, exist_ok=True)
    jobq = queue.Queue()
    for j in JOBS:
        jobq.put(j)
    threads = [threading.Thread(target=worker, args=(g, jobq)) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
