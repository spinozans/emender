#!/usr/bin/env python3
"""Iso-geometry R confound control for cmaes-m2-1p3b.

The CMA search's best-per-R curve (R1 6.2424 .. R3 6.1843) is confounded: each R
was evaluated at its own CMA-found geometry and with UNEQUAL sampling (R1 got 14
evals, R3 got 30). This control removes both confounds: it fixes the geometry
basin (dim2816 / n_state16 / depth10, the CMA-best shape) and the optimizer
(lr / batch), varies ONLY the readout rank R, adjusts n_heads per R to hold
params at ~1.3B (iso-param), and runs MULTIPLE SEEDS to expose the noise floor.

Metric == the search fitness: parse_average_loss over all logged steps, 15 min,
bf16 + FUSED (E97-M2 forces the Triton kernel + fused-guard; no eager).

Run inside a 2-GPU lease:  python iso_geometry_R_control.py 6,7
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/home/erikg/ndm/.wg-worktrees/agent-1514"
sys.path.insert(0, os.path.join(REPO, "scripts"))
import cmaes_search_v2 as C

C.DATA_PATH = "/home/erikg/elman/data/pile.txt"
C.CHUNK_SIZE = 2048
C.TOKENIZER_NAME = "p50k_base"
C.USE_TRITON_E88 = True
C.PARAM_VOCAB_SIZE = C.resolve_vocab_size("p50k_base")

OUT = os.path.join(REPO, "experiments/cmaes_m2_1p3b_20260616/iso_geometry_R_control")
os.makedirs(OUT, exist_ok=True)

DIM, NS, DEP = 2816, 16, 10
LR, BS, MINUTES = 7.1e-4, 2, 15
NH_PER_R = {1: 364, 2: 284, 3: 232, 4: 197}   # -> ~1.3B each (verified)
SEEDS = [42, 123, 7]

gpus = [int(g) for g in sys.argv[1].split(",")]

def run_one(R, seed, gpu_id):
    nh = NH_PER_R[R]
    params = dict(dim=DIM, n_heads=nh, n_state=NS, depth=DEP, lr=LR,
                  batch_size=BS, multiquery_r=R)
    odir = os.path.join(OUT, f"R{R}_seed{seed}")
    os.makedirs(odir, exist_ok=True)
    cmd, est = C.build_train_command(params, "e97-m2", MINUTES, odir)
    cmd = C.strip_cmd_arg(cmd, "--seed")
    cmd += ["--seed", str(seed)]
    env = C.prepare_worker_env("e97-m2", gpu_id)
    import subprocess
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO,
                         timeout=MINUTES * 60 + 600)
    avg = C.parse_average_loss(res.stdout)
    final = C.parse_final_loss(res.stdout, odir)
    guard = "NO eager fallback" in res.stdout
    nparam = None
    for line in res.stdout.split("\n"):
        if "parameters" in line and "Level" in line:
            import re
            m = re.search(r"([0-9,]+) parameters", line)
            if m: nparam = int(m.group(1).replace(",", ""))
    with open(os.path.join(odir, "stdout.txt"), "w") as f:
        f.write(res.stdout)
    rec = dict(R=R, seed=seed, n_heads=nh, est_params=est, real_params=nparam,
               avg_loss=avg, final_loss=final, fused_guard=guard,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(odir, "result.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  done R={R} seed={seed} nh={nh} | avg={avg:.4f} final={final} "
          f"| params={nparam} guard={guard}", flush=True)
    return rec

jobs = [(R, s) for R in sorted(NH_PER_R) for s in SEEDS]
print(f"iso-geometry R control: {len(jobs)} runs on GPUs {gpus} "
      f"(dim{DIM}/ns{NS}/dep{DEP}, lr{LR}, bs{BS}, {MINUTES}min, seeds {SEEDS})", flush=True)

results = []
from queue import Queue
pool = Queue()
for g in gpus: pool.put(g)
def worker(job):
    R, s = job
    g = pool.get()
    try:
        return run_one(R, s, g)
    finally:
        pool.put(g)

with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
    futs = {ex.submit(worker, j): j for j in jobs}
    for fut in as_completed(futs):
        results.append(fut.result())

# Aggregate: mean/min avg_loss per R across seeds
import statistics as st
byR = {}
for r in results:
    byR.setdefault(r["R"], []).append(r["avg_loss"])
summary = {}
for R in sorted(byR):
    vals = byR[R]
    summary[R] = dict(mean=round(st.mean(vals), 4),
                      std=round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                      min=round(min(vals), 4), n=len(vals), vals=[round(v,4) for v in vals])
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(dict(config=dict(dim=DIM, n_state=NS, depth=DEP, lr=LR, bs=BS,
                               minutes=MINUTES, nh_per_R=NH_PER_R, seeds=SEEDS),
                   per_R=summary, all_runs=results), f, indent=2)
print("\n=== ISO-GEOMETRY R CONTROL SUMMARY (avg-loss; lower=better) ===", flush=True)
print(f"{'R':>2} {'n_heads':>7} {'mean':>8} {'std':>7} {'min':>8}  vals", flush=True)
for R in sorted(summary):
    s = summary[R]
    print(f"{R:>2} {NH_PER_R[R]:>7} {s['mean']:>8.4f} {s['std']:>7.4f} {s['min']:>8.4f}  {s['vals']}", flush=True)
print("\nall fused:", all(r["fused_guard"] for r in results),
      "| all ~1.3B:", all(abs((r["real_params"] or 0)-1.3e9)/1.3e9 < 0.03 for r in results), flush=True)
