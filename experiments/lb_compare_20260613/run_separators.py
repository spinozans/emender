#!/usr/bin/env python3
"""lb-compare — formal-separator length-extrapolation at each CMA-best model's
found CELL/head-composition, MATCHED capacity (dim/depth fixed => capacity/width
control). Running the literal 1.3B width on 10k-step synthetic tasks is infeasible
and confounds capability with parameter count; capability is an architectural
property of the cell + head mixture, so we hold capacity fixed and vary only the
architecture the CMA actually selected. Per-arm param counts are reported as the
explicit width control.

Separators (real algorithmic tasks, dense per-position supervision):
  anbncn_viability       a^n b^n c^n viability (unbounded 3-counter comparisons)
  dyck_depth_unbounded   unbounded Dyck nesting depth (Weiss-Goldberg-Yahav)
  modular_counter        modular counting (bounded-count separator)
Trained at T=128, evaluated at T in {128,256,512,1024} (Deletang length-extrap).

REAL training. No fabrication.
"""
import os, sys, json, time, subprocess, argparse, re
from pathlib import Path
import concurrent.futures as cf

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import cmaes_search_v2 as C

HYBRID = str(ROOT / 'experiments/expressivity_tasks/train_hybrid.py')
EMENDER_LOGITS = ",".join(repr(float(x)) for x in C.emender_head_type_logits(0.9707993613680964))

# Matched capacity: dim/depth fixed for ALL arms (capacity/width control).
DIM = 512
DEPTH = 4
N_HEADS = 16

# Each arm = the found CELL/head-composition (the architectural identity CMA chose).
# extra_flags carry the cell switches; n_state matches the found geometry.
ARMS = {
    "pure-E97":    dict(pattern=["E97"],        n_state=16, extra=["--e88_raw_write","1"]),
    "Emender-mix": dict(pattern=["typed-gdn2"], n_state=32,
                        extra=["--head_type_logits=" + EMENDER_LOGITS,
                               "--gdn_allow_neg_eigval","1",
                               "--e97_state_nonlin","tanh",
                               "--use_chunked_e97_delta","0",
                               "--lam_max","1.585","--beta_max","2.747"]),
    "gdn2-mlp":    dict(pattern=["fla-gdn"],    n_state=32,
                        extra=["--gdn_allow_neg_eigval","1","--mlp_ratio","3.2587"]),
    "m2rnn":       dict(pattern=["m2rnn"],      n_state=16, extra=[]),
    "emender-mlp": dict(pattern=["E97"],        n_state=32,
                        extra=["--e88_raw_write","1","--mlp_ratio","2.2623"]),
}

TASKS = {
    "anbncn_viability":     dict(steps=10000, seq_len=128, K=2,
                                 eval_lengths=[128,256,512,1024]),
    "dyck_depth_unbounded": dict(steps=10000, seq_len=128, K=8,
                                 eval_lengths=[128,256,512,1024]),
    "modular_counter":      dict(steps=10000, seq_len=128, K=5,
                                 eval_lengths=[128,256,512,1024]),
}


def build_cmd(arm, task, seed, outdir, steps_scale):
    a = ARMS[arm]; t = TASKS[task]
    steps = max(1, int(round(t["steps"] * steps_scale)))
    cmd = [sys.executable, HYBRID, "--task", task,
           "--layer_pattern", *a["pattern"],
           "--dim", str(DIM), "--depth", str(DEPTH),
           "--n_heads", str(N_HEADS), "--n_state", str(a["n_state"]),
           "--expansion", "1.0",
           "--steps", str(steps), "--seq_len", str(t["seq_len"]),
           "--batch_size", "32", "--lr", "3e-4", "--optimizer", "schedulefree",
           "--K", str(t["K"]), "--seed", str(seed),
           "--label", f"sep_{task}__{arm}__seed{seed}",
           "--output_dir", str(outdir),
           "--eval_lengths", *[str(x) for x in t["eval_lengths"]],
           "--eval_lengths_n_batches", "8"]
    cmd += a["extra"]
    return cmd


def run_one(arm, task, seed, gpu, outdir, steps_scale):
    label = f"sep_{task}__{arm}__seed{seed}"
    jpath = outdir / f"{label}.json"
    logf = outdir / f"{label}.log"
    cmd = build_cmd(arm, task, seed, outdir, steps_scale)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("XMA_PATH", "/home/erikg/xma")
    t0 = time.time()
    with open(logf, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n"); lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    out = dict(arm=arm, task=task, seed=seed, gpu=str(gpu), returncode=p.returncode,
               wall_s=round(wall,1))
    if jpath.exists():
        try:
            d = json.load(open(jpath))
            out["final_acc"] = d.get("final_acc")
            out["final_loss"] = d.get("final_loss")
            out["length_extrap"] = d.get("length_extrap")
            out["n_params"] = d.get("params")
            out["random_baseline"] = d.get("random_baseline_acc")
        except Exception as e:
            out["parse_error"] = str(e)
    else:
        txt = logf.read_text(errors="replace")
        out["no_json"] = True
        out["tail"] = txt[-400:]
    print("SEP " + json.dumps({k: out.get(k) for k in
          ("arm","task","seed","returncode","final_acc","n_params")}), flush=True)
    return out


def main():
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()!=""]
    outdir = THIS / "sep_runs"; outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    arms = args.arms.split(",") if args.arms else list(ARMS.keys())
    tasks = args.tasks.split(",") if args.tasks else list(TASKS.keys())
    seeds = [int(s) for s in args.seeds.split(",")]
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                jobs.append((arm, task, seed))
    print(f"{len(jobs)} jobs on {len(gpus)} GPUs", flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, (arm, task, seed) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, arm, task, seed, gpu, outdir, args.steps_scale)] = (arm,task,seed)
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    out = THIS / "sep_results.json"
    prior = {}
    if out.exists():
        for r in json.loads(out.read_text()):
            prior[(r["arm"],r["task"],r["seed"])] = r
    for r in results:
        prior[(r["arm"],r["task"],r["seed"])] = r
    out.write_text(json.dumps(list(prior.values()), indent=2))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps_scale", type=float, default=1.0)
    args = ap.parse_args()
    main()
