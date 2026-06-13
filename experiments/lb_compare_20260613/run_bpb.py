#!/usr/bin/env python3
"""lb-compare — held-out BPB leaderboard for the 5 CMA-best models at their OWN
found geometries, SAME disjoint held-out slice, SAME protocol (bf16 + fused,
15-min train budget matching the CMA search, p50k_base, pile.txt seed42).

Reuses scripts/cmaes_search_v2.build_train_command so every model is constructed
BYTE-IDENTICALLY to its CMA search. Appends --heldout_tensor (fixed disjoint
slice) + --final_heldout_eval so train.py prints FINAL_HELDOUT_CE / BPB on the
schedule-free AVERAGED weights. One GPU per model (no cross-agent contention:
this single driver owns the lease set passed via CUDA_VISIBLE_DEVICES csv).

REAL training + REAL held-out eval. No fabrication.
"""
import os, sys, json, time, subprocess, argparse, re
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import cmaes_search_v2 as C
# Pin the search protocol globals (pile.txt, ctx2048, p50k_base).
C.DATA_PATH = '/home/erikg/elman/data/pile.txt'
C.CHUNK_SIZE = 2048
C.TOKENIZER_NAME = 'p50k_base'

HELDOUT = str(THIS / 'heldout_p50k_2048.pt')

# (model_type, params dict at found geometry, extra train.py flags, search_avg_loss, label)
MODELS = [
    ("e97-raw",
     dict(dim=2432, n_heads=416, n_state=16, depth=10, lr=0.0009851067699366818, batch_size=3),
     [], 5.9511, "pure-E97"),
    ("emender",
     dict(dim=2432, n_heads=212, n_state=32, depth=10, mixture_nonlin=0.9707993613680964,
          lr=0.0011443458778126467, batch_size=2),
     [], 6.0756, "Emender-mix"),
    ("gdn2-mlp",
     dict(dim=2176, expansion=1, n_heads=30, depth=12, mlp_ratio=3.258732449079677,
          lr=0.00047431158698290157, batch_size=4),
     [], 5.8949, "gdn2-mlp"),
    ("m2rnn",
     dict(dim=3072, n_heads=346, n_state=16, depth=13, lr=0.0010395553876216513, batch_size=4),
     [], 6.0636, "m2rnn"),
    # emender-mlp = pure-E97 cell (e97-raw, E88FLAHybrid split-edit raw-write) + SwiGLU MLP,
    # n_state PINNED 32, mlp_multiple 64 (the FAIR MLP counterpart of gdn2-mlp).
    ("e97-raw",
     dict(dim=1792, n_heads=216, n_state=32, depth=11, lr=0.0010071509461604343, batch_size=4),
     ["--mlp_ratio", "2.262336203876648", "--mlp_multiple", "64"], 5.8606, "emender-mlp"),
]


def build_cmd(model_type, params, extra, outdir):
    cmd, est_params = C.build_train_command(params, model_type, args.train_minutes, str(outdir))
    cmd = list(cmd) + list(extra)
    cmd += ["--heldout_tensor", HELDOUT, "--final_heldout_eval"]
    return cmd, est_params


def run_one(idx, model_type, params, extra, search_loss, label, gpu):
    outdir = THIS / "runs" / f"{label}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd, est_params = build_cmd(model_type, params, extra, outdir)
    logf = outdir / "train.log"
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("XMA_PATH", "/home/erikg/xma")
    env["HELDOUT_REPORT_NONAVG"] = "1"   # report both averaged + non-averaged held-out
    t0 = time.time()
    with open(logf, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n"); lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    txt = logf.read_text(errors="replace")
    def grab(pat):
        m = re.findall(pat, txt)
        return m[-1] if m else None
    res = dict(
        label=label, model_type=model_type, gpu=gpu, params=params, extra=extra,
        est_params=est_params,
        search_avg_loss=search_loss, returncode=p.returncode, wall_s=round(wall, 1),
        n_params=grab(r"Model: Level \S+, ([\d,]+) parameters"),
        final_loss_last100=grab(r"FINAL_LOSS_LAST100: ([\d.]+)"),
        heldout_ce=grab(r"FINAL_HELDOUT_CE: ([\d.]+)"),
        heldout_bpb=grab(r"FINAL_HELDOUT_BPB: ([\d.]+)"),
        heldout_ce_nonavg=grab(r"FINAL_HELDOUT_CE_NONAVG: ([\d.]+)"),
        heldout_bpb_nonavg=grab(r"FINAL_HELDOUT_BPB_NONAVG: ([\d.]+)"),
        heldout_tokens=grab(r"FINAL_HELDOUT_TOKENS: (\d+)"),
        heldout_bpt=grab(r"FINAL_HELDOUT_BYTES_PER_TOKEN: ([\d.]+)"),
        steps=grab(r"Training complete! Final step: (\d+)"),
        peak_mem_mb=grab(r"PEAK_MEMORY_MB: ([\d.]+)"),
        nonfinite=("non-finite" in txt.lower() or "FINAL_HELDOUT_CE" not in txt),
    )
    (outdir / "result.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps({k: res[k] for k in
          ("label","returncode","n_params","heldout_ce","heldout_bpb","final_loss_last100","steps","wall_s")}), flush=True)
    return res


def main():
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if args.only:
        sel = [m for m in MODELS if m[4] in args.only.split(",")]
    else:
        sel = MODELS
    import concurrent.futures as cf
    results = [None] * len(sel)
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, m in enumerate(sel):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, i, m[0], m[1], m[2], m[3], m[4], gpu)] = i
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    out = THIS / "bpb_results.json"
    # merge with any prior results so --only runs accumulate
    prior = {}
    if out.exists():
        for r in json.loads(out.read_text()):
            prior[r["label"]] = r
    for r in results:
        prior[r["label"]] = r
    out.write_text(json.dumps(list(prior.values()), indent=2))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="csv of GPU ids (from lease)")
    ap.add_argument("--train_minutes", type=float, default=15.0)
    ap.add_argument("--only", default=None, help="csv of labels to run")
    args = ap.parse_args()
    main()
