#!/usr/bin/env python3
"""clip-sensitivity-control: grad-clip ON-vs-OFF A/B for the two lb-compare arms.

The §6 control proposed in docs/GRAD_CLIP_ACCOUNTING.md: convert the argued
"low confound risk" of grad clipping into a MEASURED result. There is no
committed clip-off run anywhere in the tree (§4c); this fills that gap.

2x2 grid (4 REAL fused-kernel runs):
    {emender-mlp (E97 split-edit), gdn2-mlp} x {--grad_clip 1.0, --grad_clip 0}

MATCHED-TOKEN protocol (per task: "matched tokens"):
  - Each arm is built BYTE-IDENTICALLY to lb-compare via
    scripts/cmaes_search_v2.build_train_command (same level / kwargs / fused
    kernels as the committed run_bpb.py MODELS entries).
  - The wall-clock budget (--train_minutes) is STRIPPED and replaced by a fixed
    --steps budget. should_continue() (train.py:1425) then terminates on
    step-count, not wall-clock, so every cell sees EXACTLY the same number of
    optimizer steps. Both arms are batch_size=4, chunk_size=2048 => 8192
    tokens/step => equal steps == equal tokens, within AND across arms.
  - Each arm holds its CMA-tuned LR FIXED (emender 1.0071e-3, gdn2 4.7431e-4)
    and seed=42, so grad_clip is the ONLY variable within each arm.
  - bf16 + fused per NON-NEGOTIABLE #1: emender uses the repo Triton split-edit
    kernel (--use_triton 1, [fused-guard] assert); gdn2 uses the external FLA
    chunked GDN-2 fused kernel ([fused-guard] GDN2_PATH assert). Both guards are
    parsed back out of each log to PROVE no eager fallback ran.

Held-out: the SAME fixed disjoint Pile-tail slice lb-compare used
(experiments/lb_compare_20260613/heldout_p50k_2048.pt, rebuilt byte-identically
by build_heldout_tensor.py, SEED=7777). BPB scored on schedule-free AVERAGED
weights (FINAL_HELDOUT_BPB) AND non-averaged final weights
(FINAL_HELDOUT_BPB_NONAVG, via HELDOUT_REPORT_NONAVG=1). The lb-compare
correction prefers the NON-AVG basis (averaged weights are a known short-budget
artifact), so the gap analysis leads with non-avg and reports avg for robustness.

Per cell we record: held-out BPB (avg + non-avg), non-finite SKIP count
(train.py:1555 "SKIPPING this step"), non-finite-loss STOP (train.py:1496
=> honest divergence, no FINAL_HELDOUT emitted), pre-clip grad-norm distribution
(the logged `grad N.NN` column is the pre-clip total norm, train.py:1538/1540),
steps reached, fused-guard confirmation.

REAL training, REAL Pile data, REAL fused kernels. No fabrication.
"""
import os, sys, json, time, subprocess, argparse, re, statistics
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]                                  # repo root
LBC = ROOT / 'experiments' / 'lb_compare_20260613'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import cmaes_search_v2 as C
# Pin the search/lb-compare protocol globals (pile.txt, ctx2048, p50k_base).
C.DATA_PATH = '/home/erikg/elman/data/pile.txt'
C.CHUNK_SIZE = 2048
C.TOKENIZER_NAME = 'p50k_base'

HELDOUT = str(LBC / 'heldout_p50k_2048.pt')             # SAME disjoint slice as lb-compare

# (model_type, params, extra-flags) — copied verbatim from the committed
# experiments/lb_compare_20260613/run_bpb.py MODELS entries for these two arms,
# and re-verified against the committed args.json of each arm's final run.
ARMS = {
    "emender-mlp": ("e97-raw",
        dict(dim=1792, n_heads=216, n_state=32, depth=11,
             lr=0.0010071509461604343, batch_size=4),
        ["--mlp_ratio", "2.262336203876648", "--mlp_multiple", "64"]),
    "gdn2-mlp": ("gdn2-mlp",
        dict(dim=2176, expansion=1, n_heads=30, depth=12,
             mlp_ratio=3.258732449079677,
             lr=0.00047431158698290157, batch_size=4),
        []),
}


def build_cmd(arm, clip, steps, outdir, seed=42):
    model_type, params, extra = ARMS[arm]
    # train_minutes placeholder (15.0); stripped below so --steps governs.
    cmd, est = C.build_train_command(params, model_type, 15.0, str(outdir))
    cmd = list(cmd)
    while '--train_minutes' in cmd:                     # strip wall-clock budget
        i = cmd.index('--train_minutes'); del cmd[i:i + 2]
    if '--seed' in cmd:                                 # override the pinned seed (42)
        i = cmd.index('--seed'); cmd[i + 1] = str(seed)
    else:
        cmd += ['--seed', str(seed)]
    cmd += list(extra)
    cmd += ['--steps', str(steps)]
    cmd += ['--grad_clip', str(clip)]                   # the ONLY variable
    cmd += ['--heldout_tensor', HELDOUT, '--final_heldout_eval']
    return cmd, est


def parse_log(txt):
    def last(pat):
        m = re.findall(pat, txt)
        return m[-1] if m else None
    grads = [float(x) for x in re.findall(r'\| grad ([0-9.]+) \|', txt)]
    res = dict(
        n_params=last(r"Model: Level \S+, ([\d,]+) parameters"),
        steps_done=last(r"Training complete! Final step: (\d+)"),
        final_loss_last100=last(r"FINAL_LOSS_LAST100: ([\d.]+)"),
        heldout_ce_avg=last(r"FINAL_HELDOUT_CE: ([\d.]+)"),
        heldout_bpb_avg=last(r"FINAL_HELDOUT_BPB: ([\d.]+)"),
        heldout_ce_nonavg=last(r"FINAL_HELDOUT_CE_NONAVG: ([\d.]+)"),
        heldout_bpb_nonavg=last(r"FINAL_HELDOUT_BPB_NONAVG: ([\d.]+)"),
        heldout_tokens=last(r"FINAL_HELDOUT_TOKENS: (\d+)"),
        heldout_bpt=last(r"FINAL_HELDOUT_BYTES_PER_TOKEN: ([\d.]+)"),
        peak_mem_mb=last(r"PEAK_MEMORY_MB: ([\d.]+)"),
        # instability instrumentation
        skip_count=len(re.findall(r"SKIPPING this step", txt)),
        nonfinite_loss_stop=("Non-finite loss" in txt),
        nonfinite_grad_stop=("Non-finite grad norm" in txt and "Stopping before optimizer step" in txt),
        # fused-guard proof (NON-NEGOTIABLE #1)
        fused_e97=("-> fused split-edit Triton kernel, NO eager fallback" in txt),
        fused_gdn2=("FLA chunked GDN-2 fused kernel, NO eager fallback" in txt),
        # pre-clip grad-norm distribution (sampled @ log_every=10)
        grad_n=len(grads),
        grad_mean=round(statistics.mean(grads), 4) if grads else None,
        grad_median=round(statistics.median(grads), 4) if grads else None,
        grad_max=max(grads) if grads else None,
        grad_p90=round(sorted(grads)[int(0.9 * len(grads))], 4) if grads else None,
        grad_gt1=sum(g > 1.0 for g in grads),
        grad_gt2=sum(g > 2.0 for g in grads),
        grad_frac_gt1=round(sum(g > 1.0 for g in grads) / len(grads), 4) if grads else None,
    )
    res["fused_guard_ok"] = bool(res["fused_e97"] or res["fused_gdn2"])
    return res


def run_one(arm, clip, steps, gpu, seed=42):
    onoff = 'on' if float(clip) > 0 else 'off'
    tag = f"{arm}__clip{onoff}" + (f"__s{seed}" if seed != 42 else "")
    outdir = THIS / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cmd, est = build_cmd(arm, clip, steps, outdir, seed=seed)
    logf = outdir / "train.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HELDOUT_REPORT_NONAVG"] = "1"
    env.setdefault("GDN2_PATH", "/home/erikg/GatedDeltaNet-2")
    env.setdefault("XMA_PATH", "/home/erikg/xma")
    t0 = time.time()
    with open(logf, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    txt = logf.read_text(errors="replace")
    res = dict(tag=tag, arm=arm, grad_clip=float(clip), seed=seed, gpu=str(gpu),
               steps_budget=steps, est_params=est, returncode=p.returncode,
               wall_s=round(wall, 1))
    res.update(parse_log(txt))
    (outdir / "result.json").write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps({k: res.get(k) for k in (
        "tag", "returncode", "steps_done", "heldout_bpb_nonavg", "heldout_bpb_avg",
        "skip_count", "nonfinite_loss_stop", "fused_guard_ok",
        "grad_mean", "grad_max", "grad_frac_gt1", "wall_s")}), flush=True)
    return res


def main():
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    cells = [(arm, clip) for arm in ("emender-mlp", "gdn2-mlp") for clip in ("1.0", "0")]
    if args.only:
        keep = set(args.only.split(","))
        cells = [c for c in cells if f"{c[0]}__clip{('on' if float(c[1])>0 else 'off')}" in keep
                 or c[0] in keep]
    import concurrent.futures as cf
    results = [None] * len(cells)
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {}
        for i, (arm, clip) in enumerate(cells):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(run_one, arm, clip, args.steps, gpu, args.seed)] = i
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    out = THIS / ("clip_ab_results.json" if args.seed == 42
                  else f"clip_ab_results_s{args.seed}.json")
    prior = {}
    if out.exists():
        for r in json.loads(out.read_text()):
            prior[r["tag"]] = r
    for r in results:
        prior[r["tag"]] = r
    out.write_text(json.dumps(list(prior.values()), indent=2))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="csv of leased GPU ids")
    ap.add_argument("--steps", type=int, default=850, help="fixed matched-token step budget")
    ap.add_argument("--seed", type=int, default=42, help="training seed (data order + init)")
    ap.add_argument("--only", default=None, help="csv of tags/arms to run")
    args = ap.parse_args()
    main()
