#!/usr/bin/env python3
"""LMC seed-maturity threshold probe  (task: seed-maturity-threshold).

QUESTION. What is the minimum single-model training maturity at which horizontal
DiLoCo weight-averaging is SAFE -- i.e. the data-diverged replicas stay in the
SAME loss basin so the average is benign/beneficial (SWA), vs the under-trained
regime where replicas land in DIFFERENT basins and the average crosses a loss
barrier (blow-up)?

THEORY (Frankle et al., "linear mode connectivity / the instability analysis").
Take a network at some point in training (the shared PARENT), make I copies, and
train each on different SGD/data noise. After a short early "SGD-stability" phase
the children become linearly mode-connected: the loss along the line between them
has no barrier, so their AVERAGE is as good as the children (SWA). Before that
point the children fall into different basins and the average sits on a loss
barrier. The DiLoCo merge IS this average; the barrier worsens with MORE islands
(centroid further from each child) and LARGER drift per window (children wander
further before the merge). So the threshold is the maturity above which the
merge barrier is <= 0 at the island count / drift we intend to run.

WHAT THIS PROBE MEASURES (REAL, FUSED, no eager, no mock):
  parent  = a single-GPU emender checkpoint at maturity m (full schedule-free
            state incl. base-iterate z).
  replica = parent resumed via the VALIDATED train.py single-GPU path and
            trained independently on a DISJOINT data stream (--seed = base+isl).
            The fused split-edit Triton kernel is asserted on every replica
            (the [fused-guard] line) -- there is no eager path.
  snapshot= each replica is saved every `snap` steps up to K, so a single pair/
            tuple of replica runs yields the barrier at drift d in
            {snap, 2*snap, ..., K} (the Frankle "instability vs amount of SGD"
            curve) without re-training per drift.
  consensus(d) = the SF-aware average of the I matched snapshots at drift d,
            computed offline with the SAME semantics as train.py:diloco_merge
            (mean of the eval-x model weights + mean of the SF base-iterate z;
            schedule-free clocks preserved). This is exactly the plain-average
            local-SGD DiLoCo outer step (outer_lr=1, outer_beta=0).
  scoring = OFFLINE only (scripts/eval_checkpoint.py, --y-mode train, fused) on
            a FIXED held-out tensor. No inline held-out during training.

  LMC barrier(d) = bpb(consensus_d) - mean_i bpb(replica_i,d)
        barrier <= ~0 : in-basin -- averaging benign/beneficial (SWA regime)
        barrier  > 0  : basin mismatch -- the average crosses a loss barrier

Optional --merges M>1 re-seeds all islands from the consensus and repeats,
measuring whether a positive barrier COMPOUNDS (runaway, like the beta=0.9 case)
or self-heals across merges (loss continuity through merges).

Runs entirely on ONE leased GPU (replicas sequential), so it queues behind the
saturated box on a single idle GPU rather than waiting for 4-8. Results are
written incrementally and the campaign is resumable (completed cells skipped).
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUTROOT = Path(os.environ.get(
    "LMC_OUTROOT", "/mnt/nvme1n1/erikg/seed_maturity_threshold"))
RESULTS_CSV = Path(__file__).resolve().parent / "lmc_barrier_results.csv"
HELDOUT = os.environ.get(
    "HELDOUT",
    "/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057/"
    "heldout_pile_tail_p50k_2048_1m.pt")

# --- frozen geometry: byte-identical to the reference run's args.json and the
#     seed-race launch (scripts/run_diloco_seed_race_i4.sh). DO NOT change. -----
TOK_PER_STEP = 4 * 2048          # bs4 * chunk2048 per single-GPU replica step
RECIPE = [
    "--level", "E97", "--params", "100m",
    "--embed_dim", "1024", "--dim", "1792", "--depth", "11",
    "--n_heads", "216", "--n_state", "32",
    "--expansion", "1.0", "--state_expansion", "2",
    "--n_groups", "32", "--n_slots", "64",
    "--mlp_ratio", "2.2623", "--mlp_multiple", "64",
    "--use_gate", "1", "--use_permutation", "1", "--gate_activation", "silu",
    "--gdn_allow_neg_eigval", "1", "--mamba_expand", "2",
    "--use_triton", "1", "--bf16",
    "--optimizer", "schedulefree", "--lr", "0.001007", "--warmup_steps", "0",
    "--batch_size", "4", "--chunk_size", "2048",
    "--data", os.environ.get("DATA", "/home/erikg/elman/data/pile.txt"),
    "--tokenizer", "p50k_base",
]

# --- the single-GPU maturity ladder (step -> tokens = step * 8192) ------------
REFDIR = "/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
LADDER = {  # name -> (checkpoint path, parent_step, tokens)
    "176M":  (f"{REFDIR}/checkpoint_step_021500_loss_3.7778.pt", 21500, 176.1e6),
    "528M":  (f"{REFDIR}/checkpoint_step_064500_loss_3.1246.pt", 64500, 528.4e6),
    "1.06B": (f"{REFDIR}/checkpoint_step_129000_loss_3.1436.pt", 129000, 1.057e9),
    "2.0B":  (f"{REFDIR}/checkpoint_step_244141_loss_3.1168.pt", 244141, 2.000e9),
}

CKPT_RE = re.compile(r"checkpoint_step_(\d+)_loss_([0-9.]+)\.pt$")


def log(msg):
    print(f"[lmc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require_leased_gpu():
    """HARD safety gate. NEVER run on the default GPU 0 (the racers live there).

    The broker pins exactly the leased GPU(s) via CUDA_VISIBLE_DEVICES; if it is
    unset/empty we have NO lease and must abort rather than collide with another
    agent's run. This is the guard whose absence let an earlier launch OOM on the
    racer's GPU 0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip() == "":
        log("FATAL: CUDA_VISIBLE_DEVICES is unset/empty -> no GPU lease. "
            "Refusing to run (would default to GPU 0 and clobber a racer). "
            "Launch via run_lmc_probe.sh which self-leases one IDLE GPU.")
        sys.exit(2)
    return cvd


def newest_run_subdir(parent: Path):
    """train.py creates <parent>/levelE97_100m_<ts>/. Pick newest with ckpts."""
    cands = sorted(parent.glob("level*_*/"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    for d in cands:
        if list(d.glob("checkpoint_step_*.pt")):
            return d
    return None


def run_replica(parent_ckpt, parent_step, K, snap, data_seed, out_dir):
    """Train ONE replica K steps from parent_ckpt on a disjoint data stream.

    Reuses the validated train.py single-GPU path (NO --diloco, world=1). Saves
    a snapshot every `snap` steps. Returns the run subdir (with the snapshots).
    Skips training if a completed run subdir already exists (resumable).
    """
    out_dir = Path(out_dir)
    existing = newest_run_subdir(out_dir) if out_dir.exists() else None
    if existing is not None and (existing / "DONE").exists():
        log(f"  resume-skip replica (already done): {existing}")
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    target_step = parent_step + K
    keep = K // snap + 3
    cmd = [sys.executable, str(REPO / "train.py"),
           *RECIPE,
           "--resume", str(parent_ckpt),
           "--seed", str(data_seed),
           "--steps", str(target_step),
           "--save_every", str(snap),
           "--keep_checkpoints", str(keep),
           "--output", str(out_dir)]
    log(f"  train replica seed={data_seed} K={K} snap={snap} -> {out_dir}")
    t0 = time.time()
    env = dict(os.environ)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"train.py replica failed rc={proc.returncode}")
    sub = newest_run_subdir(out_dir)
    if sub is None:
        raise RuntimeError(f"no checkpoints produced in {out_dir}")
    (sub / "DONE").write_text(f"K={K} snap={snap} secs={time.time()-t0:.0f}\n")
    log(f"  replica done in {time.time()-t0:.0f}s -> {sub}")
    return sub


def snapshots_by_step(run_subdir):
    """Map parent->{step: (path, train_loss)} for all snapshots in a run dir."""
    out = {}
    for p in Path(run_subdir).glob("checkpoint_step_*.pt"):
        m = CKPT_RE.search(p.name)
        if m:
            out[int(m.group(1))] = (p, float(m.group(2)))
    return out


def sf_aware_merge(replica_ckpts, out_path, args_json_src):
    """Offline SF-aware average of I matched snapshots -> consensus checkpoint.

    Mirrors train.py:diloco_merge local-SGD branch EXACTLY (outer_lr=1,beta=0):
      * model_state_dict (the eval-x weights, saved in optimizer.eval() mode):
        arithmetic mean across replicas (fp32 accumulation, cast back).
      * optimizer base-iterate z: arithmetic mean across replicas.
      * schedule-free clocks (weight_sum,k,lr_max,step) and other state are
        identical across replicas at the same step -> taken from replica 0.
        (exp_avg_sq stays per-rank in DiLoCo; replica-0's is the representative
        for a resumable consensus and is irrelevant to forward/eval bpb.)
    """
    dicts = [torch.load(p, map_location="cpu", weights_only=False)
             for p in replica_ckpts]
    base = dicts[0]
    # 1. average model weights
    msd = base["model_state_dict"]
    for key in msd:
        if not torch.is_floating_point(msd[key]):
            continue
        acc = torch.zeros_like(msd[key], dtype=torch.float32)
        for d in dicts:
            acc += d["model_state_dict"][key].to(torch.float32)
        msd[key] = (acc / len(dicts)).to(msd[key].dtype)
    # 2. average schedule-free base-iterate z
    osd = base["optimizer_state_dict"]
    for pidx, st in osd.get("state", {}).items():
        if "z" not in st or not torch.is_tensor(st["z"]):
            continue
        acc = torch.zeros_like(st["z"], dtype=torch.float32)
        for d in dicts:
            acc += d["optimizer_state_dict"]["state"][pidx]["z"].to(torch.float32)
        st["z"] = (acc / len(dicts)).to(st["z"].dtype)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out_path)
    # eval_checkpoint reads model args from the sibling args.json
    aj = out_path.parent / "args.json"
    if not aj.exists():
        shutil.copy(args_json_src, aj)
    return out_path


def _eval(selector, out_csv):
    out_csv = Path(out_csv)
    cmd = [sys.executable, str(REPO / "scripts" / "eval_checkpoint.py"),
           *selector,
           "--heldout_tensor", HELDOUT,
           "--out", str(out_csv),
           "--y-mode", "train",
           "--batch-size", os.environ.get("HELDOUT_EVAL_BS", "16")]
    env = dict(os.environ)
    env["EVAL_CHECKPOINT_GPU_LEASED"] = "1"   # reuse this campaign's lease
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"eval_checkpoint failed rc={proc.returncode}")
    out = {}
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["step"])] = float(row["bpb"])
            except (KeyError, ValueError):
                pass
    return out


def eval_dir(run_subdir, out_csv):
    """Score every checkpoint in run_subdir offline (one model build). Returns
    {step: bpb}. Reuses the already-leased GPU (no re-lease)."""
    return _eval(["--run-dir", str(run_subdir)], out_csv)


def eval_one(checkpoint, out_csv):
    """Score a SINGLE checkpoint (does not glob the sibling ladder)."""
    return _eval(["--checkpoint", str(checkpoint)], out_csv)


def append_result(row):
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_cell(maturity, K, I, snap, merges, base_seed=1000):
    """One (maturity, K, island-count) cell. Writes barrier(drift) rows."""
    ck, p_step, p_tok = LADDER[maturity]
    if not Path(ck).exists():
        log(f"SKIP cell {maturity} K={K} I={I}: parent ckpt missing {ck}")
        return
    cell = f"{maturity}_K{K}_I{I}_M{merges}"
    cdir = OUTROOT / cell
    log(f"=== CELL {cell}  parent_step={p_step} tokens={p_tok/1e6:.0f}M ===")
    parent_ckpt, parent_step = ck, p_step
    # seed bpb anchor (the parent itself, scored once)
    seed_eval_dir = cdir / "seed_eval"
    seed_eval_dir.mkdir(parents=True, exist_ok=True)
    seed_bpb = None
    try:
        seed_csv = seed_eval_dir / "seed.csv"
        seed_bpb = eval_one(ck, seed_csv).get(parent_step)
    except Exception as e:  # noqa
        log(f"  (seed anchor eval skipped: {e})")

    for merge_round in range(merges):
        rdir = cdir / f"round{merge_round}"
        replica_subdirs = []
        for isl in range(I):
            ds = base_seed + merge_round * 100 + isl
            sub = run_replica(parent_ckpt, parent_step, K, snap, ds,
                              rdir / f"isl{isl}")
            replica_subdirs.append(sub)
        # matched-step snapshots (intersection across islands)
        per_isl = [snapshots_by_step(s) for s in replica_subdirs]
        common = set(per_isl[0])
        for d in per_isl[1:]:
            common &= set(d)
        common = sorted(common)
        log(f"  round{merge_round}: {len(common)} matched snapshots "
            f"steps {common[:3]}...{common[-1:]}")
        # offline eval each replica run-dir once
        repl_bpb = []
        for isl, sub in enumerate(replica_subdirs):
            repl_bpb.append(eval_dir(sub, rdir / f"isl{isl}_bpb.csv"))
        args_json_src = replica_subdirs[0] / "args.json"
        # build consensus dir over all drifts, eval once
        cons_dir = rdir / "consensus"
        cons_dir.mkdir(parents=True, exist_ok=True)
        for step in common:
            ck_paths = [per_isl[isl][step][0] for isl in range(I)]
            sf_aware_merge(
                ck_paths,
                cons_dir / f"checkpoint_step_{step:06d}_loss_0.0000.pt",
                args_json_src)
        cons_bpb = eval_dir(cons_dir, rdir / "consensus_bpb.csv")
        # barrier per drift
        last_cons_step = None
        for step in common:
            drift = step - parent_step
            rbpbs = [repl_bpb[isl].get(step) for isl in range(I)]
            if any(b is None for b in rbpbs) or cons_bpb.get(step) is None:
                continue
            mean_repl = sum(rbpbs) / len(rbpbs)
            barrier = cons_bpb[step] - mean_repl
            append_result({
                "cell": cell, "maturity": maturity,
                "parent_tokens": f"{p_tok:.0f}", "parent_step": parent_step,
                "K": K, "islands": I, "merge_round": merge_round,
                "drift_steps": drift,
                "drift_tokens": drift * TOK_PER_STEP,
                "seed_bpb": f"{seed_bpb:.4f}" if seed_bpb else "",
                "mean_replica_bpb": f"{mean_repl:.4f}",
                "consensus_bpb": f"{cons_bpb[step]:.4f}",
                "barrier_bpb": f"{barrier:.4f}",
                "in_basin": "yes" if barrier <= 0.02 else "NO",
            })
            last_cons_step = step
        # multi-merge: re-seed all islands from the final consensus
        if merges > 1 and last_cons_step is not None:
            parent_ckpt = cons_dir / \
                f"checkpoint_step_{last_cons_step:06d}_loss_0.0000.pt"
            parent_step = last_cons_step
    log(f"=== CELL {cell} COMPLETE ===")


# Ordered cell plan: CHEAP + DECISIVE first, so even a SHORT idle-GPU window (the
# box is saturated with indefinite racers + the outer-mom test) yields a real
# barrier row. Each I=2 K=2000 cell is ~2 windows*2000 steps (~30 min) and spans
# drift {250,500,1000,1500,2000}. (maturity, K, I, snap, merges)
DEFAULT_PLAN = [
    # 0-1: cheap early-vs-mature barrier-vs-drift (the core threshold signal).
    ("176M",  2000, 2, 250, 1),   # very-early seed = most likely to show a barrier
    ("2.0B",  2000, 2, 250, 1),   # mature seed = the safe anchor
    # 2-3: extend the drift axis to K=4000 at the two ends.
    ("176M",  4000, 2, 500, 1),
    ("2.0B",  4000, 2, 500, 1),
    # 4-5: fill the maturity ladder (>=3 maturities).
    ("528M",  2000, 2, 250, 1),
    ("1.06B", 2000, 2, 250, 1),
    # 6-7: island-count modifier (I=4) at the two ends (>=2 island counts).
    ("176M",  2000, 4, 250, 1),
    ("2.0B",  2000, 4, 250, 1),
    # 8-9: multi-merge continuity (does a barrier compound or self-heal?).
    ("176M",  1000, 2, 250, 3),
    ("2.0B",   250, 4, 250, 3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma list of cell indices to run (0-based)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    plan = list(enumerate(DEFAULT_PLAN))
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        plan = [(i, c) for i, c in plan if i in want]
    log(f"held-out tensor: {HELDOUT}")
    if not args.dry_run:
        log(f"leased CUDA_VISIBLE_DEVICES={require_leased_gpu()}")
    log(f"plan: {len(plan)} cells -> {RESULTS_CSV}")
    for i, (maturity, K, I, snap, merges) in plan:
        log(f"--- plan[{i}]: {maturity} K={K} I={I} snap={snap} merges={merges} ---")
        if args.dry_run:
            continue
        try:
            run_cell(maturity, K, I, snap, merges)
        except Exception as e:  # noqa  -- keep the campaign alive across cells
            log(f"!!! cell {maturity} K={K} I={I} FAILED: {e}")
    log("CAMPAIGN DONE")


if __name__ == "__main__":
    main()
