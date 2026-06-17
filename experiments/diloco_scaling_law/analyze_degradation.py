#!/usr/bin/env python3
"""DiLoCo island-count x seed-quality degradation analysis.

Reads the reference single-GPU held-out curve and each DiLoCo cell's scored
checkpoints, recomputes the TRUE matched total-token budget for every point
(eval_checkpoint.py does NOT propagate _world_size into model_args, so its
`tokens` column is per-replica, not total), interpolates the reference BPB at
each cell point's matched token budget, and reports:

    degradation(cell, T) = consensus_bpb(T) - reference_bpb(T)

Total-token convention (matches the task: consensus vs single-GPU at SAME total
tokens). bs*chunk = 4*2048 = 8192 tokens per optimizer step per replica.

  * from-scratch cell (no --resume):   total = step * 8192 * W
  * from-seed cell (--resume @ S0):     total = S0*8192 + (step - S0)*8192*W
    (the seed phase was single-GPU; only the DiLoCo phase is x W islands)

Reference run is world_size=1, no resume -> eval_checkpoint tokens column is
already correct (= step*8192).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BS_CHUNK = 4 * 2048  # 8192 tokens / step / replica (frozen recipe)


def read_curve(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split", "primary") != "primary":
                continue
            rows.append({"step": int(r["step"]), "bpb": float(r["bpb"]), "ce": float(r["ce"])})
    return sorted(rows, key=lambda x: x["step"])


def total_tokens(step: int, world: int, seed_step: int | None) -> int:
    if seed_step is None:           # from-scratch
        return step * BS_CHUNK * world
    # from-seed: single-GPU seed phase + x-W DiLoCo phase
    return seed_step * BS_CHUNK + (step - seed_step) * BS_CHUNK * world


def interp(curve_T_bpb, T):
    """Linear interpolation of reference bpb at total tokens T (clamped)."""
    pts = sorted(curve_T_bpb)
    if T <= pts[0][0]:
        return pts[0][1], "clamp-low"
    if T >= pts[-1][0]:
        return pts[-1][1], "clamp-high"
    for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            f = (T - t0) / (t1 - t0)
            return b0 + f * (b1 - b0), f"interp[{t0/1e6:.0f}M,{t1/1e6:.0f}M]"
    return pts[-1][1], "clamp-high"


def main():
    ref_rows = read_curve(HERE / "reference_curve.csv")
    # reference world_size=1, no resume -> tokens = step*8192 (correct already)
    ref_T_bpb = [(r["step"] * BS_CHUNK, r["bpb"]) for r in ref_rows]
    print("=== REFERENCE single-GPU held-out curve (clean disjoint p50k_2048) ===")
    for T, b in ref_T_bpb:
        print(f"  {T/1e6:7.1f}M tok   BPB {b:.4f}")
    ref_bpb_band = max(b for _, b in ref_T_bpb) - min(b for _, b in ref_T_bpb)
    print(f"  reference noise band (max-min over curve): {ref_bpb_band:.4f} BPB\n")

    # cell registry: (csv, world, seed_step, label)
    cells = [
        ("stab_scratch_i4_curve.csv", 4, None,   "S0 scratch / I=4"),
        ("swell_i2_curve.csv",        2, 64500,  "S_well(528M) / I=2"),
        ("swell_i4_curve.csv",        4, 64500,  "S_well(528M) / I=4"),
        ("swell_i4_mom_curve.csv",    4, 64500,  "S_well(528M) / I=4 +mom0.9"),
        ("swell_i2_mom_curve.csv",    2, 64500,  "S_well(528M) / I=2 +mom0.9"),
        # diloco-seed-race I=6: seeded from the LATEST emender ckpt (step 129000,
        # ~1.057B per-replica tok), W=6 plain averaging (outer_beta=0), ~3B total race.
        ("swell_i6_curve.csv",        6, 129000, "S_mature(1.057B) / I=6"),
    ]
    print("=== DEGRADATION (consensus_bpb - reference_bpb at matched total tokens) ===")
    summary = []
    for csv_name, world, seed_step, label in cells:
        rows = read_curve(HERE / csv_name)
        if not rows:
            continue
        print(f"\n-- {label}  [{csv_name}] --")
        for r in rows:
            T = total_tokens(r["step"], world, seed_step)
            ref_b, how = interp(ref_T_bpb, T)
            deg = r["bpb"] - ref_b
            flag = " (WITHIN ref noise)" if abs(deg) <= ref_bpb_band else ""
            print(f"  step {r['step']:6d}  total {T/1e6:7.1f}M  BPB {r['bpb']:.4f}  "
                  f"ref~{ref_b:.4f} [{how}]  deg {deg:+.4f}{flag}")
            summary.append({"cell": label, "world": world, "seed_step": seed_step,
                            "step": r["step"], "total_tokens": T, "bpb": r["bpb"],
                            "ref_bpb": round(ref_b, 4), "degradation": round(deg, 4)})
    (HERE / "degradation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote degradation_summary.json ({len(summary)} points); "
          f"reference noise band = {ref_bpb_band:.4f} BPB")


if __name__ == "__main__":
    sys.exit(main())
