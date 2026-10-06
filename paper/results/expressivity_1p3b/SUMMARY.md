# 1.3B Expressivity on the E97 / GDN2-MLP Pair — Results Summary

Task: `taskA-expressivity` (paper-fanout). Protocol, checkpoint derivation,
and the pre-registered budget: [`PLAN.md`](PLAN.md). Runner:
`scripts/expressivity_1p3b.py`. All numbers below are real measured
accuracies from the committed JSONs (batch 32/64, seed 42, bf16; see the
recipe block in each JSON). Machine-readable files:
`zeroshot_{e97,gdn2}.json`, `ft_{e97,gdn2}_{s5_permutation,s3_permutation}.json`.

Checkpoints (the ones the paper's 1.3B loss-parity claims bind): E97 final
`checkpoint_step_2300930_loss_2.4365.pt` (150.79B tokens) and GDN2-MLP final
`checkpoint_step_2323616_loss_2.4034.pt` (152.28B tokens), both loaded with
Schedule-Free y-mode (train) weight recovery.

---

## Verdict (stated plainly)

**The 8M S5 separation does NOT replicate at 1.3B on this pair.** Under the
matched finetune protocol (identical recipe and budget for both models),
**GDN2-MLP — a linear-recurrent baseline — is at parity or slightly AHEAD of
E97 on S5 at every evaluated length**, including all extrapolated lengths.
Both models learn S5 at trained lengths and both decay under length
extrapolation, with E97 decaying marginally faster. On the solvable S3
control, E97 is ahead (learns faster, generalizes further), so the models are
not interchangeable — but the NC1-witness separation that carries the 8M
result is absent here in the predicted direction.

If the paper's spine needs the S5 expressivity claim, the §6 1.3B slot must
either present this as an honest boundary finding (8M separation +
realizability do not transfer to the CMA-hardened 1.3B pair under a
budget-matched finetune) or fund a larger budget/seed sweep. Zero-shot does
not separate either: both models sit at the random baselines on all five
tasks, as the old paper's frozen-probe finding predicts.

---

## Table 1 — Zero-shot (no finetune), full-vocab argmax accuracy

Task generators and metrics identical to the 8M suite (`mode="running"`).
Both models are at/near the random baseline everywhere; no separation.
Restricted (forced-choice among valid class ids) accuracies, in
`zeroshot_*.json`, are also at baseline and are omitted here.

| Task (baseline) | T | E97 | GDN2-MLP |
|---|---:|---:|---:|
| parity (0.500)   | 512 | 0.385 | 0.496 |
| S3 (0.167)       | 512 | 0.129 | 0.165 |
| **S5 (0.0083)**  | 512 | 0.0085 | 0.0084 |
| mod-counter K=5 (0.200) | 512 | 0.198 | 0.198 |
| FSM K=4 (0.250)  | 512 | 0.249 | 0.249 |

(All four lengths T ∈ {64,128,256,512} show the same picture; see JSONs.
Full-vocab zero-shot accuracy is polluted by out-of-class predictions —
e.g. E97 parity 0.385 < 0.500 baseline means it often predicts a non-class
token — which is why the restricted readout is the cleaner baseline check,
and it is at baseline for every task/model/length.)

## Table 2 — Matched finetune, S5 (headline, 2000 steps each)

Identical recipe: full fine-tune, AdamW (0.9, 0.95), lr 5e-5 const, warmup
300, wd 0.01, grad-clip 0.5, batch 32, train lengths {16,32,48,64}, seed 42.

| T | E97 | GDN2-MLP | 8M reference (E88 / GDN, from-scratch 20K) |
|---:|---:|---:|---|
| 16 | **1.0000** | 0.9998 | — |
| 32 | 0.9745 | **0.9872** | — |
| 48 | 0.8486 | **0.9187** | — |
| 64 | 0.7108 | **0.7869** | 0.79 / 0.36 @T128 |
| 96 | 0.4946 | **0.5578** | — |
| 128 | 0.3817 | **0.4187** | — |
| 192 | 0.2552 | **0.2860** | — |
| 256 | 0.2002 | **0.2141** | 0.42 / 0.18 |
| 384 | 0.1335 | **0.1478** | — |
| 512 | 0.1058 | **0.1119** | 0.22 / 0.10 |
| 768 | 0.0732 | **0.0781** | — |
| 1024 | 0.0561 | **0.0595** | 0.11 / 0.05 |

GDN2-MLP ≥ E97 at every length (bold = ahead; both far above the 0.0083
baseline). Trajectory (T=64): GDN2-MLP 0.33 → 0.55 → 0.70 → 0.79 at
steps 500/1000/1500/2000; E97 0.35 → 0.54 → 0.59 → 0.71. **Both models were
still climbing when the pre-registered budget ended**; neither plateaued.
The 8M column shows the 8M from-scratch reference ordering (E88 ≫
FLA-GDN) that this pair/budget does not reproduce.

## Table 3 — Matched finetune, S3 solvable control (750 steps each)

| T | E97 | GDN2-MLP |
|---:|---:|---:|
| 16 | **1.0000** | **1.0000** |
| 32 | **1.0000** | 0.9979 |
| 48 | **0.9993** | 0.9757 |
| 64 | **0.9993** | 0.9122 |
| 96 | **0.9794** | 0.7488 |
| 128 | **0.9242** | 0.6196 |
| 192 | **0.7596** | 0.4651 |
| 256 | **0.6727** | 0.3919 |
| 384 | **0.5142** | 0.3177 |
| 512 | **0.4370** | 0.2799 |
| 768 | **0.3391** | 0.2421 |
| 1024 | **0.2927** | 0.2208 |

E97 learns the solvable-group control faster (0.999 vs 0.912 at T=64 within
750 steps) and generalizes further at every extrapolated length.

## What this means for the paper

1. **The theory→empirics loop does not close in the predicted direction at
   1.3B on this pair.** The Lean results (§7) establish realizability and
   per-step expressivity of the delta-correcting update; they do not predict
   that every 1.3B SGD-trained network with a linear-state mixer fails S5.
   The 8M S5 result is an inductive-bias-under-matched-optimization finding
   at 8M; at 1.3B, with 150B-token pretrained weights, 12–17 layers, convs,
   MLPs, and a 50k-token vocabulary, both mixers find S5-tracking solutions
   within 2000 matched steps. The 8M "linear state cannot track S5" empirical
   separation is thus **scale/pairing-conditional**, and this suite is the
   direct evidence for that boundary.
2. **GDN2-MLP is qualitatively stronger on these probes than the FLA-GDN
   baseline the old §6 used.** Old-pair 1.3B symmetric-budget (24,000 steps):
   GDN reached only 0.117 @T=64 and 0.022 @T=512 on S5. This CMA-hardened
   GDN2-MLP reaches 0.787 @T=64 in only 2,000 steps. Any §6 claim must name
   the baseline precisely; "GDN" ≠ "GDN2-MLP".
3. **E97 retains a real edge on the solvable control** (S3, all lengths) —
   the models differ in-state-tracking *speed and length robustness*, but the
   NC1-witness task itself does not separate them at this scale/budget.
4. **Parity finetune was not run** (GPU timebox); zero-shot parity is at
   baseline for both. Modular counter and FSM likewise have zero-shot numbers
   only. These gaps are reported, not smoothed over.

## Caveats (all of them)

- Single seed (42), matching the prior 1.3B protocol's per-model single seed;
  no seed variance available.
- 2000/750 steps is ~1/12 of the 8M from-scratch budget and ~1/12 of the
  old-pair 24K symmetric budget; both S5 curves were still rising at the end.
  A larger matched budget is the single most obvious follow-up (it would not
  be S5-tuning only if raised for BOTH models symmetrically and a priori).
- The finetune budget was fixed from measured throughput before any
  accuracy-bearing result existed (see PLAN.md §3 for the adjustment rule
  actually applied: 3000→2000 S5, 1500→750 S3, equally for both models).
- The task-token interface is the raw 8M-suite id space (ids 0..n−2 in,
  full-vocab argmax must equal the class id), identical to the prior 1.3B
  protocol; zero-shot under this interface is a category mismatch and is
  reported only for completeness.
- GPU usage: ~95 GPU-minutes total (zero-shot ≈ 5, loader smoke/probes ≈ 10,
  finetunes 80.3), wall ≈ 75 min, all under `scripts/gpu_lease.sh` leases,
  released after each phase.

## Files

| File | Content |
|---|---|
| `PLAN.md` | Pre-registered plan, checkpoint derivation, budget rule |
| `e97_args.json`, `gdn2_args.json` | Frozen training args of the bound checkpoints |
| `zeroshot_e97.json`, `zeroshot_gdn2.json` | Zero-shot, 5 tasks × 4 lengths, full-vocab + restricted |
| `ft_e97_s5_permutation.json` | S5 matched finetune, E97 (recipe, train curve, trajectory, eval grid) |
| `ft_gdn2_s5_permutation.json` | S5 matched finetune, GDN2-MLP |
| `ft_e97_s3_permutation.json` | S3 control, E97 |
| `ft_gdn2_s3_permutation.json` | S3 control, GDN2-MLP |
