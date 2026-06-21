# fix-long-horizon — RESULTS (measured)

Follow-up to `diloco-loss-parity-longhorizon`, which discovered that the held-out BPB of
the `emender-mlp` 1.3 B recipe **collapses over a long horizon for BOTH the DDP baseline
and DiLoCo** — not a parallelism bug but a **training-recipe** bug: schedule-free AdamW at
a constant CMA-tuned LR≈1e-3 with `warmup_steps=0` and no decay. This task fixes the
recipe and re-confirms the DiLoCo-vs-DDP gap against a **non-degrading** baseline.

**All numbers are REAL**, measured on the 8×RTX 6000 Ada box (49 GB, PCIe, NO NVLink),
commapile_mainmix 1 TB, p50k_base, ctx 2048, bf16 + FUSED Triton, emender-mlp
**1,286,589,072 params** (dim1792 nh216 ns32 dep11 mlp2.2623), bs6, seed 42. Held-out BPB
= the **same** agent-1433 preflight tensor used by the predecessor
(`heldout_comma_p50k_2048.pt`, bytes/token 3.938, 65 536 tokens), evaluated on the saved
checkpoint (DDP: synchronized weights; DiLoCo: post-merge consensus; schedule-free: the
eval/x-averaged weights via the save-time `optimizer.eval()` swap — same metric as the
predecessor). 7-GPU global tokens/step = 6·2048·7 = **86 016**.

Reproduce:
```
bash experiments/diloco_100b/longhorizon_fix/run_arm_live.sh ddp_cos /mnt/nvme1n1/erikg/lh_fix_ddpcos 2500 250 0.001 --optimizer adamw --min_lr_frac 0.1
bash experiments/diloco_100b/longhorizon_fix/run_arm_live.sh dil_cos /mnt/nvme1n1/erikg/lh_fix_dilcos 2500 250 0.001 --optimizer adamw --min_lr_frac 0.1 --diloco --diloco_k 250 --diloco_outer_beta 0.0 --diloco_outer_lr 1.0
python3 experiments/diloco_100b/longhorizon_fix/analyze.py experiments/diloco_100b/longhorizon_fix/results.txt
python3 tests/test_lr_schedule.py
```

---

## HEADLINE

1. **The bug is the constant-LR schedule, not warmup alone.** With schedule-free at a
   constant LR, held-out BPB on the eval (x-average) weights **rolls over** mid-run while
   the train (y) loss keeps falling. Adding the (previously-unplumbed) warmup AND halving
   the LR to 5e-4 only **postpones** the rollover; it does not remove it. A real **LR
   decay** is required.
2. **AdamW + linear warmup + cosine decay is monotone and far better.** 7-GPU DDP to
   215 M tokens: held-out BPB **strictly decreases** 1.872 → … → **1.205**, beating the
   broken recipe's *global minimum* (1.571) by step 750 and its 215 M endpoint (3.234) by
   **−2.03 BPB**. No mid-run rise.
3. **DiLoCo-vs-DDP gap re-confirmed against the healthy baseline** (§3): the periodic-merge
   penalty is a **persistent ~+0.35 BPB that does not close** (stable +0.35 plateau from 64.5 M through 215 M against the healthy baseline).

---

## 0. The two plumbing bugs fixed in `train.py`

- **`warmup_steps` never reached the schedule-free optimizer.** `schedulefree.AdamWSchedule
  Free(...)` was constructed without `warmup_steps`, so the SCALE_PLAN §2.2 "add a 2–5 k
  warmup" instruction was a silent no-op (the optimizer ran at `warmup_steps=0`). Fixed:
  pass `warmup_steps=args.warmup_steps`. Unit-guarded by `tests/test_lr_schedule.py::
  test_schedulefree_receives_warmup`.
- **`get_lr` cosine schedule was malformed.** It used `step/warmup_steps` as the cosine
  phase → LR oscillated with period `2·warmup_steps` and collapsed to `min_lr` right after
  warmup, and it never referenced the total step count. Replaced by
  `lr_scale_at(step, warmup_steps, total_steps, min_lr_frac)`: linear warmup, then a single
  cosine decay from peak to `min_lr_frac·peak` over the full run, scaling each param
  group's base LR (so `--knob_lr_mult` ratios are preserved). New `--min_lr_frac` flag
  (default 0.1). Unit-guarded by `test_warmup_ramps_linearly`,
  `test_cosine_decays_monotone_to_floor`, `test_per_group_ratio_preserved`.

---

## 1. The recipe bug: constant-LR schedule-free rolls the held-out average over

7-GPU DDP, emender-mlp 1.286 B, to 215 M tokens (2500 steps), held-out BPB every 250 steps.
Three recipes (broken numbers are the predecessor `longhorizon/RESULTS.md`; the two
corrected-recipe curves are measured here):

| tokens (step) | **broken** SF const lr=1e-3 wu=0 | SF wu=500 lr=5e-4 | **AdamW wu=250 lr=1e-3 cos→0.1** |
|---|---:|---:|---:|
| 21.5 M (250)  | 1.770 | 2.135 | 1.872 |
| 43.0 M (500)  | 1.619 | 1.822 | 1.647 |
| 64.5 M (750)  | **1.571** (min) | 1.696 | **1.523** |
| 86.0 M (1000) | 1.596 ↑ | 1.653 (min) | 1.431 |
| 107.5 M (1250)| 1.690 ↑ | 1.657 | 1.360 |
| 129.0 M (1500)| 1.858 ↑ | 1.697 ↑ | 1.306 |
| 150.5 M (1750)| 2.112 ↑ | 1.777 ↑ | 1.264 |
| 172.0 M (2000)| 2.456 ↑ | — | 1.231 |
| 193.5 M (2250)| 2.844 ↑ | — | 1.213 |
| 215.0 M (2500)| **3.234 ↑** | — | **1.205** |

- **Broken** (constant LR=1e-3, no warmup): bottoms at 1.571 @ 64.5 M then climbs
  monotonically to 3.234 @ 215 M. Train CE keeps falling the whole time (→ 3.4) — the
  degradation is in the schedule-free *eval (x) average*, not the train iterate.
- **SF warmup=500 + lr halved to 5e-4** (measured here, killed at 1750 once the rollover
  was confirmed): the warmup + lower LR **postpone** the rollover (min 1.653 @ 86 M vs the
  broken 1.571 @ 64.5 M) and slow the rise, but it **still rolls over** (1.653 → 1.777 by
  150 M). ⇒ warmup alone, even with a halved constant LR, does not fix it.
- **AdamW + warmup + cosine decay** (measured here): **strictly monotone** to 1.205 @
  215 M. Crosses below the broken recipe's *global min* (1.571) by step 750 and never
  rises. **This is the corrected recipe.**

Mechanism: the schedule-free x-average is a (lr-weighted) Polyak average of the y
iterates; under a sustained high constant LR the y trajectory's stationary noise floor is
high and the average sits in a progressively worse-generalizing region of the high-LR
basin. Decaying the LR collapses that noise floor, so the late iterates (and any average
of them) land in a sharp low-loss region — held-out improves monotonically. Warmup only
fixes the *early* transient, which is why it merely postpones the late rollover.

---

## 2. Validation item 2 — monotonic improvement below the broken baseline ✅

`analyze.py` flags the corrected DDP arm **MONOTONIC (no mid-run rise)**, min/endpoint
**1.205 @ 215 M**, vs the broken recipe min 1.571 / endpoint 3.234. Requirement met:
the corrected held-out BPB monotonically improves and reaches **below** (in fact far
below) the broken recipe baseline.

---

## 3. DiLoCo-vs-DDP matched-token gap, re-confirmed on the HEALTHY baseline

The predecessor measured the DiLoCo penalty against a baseline that was itself collapsing
past ~64 M, so the "gap" was confounded (it appeared to "close" to ~0 at 215 M only
because DDP collapsed to meet DiLoCo — *mutual collapse, not parity*). Here both arms use
the **same healthy recipe** (AdamW wu=250 lr=1e-3 cos→0.1), so the gap isolates the
DiLoCo periodic-merge penalty. DiLoCo β=0 (local-SGD), K=250; held-out BPB of the
post-merge consensus checkpoint vs DDP at matched tokens:

| tokens (step) | **DDP** (healthy) | **DiLoCo β=0** K=250 | gap (DiLoCo−DDP) |
|---|---:|---:|---:|
| 21.5 M (250)  | 1.8718 | 2.3126 | +0.441 |
| 43.0 M (500)  | 1.6471 | 2.0707 | +0.424 |
| 64.5 M (750)  | 1.5226 | 1.8804 | +0.358 |
| 86.0 M (1000) | 1.4310 | 1.7802 | +0.349 |
| 107.5 M (1250)| 1.3602 | 1.7072 | +0.347 |
| 129.0 M (1500)| 1.3058 | 1.6557 | +0.350 |
| 150.5 M (1750)| 1.2636 | 1.6099 | +0.346 |
| 172.0 M (2000)| 1.2314 | 1.5842 | +0.353 |
| 193.5 M (2250)| 1.2128 | 1.5691 | +0.356 |
| 215.0 M (2500)| **1.2049** | **1.5620** | **+0.357** |

- **Both arms are now MONOTONIC** the whole way (DiLoCo 2.313 → 1.562; DDP 1.872 → 1.205) —
  the recipe fix works for DiLoCo too. `analyze.py` flags both **MONOTONIC (no mid-run
  rise)**.
- **The DiLoCo matched-token penalty is a persistent ~+0.35 BPB that does NOT close.**
  After the warmup transient (+0.44 → +0.42 in the first 43 M) the gap settles to a
  **dead-flat +0.35** plateau from 64.5 M through 215 M (+0.358, +0.349, +0.347, +0.350,
  +0.346, +0.353, +0.356, +0.357). This is the clean measurement the predecessor could not
  make: against a **non-degrading** baseline the gap is stable and **does not shrink** — the
  predecessor's apparent "closing to ~0 at 215 M" is now confirmed to have been **mutual
  collapse** of the broken recipe, not parity.
- The magnitude is a bit lower than the predecessor's ~0.44–0.47 (measured under the broken
  schedule-free recipe); under AdamW+cosine it is ~0.35. The **qualitative verdict is
  unchanged**: plain local-SGD DiLoCo carries a real, persistent matched-token loss penalty
  that does not close with more tokens.

---

## 4. Verdict + scale-plan implication

- **Recipe fix (binding blocker from the predecessor): RESOLVED.** The 100 B seed run must
  use **AdamW + linear warmup + cosine decay to a small floor** (not constant-LR
  schedule-free). Measured monotone to 1.205 BPB @ 215 M at the `emender-mlp` geometry;
  scale `warmup_steps` to ~1–2 % of the real step budget. `docs/SCALE_PLAN.md` §2.2
  updated.
- **DiLoCo:** is a **persistent ~+0.35 BPB that does not close** (stable +0.35 plateau from 64.5 M through 215 M against the healthy baseline) (full table §3). The predecessor's not-a-viable-loss-parity-path
  conclusion therefore **stands, now re-confirmed against a non-degrading baseline** — the earlier "gap closes at 215 M" was mutual collapse, not parity.
- The schedule-free path is **not** retired in principle (the E88 0.966 precedent used it),
  but it is **unsafe at this geometry/LR** without decay; if horizon-freedom is needed
  later, re-validate schedule-free monotonicity at a much lower constant LR or use a
  WSD/trapezoidal schedule.
