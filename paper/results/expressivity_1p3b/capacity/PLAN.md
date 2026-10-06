# Task A2 — 1.3B S5 Capacity-Exceeded / Generalization-Regime Experiment — Plan

Task: `taskA2-capacity` (paper-fanout). Follow-up to Task A
(`../PLAN.md`, `../SUMMARY.md`): the matched 2k-step budget at train lengths
{16..64} left both models still descending — no plateau, memorization never
exhausted, so the E97-vs-GDN2 separation was never tested in the regime the
operator directive (2026-10-06) demands: *"learning long enough and capacity
exceeded enough that GDN2 can't just memorize the result."*

Runner: `scripts/expressivity_1p3b_capacity.py` (derives loaders/protocol from
`scripts/expressivity_1p3b.py`; same checkpoints E97 `step_2300930` /
GDN2-MLP `step_2323616`, same Schedule-Free y-mode recovery, same seed 42,
same lr recipe: AdamW (0.9,0.95), lr 5e-5 const, warmup 300, wd 0.01,
grad-clip 0.5, batch 32, bf16, full fine-tune).

## 1. Design (decided before any result; matched across both models)

1. **Long-horizon training at T=256** (single trained length; 256 is a
   multiple of 16 for the E97 Triton `checkpoint_interval=16` constraint).
   Train until *validation loss genuinely plateaus*, with a hard step cap
   derived ONLY from measured throughput (§2). Full train/val trajectories
   are logged — the memorization-plateau vs continued-descent shapes ARE the
   data.
   - Plateau rule (pre-registered): val loss probes every 100 steps on a
     fixed val set (4 batches × 32 @ T=256, seen-pair positions only, seed
     42-derived). Plateau = relative improvement over the trailing 300 steps
     < 1% AND over the trailing 200 steps < 1%, evaluated only at steps ≥
     `min_steps`; OR val loss < 1e-4 (memorized — no signal left).
   - `min_steps = min(1000, cap)` so the budget is never dominated by an
     early accidental plateau.
   - Curriculum escape (the task's "T=256->512 if loss stalls"): if the
     plateau fires with val loss > 0.01 while < 60% of the cap is consumed,
     continue training at T=512 until a fresh plateau (≥ 400 more steps) or
     the cap. If this fires, 512 becomes a *trained* length and is flagged
     out of the held-out sweep set.
2. **Length extrapolation sweep at convergence** (never-trained lengths):
   T ∈ {512, 1024, 2048, 4096, 8192}, plus the trained length(s) for
   reference. Full-vocab argmax accuracy (identical metric/interface to Task
   A). Batch chosen so B×T ≤ 32768 (min 4), 8 batches per length, OOM-failover
   by halving batch. Same eval seeds for both models.
3. **Held-out structure split**: 20% of the 480 S5 (state, generator)
   transition pairs — 96 pairs — chosen by `numpy.random.default_rng(42)`
   permutation of ids `prev_state*4 + gen`. Supervision on held-out-pair
   positions is REMOVED from the training loss (inputs still flow through
   them; only the targets are masked). At eval, accuracy is broken out as
   seen-pair vs held-out-pair at every length. Pattern-matching cannot cover
   unseen structure; composition can.

## 2. Hard cap derivation (throughput only, fixed before any result)

Budget: ~4 GPU-hours total across both models = 14400 GPU-seconds,
including load, probes, final sweeps, and the throughput smoke itself.

1. Smoke: 20 training steps at T=256, batch 32, per model (under lease);
   take the median s/step of steps 5..20 (excludes load/compile outliers).
2. `cap_steps = int(0.95 * (14400 - 1500_reserve) / (s_E97 + s_GDN2))`,
   identical for both models. Reserve 1500 s covers: 2× checkpoint load
   (~2 min), 2× final sweep (~4-5 min), probe overhead, smoke slack.
3. If the plateau fires earlier, the run stops earlier and releases the GPU.

GPU discipline: `scripts/gpu_lease.sh acquire 1 --wait` per model wrapper
(one GPU per model, parallel), released at wrapper exit per phase. No
squatting.

## 3. Outputs

`paper/results/expressivity_1p3b/capacity/`:

- `budget.json` — measured throughput + cap derivation (this section's rule,
  applied).
- `cap_<model>_s5_train.json` — phase=train: split spec, recipe, cap,
  train/val curves, trajectory probes, plateau verdict, stage info, timing.
- `cap_<model>_s5_sweep.json` — phase=length_extrapolation: acc vs T
  (overall / seen / held-out) at trained + held-out lengths.
- `cap_<model>_s5_heldout.json` — phase=heldout_structure: seen vs held-out
  accuracy at trained lengths, deltas.
- `SUMMARY.md` — honest verdict: accuracy-vs-length table for both models,
  plateau evidence, held-out split deltas. Negative results verbatim; either
  outcome is a paper result (separation in the capacity-exceeded regime
  closes the operator's loop; parity there is a boundary finding).

## 4. If the cap hits before plateau

Report plainly: trajectory shapes at cutoff (steepest-descent slopes,
length-extrapolation curves still discriminate the hypotheses) — per task
brief §6.
