<!-- DRAFT SKELETON — numbers filled in from experiments/emender_real_cap/AGG_TABLES.md after the batteries + LM cohort complete. Do not cite until the PRELIMINARY tags are removed. -->
# The REAL Emender — sparse nonlinear-emendment sprinkle vs pure GDN-2 (matched precision)

**Task:** `emender-real-cap` · **Hardware:** 8× RTX 6000 Ada · bf16, fused Triton (loud no-eager guard) · REAL data/algorithms, no mocks.

## 0. What "the REAL Emender" is (and why opt-1p3b was a strawman)

The prior `opt-1p3b` arm was **50% nonlinear heads, fp32, ~4.3× token-starved** vs the bf16
controls — not the Emender thesis. The REAL Emender is:

- a **SEA of `gdn2_recall`** heads (GDN-2 delta-memory, `allow_neg_eigval` on → recall+track), plus
- a **SMALL fraction** (6.25% / 12.5%) of **nonlinear EMENDMENT heads**. The capability-adding
  emendment head is the **`e97_delta` split-edit cell with a per-step bounded (`tanh`) state map**
  — *not* the throughput-friendly `gdn2_nonlin_shell` (phi-explore: bounded saturation unlocks the
  depth capability on **split-edit**; it is "nearly inert on gated-delta"/shell).
- **bf16**, matched to the GDN-2 control (no token starvation), **fused Triton only** (asserted).

## 1. Throughput @ 1.3B head shape (the central tension)

<!-- from throughput.json -->
The capability head (`e97_delta`-tanh split-edit) is **SEQUENTIAL** (non-chunkable: `tanh ⊥ chunkable`).
Stream overlap hides part of it under the chunkable GDN-2 bulk, but cannot reach 0.95×:

| config | ratio vs GDN-2 |
|---|---|
| emender 4/64 (e97_delta-tanh, overlap) | **0.75×** |
| emender 8/64 (e97_delta-tanh, overlap) | 0.73× |
| shell 4/64 (gdn2_nonlin_shell — capability-INERT) | (0.95× in HETERO_KERNEL_NOTE) |

**The throughput-friendly head (shell, 0.95×) does NOT add capability; the capability head
(split-edit, 0.73–0.75×) is NOT throughput-friendly.** This reproduces
[hetero-cma-knee] (split-edit = 0.73× GDN-2). The "~same speed" premise of the thesis is **not met**
at matched precision: the real Emender costs ~25–27% throughput.

## 2. Expressivity battery — does the sprinkle add capability GDN-2 can't reach?

Two scales, **same sparse fractions** (6.25% / 12.5%); head_dim=8, n_state=32 identical, only the
SEA size differs:
- **small** dim256 nh32 (2/32, 4/32) — the documented separation regime.
- **large** dim512 nh64 (4/64, 8/64) — the literal task fractions / 1.3B head count.

<!-- from AGG_TABLES.md: per-task tables + separations -->
### Scale story
- (PRELIMINARY) At the **large** sea size, pure GDN-2 already **saturates** modular_quadratic
  (≈1.00 incl. T=512), leaving no separation headroom on the primary cliff task.
- (PRELIMINARY) The separation, if any, concentrates where GDN-2 has extrapolation headroom
  (modular_counter, s5) — TBD from the small-regime battery.

## 3. Convergent-loss tie — held-out BPB (REAL Comma-Pile, bf16 matched)

<!-- from results_lm: matched-param control gdn2typed vs emender; token + wall matched -->
- **Matched-param control** = `gdn2typed` (typed all-`gdn2_recall`, same path/dim/heads as the
  Emender, differs ONLY in 4 head types). `gdn2` (fla-gdn) is the external incumbent reference.

| arm | params(M) | held-out BPB | tokens | tok/s |
|---|---|---|---|---|
| gdn2typed (matched control) | TBD | TBD | TBD | TBD |
| emender 4/64 | TBD | TBD | TBD | TBD |

## 4. Verdict + best fraction for the 1.3B run

<!-- final synthesis -->
- Best sparse fraction for 1.3B: **TBD** (4/64 favored on throughput if it already carries the
  capability + ties on loss).
- Honest verdict: **TBD** — does a sparse capability sprinkle TIE on loss AND add reachable
  expressivity at ~same speed? (Throughput premise already challenged in §1.)

## Reproduction
- Throughput: `python experiments/emender_real_cap/throughput.py`
- Expressivity: `python experiments/emender_real_cap/run_cap_battery.py --scale {small,large}`
- Convergent-loss: `python experiments/emender_real_cap/run_lm_cohort.py --train_minutes 12`
- Aggregate: `python experiments/emender_real_cap/aggregate.py`
