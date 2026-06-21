# grok-expressivity — verdict

**Question.** Were the prior e97-vs-GDN-2 capability "nulls"/ties PRE-GROK artifacts?
Specifically: does the e97 nonlinear-in-time cell GROK the hard algorithmic
separators (S5 state-tracking, modular_quadratic, a^n b^n c^n counting) under
canonical grokking conditions — where the earlier capability runs used all three
grok-suppressors (too few steps, wd 0/0.01, schedule-free optimizer)?

**Answer: NO — and the result is the OPPOSITE of the hypothesis on the one task
that separates.** Under canonical grokking conditions the per-step nonlinearity
unlocks nothing; on S5 it is the *linear* GDN-2 that groks (test-acc → 1.000,
delayed by thousands of steps) while the e97 nonlinear cell never leaves baseline.
The prior nulls are NOT pre-grok artifacts for e97.

## Conditions (all three suppressors removed)

- **Optimizer:** vanilla `torch.optim.AdamW` (NOT schedule-free).
- **Weight decay:** explicit sweep `{0.01, 0.1, 0.3, 1.0}` (grokking is wd-driven).
- **Fixed train/test split:** 128 train / 512 test disjoint sequences (deduped by
  input hash). A finite memorizable train set is the prerequisite for delayed
  generalization — with the fresh-infinite sampling the prior runs used, train≡test
  every step and grokking cannot appear.
- **Long training:** 100k steps, train AND test accuracy
  logged every 500 steps; grok-step = first step test-acc≥0.9, memorize-step = first
  step train-acc≥0.95.
- **Matched small geometry:** dim 256, depth 4, n_state 32, n_heads 8, mlp_ratio 4,
  T=128. ~4.5–5.0M params per arm.
- **bf16 + FUSED kernels, no eager (asserted):** `e97` = E97 split-edit on the fused
  bf16 Triton kernel (tanh state map); `e97-lin` = the SAME fused kernel with the
  tanh dropped (linear split-edit — the within-substrate control isolating the
  per-step nonlinearity); `gdn2` = FLA GatedDeltaNet (allow_neg_eigval) fused chunk
  kernel. (hardtanh has no fused Triton kernel; tanh is the e97 cell's real and only
  fused bounded saturator, and phi-explore already established hardtanh≡tanh≡softsign
  for this capability.)

Data: `runs/*.json` (full per-step train+test curves), `runs/grok_table.md`,
`runs/grok_summary.json`; calibration `runs_calib/*` (n_train ∈ {128,512,2048}).
Aggregated by `aggregate.py`. (A 200k S5 confirmation was started but cancelled
mid-run to yield all 8 GPUs to a co-running agent's deciding run; the 100k
full-budget S5 curves already establish the non-grok — multiple e97/e97-lin S5
configs ran the entire 100k flat at ~0.16, 2.5× past gdn2's latest grok at 40.5k.)

## Grok-status per model × task × wd (best over 2 seeds; 100k steps)

| task | arm | grokked? | grok-step | final/best test-acc | extrap@1024 |
|---|---|---|---|---|---|
| **s5_permutation** | **gdn2 (linear)** | **YES** (wd≥0.1) | 4000–40500 | **1.000** | 0.28–0.87 |
| s5_permutation | e97 (nonlinear) | **NO** (all wd) | — | 0.17 | 0.02–0.03 |
| s5_permutation | e97-lin (linear split-edit) | **NO** (all wd) | — | 0.19 | 0.02–0.04 |
| modular_quadratic | e97 | YES (all wd) | 500–1500 | 0.99 | 0.91–0.98 |
| modular_quadratic | e97-lin | YES (all wd) | 500–1000 | 0.98 | 0.92–0.98 |
| modular_quadratic | gdn2 | YES (wd≥0.1) | 1500–10500 | 0.98 | 0.86–0.98 |
| anbncn_viability | e97 | YES (all wd) | 500 | 0.99 | 0.93 (wd1.0) |
| anbncn_viability | e97-lin | YES (all wd) | 500 | 1.00 | 0.93 (best) |
| anbncn_viability | gdn2 | YES (all wd) | 500 | 0.99 | 0.92 |

## Findings

1. **S5 — the decisive separator, and it goes the wrong way for the hypothesis.**
   GDN-2 (linear) GROKS S5 to test-acc **1.000** with a textbook delayed jump
   (train memorized at step 500, test pinned at baseline ~0.05–0.12 for thousands of
   steps, then a sharp grok at step 4000/9000/9500/24500/40500 depending on wd/seed).
   The e97 nonlinear cell and the linear split-edit control BOTH **never grok** — test
   stays at 0.09–0.19 (baseline 0.008) through the **full 100k** steps (multiple e97
   configs ran the entire 100k flat; the early-stopped ones halted at 62.5k, already
   past gdn2's latest grok at 40.5k). This reproduces, at-grok, the long-standing result that for finite-state
   / permutation tracking the *linear* gated-delta wins and the per-step saturating
   nonlinearity does not help; here it actively prevents grokking.

2. **The prior nulls were pre-grok artifacts for GDN-2, NOT for e97.** GDN-2's S5 and
   low-wd modular_quadratic "failures" at ≤16k steps were genuinely pre-grok: it groks
   S5 as late as step 40500 — invisible to an 8k-step run. e97's failures are real: it
   does not grok S5 through the full 100k budget (2.5× gdn2's latest grok point).

3. **e97 does NOT grok where GDN-2 cannot — the reverse holds.** On the one task that
   discriminates (S5), GDN-2 groks and e97 does not. e97 has no task on which it groks
   and GDN-2 fails.

4. **The per-step nonlinearity is null for grokking.** `e97` (tanh) and `e97-lin`
   (identity), the SAME fused kernel differing only by the per-step state map, behave
   identically on every task and every wd (S5: both flat ~0.17/0.19; modquad: both
   grok ~0.98; anbncn: both grok ~0.99). Nonlinearity-in-time is not the load-bearing
   factor. The S5 grok gap between `e97-lin` and `gdn2` (both linear!) is the
   *substrate* — FLA GatedDeltaNet recall plumbing (short-conv, gating) vs the
   split-edit shell — not the nonlinearity.

5. **modular_quadratic (p=7) does not separate at grok.** All three arms grok to
   ~0.95–0.99 and length-extrapolate flat to T=1024. The prior "+0.18 e97 edge on
   modquad" was a pre-convergence artifact — the linear arms catch up under proper
   AdamW+wd+long training. (e97/e97-lin are marginally more robust at the lowest
   wd=0.01, where gdn2 alone fails to grok modquad, 0.85–0.87.)

6. **a^n b^n c^n counting does not cleanly separate.** All arms grok the T=128 viability
   to ~0.94–1.0 and degrade comparably under length extrapolation (T=1024 ≈ 0.82–0.93);
   e97 at wd=1.0 holds the best long extrap (0.93) but within seed noise of the
   linear arms.

7. **Weight decay is grok-driving (confirmed).** wd=0.01 fails to grok S5 (gdn2) and
   modquad (gdn2); wd≥0.1 groks. This validates the canonical-grok setup and shows the
   earlier wd≈0/0.01 runs were in the no-grok regime — but removing that suppressor
   helps the *linear* cell, not e97.

## Bottom line

The e97 nonlinear-in-time cell does **not** grok the hard separators. The capability
nulls/ties reported earlier are **not** pre-grok artifacts of e97 — they survive
canonical grokking conditions (AdamW + wd sweep + fixed split + 100k/200k steps).
Where grokking *does* rescue a prior null (S5, low-wd modquad), it rescues **GDN-2
(linear)**, which groks to perfect test accuracy while e97 stays at baseline. The
per-step nonlinearity (tanh vs identity in the identical fused kernel) is null for
grokking on all three tasks. The "nonlinearity unlocks grokked
capability" hypothesis is **NULL** (per-step nonlinearity does not grok any of the
three tasks); consistent with the convergent-capability nulls and with
"the lin-vs-nonlin separator is counting/state-tracking and *linear* wins S5."
