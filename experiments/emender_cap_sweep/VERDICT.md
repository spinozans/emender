# emender-cap-sweep — VERDICT (grounded in THIS run's measurements)

**Question.** With a *properly-budgeted* CMA over the Emender mixture (a sea of
GDN-2 recall heads + a discovered fraction of nonlinear e97_delta-tanh emendment
heads), (a) what mixture does CMA find, (b) across model dim {256,384,512,768,
1024} where does the Emender's capability separation close, and (c) what is the
efficiency cost? All measured fresh, bf16-uniform, fused split-edit asserted.

## VERDICT: NULL (no separation at matched compute).

The CMA-found Emender adds **no capability that best-vs-best GDN-2 (typed) cannot
reach**, **ties** GDN-2 on iso-param/iso-token loss, is **strictly worse on S5
state-tracking and counting at every tested dim**, and runs at **0.54–0.74×
throughput**. There is **no capacity boundary where an Emender advantage closes**
— measured against the clean `gdn2typed` control, no advantage exists at any of
the five dims.

## What the measurements say (each traces to a committed file)

1. **CMA kept a 12.5% sprinkle — but on a FLAT loss basin.** popsize 8 × 15 gens,
   120 candidates, param-locked, bf16+fused (all asserted). Found 28 gdn2_recall
   + 1 e97_track + 3 e97_delta (BPB 2.371 vs GDN-2 anchor 2.450). The
   "EMENDER kept a fraction" verdict is real on the raw fitness — **but the
   convergence curve shows best-BPB spanning only 2.370–2.389 while the visited
   e97_delta fraction spans 0.094–0.188.** Loss is flat across mixtures → the kept
   sprinkle is CMA wandering in a basin, not loss evidence for nonlinearity.
   (`results_cma/cma_result.json`, `generations.jsonl`.)

2. **The assumed small-scale separation does NOT reproduce vs a fair control.**
   The task's premise ("Emender separates: S5 ~0.79 vs FLA-GDN 0.36") does not
   hold here: at dim256 the Emender's S5 is 0.774 (≈ the cited 0.79) **but
   FLA-GDN solves S5 at 0.999, not 0.36.** The Emender is *below* both GDN-2
   controls on S5 — the opposite of a separation. The one place the Emender beats
   FLA-GDN (modquad@dim256, 0.990 vs 0.675) is matched exactly by `gdn2typed`
   (0.996): that win is the **typed substrate, not the emendment heads**.
   (`results_sweep/CAPACITY_TABLE.md`, `sweep_mean.csv`.)

3. **The emendment heads HURT, increasingly with scale.** S5: emender−gdn2typed =
   −0.22, −0.43, −0.68, −0.45, −0.68 across dims (bimodal per-seed collapse; all
   seeds collapse to ~0.31 by dim512). Counting: worse at every dim (−0.21 peak).
   GDN-2 (typed and fla) is robust ~0.99 on S5 throughout. So scaling the model
   does not *open* an Emender niche — it makes the nonlinear heads break
   state-tracking more reliably.

4. **Efficiency cost is real and one-sided.** Iso-param/iso-token BPB is a tie
   (Δ +0.0003). Throughput is 0.54× at the 41M proxy and 0.74× at the 1.3B head
   shape (overlap on). Consequence: to reach the tied loss, GDN-2 needs 1.7 min
   vs the Emender's 3.2 min — **1.85× slower to equal loss.** The 0.95×-throughput
   target is met only by the capability-**inert** shell head.

## Honest niche statement

Is there *any* niche for the CMA-found Emender? On these measurements, **no
positive niche on capability**: it never beats the typed GDN-2 control on any of
the three documented separation tasks at any of the five dims, and it is markedly
worse on two of them. Its loss is tie-level and its throughput is a 26–46% tax.
The only "win" it shows is over the *fla-gdn* incumbent on the modquad cliff at
the smallest dim — and that win is fully explained by the typed substrate (the
all-`gdn2_recall` typed control wins it identically), not by the nonlinear
emendment heads. This is consistent with, and extends to a measured capacity
sweep, the convergent-loss-null line for the real sparse Emender.

## Threats to validity (stated honestly)

- Capability sweep trains under autocast (the standard capability harness,
  matching the accepted `emender_real_cap` battery); the e97_delta-tanh head math
  is the **fused** bf16 split-edit Triton kernel either way (`use_triton_e97=True`,
  seq path verified) — fusion is a throughput optimization, identical accuracy.
  The CMA loss fitness and the Phase-3 efficiency runs are bf16-uniform+fused.
- The direct iso-wall BPB probe (6-min horizon, lr 1.2e-3) was confounded by
  late-training drift in **both** arms; the reliable iso-wall conclusion is the
  token-tie + measured-throughput inference (§4), not the raw 6-min BPBs.
- S5 per-seed bimodality means the mean understates how often the Emender fully
  solves S5 — but it never matches GDN-2's robustness, and the collapse fraction
  grows with dim. Per-seed values are in `sweep_per_seed.csv`.
