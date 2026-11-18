# RESULT_OPTIMIZER_BAKEOFF.md — the optimizer/schedule bake-off on the winner

**Date:** 2026-11-19 · **Status: complete** — all five arms landed at the
protocol and committed as they landed (plus two launch-time failures on the
exploratory arm, receipted honestly); the assemble summary committed with this
doc. · **Authorisation:** the operator's bake-off directive — the side-by-side
dissection receipts the remaining gap to GDN2 as earned in the first ~10-20
steps, so the operator asks the direct question: *schedule-free? Muon? what is
the best optimizer/schedule for this machine?* — answered with receipts.

## 0. The answer, stated first

**The best optimizer/schedule for THIS machine — at probe scale, seed 3401, the
ladder's one protocol — is hybrid Muon at 3× the AdamW LR point.** It lands at
**1.5146 nats/byte, BEATING gdn2's banked 1.5837 by 0.069 nats at the matched
protocol** and beating the incumbent (tuned AdamW) by **0.308 nats** — the
largest single-arm gain any lane has banked on this machine, and it is pure
training surface: the mechanism is unchanged (same identity-CORRECT cache
regime, the key-collapse guard holds, no model/semantics file touched).

Two honest qualifications frame that answer:

* **The early hole and the endpoint are DIFFERENT axes, and no arm dominates
  both.** Schedule-free AdamW ELIMINATES the first-10-steps hole exactly as
  pre-stated (it LEADS gdn2 by 1.41 nats at step 1 and never trails through
  step 20) but does not move the steady-state deficit. Muon opens a MUCH
  WORSE early hole (deficit +1.96 at step 10, vs the incumbent's +0.36) and
  then descends faster than everything else, crossing gdn2 at ~step 80 and
  leading it through the landing. If the metric is the endpoint, Muon-3x
  wins; if the metric is the first megabyte, SF wins; the incumbent's
  AdamW+warmup is dominated on both axes by SOME other arm.
* **The C5 re-read is the scientific headline.** The side-by-side
  (`RESULT_SIDE_BY_SIDE_0239.md` §5) attributed the ~0.16-nat steady deficit
  to the value/read form (C5, with C4 as its per-layer face) **at the
  AdamW-trained endpoint**. Muon-3x, with the mechanism receipted unchanged,
  closes that steady deficit and inverts it (a −0.032-nat LEAD over gdn2
  across steps 80–160, four consecutive checkpoints). This does not refute
  C5 — it re-scopes it: under AdamW the write form UNDERTRAINS; the same
  sparse byte-identity-CORRECT store, trained by orthogonalized momentum,
  reaches gdn2-parity. The binding constraint was the training surface, not
  the architecture's ceiling. The named next lever (a gdn2-style dense gated
  delta write) remains the structural lever, but the optimizer surface must
  be fixed FIRST or it will silently absorb the next structural comparison
  too.

## 1. The protocol and the arms (the approved conventions)

Every arm runs the WINNER (K10 per-token + separated keys, 149,981,808
parameters, the parameter-neutral swap asserted) at the **exact probe
protocol verbatim**: the 20,000,000-target matched-exposure rule (landing at
**20,022,866 targets in 180 steps**, every arm identical), **seed 3401**, the
registered flagship stream (identity asserted), the document-disjoint
calibration eval at 7 windows / 14,336 targets (recorded-read path), the
honest clock, and the protocol's own grad-clip (1.0, error_if_nonfinite) —
the symmetry rule, applied identically in spirit across arms and receipted in
each run's `result.json`. The banked builder `velocity_ladder._build_k10` is
imported verbatim (the tuned winner's own config: lr 3.3427e-4, wd 0.17422,
warmup 45, read_scale 2.787, the transferred betas/beta2 0.9392 and
microbatch 8). `k10_*`/`k6_*`/side-by-side sources, the PREREGISTRATIONs and
`lattice.py` are untouched; cards 2/3 only (the side-by-side lane's cards
0/1 untouched).

**Checkpoint-eval curve, the primary instrument:** state snapshots (CPU
clones) taken inside the training loop at steps 1/5/10/20/40/80/120/160/180,
every eval on restored snapshots after training — the training path
unperturbed by construction (the side-by-side's own rule). For the
schedule-free arms the snapshot is the **primal-averaged eval point** (the
package's own `eval()` semantics, `x = y + (z - y)·(1 − 1/β₁)`), computed
non-destructively (`bakeoff_optimizers.sf_x_snapshot`, unit-tested against the
package's own swap). Protocol-identical starts, receipted: every arm's step-1
train loss is bit-identical to the banked incumbent's (5.785070835425993).

**The arms** (conventions approved by the supervisor; LR mapping and
provenance stated per arm in the receipts):

| arm | optimizer (provenance) | schedule | LR point |
|---|---|---|---|
| incumbent (cited, NOT re-run) | torch AdamW fused | warmup 45 → constant | the tuned 3.3427e-4 |
| `sf` | schedulefree **AdamWScheduleFree 1.4.1** (the same package the prior invertible line receipted) | **none, by design** (warmup_steps=0) | the incumbent's tuned PEAK LR — same LR, schedule removed; no scheduler is needed with schedule-free, so there is no warmup/decay curve to tune against the probe's FIXED 180-step budget (the averaging IS the schedule) |
| `sf_prior` | the same, at the prior line's own point | none | lr 1e-3, betas (0.9, 0.95), wd 0.01 (the LR/beta bracket) |
| `muon1x` | `bakeoff_optimizers.HybridMuonX45` — the prior line's pinned KellerJordan quintic VERBATIM (`hybrid_muon.scaled_update`, FP32 NS arithmetic, momentum .05-lerp + Nesterov .95, Moonshot 0.2·√max(shape) scaling), Muon on embeddings + projections (embed, head, mixer value/key/query_key/out, mlp w1-3), AdamW on norms + the decay gate | the incumbent's warmup 45, BOTH groups (the prior line scaled both equally) | 1× the tuned LR (the update-RMS normalization makes 1× the AdamW-matched point) |
| `muon3x` | the same | the same | 3× the tuned LR (the prior receipt's bracket: 3x = the neutral point, 10x collapses) |
| `sf_muon` (exploratory) | the package's `ScheduleFreeWrapper` primal averaging around the hybrid | none | the incumbent's LR, wrapper momentum 0.9, decay at y |

Receipts: `bakeoff_runs/<label>_seed3401/` (`start.json` with the full
convention block + versions, `training.jsonl` live per-step, `checkpoints.jsonl`,
`result.json` with the curve, walls, mechanism diagnostics and the
muon-update RMS), the assemble summary
`receipts/optimizer_bakeoff_summary.json`, and the lane's sources
(`optimizer_bakeoff.py`, `bakeoff_optimizers.py`,
`run_optimizer_bakeoff_leased.sh`, tests in `tests/test_x45_optimizer_bakeoff.py`).
Session lambda01, cards 2/3, RTX 6000 Ada, torch 2.9.1+cu128,
schedulefree 1.4.1.

**Walls, honestly confounded:** round 1 (`sf`, `muon1x`) ran while the box was
at load 234/256 from unrelated user jobs — the per-token arm is host-loop
bound, so its honest clock inflated ~6× (2,782 / 3,009 s train). Round 2 and
`sf_muon` ran on a quieter box and match the incumbent's class
(1,854–1,920 s vs the banked 1,858 s; 10.4–10.8k bytes/s). Bytes/s are NOT
comparable across rounds and are not load-bearing for any conclusion here.
Muon's own optimizer cost is receipted separately: **+0.36 s/step steady**
(the pinned FP32 Newton-Schulz over 72 MLP/projection matrices — the honest
price of the convention; measured `optimizer_step_seconds.mean_steady`).

## 2. The overlaid curves (THE artifact)

Calibration eval NLL (nats/byte, 7 windows / 14,336 targets) at matched
checkpoints — every arm on byte-identical windows at the same step; gdn2 and
the incumbent are the side-by-side's banked armed runs (cited, unmodified).
**Deficit = arm − gdn2; positive = worse than gdn2.**

| step | MB | gdn2 (banked) | incumbent AdamW (banked) | sf | sf_prior | muon1x | muon3x | sf_muon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.7045 | 5.6224 | **4.2919** | 4.3973 | 5.8092 | 5.8013 | 5.6296 |
| 5 | 0.57 | 3.8999 | 4.1586 | **3.2763** | 3.3146 | 5.7385 | 5.6138 | 5.2358 |
| 10 | 1.19 | 3.1024 | 3.4579 | **3.0784** | 3.2143 | 5.5579 | 5.0646 | 4.7827 |
| 20 | 2.37 | 2.6675 | 3.0853 | **2.6615** | 2.9991 | 4.8695 | 3.8080 | 4.1610 |
| 40 | 4.54 | 2.2309 | 2.4607 | 2.3184 | 2.5152 | 3.5587 | 2.7015 | 3.5749 |
| 80 | 8.88 | 1.8528 | 1.9787 | 2.0040 | 2.1499 | 2.3307 | **1.8427** | 3.0396 |
| 120 | 13.28 | 1.6290 | 1.7953 | 1.8273 | 1.9791 | 1.8775 | **1.5826** | 2.6969 |
| 160 | 17.72 | 1.5056 | 1.6696 | 1.7180 | 1.8561 | 1.6211 | **1.4649** | 2.4316 |
| 180 | 20.02 | **1.5837** | 1.8227 | 1.7174 | 1.8550 | 1.6542 | **1.5146** | 2.3561 |

The same table as deficits vs gdn2 (the bake-off's primary readouts):

| step | incumbent | sf | sf_prior | muon1x | muon3x | sf_muon |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.082 | **−1.413** | −1.307 | +0.105 | +0.097 | −0.075 |
| 5 | −0.259 | **−0.624** | −0.585 | +1.839 | +1.714 | +1.336 |
| 10 | −0.356 | **−0.024** | +0.112 | +2.456 | +1.962 | +1.680 |
| 20 | −0.418 | **−0.006** | +0.332 | +2.202 | +1.140 | +1.493 |
| 40 | −0.230 | +0.088 | +0.284 | +1.328 | +0.471 | +1.344 |
| 80 | −0.126 | +0.151 | +0.297 | +0.478 | **−0.010** | +1.187 |
| 120 | −0.166 | +0.198 | +0.350 | +0.249 | **−0.046** | +1.068 |
| 160 | −0.164 | +0.213 | +0.351 | +0.116 | **−0.041** | +0.926 |
| 180 | −0.239 | +0.134 | +0.271 | +0.071 | **−0.069** | +0.772 |

**What the curves say:**

* **The early hole is warmup-shaped — the SF prediction is MET.** `sf` leads
  gdn2 by 1.41 nats at step 1 (the primal average moves at the full tuned LR
  from the first update while every warmup arm is still crawling), tracks it
  evenly through step 20 (−0.024 / −0.006), and only then opens a deficit.
  The pre-stated refutation condition (opens the same hole or worse) is far
  from met. But SF does NOT move the steady state: its steps-80–160 mean
  deficit is +0.187 vs the incumbent's +0.152 — schedule-free fixes the
  TRANSIENT, not the deficit the side-by-side attributed to C5. And its
  final 1.7174 flatters it: the incumbent's 1.8227 is the endpoint draw the
  side-by-side receipts as ~0.075 nats of variance (at step 160 the incumbent
  1.6696 is BETTER than SF 1.7180).
* **The prior line's SF point is wrong for this machine.** `sf_prior`
  (lr 1e-3, betas (0.9, 0.95), wd 0.01) opens a steady +0.33 deficit and
  lands at 1.8550 — WORSE than the incumbent. The prior regime's SF collapse
  (0.064 there) does not reproduce here; what reproduces is the direction:
  this machine's SF sweet spot is at/near the AdamW-tuned LR, not 3× above
  it.
* **Muon trades a worse early phase for a much better everything-else.** Both
  Muon arms leave init slowly (the hole PEAKS at step 10: +2.46 at 1×, +1.96
  at 3× — the orthogonalized-momentum machinery has ~20 steps of momentum
  buildup and the warmup-45 schedule holds BOTH groups at fraction-LR
  exactly when the store must learn its regime), then descend faster than
  any other arm. From step 40 on, the ranking is strictly
  **muon3x > muon1x > incumbent** at every checkpoint. `muon3x` crosses
  gdn2 at step 80 and LEADS it at every checkpoint after (−0.010 / −0.046 /
  −0.041 / −0.069) — a four-checkpoint lead, not an endpoint draw.
* **The exploratory Muon+SF combination is REFUTED at this protocol.**
  `sf_muon` takes SF's step-1 lead (−0.075) and then Muon's hole, and never
  recovers: it lands at 2.3561, the worst arm of the bake-off. The wrapper's
  outer momentum on top of Muon's internal momentum at the full (unscheduled)
  LR moves too far per step and the averaging never lets the orthogonalized
  late phase compound. Stated honestly: two launch-time bugs were found and
  fixed on this arm first (receipted: the wrapper pre-populates the base
  optimizer's state, `bakeoff_optimizers` now guards on the momentum-buffer
  key; `ScheduleFreeWrapper` does not forward `last_metrics`), both with
  regression tests, both failures' consoles retained — the 2.3561 is the
  fixed code's honest result, not an artifact of the failures.

## 3. The readouts per arm (final NLL vs the ladder rows, hole, steady)

| arm | final NLL | vs gdn2 1.5837 | vs incumbent 1.8227 | step-10 hole (deficit) | steady deficit (steps 80–160) |
|---|---:|---:|---:|---:|---:|
| gdn2 (banked row 5) | 1.5837 | — | −0.2391 | — | — |
| incumbent AdamW tuned (banked row 6) | 1.8227 | +0.2391 | — | +0.356 | +0.152 |
| **muon3x** | **1.5146** | **−0.0691** | **−0.3081** | +1.962 | **−0.032** |
| muon1x | 1.6542 | +0.0705 | −0.1686 | +2.456 | +0.281 |
| sf | 1.7174 | +0.1337 | −0.1053 | **−0.024** | +0.187 |
| sf_prior | 1.8550 | +0.2713 | +0.0323 | +0.112 | +0.333 |
| sf_muon (exploratory) | 2.3561 | +0.7724 | +0.5334 | +1.680 | +1.060 |

*(The incumbent's step-10 hole here is its banked side-by-side number; its
steady mean is computed on the same 80/120/160 grid for like-for-like.)*

**The mechanism is receipted UNCHANGED in every arm** (the bake-off's scope
rule: training surface only) — final-checkpoint diagnostics, the banked
channels:

| arm | live slots /128 | read entropy (nats) | identity share | CORRECT/token | MERGE/token | key guard |
|---|---:|---:|---:|---:|---:|---|
| incumbent (banked) | 27.3 | 1.732 | 0.113 | 0.9789 | 0.00227 | ok |
| sf | 29.6 | 1.745 | 0.111 | 0.9767 | 0.00240 | ok |
| sf_prior | 23.0 | 1.474 | 0.130 | 0.9773 | 0.00479 | ok |
| muon1x | 35.9 | 1.822 | 0.117 | 0.9751 | 0.00125 | ok |
| muon3x | 35.6 | 1.664 | 0.123 | 0.9735 | 0.00251 | ok |
| sf_muon | 12.7 | 1.571 | 0.100 | 0.9848 | 0.00387 | ok |

Same regime everywhere (span_key_std min ≥ 0.05 in every layer, every arm —
the KEY-COLLAPSE guard holds throughout): a sparse byte-identity-CORRECT hot
set read at ~1.7 nats. Muon-3x reaches gdn2-parity **in the same regime** —
better weights, not a different machine.

## 4. The prior receipts, with the regime caveat

The prior optimizer bake-off (the invertible line,
`results_invertible_stage10_optimizers_8m` + `…_stage10b_muonlr_8m`): the e97
invertible tasks at 8 M parameters, 500,000 updates, warmup+cosine schedules,
`HybridMuon` with the routing that EXCLUDED embeddings, aux AdamW at betas
(.9, .999)/wd .01, and `schedulefree.AdamWScheduleFree` at lr .001 /
betas (.9, .95) / wd .01. Best accuracies (the directive's quoted prior,
matched to the receipts):

| arm | best accuracy (receipt) |
|---|---:|
| AdamW | 0.2522 |
| Muon | 0.2424 |
| Muon-3x | 0.2521 |
| SF | 0.0643 |
| Muon-10x | 0.0631 |

**The caveat, stated as the prior's own numbers demand:** that regime is
500k-step synthetic invertible tasks at 8 M parameters; this one is a
180-step real-text probe at 149.9 M parameters. Only the DIRECTION transfers,
and it transfers cleanly: Muon ≈/≥ AdamW at the matched LR, 3× is Muon's
neutral-to-best point, and the far points collapse. What does NOT transfer:
the prior's SF collapse (0.064) — SF is healthy here at the AdamW-matched LR
(1.7174, better than the incumbent's own endpoint) and only degrades at the
prior's 1e-3 point; and the prior's "3x = neutral" — here 3× is not neutral,
it is the best arm of the bake-off. The 10× point was NOT run here (outside
the approved bracket; the prior's collapse at 10× is cited, untested at this
protocol — an honest residual).

## 5. The verdict on the operator's question (honestly scoped)

* **Best optimizer/schedule for THIS machine, at probe scale, single seed,
  single protocol: hybrid Muon (the repo's own pinned implementation) at 3×
  the AdamW LR point, on the incumbent's warmup-45 schedule.** 1.5146 vs the
  incumbent's 1.8227 and gdn2's 1.5837; a steady-state LEAD over gdn2 across
  steps 80–160 (−0.032 mean, four consecutive checkpoints); the largest gain
  any single arm has banked on this machine, at +0.36 s/step optimizer cost.
* **Schedule-free is the right answer to the EARLY hole only.** It eliminates
  the first-10-steps hole by design (prediction MET) and nothing else: its
  steady deficit is no better than AdamW's, and its final number's advantage
  over the incumbent sits inside the endpoint-variance band the side-by-side
  receipts. If a future registration cares about early-phase loss (e.g., a
  short-horizon deploy), SF at the AdamW-tuned LR is the honest choice; at
  this protocol's single-endpoint metric it is dominated by Muon-3x.
* **Do not combine them here:** the package's own wrapper around the hybrid
  is receipted at 2.3561 — refuted at this protocol.
* **Scoping, stated:** single seed (3401 — the protocol's own fixed-seed
  rule, so every number is a seed-3401 number), one checkpoint eval per step
  on a 7-window / 14,336-target boundary (the ~0.075-nat endpoint-variance
  band the side-by-side receipts applies to every endpoint here), probe scale
  only (the 2700 s board protocol and C14+ training are NOT tested), the
  Muon LR axis is a two-point bracket (1×/3×) around the AdamW-tuned LR, not
  a search, and Muon's routing/aux knobs are the approved conventions, not a
  tuned optimum. The 0.308-nat Muon gain dwarfs the ±0.1-nat variance band
  and is the load-bearing conclusion; the 0.069 gdn2-beating margin is
  inside the band's reach and needs the confirmation below before it is
  bank-grade.

## 6. The registration recommendation

1. **Register hybrid Muon (this repo's `hybrid_muon` convention, routed
   embeddings + projections, both groups on the incumbent schedule) at 3×
   the AdamW LR point as the winner-line's default optimizer**, replacing
   the transferred AdamW — with a confirmation cell at the 2700 s board
   protocol (the same one-cell confirm discipline the tuned winner's own
   ladder row used) before it displaces AdamW in the scale-up registrations.
   The Muon LR axis (between 1× and 10×, and no-warmup variants) is then the
   first candidate for any re-opened CMA-ES search — the bracket says the
   incumbent LR is 3× too low for this machine's matrix params.
2. **Keep the C1 receipt's scope statement:** optimizer choice is now shown
   to be worth ~10× the whole CMA-ES tuning budget (0.308 vs ≤0.03 nats) —
   "structure first, tuning second" gains its third clause at this machine:
   **training surface before hyperparameters**. Any future structural
   comparison (the C5 write-form lever included) must be run under the best
   training surface or it will mis-attribute the optimizer's deficit to the
   structure — exactly what the side-by-side's C5-vs-AdamW endpoint did.
3. **Do not register SF or the Muon+SF wrapper** for this machine at this
   protocol (SF fixes only the transient; the wrapper is refuted).

## 7. Residuals, stated

* **Single seed, single protocol, probe scale** — the whole bake-off is
  seed-3401 at the 20 M probe; the 2700 s board confirmation (and eventually
  a multi-seed read) is required before the registration acts on the
  gdn2-beating margin. The Muon-beats-AdamW conclusion is far above the
  variance band and does not share this fragility.
* **The 10× Muon point is untested** at this protocol (outside the approved
  bracket; the prior's collapse at 10× is cited as the reason it was not
  spent).
* **Muon's early hole is unexplained beyond its shape** — the receipt shows
  the hole peaking at step 10 (+1.96) with the momentum-buildup/warmup
  interaction as the stated hypothesis; a no-warmup Muon arm (schedule ×
  optimizer interaction) is the natural next cell and was NOT run (outside
  the approved conventions).
* **Round-1 walls are load-confounded** (box at 234/256); no conclusion here
  rests on throughput. Muon's +0.36 s/step optimizer cost is the only
  velocity claim, measured steady on a quiet box.
* **Two launch-time failures on the exploratory arm** (`sf_muon`), both
  fixed with regression tests, both consoles retained as receipts
  (`bakeoff_runs/sf_muon.console.txt`, `smoke_sf_muon.console.txt`) — the
  arm's final number is the fixed code's honest result.

## Commands

```bash
# one arm, detached on a leased card from the diagnosis lane's pool (cards 2/3)
setsid bash -c 'OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=2,3 \
  experiments/expressivity_tasks/x45_matrix_emender/run_optimizer_bakeoff_leased.sh muon3x \
  > …/bakeoff_runs/muon3x.console.txt 2>&1' &

# the overlay summary (gdn2's + the incumbent's banked curves vs every arm)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff assemble

# the convention tests (CPU)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_x45_optimizer_bakeoff.py -q
```
