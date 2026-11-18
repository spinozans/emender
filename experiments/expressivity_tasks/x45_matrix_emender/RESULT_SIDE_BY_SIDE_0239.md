# RESULT_SIDE_BY_SIDE_0239.md — why GDN2 (1.5837) beats the tuned winner (1.8227) by 0.2390 nats

**Date:** 2026-11-18/19 · **Status: complete** — §0 pre-stated and committed
before any run; §1/§2/§4 (the curve, per-layer, gdn2-side tables) landed and
committed with the curve, all three armed runs protocol-equivalent
bit-for-bit; §3's C1 extension generation landed and the full analysis
(§5/§6) is committed with it. · **Authorisation:** the operator's
side-by-side gap-dissection directive — DIAGNOSIS ONLY, no model changes, no
fixes; ranked causes with receipts.

## The question, and the two models

At the matched-exposure probe protocol (20,022,866 targets, seed 3401, the
registered flagship stream, the document-disjoint calibration eval at 7
windows / 14,336 targets, the honest clock — the ladder's one protocol,
`RESULT_VELOCITY_LADDER.md`):

| arm | eval NLL (nats/byte) | config status |
|---|---:|---|
| gdn2 (row 5) | **1.5837** | its own 144-candidate-search-tuned nomination, verbatim; not re-tuned here |
| tuned winner (row 6: K10 per-token + separated keys, CMA-ES-tuned) | **1.8227** | 9 CMA-ES evaluations (one generation), then confirmed at the protocol |
| **the gap** | **0.2390** | |

Both are ~149.9 M parameters, 24 layers, the same stream, the same data order,
the same stopping rule, the same eval. The operator's directive: put them side
by side and figure out WHY.

## 0. The pre-stated candidate causes, each with its predicted signature

Stated before the runs. Every candidate maps to a measured signature in the
artifacts this lane produces; the verdict (§5) ranks them by what the data
shows, whatever it shows.

* **C1 — Tuning-budget asymmetry.** GDN2's config came from a 144-candidate
  byte-velocity search at this scale; the winner's tuning is ONE CMA-ES
  generation (9 evaluations total, budget-starved — the search's own receipt
  says the wall guard stopped cleanly before generation 2) AND tuned-to-a-
  horizon (the search's fixed-exposure scoreboard 1.6546 at 16.8 M targets vs
  the protocol number 1.8227 at 20 M — the search's own horizon effect,
  receipted in `RESULT_VELOCITY_LADDER.md` §3/§4). **Predicted signature:**
  (a) the bounded fair-tuning extension below buys O(0.01–0.03) nats at the
  probe objective (the incumbent trajectory's concave shallow shape
  continues) and cannot approach GDN2's 1.5163 anchor at that exposure —
  leaving ≥ 0.2 nats of the 0.2390 NOT owned by tuning; OR (b) the
  gap-vs-phase curve shows config-shaped artifacts: the tuned config (warmup
  45, lr 0.62× the transfer — a short-horizon optimum) LEADS the transferred
  config's curve early and the two converge late in training. Whatever the
  extension shows is attributed to C1 by the pre-stated rule.
* **C2 — Init/early regime.** The gap exists at step ~0–20 and never widens.
  **Predicted signature:** the checkpoint evals at steps 1/5/10/20 show the
  full ~0.24-nat gap already present; the gap-vs-phase curve is flat from the
  first checkpoint; the per-layer identity-rewrite share is ≥ ~0.5 everywhere
  at the first checkpoint and the arm never leaves the byte-identity regime
  it was born in.
* **C3 — Late plateau divergence (the `CMAES_TUNING.md` prior).** The 150 M
  finding: the emender config LED GDN2 for ~2 MB then hit a plateau the
  search couldn't move. The key prior to re-check: does the per-token+keys
  form STILL plateau, or does it now diverge elsewhere? **Predicted
  signature:** the gap-vs-step curve is flat-or-negative early (the winner
  leads), then opens after step K; the train-loss curves at matched bytes
  cross once and never re-cross; the winner's per-layer read entropy/occupancy
  freeze (the store stops changing) around the same K.
* **C4 — Layer-0 (or specific-layer) drag.** The identity-cache residue: layer
  0's identity-rewrite share is 0.614 at the final checkpoint where layers
  5–23 hold 0.072–0.085 (the K10 doc's §3.2 open item). One or few layers may
  earn the gap. **Predicted signature:** the per-layer store telemetry shows
  the affected layers' op/entropy/occupancy signatures anomalous across ALL
  checkpoints (not just the last); the per-layer read-mask probes (an
  eval-time diagnostic on the trained checkpoint: one layer's mixer output —
  which IS its store read — zeroed, eval NLL re-measured) show the affected
  layer's store contributing anomalously little (a decorative store) or
  anomalously much.
* **C5 — Value/read-form residual.** The fixed value-side structure
  underuses the store vs GDN2's state utilization. **Predicted signature:**
  our arm's effective live-mass/entropy metrics (occupancy ~24/128 mean live
  slots, read entropy ~1.65 nats over the live set — the banked mechanism
  receipts) sit materially below GDN2's effective state activity, measured by
  GDN2's own gate telemetry at its checkpoints (the channel-wise
  decay/erase/write gates its forward computes — the per-token state
  retention horizon and the write-gate mass, from its own projections, read
  non-invasively; its arm code is NOT modified). If GDN2's state is not
  instrumentable without touching its code, that is stated honestly and the
  banked profile receipts (`PROFILE_STEP_VS_GDN2.md`, the CMA-ES diagnosis)
  carry the comparison.

### 0.1 The C1 bounded fair-tuning extension — the exact protocol, pre-stated

The line the operator drew: the extension must be symmetric-in-principle and
pre-stated, and its result is attributed to C1 whatever it shows. This is the
whole protocol, fixed before launch:

* **Machinery:** the banked `velocity_ladder search` verbatim (imported,
  unedited, run through the banked `run_velocity_ladder_leased.sh` driver).
* **Objective:** the search's own fixed-exposure probe objective
  (16,777,216 targets → the 16,817,493 landing, seed 3401, the same stream) —
  the same scoreboard the banked generation-1 trajectory is on, so the two
  generations are directly comparable.
* **Space:** phase 1 (lr, weight_decay, warmup_steps, read_scale) — the same
  four dimensions as the banked generation.
* **x0:** the banked incumbent (eval 4: lr 3.3427e-4, weight_decay 0.17422,
  warmup 45, read_scale 2.787).
* **sigma0 0.15** (the banked search's own phase-2 refinement precedent for
  refining around an incumbent), **popsize 8, ONE generation, CMA-ES seed
  20261116** (fresh, stated — not the banked generation's seed).
* **Receipts:** appended to the banked per-evaluation log
  `receipts/velocity_k10_cmaes_evals.jsonl` (the machinery's own resume
  convention — the chunked-arm search extended the same log from phase 1 to
  phase 2, the lane's precedent) and a new result receipt
  `receipts/velocity_k10_cmaes_result_gen2.json`.
* **Symmetry, stated honestly:** GDN2's own config was found by a
  144-candidate search; this extension lifts the winner's search from 9 to 17
  candidates. It does not close the asymmetry — it bounds it. Phase 2
  (tau_match / tau_pop) remains unrun (budget: one generation ≈ 3.5 h of the
  ~4–5 h card budget this lane can spend, stated in §3 whatever the outcome).
* **Attribution rule, pre-stated:** whatever the extension shows — a big
  gain, a small gain, no gain, or a regression — is attributed to C1.

### 0.2 The build (what runs, on which receipts)

`side_by_side.py` (new file; `k10_*`/`k6_*`/gdn2 arm code, the
PREREGISTRATIONs and `lattice.py` untouched — the banked builders
`matched_arms.build_gdn2` / `velocity_ladder._build_k10` are imported
verbatim) runs BOTH arms at the exact probe protocol:

* **The armed runs** (both arms, the same checkpoints, byte-identical
  bindings): the K10 tuned winner, the K10 arm at the transferred (untuned)
  config — the C1 config-shape contrast — and GDN2, each trained to the
  20,000,000-target matched-exposure rule (landing at 20,022,866, seed 3401,
  the honest clock, the per-step log written live). **Checkpoint evals at
  steps 1/5/10/20/40/60/80/100/120/140/160/180** — the model state is
  snapshotted (CPU copies of `state_dict`) at each checkpoint step INSIDE the
  training loop and every evaluation/telemetry pass runs on the restored
  snapshots AFTER training completes, so the training path is unperturbed by
  construction and stays comparable row-for-row with the banked runs. The
  protocol-equivalence proof is the banked template's own: every training-log
  row must be bit-identical to the banked run's row (loss and bytes), and the
  step-180 checkpoint's eval must reproduce the banked ladder number
  bit-for-bit — asserted in-code, fail loudly.
* **Per-layer store telemetry for our arm at the checkpoints** (the existing
  channels, imported unchanged: `probe_diagnostics` — per-layer op rates,
  occupancy, read entropy, identity-rewrite share, the key-collapse watch —
  plus the DECAY gate's per-layer mean from the trained walk's own
  projections, a non-invasive forward hook in this file).
* **GDN2 telemetry where its arm code exposes it:** the per-checkpoint
  calibration eval, the train-loss curve, and the gate diagnostics its
  forward computes (the channel-wise log-decay `g`, the erase gate `b`, the
  write gate `w` — read from its own projection modules in a forward hook;
  its source is not modified). Its recurrent state itself is internal to the
  FLA chunk kernel and is not exposed by its arm wrapper — stated honestly;
  the gates are the state's own write/forget coefficients, so their telemetry
  IS the state-utilization receipt.
* **The C4 read-mask probes** (a diagnostic receipt only, no model change):
  at the winner's step-180 checkpoint, one layer at a time, the layer's mixer
  output — which is exactly its store read, projected — is zeroed for the
  eval pass and the calibration NLL is re-measured; plus one all-layers
  masked cell (the storeless floor). This measures each layer's store-read
  contribution to held-out NLL.
* **The C1 extension** per §0.1.

GPU: cards 0/1, one leased card for the armed runs (sequential), one for the
C1 extension, per the lease broker. Python `/home/erikg/emender/.venv/bin/python`,
OMP_NUM_THREADS=4, detached launchers per the long-session survival playbook.

## 1. The gap-vs-training-phase curve (THE headline artifact)

**Receipts:** `side_by_side_runs/{k10_tuned_armed,gdn2_armed,k10_transferred_armed}_seed3401/`
(`start.json` / `training.jsonl` / `checkpoints.jsonl` / `result.json`), session
lambda01 card 1 (RTX 6000 Ada, torch 2.9.1+cu128, triton 3.5.1), and
`receipts/side_by_side_summary.json` (the assembler's summary).

**Protocol equivalence, receipted bit-exact on every armed run** (the GDN2
row's own proof template, asserted in-code): all three armed runs reproduce
their banked ladder rows' 180 training-log rows bit-for-bit (bytes AND loss),
and every final-checkpoint eval reproduces the banked ladder number
bit-for-bit — the tuned winner **1.822737353188651** (banked confirm), gdn2
**1.5836738858904158** (banked gdn2 row), the transferred config
**1.841756020273481** (banked untuned probe). The checkpoint design (state
snapshots inside the loop, every eval/telemetry pass on restored snapshots
after training) perturbs nothing: the training path is provably the banked
path. Walls: tuned 1,858.2 s (10.8k tok/s), transferred 1,858.6 s, gdn2
536.8 s (37.3k tok/s) — matching the banked rows.

The curve — calibration eval NLL (nats/byte, 7 windows / 14,336 targets) at
matched checkpoints (both arms see byte-identical windows at the same step),
seed 3401:

| step | MB seen | gdn2 | tuned winner | transferred | **winner − gdn2** | tuned − transferred |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.7045 | 5.6224 | 5.6188 | **−0.0821** | +0.0036 |
| 5 | 0.57 | 3.8999 | 4.1586 | 4.1257 | **−0.2587** | +0.0329 |
| 10 | 1.19 | 3.1024 | 3.4579 | 3.4279 | **−0.3555** | +0.0300 |
| 20 | 2.37 | 2.6675 | 3.0853 | 3.0668 | **−0.4178** | +0.0185 |
| 40 | 4.54 | 2.2309 | 2.4607 | 2.4529 | **−0.2298** | +0.0078 |
| 60 | 6.71 | 1.9459 | 2.1096 | 2.1503 | **−0.1637** | −0.0407 |
| 80 | 8.88 | 1.8528 | 1.9787 | 1.9977 | **−0.1259** | −0.0190 |
| 100 | 11.09 | 1.7232 | 1.9000 | 1.9190 | **−0.1768** | −0.0190 |
| 120 | 13.28 | 1.6290 | 1.7953 | 1.8312 | **−0.1664** | −0.0358 |
| 140 | 15.52 | 1.5614 | 1.7345 | 1.7409 | **−0.1731** | −0.0064 |
| 160 | 17.72 | 1.5056 | 1.6696 | 1.6775 | **−0.1640** | −0.0079 |
| 180 | 20.02 | **1.5837** | **1.8227** | 1.8418 | **−0.2391** | −0.0190 |

*(negative = gdn2 better; the winner−gdn2 column is the gap the headline
comes from.)*

**What the curve says, read honestly:**

* **The gap does NOT grow late and does NOT appear only late.** The winner's
deficit is at its LARGEST at steps 10–20 (0.36–0.42 nats, at 1.2–2.4 MB),
narrows to a **flat plateau of −0.162 ± 0.018 nats over steps 60–160**, and
the headline **−0.2391 is the endpoint draw on top of that plateau**: at the
last checkpoint BOTH arms' evals jumped up (the winner +0.153 over its
step-160 value, gdn2 +0.078 — the same late-train-loss noise and
7-window eval-boundary variance the ladder doc's §1 blemish receipts on
gdn2's own row). Had the endpoint continued the step-160 draw, the headline
gap would read ≈ 0.164, not 0.239. **The steady-state deficit at matched
exposure is ≈ 0.16 nats; ≈ 0.075 nats of the 0.239 headline is
checkpoint-endpoint variance, not mechanism.**
* **The early phase (steps 1–20) is where the arms diverge — and the winner
briefly LEADS at step 1.** At step 1 the winner evals 0.082 BETTER than gdn2;
by step 20 it is 0.418 WORSE. The mechanism is receipted in the winner's own
telemetry (§2): at steps 1–20 its store is nearly empty (mean live slots
8.8 → 5.7, occupancy 4–7 % of rank) with a slow DECAY gate (0.98/token ≈ a
40-token row half-life) — the schedule-driven store has not yet learned what
to keep, while gdn2's delta-rule writes a dense state update at EVERY token
from step 1 (no learned admission policy: its write gate is ~0.5 on every
channel at init and stays there, §4). gdn2's memory is useful immediately;
the winner's memory comes online between steps 20 and 80 (occupancy 5.7 →
20.7, DECAY 0.9804 → 0.9108, read entropy 0.49 → 1.77), exactly the window
where the gap narrows from 0.42 to 0.13.
* **No plateau, no stall, no widening after step K (C3's signature is
ABSENT).** The winner's eval descends steadily 2.46 → 1.67 over steps 40–160,
tracking gdn2 at a constant offset. The `CMAES_TUNING.md` plateau (the chunked
arm under content addressing, locked at ~2.6–2.7 while gdn2 descended)
 **does not reproduce** in the per-token+separated-keys form at this
protocol: whatever remains is a constant-offset deficit, not a diverging one.
* **The C1 config-shaped artifact is present exactly as pre-stated (the
tuned-to-a-horizon signature).** The tuned incumbent LEADS the transferred
config by +0.033/+0.030/+0.019/+0.008 nats at steps 5/10/20/40 — and the
lead EVAPORATES and REVERSES from step 60 on (−0.006 to −0.041 through the
last six checkpoints). The search's 16.8 M scoreboard sits at ~step 152,
right where the two curves cross back; the search's −0.0229 objective delta
is an early-horizon artifact that does not survive to the protocol — the
endpoint difference (−0.019) is inside the variance band and MID-training
the transferred config is slightly BETTER. Tuning is not where the gap
lives.

## 2. The per-layer table (the winner's store, at every checkpoint)

**Receipts:** every checkpoint's full `probe_diagnostics` (op rates, occupancy,
read entropy, identity-rewrite share, key-collapse watch — the banked
instruments, imported unchanged) plus the DECAY-gate telemetry and the C4
read-mask sweep, all in `side_by_side_runs/k10_tuned_armed_seed3401/`.

The final checkpoint (step 180), per layer — the winner's store side by side
with gdn2's state telemetry (§4):

| L | live slots /128 | read entropy (nats) | identity share | DECAY gate | row half-life (tok) | read-mask ΔNLL | gdn2 retention exp(g) | gdn2 horizon (tok) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.4 | 2.636 | 0.796 | 0.9736 | 25.9 | **+0.0794** | 0.635 | 2.7 |
| 1 | 35.9 | 1.507 | 0.094 | 0.8133 | 3.4 | +0.0436 | 0.731 | 3.7 |
| 2 | 30.9 | 1.440 | 0.092 | 0.8256 | 3.6 | **+0.0603** | 0.658 | 2.9 |
| 3 | 25.6 | 1.772 | 0.081 | 0.8699 | 5.0 | +0.0328 | 0.705 | 3.4 |
| 4 | 29.0 | 1.553 | 0.083 | 0.8211 | 3.5 | +0.0208 | 0.701 | 3.3 |
| 5 | 27.1 | 1.479 | 0.084 | 0.8272 | 3.7 | +0.0207 | 0.864 | 7.4 |
| 6 | 33.0 | 1.368 | 0.090 | 0.8047 | 3.2 | +0.0170 | 0.707 | 3.4 |
| 7 | 27.4 | 1.456 | 0.085 | 0.7910 | 3.0 | +0.0025 | 0.716 | 3.5 |
| 8 | 28.1 | 1.116 | 0.091 | 0.7881 | 2.9 | +0.0124 | 0.563 | 2.3 |
| 9 | 24.3 | 1.719 | 0.080 | 0.8773 | 5.3 | +0.0065 | 0.784 | 4.6 |
| 10 | 27.7 | 1.682 | 0.084 | 0.8602 | 4.6 | +0.0022 | 0.615 | 2.6 |
| 11 | 22.7 | 2.231 | 0.076 | 0.9440 | 12.0 | −0.0004 | 0.696 | 3.3 |
| 12 | 27.5 | 2.076 | 0.078 | 0.9053 | 7.0 | +0.0018 | 0.714 | 3.5 |
| 13 | 25.9 | 1.394 | 0.092 | 0.8440 | 4.1 | +0.0010 | 0.750 | 4.0 |
| 14 | 27.4 | 1.724 | 0.083 | 0.8778 | 5.3 | +0.0035 | 0.674 | 3.1 |
| 15 | 27.1 | 1.862 | 0.079 | 0.8820 | 5.5 | +0.0010 | 0.588 | 2.4 |
| 16 | 27.5 | 2.153 | 0.080 | 0.9260 | 9.0 | −0.0016 | 0.647 | 2.8 |
| 17 | 28.6 | 1.728 | 0.081 | 0.8521 | 4.3 | +0.0041 | 0.529 | 2.1 |
| 18 | 26.3 | 1.770 | 0.083 | 0.8656 | 4.8 | +0.0012 | 0.686 | 3.2 |
| 19 | 27.1 | 1.954 | 0.080 | 0.8990 | 6.5 | −0.0001 | 0.635 | 2.7 |
| 20 | 28.8 | 1.688 | 0.083 | 0.8287 | 3.7 | +0.0025 | 0.745 | 3.9 |
| 21 | 22.5 | 1.644 | 0.081 | 0.8230 | 3.6 | +0.0046 | 0.700 | 3.3 |
| 22 | 24.6 | 1.787 | 0.080 | 0.8693 | 4.9 | +0.0045 | 0.683 | 3.2 |
| 23 | 18.9 | 1.841 | 0.079 | 0.9117 | 7.5 | +0.0119 | 0.683 | 3.2 |

*(read-mask ΔNLL: the C4 diagnostic — that layer's mixer output, which IS its
store read projected, zeroed for the eval pass. Ablation effects, not
marginals: the sum of single-layer deltas is +0.332 where the all-layers cell
(the storeless floor, 3.1573 nats) is +1.3346 — the stores' contributions
are heavily sub-additive/redundant.)*

**What the per-layer receipts say:**

* **Layer 0 never leaves the byte-identity regime — at EVERY checkpoint**
(identity share 0.797 / 0.803 / 0.810 / 0.832 / 0.848 / 0.868 / 0.838 /
0.829 / 0.819 / 0.816 / 0.828 / 0.796 at steps 1→180; layers 1–23 sit at
0.076–0.102 throughout — the C4 predicted signature, confirmed). The
K10 doc's open item (0.614 at the untuned probe) is not a tuning artifact:
under the tuned config layer 0's share is even HIGHER (~0.80), and it is
structural — at the first store layer the payload is a near-pure function of
the byte, so even context-addressed keys cannot avoid byte-identity
CORRECTs there.
* **But layer 0's store is NOT decorative — it is the single most load-bearing
read** (read-mask Δ +0.0794, 24 % of the single-layer sum). The C4 "layer-0
drag" framing splits: layer 0 is anomalous (identity-bound) AND earning its
keep; its 25.9-token half-life (DECAY 0.974 — the slowest of any layer) is
the one store behaving like a long-lived byte cache rather than the
short-horizon contextual memory every other layer learned.
* **The store's eval contribution is concentrated in the first ~7 layers**
(Δ +0.079 / +0.044 / +0.060 / +0.033 / +0.021 / +0.021 / +0.017); layers
7–23 are near-decorative individually (|Δ| ≤ 0.0125, seven of them ≤ 0.002,
three of those slightly NEGATIVE). Most of the 24 stores' reads barely matter to
held-out NLL at this exposure — the machine runs 24 stores to bank ~0.33
nats of single-layer ablation effect, mostly from its first seven.
* **The store's regime is LEARNED and arrives late — the C2/C5 mechanism
receipt.** Aggregate over checkpoints: mean live slots 8.8 / 8.3 / 6.5 /
5.7 / 7.2 / 14.8 / 20.7 / 23.4 / 27.0 / 29.9 / 28.2 / 27.3; mean DECAY gate
0.9826 / 0.9816 / 0.9824 / 0.9804 / 0.9651 / 0.9422 / 0.9108 / 0.8895 /
0.8883 / 0.8736 / 0.8654 / 0.8617; read entropy 0.844 / 0.754 / 0.517 /
0.488 / 0.865 / 1.560 / 1.767 / … / 1.732. The arm STARTS as a nearly
empty, slow-forgetting cache and LEARNS its way to a ~27-slot hot set with
a ~4–5-token row half-life — converging on gdn2's own memory horizon
(§4: gdn2's implied horizons are 2.1–7.4 tokens per layer). By the end the
 two machines keep memory on the same timescale; the winner's rows are
just sparse-addressed rather than dense.
* **The op mix at the end is still the identity-cache arithmetic**: CORRECT
0.9789/token, APPEND 0.0188, MERGE 0.0023, POP 0.0000 — the schedule is
overwhelmingly update-in-place over the hot set (the banked mechanism,
unchanged by tuning, unchanged by training phase).

## 3. The C1 fair-tuning extension

**Run** (the pre-stated §0.1 protocol, verbatim, on card 0, detached through
the banked driver): the banked `velocity_ladder search` machinery, x0 = the
banked incumbent (eval 4, 1.6546), sigma0 0.15, popsize 8, ONE generation,
CMA-ES seed 20261116, the search's own 16,777,216-target objective, appended
to the banked per-eval log (its resume convention — receipts
`velocity_k10_cmaes_evals.jsonl` rows 9–16,
`receipts/velocity_k10_cmaes_probes/eval_009.json` … `eval_016.json`, result
`receipts/velocity_k10_cmaes_result_gen2.json`).

**IN FLIGHT at the time of this section's first commit** (the curve sections
§1/§2/§4 landed and were committed first, per the directive's commit order):
the first three evaluations of the extension's generation had landed —
1.6748 / 1.6677 / 1.7089, **all three WORSE than the incumbent's 1.6546** —
consistent with C1's pre-stated predicted signature (a). Completed below.

**The generation's result** (all 8 evaluations on the search's own
16.8 M-target scoreboard, seed 3401):

| eval | config (lr / wd / warmup / read_scale) | NLL |
|---:|---|---:|
| 9 | 8.69e-4 / 0.0711 / 57 / 3.218 | 1.6748 |
| 10 | 3.37e-4 / 0.1753 / 57 / 3.183 | 1.6677 |
| 11 | 2.05e-4 / 0.5390 / 43 / 3.324 | 1.7089 |
| 12 | 3.57e-4 / 0.1672 / 64 / 3.934 | 1.6604 |
| 13 | 2.75e-4 / 0.1922 / 18 / 2.725 | 1.6948 |
| **14** | **3.96e-4 / 0.1486 / 21 / 3.162** | **1.6514** |
| 15 | 2.04e-4 / 0.0679 / 84 / 2.865 | 1.7388 |
| 16 | 2.27e-4 / 0.1772 / 64 / 1.999 | timeout (scored inf by the machinery's rank rule; the subprocess exceeded the 3000 s per-eval guard — stated, not smoothed) |

* **The extension buys −0.0032 nats at the probe objective** (1.6546 →
1.6514, eval 14) — one candidate in eight. Seven of eight no better than
the incumbent; the trajectory stays concave and shallow, exactly the
pre-stated signature (a). The whole tuning story so far: the transfer
1.6775 → gen-1 1.6546 → gen-2 1.6514 = **−0.0261 nats of search on a
0.161-nat probe-objective deficit to gdn2's banked 1.5163 anchor** — 16 %
of it — on a scoreboard (§1's config-shape artifact) that itself inflates
tuning's protocol-scale value.
* **The honest bound on C1 at the ladder's own protocol:** the banked
confirmation already showed the gen-1 incumbent's probe-objective gain
shrinks to −0.0191 at the protocol (inside the ~0.1-nat checkpoint-variance
band), and §1's tuned-vs-transferred curve shows the incumbent's advantage
is an EARLY-phase artifact (+0.03 at steps 5–20) that reverses mid-training.
A gen-2 incumbent's −0.0032 at the probe objective cannot own more than
~0.02–0.03 nats of the 0.239 headline. **C1 is bounded, small, and not where
the gap lives.** Symmetry, restated: gdn2's nomination came from a
144-candidate search; the winner's arm has now had 17 evaluations. Phase 2
(tau_match / tau_pop) remains unrun — one more generation ≈ 3.5 h did not
fit the lane's budget, and both generations' receipts say the space around
the incumbent is flat.

## 4. GDN2's side of the table (gate/state telemetry)

**Receipt:** `side_by_side_runs/gdn2_armed_seed3401/checkpoints.jsonl` (the
gate telemetry at every checkpoint) — gdn2's own forward's gate arithmetic,
read non-invasively from its own projection modules (its source untouched;
its recurrent state is internal to the FLA chunk kernel and NOT exposed by
its arm wrapper — stated honestly; the gates are the state's own
write/forget coefficients, so this telemetry IS the state-utilization
receipt).

| quantity (final checkpoint, per-layer mean over the probe windows) | value |
|---|---|
| state retention per token, exp(g) | 0.529–0.864 per layer (mean 0.684) |
| implied memory horizon (1/(1−retention)) | 2.1–7.4 tokens per layer |
| write gate w (every channel, every token) | 0.49–0.53 (mean 0.515) |
| erase gate b (every key channel, every token) | 0.49–0.53 (mean 0.494) |
| at step 1: retention / write gate | 0.702–0.720 / 0.50 — essentially unchanged by training |

**What gdn2's side says:**

* **gdn2's memory regime is architectural and available from step 1**: it
writes a dense, channel-gated rank-1 delta to the FULL state at EVERY token
(w ≈ 0.5 on every channel from init, moving to ≈ 0.52 by the end), with a
short input-driven decay horizon (2–7 tokens). Nothing about its memory has
to be learned before it is useful — exactly why its eval descends through
steps 1–20 while the winner's nearly-empty learned store cannot follow
(§1).
* **By the end of the matched exposure, both machines keep memory on the same
timescale** (gdn2 horizons 2.1–7.4 tokens; the winner's learned row
half-lives 2.9–25.9 tokens, layers 1–23 mostly 3–7). The winner's training
moved its DECAY gate 0.983 → 0.862 and its occupancy 6 → 27 — it CONVERGED
on gdn2's regime. The steady-state ≈ 0.16-nat deficit is therefore NOT a
memory-timescale mismatch; it is in WHAT is written and read back: gdn2's
state is a dense superposition of every recent token (full-rank write, every
token), while the winner's read addresses ~e^1.73 ≈ 5.6 effective rows out
of ~27 live slots — a sparse, near-argmax-addressed cache over
payload directions whose top write op is byte-identity CORRECT (§2).

## 5. The C1–C5 disposition and the ranked residual attribution

Every candidate cause, its pre-stated signature (§0), and what the data
showed:

| cause | pre-stated signature | what the receipts show | verdict |
|---|---|---|---|
| **C1** tuning-budget asymmetry | the extension buys O(0.01–0.03) and stays an order of magnitude short; and/or the curve shows the tuned config leading early, converging late | the extension bought **−0.0032** at the probe objective (7 of 8 candidates no better than the incumbent); the whole 17-eval search owns −0.0261 on a 0.161 probe-objective deficit; the tuned-vs-transferred curve shows the incumbent's advantage is an early-phase artifact (+0.033 at step 5) that REVERSES from step 60 on | **CONFIRMED as bounded-and-small** — signature (b) exactly, and signature (a)'s shape at a magnitude slightly UNDER the pre-stated band (−0.0032 < O(0.01–0.03)); the pre-stated attribution rule assigns it to C1 as shown. C1 owns ≤ ~0.02–0.03 nats of the headline |
| **C2** init/early regime | the full gap present at steps 1/5/10/20, flat after; identity share ≥ ~0.5 everywhere from the start | the deficit is present EARLY and at MORE than its final size (0.36/0.42 at steps 10/20) — but it is NOT flat after: it NARROWS to ~0.16 by step 60 as the store comes online; identity share ≥0.5 holds ONLY in layer 0 (0.80–0.87 at every checkpoint), not everywhere | **PARTIAL — the divergence is early, but it is transient.** C2 receipts WHERE the arms diverge, not the endpoint residual |
| **C3** late plateau (the CMAES prior) | gap flat early, opening after step K; the winner keeps pace then stalls | the OPPOSITE shape: the deficit is LARGEST early (0.42 at step 20), narrows to a **flat 0.162 ± 0.018 plateau over steps 60–160**; the winner's eval descends steadily 2.46 → 1.67 with no stall and no widening | **REFUTED at this protocol** — the per-token+keys form does NOT hit the chunked arm's plateau |
| **C4** layer-0 / specific-layer drag | the affected layers anomalous at all checkpoints; the read-mask probes show the layer's store decorative or load-bearing | layer 0 IS anomalous at every checkpoint (identity 0.80–0.87, never leaves the byte regime) — and its read is the **most load-bearing single layer** (+0.0794 mask-delta), NOT decorative; layers 7–23 are near-decorative (|Δ| ≤ 0.0125); the store's eval value concentrates in layers 0–6 | **SPLIT verdict:** the anomaly is real and structural, but it EARNS its keep — "layer-0 drag" as dead weight is refuted; the per-layer concentration is C5's face |
| **C5** value/read-form residual | our live-mass/entropy materially below gdn2's effective state activity | the winner's store: CORRECT 0.979/token (byte-identity update-in-place), reads over ~27 live slots with entropy 1.73 (≈ **e^1.73 ≈ 5.6 effective rows**); gdn2: dense channel-gated writes EVERY token (w ≈ 0.5 on every channel), full-state reads, horizons 2.1–7.4 tokens; the winner LEARNED gdn2's memory timescale over the run (DECAY 0.983→0.862, occupancy 6→27) so the timescale is NOT the residual — the WRITE/READ FORM is | **CONFIRMED as the load-bearing mechanism** |

### The ranked residual attribution of the 0.2391 headline

Stated as an honest decomposition of the measured number — with the caveat
that ablation deltas are sub-additive (§2) and causes C4/C5 are faces of one
value-side structure, not separable marginals:

1. **Endpoint checkpoint-eval variance: ≈ 0.075 nats (~30 %) — measurement,
   not mechanism.** At step 160 the gap is 0.164; both arms' evals JUMPED at
   the step-180 boundary (the winner +0.153, gdn2 +0.078 — the same
   late-train noise and 7-window eval-boundary variance the ladder doc
   blemish receipts on gdn2's own row). The steady-state gap at matched
   exposure is **≈ 0.16 nats**; the headline 0.2391 is the endpoint draw on
   top of it. (A slice of the late rise could be real late-training noise
   in the winner's regime — not separable at one seed, stated.)
2. **The value/read-form residual (C5, with C4 as its per-layer face): ≈
   0.14 nats — THE mechanism.** Receipts: the op mix is byte-identity
   update-in-place (CORRECT 0.979/token, APPEND 0.019, MERGE 0.002, POP 0);
   the read supports ≈ 5.6 effective rows of ~27 live slots (entropy 1.73)
   against gdn2's dense every-token delta writes and full-state reads; most
   of 24 stores are individually near-decorative at eval (layers 7–23,
   |Δ| ≤ 0.0125) — the machine spends 24 stores to bank a sparse hot-set
   cache whose value concentrates in its first seven layers; layer 0 is
   structurally byte-identity (0.80–0.87 at every checkpoint) and the most
   load-bearing single read. The address side is NOT the residual (the keys
   work: layers 1–23 identity share 0.08; the KEY-COLLAPSE guard holds
   throughout); the memory TIMESCALE is not the residual (the winner
   converged on gdn2's 3–7-token horizon by itself). What remains is what
   the value channel writes and what the read can use.
3. **C1 tuning asymmetry: ≤ ~0.02–0.03 nats (~10 %), bounded by the
   extension.** §3's receipts. Inside the variance band at the protocol;
   the tuned-to-a-horizon artifact means even this bound is generous.
4. **C2's early-regime divergence (transient; ~0 at the endpoint, but the
   mechanism's onset receipt):** the deficit was 0.36–0.42 at steps 10–20
   because gdn2's memory is architectural (on from step 1) while the
   winner's store must LEARN its regime (occupancy 5.7 → 27, DECAY 0.98 →
   0.86 over the first 5–9 MB). Same root as (2): the write policy is
   learned, not given.
5. **C3: 0 — refuted** (no late plateau at this protocol).

**Bottom line.** GDN2's 0.2390-nat lead at the matched protocol decomposes
into ~0.075 nats of endpoint eval variance and ~0.16 nats of steady-state
deficit, of which tuning owns ≤ ~0.02–0.03 and the rest is the store's
**write/read form**: a sparse, byte-identity-CORRECT-dominated cache that
learns its memory regime late, versus gdn2's always-on dense gated delta
writes. Structure first, tuning second — the ladder's spine, now with the
residual's mechanism receipted layer by layer.

## 6. The named next lever

The dissection names ONE lever, and it is not a hyperparameter and not the
key side: **the value channel's write form**. The receipts converge on it
from three directions —

* the address side already works (layers 1–23 identity share 0.08, keys
  spread, KEY-COLLAPSE guard held throughout) and layer 0's byte-identity
  is structural (§2), so further addressing work cannot move the residual;
* the memory timescale already converged on gdn2's (the winner LEARNED
  3–7-token horizons by itself), so decay/schedule policy is not the
  binding constraint;
* the write op is 97.9 % byte-identity CORRECT over a ~27-slot hot set, and
  17 of 24 stores are individually near-decorative at eval — the store's
  problem is WHAT the value channel writes (near-functions of the byte)
  and how little of it the read can use (≈ 5.6 effective rows).

The concrete next stage this points at, in the lane's own registered order:
a write-side change that makes payloads contextual rather than byte-derived
— the K10 registration's own named follow-up (**ordered / Wheeler-style
keys: addresses of addresses**, `K10_SEPARATED_KEYS.md` §5) is the
key-side version, but this dissection's evidence says the VALUE side is
the binding one: payloads/DECAY structured so a write carries
context-conditioned content gdn2-style (a dense, gated delta at every
token) instead of a byte-identity replay. The C1 receipt bounds what
optimizer work can buy on top (≤ ~0.02–0.03); the C2 receipt adds the
secondary benefit: a write form that is useful from step 1 removes the
5 MB of learning the learned-schedule store currently spends reaching
the regime gdn2 has by construction. Diagnosis only — no model was changed
in this lane; any such change needs its own registration and gates.
