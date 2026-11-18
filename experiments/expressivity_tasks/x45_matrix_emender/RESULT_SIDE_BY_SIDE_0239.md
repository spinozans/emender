# RESULT_SIDE_BY_SIDE_0239.md — why GDN2 (1.5837) beats the tuned winner (1.8227) by 0.2390 nats

**Date:** 2026-11-18/19 · **Status: §0 (the pre-statement) committed BEFORE any run
of this lane; §1+ to be filled as the receipts land.** · **Authorisation:** the
operator's side-by-side gap-dissection directive — DIAGNOSIS ONLY, no model
changes, no fixes; ranked causes with receipts.

**STATUS: PRE-STATEMENT.** The candidate causes below and their predicted
signatures, the checkpoint plan, and the C1 fair-tuning extension's exact
protocol are stated and committed BEFORE any run of this lane launches. The
build (`side_by_side.py`) is committed with them. No result in this document
predates its run.

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

TO BE FILLED when the armed runs land.

## 2. The per-layer table

TO BE FILLED.

## 3. The C1 fair-tuning extension

TO BE FILLED.

## 4. GDN2's side of the table (gate/state telemetry)

TO BE FILLED.

## 5. The C1–C5 disposition and the ranked residual attribution

TO BE FILLED.

## 6. The named next lever

TO BE FILLED.
