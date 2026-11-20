Pre-registration draft — THE ADDITIVE READ (K13): GDN2's read form
transplanted into the K10 winner — the row-softmax read replaced by the
additive sum of the banked score channel's raw similarities, gated by the
committed `g_proj` read gate

**Date:** 2026-11-24 · **Status: DRAFT, operator sign-off pending.** This is
a NEW registration (it changes the store's READ FORM — a value channel, the
recorded schedule's own class; the discrete op selection is untouched),
written and committed BEFORE any probe measurement is claimed. Probe-scale
engineering checks (parity gates, gradchecks, memory, timing) are
pre-registration evidence and are in scope now; **no board-scale (2700 s)
claim is run before the operator signs this document off.** It does not
touch any `k10_*`/`k11*`/`k12_*`/`x46_*` source, any committed
PREREGISTRATION, `lattice.py`, or GDN2 model code — those are imported or
read, never edited. New files only: `k13_*.py`, `test_k13_*.py`, the
receipts, this document.

## 0. The chain this stage closes, the enabling fact, and the ruling

The side-by-side dissection (`RESULT_SIDE_BY_SIDE_0239.md` §5) receipted the
winner's residual as **C5, the store's write/read form**: byte-identity
CORRECT on 0.979/token, **reads over ~5.6 effective rows of ~27 live** —
the read-dilution finding — and 17/24 stores near-decorative at eval. The
follow-up levers tested since (K11 per-head scalar gates, K11B GDN2-granular
per-channel gates, K12 learned placement) all receipted honest nulls at
parity: no mechanism number moved. The one C5 sub-lever never isolated is
the read's **normalization law itself**. Our read is COMPETITIVE — a
softmax over live rows (temperature-tuned, `read_scale` 2.0), so every row
competes for a probability simplex and the mass dilutes across the live
set. GDN2's read is ADDITIVE — `out_t = S_t^T q_t` with an output gate:
every matching direction contributes, magnitude carries information,
nothing competes. The read form is transplantable WITHOUT changing the
store or the ops: it is a value-channel change, and this lane is that
transplant.

**The enabling fact:** the read already feeds NO decision. The K10 decision
chain runs on the untempered key-channel similarity; the read weights are
a value channel of the recorded schedule (P1's class: recorded channels
only). Swapping the softmax for the raw score changes what the reader
emits, and NOTHING else — the discrete op sequence is bit-invariant by
construction (receipted at probe scale, §2.2). The K7 full-population read
law governed the softmax's in-kernel normalization (ONE division by the
full-population exact mass, never a per-block renormalisation); the
additive read needs its own in-kernel accumulation — a plain
defined-accumulation sum over live rows with NO denominator at all, so the
K7 obstruction (per-block renormalisation) cannot arise by construction:
there is no renormalisation anywhere in the form. The read keeps the K6
floor-channel discipline (the read feeds no decision, so no flip can come
of it).

**Supervisor ruling received and incorporated before this document was
committed** (the task's escalate-before-commit rule):

* **R13-1 — the score form: the banked score channel VERBATIM.**
  `w_j = read_scale · cos(q_key, K_j)` on the key channel, live-gated,
  softmax REMOVED, nothing else in the score path changed.
  `read_scale = 2.0` is disclosed as **INHERITED-FROM-BANKED, not
  re-tuned** — under an additive read it is a gain rather than a
  temperature, absorbed by the downstream learned projection within steps.
  Rationale: K13's question is whether the residual's owner is the
  NORMALIZATION LAW (competitive simplex vs additive superposition), and
  the verbatim score channel isolates exactly that — the score channel is
  bit-identical to the banked read's, the decisions (already on untempered
  cos) are untouched, and the dilution instrument measures the same
  quantity through both normalization laws. The alternatives were refused:
  untempered cos would conflate the form change with a gain change; the raw
  dot `q_key · K_j` would conflate it with two learned magnitudes.
* **R13-2 — strength-drop, CONFIRMED.** The banked `||S_j||` multiplier
  goes WITH the softmax: magnitude enters through `v_j = S[j]` itself
  (GDN2-faithful — `out = S^T q` has no separate strength channel). A POP'd
  row contributes exactly 0 through `S[j] = 0`; the live mask still gates
  `w_j` (dead rows' keys are zeroed, and the masked score is not a number
  the additive read may consume).
* **R13-3 — the read gate, CONFIRMED: the committed k11b `g_proj` pattern
  verbatim.** `r_vec = 1 + m_o·sigmoid(g_proj(x))` with
  `g_proj = Sequential(Linear(width, d_v, bias=False),
  Linear(d_v, heads·d_v, bias=True))` zero-born and `m_o` a per-head scalar
  born at EXACT zero, applied `reads ⊙ r_vec` before the output projection,
  OUTSIDE the scan (K11/K11B's read-out convention). Born at exact 1.
* **R13-4 — the M-R1 instrument under signed weights, CONFIRMED.**
  Effective rows := `e^H` of the **|w|-mass distribution**
  (`|w_j| / Σ_j |w_j|` over live rows), reported with the top-row |w|-mass
  share (the collapse detector) and the mean live-row count.
* **R13-5 — the birth-magnitude diagnostic, REQUIRED.** The additive read
  at birth produces ~O(live-rows)× read-output magnitudes (no simplex
  renormalisation); the arm's receipt MUST carry the per-head read-output
  RMS at birth (additive arm vs the banked read's, same seed, same batch),
  so that if the arm loses, the magnitude-blowup-vs-form attribution is
  inspectable rather than guessed, and any early-loss divergence is
  reported with this diagnostic attached. Pre-stated here: the first
  training steps adapt the downstream scale; **NO birth parity is possible
  (a different function)** — the step-1 train loss will NOT match the
  banked row's, and that is disclosed as the machine-variant consequence
  (the K10 tradition: like separated keys were, a different machine), not
  excused.

## 1. The architecture under test

**The K10 winner (per-token + separated keys, the tuned config) with its
read form replaced: `y = gate ⊙ (Σ_j w_j · v_j)` over live rows, `w_j` the
banked score channel's raw similarity, `gate` the k11b `g_proj` per-token
read gate** — at the matched training surface (the banked winner lineage:
hybrid Muon at 4× the tuned AdamW LR, the gate projections on the AdamW
side, the DECAY gate's own convention). Same five-op schedule, same store,
same thresholds, same DECAY, same key channel, same projections, same
recorded-schedule discipline; only the reader's normalization law and the
gate change.

Stated exactly once, the registered read:

```
score_j = read_scale · cos(q_key(t), K_j)        # the banked score channel, VERBATIM
w_j     = score_j  if slot j is live else 0       # raw score; NO softmax, NO ||S_j|| factor
y_raw   = Σ_j w_j · S[j]                          # in-kernel, the floor-channel rank sum
y       = r_vec ⊙ y_raw                           # the g_proj gate, outside the scan
r_vec   = 1 + m_o·sigmoid(g_proj(x_t))            # zero-born, exact 1 at birth
```

Everything else — decisions, payload lanes, MERGE blend, DECAY (values
only), the key lane, POP/MERGE zeroing — is the K10 winner VERBATIM. The
score channel is bit-identical to the banked read's own (the same
defined-order folds, the same `read_scale`, the same live mask); the change
is the normalization law and nothing else. Decisions remain inviolable
(P2): the read form cannot feed op selection, and the recorded op sequence
is bit-invariant to it — receipted at probe scale as the
**decisions-bit-identical** birth receipt (§2.2), which replaces the
0-flip-at-birth receipt a mimetic-neutral birth would carry.

### 1.5 The parameter accounting

The read gate is the k11b read channel's own shape and count: per mixer,
`g_proj` = `width·d_v` (576·128 = 73,728) + `d_v·heads·d_v` + `heads·d_v`
(128·256 + 256 = 33,024) plus the per-head scalar `m_o` (2) = **106,754**.
At depth 24: **+2,562,096** parameters → 152,543,904 total, **+1.70 %** over
the audited 149,981,808. K11B's R4 precedent applies verbatim: the 0.1 %
capacity tolerance is DROPPED for a real GDN2-granular transplant (a gate
channel cannot hold it); the delta is stated, asserted by the probe, and
the capacity asymmetry vs GDN2's 18.3 % gate share is the finding, not a
caveat to be dropped. The read-form change itself is parameter-FREE (the
score channel and the sum are the banked read's own machinery).

## 2. The protocol, the arms, the birth receipts

### 2.1 The arms and the protocol

**One arm — `additive` — at BOTH seeds (3401, 3402), both primary.** No
birth parity is possible, so the second seed is not a confirmation luxury
but primary evidence; the pre-stated M-2 verdict fires on the seed-matched
comparisons at both seeds where the anchor exists. Every arm is the banked
winner build VERBATIM (`velocity_ladder._build_k10`, the tuned winner
config, imported) then `convert_model_to_additive_read` (the K10 conversion
pattern: a shallow re-wrap sharing every parameter object, the `g_proj`
gate added zero-born), at the matched-exposure probe protocol: the
20,022,866-target landing, 180 constant-LR steps, the registered flagship
stream, the document-disjoint calibration eval (7 windows / 14,336
targets), the honest clock — then the CORRECTED scoring (the standing
amendments, imported verbatim): 24-step linear cooldown PRIMARY, EMA-0.99
at step 180 SECONDARY (ranking-only), constant-180 retained for banked
comparability.

**Both comparison anchors are quoted everywhere:** the constant-LR banked
best **1.4899048124040877** (seed 3402; the same config's seed-3401 row is
**1.4907409463609969**) and the corrected post-cooldown banked best
**1.4338248457227434** (seed 3402). The GDN2 context rows (constant
1.3951, post-cooldown **1.3521734476089478**, EMA 1.3803) are quoted as
the gap reference: the corrected-gap is **0.0816 nats** (1.4338 − 1.3522),
the residual C5 owns after the optimizer-recipe and instrument alibis were
removed (`RESULT_PROTOCOL_CORRECTION.md`).

### 2.2 The birth receipts (the machine-variant's own)

1. **DECISIONS-BIT-IDENTICAL at birth** (the read form's 0-flip analog):
   on the same seed, same build order, same first batch, the additive
   arm's recorded schedule — every `slot`, `op`, `erase`, `live_after` and
   `payload` lane, and the pre-decision liveness — must be `torch.equal`
   to the banked winner twin's (a K10 build with the banked read), layer
   for layer. The read form cannot touch the discrete schedule; this
   receipts that at probe scale.
2. **THE BIRTH-MAGNITUDE DIAGNOSTIC (R13-5)**: the per-head read-output RMS
   of the additive arm vs the banked read's, same seed, same batch, at
   birth — the honest number for the ~O(live-rows)× magnitude consequence
   of removing the simplex, attached to any early-loss divergence report.
3. **No step-1 loss parity, disclosed**: the step-1 train loss is a
   DIFFERENT function's value and will not match the banked row's
   (5.785070835425993 at 3401 / 5.768619428012168 at 3402); asserting it
   would be a category error, and the two receipts above carry the birth
   honesty instead.
4. **The gate is born at exact 1** (`m_o` = 0, `g_proj` zero-born):
   at birth the arm is the PURE additive read — the gate's growth (if any)
   is training's own verdict, reported per layer as gate telemetry (the
   k11b readout convention: distance from 1, min/max).

### 2.3 The mechanism instrument, per arm, at BOTH states

The K10/K11 instrument, run at the step-180 constant snapshot (the
banked-comparable state) AND the post-cooldown final (the primary state):
op rates, occupancy (mean live slots), the identity-rewrite share
(`_walk_stats`' raw-payload cosine instrument, anchor-continuous), plus
the additive read's own channels under R13-4: effective rows `e^H` of the
|w|-mass distribution, the top-row |w|-mass share, the mean live-row
count. The decorative-store sweep (`side_by_side.k10_read_mask_sweep`,
imported verbatim: each layer's store read masked at eval, the dNLL
profile) at both states — the receipt for whether the additive read makes
more stores load-bearing. The gate telemetry (per-layer `r_vec` distance
from 1) at both states.

Anchors, both quoted with their eras (the K11 convention): the
side-by-side's ~5.6 effective rows of ~27 live (the fused machine, the
read-dilution finding's own number) and the banked same-config anchors
loaded at run time from the banked receipts (entropy 1.5150/1.6002 →
e^H ≈ 4.55/4.95, occupancy 33.5–35.2, identity 0.1222/0.1180, the
AdamW-era sweep's 7 load-bearing of 24 / 17 near-decorative; the
muon4x-era sweep was never banked — that caveat stands).

### 2.4 The engine gates (pre-registration evidence, in scope now)

New file `k13_additive_read.py`, committed engines imported verbatim:

* **the eager oracle** `additive_oracle_scan` — the parity fixed point,
  differentiable through the key channel and the read (the raw-score
  weights are ATTACHED; the decisions stay detached, the standing
  doctrine). The K10 oracle with the read block replaced per §1 and
  nothing else touched.
* **`_k13_seq_fwd_kernel`** — K10's persistent whole-window walk with the
  read block replaced: `w = where(live, score, 0)`, the read the same
  native rank-axis floor sum; the exp/max/denominator of the softmax are
  GONE (no renormalisation exists to obstruct the K7 law).
* **`_k13_bwd_prep_kernel`** — K10's token-parallel prep with the read
  adjoint's coefficients replaced: `dscores_j = dw_j` on live rows (the
  identity — `w_j = score_j`), the strength channel written as EXACT
  ZEROS, `z`/`duu`/`dc`/`dnk` on the new `dscores` (the same clamp-select
  identities; the decomposition is form-agnostic).
* **the backward walks imported VERBATIM**: K10's key-traj replay, value
  walk and key walk kernels are consumed unchanged — the value walk's
  strength term dies by the zero `dstrength` channel (IEEE add of 0.0 is
  exact), its direct term consumes the score weights; the key walk's
  c-term/nk-term consume the prep's channels. No committed kernel is
  edited; a lane-specific backward body in the new file wires them.
* **gate A**: the kernel vs the oracle — decisions, payload lanes, read
  WEIGHTS, state, liveness, pre-decision liveness AND the key table
  bit-exact (`torch.equal`), 0 flips, at the registered shape (bf16
  autocast corners included — the K11B lesson: bit-parity fixtures go
  through the production expression under autocast).
* **gate B**: all six gradients (`dpayload`, `dwrite_keys`,
  `dquery_keys`, `ddecay`, `dstate`, `dkeys`) vs the oracle's own autograd
  at the measured fp32-vs-fp64 floors, `NOISE_FACTOR`× — with the **fp64
  gradcheck of the oracle THROUGH THE ADDITIVE READ PATH mandatory** in
  the suite (every adjoint branch, including the read's direct term and
  the score path's query-key/key-table terms).
* **refusals and determinism**: the K10 wrapper's own (append-only
  refused, growth refused, rank prefix refused by name; a store the
  persistent walk cannot holds takes the oracle verbatim); run-to-run
  determinism receipted.

## 3. The pre-stated predictions (with refutation clauses)

Stated BEFORE any probe measurement, against the anchors of §2.1/§2.3:

* **M-R1 — the read-dilution instrument MOVES.** The additive read's
  effective rows (|w|-mass `e^H`, R13-4) rise from the banked ~5 (the
  side-by-side's ~5.6, the same-config 4.55/4.95) TOWARD the live-row
  count — the competitive→additive signature: nothing competes for a
  simplex. **Bar:** effective rows ≥ 2× the seed-matched banked anchor at
  BOTH seeds. **Refuted if** the additive read collapses to few dominant
  rows anyway (below the bar at both seeds — the top-row |w|-mass share
  is the collapse detector, and a collapse-refutation REFINES the C5 read
  to "dilution is not the softmax's doing").
* **M-R2 — the NLL question, the attribution this stage exists for.** If
  the additive arm closes ≥ 0.02 of the ~0.08 corrected gap (post-cooldown
  primary, seed 3402 vs the 1.4338 anchor: arm ≤ **1.4138**; the
  constant-180 and EMA scorings quoted as cross-checks at both seeds),
  **the C5 attribution refines to THE READ FORM (competitive softmax
  dilution)**, the terminal reading is amended, and the read form joins
  the leaner registration. **If parity** (every seed-matched delta inside
  the 0.075-nat endpoint-variance band), **the read form is NOT the
  residual's owner and the superposition story stands with one more
  receipt.** No third outcome is pre-stated: a WORSE landing (outside the
  band on the losing side) is reported with the birth-magnitude diagnostic
  attached (R13-5) — the magnitude-blowup-vs-form attribution is then an
  inspectable question for a named follow-up (a gain-tuned read), not a
  post-hoc rescue.
* **M-R3 — the mechanism instrument, reported per arm** (P2, no win/lose
  bar): identity share, occupancy, op mix at both states, vs both anchor
  sets; the decorative-store sweep's dNLL profile compared — the additive
  read MAY make more stores load-bearing (the sweep is the receipt); the
  gate telemetry (did training pull `r_vec` off 1, per layer).
* **Pre-state: NO directional prediction on the headline.** The arm is a
  different machine's read; the honest pre-state is that both directions
  are live outcomes, and §M-R2's two-sided clause is the pre-committed
  reading.

**The refutation clause, stated verbatim before the arms:** if NO
mechanism number moves AND the NLL does not improve beyond parity, the
read-form lever does not own the residual and that is reported as refuted,
not excused — the terminal reading (superposition, the C5 residual owned by
the write geometry) then stands on one more null receipt.

## 4. Claim set (registered, pending sign-off)

1. **Probe scale only** (20 M-target matched exposure, 180+24 steps), one
   protocol, both seeds; no board-scale (2700 s) claim before the operator
   signs this document off.
2. Any NLL delta is reported against BOTH anchors, seed-matched, with the
   0.075-nat endpoint-variance band and the winner's own seed delta
   (0.0119) quoted; nothing inside the band is bankable.
3. The engine claims (gate A bit-parity, gate B floors, the fp64
   gradcheck) are the lane's own receipts, in `receipts/k13_gates.json` +
   `K13_ADDITIVE_READ.md`; the arms' receipts land per arm under
   `k13_runs/additive_seed<seed>/`, committed per artifact (the ceiling is
   real; a committed partial beats an uncommitted complete).
4. The card discipline: cards 0/1 (this lane's own pool); cards 2/3 are
   the signed X46 sweep's, card 4 the detached emender run — untouched;
   card 7 forbidden.

## 5. Deliverables

| file | role |
| --- | --- |
| `PREREGISTRATION_ADDITIVE_READ.md` | this DRAFT (committed before any probe measurement) |
| `k13_additive_read.py` | the oracle, the two lane kernels, the imported-walk backward body, the op pair, the mixer and its conversion |
| `test_k13_additive_read.py` | gates A and B (the fp64 gradcheck through the additive path), the birth receipts' fixtures, refusals, determinism |
| `k13_gate_receipt.py` | the gate receipts into `receipts/k13_gates.json` |
| `k13_probe_additive_read.py` | the arm runner + the assemble (predictions evaluated against the anchors) |
| `run_k13_additive_read_leased.sh` | the detached leased driver (cards 0/1) |
| `K13_ADDITIVE_READ.md` / `RESULT_ADDITIVE_READ.md` | the engine document / the arms' result document |

## 6. Commands

```bash
# the suite (the fp64 gradcheck through the additive read path is mandatory):
/home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k13_additive_read.py -x
# the gate receipts:
/home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k13_gate_receipt
# one arm (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k13_additive_read_leased.sh additive 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k13_runs/\
additive_seed3401.console.txt 2>&1 < /dev/null &
# the assemble:
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.\
x45_matrix_emender.k13_probe_additive_read assemble
```
