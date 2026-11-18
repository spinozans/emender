# Pre-registration draft — LEARNED PLACEMENT POLICY (K12): the APPEND's free-slot choice made a learned policy on the dead-key residue

**Date:** 2026-11-23 · **Status: DRAFT, operator sign-off pending.** This is a
NEW registration (it changes WHICH free slot an APPEND lands in — the op-set
audit's ALLOCATION half, the one primitive GDN2 lacks and our machine had as a
fixed rule), written and committed before any probe measurement is claimed.
Probe-scale engineering checks (parity gates, gradchecks, the estimator
receipts, memory, timing) are pre-registration evidence and in scope now;
**no board-scale (2700 s) claim is run before the operator signs this
document off.** No `k10_*`/`k11_*` source, PREREGISTRATION, `lattice.py` or
`matched_arms.py` is edited — K10's spec helpers, its backward body, its
selection chain and its mixer/conversion are IMPORTED; the K11 lane (cards
0/1) is untouched.

## 0. The gap this stage closes, and the structural facts that size it

The op-set audit (`RESULT_OP_SET_AUDIT.md` 1.1/3) named it: our APPEND's
WRITE half is instantiated (the rank-1 lane), but its ALLOCATION half — WHICH
free slot the new row lands in — is the fixed first-free-ordinal rule
(`free_slot = the lowest free index`), a rule the model cannot choose.  The
operator named the lever: a learned placement policy, natural form **"write
where the evicted content lived"** — score the free slots by the similarity
of the new write key to the DEAD row's last-key residue — generalising to
learned ADDRESS LOCALITY (related content clustering in slot space).

Three structural facts, stated BEFORE the probe because they size every
prediction honestly:

* **F1 — the decision window is rare.**  Placement can differ from
  first-free ONLY when ≥ 2 slots are simultaneously free.  In this op chain
  POP requires a FULL store (it cannot create a second free slot), so the
  only window is **a POP-then-MERGE (or MERGE-then-MERGE) sequence before an
  APPEND consumes a slot**.  The banked K10-keys machine's op mix (seed 3401:
  CORRECT 0.97939, APPEND 0.01748, MERGE 0.00313, POP **0.00000** per
  token) makes that window rare: deaths are MERGE-only, and an APPEND
  consumes the freed slot within ~57 tokens on average.  The honest prior
  on this lever's NLL headroom is SMALL, and the pre-stated band below is
  sized accordingly.
* **F2 — the zero-gradient finding (the supervisor's own ruling, receipted
  prominently).**  Under P2+P3 the placement effect is purely DISCRETE
  (which free slot receives the APPEND), and the recorded-schedule doctrine
  makes the slot a backward constant — so the STRUCTURAL gradient to the
  placement weight is EXACTLY ZERO (the eager oracle's autograd returns
  `dL/dw == 0` on every fixture; asserted in the tests and in the gate
  receipt).  **"The model can learn where to write" therefore REQUIRES an
  admitted estimator; that requirement is itself a structural fact about
  this architecture** — the same wall the zero-learned-management doctrine
  built Paper 1 on, now surfaced from the placement side.
* **F3 — the falsifier is bit-exact by construction.**  The placement weight
  is born at exact `+0.0` and the score spec adds one `+0.0`
  (IEEE: `-0.0 + +0.0 = +0.0`), so at birth EVERY free slot scores exactly
  `+0.0` and the argmax's lowest-index tie-break reproduces K10's first-free
  rule BIT-EXACTLY — and stays bit-equal FOREVER while `w` is frozen at
  `+0.0` (the never-fires falsifier control; a trained 3-step receipt in the
  tests, birth-parity 0 flips over 17.4 M lanes vs K10's own launch in the
  gate receipt).

## 1. The semantics (pre-stated; the supervisor's rulings cited)

The K10 separated-keys machine (per-token, key_dim 16, mimetic key birth,
free drift, keys undecayed — that draft's semantics verbatim, its pending
sign-off inherited) with ONE new recorded channel and ONE new choice:

1. **The residue channel (P1).**  `Res` (per head, `rank x key_dim`): at a
   row's DEATH token — POP's target row, MERGE's source row —
   `Res[j] := K[j]`, the dying row's last key, recorded PRE-CLEAR.  Never-used
   slots keep a zero residue.  The residue is state dependence EXACTLY through
   a recorded channel: the placement scores consume it, the recorded
   liveness, and the current token's write key — nothing else.
2. **The placement score and the birth tie-break (P5).**  At an APPEND
   decision the score of a FREE slot `j` is
   `s_j = w * cos(wk_t, Res_j) + (+0.0)` in the K10 defined-order arithmetic
   (the same folds, EPS-clamped norms — FINITE at a zero residue), and the
   APPEND lands in `argmax over free slots of s_j`, lowest index wins ties —
   the lane's established first-max semantics, the SAME construction in the
   oracle and the kernel (gate A: bit-equal at `w` ∈ {0, 0.5} on every
   channel, the residue table included).  `w` is ONE per-head scalar per
   mixer born at exact `+0.0` (**+48 parameters** at the registered shape —
   24 mixers × 2 heads — stated, asserted by the probe, not hidden).
3. **P2 — decisions inviolable.**  Placement selects WHICH free slot an
   APPEND lands in; it never touches the op code, the thresholds, the
   payload, the key values, the liveness arithmetic, or any live row (P3:
   the APPEND target is a free slot by definition).  The discrete op SEQUENCE
   is invariant to placement in the sense the lane receipts: op/erase/
   live_after are functions of the op chain alone; only the APPEND lane's
   slot field can move, and only among free slots.
4. **The admitted estimator (the supervisor's option (a), approved as
   specified; option (c) — attaching the score inputs — REJECTED for this
   registration: it amends the detached-decision doctrine and muddies
   mechanism attribution; it stays the named follow-up only under outcome
   (ii) below).**  The kernel path's backward is K10's own
   `_k10_backward_body` imported VERBATIM (the six value/key gradients are
   UNCHANGED — the recorded schedule fully determines them) plus ONE small
   reduction: at each APPEND token, with the write-content sensitivity
   `g_t = <dpayload_t, p_t>` (the EXISTING value walk's own output — no new
   walk), the placement scores `s_j = w * sim_j` over free slots and
   `p = softmax(s over free)`:

   `dL/dw = Σ_t g_t · p_{j*} · (sim_{j*} − Σ_j p_j · sim_j)`.

   **The estimator's bias, disclosed verbatim:** it captures the
   write-content sensitivity of the chosen slot, NOT the full trajectory
   sensitivity of the slot choice (which rows become neighbours, which
   subsequent reads and matches change).  The score inputs are DETACHED (the
   write key is decision-side, the residue a detached recording): ONLY the
   per-head `w` learns.  The gradient reaches the zero-born `w` at step 0
   (P5's discipline, receipted).  The oracle (refusal) path keeps the HONEST
   hard semantics — its `w`-gradient is the structural zero of F2 — and the
   probe asserts the kernel path is the path taken at the registered shape.
5. **The arms (the supervisor's ruling adds the bracket).**  Besides the
   zero-born LEARNED arm, a small FIXED-`w` bracket {+0.25, +0.5, +1.0}
   (positive side only — the natural "write where the evicted content lived"
   directions) as control arms at the same protocol: they distinguish "the
   surrogate is underpowered" from "placement is not load-bearing" — the
   fixed arms need no estimator at all.

## 2. The probe protocol (the lane's matched-exposure discipline, the
corrected scoring instrument)

Every arm runs the banked K10-keys probe protocol VERBATIM —
`matched_arms.build_matrix("matrix_ops", 128, seed)` + the parameter-neutral
separated-keys conversion (the audited 149,981,808; the placement arms ADD
48, asserted) + the registered flagship stream + the same data order + the
same AdamW (the banked row's own optimizer config, fused) + the same warmup,
grad clip and honest clock — to the matched-exposure landing (20,022,866
targets, 180 constant-LR steps), then the CORRECTED SCORING
(`RESULT_PROTOCOL_CORRECTION.md`, the standing discipline):

* **PRIMARY** — the post-COOLDOWN eval: a 24-step linear cooldown
  (13.3 % of 180; factor `(25−k)/24`, never zero) continues the same stream,
  then the document-disjoint calibration eval (7 windows / 14,336 targets).
* **SECONDARY (ranking-only)** — the EMA-0.99 iterate at step 180.
* **Banked comparability** — the constant-LR step-180 eval, retained; the
  BASELINE arm is the banked K10-keys config itself re-scored, so its
  step-180 eval must reproduce the banked **1.841756020273481** and its
  180 constant-phase step losses must be bit-identical to the banked run's
  own `training.jsonl` (asserted in-code, the protocol-correction lane's
  strongest receipt); the LEARNED arm's step-1 loss must equal the banked
  step-1 loss bit-identically (the 0-flip-at-birth bar at probe scale).

**Arms (seed 3401 unless the second-seed rule fires):**

| arm | machine | `w` |
|---|---|---|
| `baseline` | the banked K10-keys config (the 1.8418 lineage) | — (no placement channel) |
| `placement_learned` | + the K12 module | zero-born, trainable via the estimator |
| `placement_fixed0p25` | + the K12 module | pinned 0.25, frozen |
| `placement_fixed0p5` | + the K12 module | pinned 0.5, frozen |
| `placement_fixed1p0` | + the K12 module | pinned 1.0, frozen |

**Second seed:** on any arm whose post-cooldown NLL beats the baseline's by
≥ 0.01 nats at seed 3401 (the pre-stated win bar; anything smaller is inside
the lane's 0.075-nat endpoint-variance band and is reported as directional,
not banked), that arm re-runs at seed 3402.

Mechanism telemetry on every arm (the K10 probe's instruments — op rates,
per-token occupancy, read entropy, identity share, the KEY-COLLAPSE watch —
imported) PLUS this registration's own: the placement FIRE RATE (APPENDs
whose slot differs from first-free), the chosen-slot residue similarity vs
the free-slot mean (the margin), the learned `w` per layer (the learned
arm), the ADDRESS-LOCALITY instrument (the Pearson correlation between slot
distance |j−j'| and key cosine over live-row pairs, per layer), and the
read-mask sweep (`side_by_side.k10_read_mask_sweep`, the C4 diagnostic).

## 3. PRE-REGISTERED MECHANISM PREDICTIONS (committed before the probe)

The anchors are the BANKED K10-keys probe row (seed 3401,
`receipts/k10_probe.json`): eval NLL **1.841756020273481**, occupancy
**24.03** live slots, read entropy **1.6541** nats (≈ **5.23** effective
rows), CORRECT **0.97939**, APPEND **0.01748**, MERGE **0.00313**, POP
**0.00000** per token, identity share **0.1032**; and the side-by-side's
AdamW-era read-mask anchor: **7 of 24** layers with ΔNLL ≥ 0.02, **17**
near-decorative (|Δ| ≤ 0.0125) — stated with its caveat (one machine back;
the K12 baseline re-measures its own sweep under the same instrument).

* **P-1 (NLL, the primary claim).**  At least one placement arm's
  post-cooldown NLL improves on the baseline's post-cooldown NLL at seed
  3401.  Band, honestly wide and sized by F1 (the decision window is a
  small fraction of a percent of tokens): **improvement in [0.0, 0.03]
  nats**.  The win bar for the second seed is ≥ 0.01 nats; improvements
  inside the 0.075 endpoint band are reported as directional, never banked.
  Refuted if every placement arm is worse than the baseline.
* **P-2 (address locality forms — the lever's own mechanism).**  The
  live-pair slot-distance/key-cosine Pearson `r`, mean over layers: the
  placement arms go materially NEGATIVE relative to the baseline's own
  measured `r` (near slots hold similar keys): **Δr ≤ −0.02**; unchanged if
  |Δr| ≤ 0.01.  The direct receipts read with it: the fire rate > 0 on
  multi-free windows, the chosen-residue margin > 0, and (the learned arm)
  `|w|` materially off zero in some layers.
* **P-3 (read-back sharpness).**  Effective rows `e^H` from the read
  entropy: **rises ≥ 0.5 rows** over the baseline's own re-scored value
  (the banked row's 5.23 cited); unchanged within ±0.25 rows; refuted if
  lower.
* **P-4 (decorative stores).**  Layers with read-mask ΔNLL ≥ 0.02:
  **≥ 10 of 24** (the baseline's own count is the matched anchor; the
  banked era's 7-of-24 / 17-decorative cited as the one-machine-back
  anchor); refuted if not above the baseline's own count.
* **P-5 (MERGE may revive — near-address slots coalesce).**  MERGE rate
  rises to **≥ 0.01 per token** (≈ 3× the banked 0.00313) — this would be
  the first positive MERGE receipt at per-token granularity.  Reported
  honestly either way; no other band.
* **The three-way read (the supervisor's ruling, pre-stated).**
  (i) learned-`w` ≈ the best fixed point or better ⇒ the estimator works,
  placement is learned; (ii) learned-`w` stalls near 0 while a fixed `w`
  helps ⇒ the surrogate's bias is binding — receipt it, option (c) becomes
  the named follow-up; (iii) nothing helps anywhere ⇒ **the honest null:
  free-slot placement is decorative on natural text, and C5's write/read
  form owns the residual.**  All three are results.
* **Refutation clause.**  If the mechanism numbers do not move while NLL
  does (or vice versa), the dissociation is reported honestly.  A null is a
  result — it would say free-slot placement is not load-bearing on natural
  text.

## 4. Claim set (registered, pending sign-off)

* **S1 (quality).**  The arms train at the registered shape under the
  corrected protocol; the baseline reproduces the banked row's constant-180
  eval bit-identically; P-1 evaluated with the receipts, the win bar and
  the second-seed rule stated above.
* **S2 (mechanism).**  P-2/P-3/P-4/P-5 evaluated against the matched
  anchors, each reported met/not-met/refuted; the three-way read applied
  verbatim.
* **S3 (algebra).**  The gates: birth-parity bit-equality (0 flips vs
  K10's own launch), parity at firing `w`, the frozen-`w` falsifier, the
  zero-gradient finding, the estimator-equals-its-reference receipt, the
  six-gradients-within-floors receipt, and the K10 suite green with the
  module loaded — all in `receipts/k12_gates.json` and the tests, BEFORE
  any probe claim.
* **S4 (cost).**  The honest walls/tok/s per arm against the baseline's own
  (gate D: the placement channel costs ~5 % forward at the registered
  shape — the residue tile; the placement math runs only under a scalar
  `if appendable` guard).

## 5. Honest scope and caveats

* Single protocol, probe scale, seed 3401 per arm (+ the second seed on a
  win); the 20 M matched exposure; the 0.075-nat endpoint-variance band
  applies to every NLL endpoint here.
* The placement arms ADD 48 parameters (+0.00003 %) — stated; the
  comparison is parameter-matched to within 0.0001 %.
* The estimator is a SURROGATE, not the true adjoint (F2 + §1.4); the
  learned arm's `w` trajectory is read with that bias disclosed, and the
  fixed-`w` bracket is the estimator-free control.
* F1 sizes everything: the placement decision window is the
  multi-free-slot APPEND — structurally rare in the banked op mix (POP 0 %,
  MERGE 0.31 %).  The probe measures whether that rare window is
  load-bearing; it does not claim the lever is large.
* One engineering finding pre-stated for the record (found by this lane's
  gate fixtures, fixed in THIS module's oracle copy only; no `k10_*`/`k11_*`
  file edited): the K10 EAGER ORACLE's MERGE branch has a LATENT MULTI-LANE
  bug — its `if merge.any()` index-put zeroes EVERY (batch, head) lane's
  second row whenever ANY lane merges.  It is unreachable by K10's own gate
  fixtures (single-lane merge fixtures; multi-lane fixtures produce zero
  merges) and inert in production (training takes the kernel path, which
  guards per lane); all banked K10 receipts stand.  The K11 oracle was
  copied from the same pattern and likely carries the same latent bug, with
  the same inertness argument — flagged to the operator, not touched here.
* The board-scale (2700 s) run is gated on the operator's sign-off of this
  document (and the K10/K11 drafts it inherits).

## 6. Files

`k12_placement.py` (the oracle, the forward kernel, the estimator's
reduction, the op pair, the wrapper, the mixer, the conversion),
`test_k12_placement.py` (18 tests: gates A/B, the finding, the estimator,
the falsifier, the heterogeneous-lane pin, refusals, determinism, the
conversion accounting), `k12_gate_receipt.py` +
`receipts/k12_gates.json` (the registered-shape gates),
`k12_probe_placement.py` + `run_k12_placement_leased.sh` (the probe and
its detached leased driver, cards 2/3), `receipts/k12_placement_*.json`
(the arm receipts + the assemble summary), `K12_PLACEMENT.md`,
`RESULT_PLACEMENT.md`.  No `k10_*`/`k11_*` source is edited.
