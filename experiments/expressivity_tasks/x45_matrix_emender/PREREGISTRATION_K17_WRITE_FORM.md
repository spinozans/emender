Pre-registration draft — THE WRITE FORM UNDER THE ADDITIVE READ (K17):
the dense gated write (G2's form, reborn at the per-channel granularity) and
the read-gain bracket, on the COMMITTED composed machine

**Date:** 2026-11-25 · **Status: DRAFT, operator sign-off pending.** This is a
NEW registration (it changes the store's CONTINUOUS write channel and one
frozen config constant; the algebra's discrete op selection is untouched),
written and committed BEFORE any probe measurement is claimed. Probe-scale
engineering checks (parity gates, gradchecks, memory, timing) are
pre-registration evidence and are in scope now; **no board-scale (2700 s)
claim is run before the operator signs this document off.** It does not
touch any `k10_*`/`k11_*`/`k11b_*`/`k13_*`/`k15_*`/`k16_*` source, any
committed PREREGISTRATION, `lattice.py`, or GDN2 model code — those are
imported, never edited.

## 0. The chain this stage closes

The K15 adjudication (the supervisor's ruling of 2026-11-24) named the
WRITE FORM (event writes vs dense superposition) as the LAST structural
candidate owner of the composed machine's remaining ~0.03 vs GDN2; the K16
config bracket (the ruling's follow-up 1) then DISCHARGED the config
alibi on every axis the program knows how to search and left the residual
UNCHANGED (+0.0289/+0.0311 vs GDN2's 1.3521734476089478,
parity-within-band under the supervisor's ruling of 2026-11-25). Two
receipts converge on this lane:

* **The morning's G2 arm HURT on the softmax machine** (+0.007
  post-cooldown vs its own banked baseline, `RESULT_LEARNED_GATES.md`):
  the per-head dense-write channel did not compose with the COMPETITIVE
  read — the softmax's simplex was starved by an extra unnormalised
  writer.
* **The machine changed fundamentally since.** The K13 ADDITIVE read
  (post-cooldown 1.3928/1.3845, ~60% of the corrected gap closed, |w|-mass
  effective rows 3.6×, no collapse) is a DIFFERENT read law: no simplex,
  no competition, magnitude rides the rows themselves. The composed
  machine (K13's read under the K11B p3 per-channel gate stack,
  1.3811/1.3832 post-cooldown, the program's best absolute rows) is the
  machine the write form must now be tested ON. The read/write interface
  is different: **the read law and the write law may compose the way the
  gates and the read law did not.** This is also the K11B registration's
  own named follow-up §7.2 ("the dense-write channel at per-channel
  granularity … a per-channel dense deposit — named, not run") and the
  side-by-side's §6 named lever in its final untested form.

The second arm is bundled because it is cheap and it is the same
interface's other never-re-tuned constant: the read GAIN.

## 1. The architecture under test

**The COMMITTED K15 composed machine VERBATIM** (the banked winner build
`velocity_ladder._build_k10` + `convert_model_to_composition` — the gates
born at the registered scalar equivalents; the machine's own banked rows
1.3811104467936925 / 1.383232252938407 post-cooldown, seeds 3401/3402 —
imported byte for byte, never modified), at the matched training surface
(the banked winner lineage: hybrid Muon at the inherited 4× point, every
gate projection on the AdamW side, the K15/K16 arms' own routing), at the
exact probe protocol (§3). Two levers, each isolated; a combined arm only
if either shows signal (§4).

### 1.1 Arm W — THE DENSE GATED WRITE (the b_proj-delta form; G2's form
reborn on the new interface)

**Every token writes a small learned-gated contribution to its
BEST-MATCH LIVE ROW, as recorded schedule lane 2, while the five ops keep
managing structure** — the K11 G2 semantics, at the composed machine's
per-channel granularity (the K11B §7.2 named follow-up):

```
dense_t = m_d[h] * sigmoid(d_proj(x)) ⊙ p_t     # per (token, head, channel)
```

with `d_proj = Linear(width, heads·d_v, bias=False)` — **the b_proj form
verbatim** (GDN2's own single bias-free write-gate Linear shape; the
composed machine's write-admission channel's own construction) — born at
exact ZERO weight, and `m_d` a per-head scalar born at EXACT zero: the
committed zero-born gate conventions (K11B's), so at birth `dense ≡ 0`
exactly and **the machine is BORN AS THE COMPOSED MACHINE bit for bit**
(the K15 birth discipline; §2). The deposit:

* **targets P3 VERBATIM**: the BEST-MATCH LIVE ROW at decision time
  (argmax over live, regardless of whether it clears `tau_match`); if NO
  live row exists at decision time the deposit goes nowhere (zero —
  writing into a free slot would be an APPEND event and change the
  decision sequence, violating P2); and the target row must STILL BE LIVE
  after lanes 0/1 (a POP can kill it in-token) — the K11 G2 guard;
* **is recorded as schedule lane 2** — `op=APPEND` (additive),
  `slot=best`, `erase=False`, `live_after=True`, `payload=dense_t` —
  applied after lanes 0/1 and the MERGE-source close, so it is ordinary
  schedule data: the imported lane-generic per-channel trajectory replay
  (`channel_persistent_trajectory`) replays it bit-exactly and the
  certified scan theorems cover it as any other scheduled additive write;
* **fires on the live guards ALONE** (never on the deposit's value — a
  value-dependent no-op rule would make the recorded op code a function
  of the deposit and cut the birth gradient, the dead-channel failure P5
  exists to prevent): at the zero birth the deposit is ±0.0, the IEEE
  identity `x + ±0 = x` (the −0.0 corner is receipted by the suite, the
  K11 precedent);
* **rides the VALUE channels only**: the dense lane never touches the
  decision channel, the key lane, the liveness arithmetic or the
  thresholds (P2 — the 0-flip receipt); it composes with the ops honestly
  (at a CORRECT token the op's write lands first and the deposit adds on
  top; at an APPEND token the op allocates the free row while the deposit
  lands on the best EXISTING row; at NONE the deposit is the only write);
* **carries NO write-strength and NO read-out gate of its own**: it
  isolates the dense channel under the composed machine's own p3 gates.

The parameter delta (stated, not hidden; the committed k11b accounting
convention): per mixer `d_proj` 576·256 = **147,456** + 2 (`m_d`) =
**147,458**; at depth 24 **+3,538,992** → **162,177,840** total
(+2.36 % over the audited 149,981,808; the composed machine's 158,638,848
+ p3's own 8,657,040 + the dense channel's 3,538,992). The K11B R4
precedent stands (the 0.1 % tolerance is dropped; GDN2's own gate share is
18.3 % — ours is ~7.3 % including p3, still ~2.5× leaner). Asserted by
the probe, not hidden.

### 1.2 Arm G — THE READ-GAIN TUNE (cheap, bundled; config arms, not
machine variants)

`read_scale` is the one FROZEN constant of the additive read's score
channel that has never been re-tuned for it: under the softmax read it
shaped a simplex (a temperature); under the additive read it is a pure
GAIN on the raw-score weights — linearly absorbed by the downstream
learned projections at convergence, but never re-tuned, and it scales the
store-facing gradients (the effective learning geometry of everything
behind the read).

**A FACTUAL CORRECTION, stated before any arm runs (the receipts are
quoted, not the registration-text folklore):** the composed machine's
ACTUAL inherited gain is the tuned softmax-machine point
`read_scale = 2.7870458737509636` (`optimizer_bakeoff.TUNED_WINNER`, the
CMAES-tuned value every banked build passes to
`velocity_ladder._build_k10`; the "2.0" of the K13 registration text is
the `RoutingPolicy` CLASS DEFAULT, which the banked build overrides —
the receipts' own `config/summary/routing/read_scale = 2.0` rows are the
BASE build's stale summary record, not the machine's policy). The
inherited 2.787 cell's rows are the K15 composition's own at BOTH seeds
(cited, never re-run); it is NOT one of the bracket cells.

The bracket (the task's own numbers): **{1.0, 2.0, 4.0}** — the near
side (1.0), the policy-default read (2.0), and the far side past the
inherited point (4.0) — each a CONFIG arm: the machine byte for byte the
K15 composition, the ONLY difference the policy constant
(`read_scale`), consumed by the committed kernels exactly as the banked
machine consumes it. No birth parity exists or is asserted for these
arms (a different gain constant is a different function at birth — a
config arm like the LR cells, disclosed); the composed machine's own
same-seed rows are the baseline.

### 1.3 Arm C — the combined arm (conditional)

The dense write + the best read-gain cell, ONLY if either lever shows
signal at seed 3401 (§4's pre-stated rule). Registered now so the arm
needs no new sign-off to launch.

### 1.4 The arithmetic spec (inherited verbatim, extended by one lane)

K6/K10/K11B/K15's defined-order spec applies unchanged (balanced binary
tree folds, fp32 decision/read arithmetic, correctly-rounded
`sqrt.rn`/`div.rn`, `mul.rn`/`add.rn` — no FMA contraction). The dense
lane adds ONE `add.rn` per firing token at the best row (`x + dense_t`)
and its recorded schedule fields; the deposit's VALUE is the gate's
output (born at exact zero). The recorded lane-2 payload is the gate's
output, so the recorded schedule replays through the lane-generic kernels
verbatim — the gate never appears in the replay.

## 2. The birth discipline (the K15 receipt, carried and sharpened)

* **BORN AS THE COMPOSED MACHINE, bit for bit**: at the zero birth
  (`d_proj ≡ 0`, `m_d ≡ 0` → `dense ≡ ±0.0`) the K17 forward is
  BIT-IDENTICAL to the K15 composed kernel's own committed launch on
  every shared channel (reads, state, liveness, keys, lanes 0/1 of the
  schedule, the read weights, the pre-decision liveness, the segment
  checkpoints), and the arm's step-1 train loss is BIT-IDENTICAL to the
  K13 additive arm's own same-seed row (the composed machine's nearest
  kin — the same chain the K15/K16 arms assert). The ±0.0 IEEE corner
  (`−0.0 + +0.0 = +0.0` on an exactly-−0.0 row element) is the only
  legal escape, receipted by the suite with its count asserted zero at
  probe scale — escalated, not excused, if it fires.
* **The 0-flip bar (P2)**: the discrete op sequence (lanes 0/1), the key
  table and the liveness are BIT-INVARIANT to EVERY dense deposit value
  at fixed inputs (the K11 receipt carried to the composed machine).
* **The frozen-write falsifier control (MANDATORY, the K11B R4
  precedent)**: one control run — the dense arm with the dense channel's
  parameters frozen at birth (`requires_grad=False`, excluded from the
  optimizer; the routing then equals the composed machine's own split
  exactly). **The bar, stated exactly:** (a) the control's step-1 train
  loss MUST be bit-identical to the K13 additive arm's own same-seed row
  (the forward-at-birth plumbing receipt); (b) the control's landing MUST
  be at parity with the composed machine's same-seed banked row (inside
  the winner's own measured seed-delta scale, 0.0119, on the primary
  scoring). **The honest limitation, stated not hidden (K11B §2.4's
  own):** full-trajectory bit-identity is NOT achievable by construction —
  the dense lane's backward emits an exact-zero payload-gradient
  contribution whose ±0.0 signs are reduction-order artifacts, and the
  difference compounds chaotically through 180 optimizer steps even with
  the channel frozen; the control measures everything-except-dense-
  learning at the band's scale (module creation, optimizer routing,
  dtype promotion, the new kernels' arithmetic), with the per-step loss
  divergence against the composed machine's own training log measured and
  quoted (expected: exact 0 at step 1, ULP-scale growth after).

## 3. The protocol

Every arm: the EXACT probe protocol — the 20,022,866-target
matched-exposure landing at 180 constant-LR steps on the registered
flagship stream (identity asserted), the document-disjoint calibration
eval (recorded-read path), the honest clock, the protocol's own grad-clip
(1.0, error_if_nonfinite), the banked knobs asserted at launch (betas
(0.9, 0.9392), weight decay 0.17422, warmup 45 on both hybrid groups,
microbatch 8), the inherited 4× hybrid-Muon point (K16's own receipt:
already the composed machine's bracket optimum) — THEN the corrected
scoring: the 24-step linear cooldown PRIMARY (post-cooldown calibration
eval), EMA-0.99 at step 180 SECONDARY (ranking-only), constant-180
retained for banked comparability. Seed 3401 first on every arm.

**The anchors (all quoted, none re-run):** the composed machine's OWN
rows at BOTH seeds — post-cooldown **1.3811104467936925 (3401) /
1.383232252938407 (3402)**, constant-180 1.433674931526184 /
1.4341776371002197, EMA 1.424389226096017 / 1.4385861498968942 — are the
BASELINE; GDN2's post-cooldown **1.3521734476089478** is the gap
reference; the K13 additive rows (1.3928/1.3845) and the banked winner's
corrected anchor (1.4338, seed 3402) are context. The bands on every
margin: the 0.075-nat endpoint-variance band and the winner's 0.0119 seed
delta.

**PRE-STATE: NO DIRECTIONAL PREDICTION** is made for either lever (the
K15/K16 precedent; the honest priors point both ways — G2 hurt on the
softmax machine, while the additive read removed the starvation mechanism
that hurt it, and K11B §7.2's granularity argument was never tested).

### 3.1 Mechanism duties (reported as measured, no bar pre-stated)

* **The write-channel telemetry (the C5 instrument)**: per layer — the
  dense channel's mass vs the EVENT writes' mass (mean |dense_t| over
  mean |lane-0 payload| on the carrying ops, and the dense share of the
  total written mass), the dense gate's distance from the zero birth
  (`|dense_t|` scale relative to the payload), the per-head scalar drift,
  and the dense lane's FIRE RATE (the live-guard coverage). This is the
  receipt that says whether the dense channel CARRIED anything at all —
  K11's own honest rows (deposit 0.0015–0.024 of payload scale) are the
  quoted reference.
* **The K15 mechanism instrument UNDER the dense write (the interesting
  receipt)**: |w|-mass effective rows, the top-row-share collapse
  detector, occupancy, the identity-rewrite share, the op mix, and ALL
  THREE p3 gate channels' born-identity drift — imported verbatim,
  re-run on the dense machine's own launch.
* **The decorative-store sweep** (`k10_read_mask_sweep`, imported
  verbatim) at BOTH the step-180 and post-cooldown states.
* The dissociation clause (K11 §4's own): if the mechanism numbers do
  not move while NLL does (or vice versa), the dissociation is reported
  honestly.

## 4. The pre-stated rules (the K13/K15/K16 precedent)

* **W-1 (the compose bar)**: the dense arm's post-cooldown row improves
  on the composed machine's same-seed banked row by **≥ 0.01 at BOTH
  seeds** ⇒ the dense gated write COMPOSES with the additive read; the
  write-form attribution owns that much of the residual and the gap quote
  is updated.
* **W-2 (parity)**: improvement at one seed inside the 0.0119 seed band
  and no replication at the other ⇒ **PARITY-grade evidence** (the C-2
  read-form-dependence verdict's own phrasing) — reported as measured,
  NOT banked as composing.
* **W-3 (interference)**: the dense arm lands WORSE than its same-seed
  baseline beyond the band ⇒ the G2-under-softmax result REPLICATES under
  the additive read (the write form does not compose) — reported as the
  refutation, with the mechanism rows attached (the C5 instrument's
  answer to WHAT the dense channel did).
* **G-1 (the read-gain)**: any gain cell improving the same-seed
  baseline by ≥ 0.01 at 3401 ⇒ the gain owns part of the residual
  (disclosed as a config yield, the K16 convention); inside the seed band
  ⇒ the inherited point stands, receipted.
* **THE COMBINED ARM** fires ONLY if either lever's seed-3401 row
  improves ≥ 0.01 (the signal bar): dense + the best gain cell.
* **SECOND SEEDS (the standing win-bar rule)**: any arm whose seed-3401
  improvement is ≥ 0.01 owes seed 3402 BEFORE any claim.
* **THE BAND DISCIPLINE (the K15/K16 clause, verbatim)**: any arm whose
  post-cooldown row would put the machine AT OR PAST GDN2's
  1.3521734476089478 is reported with BOTH scorings (post-cooldown
  PRIMARY, constant-180 retained) and BOTH seeds before any claim, and
  ESCALATED, not adjudicated here. Any inversion of a standing
  quality conclusion escalates likewise.
* **The final gap-quote phrasing is the supervisor's to adjudicate**
  (the K13 clause-conflict precedent): this lane's assemble states the
  measured rows and the flags.

## 5. The parity gates (committed before any probe measurement)

* **Gate A** — oracle-vs-kernel bit-equality on the dense machine: the
  3-lane schedule fields, the payload lanes, the read WEIGHTS, the
  pre-decision liveness, the exit state/liveness/key table
  (`torch.equal`) in both GATED combinations, the bf16 autocast regime,
  ragged/edge fixtures covering all five ops PLUS the dense lane's guards
  (no-live-row, POP-kills-best, APPEND-token deposit-on-existing-row);
  the read at the measured fp32-vs-fp64 floor; determinism.
* **Gate A2 — the birth receipts**: at the zero birth the K17 forward is
  BIT-IDENTICAL to the K15 composed kernel's own launch on every shared
  channel (the born-as-the-composed-machine receipt); the 0-flip
  decision-invariance receipt under RANDOM dense deposits (P2); the
  mixer-level birth receipt (the converted dense mixer's output
  bit-identical to the composed mixer's at birth).
* **Gate B — MANDATORY: the fp64 gradcheck THROUGH THE DENSE-WRITE PATH**
  (the oracle's every adjoint branch — NONE/APPEND/CORRECT/POP/MERGE, the
  dense lane's own `ddense = dS[best]`, the additive read's raw-score
  weights attached, the per-channel DECAY vector and write gate) — then
  the kernel path's EIGHT gradients (dpayload, ddense, dwrite_keys,
  dquery_keys, ddecay, dgates, dstate, dkeys) within NOISE_FACTOR (8×)
  of the measured fp32-vs-fp64 floor; the 3-lane trajectory replay
  bit-exact against the forward's own segment checkpoints; the zero-born
  scalars' step-0 gradient receipt and the two-step wake-up.
* **Gate D** — the honest walls vs the K15 composed machine's own
  same-session launches (the dense lane's cost, stated not estimated).
* The committed K10/K13/K11B/K15 suites re-run green with the new module
  loaded.

## 6. Honest scope

* Probe scale only (the 20 M matched exposure, 180+24 steps, 7-window
  eval); no board-scale claim before the operator signs off.
* The dense arm ADDS parameters (§1.1): the comparison is config- and
  protocol-matched, and the parameter asymmetry vs GDN2's 18.3 % gate
  share is quoted in every receipt.
* The dense gate is the b_proj FORM (a single bias-free per-channel
  Linear + per-head scalar) — the K11B §7.2 named granularity; richer
  forms (per-slot targets, match-similarity conditioning, the f_proj
  two-layer shape) are named variants, not this registration.
* The gain bracket is three cells of a one-dimensional constant: it is a
  cheap tune, not a search; the inherited 2.787 point is cited, and no
  interpolation claim is made.
* Single-protocol rows; the endpoint-variance band (0.075) and the
  winner's seed delta (0.0119) are quoted with every delta; the
  composed-machine-vs-GDN2 row is a cross-machine context row at the same
  probe protocol, not a controlled comparison.

## 7. Files

`k17_dense_write.py` (the dense oracle, the ONE new forward kernel, the
K17 value-walk backward, the op pair, the wrapper, the mixer, the
conversion, the parameter accounting), `test_k17_dense_write.py` (gates
A/A2/B), `k17_gate_receipt.py` (the registered-shape receipts into
`receipts/k17_gates.json`), `k17_probe_write_form.py` (the arm runner +
the assemble + the mechanism instruments), `run_k17_write_form_leased.sh`
(the detached leased driver, cards 0/1), `K17_WRITE_FORM.md` /
`RESULT_WRITE_FORM.md` (the receipts / the arms), the per-arm receipts
under `k17_runs/`. No committed source or PREREGISTRATION is edited.

## 8. Open registration questions

1. **The dense deposit's own decay class** (K11 §7.4, carried): the
   deposit rides the row it lands on and the row's DECAY applies to it
   thereafter; a separate decay class is a named variant.
2. **Match-similarity conditioning (P1(b))** for the dense gate (GDN2's
   "how well does this token match its row" analog): named, not run.
3. **The per-slot/address-axis variants** (K11B §7.1's named follow-up):
   untouched by this lane.
4. **Board-scale**: the operator's call, pending this lane's outcome and
   sign-off.
