# Pre-registration draft — THE SEPARATED-KEY ARM at 150 M scale

**Date:** 2026-11-17 · **Status: DRAFT, operator sign-off pending.** This is a NEW
registration (it changes the store's ADDRESSING, not the algebra's operators), written
before any probe measurement is claimed. Probe-scale engineering checks (parity gates,
memory, timing) are pre-registration evidence and are in scope now; **no board-scale
(2700 s) claim is run before the operator signs this document off.** It does not touch
`PREREGISTRATION_SEQUENTIAL_MATRIX.md` (whose own sign-off is separately pending), the
matrix registration, or any `k6_*`/`k7_*`/`k8_*`/`k9_*` source — those are imported,
never edited.

## 0. The chain this stage closes

The 150 M board arm's residual (`CMAES_TUNING.md`: matrix_ops 1.7576 vs gdn2 1.1976
nats/byte; ≥ 0.74 nats architectural at probe exposure) was diagnosed as the
identity-cache regime: CORRECT 60–71 % of decisions, firing mostly as exact-match
*identity rewriting* over repeated byte-level payload directions — a store whose keys
ARE its contents has compression ratio 1. SEP-KV (`RESULT_KEY_SEPARATION.md`) proved
the cause at probe scale (key=value fusion is mechanically role-blind; the fused arm
pinned at the derived 0.5 set-retrieval ceiling across all 48 depth-1 cells while the
separated arm reached 1.000) and receipted the fix (small separated write-context keys;
key_dim 4–16 all 1.000; the free-key collapse basin killed by mimetic birth,
`RESULT_WHEELER_CAPACITY.md` §4b: span_key_std 0.68–0.90 in every mimetic cell, the
collapse signature nowhere). **The fix has never touched real text at scale. This
registration defines that application.**

## 1. The architecture under test

**The K6 per-token sequential arm (`PREREGISTRATION_SEQUENTIAL_MATRIX.md`, the untangled
machine) with its ADDRESSING moved from content to a small learned key channel.** Same
five-op algebra, same thresholds, same values, same per-token granularity, same
defined-order arithmetic spec. The semantics, pre-stated:

1. **A small key table** `K ∈ R^{rank × key_dim}` per head, `key_dim = 16` (the
   SEP-KV key-dim curve's Receipted-sufficient size; one row per slot). The WRITER emits
   a key `k_t ∈ R^{16}` — a learned projection of its CAUSAL WRITE CONTEXT (the last 4
   positions of the mixer's input, per-position RMSNormed, the SEP-KV receipted minimal
   form) — and the table is updated by THE SAME SCHEDULE'S WRITE EVENTS, exactly as the
   SEP-KV oracle (`key_separation_probe.separated_oracle_scan`, the minimal
   separated-key oracle form) files it: APPEND/CORRECT/MERGE re-file the target row's
   key to `k_t`; MERGE additionally zeroes the source row's key (its value, key and
   liveness all close); POP clears the address (`K_j := 0`); NONE leaves it untouched.
   A fresh row's key is the zero vector until first written, and matching against it is
   liveness-masked exactly as the value channel is.
2. **Routing similarity AND read scores switch from `cos(q, S_j)` to `cos(q_key, K_j)`.**
   The decision chain (`schedule.route_chunk`'s logic at chunk granularity 1) runs
   verbatim with the similarity taken on the key channel: unit write key vs the live
   rows' unit keys, dead rows masked to −2, the index tie-break penalty, the first-index
   rule, the five thresholds UNCHANGED. The read keeps the certified form with the score
   on the key channel: `out_t = Σ_j softmax_j(scale·cos(qk_t, K_j) over LIVE slots) ·
   ‖S_j‖ · S_j` — values, payloads, DECAY, growth-rule arithmetic, and all five ops'
   arithmetic UNCHANGED (the MERGE blend still uses the registered detached VALUE
   strengths). The value-space content query is REPLACED by the key-space query (a
   learned projection of the same causal context): the mixer's `query` projection is
   REMOVED, not retained dead.
3. **Keys are NOT decayed** — an address is categorical (the SEP-KV lane's flagged open
   question, inherited here as this registration's first open question, §7.1).
4. **Mimetic birth, free drift.** The key and query-key projections are INITIALISED AT
   the fixed orthonormal family: only the CURRENT-token context position is non-zero,
   a fixed seeded QR-orthonormal map `Q_h ∈ R^{width×16}` (seed 3401 + head) applied to
   the RMSNormed position, identical for the write-key and query-key projections — so
   at step 0 the emitted key is a fixed function of the token, keys are BORN SPREAD
   (distinct byte directions project to distinct 16-d addresses; spread is an invariant
   of the embeddings, not a training outcome), and the birth machine MIMICS the
   registered fused content-addressing (the read score at birth ≈ the 16-d JL sketch
   of the content cosine; the X42 mimetic precedent:
   `x42_pile_scale/receipts/mimetic_geometry/VERDICT.md` — "mimicry improves materially
   while unfreezing does not"). Training then drifts the projections FREELY — no
   rotation constraint, no re-orthonormalisation, weight decay as everywhere (the
   Wheeler mimetic receipt: birth kills the collapse basin; drift is earned by
   training).
5. **Windows are fresh stores**: the key table resets to zeros with the value state at
   every (layer, window) scan boundary (the K6 semantic; the cross-window persistence
   question is §7.2's, not resolved here).

### 1.1 The arithmetic spec (inherited verbatim from K6, extended to the key axis)

K6's defined-order spec applies unchanged (balanced binary tree over each reduced
axis, exact-zero padding; fp32 decision/read arithmetic upcast once at entry;
correctly-rounded `sqrt.rn`/`div.rn`; the softmax's one transcendental as torch's own
libdevice `exp`; `mul.rn`/`add.rn` on every spec-path product — no FMA contraction).
The key channel adds folds over the key axis (`key_dim` 16, a power of two, LOGK = 4
levels): the write/query-key norms, the live rows' key norms, the read score
`scale·fold(uqk ⊙ K_j)/max(nk_j, EPS)` (the division after the fold, K6's own defined
order), and the routing similarity `fold(uk_t ⊙ K_j)/max(nk_j, EPS)` (dead rows −2).
Decisions, payload lanes, read weights, the value state, the key table and the reads
are therefore **bit-exact against the eager per-token oracle by construction** — the
gate-A bar is `torch.equal` on every channel the K6 bar covers plus the key table.

### 1.2 Parameter accounting (stated, not hidden)

The arm is built by the matched-arms build order (`matched_arms.build_matrix`, the
150 M capacity audit lands exactly) and then converted in place. The conversion
REMOVES the content-query projection (`width × heads·value_dim` = 147,456 per mixer)
and ADDS the write-key and query-key projections, both over the same 4-position
causal context (`4·width × heads·key_dim` = 2304 × 32 = 73,728 per mixer each,
147,456 per mixer together) — **the conversion is parameter-NEUTRAL to the parameter**:
the model stays at exactly the audited 149,981,808 parameters, and the key channel
replaces the content query it makes dead. The exact count is asserted in the probe's
receipt. The SEP-KV lane's caveat is inherited anyway: the probe licenses the
mechanism, not a superiority claim beyond this parameter-matched swap.

## 2. The implementation (K6's discipline: oracle first, kernels match it)

* **The eager per-token oracle** (`k10_separated_keys.separated_sequential_oracle_scan`):
  the K6 oracle's structure (decisions from the DETACHED current state and key table,
  read weights from the ATTACHED ones, DECAY then lanes in the registered order) with
  the key channel — the oracle DEFINES the reduction order and is the parity fixed
  point, differentiable through the key table (the read scores and the re-file lanes
  carry gradient to the write/query-key inputs; the decisions are detached constants).
* **The persistent-walk kernel extension** (`_k10_seq_fwd_kernel`): K6's
  whole-store-register walk with the key-match side — the key table (`rank × 16` fp32)
  is register-resident beside the value tile; per token the defined-order key norms,
  read scores, softmax, read (value side, the K6 floor channel), decision similarity,
  five-op selection, MERGE blend, then DECAY (values only) and BOTH lanes (value lane
  as K6; key lane per §1). It records the same decision channel shapes as K6 (lanes 2,
  `MatrixSchedule`-compatible, so K4's `persistent_scan_trajectory` replays the VALUE
  side verbatim), the pre-decision liveness, and the segment-entry checkpoints of the
  value state, liveness AND key table.
* **The backward** (`x45::k10_scan_fwd`/`x45::k10_scan_bwd`, ONE opaque op pair per
  (layer, window) scan, the K4/K6 hosting pattern): decisions are recorded constants.
  Per 64-token segment, in reverse: K4's `persistent_scan_trajectory` (imported
  verbatim) replays the value trajectory; a new `_k10_key_traj_kernel` replays the KEY
  table's trajectory from its segment-entry checkpoint (the key lane is a
  schedule-driven lane exactly like the value lane, `key_dim`-wide); a token-parallel
  `_k10_bwd_prep_kernel` recomputes the per-token read weights (bit-identically to the
  forward) and their adjoint coefficients on BOTH channels; a value-blocked
  `_k10_bwd_value_walk_kernel` (K6's walk with the score-path state term removed — the
  read scores no longer touch the value state, only the direct and strength terms
  remain) produces `dpayload`, `ddecay`, `dstate`; and a `_k10_bwd_key_walk_kernel`
  walks the key-table adjoint tile (`rank × 16`) in reverse — the read adjoint's
  c-term and key-norm term accumulate into the key table, the lane adjoint emits
  `dwrite_keys` at re-file events (the row's pre-write key gradient terminates at its
  last re-file, the POP/MERGE-source analog of K4's case-iii), and the query-key
  adjoint's clamp-select identity produces `dquery_keys`. **Keys are parameters/inputs
  — gradients flow through the key projections; decisions stay constants.**
* **The wiring** (`separated_scan`): the drop-in at the mixer's boundary. Refusals by
  name, the K6 pattern: `ops=False` and `grow` refused outright; a rank prefix refused;
  a store the persistent walk cannot hold takes the EAGER ORACLE verbatim (the same
  semantics, the parity fixed point, never a silent granularity change).
* **K8/K9 composition, API level** (full wiring deferred, stated): the key lane is a
  schedule-driven lane over the same recorded `MatrixSchedule` the K8 growth
  re-dispatch and the K9 distributed walk already consume, so the composition contract
  is unchanged — growth reveals zero-keyed rows that join the address space at their
  first write (liveness-masked before it), and the K9 grid's `_k9_score/_k9_decide/
  _k9_apply` split extends to the key tile by the same scratch-channel discipline. No
  K8/K9 kernel is edited or built here; the probe runs the K6-path integration fully
  (fixed rank 128, the registered shape).

## 3. Gates before any probe claim

1. **Gate A — oracle parity**: decisions/reads bit-exact (`torch.equal`) vs the eager
   oracle on every schedule field, payload lane, read weight, pre-decision liveness,
   exit state/liveness AND the exit key table, on ragged/edge fixtures covering all
   five ops, at the registered shape under the bf16 autocast regime; the decision-flip
   test at the K5A lane bar (0 flips). **The KEY-COLLAPSE watch** (the Wheeler
   receipt's guard) runs on every probe fixture: `span_key_std` — the std over a
   window's tokens of the emitted write keys, per key-dim, mean — must stay bounded
   (≥ 0.05 absolute; the Wheeler collapse signature is std → 0 with one run absorbing
   the window).
2. **Gate B — gradient parity vs the oracle within measured floors**: the fp64
   `gradcheck` of the oracle first (every adjoint branch: NONE, APPEND, CORRECT, POP,
   MERGE — including the key-lane branches), the fp32-vs-fp64 floor measured per
   output (`dpayload`, `dwrite_keys`, `dquery_keys`, `ddecay`, `dstate`), the kernel
   path within `NOISE_FACTOR` (8×) of it, each output's exactness status stated. The
   value-trajectory replay asserted bit-equal to the forward's own checkpoints.
3. **Gate C — THE PROBE**: the 20,022,866-target matched-exposure probe — seed 3401,
   the transferred LR (the gdn2 nomination's, the sweep's own within-sweep rule), the
   K6 protocol verbatim (the same `matrix_ops` build order, the same registered
   flagship stream, the same matched-exposure stopping rule and honest clock, the same
   document-disjoint calibration eval) — against the banked K6 per-token row
   (**2.0278 nats/byte**), with the mechanism predictions of §4 pre-stated.
4. **Gate D — timing**: block-call walls (forward/backward per (layer, window) scan),
   the projected step tok/s, and the honest key-channel cost statement (the key channel
   adds per-token match work: two `rank × 16` defined-order folds and a `rank × 16`
   adjoint walk against the value side's `rank × 128` folds — measured, not estimated).

## 4. PRE-REGISTERED MECHANISM PREDICTIONS (committed before the probe runs)

### 4.0 Baseline provenance (the conflation, corrected on record)

The task brief's "banked: 97.7 % identity-rewrite share at probe scale" and
"~48/128 occupancy" were CONFLATIONS, corrected here before anything is pre-registered
against them: the 97.66 % figure is the K8 GROWTH SMOKE's **CORRECT op-share of
DECISIONS at INIT** on real stream data (`K8_GROWTH.md` §5: "97.66 % CORRECT / 2.34 %
APPEND / 0 POP / 0 NONE, holding ~48 live slots"), never an identity-rewrite SHARE;
the identity-rewriting claim was QUALITATIVE from the 150 M CMA-ES diagnosis
(`CMAES_TUNING.md`: CORRECT 60–71 % of decisions "firing mostly as exact-match identity
rewriting"); **no identity-rewrite-share channel was ever banked at any scale**, and the
receipted K6 per-token probe occupancy is 73.7/128 mean live slots. This registration
therefore defines the instrument and MEASURES its own fused-path anchors:

* **The instrument** (first precisely-defined measurement of the diagnosed regime, at
  any scale): *identity-rewrite = a CORRECT decision whose write payload's fp32
  value-channel cosine to the targeted row's pre-write content is ≥ 0.999; the share is
  taken over CORRECT decisions.* Computed from the recorded schedule and the
  `persistent_scan_trajectory` replay (K4's kernel, imported) of the same walk.
* **Anchor (a) — the fused path at INIT** on the probe windows (the embedding-dominated
  regime the CMA-ES diagnosis receipts; expected near the mechanical ceiling).
* **Anchor (b) — the paired ablation**: the fused content-addressing walk (K6's own
  launch) run over the SAME trained key-arm parameters' value/payload/decay channels —
  isolating the addressing channel at fixed payloads (it turns "did the whole run
  differ" into "did the addressing change cause the mechanism change").

### 4.1 The predictions

* **K-1 (the NLL).** The key arm's eval NLL **improves on the banked 2.0278**
  nats/byte. Direction: improves. Magnitude band, honestly wide: **−0.02 to −0.35**
  (lands in [1.68, 2.01]); the honest uncertainty is that SEP-KV's necessity claim was
  PER-LAYER (depth substitution is available from layer 2 at 150 M), so the gain may be
  a fraction of the 0.51-nat probe-scale gap the fused per-token arm still leaves to
  gdn2's 1.5163 anchor.
* **K-2 (the identity-rewrite share falls).** The key arm's mean-over-layers share
  (§4.0's instrument) falls **materially below BOTH fused-path anchors** — materially
  = at least 0.10 absolute below the lower of them. If the share is within 0.02 of
  either anchor, the mechanism did not move.
* **K-3 (occupancy).** Mean live slots falls from the receipted fused per-token
  73.7/128 toward the store's useful floor; band **≤ 65**, honestly wide in both
  directions (a context-addressed store could plausibly RAISE occupancy — more
  context-distinct addresses means more rows claimed — and the refutation clause covers
  exactly that).
* **K-4 (read-weight entropy rises).** From the banked per-token 1.085 nats; band
  **> 1.2**, honestly wide (the sharpening alternative — a context-addressed read
  locking onto the right row — would push entropy DOWN; the direction is pre-stated
  from the diagnosis, and either landing is reported as measured).
* **K-5 (MERGE): NO PREDICTION.** Its recovery prediction was already falsified once
  (K6 §4: MERGE completes its collapse to 0.0002/token, the dial's staleness
  artifact); its rate is reported, not re-predicted.
* **The KEY-COLLAPSE watch (a guard, not a prediction).** `span_key_std` (§3 gate A's
  definition) bounded ≥ 0.05 in every layer of every probe run — the Wheeler collapse
  signature (std → 0, one run absorbing the window) appears nowhere; mimetic birth
  receipts its purpose.
* **Refutation clause (verbatim from the brief).** If the key arm shows the
  identity-rewrite regime UNCHANGED (share within noise — 0.02 — of the fused anchors)
  **or** NLL worse than 2.0278, **the fix does not transfer to real text and the
  residual needs a different owner**; it is reported as refuted, not excused. If K-1's
  NLL improves while K-2 does not move, that dissociation is reported as found — the
  diagnosis would be partially wrong about the residual's owner.

## 5. Claim set (registered, pending sign-off)

* **S1 (transfer).** The separated-key arm trains at the registered shape (149 M
  parameters, 24 layers, rank 128, `key_dim` 16) within the probe budget, and its
  held-out NLL at the 20,022,866-target matched exposure beats the banked K6 fused
  per-token row (2.0278) — the K-1 band of §4.1.
* **S2 (mechanism).** The pre-registered mechanism predictions K-2/K-3/K-4 evaluated
  against the measured anchors of §4.0, each reported as met/not-met/refuted with the
  receipts.
* **S3 (trainability).** The arm trains within the session budget; the projected step
  tok/s and the honest per-scan walls are reported first-class (gate D), including the
  key channel's per-token match cost.

Single seed (3401, the standing one), stated in every receipt; ~20 M probe exposure
(the sweep's own matched-exposure rule, its honest caveat inherited verbatim:
probe-scale numbers, one transferred LR, not board-scale claims). The board-scale
(2700 s) run is gated on the operator's sign-off of BOTH this document and the K6
registration it builds on.

## 6. Honest scope

* This is a registration of ADDRESSING (what the slot is addressed by), not of the
  algebra: the five ops, thresholds, values, granularity and arithmetic spec are the
  K6 draft's verbatim, and this arm inherits that draft's pending status.
* The necessity evidence is per-layer and probe-scale (SEP-KV); at 24 layers depth
  substitution is real (receipted, §5 of the SEP-KV result), so a null K-2 at 150 M
  would not refute the necessity claim — only its ownership of THIS residual (the
  refutation clause's exact scope).
* `key_dim` 16, `ctx` 4, keys undecayed, mimetic birth with free drift, windows as fresh
  stores: all five are scoping choices receipted at probe scale; §7 lists each as an
  open registration question.
* The probe licenses the mechanism at exactly the audited parameter count (§1.2's
  swap); it is not a board claim.
* K8 (growth) and K9 (high-rank) integration is API-level here (§2); the probe runs
  fixed rank 128 on the K6 path fully.

## 7. Open registration questions (inherited and new)

1. **Keys and decay** (the SEP-KV lane's most visible choice, inherited FIRST):
   addresses are not decayed here — an address is categorical. Whether addresses
   should age with their values (address persistence vs address aging) interacts with
   run-boundary semantics and growth and is unresolved.
2. **Run-boundary semantics for the key channel**: this registration resets the key
   table with the store each window. Whether keys persist across run boundaries (and
   what a carried key means after its value has decayed away) is a registration
   decision, not receipted here.
3. **Orthonormality vs statistical margin**: with learned content-free keys the
   per-slot ops remain exact (leak-freedom is by slot index), but the matching margin
   is a statistical property — distinct contexts must keep key-cos below `tau_match`;
   the SEP-KV key-dim curve (4–16 dims all 1.000 at that store size) is the only
   margin evidence, and it is probe-scale.
4. **Ordered / Wheeler-style keys** (addresses of addresses) remain the named follow-up
   variant once the flat small-key case is receipted on real text.
5. **Growth interaction**: growth reveals zero-keyed rows (liveness-masked until first
   written); whether growth should instead reveal PRE-ASSIGNED key directions (a
   key-family prefix) is open — and is why the K8 integration stays API-level here.
6. **Key-dim scaling on real text**: the registration picks 16; the real-text curve is
   unmeasured.

## 8. Files

`k10_separated_keys.py` (the oracle, the kernels, the op pair, the wrapper, the mixer
conversion), `test_k10_separated_keys.py` (gates A/B), `k10_gate_receipt.py`
(the registered-shape receipts), `k10_probe_separated_keys.py` +
`run_k10_probe_leased.sh` (the gate-C probe and its detached leased driver),
`K10_SEPARATED_KEYS.md` (the results doc). No `k6_*`/`k7_*`/`k8_*`/`k9_*`,
`PREREGISTRATION_*`, `lattice.py` or `matched_arms.py` file is edited.
