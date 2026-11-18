Pre-registration draft — LEARNED GATES ON THE EMENDER (K11): input- and
state-dependent learned write/read gates inside the certified algebra

**Date:** 2026-11-21 · **Status: DRAFT, operator sign-off pending.** This is a
NEW registration (it changes the store's CONTINUOUS write/read parameters, not
the algebra's discrete op selection), written before any probe measurement is
claimed. Probe-scale engineering checks (parity gates, gradchecks, memory,
timing) are pre-registration evidence and are in scope now; **no board-scale
(2700 s) claim is run before the operator signs this document off.** It does not
touch `PREREGISTRATION_SEPARATED_KEYS.md` or
`PREREGISTRATION_SEQUENTIAL_MATRIX.md` (whose own sign-offs are separately
pending), any `k10_*`/`k6_*`/`k1_*` source, the matrix registration, or
`lattice.py` — those are imported, never edited.

## 0. The chain this stage closes, and the enabling fact

The side-by-side dissection (`RESULT_SIDE_BY_SIDE_0239.md` §5) attributed the
winner's steady deficit to C5 — the store's write/read FORM: byte-identity
CORRECT on 0.979/token, reads over ≈ e^1.73 ≈ 5.6 effective rows of ~27 live,
17 of 24 stores near-decorative at eval. The optimizer bake-off
(`RESULT_OPTIMIZER_BAKEOFF.md`) re-scoped C5 (the write form UNDERTRAINS under
AdamW) and fixed the training surface (hybrid Muon at 3× the AdamW LR); the
symmetric control (`RESULT_MUON_CONTROL.md`) retracted the lead claim and
restored GDN2 ahead at the matched surface — **the residual that remains at the
matched surface is structural, and the side-by-side's §6 named lever (a
gdn2-style dense gated delta write) is what this registration defines.**

The receipts converging on the lever:

* **GDN2 spends its parameters on an admission controller.** Its gate matrices
  (erase `b`, write `w`, decay pre-activation `f`, output `g`) are ~18 % of its
  parameters and they are INPUT-DEPENDENT: its state takes a dense, channel-
  gated rank-1 delta at EVERY token from step 1 (write gate ≈ 0.5 on every
  channel at init and trained).
* **Our machine gates by THRESHOLD.** 27,648 gate parameters (the 24 DECAY
  projections, 0.018 %) — the discrete op SELECTION is thresholded, and the
  continuous op parameters (how strongly a matched write is admitted, whether
  a read is passed on) are fixed constants of the arithmetic, not learned.
* **The enabling fact — gates are SCHEDULE DATA.** K1's injection list already
  carries a gate scalar per write (the decay weight `exp(gc_ref − gc_u)` of
  each injection, `K1_SCHEDULE_SCAN.md` §1.3): per-write scalars are an
  existing, certified channel of this algebra. The Lean algebra's writes are
  gated by construction — `EmenderMemory` APPEND writes `(key, gate·value)`
  and MERGE is `old + beta·(new − old)`, with "all gate magnitudes" part of
  the fixed `TokenDecision` schedule (`formal/lean/OPERATOR_ALGEBRA_STATUS.md`
  §"Proved", §"Phase one assumed": "Complete fixed schedule: targets, binary
  enables, **all gate magnitudes** … must be supplied"). The affine-in-values
  scan theorems hold for ANY supplied gate values; the obstruction list pins
  exactly what must NOT happen (state-dependent DEATH recomputation,
  threshold-shaping, state-derived read alphas) and explicitly receipts that
  "fixed learned alpha remains linear" when the alpha is schedule data
  (`AttentionObstructions.fixed_learned_alpha_linear`). **Making the gate
  magnitudes input-dependent adds ZERO new math**: the schedule is pinned per
  token either way; only the PRODUCER of the scalar changes.
* **The discrete op SELECTION stays thresholded and recorded** — the doctrine
  the expressivity results and the fusion program are built on. Only the
  continuous op PARAMETERS become learned.

**Semantics rulings received from the supervisor before this document was
committed** (the task's escalate-before-commit rule), cited at each choice
below:

* **P1 — state-dependence only through recorded channels.** A gate projection
  may consume (a) the current token's features and (b) decision-side channels
  already recorded per token (match similarity, slot strengths, live counts).
  Nothing else.
* **P2 — decisions are inviolable.** No gate value may feed op selection:
  gates scale payload magnitudes, write strengths, and read outputs ONLY —
  never thresholds, never routing similarity, never liveness. The
  bit-exact-decision gate (0 flips) is the receipt that this held.
* **P3 — the dense-write target rule.** G2's per-token contribution goes to
  the BEST-MATCH LIVE ROW (argmax over live, regardless of whether it clears
  `tau_match`). If NO live row exists, the dense contribution goes nowhere
  (zero) — writing into a free slot is an APPEND event: it would make a
  decorative row live and change the decision sequence, violating P2. Dense
  writes touch live rows only; structure creation stays the ops' job.
* **P4 — granularity: simplest form first.** G1's write-strength gate is a
  per-token per-head scalar projection; richer per-slot forms are a later
  variant. Read-out gating is included in G1 only if it costs less than ~20
  lines of oracle (it costs four mixer lines and zero oracle lines — it is
  included; see §1.2).
* **P5 — mimetic-neutral birth.** Every new gate channel is born so that the
  arm's step-0 behavior IS the banked winner's (the X42/K10 mimetic precedent:
  the new machinery must not perturb the banked machine at birth; the channel
  grows by training). The birth form is pre-stated in §1.3.

## 1. The architecture under test

**The K10 winner (per-token + separated keys, the tuned config) with its
continuous write/read parameters made learned gates, at the matched training
surface (hybrid Muon at 3× the tuned LR, the bake-off's banked best).** Same
five-op algebra, same thresholds, same key channel, same per-token
granularity, same defined-order arithmetic spec, same discrete op selection.
Three arms, staged minimal → radical:

### 1.1 Arm G1 — the minimal learned-gate arm (write-strength + read-out gates)

1. **The write-strength gate** `g_write ∈ (0,1]`, per (token, head): a
   per-token per-head scalar produced by a small learned projection of the
   mixer's input `x` (P1(a): input-dependent; the P1(b) recorded-channel
   variants — match-similarity conditioning — are named in §7 and NOT run in
   this registration). It is consumed through the existing write/injection
   channel, at the two content-carrying ops:
   * **APPEND** (the Lean APPEND form, `gate·value`, verbatim):
     `S[j_free] := S[j_free] + g_write · p_t` — the admitted payload. (A free
     row's content is exact zero — POP/MERGE write exact zeros and the dense
     channel never touches dead rows, P3 — so this is `g_write · p_t` on a
     fresh row.)
   * **CORRECT** (the Lean MERGE arithmetic applied at the CORRECT write
     enable, `old + beta·(new−old)` with the gate as schedule data): the
     overwrite becomes the delta-rule blend
     `S[j_best] := v + g_write · (p_t − v)`, where `v` is the row's
     post-DECAY pre-write content. This is the GDN2 move on the dominant op
     (97.4 % of the winner's decisions): a matched row becomes a running
     average of its recent payloads instead of a byte-identity replay.
   * **MERGE/POP/NONE carry no content and stay VERBATIM** (MERGE's blend is
     the router's own row-mix, computed from the pre-DECAY rows exactly as
     registered; POP writes the constant zero; NONE writes nothing).
2. **The read-out gate** `g_out`, per (token, head): the g_proj analog —
   `reads' = reads ⊙ g_out` applied in the mixer before the output projection
   (outside the scan op; the read stays affine in values with the alpha as
   schedule data, the Lean `fixed_learned_alpha_linear` receipt). Born at
   identity (§1.3).
3. **The gate forms** (P4's simplest-first, with P5's zero-born scalar):
   `g_write = 1 − m_w · σ(w_w·x + b_w)` and `g_out = 1 + m_o · σ(w_o·x + b_o)`,
   each per head, where `σ` is the logistic and `m_w, m_o` are per-head
   learnable scalars **born at exact zero** — so at birth `g_write ≡ 1` and
   `g_out ≡ 1` exactly: the arm is born as the banked winner (P5), and
   training opens the gates (the gradient reaches `m` at step 0; the
   projections train once `m` moves — no dead zone beyond the first step).

### 1.2 Arm G2 — the dense-write arm (the direct GDN2 move)

**Every token writes a small learned-gated contribution to its matched row,
as a third scheduled lane, while the five ops keep managing structure.**
The deposit is `dense_t = m_d · σ(w_d·x + b_d) · p_t` per (token, head)
(`m_d` a per-head scalar **born at exact zero**, P5), and its target rule is
P3 verbatim: **the dense contribution goes to the BEST-MATCH LIVE ROW (argmax
over live, regardless of whether it clears `tau_match`); if NO live row
exists, the dense contribution goes nowhere (zero).** The deposit:

* is recorded as **schedule lane 2** — `op=APPEND` (additive), `slot=best`,
  `erase=False`, `live_after=True`, `payload=dense_t` — applied after lanes
  0/1, so it is ordinary schedule data: K4's `persistent_scan_trajectory`
  (imported verbatim, lane-generic) replays it bit-exactly and the certified
  scan theorems cover it as any other scheduled additive write;
* **touches live rows only** (P3): it never creates liveness, never re-files
  a key, never enters the decision chain (P2) — the discrete op sequence is
  bit-invariant to the deposit at fixed inputs, receipted by the 0-flip gate;
* **composes with the ops, stated honestly**: at a CORRECT token the op's
  overwrite lands first and the deposit adds on top (`S[best] := p` then
  `S[best] += dense_t`); at an APPEND token the op allocates the free row
  while the deposit lands on the best EXISTING row (both GDN2-like: the token
  both extends the store and blends into its nearest neighbor); at POP the
  op kills the worst row and the deposit lands on the best; at NONE (≈ 0 % of
  the winner's tokens) the deposit is the only write. **The five ops
  (APPEND/POP/MERGE unchanged), the thresholds, the key lane and the
  liveness arithmetic are the K10 registration's verbatim.**
* **A zero deposit is a no-op** (defined, tested): when the token's `dense_t`
  is exactly zero (at birth, always; after training, only if the gated payload
  vanishes), lane 2 is recorded `op=NONE` and nothing is applied — this is
  what makes the birth bit-parity exact (§3 gate A).

G2 carries NO write-strength gate and NO read-out gate (its lane-0 arithmetic
is the incumbent's verbatim): it isolates the dense channel.

### 1.3 Arm G3 — combined

G1 + G2 together (write-strength gate + read-out gate + the dense channel).
Run "if both help" per the task's staging; the driver supports it from the
start and the arm launches if the single-mechanism arms land clean within the
card budget.

### 1.4 The arithmetic spec (inherited verbatim, extended by two scalars)

K6/K10's defined-order spec applies unchanged (balanced binary tree folds,
fp32 decision/read arithmetic, correctly-rounded `sqrt.rn`/`div.rn`, libdevice
`exp`, `mul.rn`/`add.rn` — no FMA contraction). The gates add: one
`sub.rn`/`mul.rn`/`add.rn` triple per gated CORRECT (the blend) and one
`mul.rn` per gated APPEND (the admission), on the value lane; the deposit adds
one `add.rn` at the best row. The recorded lane-0 payload is the BLENDED lane
value, so the recorded schedule replays through K4's kernel verbatim — the
gate never appears in the replay at all. Decisions, payload lanes, read
weights, the value state, the key table and the reads are therefore
**bit-exact against the eager per-token oracle by construction** — the gate-A
bar is `torch.equal` on every channel the K10 bar covers, plus the 3-lane
schedule.

### 1.5 Parameter accounting (stated, not hidden)

Each gate channel is one `nn.Linear(width, heads, bias=True)` plus one per-head
scalar: 576·2 + 2 = **1,154 parameters per mixer per channel**. The arms ADD
parameters to the audited winner (no MLP re-solve: the backbone stays the
banked winner's byte-for-byte, so each arm differs from the winner ONLY by its
gate projections):

| arm | added channels | added params | total | vs 149,981,808 |
|---|---|---:|---:|---:|
| G1 | write gate + read-out gate | 55,392 | 150,037,200 | +0.037 % |
| G2 | dense gate | 27,696 | 150,009,504 | +0.018 % |
| G3 | all three | 83,088 | 150,064,896 | +0.055 % |

All three sit inside the capacity tolerance (0.1 %); the additions are 0.3–0.5
of GDN2's own gate-matrix share expressed as a fraction of parameters
(18 % → ours 0.02–0.06 %) — the honest asymmetry is that our gates are per
HEAD scalars (2 per token), not per-channel matrices; the per-slot/per-channel
richer forms are §7's named variants. The exact count is asserted in the
probe's receipt.

## 2. The implementation (K10's discipline: oracle first, kernels match it)

* **The eager per-token oracle** (`k11_learned_gates.gated_oracle_scan`):
  K10's oracle structure (decisions from the DETACHED key channel, read
  weights from the ATTACHED state/keys, DECAY then lanes in the registered
  order) extended with the gate channel and the dense lane — the oracle
  DEFINES the reduction order, is the parity fixed point, and is
  differentiable through payload, dense, gates, keys and decay.
* **The persistent-walk kernel** (`_k11_seq_fwd_kernel`): K10's
  whole-store-register walk with (a) the gated lane-0 arithmetic (a `GATED`
  constexpr: 0 = the incumbent's lane verbatim for G2, 1 = the blend/
  admission), (b) the dense lane (a `DENSE` constexpr: lane 2 loaded, targeted
  at `best` per P3, applied additively, recorded; the zero-deposit no-op), and
  (c) the 3-lane recorded schedule. K10's `_k10_seq_fwd_kernel` is NOT edited
  — the k11 kernel is new code beside it, importing K1's `_sub_rn` and K6's
  fold helpers.
* **The backward** (`x45::k11_scan_fwd`/`x45::k11_scan_bwd`, ONE opaque op
  pair per (layer, window) scan, the K4/K6/K10 hosting pattern): decisions
  are recorded constants. Per 64-token segment, in reverse: K4's
  `persistent_scan_trajectory` **imported verbatim** replays the 3-lane value
  trajectory (the blended lane-0 payload and the dense lane are ordinary
  schedule data to it); a k11 key-lane replay (K10's kernel generalized to the
  3-lane channel strides — keys are untouched by gates); K10's
  `_k10_bwd_prep_kernel` **imported verbatim** (it recomputes the read weights
  from the trajectory and the key channel — gate-agnostic); a k11 value walk
  (K10's walk plus the gate adjoint: `dpayload = g·dpay0`,
  `dgates = <dpay0, p − v>` per token — a per-block partial reduction like
  `dlam`'s — the `(1−g)` old-row path at CORRECT, the dense lane's
  `ddense = dS[best]`, and the L-generic channel strides); and a k11 key walk
  (K10's, stride-generalized). **Kernel changes were needed** (the blend
  arithmetic and the dense lane live inside the walk): receipted here per the
  task's gate-A clause, with the birth-parity receipt (§3) proving they reduce
  to the incumbent's arithmetic at unity/zero gates.
* **The wiring** (`gated_scan`): the K10 wrapper's drop-in with the same
  refusals by name (`ops=False`, `grow`, a rank prefix), the eager oracle as
  the fallback, and the gate projections living in the mixer
  (`GatedSeparatedMixer`), added by `convert_model_to_gated_writes` on top of
  the banked `convert_model_to_separated_keys` (the conversion shares the
  K10 mixer's `nn.Linear` modules — the optimizer sees the winner's own
  parameters plus the gate projections).
* **The optimizer routing** (the incumbent's own convention): the bake-off's
  `split_parameters` routes `mixer.{value,key,query_key,out}` + embeddings +
  MLP to Muon; the gate projections (`write_gate`, `readout_gate`,
  `dense_gate` and their scalars) do not match the Muon patterns and land on
  the hybrid's AdamW side, exactly as the incumbent's own DECAY gate does
  ("a gate, not a hidden map", `bakeoff_optimizers.py`) — asserted in a test
  and in the probe's receipt.

## 3. Gates before any probe claim

1. **Gate A — oracle parity and the 0-flip bar.** The kernel path vs the eager
   oracle: `torch.equal` on every schedule field (3 lanes), the payload lanes
   (the blend included), the read weights, pre-decision liveness, exit state/
   liveness/key table, on ragged/edge fixtures covering all five ops at gate
   values off identity (e.g. g=0.7, random dense), at the registered shape
   under the bf16 autocast regime. **The decision-invariance receipt (P2's
   0-flip bar): the discrete op sequence (lanes 0/1) is bit-identical under
   ANY gate values and ANY dense deposits at fixed inputs** — tested by
   re-running the same fixture at identity gates and at random gates and
   asserting the decision channels equal. **The birth-parity / plumbing
   receipt (P5): at `g_write ≡ 1`, `g_out ≡ 1`, `dense ≡ 0` the k11 scan's
   discrete schedule, key table and liveness are bit-identical to K10's own
   launch on the same inputs; the value state and reads sit within the
   measured fp32 floor** (the CORRECT blend's `v + 1·(p−v)` re-rounds the
   overwrite — ≤ 1 ulp per write, Sterbenz-exact in the identity regime —
   stated, measured, not hidden). **The replay receipt:** the 3-lane
   recorded schedule rebuilds the forward's own segment checkpoints
   bit-exactly through K4's imported kernel, dense lane included.
2. **Gate B — gradient parity within measured floors.** The fp64 `gradcheck`
   of the oracle FIRST (every adjoint branch: NONE, APPEND admission, CORRECT
   blend, POP, MERGE, the dense lane, the read-out gate), then the
   fp32-vs-fp64 floor per output (`dpayload`, `ddense`, `dgates`,
   `dwrite_keys`, `dquery_keys`, `ddecay`, `dstate`, `dkeys`), the kernel
   path within `NOISE_FACTOR` (8×) of it, each output's exactness status
   stated. K10's structural boundaries carry over (POP terminates the row
   gradient; a re-filed row's earlier key gradient terminates; the MERGE blend
   reaches both source rows) and are re-pinned on the gated oracle.
3. **Gate C — THE PROBE.** 20,022,866 targets (the 20,000,000-target
   matched-exposure rule), seed 3401, the ladder protocol verbatim (the
   registered flagship stream, the document-disjoint calibration eval at 7
   windows / 14,336 targets, the honest clock, the protocol's grad clip), the
   arms G1/G2/G3 with checkpoint curves (steps 1/5/10/20/40/80/120/160/180,
   CPU snapshots inside the loop, every eval on restored snapshots) overlaid
   on the banked curves (the winner's muon3x rows, the incumbent, gdn2).
   **Second seed (3402) on the best arm.**
4. **Gate D — walls.** Walls/tok/s stated per arm (the gates add a small
   projection, a per-token gate load, and — for G2/G3 — the dense injection
   traffic: measured against the winner's banked muon3x row, not estimated).

## 4. PRE-REGISTERED MECHANISM PREDICTIONS (committed before the probe runs)

Stated from C5's receipts BEFORE the probe. The anchors are the banked
matched-surface winner rows (the bake-off's muon3x arm, `RESULT_OPTIMIZER_
BAKEOFF.md` §3): final NLL **1.5146** (seed 3401), identity-rewrite share
**0.123**, read entropy **1.664 nats** (≈ e^1.664 ≈ 5.3 effective rows of
35.6 live slots), CORRECT/token **0.9735**.

* **M-1 (NLL).** Each gate arm's final eval NLL **improves on the winner's
  matched-surface row at the same seed (1.5146)**. Band, honestly wide:
  **−0.02 to −0.25 nats**; refuted if worse than the banked row.
* **M-2 (the identity-rewrite share falls).** The mean-over-layers identity
  share (the K10 registration's instrument: a CORRECT whose payload cosine
  to the pre-write row is ≥ 0.999) **falls at least 0.03 absolute below the
  0.123 anchor**; unchanged if within 0.02. This is the C5 headline: under
  the blend, matched rows become superpositions of their recent payloads, so
  the pre-write row stops being the last payload's direction.
* **M-3 (effective read rows rise).** Read entropy from 1.664 nats: **> 1.8
  nats** (≈ 6 effective rows); unchanged if within 0.05. The gated write
  redistributes row strengths and makes rows content-distinct, so the read
  spreads.
* **M-4 (near-decorative stores fall).** The read-mask sweep's per-layer
  deltas: **≥ 10 of 24 layers with ΔNLL ≥ 0.02** (the banked winner: 7 layers
  ≥ 0.02, 17 near-decorative at |Δ| ≤ 0.0125 — the side-by-side's §2 sweep,
  its AdamW-era anchor stated as the caveat: the muon3x-era sweep was never
  run, so this prediction is anchored one surface back).
* **M-5 (op mix): NO PREDICTION.** The op mix is a decision-channel property
  the gates cannot move at fixed weights (P2); across training it drifts with
  the projections. Reported, not predicted.
* **The gate telemetry, reported first-class:** per-layer means of the
  write-gate distance from 1, the dense-gate magnitude, the read-out gate —
  the learned admission controller's own readout, the GDN2-analog of the
  side-by-side's §4 table.
* **Refutation clause (the task's own).** If the mechanism numbers do not
  move while NLL does (or vice versa), the dissociation is reported honestly
  — it re-scopes C5 again. If NO mechanism number moves and NLL does not
  improve, the learned-gate lever does not own the residual and that is
  reported as refuted, not excused.

## 5. Claim set (registered, pending sign-off)

* **S1 (quality).** The gate arms train at the registered shape within the
  probe budget, and the best arm's held-out NLL at the 20,022,866-target
  matched exposure beats the winner's matched-surface row (1.5146, seed 3401)
  — the M-1 band — with the second seed (3402) read against 1.5027.
* **S2 (mechanism).** M-2/M-3/M-4 evaluated against the banked anchors, each
  reported met/not-met/refuted with the receipts, the dissociation clause
  applied verbatim.
* **S3 (algebra).** The 0-flip decision-invariance receipt and the
  birth-parity receipt hold (the gates are schedule data end to end; the
  recorded-schedule machinery is untouched), and the gate plumbing is
  receipted (K4's replay consumes the 3-lane schedule verbatim, dense lane
  included).
* **S4 (cost).** The honest walls/tok/s per arm against the winner's banked
  row (gate D).

Single seed (3401) per arm plus the second seed on the best arm; probe-scale
exposure, stated in every receipt. The board-scale (2700 s) run is gated on
the operator's sign-off of this document.

## 6. Honest scope

* This is a registration of the CONTINUOUS write/read parameters (the gate
  magnitudes), not of the algebra: the five ops, thresholds, key channel,
  granularity and arithmetic spec are the K10 draft's verbatim, and this arm
  inherits that draft's pending status.
* The gate arms ADD parameters (§1.5): the comparison is parameter-matched to
  within 0.06 %, not to the parameter; the additions are stated per arm and
  the asymmetry to GDN2's 18 % gate-matrix share is stated.
* The gates are per-token per-head scalars (P4's simplest form). Per-slot
  gates, per-channel gates, and the match-similarity state-dependent variant
  (P1(b)) are named variants, not this registration.
* Read-out gating is included in G1 per P4 (it costs four mixer lines outside
  the scan; the oracle is untouched). Its gradient path is plain autograd
  outside the opaque op pair.
* Everything is single-protocol probe scale (the 20 M matched exposure, the
  7-window eval, the ~0.075-nat endpoint-variance band the side-by-side
  receipts applies to every endpoint here); the Muon LR axis and the
  gates-on-Muon routing question (the muon control's §6 residual) are NOT
  searched here — the arms run the banked best surface verbatim.
* The read-mask anchor of M-4 is the AdamW-era winner's sweep (the muon3x-era
  sweep was never banked); stated as the caveat on that prediction.

## 7. Open registration questions

1. **Match-similarity state-dependence (P1(b))**: the write gate could consume
   the recorded best-match similarity (a decision-side channel) instead of
   only the input — the GDN2-analog "how well does this token match its row"
   conditioning. Named, not run.
2. **Per-slot / per-channel gates**: GDN2's gates are channel matrices (18 %
   of parameters); ours are per-head scalars (0.02–0.06 %). Whether the
   scalar form captures the mechanism or only its sign is the natural next
   arm if M-1 lands.
3. **Gates on MERGE/POP payloads**: this registration gates only the
   content-carrying ops (APPEND/CORRECT); MERGE's blend and POP's zero are
   kept verbatim (structure). Whether MERGE's blend strength should also be
   learned is open.
4. **The dense lane and DECAY interaction**: the deposit is not decayed
   separately (it rides the row it lands on; the row's DECAY applies to it
   thereafter). Whether a deposit should carry its own decay class is open.
5. **The 10× Muon point and the gates-on-AdamW routing variant** (the muon
   control's §6 residual, inherited): untested here, outside the approved
   bracket.

## 8. Files

`k11_learned_gates.py` (the oracle, the kernels, the op pair, the wrapper, the
gated mixer, the conversion), `test_k11_learned_gates.py` (gates A/B),
`k11_gate_receipt.py` (the registered-shape receipts),
`k11_probe_learned_gates.py` + `run_k11_learned_gates_leased.sh` (the gate-C
probe and its detached leased driver, cards 0/1), `receipts/k11_gates.json`,
`receipts/k11_learned_gates_summary.json`, `RESULT_K11_LEARNED_GATES.md`.
No `k10_*`/`k6_*`/`k1_*`/`k4_*` source, `PREREGISTRATION_*` file,
`lattice.py`, `matched_arms.py`, `velocity_ladder.py`, `cmaes_search.py`,
`optimizer_bakeoff.py` or `side_by_side.py` is edited — all imported.
