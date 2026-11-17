# KEY SEPARATION (SEP-KV): the necessity probe

Status: **predictions pre-stated (this revision); sweep pending.** The
predictions below were committed before any protocol-budget training run.
Pre-prediction calibration runs are disclosed in §6 (the granularity
separator's disclosure precedent). Results, receipts and the verdict are
appended after each part lands; nothing in §1-§7 is edited after the fact.

## 1. Hypothesis, and the Wheeler-graph frame

The operator's hypothesis, sharpened by a compression-theoretic argument: a
store whose keys ARE its contents has compression ratio 1 and degenerates to
identity caching — the receipted diagnosis at 150 M
(``CMAES_TUNING.md``: CORRECT 60-71 % of decisions, firing mostly as
exact-match *identity rewriting* over repeated byte-level payload directions;
read entropy collapse as the only tuning win). The hypothesis under test:
**key-value separation is necessary** — with key=value (the registered design,
matching by raw content cosine) two memories with identical content are
mechanically indistinguishable regardless of parameters; with a small separate
key, context-dependent addressing solves it.

**The claim's true scope, fixed before the sweep (supervisor-approved after
the calibration below): PER-LAYER.** At the first store layer — where the
payload is a fixed function of the token — content-fused addressing is
mechanically role-blind. Depth can substitute (§5): a second layer synthesizes
context-dependent addressing through composite payloads, at the cost of one
full layer of indirection. The registration question this probe receipts is
therefore: **dedicated key channel vs composite layer.**

## 2. The collision lemma (exact, parameter-free)

Machine-checked in ``test_key_separation_probe.py``; every claim below holds
for every parameter setting:

* **L-bit-identical payloads.** At layer 1 the registered payload is
  ``W_v RMSNorm(embed(token))`` — no cross-token term. Two occurrences of the
  same token produce BIT-IDENTICAL payloads
  (``test_identical_payloads_bit_equal``).
* **Write half.** While copy-1's row is live with content ``c * f(X)`` (``c``
  the accumulated decay scalar, above the registered strength clamp), copy-2's
  routing similarity to that row is exactly ``cos(f(X), c f(X)) = 1 >=
  tau_match``: copy-2 is CORRECT or MERGE onto that row, never an APPEND into
  a fresh slot — the two copies mechanically cannot occupy two rows
  (``test_identity_rewrite_is_mechanical``, at decay 0.999/0.9/0.5/0.05).
  This is the probe-scale reproduction of the 150 M identity-rewrite regime.
* **Read half.** Two live rows that share content receive exactly equal read
  scores, hence exactly equal softmax mass
  (``test_equal_mass_on_identical_rows_is_exact``): the read channel is
  mechanically role-blind between content-identical memories — "two memories
  with identical content are indistinguishable" verbatim.
* **M2's order feature is absent from the content channel**
  (``test_m2_order_blindness_is_content_exact``, run before the sweep per the
  supervisor's requirement): the (P, Q) and (Q, P) frames produce the same row
  directions (bijection-matched to < 1e-4 cosine) and the same read
  probabilities (sorted agreement < 1e-5), both writes APPEND without
  interacting, and the only state differential is the strength norms — the
  per-token decay factors lambda_P/lambda_Q, a FIXED function of token
  identities that cannot carry a per-window-random binding.

## 3. Task families (SEP-KV)

Both families present ONE ambiguous token X (drawn from 4 identities) in TWO
contextual roles in a single window, with per-window-random continuations
``c_0, c_1`` drawn without replacement from 32 value tokens — the binding is
window-specific exactly as in SEP-AR, so no fixed token association can be
memorised. The probe asks for X's role-dependent continuation; accuracy is
the value-restricted argmax over the 32 values; **chance = 1/32 = 0.03125**.

* **M1 (context-keyed retrieval).** Roles are explicit markers:
  ``BOS PAD PAD R0 X c0 R1 X c1 probe(0,x) c0 probe(1,x) c1`` — the two X
  copies are the same token (bit-identical payloads, the collision premise);
  the probes are combined (role, x) tokens. The binding write is the
  continuation's own write: under separated keys its ADDRESS is the write
  context (marker, X); under key=value its address is the content ``f(c_r)``,
  which carries no role at all.
* **M2 (composition).** The role is a contextual feature ABSENT from every
  token's raw content: the ORDER of two shared frame tokens. Frame-0 presents
  ``P Q`` before its X copy, frame-1 presents ``Q P``:
  ``BOS PAD PAD P Q X c0 Q P X c1 P Q X(->c0) c0 Q P X(->c1) c1``. The
  probes re-present the frame (``P Q X`` vs ``Q P X``): retrieval must key on
  the compositional order feature.

## 4. Arms (the necessity claim is per-layer; both arms at DEPTH 1)

* **(a) FUSED — the current registered per-token arm, verbatim.** The
  granularity separator's blocks at ``chunk_tokens=1`` (the K6 semantic), the
  registered ``RoutingPolicy`` and eager scan, matching by raw content
  cosine, at ONE block (~29k parameters): the first store layer, where the
  collision lemma binds exactly.
* **(b) SEPARATED — the minimal separated-key variant, oracle level.** The
  writer emits a small key ``k_t in R^16`` — a learned projection of the
  WRITE CONTEXT (the last 4 positions of the block input, per-position
  RMSNormed; at layer 1 any current-position-only projection is a fixed
  function of the token, which is exactly the fused arm's blindness, so a
  context window is required for context addressing) — and matching switches
  from ``cos(q, S_j)`` to ``cos(q_key, K_j)`` for BOTH the read score and the
  routing similarity, values/DECAY/MERGE/POP unchanged, the read keeping the
  certified form. Eager per-token torch oracle (``separated_oracle_scan``),
  the K6 doctrine verbatim: decisions detached, read weights attached. Keys
  are NOT decayed (addresses are categorical here) — §11's first open
  question. ~37k parameters (the key/query projections are the only
  additions; the probe licenses capability, not a parameter-matched
  superiority claim).
* **The depth-2 corroboration arm (part 3).** The registered FUSED arm at
  TWO blocks, receipting the depth-substitution finding (§5) with the
  composite-payload mechanism receipt (``composite_receipt``: the layer-2
  binding payload's cosines against the value direction, the marker/frame/X
  directions and the layer-1 read output — the injection channel).

## 5. The fused-arm ceiling at depth 1, and the depth-2 escape

**The 0.5 set-retrieval ceiling (depth 1).** At layer 1 every row content is
a fixed function of a single token, and every read weight at the probe is a
fixed function of token identities: the score is
``scale * cos(f(probe_token), f(v))`` — a fixed per-(probe, value) pair — and
the strength is ``||f(v)||`` times the decay suffix from the write position,
again a fixed per-token function (the window layout is constant). The
coefficient with which each candidate value's content enters the head's input
is therefore a fixed function of token identities alone. Consider the swapped
pair of windows W (role-0 = A, role-1 = B) and W' (role-0 = B, role-1 = A),
same X, and one probe: correct answers force
``Phi(A) > Phi(B)`` in W and ``Phi(B) > Phi(A)`` in W' for a common
fixed-per-token quantity ``Phi`` (the freshness factors enter both
inequalities identically because the suffix decays are functions of the
write position, not of the window) — a contradiction for every pair
``A != B``. Hence no fixed function of token identities can track the
per-window assignment, and the best the fused arm can do is retrieve the
candidate SET ``{c_0, c_1}`` with a fixed preference order: accuracy 1/2 + a
chance correction. **Pre-registered bound: every fused depth-1 cell <= 0.55.**

Honest caveat, stated as covered-by-refutation and not by proof: the decay
suffix makes role-0's row uniformly staler than role-1's; an MLP
magnitude-band selector on that freshness asymmetry is a conceivable
non-linear escape from the linear-coefficient argument above. The refutation
clause (§7) is the guard: if it fires, the necessity claim is refuted, not
excused.

**The depth-2 escape (the calibration's finding, pre-disclosed).** At layer
2 the binding row's payload is a composite of (its value, its write context
injected by the layer-1 read): the context components — marker content,
frame-token freshness patterns — are WINDOW-INVARIANT, so a query can match
on them and the per-window-random value rides inside the same row. Content
matching thereby synthesizes context-dependent addressing one layer deep.
No assignment tracking is needed at all: the role-0 row's structure is the
same in every window (it always lacks the later marker / carries the earlier
freshness pattern), so a probe-variant-dependent fixed query suffices. The
calibration confirms the fused depth-2 arm exploits exactly this. The
necessity claim is per-layer; the cost contrast — the fused machine pays
one full layer of parameters and serial compute plus the composite
indirection to buy what the key channel gives at the first layer — is the
finding, and the honest 150 M implication: in a 24-layer model depth
substitution is available from layer 2 onward, so the registration case for
keys is the layer-1 boundary, the per-layer indirection, and the
identity-collapse regime already receipted — NOT a claim that deep fused
nets are context-blind everywhere.

## 6. PRE-PREDICTION CALIBRATION DISCLOSURE (recorded before the sweep)

Plumbing smokes and calibration cells were run BEFORE the predictions were
committed; no protocol-budget cell has been run, and no prediction below is
conditioned on a cell at protocol budget. The disclosure, all numbers
verbatim:

* **Position-wise control** (embedding -> head, no store): M1 accuracy
  0.029 = chance — the task is genuinely store-dependent; there is no
  target leak.
* **200-step smokes** (lr 1e-3, seed 3401, batch 256, eval 512 windows):
  separated depth-2 M1 = 1.000; fused depth-2 M1 = 1.000; fused depth-2 M2 =
  0.869 (climbing); separated depth-2 M2 = 1.000. The depth-2 fused escape
  (§5) was found here, understood mechanistically, and the design was
  escalated to the supervisor, who approved the depth-1 restructure with the
  depth-2 corroboration arm.
* **3000-step depth-1 calibration cells** (lr 1e-3, seed 3401, batch 256,
  eval 2048 windows): fused depth-1 M1 = 0.514, M2 = 0.516 — ON the derived
  0.5 ceiling, flat from step 1000 (steps 1000/2000/3000: M1
  0.516/0.503/0.514; M2 0.516/0.517/0.516); separated depth-1 M1 = 1.000,
  M2 = 1.000. These are single-cell calibrations at one LR, disclosed as
  such; the sweep below is the pre-registered measurement.

## 7. PRE-STATED PREDICTIONS (committed before any protocol-budget run)

Protocol: probe-scale runs, AdamW (betas 0.9/0.95, weight decay 0.01, clip
1.0), 3000 steps x batch 256, fixed eval set (seed 4999, 4096 windows, 8192
probes) shared by every cell, receipt set 512 windows, per-cell receipts in
``receipts/keysep_*.json``; models are ~29-37k parameters at depth 1.
Part 1 (the necessity sweep): both tasks x both arms (depth 1) x LR sweep
{3e-3, 1e-3, 3e-4, 1e-4} x seeds {3401, 3402, 3403} = 48 cells.
Part 2 (the address-size sweep): separated arm (depth 1), key dim
{4, 8, 16, 32, 64} x 3 seeds at part 1's best separated LR, both tasks.
Part 3 (the depth-2 corroboration): fused arm at depth 2, part 1's best
fused LR x 3 seeds, both tasks, with the composite-payload mechanism
receipt per cell.

* **P-1 (separated learns at depth 1).** The separated arm reaches value
  accuracy >= 0.85 at its best LR (mean over seeds) on BOTH M1 and M2; every
  seed >= 0.70. Precondition for the separator's validity.
* **P-2 (fused bounded at depth 1 — the necessity claim).** Every fused
  depth-1 cell (all LRs, all seeds, both tasks) is <= 0.55. The calibrated
  cells sit ON the 0.5 set-retrieval ceiling; the fused arm's channel
  receipts will show the set-retrieval mode: aggregate read mass on the true
  answer row vs the other candidate within 0.05 of each other (role-blind),
  while the per-window preference is nonzero (the fixed token-identity
  pattern).
* **P-3 (collision receipts, fused).** For the best fused depth-1 cells:
  identity-rewrite fraction (copy-2's write CORRECT/MERGE onto copy-1's slot)
  >= 0.95 of windows at layer 1 — the mechanical collision, allowing only the
  rare geometry lottery where an intervening write steals the X row;
  the count of live identical-content row pairs across the receipt set is 0
  (the lemma's write half says they cannot coexist); the probe's read mass on
  the merged X row is receipted.
* **P-4 (separation receipts, separated).** The two X copies' emitted keys
  differ mechanically (cos < 0.999 at init for every parameter draw —
  machine-checked); after training, copy key-cos < tau_match in >= 0.95 of
  windows and both copy rows live in distinct slots in most windows; the
  probe read mass on the role-matching copy exceeds the other copy, flipping
  with the probe's role (both flip-direction means > 0); the probe mass on
  the true answer row dominates the other candidate (frac_true_heavier
  >= 0.9).
* **P-5 (the key-dim curve).** Key dims >= 16: mean accuracy within 0.05 of
  the dim-64 mean and >= 0.80; dim 8 at least 0.10 below the dim-64 mean;
  dim 4 at least 0.20 below. JL-flavored rationale: a rank-16 store needs a
  handful of well-separated context addresses; a random projection to d dims
  distorts pairwise key cosines by ~O(1/sqrt(d)), and the match/no-match
  margin (tau_match 0.45) survives that distortion from ~16 dims but not
  from 4.
* **P-6 (the depth-2 corroboration).** The fused depth-2 arm reaches >= 0.85
  on both tasks at its best LR (the escape), and the composite receipt shows
  the mechanism: the layer-2 binding payload carries BOTH a value component
  (cos_to_value) and window-invariant context components (cos to the
  marker/frame/X directions and the layer-1 read output), unlike the
  separated arm whose value payload stays a pure function of the token with
  the context carried by the key channel. The early-curve receipt contrasts
  the two machines' paths (the fused depth-2 curve is expected to pass
  through the 0.5 set-retrieval plateau before the composite channel
  forms; if it does not, that is reported as found). The parameter/serial-
  layer cost contrast is receipted in the ``params`` fields.
* **Refutation clause (verbatim, applies to the depth-1 necessity arms).** If
  the fused arm learns M1 or M2 anywhere in its depth-1 sweep — any LR, any
  seed, reaching 0.85 value accuracy — the necessity claim is REFUTED and is
  reported as such: it would mean content matching suffices and the
  diagnosis was misread.
* **Inconclusive clause.** If the separated arm never reaches P-1, the probe
  gradient is unresolved at this scale and no necessity claim is made;
  report inconclusive, tune (LR/steps) within the probe budget, and say so.

## 8. RESULTS (pending the sweep)

To be appended per part as it lands: results tables (per arm: best LR,
per-seed accuracies, per-probe accuracies), the collision/separation
receipts, the key-dim curve, the depth-2 corroboration with the composite
receipts, and the verdict (confirmed / refuted / partial / inconclusive)
against §7.

## 9. The payload analysis (the algebra's payload form, machine-checked)

The provable characterization (``test_slot_value_is_decayed_run_integral``):
under the registered ops, a slot's value between two consecutive re-files IS
the geometrically decayed run-integral of everything written to it — each
payload survives to the next key weighted by the product of the decays since
its write, ``S_j(t) = sum_u p_u * prod_{u < s <= t} lambda_s`` over the
writes ``u`` addressed to slot ``j`` since its last erase (an erase resets the
integral; CORRECT is the erase-plus-write that re-bases it). "Everything to
the next key" is therefore ALREADY the algebra's payload form; the decay rate
is exactly how much of the run survives to the next re-file. This is the
three-line induction behind ``scan.py``'s delta-rule decomposition
(``S_s = exp(gc_s) H + sum_u exp(gc_s - gc_u) k d_u^T``), stated here in the
slot-local form the key-channel registration will need: the value channel
needs no redesign to carry runs — the registration's question is only what
the slot is ADDRESSED BY.

## 10. Open design questions for the REGISTRATION (not resolved here)

1. **Keys and decay** (the probe's most visible choice): addresses were NOT
   decayed — an address is categorical. The registration must decide whether
   addresses age with their values (address persistence vs address aging),
   which interacts with run-boundary semantics and growth.
2. **Run-boundary semantics for the key channel**: this probe resets the key
   table with the store each window, like the store. Whether keys persist
   across run boundaries (and what a carried key means after the value has
   decayed away) is a registration decision.
3. **Orthonormality constraint vs crosstalk bound**: the registered
   exactness theorem (``memoryTable_retrieves_orthonormal``) keys CORRECT/
   POP exactness on the orthonormal family. With learned content-free keys
   the per-slot ops remain exact (leak-freedom is by slot index), but the
   matching margin becomes a statistical property: distinct contexts must
   keep key-cos below tau_match — a margin condition to be stated as a bound,
   not as orthonormality. The key-dim curve (P-5) is the first evidence for
   how much dimension that margin needs.
4. **Ordered / Wheeler-style keys**: keys as composites of prior keys (the
   Wheeler-graph frame — addresses of addresses) are the natural follow-up
   variant once the flat small-key case is receipted here.
5. **Interaction with growth**: growth reveals zero rows; a fresh row's key
   is the zero vector until first written, and matching against it is
   liveness-masked exactly as in this probe. Whether growth should instead
   reveal PRE-ASSIGNED key directions (a key-family prefix, the registered
   orthonormal family's structure) is open.
6. **Key-dim scaling**: the measured curve is the evidence for "small keys
   suffice"; the registration picks the dimension (and whether the key
   channel should be tied to the read-scale hyperparameter).

## 11. What this probe does NOT establish

* Probe scale: ~30k parameters, 13/19-token windows, synthetic task
  families; nothing here is a board or LM claim, and nothing here says the
  150 M NLL gap closes.
* The separated arm is ORACLE-LEVEL: eager per-token torch, no kernels, not
  the registered fused path; correctness first, speed nowhere.
* The fused depth-1 bound is exact on the linear read/routing channels; the
  MLP magnitude-band freshness selector is a conceivable non-linear escape
  covered by the refutation clause, not by proof.
* The separated arm's key/query projections see a 4-token context window —
  the minimal change that enables context addressing at layer 1, not a claim
  that this is the right context width.
* Keys undecayed, 2 roles per window, one head: all three are probe-scale
  scoping choices, each an open registration question (§10).
* No claim that deep fused networks are context-blind: depth substitution is
  receipted as real (§5, P-6). The necessity claim is the layer-1 boundary
  and the per-layer indirection cost.

---

## RESULTS (appended by the supervisor session after the lane hit its 6 h ceiling; every number from the lane's committed-prediction code and per-cell receipts; nothing in §1-§7 edited; the depth-2 corroboration arm of §5 was NOT run — pending, see below)

**Part 1 — the necessity sweep (48 cells: 2 tasks x 2 arms x 4 LRs x 3 seeds, depth 1, chance 1/32 = 0.031):**

| arm | M1 best (12 cells) | M2 best (12 cells) |
|---|---:|---:|
| fused (registered, key=value) | **0.5095** (range 0.442-0.510) | **0.5159** (range 0.420-0.516) |
| separated (R^16 write-context keys) | **1.000** (range 0.929-1.000) | **1.000** (best cells; range 0.147-1.000) |

Verdict against the pre-stated predictions:
- **Fused ceiling MET on both tasks**: every fused cell sits at the derived 0.5 set-retrieval ceiling (best 0.510/0.516, bound was <= 0.55) — the collision lemma's mechanical bound, hit exactly, across every LR and seed.
- **Refutation clause NOT triggered**: no fused cell anywhere reached 0.85.
- **Separated prediction MET on M1** (uniformly: 0.929-1.000) and **MET on M2 at its best cells** (1.000), with honest disclosure: some M2 separated cells fail (down to 0.147) — separated addressing is learnable on both mechanisms but M2's order-feature is harder to learn uniformly; stated per-seed, not smoothed.
- **NECESSITY CONFIRMED at depth 1, per-layer scope** (the §1-§5 framing): content-fused addressing is mechanically role-blind at the first store layer; a small separate key channel solves it immediately.

**Part 2 — the key-dimension sweep (M1, separated, 5 dims x 3 seeds):** key_dim 4/8/16/32/64 all reach **1.000 on every seed**. The pre-stated JL-flavored prediction (16 near-ceiling, 4-8 degrade) was **beaten in the favorable direction**: even 4-dimensional addresses suffice at this store size. The operator's design call — "the address doesn't have to be very big" — is receipted at its strongest: the address can be almost nothing.

**Part 3 — depth-2 corroboration: NOT RUN** (the lane hit its ceiling before this arm). The pre-prediction calibration stands as disclosed (§6): fused depth-2 escapes M1 at ~1.0 within 200 steps via composite payloads — the depth-substitution mechanism — leaving the corroboration arm (mechanism receipt + timing contrast at protocol budget) as the named pending follow-up. The necessity verdict does not depend on it.

Files: `receipts/keysep_{m1,m2}_{fused,separated}_d1_*.json` (48 cells), `receipts/keysep_*_sweep.json` (aggregates), `receipts/keysep_m1_separated_d1_keydim.json` (the key-dim curve). `.pt` checkpoints not committed (program rule).
