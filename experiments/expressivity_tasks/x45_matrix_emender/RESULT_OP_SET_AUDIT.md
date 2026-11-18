# RESULT_OP_SET_AUDIT.md — the five ops vs GDN2, op by op, in both registers

**Date:** 2026-11-22 · **Status:** analysis complete; no new training, no model
code touched · **Authorisation:** the operator's structural-audit directive —
an exact op-by-op comparison of our five ops against GDN2's mechanics, formal
and empirical, per-op honest verdicts, and the leaner-registration question —
plus the design of the GDN2+growth control experiment
(`PREREGISTRATION_GDN2_GROWTH.md`, its own document).

**Sources read (the audit's evidence base):**
`formal/lean/OPERATOR_ALGEBRA_STATUS.md` (GDN2 as an instance of the
memory-operator monoid), `ElmanProofs/Architectures/GDNMemoryOperator.lean`,
`SplitGatedDelta.lean`, `Expressivity/CompositionalDepthDuality.lean`; the
usage receipts `receipts/{k6_probe,k8_probe_grow,k8_probe_fixed2048,k10_probe}.json`
and `side_by_side_runs/` (the K6/K8/K10/side-by-side diagnostics);
`RESULT_WHEELER_CAPACITY.md` (the coalescing refutation);
`RESULT_GRANULARITY_SEPARATOR.md` (the separator's within-family scope);
`RESULT_MATRIX_EMENDER.md`, `RESULT_SIDE_BY_SIDE_0239.md`,
`RESULT_GDN2_LR_BRACKET.md`, `../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md`,
`K6_SEQUENTIAL_MATRIX.md`, `K10_SEPARATED_KEYS.md`, `K8_GROWTH.md`,
`CMAES_TUNING.md`; `../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`
(the ops' registered purposes and claims C1–C4), `ARCHITECTURE.md` and
`DESIGN_SUFFIX_ATTENTION.md` (the controller/spine story); GDN2's model source
read-only (`lit_gpt/gdn2.py`, `lit_gpt/gdn2_ops/chunk_gdn2.py`) and the
registered arm's build path (`part8_surface.GDNBlock` →
`GDN2ExternalMLPLayer(expansion=2., head_dim=128, num_heads=2)`).

**The registered GDN2 arm's actual config, from the receipts and build path**
(width 576, depth 24, 149,855,670 parameters): per head `d_k = 128` key
directions, `d_v = 256` value dims (`expand_v = 2`), **2 heads**; per-token
per-head inputs `q, k, g, b ∈ ℝ^{128}` (key side) and `v, w ∈ ℝ^{256}` (value
side); state `S ∈ ℝ^{128×256}` per head. The one-step law, with k̂ the
l2-normalised key and `D = diag(exp(g))` the channel-wise decay:

```
S_t = (I − k̂_t (b_t ⊙ k̂_t)^T) D_t S_{t−1} + k̂_t (w_t ⊙ v_t)^T
        erase along the current key        decay        dense rank-1 write
o_t = S_t^T q̂_t            (the full-state read; no candidate subset)
```

---

## 0. The two machines in one formal frame

The Lean algebra puts both machines in **one monoid**
(`MemoryOperator Op S`: a step function, an affine embedding, equality on every
state; `MemoryAlgebra.AffineOperator` product applies the left operand first).
GDN2 is *not* a rival formalism — it is a named instance:

* `GDNMemoryOperator.embed_correct` / `GDNMemoryOperator.gdn2_conditional_scan`
  / `GDNMemoryOperator.gdn2_fold_scan`: one GDN2 step
  `gdnStep p H = (splitGatedTransition 1 p.k p.b * p.decay) H + outerKV p.k (w ⊙ v)`
  is the affine operator `(L, b)` with `L = (λI − k(b⊙k)^T)·decay` and
  injection `k(w⊙v)^T`; GDN2 scan-composes and chunk-scan-equals-sequential
  under a fixed (gate, decay, value) schedule, by the *same theorems* as our
  store (`EmenderMemory.token_conditional_scan`,
  `Event.combined_chunked_scan`).
* `gdn_state_channel_depth_bounded_in_time`: the GDN state channel is affine at
  every step — a fixed-dimension linear summary `(A, C)` exists for any finite
  schedule. Our store has the same shape at fixed rank
  (`EmenderMemory.step_decomposition`: `D ⊙ s + Σ_j e_j ⊗ b_j`).
* Both instantiate `IsMultiProgrammedRecurrentStack`
  (`RecurrentResourceFormalism.gdn2_multiProgrammed`,
  `slotStore_multiProgrammed`).

**The theorem-level consequences that bind BOTH machines at fixed rank:**

1. **The fixed-state automata bound applies to BOTH.** At fixed rank (ours) or
   fixed `(d_k, heads)` (GDN2), both state channels are fixed-dimensional
   affine maps under a fixed schedule; neither can represent behaviour needing
   more state than its fixed dimension (the Elman-class ceiling; the empirical
   shadow is receipted by the Wheeler probe's pointwise collapse past capacity
   and by the C3 claim's own framing). The only exit in the whole formalism is
   a **literal change of state space**: `LiftedMemory` lifts arbitrary
   transitions only by embedding the *full* state in a free vector space —
   no compactness claim — and the legal practical form is zero-injection growth
   (`injection_propagation`, `injection_eq_suffix_sum`: new state components
   enter as exact zeros and every monoid/scan theorem continues to hold).
2. **POP/MERGE/growth are the only exact-capacity / unbounded-state
   operations in the program.** `only_append_allocates`,
   `append_allocates_only_selected`, `pop_death_iff`, `freeMask_iff`: only
   APPEND allocates, only POP frees (and MERGE frees one slot while moving its
   mass); deaths are irreversible (zero diagonals, no inverse ops —
   `injection_propagation`). Growth is the only op that raises the state
   ceiling itself. **GDN2 has none of the three** (§2 shows its erase/decay are
   never exact and its shape is constitutionally fixed) — at fixed config GDN2
   is permanently inside the fixed-state class.
3. **Chunked-equals-sequential licenses both chunk kernels, with different
   premises.** For GDN2 the "schedule" is the gate/decay/value sequence, a
   deterministic pointwise function of the inputs — chunking GDN2 is legal with
   no expressivity caveat (this is what its Triton WY kernel rests on). For
   our machine the same law holds **conditional on the fixed decision
   schedule**; the discrete schedule is a *generated* object, and the
   granularity separator receipts that generating it at chunk granularity is an
   expressivity weakening (entry-pinned reads put exactly 0 weight on
   mid-chunk-written rows). The separator's scope is **within-family**: it is
   a theorem-shaped claim about *this algebra's* chunked router
   (`RESULT_GRANULARITY_SEPARATOR.md` §1, §7). It does not touch GDN2, whose
   "decisions" (the gates) are pointwise in the input with no state-dependent
   discrete selection to delay. **No cross-family separator
   emender-vs-GDN2 exists yet** — not formal, not empirical: no task family has
   been registered or run where the two machines' *solved classes* are proven
   or measured to differ. Every measured difference so far is a quality margin
   on tasks both solve (nats/byte), not a capability separation.
4. **The truncation laws bind our read family, not GDN2's.**
   `AttentionOperator.aggregates_independent_of_truncation`,
   `truncation_sharpens_survivor`, `full_normalization_loses_omitted_mass`:
   when a read truncates its candidate set, retained-normalisation can sharpen
   survivors and full-normalisation drops mass, so truncation must be
   disclosed. Our read computes softmax weights over the **live subset**
   (entry-pinned at chunk granularity) — the laws bind it and the Wheeler
   probe receipted the omitted-mass duty. GDN2's read `o = S^T q` at fixed rank
   is a **full-population** read over the whole state — no candidate subset, no
   truncation, the laws vacuous. They become live for GDN2 only in a *growth*
   design that ever caps the read (the GDN2+growth registration does not).

---

## 1. The op-by-op formal mapping

| our op (registered form) | GDN2's mechanics | formal verdict |
|---|---|---|
| **APPEND** `S += v k_j^T`, allocate a free slot | the rank-1 injection `k̂(w⊙v)^T` fires at **every token** into the dense state | **write half: instantiated (denser)**; **allocation half: LACKED** |
| **CORRECT** `S[j] = v'` exactly, preserve key/strength/age | the delta `(w⊙v) − S^T(b⊙k̂)` corrects the readout along the *current* key, blended by the erase gate `b` | **instantiated in a strictly richer, softer form** (per-token, per-channel, always-on) |
| **POP** `S[j] = 0` exactly, free the slot | no slot, no liveness, no allocator; the erase `(I − k̂(b⊙k̂)^T)` acts only along the *current* key, and `D = diag(exp(g))` with `g < 0` strictly | **LACKED** (approximated along the current key only; never exact) |
| **DECAY** `S ← λS`, learned per-token per-head scalar | `D = diag(exp(g))`, per-channel, input-driven (`A_log` per head × `softplus(f_proj + dt_bias)` per channel) | **instantiated in a strictly richer form** |
| **MERGE** `old + β(new − old)`, free the source slot | the dense state superimposes every write; there are no two slots to consolidate | **LACKED — and vacuous** in a superposed state |
| **growth** rank `r' = snap(⌈g·r⌉)` at boundaries, zero-injection | nothing: `d_k`, `d_v`, heads are fixed configuration constants | **LACKED categorically** |

### 1.1 APPEND — the write is instantiated, the allocation is not

GDN2's injection `outerKV k̂ (w⊙v)^T` *is* a rank-1 APPEND into the state, at
every token, with a per-channel value gate `w` our APPEND does not have. What
GDN2 does **not** have is the allocation act: `only_append_allocates` /
`append_allocates_only_selected` — the choice of *which new address* the write
creates, the liveness channel, and the free-slot set. In GDN2 the write lands in
a fixed superposition; no address is ever "new".

### 1.2 CORRECT — the delta rule is a strictly richer soft CORRECT

With `b = 1` (and `‖k‖ = 1` by the kernel's l2norm) the GDN2 step along the
current key direction is *exactly* our CORRECT: the old readout
`S^T(b⊙k̂) = S^T k̂` is subtracted and `(w⊙v)` written in its place. With
`b = 0` it degenerates to pure APPEND along `k̂`; with `b ∈ (0,1)` it is a
convex blend — the *same interpolation our MERGE performs between two slots*,
executed inside one update along the current key. So GDN2 instantiates
APPEND/CORRECT/blend in a single always-on op, per channel. What it lacks is
exactness (`b` is a continuous gate, not a schedule bit), and *addressing of a
stored key*: the correction always targets the current token's key direction,
never a stored `k_j` chosen by match — `correct_preserves_key` (our CORRECT
preserves the stored key, strength, age) has no GDN2 analog because nothing is
stored discretely at all.

### 1.3 POP — the one exact-capacity op, and GDN2 has no analog

To free a stored direction exactly, a machine must zero a row and mark it
reusable (`pop_death_iff`, `freeMask_iff`). GDN2's two erasure channels cannot:
the rank-1 erase `I − k̂(b⊙k̂)^T` acts only along the *current* key (trace
`K − Σ b_i k̂_i² ≥ K − 1`: it removes at most one direction's worth of mass,
partially, and only the incoming one), and the decay `exp(g) < 1` strictly, so
the state is emptied only asymptotically, never at a step. There is no
allocator to return a slot to. **Consequence (fixed-state bound, §0.1–2):**
our POP/MERGE/growth are the program's only exits from the fixed-state class;
GDN2 has none of them.

### 1.4 DECAY — ours is the strictly poorer instance

Our DECAY is the *only learned gate* in our five-op set: a per-token, per-head
scalar `λ` — one `nn.Linear(576, 2)` per layer, the 27.7K-parameter "decay pair"
of the whole 150 M machine, floored at 1e-4). GDN2's
decay is per-channel, input-driven, per token (the `g` projection stack), part
of **27.4M learned gate-matrix parameters — 18% of the machine against our
0.018% plus three hyperparameter thresholds** (`RESULT_GDN2_LR_BRACKET.md`
§5). Formally our `S ← λS` is the special case `g_i = log λ ∀i`. The running
K11 LEARNED-GATES registration is the attack on exactly this deficit.

### 1.5 MERGE — lacked, and vacuous where it is lacked

MERGE needs two distinct slots holding similar content
(`merge_delta_rule`: `old + β(new − old)`, source freed). GDN2 has no two
slots: every write interferes with the whole superposition, i.e. **every GDN2
write is already "merged"** — consolidation is not an operation a dense state
can lack and miss, it is its default condition. The empirical question is
whether *addressable* consolidation ever pays; §2/§3 receipt that it has not.

### 1.6 growth — the categorical difference

GDN2's state dimensions are configuration constants; nothing in its update or
kernels contemplates a state that changes shape mid-sequence. Our growth is
Lean-legal by zero-injection, wired (K8), gated (0 decision flips in 2.16 M
lanes; gradients within floors across rank changes), and the only unbounded-
state op in either machine's vocabulary.

---

## 2. The empirical register — the receipts per op

**Usage (lane-0 op rates per token, the receipted probes; chance for context:
the arms' evals are in the cited receipts):**

| op | K6 per-token (banked, 2.0278) | K8 grow (2.0186) / fixed2048 (2.0240) | K10 separated keys (1.8418) | side-by-side tuned winner @180 (1.8227) |
|---|---:|---:|---:|---:|
| CORRECT | 0.884 | 0.928 / 0.929 | **0.979** | 0.9789 |
| APPEND | 0.084 | 0.072 / 0.071 | 0.017 | 0.019 |
| POP | **0.031** | 0 / 0 | 0 | 0 |
| MERGE | **1.7e-4** (134 events / 20 M targets) | 1.9e-4 / 2.1e-4 | 3.1e-3 | 2.3e-3 |
| (live slots / rank) | 73.7 / 128 | 84.7 / up to 2048 | 24.0 / 128 | 27 / 128 |

Per-op receipts:

* **CORRECT is the byte-identity op.** The best arms run CORRECT at 97.9–97.9%
  of tokens, and its content is *identity rewriting*: identity share 0.103
  (K10 separated, from 0.519 at init), layer 0 pinned at 0.61–0.87 at every
  checkpoint (structural: the first store layer's payload is a near-pure
  function of the byte); the 150 M diagnosis receipts 60–71% identity
  rewriting (`CMAES_TUNING.md`). K6's dial falsified the draft's MERGE-recovery
  prediction and showed redundancy is resolved *at write time* by CORRECT.
* **POP is idle in every winning arm** (0 in K8/K10/side-by-side; 3.1% in the
  banked K6 probe; up to 7.7% in per-layer/init-anchor cells; "marginal" at
  150 M). Occupancy never approaches the ceiling in the trained probes
  (24–85 live slots of 128+); the chunked arms reached 92% at chunk *entries*
  — near, not at, capacity.
* **MERGE is ~dead.** 134 events in 20 M targets (K6); 0.02–0.3% everywhere
  else; starved 27× from chunk 128 to 8 in the dial; the recovery prediction
  falsified; and the granularity separator's §2 analysis shows *why*
  structurally: a `tau_merge=0.85`-similar payload forces the candidate rows
  within cosine 0.445 of each other, but `tau_match=0.45` self-deduplication
  CORRECTs the second write before the pair can exist — the mergeable pair
  cannot arise under the policy that would exploit it.
* **DECAY: our only learned gate vs GDN2's richer one, with the convergence
  receipt.** Our gate *learned its way* from λ=0.983 (a ~40-token half-life,
  a nearly-empty store) to λ=0.862 (2.9–25.9-token row half-lives) — the
  trained machine **converged on GDN2's own memory horizon** (GDN2's implied
  horizons 2.1–7.4 tokens/layer, retention exp(g) 0.529–0.864). GDN2's gates
  were ≈0.5 on every channel at init and essentially unchanged by training:
  its memory regime is *architectural*, on from step 1 — which is exactly the
  side-by-side's early-phase receipt (the deficit is largest at steps 10–20
  while our store learns its regime).
* **The dense-write C5 receipts (the load-bearing comparison).** GDN2 writes a
  dense, channel-gated rank-1 delta into the FULL state at EVERY token (write
  gate w ≈ 0.5 on every channel from init); our winner writes a sparse hot set
  (~27 live slots) whose reads support ≈ e^1.73 ≈ 5.6 effective rows. The
  side-by-side verdict: C5 — the value/read **form** — owns the ~0.16-nat
  steady-state deficit (timescale and addressing are receipted NOT the
  residual); the config-symmetry program closed every surface alibi and
  settled the structural residual at **~0.09 nats best-vs-best, GDN2's favour**
  (`RESULT_GDN2_LR_BRACKET.md`), with the architectural face receipted as
  27.4M learned gate params vs our 27.7K + three thresholds. This is a
  receipt *against our op vocabulary earning its cost at byte-LM scale*, and
  it is what K11 is attacking.
* **The Wheeler coalescing refutation (the MERGE-analog control).** The
  ordered run-coalescing store — the interval-read variant whose
  update-in-run *accumulate* is MERGE's semantics applied to runs — was
  **refuted at this scale**: it lost to the pointwise store at S = 8 and 16,
  tied at 32, chance at 64 (pre-registered P-5 failed at every long span).
  The mass-preserving machinery itself worked (runs built, nothing evicted,
  Wheeler contiguity 0.92) — what failed is converting held mass into
  per-address resolution (dilution law ≈ mass-on-row / run-length). Receipt:
  coalescing does not buy resolution at this scale; it is evidence *for*
  demoting MERGE in the byte-LM arm, scoped honestly as one probe's scale and
  budget.
* **The granularity separator's scope, stated exactly.** Its result —
  chunk-granular decision *generation* is an expressivity weakening (exact-0
  read weight on live mid-chunk answer rows) — is a claim about *this
  algebra's* chunked router with entry-pinned reads and free-slot sets
  (`RESULT_GRANULARITY_SEPARATOR.md` §1). **No cross-family separator
  emender-vs-GDN2 exists yet**: no registered or run task family separates the
  two machines' solved classes. The audit's formal section (§0.3) gives the
  reason no such separator is currently constructible from the banked theorems
  either: the Lean results are schedule-conditional and cover both machines
  identically; the differentiators (§0.2) are capability classes whose value
  tests (C3) have not been run at any scale.

---

## 3. The honest verdict, per op

The task's three-way taxonomy, applied with receipts:

| op | verdict | one-line justification |
|---|---|---|
| **CORRECT** | **redundant-with-GDN2's-analog** | the delta rule instantiates it per-token, per-channel, always-on; empirically it is our dominant op (97.9%) but its content is byte-identity — the very channel GDN2's dense gated write does richer (C5) |
| **APPEND** | **write: redundant-with-GDN2's-analog; allocation: idle-for-byte-LM, essential-for-controller-story** | the rank-1 write is GDN2's every-token injection; the *allocation* half (liveness, free-slot choice) is lacked by GDN2, is the addressable-slot act, and is empirically idle at byte-LM scale (occupancy 24–85 of 128+; 1.8–8.4% of tokens) |
| **POP** | **idle-for-byte-LM, essential-for-controller-story** | 0–3.1% everywhere, 0 in every winning arm — but it is the *only exact-capacity op in either machine's vocabulary* (GDN2 lacks any analog, §1.3), and the controller program's ownership-expiry verb |
| **DECAY** | **redundant-with-GDN2's-analog** | strictly poorer instance (per-head scalar vs per-channel input-driven); the trained machine converged on GDN2's timescale anyway; K11 is the registered attack on the gap |
| **MERGE** | **~dead at byte-LM; demote-to-controller-only is the honest reading** | geometry-pinned (the mergeable pair cannot arise), 0.02–0.3% usage, dial-falsified recovery, Wheeler-refuted coalescing payoff — but consolidation is live in the controller setting (below) |
| **growth** | **essential-for-the-state-class-story** | the one capability GDN2 categorically lacks (§1.6); Lean-legal, wired, gated; probe-scale utility null receipted honestly (2.0278 / 2.0186 / 2.0240 three-way tie — content diversity, not size); C3 untested |

**The controller/spine position (why the idle ops are not dead weight).** The
byte-LM arm is one consumer of the op vocabulary. The registered program's
architecture (`ARCHITECTURE.md` §3/§8, `DESIGN_SUFFIX_ATTENTION.md` §3/§7)
defines the ops as the **lifecycle verbs of a weighted interval set — the
controller vocabulary for corpus-index operations**: APPEND = hypothesis
enters; CORRECT = evidence reweighting; POP = ownership expiry (tombstone
first, filter before support, compact later); DECAY = staleness weighting;
MERGE = consolidation / component collapse. In that setting POP and MERGE are *not* optional: the C14-B
store door receipts the ops firing meaningfully (answers APPENDed under native
keying, CORRECT/MERGE by slot metadata, POP/DECAY per ownership — capacity
merges and refusals as receipted events; PB3 telemetry "all op classes firing"),
and the suffix design's MERGE row is the *physical cross-LCP consolidation*,
semantics-preserving by construction. The string-kernel position
(`DESIGN_SUFFIX_ATTENTION.md` R2.1–R2.4): the ops manage a
maximal-exact-match attention object over the model's own context and the
corpus spine — a program that is designed, prototyped at the chaining level,
and **unexplored as a trained controller**. The honest statement: POP/MERGE are
idle-for-byte-LM on today's receipts, essential-for-controller-story on the
program's registered design, and the controller story itself remains
unreceipted beyond the C14-B store door.

---

## 4. The leaner-registration question (a registration OPTION, receipts on both sides)

> Is there a minimal op set for the byte-LM arm — dense/gated write + DECAY +
> POP + growth, with MERGE demoted to controller-only?

**Receipts FOR the leaner set:**

1. MERGE: geometry-pinned + 0.02–0.3% + dial-falsified + Wheeler-refuted
   (§2). Demoting it costs nothing any receipt has measured at byte-LM scale.
2. POP is 0 in every winning arm — but note it is *retained* in the leaner set
   precisely because it is the only exact-capacity op and the C3/saturation
   story is its home: K8's own verdict names the saturation regime as "the one
   place the 2048 target is testable in value".
3. CORRECT's byte-identity content + the C5 receipts say the *value/write form*
   is the binding residual, not the discrete op vocabulary: a byte-LM arm whose
   write form is GDN2-like (dense, gated, per-channel) needs CORRECT only as
   the discrete special case, if at all.
4. Growth: legal, wired, and the one capability GDN2 lacks — the leaner set
   keeps exactly the ops that differentiate (POP + growth) plus the ones GDN2
   proved necessary (dense gated write + DECAY).

**Receipts AGAINST (or: against deciding now):**

1. **C1's registered test is the five-op bundle at the 2700 s board, and its
   board-scale verdict has already landed the other way once**:
   `matrix_ops` 1.7576 vs `gdn2` 1.1976 same-session — the ordering
   *five-ops > append-only > no-management* is **not met at 150 M** and C1's
   falsifier fired with its scoping clause applied
   (`RESULT_150M_MATCHED_ARMS.md`). A leaner registration would re-enter the
   same comparison with fewer moving parts — which is an argument *for* the
   leaner set, but it also means the demotion decision is entangled with C1's
   already-falsified form and should be registered as its own claim, not
   smuggled in as a simplification.
2. **K11 LEARNED-GATES is running now on exactly the write-form axis**
   (cards 0/1). The leaner set's "dense/gated write" item is K11's territory:
   if K11's learned-gate arm closes the C5 gap, the op-set question changes
   shape (the discrete ops become clearly residual); if it does not, the
   leaner set is the honest byte-LM arm. Registering the minimal set *before*
   K11 lands risks double-registering the same axis.
3. MERGE's controller role fires on real receipts (C14-B, PB3), so "demoted"
   must mean **byte-LM-arm-only demotion**, never program-wide removal —
   the suffix/spine program's MERGE is the physical-consolidation verb.

**Stated as an option:** a `byte-LM-minimal` arm (dense gated write + DECAY +
POP + growth; MERGE controller-only; CORRECT as the delta rule's discrete
special case or removed if K11's gates subsume it), sequenced **behind K11's
result**, registered against C3's saturation regime rather than the plain
board. The decision is the operator's; this audit's receipts support the
demotion of MERGE and the retention of POP+growth on the byte-LM arm, and
defer everything else to K11.

---

## 5. Findings where the audit's verdicts conflict with registered claims (receipted, per the directive)

These are **findings, not problems** — each is a place where the honest
per-op reading diverges from what a registration anticipated:

1. **C1 (management) is falsified at board scale, and the audit explains the
   mechanism.** The registered ordering expected the five ops to *earn their
   cost*; the receipts show the ops that differentiate us from GDN2 (POP,
   MERGE, growth) are precisely the idle ones at byte-LM scale, while the op
   GDN2 shares (the correction/decay/dense-write family) is the one where
   GDN2 is *richer* (per-channel, always-on, 27.4M gate params). C1's
   scoping clause was applied honestly at 150 M; the audit's contribution is
   the decomposition of *why*.
2. **The registered growth trigger fires on an idle store** (K8 §5): the
   append-collision statistic counts identity-CORRECT write-reuse as
   collision, so the rank climbs 128→2048 at rate ~1.0 while occupancy sits at
   ~48/128. The registration's intent (a capacity signal: "write-residual
   mass and/or append-collision rate") and the implemented statistic's
   behaviour diverge; K8 retained the implemented statistic verbatim per the
   operator's instruction, and the divergence is receipted. Any future
   growth statistic (including GDN2+growth's, the companion document) must
   answer this hazard explicitly.
3. **MERGE's registered management role vs its receipts**: registered as
   "consolidation of indistinguishable slots; capacity merges under pressure"
   — receipted as structurally unable to arise under the arm's own thresholds
   (§2), falsified in the dial, and its coalescing payoff refuted by Wheeler.
   The role survives only in the controller setting (C14-B metadata
   consolidation), where it fires.
4. **C3 (state class) is untested at any scale** — and the audit sharpens
   this: the *only* formal capability advantages our machine holds over GDN2
   (exact-capacity POP, addressable allocation, legal growth) are exactly
   the claims C3/C2 would test. Until the saturation regime runs, the
   program's differential value over GDN2 is a proven capability with a
   receipted utility null, and the honest headline is the audit's §3 table.

## 6. What this audit does NOT establish

* No new measurement: every number is from the banked receipts cited.
* No cross-family separator (§2, stated explicitly) — the capability
  differentials are formal/structural, their *value* is untested (C3).
* No claim about the controller program's trained value (unexplored, §3).
* The verdicts are op-level, at the byte-LM arm's receipted scales (2.13 M
  probes, 150 M board); the C14-B/C14 receipts live at the 2.13 M class and
  are cited as such.

## 7. Companion deliverable

`PREREGISTRATION_GDN2_GROWTH.md` (repo root, draft, operator sign-off pending):
the control experiment this audit sets up — give GDN2 the one capability it
categorically lacks (growth) and test, in the C3 saturation regime, whether
*growth* is the differentiator independent of slot architecture, or whether
addressable slots earn their keep.
