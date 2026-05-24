# Lean Formalization Gap Audit — Against Revised Paper Claims

**Generated:** 2026-05-24
**Branch:** wg/agent-46/lean-formalization-gap-audit
**Task:** lean-formalization-gap-audit
**Audience:** team lead (internal coordination memo)

This audit maps the revised three-pillar paper framing onto the current trusted
Lean core (`ElmanProofs.PaperCore` import closure) and recommends what to
formalize next. It does not modify any `.lean` file.

Sources read:
- `formal/lean/PROOF_INVENTORY.md`
- `formal/lean/TRUSTED_PROOF_SURFACE.md`
- `formal/lean/S5_NC1_WITNESS_PLAN.md`
- `formal/lean/S5_FULL_PROOF_STATUS.md`
- `formal/lean/TC0_CLAIMS_STATUS.md`
- `formal/lean/ElmanProofs/PaperCore.lean`
- `formal/lean/ElmanProofs/Expressivity/S5NDMRealization.lean`
- `paper/OUTLINE.md`
- `paper/ndmpapernotes.md`

The trust gate task `lean-fix-trust-gate` is **still in-progress** (assigned to
agent-45); `formal/lean/build_logs/` does not yet exist. This audit therefore
treats the inventory's claim of an empty placeholder set in PaperCore as
**unverified**. Any new placeholders surfaced by that task will need to be
folded into the follow-on list below.

---

## 1. Revised-Claim → Lean-Status Mapping

The three pillars (from the task brief):

> **P1.** Generality of multi-programming — pure nonlinear recurrent LMs can be
> trained at scale; multi-programming is the enabling systems idea, **not
> specific to NDM**.
>
> **P2.** Update rule matters — the delta-correcting write `v − Sᵀk` separates
> from raw-write matrix RNNs (M2RNN family) and **approaches the NC1 limit** in
> expressivity.
>
> **P3.** FLOPs-per-bit convergence under CMA-ES — all sensibly-tuned recurrent
> baselines learn at the same FLOPs-per-bit (N=4). NDM's contribution is the
> architectural option + the mechanism, not speed.

| Pillar | Sub-claim | Status | Lean witness (precise name) |
|---|---|---|---|
| **P1** | NDM at 1.27B IS a multi-programmed shape (per-head, per-state-tile, per-batch) | **(a) formalized** | `RecurrentResourceFormalism.ndm_1p27B_is_pure_nonlinear_recurrent_stack`; `ndm_1p27B_programs_per_batch_token` (= `12 * 370 * batch`); `ndm_1p27B_programs_per_batch_token_bs5` (= 22200); `ndm_1p27B_state_scalars_per_layer` (= `370 * 32 * 32`) |
| **P1** | NDM has matrix state and is a pure nonlinear recurrent stack | **(a) formalized** | `ndm_1p27B_has_matrix_state`, `ndm_1p27B_has_delta_memory`, `pure_m2rnn_is_nonlinear_matrix_recurrent` (M2RNN counterpart) |
| **P1** | Multi-programming is *general* (other update families admit the same shape) | **(c) not formalized** | None. No type-level predicate exists for "is a multi-programmed recurrent stack" parameterized over update family. The current 1.27B theorems are NDM-specific aliases over `e88NDM_1p27B`. |
| **P1** | "Pure nonlinear recurrent LMs **can be trained** at scale" | **(c) not formalizable as stated** | This is an empirical training-trajectory claim. No Lean witness; should not be formalized — paper must carry this on the wallclock racer plots. |
| **P2** | NDM update family ≠ M2RNN update family (feature/signature level) | **(a) formalized** | `M2RNNComparison.m2rnn_features_not_e88_features`; `RecurrentResourceFormalism.e88_and_m2rnn_differ_as_one_step_transition_families`; `ndm_and_m2rnn_differ_as_one_step_transition_families` |
| **P2** | Delta write `v − Sᵀk` exactly overwrites under a unit key (the mechanism, in isolation) | **(a) formalized** | `OnlineMemory.linearDeltaWrite_exact_overwrite`, `linearDeltaWrite_preserves_orthogonal_query`, `linearDeltaWrite_uniformOneStepOverwrite` |
| **P2** | Raw outer-product write (M2RNN) cannot satisfy uniform one-step overwrite | **(a) formalized** | `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite`; `stateIndependentAdditiveWrite_not_uniformOneStepOverwrite`; combined: `delta_core_separates_gdn_ndm_from_m2rnn_raw_write` |
| **P2** | One-step update separation: no fixed-W, fixed-row/column/cell-forget M2RNN matches NDM's mixed-key delta correction | **(a) formalized** (at K=2, V=1 and in general K≥2, V≥1) | `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation`; `ndm_m2rnn_one_step_resource_separation_embeds`; constituent fail-lemmas for row/column/cell forget gates |
| **P2** | M2RNN *can* simulate one NDM step given an extra read-then-delta resource | **(a) formalized** (positive direction) | `M2RNNComparison.m2rnn_read_then_delta_embeds_e88_delta_update` |
| **P2** | NDM approaches the **NC1 limit** | **(b) partial** | Trusted core has the *ceiling* (`S5Witness.fixed_precision_state_space_finite`) and the *abstract lookup realization* for ANY finite recognizer (`S5NDMRealization.exactTransitionMemory_run`) instantiated to S5. The bridge "NDM-architecture parameters realize the S5 lookup table" is **missing**. See §2 for the recommendation. |
| **P2** | Multi-step / training-time separation between NDM and M2RNN | **(c) not formalized** | The current separation is per-step on a fixed shape. See §3 for whether this is worth extending. |
| **P3** | All sensibly-tuned recurrent baselines learn at the same FLOPs-per-bit | **(c) not formalized; not formalizable as stated** | Empirical learning-curve claim with N=4 baselines. No Lean theorem exists. A *weak* architectural anchor (per-token FLOP cost class) could be formalized; see task F3 in §4. |
| **P3** | NDM's contribution is the architectural option + the mechanism, not speed | **(b) partial** (the mechanism part) | Architectural option is captured by `ndm_1p27B_*` and the M2RNNComparison witnesses. "Mechanism" coverage is the delta-write separation in P2. The "not speed" framing is a negation of a claim the paper does not need to prove. |

### Aggregate read

- **Pillar 2 is the best-covered pillar.** All separation results the paper
  needs in §6/§7 (one-step resource separation, embedding theorems, mechanism
  lemmas) are in the trusted core.
- **Pillar 1 has a witness but no generality theorem.** The Lean core proves
  NDM *is* a multi-programmed stack; it does not prove the *category* is
  achievable by other update families. The paper currently leans on the
  empirical M2RNN-CMA shape for this. A small Lean addition would tighten it.
- **Pillar 3 has no Lean coverage** and most of it should not. The
  FLOPs-per-bit convergence claim is empirical. Only a per-token cost-class
  claim is formalizable, and only weakly supports the paper.

---

## 2. NC1 Claim — Special Attention

### What the trusted core already proves

1. **Finite-state ceiling at fixed precision.**
   `S5Witness.fixed_precision_state_space_finite` shows every
   `FixedPrecisionOnlineRecognizer` has a finite state space (typeclass
   witness). This bounds the model class to regular languages, which sit
   inside NC1. The ceiling is real.

2. **S5 word problem witness.**
   `S5Witness.s5_state_count` (= 120), `S5Witness.s5_not_solvable`,
   `S5Tracker.recognizer_state_count`, `S5Tracker.run_append`,
   `S5Tracker.pythonRun_eq_tracker_tuple`, and
   `S5NDMRealization.s5_transition_key_count` (= 480) jointly check that the
   S5 prefix-product tracker is a 120-state non-solvable DFA whose Lean
   semantics agree with the Python evaluation harness.

3. **Abstract lookup-table realization.**
   `S5NDMRealization.exactTransitionMemory_run` proves that **every** finite
   fixed-precision recognizer has an exact `(state, input) → state` lookup
   table; `s5TransitionMemory_run` instantiates this for the S5 tracker. This
   is the closest existing theorem to "an NDM-shaped read-then-write can
   realize S5".

### What the trusted core does NOT prove

- It does not prove **NDM-architecture parameters** (orthonormal key family,
  value vectors, decay schedule, output gate) can be *configured* to realize
  the `s5TransitionMemory` lookup. The lookup memory is an abstract
  `LookupMemory` record; the bridge to `ndm/models/e88_fused.py:298-316`
  semantics is informal.
- It does not prove **Barrington's theorem** (S5 word problem is NC1-complete)
  nor any S5 lower bound for linear/scan models. Both remain external citations.
- It does not prove **NDM at width d ≥ X tracks S5 exactly**. There is no
  S5NDMRealization theorem whose statement names the NDM update equation.

### Evaluation of the three suggested formalization scopes

| Option | Statement (informal) | Already in core? | Effort to formalize | Recommendation |
|---|---|---|---|---|
| (i) NDM with d ≥ X width tracks S5 exactly | "There exist NDM parameters (K, V, d, decay) at d ≥ 12 such that the NDM update produces the same state trajectory as `S5Tracker.run` modulo an orthonormal embedding of `S5State → ℝ^d²`." | **No.** S5NDMRealization proves only the abstract-memory realization. The NDM-update-equation bridge is missing. | **M** (orthonormal embedding + key/value construction; reuse of `linearDeltaWrite_uniformOneStepOverwrite` and the orthonormal-table theorems `memoryTable_retrieves_orthonormal` and `linearDeltaWrite_overwrites_one_preserves_others` from `OnlineMemory.lean`) | **Formalize.** This is the right scope. |
| (ii) NDM update family can implement any Barrington program of bounded width | "For every Barrington branching program (M, w) recognizing a language L in NC1, there exist NDM parameters whose state trajectory accepts L." | **No.** | **L** (requires formalizing branching programs and Barrington's reduction; mathlib has no Barrington formalization) | **Do not formalize before submission.** Out of scope for one paper cycle. |
| (iii) NDM has finite-state ceiling at fixed precision | Already true. | **Yes.** `S5Witness.fixed_precision_state_space_finite` | trivial | Keep and **cite explicitly** in the paper's §7 non-claims paragraph as the *ceiling* the model lives under. |

### Recommendation (explicit, per task spec)

**Formalize option (i).** Add a `NDMRealizesS5` module proving an orthonormal
key/value embedding under which the NDM update equation realizes the
S5 transition table, with the trajectory bridge to `S5Tracker.run` proved via
`linearDeltaWrite_overwrites_one_preserves_others` and the existing
`s5TransitionMemory_run`.

**Scope to S5 only.** Do not generalize to Barrington width-5 programs in
Lean.

**Do not claim:** (a) Barrington's theorem inside Lean; (b) "NDM exceeds NC1"
at any precision regime; (c) any linear-scan lower bound. These remain cited
background, per `S5_NC1_WITNESS_PLAN.md` and `TRUSTED_PROOF_SURFACE.md`.

**Paper wording to match the formalization scope.** Replace any phrase like
"approaches the NC1 limit" with:

> NDM is, at fixed width and precision, a finite-state recognizer (proved in
> Lean as `fixed_precision_state_space_finite`). Within that ceiling, an
> orthonormal-key configuration of the NDM update realizes the S5 prefix
> tracker (proved in Lean as `NDMRealizesS5.ndm_realizes_s5_tracker`); S5 is
> non-solvable (`s5_not_solvable`) and by Barrington's theorem (cited, not
> formalized) the S5 word problem is NC1-complete. NDM therefore reaches the
> top of NC1 in the canonical regular-language witness.

This is the strongest *responsibly* formalizable version. The paper should
state "reaches the top of NC1 in the canonical witness", **not** "approaches
the NC1 limit in expressivity", because the latter implicates a families-wide
claim Lean does not check.

---

## 3. Delta vs Raw-Write Separation — Scope Verification

### What is proved

The trusted core proves a **one-step** resource separation, in two strengths:

- 2D base case: `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation`
- General embedding: `ndm_m2rnn_one_step_resource_separation_embeds` (every
  K ≥ 2, V ≥ 1).

Both quantify over *fixed* M2RNN parameterizations (W, f) including
row-/column-/cell-wise forget gates. The constituent lemmas
`row_forget_m2rnn_fails_embedded_mixed_key_delta_correction` (and column/cell
analogues) close every combination of forget-gate placement that respects
M2RNN's signature.

The mechanism-side analogue in `OnlineMemory.lean`
(`stateIndependentAdditiveWrite_not_uniformOneStepOverwrite`,
`rawOuterWrite_not_uniformOneStepOverwrite`) shows the obstruction is intrinsic
to state-independent additive writes; no parameter choice escapes.

### Could a stronger multi-step or training-time separation be formalized?

**Multi-step (deterministic-trajectory) separation — feasible, medium effort.**
Statement candidate: *for every fixed M2RNN parameterization (W, f) with K ≥ 2
input keys and a key/value stream that hits the mixed-key delta condition at
some step t ≤ N, NDM and M2RNN trajectories diverge by step t + 1; this divergence
is preserved under the M2RNN forget mechanism for N additional steps.* This
extends the one-step witness into the recurrence and would block any "small
constant factor away" argument. Effort: **L** (induction over trajectory
length, definitionally heavy).

**Training-time separation — infeasible.** Would require formalizing AdamW or
schedule-free optimization, gradient flows, and a target loss surface. No
realistic path to Lean.

**Generalization to all "matrix-state nonlinear-recurrent without delta
correction" families — feasible only as a type-level signature claim, not a
families-wide impossibility.** The `M2RNNFeatures` predicate already does this
at the signature level. A semantic families-wide claim would require enumerating
or quantifying over an *infinite* parametric family, which Lean can do but at
limited paper value.

### Recommendation

**Keep the one-step separation as the formal anchor for the paper's §7 Theorem
Set A.** It is sharp, general (K ≥ 2, V ≥ 1), covers all three external-forget
shapes, and is exactly the result the empirical S5 evidence in §6 corroborates.

**Optionally** extend with the multi-step trajectory separation (task F2 below)
if a reviewer asks for it; otherwise the paper text should be careful to say
*"in one recurrent step"* every time. The one-step result is enough to support
the §6 mechanism witness.

**Do not** attempt training-time or families-wide impossibility theorems.

---

## 4. Prioritized Follow-On Formalization Task List

Listed in suggested dispatch order. Each task identifies the candidate theorem
statement (informal English), effort, and which paper claim it unlocks. Every
named existing theorem is cross-checked against the inventory.

### F1. NDM-architecture realization of the S5 tracker  *(Effort: M)*

**Candidate theorem (informal):**
> *(NDMRealizesS5.ndm_realizes_s5_tracker)* There exist an integer `d` (e.g.
> `d = 12`), an orthonormal family of keys `{k_g : Fin d → ℝ}` indexed by
> `S5Tracker.AdjacentGenerator`, a value family `v : S5Tracker.AdjacentGenerator
> → Fin d → ℝ`, and a decay scalar `λ = 1` such that the NDM update equation
> `S_t = tanh(λ · S_{t-1} + k_{g_t} · (v_{g_t} − S_{t-1}ᵀ k_{g_t})ᵀ)` produces
> a state trajectory that, decoded through a fixed linear readout `S_t ↦ q · S_t`
> reconstructs `S5NDMRealization.s5TransitionMemory.read` on every input word.

**Bridges:** `OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others`,
`OnlineMemory.memoryTable_retrieves_orthonormal`,
`S5NDMRealization.s5TransitionMemory_run`, `S5Tracker.run_append`,
`Activation.tanh_strictMono` (for the readout invertibility step).

**Unlocks:** the cleanest paper wording for Pillar 2's "approaches the NC1
limit" — restated as "NDM realizes the canonical NC1-complete regular-language
witness". Also tightens the §6 interpretation paragraph from a parameter-count
match into an architecture-level realization argument.

**Risks:** the orthonormal-key construction must avoid `native_decide` and any
opaque numeric literal. Reuse the constructive `Fin n → ℝ` basis already used
in `e88_two_keys_induce_distinct_left_transitions`.

---

### F2. Multi-step trajectory separation (NDM ≠ raw-write M2RNN over N steps)  *(Effort: L)*

**Candidate theorem (informal):**
> *(RecurrentResourceFormalism.ndm_m2rnn_trajectory_separation)* For every
> fixed `W : Matrix d d ℝ` and external forget schedule (row, column, or
> cell-wise) on M2RNN, and for every N ≥ 2, there exists a length-N input
> stream `((k_t, v_t))_{t=1..N}` over K ≥ 2 keys such that the NDM trajectory
> `S_t` and the M2RNN trajectory `H_t` differ at some `t ≤ N`, and this
> divergence persists under both architectures' subsequent updates.

**Bridges:** the existing one-step
`ndm_m2rnn_one_step_resource_separation_embeds` is the base case; induction on
`N` carrying the inequality through `tanh_strictMono` and a per-step
"divergence persists" lemma.

**Unlocks:** lets the paper drop the qualifier "in one recurrent step" from the
mechanism-witness sentence and strengthens the §7 theorem set A.

**Risks:** the "divergence persists" lemma is the hard part — `tanh` is
strictly monotone but the M2RNN forget mechanism can in principle compress
arbitrarily; this needs a careful argument that the obstruction is not
"forgotten" by `f`.

**Dispatch posture:** optional. Only run if F1 lands cleanly and there is
reviewer time before submission. The paper does not require it.

---

### F3. Per-token FLOP-class equivalence  *(Effort: S)*

**Candidate theorem (informal):**
> *(RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv)* The per-token
> floating-point operation count for one NDM head and one M2RNN head (in the
> shared signature `ArchitectureSignature`) are both bounded by
> `c₁ · d² + c₂ · d` for explicit constants `c₁, c₂` depending only on whether
> the architecture has an external forget gate. Therefore equal-token-budget
> comparisons at matched `d, H, depth` are within a constant factor.

**Bridges:** add a `flopsPerToken : ArchitectureSignature → ℕ` field
or definition computed from existing `ArchitectureSignature` fields (see
`RecurrentResourceFormalism.lean:128`); prove the bound by `decide` or
arithmetic.

**Unlocks:** Pillar 3 ("FLOPs-per-bit convergence under CMA-ES") — provides
the only architectural anchor Lean can responsibly give. Paper text:
*"the FLOP cost per token is in the same class for all four baselines (proved
in Lean as `ndm_m2rnn_flop_class_equiv`); the empirical FLOPs-per-bit
convergence (Figure 3) is therefore not a budget artifact."*

**Risks:** the FLOP counting model must be transparent. Avoid pretending Lean
proves a *learning-rate* convergence; the theorem is purely combinatorial cost.

---

### F4. Multi-programmed-recurrent type predicate (Pillar 1 generality)  *(Effort: S)*

**Candidate theorem (informal):**
> *(RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm)* Define a
> predicate `IsMultiProgrammed (sig : ArchitectureSignature) : Prop` that
> captures the three multi-programming features (many independent heads per
> layer, per-head state tile, per-batch independence). Then both the 1.27B
> NDM signature (`e88NDM_1p27B`) and a CMA-ES-reshaped M2RNN signature
> instantiated from `m2rnnPure layers heads stateScalarsPerHead` satisfy
> `IsMultiProgrammed`.

**Bridges:** uses existing `ArchitectureSignature` from
`RecurrentResourceFormalism.lean:128` and the existing `m2rnnPure` /
`e88NDM_1p27B` definitions (no new signatures need to be invented —
parameterize `m2rnnPure` with the CMA-search head/state numbers and prove the
predicate on it directly). `m2rnnHybrid` provides a useful negative witness
(should *fail* the predicate, because it mixes in non-recurrent layers).

**Unlocks:** Pillar 1's "not specific to NDM" — gives the paper a one-line
formal anchor: *"Multi-programming is not architecture-specific; both NDM and
CMA-ES-shaped M2RNN satisfy the Lean-checked multi-programming predicate
(`multiProgrammed_admits_m2rnn_and_ndm`)."*

**Risks:** the predicate must be tight enough to exclude attention/scan
hybrids; otherwise it is vacuous. Coordinate definition with the §4
geometry-search discussion.

---

### F5. Retire or quarantine the TC0 / E88Definition sorries  *(Effort: S, but coordination-heavy)*

**Candidate action (no theorem):** Audit `TC0Qualifications.lean` and
`E88Definition.lean` (both currently outside `PaperCore`). Per the revised
framing, the paper drops the "exceeds TC0" line and the parity/binary-latching
sketches. These files exist as historical sketches per
`TRUSTED_PROOF_SURFACE.md` §"Historical Sketches".

**Recommended action:** move both files into a `formal/lean/historical/`
subdirectory (or add a header comment marking them as superseded), to prevent
future inventory passes from re-surfacing their `sorry`-bearing theorems as
candidate paper claims.

**Unlocks:** clean trust-gate output; reduces the risk that a reviewer probes
the `sorry`s and questions the paper's "no placeholders" framing.

**Coordination:** must not happen until the `lean-fix-trust-gate` task
(agent-45) finishes and confirms the PaperCore boundary. If that task surfaces
that any of these "sketch" theorems are inadvertently imported into PaperCore,
the move-to-historical action is blocked by a fix first.

---

### F6. Coordinate with `lean-fix-trust-gate` for placeholder verification  *(Effort: S)*

**Candidate action (no theorem):** When `lean-fix-trust-gate` completes,
re-read its outputs at `formal/lean/build_logs/2026-05-24_trust_gate.log` and
`formal/lean/TRUST_GATE_FINDINGS.md`. If new placeholders are surfaced inside
PaperCore's import closure, fold them into an addendum to this audit and into
the F-task list above.

**Unlocks:** turns the inventory's currently-vacuous "no placeholders" claim
into a verified claim. The paper's §7 trust-gate paragraph
(`OUTLINE.md` lines 209–213) depends on this verification.

---

## 5. Prioritization Summary

| Order | Task | Effort | Pillar unlocked | Required for submission? |
|---|---|---|---|---|
| 1 | F6 (verify trust gate) | S | All (gate veracity) | **Yes** — paper claims a non-vacuous gate. |
| 2 | F1 (NDM realizes S5) | M | P2 (NC1 wording) | **Strongly recommended** — current "approaches NC1" wording is unsupported without this. |
| 3 | F4 (multi-programming predicate) | S | P1 (generality) | Optional — paper can lean on empirical evidence for P1, but a tiny formal anchor is cheap. |
| 4 | F3 (FLOP-class equivalence) | S | P3 (architectural cost) | Optional but cheap; gives Pillar 3 something to point at. |
| 5 | F5 (quarantine historical sketches) | S | hygiene | Optional; do after F6. |
| 6 | F2 (multi-step separation) | L | P2 (stronger wording) | **No** — defer unless reviewers push back. |

## 6. Out-of-Scope Reminders

This audit does not modify `.lean` files. Per the task brief, the follow-on
tasks above are the executors. None of the candidate theorem statements above
should be interpreted as already proven; they are dispatch targets.

The audit does not address:
- Out-of-repo training/eval artifacts (see `paper/OUTLINE.md` §5.2).
- The `huggingface-release-plan` task chain.
- The Triton kernel correctness (no Lean formalization planned).
- Mathlib upgrade or Lean toolchain version (`lean-toolchain` file).

---

*End of audit.*
