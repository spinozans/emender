# X22T — Indefinitely running programs, the precision regime split, and the memory boundary

Operator-funded 2026-10-22, re-scoped the same day by an operator steer that supplied
new source material: **Merrill, Jiang, Li, Lin, Sabharwal, *Why Are Linear RNNs More
Parallelizable?*, ICML 2026 (PMLR 306:88230–88260)** — read in full this session. That
paper changes what the targets mean:

* its **precision axis is a function of input length** (Def. 10), so its entire analysis
  sits on the *growing*-precision side, while X20T/X21T/X22T-F1/F2 study a *fixed* budget;
* its **nonlinear arm reaches counter machines / `L` / `P`** (Thm. 1, Thm. 2, Cor. 2),
  which subsumes "TANH realizes finite automata" (target F1) and supplies a cited positive
  direction for target F3 under the paper-compatible semantics;
* its **DPLR arm is bounded by `PNC¹`** (Thm. 3, Thm. 4) and reaches WFA/`PNC¹`
  completeness with 4-layer stacks (Thm. 5, Thm. 6), under paper-compatible
  rational-approximation semantics — not an automatic placement of our exact-real
  fla-gdn model.

**Pass 2 (superseding operator directive): RE-PROVE, do not cite.** Because a mistake is
possible — theirs, ours, or in the mapping — the algebraic arm mapping (R1), an
exact embedding of additive-update counter machines
with an arithmetic zero-mask gadget (R2), one-layer unrestricted LRNN WFA evaluation
and a literal matrix obstruction (R3), and homogeneous-product identities with
padding/depth arithmetic (R4) are independently proved in Lean. These are not complete
re-proofs of the paper: closed-form transition assembly is **O14**, the precision
bridge is **O4**, and circuit-class membership is **O13**.

Formal receipts (`lake build` green, no `sorry`, headline declarations depend only on
`propext`, `Classical.choice`, `Quot.sound`):

| file | role |
|---|---|
| `formal/lean/ElmanProofs/Expressivity/ArmMapping.lean` | **R1** — gdn step = Def. 4 LRNN step; `D_t = I` ⇒ DeltaNet exactly; DPLR in three decay placements; Taylor-`exp` datatype; `β ∈ (0,2)` |
| `formal/lean/ElmanProofs/Expressivity/CounterMachine.lean` | **R2** — counter machines, exact closed-form zero mask, additive-update embedding, balanced-block discrimination witness, `3·(n+1)`-state bound |
| `formal/lean/ElmanProofs/Expressivity/WeightedAutomata.lean` | **R3/R4** — WFA simulation in one layer, deterministic special case, DPLR cycle obstruction, associative-scan log-depth reduction |
| `formal/lean/ElmanProofs/Expressivity/PrecisionRegime.lean` | fixed-budget collapse, impossible-global-abstraction corollary + mapping predicates |
| `formal/lean/ElmanProofs/Expressivity/OmegaStreamExpressivity.lean` | **F2** — `ω`-stream transduction, collapse, bounded-precision activation independence |
| `formal/lean/ElmanProofs/Expressivity/StreamingTanhRealization.lean` | **F1** — exact saturated-corner `TANH(n)` realization + geometric corner separation |
| `formal/lean/ElmanProofs/Expressivity/MemoryBoundary.lean` | **F3** — context-free impossibility + stack augmentation |
| `experiments/expressivity_tasks/theory_x22t/MAPPING_RECEIPT.md` | definition-by-definition mapping with the re-proof status table and the five divergences |

**No training was launched and no experiment was registered.** The X21E receipts are
cited, never re-measured.

---

## 1. The regime skeleton (narrowed after X22TR review)

**Scope.** Let `p : ℕ → ℕ` be schedule notation inspired by Def. 10. Our
`BoundedAbstraction` assumptions are global and all-length, not length-restricted
precision assumptions. The cited complexity placements in this table require
**paper-compatible rational-approximation semantics**, not exact-real transfer:

| regime | both arms | the two arms |
|---|---|---|
| **`p` fixed** (`∃ B, ∀ n, p n ≤ B`) | **finite automata, uniformly in `n`** — proved here | indistinguishable at this precision: both are `N`-state machines, and exactly the `n`-state class is realized at `d = n` by both activations |
| **`p` growing** (unbounded) | exact counting escapes every finite global abstraction; the length-restricted Def.-10 schedule bridge is **UNPROVED**, not a theorem that growing budget buys counting | **cited (Merrill et al.)**: nonlinear (ReLU-MLP) arm reaches counter machines — `L`-complete at log precision (Thm. 2), `P`-complete at poly precision (Thm. 1, Cor. 2); DPLR arm `⊆ PNC¹` (Thm. 3), `⊆ AC⁰[ENC¹]` at log precision (Thm. 4). Since `PNC¹ ⊆ L`, the arms separate **iff `PNC¹ ≠ L`** |

**Precision growth distinguishes the cited regimes, not a proved schedule bridge.**
`PrecisionRegime.precision_regime_split` packages:

1. `fixed_budget_collapses_to_finite_state` — a family of models whose channel carries
   `2 ^ p n` values with `p` bounded is, *uniformly in the length*, simulated by an
   `N`-state model for one fixed `N`. Activation-agnostic: the statement only sees
   `BoundedAbstraction`. (This is X21T's collapse in schedule form, and without X21T's
   surjectivity hypothesis — `exists_fin_model_of_abstraction'`, `collapseFin`.)
2. `countModel_needs_growing_budget` — **impossible-global-abstraction corollary**.
   Its premise supplies a global, all-length `BoundedAbstraction countModel (2 ^ p n)`
   for every `n`; already `h 0` contradicts `countModel_no_omega_finite_state`,
   regardless of whether `p` grows. No schedule satisfies it. A length-restricted
   abstraction and the bridge to Def. 10 are **UNPROVED**.
3. `log2_schedule_is_growing` — `O(log n)` precision **is** a growing schedule. This is
   the sentence that keeps our theorems and theirs from being in conflict: their log
   precision is not our fixed budget (`2^{O(log n)} = poly(n)` values).

`PrecisionRegime.arm_separation_at_growing_precision` (:22) carries the cited half as an
explicit conditional: given `DPLR ⊆ PNC¹` (their Thm. 3 under paper-compatible
rational-approximation semantics, not transferred by `gatedDelta_isDPLR` alone) and
a nonlinear-arm witness outside `PNC¹` (their Thm. 2 plus
`PNC¹ ≠ L`), the nonlinear arm is **not** contained in the DPLR arm. Nothing about
complexity classes is formalized in Lean; the cited statements enter as named hypotheses
so the conditional structure is auditable.

---

## 2. Arm mapping (receipt summary; full version in `MAPPING_RECEIPT.md`)

| our arm | paper's definition | verdict | Lean |
|---|---|---|---|
| `fla-gdn`, gated delta rule, `A_t = diag(exp g_t)(I − β k kᵀ)`, `allow_neg_eigval=True` | **Def. 5 DPLR** `A_t = D_t − k_t v_tᵀ`; DeltaNet is the `D_t = I` case | **instance (DPLR)**; differs by the nontrivial decay diagonal (RWKV-7-like), by being a single head rather than their 4-layer router+buffer stack, and by fixed (bf16/fp32) precision | `gatedDelta_isDPLR` |
| β range under `allow_neg_eigval` | paper §2.3 cites Grazzi et al.: `β ∈ (0,1)` cannot recognize parity, `(0,2)` can | `β = 2·sigmoid z ∈ (0,2)` — the parity-capable range | `negEigval_beta_range` |
| E97 `state_nonlin='tanh'`: `S_t = tanh(decay ⊙ S_{t−1} + β ⊙ outer(delta,k))` | **Def. 3 nonlinear RNN** with Def. 2 closed-form `f` | **exact-real closed-form shape** — their *upper* bounds apply only under paper-compatible rational-approximation semantics (transfer unproved), and their *lower* bounds do **not**: those (Thm. 1/2, Lemma 3) are built from **ReLU** networks, whose counters are unbounded above, while `tanh` saturates into `(−1,1)` | `tanhStreamModel_isClosedFormRNN`, `ArithClosedForm.eval_tanhForm` |
| `MemoryBoundary.stackModel` (unary stack) | counter machine (Fischer et al. / Weiss et al. / their Lemma 3) | **is a one-counter machine**: the stack collapses to its depth | `stackModel_run_eq_counterModel` |
| all of the above at a fixed budget | — | **finite automata** (the fixed-budget band) | `fixed_budget_collapses_to_finite_state` |

Figure-1 placement **under paper-compatible rational-approximation semantics only**:
`fla-gdn` → **DPLR LRNN upper-bound box (`PNC¹`)**, with no WFA-completeness claim
for a single head; E97-`tanh` → **Nonlinear RNN upper-bound side only**. No exact-real
transfer theorem is proved. E97-`relu`/`softplus` are **CANDIDATE HOSTS**, requiring a
realization proof: one configured elementwise activation is not a ReLU MLP transition
(`ndm/models/e88_fla_hybrid.py:1356–1374`), and the paper gives no softplus construction.
`stackModel` → **counter machine**; at a fixed budget, the abstraction hypothesis gives
**finite automata for both arms**.

**How the X21E receipts read now.** Trained champions retain only start-anchored shallow
prefixes at extended horizons and learn the cumulative 3721-state automaton
(`(x,y) ↦ (y, x+y²+c) mod 61`) only 2–6 steps deep. These are empirical observations,
not predictions of the fixed-budget theorems.
Those theorems provide neither a 3721-state capacity obstruction for these models nor
a learnability bound, and establish no necessary cause for shallow-prefix learning.
No training was launched.

---

## 3. Verdict per target (re-scoped)

### F1 — `TANH(n)` realizes every `n`-state automaton: **LANDED, and now understood as the fixed-budget half of a bigger picture**

*Proved (kept):* `moore_realized_by_tanh_corner` — every `n`-state automaton is realized
*exactly* by a `TANH(n)` model (`s ↦ tanh(A(x)·s)`, one shared readout, horizon-free),
via the **saturated-corner** construction: code `enc j = τ·e_j` with `τ = tanh 1`, matrix
`cornerMatrix = if step j x = i then τ⁻¹ else 0`; `cornerMatrix_mul_enc` shows the
pre-activation is *exactly* the `0/1` successor indicator (so no precision hypothesis is
needed for exactness), `tanh_cornerMatrix_mul_enc` that `tanh` lands exactly on the
successor corner. `finite_program_class_equality`: both activations reach every `n`-state
behaviour at `d = n`; `enc_robust_decode` proves geometric uniqueness: two corners
cannot both be within `τ/2 ≈ 0.381` coordinatewise of one state. It does **not**
implement a robust decoder. `tanhCornerModel.out` requires exact corner equality and
otherwise returns the initial-state output; small perturbations of a noninitial corner
need not decode correctly. No finite-precision dynamical guarantee is claimed.

*Re-scoped:* at growing precision this is **subsumed** by Merrill et al. — a nonlinear
RNN with `O(log n)` bits simulates a *counter machine* (Thm. 2, Lemma 3), and counter
machines subsume finite automata. F1 is therefore **not** offered as evidence of TANH's
power; it is (i) the exact dimension-`n` construction in our framework, (ii) the
fixed-budget statement that pairs with their growing-precision one, and (iii) an
ingredient of F2's collapse sandwich. **And it does not close the arm separation**: both
arms realize every finite automaton at `d = n` (`finite_program_class_equality`).

*Ambient form impossible:* `no_ambient_abstraction_of_injective_readout` — a
distinguishing readout forces an injective quotient, so no model with a nontrivial
readout has a finite *ambient* abstraction at any budget. "Class equality at bounded
precision" is only meaningful as reachable-orbit realization or conditionally on the
abstraction hypothesis, which is exactly how everything here is stated.

### F2 — the runs-forever object: **LANDED — and this is the primary novel contribution**

Merrill et al. do not touch `ω`-streams or streaming transduction; they treat language
recognition with a final-token readout (Def. 9). This development contributes the
streaming formalization and its collapse.

*Formalism.* An `ω`-stream is `x : ℕ → α`; `StreamModel.omegaRun x t =
M.out (M.omegaState x (t+1))` is a **total deterministic sequential (Moore) transducer
over `ω`-streams**: one state update per token (`omegaState_succ`), one output per token
forever, no halting state, **no acceptance condition**, causal. `omegaRun_acceptance_invariant`
proves any predicate on the output stream is carried along by stream equality, so
Büchi/Muller/parity variants add nothing to these theorems.

*Results.* `omegaRun_sim` (the `ω` engine); `omegaRun_collapse` /
`exists_fin_omega_model_of_abstraction` (**abstraction collapse over `ω`-streams**);
`omega_class_activation_independent_at_bounded_precision` (**at a fixed budget the
`ω`-program class is activation-independent**: any bounded-precision channel of either
arm has finite-state `ω`-behaviour, and every `n`-state `ω`-behaviour is realized exactly
at `d = n` by the `LIN(n)` one-hot and the `TANH(n)` corner constructions);
`omegaRun_needs_unbounded_state`, `countModel_omegaRun` (`1,2,3,…` forever),
`countModel_no_omega_finite_state`.

*What bends.* The upper bound is conditional on the abstraction hypothesis; the sandwich
is reachable-realization equality; and with **unbounded** precision the `ω` classes are
unclassified (exact linear counting provably exceeds finite state; no matching exact-`TANH`
statement — O8).

### F3 — the memory boundary: **LANDED** (impossibility proved; positive direction now **re-proved** in R2, not derived)

**(a) Native impossibility (proved, consolidated).**
`context_free_discrimination_impossible`: for *any* `StreamModel` of either activation at
any dimension, `SeparatesBalanced M yes no` with `yes ≠ no` implies `NoBoundedAbstraction M`.
`tanh_family_context_free_impossible` is the scalar `TANH` instance. So no bounded-precision
channel computes the `aᵏbᵏ` discrimination, at any word length.

**(b) Positive direction (re-proved in R2).** Our stack augmentation is a **one-counter
machine** (`stackModel_run_eq_counterModel`), and `CounterMachine.lean` now re-proves the
simulation itself: `CounterMachineSim.cmModel_fold` shows a real-state recurrent model
exactly embeds the **defined additive-update** machine after every word. `cmRealStep`
retains discrete control, Boolean equality decisions, and arbitrary lookup functions;
only `zeroMaskForm` has an arithmetic-expression correctness theorem, with exact
`max 0 (1 − |r|)` on integers. No assembled closed-form transition is proved, even for
the one-counter witness; reset updates from the paper's Lemma 3 are not modelled.
`cmModel_separates_balanced` proves **balanced-block discrimination** for the
`StreamModel` construction, not closed-form-arithmetic completeness or exact language
recognition (the witness also accepts `abab`). `stackModel_cells_unary`
shows the memory is one cell symbol at unbounded depth and
`stackModel_no_finite_abstraction` that it lies outside every bounded-precision class.
`cmModel_reachable_card` supplies the log-precision content as a **cardinality** bound
(`3·(n+1)` states at length `n`).

**Scope.** A construction in the formal framework, **not** a claim about
`HybridLadderLM` (no native stack). The architecture-level addition that would host it is
a soft push/pop head (stack-RNN / NTM style); `stackModel` is its discrete idealization,
and the soft version is open (O7/O9).

---

## 4. The operator's claim: what survives, what bends — updated

Operator (2026-10-22): *"we want to prove we can have indefinitely running programs in the
Emender series… the whole class that's expressible as an indefinitely running program is
different."*

**A conditional separation for the paper's growing-precision classes, not an
exact-real transfer or a realized separation between configured Emender heads.**

> At a *fixed* budget both arms are finite automata (our theorems: X21T collapse,
> `fixed_budget_collapses_to_finite_state`), so no arm separation exists there. At a
> *growing* budget the nonlinear arm reaches counter machines and hence `L`-complete /
> `P`-complete problems (their Thm. 1, Thm. 2, Cor. 2) while every LRNN — our DPLR fla-gdn
> arm's algebraic shape included — is confined to `PNC¹` **under paper-compatible
> rational-approximation semantics** (their Thm. 3; `PNC¹ ⊆ L`). No exact-real transfer
> theorem is proved. So
> **nonlinear ⊄ DPLR, conditional on `PNC¹ ≠ L`**
> (`arm_separation_at_growing_precision`), and the pivot variable is precisely the
> length-dependence of the state channel's precision.

What bends, stated plainly:

1. It is **not** a claim about our `tanh` arm. Their lower bounds are ReLU-MLP
   constructions; a saturating `tanh` state is confined to `(−1,1)`, and whether it can
   reach counter machines at growing precision is **open** (O10). The addressable Emender
   configurations `state_activation ∈ {relu, softplus}` are only **CANDIDATE HOSTS**:
   their single elementwise activation is not the paper's ReLU MLP, and no softplus
   construction is given. A realization proof is required.
2. It is **not** an unbounded-precision claim either. Exact real arithmetic is outside all
   of these classes (their Proposition 1 caps at poly precision); X20T's exact-`LIN(1)`
   universality lives outside the picture and stays open (O6).
3. It is **conditional** on `PNC¹ ≠ L` (`NC ≠ P` for the poly-precision version) — a
   long-standing open problem, exactly as their Corollaries 4/6 state.
4. It says nothing about *learnability*. X21E's shallow-prefix finding is empirical,
   with no proved causal link to fixed-budget collapse (see §2); whether the extra
   growing-precision power is learnable is not addressed by either development.

**And the honest re-labelling of X22T's own earlier results:** F1's "TANH realizes
automata" is an *equality*, not a separation, and at growing precision it is subsumed.
F2 remains the novel formal contribution (they have no `ω`-stream theory). F3(a)'s
impossibility is unconditional and independent of their framework; F3(b)'s positive
direction is theirs.

---

## 5. The objects

Inherited from X21T: `StreamModel σ α β` (init, token-indexed step, one shared readout),
`run`, `stateAfter`, `runFrom_sim`, `BoundedAbstraction M n`, `collapse`, `onehot`,
`finLinModel`, `finModel`, `countModel`, `tanhStreamModel`, `SeparatesBalanced`.

X22T adds: `tau`/`enc`/`cornerMatrix`/`tanhCornerModel` (F1); `omegaPrefix`, `omegaState`,
`omegaRun`, `NoBoundedAbstraction` (F2); `StackPhase`, `stackPush/Pop/Step`, `stackModel`
(F3); `PrecisionSchedule`, `IsFixedBudget`, `IsGrowingBudget`, `collapseFin`,
`ArithClosedForm`, `ScalarStepClosedForm`, `IsDPLR`, `counterStep`, `counterModel`
(PrecisionRegime).

---

## 6. What is proven (all Lean, all sorry-free)

**`StreamingTanhRealization.lean` (F1)**
1. `tau_eq_tanh_one`, `tau_pos`, `tau_ne_zero`, `tau_abs_lt_one`.
2. `enc_self`, `enc_of_ne`, `enc_injective`, `enc_entries`, `enc_abs_lt_one`.
3. `enc_diff_at`, `enc_robust_decode` — geometric corner separation, not an
   implemented robust decoder or a finite-precision dynamical guarantee.
4. `cornerMatrix_mul_enc_apply`, `cornerMatrix_mul_enc` — the pre-activation is exactly
   the `0/1` successor indicator.
5. `tanh_cornerMatrix_mul_enc` — `tanh` lands exactly on the successor corner.
6. `tanhCornerModel_step_enc`, `_fold`, `_stateAfter`, `_reachable_enc`.
7. `tanhCornerModel_reachable_bounded` — reachable states inside `(−1,1)ⁿ`.
8. `moore_realized_by_tanh_corner` — exact `TANH(n)` realization of every `n`-state
   automaton.
9. `finite_program_class_equality` — both classes at `d = n`.
10. `no_ambient_abstraction_of_injective_readout` — the ambient reading is impossible.

**`OmegaStreamExpressivity.lean` (F2)**
11. `omegaPrefix_zero/succ/const`, `omegaState_zero/succ`, `omegaRun`.
12. `omegaRun_sim` — the `ω` engine.
13. `omegaRun_acceptance_invariant`.
14. `omegaRun_collapse`, `exists_fin_omega_model_of_abstraction`.
15. `omega_class_activation_independent_at_bounded_precision`.
16. `NoBoundedAbstraction`, `omegaRun_needs_unbounded_state`, `countModel_omegaRun`,
    `countModel_no_omega_finite_state`, `exists_omega_model_outside_finite_state`.

**`MemoryBoundary.lean` (F3)**
17. `context_free_discrimination_impossible`, `tanh_family_context_free_impossible`.
18. `fold_push`, `fold_push_init`, `fold_pop_underflow`, `pop_chain`, `fold_blocks`.
19. `stackModel_separates_balanced`, `stackModel_no_finite_abstraction`,
    `stackModel_cells_unary`.
20. `memory_boundary_dichotomy`.

**`PrecisionRegime.lean` (regime theorem + mapping)**
21. `PrecisionSchedule`, `IsFixedBudget`, `IsGrowingBudget`, `log2_schedule_is_growing`.
22. `boundedAbstraction_mono`, `collapseFin`, `collapseFin_step`, `collapseFin_out`,
    `exists_fin_model_of_abstraction'` (collapse **without** surjectivity).
23. `fixed_budget_collapses_to_finite_state` — uniform-in-`n` finite-state bound.
24. `countModel_needs_growing_budget` — impossible-global-abstraction corollary; premise
    contradicted by `h 0` for every schedule, not a Def.-10 schedule lower bound.
25. `precision_regime_split` — (23) ∧ (24) ∧ `log₂` growing.
26. `ArithClosedForm` (+ `eval`, `tanhForm`, `eval_tanhForm`), `ScalarStepClosedForm`,
    `tanhStep_scalarClosedForm`, `tanhStreamModel_isClosedFormRNN` — **Def. 2/3 mapping**.
27. `IsDPLR`, `gatedDelta_isDPLR`, `negEigval_beta_range` — **Def. 5 mapping** (`β ∈ (0,2)`).
28. `counterStep`, `counterModel`, `counterStep_depth`, `counterModel_out_depth`,
    `stackModel_run_eq_counterModel` — **counter-machine mapping**.
29. `arm_separation_at_growing_precision`, `arm_separation_chain` — the cited conditional.

**`ArmMapping.lean` (R1)**
30. `matMul_diagM_left`, `sum_diag_minus` (the matrix algebra the identities rest on).
31. `lrnnStep`, `lrnnStepVec` — their Def. 4 in both arities.
32. `gdnStep`, `gdnMatrix`, `deltaNetMatrix`; `gdnMatrix_apply`.
33. `gdnStep_eq_lrnnStep` — **the gdn recurrence is a Def. 4 LRNN step**.
34. `gdnMatrix_eq_decayed_deltaNet`, `gdnMatrix_zero_decay`,
    `gdnStep_zero_decay_eq_deltaNet` — **DeltaNet recovered exactly at zero decay**.
35. `deltaNetMatrix_isDPLR`, `gdnMatrix_isDPLR`, `gdnMatrixPerChannel_isDPLR`,
    `diagM_isDPLR` — Def. 5 in three placements.
36. `expTaylorForm`, `eval_expTaylorForm`, `expTaylor_two_ne_exp` — their Taylor datatype,
    and the proof that it is not `Real.exp`.

**`WeightedAutomata.lean` (R3/R4)**
37. `matMul_assoc`, `matMul_oneM_left/right`.
38. `WFA`, `rowStep`, `wfaState`, `wfaScore`, `wfaModel`, `wfaModel_readout` — **one layer
    computes every WFA score at every position**.
39. `detTrans`, `detWFA`, `detTrans_rowStep_onehot`, `detWFA_state_eq_onehot` — **the
    deterministic special case is X21T's one-hot model**.
40. `dplr_cycle_identity`, `cycle3`, `cycle3_not_dplr` — **literal matrix obstruction
    in the specified coordinates only**, not behavioral strictness.
41. `prodL`, `prodL_append`, `prodL_splitAt`, `prodL_singleton`, `foldl_matMul_eq_rev`,
    `lrnn_hom_state` — the LRNN state is an iterated product.
42. `halvingDepth`, `halvingDepth_pow_two`, `le_two_pow_log2_succ`, `prodL_pad`,
    `prodL_balanced_eval`, `log2_mono` — **balanced evaluation of depth `log₂n+1`**.

**`CounterMachine.lean` (R2)**
43. `CounterMachine`, `CMState`, `zeroMasks`, `cmStep`, `cmRun`.
44. `zeroMaskR`, `zeroMaskForm`, `eval_zeroMaskForm`, `zeroMaskR_int`, `isZeroR`,
    `isZeroR_int`, `isZeroR_natCast` — **the zero test is an exact closed form**.
45. `cmEmbed`, `cmRealStep`, `cmModel`, `cmRealStep_embed`, `fold_embed`, `cmModel_fold` —
    **exact embedding of the defined additive-update machine**, not closed-form
    transition assembly (discrete control, Boolean decisions, arbitrary lookups remain).
46. `balancedCM`, `phaseIdx`, `toReal`, `cmRealStep_balanced_toReal`, `cmStep_balanced_toCounter`,
    `cmRun_balanced_aux`, `fold_counterStep_eq_stack`, `fold_counterStep_init`, `cmRun_blocks`,
    `outR`, `outR_cmEmbed`, `cmModel_separates_balanced` — balanced-block discrimination
    (also accepts `abab`), not exact `aᵏbᵏ` recognition.
47. `counterStep_snd_le`, `counterFold_snd_le`, `cmModel_reachable_card` — the `O(log n)`
    cardinality content.

**Axiom audit** (`#print axioms`, 31 headline declarations across the seven files): 26 use at
most `propext`, `Classical.choice`, `Quot.sound`, two (`omegaRun_sim`,
`stackModel_run_eq_counterModel`) use only `propext` + `Quot.sound`, one
(`stackModel_cells_unary`) only `propext`, and `arm_separation_at_growing_precision`
depends on **no** axioms. No `sorryAx` anywhere. This covers every R1–R4 declaration plus
the F1/F2/F3 and regime-theorem headlines.

**Independent numeric sanity checks** (float, not proofs): corner identity
`tanh(A·enc j) = enc (step j x)` to `0.0` max error for `n = 3`; stack automaton satisfies
`SeparatesBalanced` for all `k, k' ∈ [1,11]` with zero violations.

---

## 7. Corrections (cumulative; X21T/X21TR items remain in force)

1. **The precision axis was mis-stated in the X22T frame.** The original task said "under
   bounded precision" as if that were Merrill et al.'s regime. It is not: their Def. 10
   precision is `p(n)` bits at length `n`, and log precision means `poly(n)` values. Our
   `BoundedAbstraction` is the `O(1)`-bit case, which **they do not analyse**. Every X22T
   theorem is therefore a statement about a *different* regime, and the regime theorem
   (`precision_regime_split`) packages fixed-budget collapse and a vacuous counting
   implication; the length-restricted Def.-10 bridge remains unproved.
2. **F1's content is subsumed at growing precision.** Counter machines subsume finite
   automata (their Thm. 2 path), so "TANH realizes automata" is not evidence of nonlinear
   power. F1 is kept as the exact dimension-`n` construction and as the fixed-budget half
   of the equality; it is **not** a separator.
3. **The F1 construction needed no precision hypothesis, and the "large-magnitude
   saturation" route was unnecessary.** The pre-activations are *solved for* and are
   exactly `0`/`1` in exact real arithmetic. The geometric margin `τ/2`
   (`enc_robust_decode`) supplies neither an implemented robust readout nor a
   finite-precision dynamical guarantee.
4. **The ambient form of "class equality at bounded precision" is impossible**
   (`no_ambient_abstraction_of_injective_readout`), generalizing X21T's `rot3Model.out = id`
   remark to every model with a distinguishing readout.
5. **`BoundedAbstraction` is data, not a proposition** — hence the `NoBoundedAbstraction`
   wrapper used for every impossibility statement. Also: the uniform fixed-budget theorem
   needs the collapse **without** a surjectivity hypothesis (`collapseFin`), because a
   budget bounds *available* values, not *used* ones.
6. **The `ω`-acceptance question dissolves** (`omegaRun_acceptance_invariant`): a total
   transducer with a shared readout needs no acceptance condition, and any stream predicate
   is carried along. This is a statement about these theorems, not a classification of
   `ω`-regular languages.
7. **Arm-mapping caveats that must not be lost** (see `MAPPING_RECEIPT.md`): our fla-gdn is
   DPLR but with a nontrivial decay diagonal (RWKV-7-like, not DeltaNet's `D_t = I`), and it
   is a single head, not their 4-layer router+buffer stack, so their *completeness* results
   (Thm. 5/6) are not claimed for it. Our E97 `tanh` arm is Def.-3 closed form, so their
   *upper* bounds apply only under paper-compatible rational-approximation semantics
   (exact-real transfer unproved); their *lower* bounds are ReLU MLP constructions
   and do not transfer to the configured elementwise activations.
8. **`SeparatesBalanced` is a finite-word statement already quantifying over all lengths**;
   the `ω` reading follows from `omegaState` being the finite-word prefix fold, and no
   separate `ω` impossibility theorem is formalized (the docstring says exactly this).
9. **`PNC¹ ⊆ L ⊆ P` and `PNC¹ ≠ L`/`NC ≠ P` are not formalized.** They enter
   `arm_separation_*` as explicit hypotheses so the conditional is auditable; the
   separation claim is only as strong as those citations.
10. **Pass-2 scope corrected by X22TR.** `cycle3_not_dplr` excludes a literal
    `3×3` transition matrix in its specified coordinates only. It proves neither
    behavioral strictness nor four-layer necessity (see correction 17).
    `zeroMaskForm` is exact on integer-valued reals; the complete closed-form
    counter-machine transition is not assembled (O14).
11. **Pass-2 correction: what we could not reproduce.** `PNC¹`-membership (their Thm. 3),
    their Thm. 1's space-complexity wrapper, and their 4-layer router/buffer construction.
    We have the associative-scan reduction and the obstruction, not the classes. **O13.**
12. **Pass-2 correction: "log precision" is re-proved as cardinality, not bits.** Their
    `p_h(n)` bit accounting is not modelled; `cmModel_reachable_card` gives `3·(n+1)`
    reachable states for the witness machine. Any statement that needs *bit-lengths* (e.g.
    accumulation error) is outside what we re-proved.
13. **X21E observations, not a theorem consequence.** No fixed-budget theorem
    predicts the shallow-prefix / 2–6-step depth finding, supplies a 3721-state
    capacity obstruction here, or proves a learnability bound. No training was launched.
14. **X22TR condition 1 — impossible global premise.** For any schedule, `h 0` in
    `countModel_needs_growing_budget` contradicts `countModel_no_omega_finite_state`.
    The theorem is demoted to an impossible-global-abstraction corollary; the
    length-restricted Def.-10 schedule bridge is **UNPROVED**. No claim that a growing
    budget buys counting follows from the regime skeleton.
15. **X22TR condition 2 — embedding, not closed-form completeness.** `cmRealStep`
    retains discrete control, Boolean equality decisions, and arbitrary lookups.
    Only `zeroMaskForm` has an arithmetic-expression correctness theorem. O14 includes
    the one-counter witness, general lookup assembly, and the paper's reset updates;
    the counter-machine class claim concerns the defined additive-update `StreamModel`.
16. **X22TR condition 3 — semantics-qualified placements.** Every cited upper-bound
    placement requires paper-compatible rational-approximation semantics; exact-real
    transfer is unproved. `expTaylor_two_ne_exp` proves a discrepancy only, not
    computational-class strictness or a transfer theorem.
17. **X22TR condition 4 — literal coordinates only; reviewer's rot3-is-DPLR
    counterexample.** `B = [[0,-1],[1,-1]] = diag(-1,0) - (1,1)ᵀ(-1,1)` **is DPLR**
    and cycles `(1,0) → (0,1) → (-1,-1) → (1,0)`, realizing the X21T three-cycle.
    Thus `cycle3_not_dplr` is only a literal matrix obstruction, not behavioral
    strictness. The paper's four-layer construction is **sufficient, not proved
    necessary**; neither minimal depth nor necessity of its router/buffer is proved.
18. **X22TR condition 5 — candidate hosts only.** E97 ReLU/softplus configurations
    require a realization proof: `e88_fla_hybrid.py:1356–1374` applies one configured
    elementwise activation, not a ReLU MLP transition; the paper gives no softplus
    construction. Non-saturation does not establish its lower bounds.
19. **X22TR condition 6 — balanced-block discrimination.** `balancedCM` also accepts
    `abab`; `SeparatesBalanced` is the correct narrow statement, not exact recognition
    of `aᵏbᵏ`. Lean comments and the witness descriptions are corrected accordingly.
20. **X22TR condition 7 — separation, not robust implementation.**
    `enc_robust_decode` proves geometric corner uniqueness; `tanhCornerModel.out`
    requires exact corner equality and otherwise emits the initial-state output.
    No implemented robust decoder or finite-precision dynamical guarantee is proved.
21. **X22TR condition 8 — no training prediction.** Removed the predictive/necessary
    causal claim about shallow-prefix learning; the fixed-budget theorems give neither
    a 3721-state capacity obstruction for these models nor a learnability bound.

---

## 8. Literature anchors

| # | Anchor | Status |
|---|---|---|
| 0 | Merrill, Jiang, Li, Lin, Sabharwal, *Why Are Linear RNNs More Parallelizable?*, ICML 2026, PMLR 306:88230–88260 | **PRIMARY — full text read this session**, definitions transcribed in `MAPPING_RECEIPT.md`; the paper's theorems are cited, never re-proved |
| 0b | Grazzi, Siems, Franke, Zela, Hutter, Pontil, *Unlocking state-tracking in linear RNNs through negative eigenvalues* (ICLR 2025, arXiv:2411.12537) | **cited-via-anchor-0** (paper §2.3) and via `fla`'s own docstring (`fla/layers/gated_deltanet.py:73–77`); original not read this session. Used only for the `β ∈ (0,2)` parity claim, which we do **not** re-prove |
| 1 | Merrill, *Sequential Neural Networks as Automata* (arXiv:1906.01615) | verified in X21T (not re-verified) |
| 2 | Merrill et al., *A Formal Hierarchy of RNN Architectures* (arXiv:2004.08500) | verified in X21T |
| 3 | Weiss, Goldberg, Yahav, *On the Practical Computational Power of Finite Precision RNNs* (arXiv:1805.04908) | verified in X21T |
| 4 | Fischer, Meyer, Rosenberg, *Counter machines and counter languages* (1968); Weiss et al.; Merrill 2021 | **cited-via-anchor-0** (their Lemma 2/3 use) |
| 5 | Caussinus et al. 1998 (iterated `3×3` matrix multiplication is GapNC¹-complete); Chiang 2025 (closed forms are `FO`-uniform `TC⁰`) | **cited-via-anchor-0** |
| 6 | Schützenberger 1961 / weighted automata | X21T: verified at definition/attribution level; WFA is now anchor 0's Def. 13 |
| 7 | Delétang et al., *Neural Networks and the Chomsky Hierarchy* (arXiv:2207.02098) | X21T: abstract-level, tested-task subset only |
| 8 | Grefenstette et al.; Joulin & Mikolov; Graves et al. (differentiable stack/tape memory) | **secondary, NOT verified** — architecture vocabulary for F3(b) only |

---

## 9. Open list (honest)

* **O1 — activation separation at equal `(d, B)`, fixed budget.** Open. F1 shows both arms
  reach every `n`-state automaton at `d = n`, so no fixed-budget separator can come from
  the automaton class.
* **O2 — formerly "does `S-TANH(d,B)` reach all `n`-state automata?"** **Closed at `d = n`**
  in the reachable-realization sense (`moore_realized_by_tanh_corner`); the ambient form is
  impossible; minimal-dimension questions remain O3.
* **O3 — minimal dimension.** `d = 1` cannot literally 3-cycle; `d = 2` can; `d = n`
  suffices for any `n`-state automaton (both activations). No lower bound `d ≥ f(n)`.
* **O4 — precision/range modeling.** `BoundedAbstraction` is an information budget with no
  rounding model; anchor 0's datatype is a rational semiring with poly/log-precision `exp`
  and `√`. Bridging the two precision formalisms is not attempted.
* **O5 — learnability, efficiency, average case.** Untouched; no training launched.
* **O6 — unbounded precision with a shared decoder over all lengths and online readout.**
  Still open, exactly as in X20T/X21T. Note this regime is outside anchor 0's classes
  (their Proposition 1 caps precision at poly).
* **O7 — the soft/differentiable stack.** No theorem relates `stackModel` to a real-valued
  soft-pointer stack, nor bounds gradients through push/pop.
* **O8 — `ω`-classes at unbounded precision.** Exact linear counting exceeds finite state
  over `ω`-streams; whether a `TANH` `ω`-transducer can do the same is open.
* **O9 — architecture-level hosting.** A soft stack/tape head would host F3(b); the current
  `HybridLadderLM` has none. Priority and cost are operator decisions.
* **O10 — the saturating nonlinear arm at growing precision.** Merrill et al.'s lower bounds
  (Thm. 1, Thm. 2, Lemma 3) are ReLU constructions. Whether a `tanh`-state RNN at log
  precision reaches counter machines — and whether our E97-`tanh` configuration therefore
  sits in the `L` box with a *lower* bound — is **open**. The `relu`/`softplus`
  configurations are **CANDIDATE HOSTS**, not configurations covered by their
  ReLU-MLP lower bounds; both require a realization proof, and no softplus construction
  is supplied by the paper.
* **O11 — how much of `PNC¹` our fla-gdn head actually reaches.** Their Thm. 5/6 provide a sufficient 4-layer
  construction with a router and a token buffer (not a minimal-depth theorem); what a single Emender head with `d ≤ 64` and
  bf16 reaches inside `PNC¹` is unaddressed.
* **O12 — `PNC¹ ≠ L`.** The separation is conditional on a longstanding open problem, as in
  their own Corollaries 4 and 6.
* **O13 — `PNC¹` membership is not re-proved.** We formalize no circuit, no uniformity, no
  space complexity. `prodL_balanced_eval` gives the log-depth *reduction* (padding to a
  power of two, `halvingDepth_pow_two`, `log₂n+1` bound) but the step from that to "the
  language is in `PNC¹`" is the paper's and remains unverified by us. This is the largest
  single gap left by pass 2.
* **O14 — complete closed-form transition assembly, including one counter.**
  `cmRealStep` retains discrete control, Boolean equality decisions, and arbitrary
  lookup functions. Only `zeroMaskForm` has an arithmetic-expression correctness
  theorem; no assembled Def.-2 transition is proved even for the one-counter witness.
  `cmModel_fold` exactly embeds the defined additive-update machine, not the paper's
  complete Lemma-3 closed-form simulation. Reset updates allowed there are not modelled.
  Realizing these controls/lookups/updates arithmetically remains unproved.
* **O15 — the 4-layer router/buffer construction.** Their Thm. 5/6 positive direction for
  DPLR stacks is not reproduced; we have the one-layer LRNN simulation and a literal coordinate-matrix obstruction
  that proves no stack-depth necessity. Whether *our* fla-gdn head reaches `PNC¹`-completeness at 4
  layers with `d ≤ 64` is unaddressed (O11).

---

## 10. Reproduce

```bash
cd formal/lean
lake build ElmanProofs.Expressivity.StreamingTanhRealization   # F1
lake build ElmanProofs.Expressivity.OmegaStreamExpressivity    # F2
lake build ElmanProofs.Expressivity.MemoryBoundary             # F3
lake build ElmanProofs.Expressivity.ArmMapping                 # R1
lake build ElmanProofs.Expressivity.CounterMachine             # R2
lake build ElmanProofs.Expressivity.WeightedAutomata           # R3/R4
lake build ElmanProofs.Expressivity.PrecisionRegime            # regime theorem + mapping
lake build                                                     # full build green
# headline axiom check (standard axioms only, no sorry):
cat > /tmp/x22t_ax.lean <<'EOF'
import ElmanProofs.Expressivity.PrecisionRegime
open PrecisionRegime
open StreamingExpressivity
#print axioms precision_regime_split
#print axioms fixed_budget_collapses_to_finite_state
#print axioms countModel_needs_growing_budget
#print axioms tanhStreamModel_isClosedFormRNN
#print axioms gatedDelta_isDPLR
#print axioms negEigval_beta_range
#print axioms stackModel_run_eq_counterModel
#print axioms arm_separation_at_growing_precision
#print axioms moore_realized_by_tanh_corner
#print axioms omega_class_activation_independent_at_bounded_precision
#print axioms memory_boundary_dichotomy
EOF
lake env lean /tmp/x22t_ax.lean
```
