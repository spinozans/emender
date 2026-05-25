/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.S5Tracker
import Mathlib.GroupTheory.Perm.Sign
import Mathlib.Tactic.FinCases

/-!
# S5 Capacity Inseparability (v18-lean-autopoetic)

This module proves the **capacity lower bound** for finite-state recognizers
that realize the S5 adjacent-transposition tracker. It is the existential
form of the S5-coset inseparability target documented in
`formal/lean/LEAN_FRONTIER.md`.

## Headline theorem

`emender_m2rnn_s5_inseparability_existential` says: no finite recognizer
`M : FixedPrecisionOnlineRecognizer AdjacentGenerator` with strictly fewer
than 120 states admits a state-decoder reproducing `S5Tracker.run` on every
input word. Symmetrically, NDM at `d = 12` realizes the S5 tracker
(`NDMRealizesS5.emender_realizes_s5_tracker`), so the 120-state ceiling
is met.

## What this is, and what it is not

The argument is a clean state-counting / pigeonhole result. It works for
*any* finite-state recognizer, regardless of architectural details, by
showing that:

* `S5Tracker.run : List AdjacentGenerator → S5` is **surjective**. This uses
  Mathlib's `Equiv.Perm.mclosure_swap_castSucc_succ` (the four adjacent
  transpositions generate `Equiv.Perm (Fin 5)` as a submonoid) and the
  standard induction principle for submonoid closure.
* Composing a state-decoder with the recognizer's run inverts `run` on a
  section, giving an injection `S5 → M.State`, which forces
  `Fintype.card M.State ≥ Fintype.card S5 = 120`.

This is **not** an explicit `T(d)` polynomial bound on the raw-write
class `FixedRightRawExternalForget2` of `RecurrentResourceFormalism.lean`.
That raw-write class is parameterised by continuous-valued matrices and
vectors, so its underlying "state space" is `Matrix (Fin K) (Fin V) Real`,
which is uncountable; the finite-state recognizer formulation here does not
directly apply. To bridge to the raw-write class, one would need either:

1. A bounded-precision discretisation (Liu et al.-style fixed-point weights
   and activations), or
2. A direct trajectory-distinguishability counting argument on the raw-write
   update rule.

Both directions are documented in `LEAN_FRONTIER.md` as the remaining
infrastructure work. The capacity bound proved here closes the **existential**
form of the inseparability claim and provides the load-bearing
state-counting lemma against which any future bounded-precision raw-write
class would have to land.

## Surjectivity of the tracker

The bridge from the four adjacent-generator tokens (`s01`, `s12`, `s23`,
`s34`) to Mathlib's `swap i.castSucc i.succ` is direct: each
`transposition g` is exactly one of the four adjacent transpositions in
`Equiv.Perm (Fin 5)`. The submonoid closure of these four swaps is `⊤` by
`Equiv.Perm.mclosure_swap_castSucc_succ 4`, hence every S5 element has a
word in adjacent generators.
-/

namespace S5Inseparability

open S5Witness S5Tracker

/-! ## Adjacent generators as Mathlib `swap` -/

/-- For each `i : Fin 4`, the swap `(i.castSucc, i.succ)` in `Equiv.Perm (Fin 5)`
is exactly the transposition of one of the four adjacent generators. -/
theorem exists_adjacent_generator_for_swap (i : Fin 4) :
    ∃ g : AdjacentGenerator,
      transposition g = Equiv.swap (i.castSucc : Fin 5) (i.succ : Fin 5) := by
  fin_cases i
  · exact ⟨.s01, by rfl⟩
  · exact ⟨.s12, by rfl⟩
  · exact ⟨.s23, by rfl⟩
  · exact ⟨.s34, by rfl⟩

/-- Running a single-token word produces the corresponding adjacent
transposition. -/
@[simp] theorem run_singleton (g : AdjacentGenerator) :
    run [g] = transposition g := by
  simp [run, runFrom, step]

/-! ## Surjectivity of `S5Tracker.run` -/

/-- **`S5Tracker.run` is surjective onto S5.** Every element of `Equiv.Perm (Fin 5)`
is realised by some word over the four adjacent generators. -/
theorem run_surjective :
    Function.Surjective (run : List AdjacentGenerator → S5) := by
  intro g
  refine Submonoid.dense_induction
    (motive := fun (x : S5) => ∃ w : List AdjacentGenerator, run w = x)
    (s := Set.range fun i : Fin 4 =>
      Equiv.swap (i.castSucc : Fin 5) (i.succ : Fin 5))
    (closure := Equiv.Perm.mclosure_swap_castSucc_succ 4)
    ?mem ?one ?mul g
  case mem =>
    rintro x ⟨i, rfl⟩
    obtain ⟨gen, hgen⟩ := exists_adjacent_generator_for_swap i
    exact ⟨[gen], by rw [run_singleton, hgen]⟩
  case one =>
    exact ⟨[], rfl⟩
  case mul =>
    rintro x y ⟨wx, hx⟩ ⟨wy, hy⟩
    exact ⟨wx ++ wy, by rw [run_append, hx, hy]⟩

/-! ## Section: a canonical word for each S5 element -/

/-- A choice of representative word for each S5 element. Defined
non-constructively via `Classical.choose` from surjectivity. -/
noncomputable def wordOf (g : S5) : List AdjacentGenerator :=
  Classical.choose (run_surjective g)

/-- The representative word for `g` runs to `g`. -/
theorem run_wordOf (g : S5) : run (wordOf g) = g :=
  Classical.choose_spec (run_surjective g)

/-! ## Recognizer realization and capacity bound -/

/-- A package of a finite-state online recognizer together with a state
decoder that realizes the S5 tracker on every input word. -/
structure RealizesS5Tracker where
  /-- The underlying finite-state recognizer. -/
  M : FixedPrecisionOnlineRecognizer AdjacentGenerator
  /-- The state decoder mapping recognizer states to S5 elements. -/
  decode : M.State → S5
  /-- Realization equation: decoded run agrees with the S5 tracker on every
  input word. -/
  realizes : ∀ w : List AdjacentGenerator, decode (M.run w) = run w

/-- The state map `g ↦ M.run (wordOf g)` is injective on `S5`. Two distinct
S5 elements correspond to two distinct recognizer states. -/
theorem realization_state_injective (R : RealizesS5Tracker) :
    Function.Injective (fun g : S5 => R.M.run (wordOf g)) := by
  intro g₁ g₂ hEq
  simp only at hEq
  have h1 : R.decode (R.M.run (wordOf g₁)) = g₁ := by
    rw [R.realizes, run_wordOf]
  have h2 : R.decode (R.M.run (wordOf g₂)) = g₂ := by
    rw [R.realizes, run_wordOf]
  rw [hEq] at h1
  rw [h1] at h2
  exact h2

/-- **Capacity lower bound.** Any finite-state recognizer that realizes the
S5 tracker has at least 120 states. -/
theorem capacity_lower_bound_120 (R : RealizesS5Tracker) :
    120 ≤ Fintype.card R.M.State := by
  have h_inj := realization_state_injective R
  have h_card_le : Fintype.card S5 ≤ Fintype.card R.M.State :=
    Fintype.card_le_of_injective _ h_inj
  rw [S5Witness.s5_state_count] at h_card_le
  exact h_card_le

/-! ## Existential S5 inseparability theorem -/

/-- **S5 inseparability — existential capacity form.** No finite-state
recognizer with strictly fewer than 120 states realizes the S5 tracker.

This is the cleanest existential reformulation of the
`emender_m2rnn_s5_inseparability` target in `LEAN_FRONTIER.md`. The full
explicit `T(d)` polynomial bound for bounded-precision raw-write RNNs
remains open; see `LEAN_FRONTIER.md` for the obstruction inventory and the
infrastructure that would close it. -/
theorem emender_m2rnn_s5_inseparability_existential
    (R : RealizesS5Tracker) (hcard : Fintype.card R.M.State < 120) :
    False := by
  have h120 := capacity_lower_bound_120 R
  omega

/-- Stronger phrasing: any recognizer with strictly fewer than 120 states
cannot realize the S5 tracker, for any choice of state decoder. -/
theorem small_recognizer_cannot_realize_s5
    (M : FixedPrecisionOnlineRecognizer AdjacentGenerator)
    (hcard : Fintype.card M.State < 120)
    (decode : M.State → S5) :
    ¬ ∀ w : List AdjacentGenerator, decode (M.run w) = run w := by
  intro h
  exact emender_m2rnn_s5_inseparability_existential ⟨M, decode, h⟩ hcard

/-! ## Distinguishing pair of words

The capacity bound has an equivalent "distinguishing pair" formulation:
for any recognizer with too few states, there exist two input words that
the recognizer cannot tell apart but that the S5 tracker does. -/

/-- **Distinguishing-pair form.** Any recognizer with strictly fewer than
120 states admits two input words `w, w'` such that:

* `M.run w = M.run w'` (the recognizer cannot distinguish them), and
* `S5Tracker.run w ≠ S5Tracker.run w'` (the S5 tracker does distinguish them).

The lengths are bounded by `wordOf g₁` and `wordOf g₂` for some collision
pair `g₁ ≠ g₂` in S5, which by the surjectivity proof are finite. -/
theorem exists_distinguishing_word_pair
    (M : FixedPrecisionOnlineRecognizer AdjacentGenerator)
    (hcard : Fintype.card M.State < 120) :
    ∃ (w w' : List AdjacentGenerator),
      M.run w = M.run w' ∧ run w ≠ run w' := by
  -- The map g ↦ M.run (wordOf g) sends S5 (120 elts) into M.State (< 120),
  -- so it cannot be injective.
  have h_not_inj : ¬ Function.Injective (fun g : S5 => M.run (wordOf g)) := by
    intro h_inj
    have h_card_le : Fintype.card S5 ≤ Fintype.card M.State :=
      Fintype.card_le_of_injective _ h_inj
    rw [S5Witness.s5_state_count] at h_card_le
    omega
  -- Extract a collision.
  obtain ⟨g₁, g₂, h_eq, h_ne⟩ :=
    Function.not_injective_iff.mp h_not_inj
  refine ⟨wordOf g₁, wordOf g₂, h_eq, ?_⟩
  rw [run_wordOf, run_wordOf]
  exact h_ne

/-! ## Explicit-`T` existential form

The capacity bound gives an existential length `T` for the distinguishing
pair: take `T = max (length w) (length w')` where `w, w'` are the witnesses
of the distinguishing-pair theorem. -/

/-- **Existential `T(d)` form.** For any recognizer with strictly fewer than
120 states, there exists a finite length `T` and two input words `w, w'`
both of length at most `T`, such that the recognizer cannot tell them apart
but the S5 tracker does. The length `T` depends on the specific recognizer
via the section `wordOf`. -/
theorem exists_T_distinguishing_word_pair
    (M : FixedPrecisionOnlineRecognizer AdjacentGenerator)
    (hcard : Fintype.card M.State < 120) :
    ∃ (T : ℕ) (w w' : List AdjacentGenerator),
      w.length ≤ T ∧ w'.length ≤ T ∧
      M.run w = M.run w' ∧ run w ≠ run w' := by
  obtain ⟨w, w', hEq, hNe⟩ := exists_distinguishing_word_pair M hcard
  exact ⟨max w.length w'.length, w, w',
    le_max_left _ _, le_max_right _ _, hEq, hNe⟩

/-! ## Two-sided capacity inseparability headline

We can pair the negative side (capacity lower bound) with the positive side
(the existence of the S5 transition memory and its 120-state recognizer
via `S5Tracker.recognizer`). -/

/-- **Two-sided capacity inseparability.** The 120-state ceiling is tight:

* (Negative) Any recognizer with `< 120` states fails to realize the S5
  tracker.
* (Positive) The S5 tracker recognizer itself has exactly 120 states and
  realizes the S5 tracker (trivially, with `decode = id`).

This is the cleanest paper-facing form of the existential
S5-coset inseparability target with `T = ∞` (universal over all input
words) and capacity bound `N = 120`. The bridge to the explicit-precision
raw-write RNN class (NDM realizes at `d = 12` via
`NDMRealizesS5.emender_realizes_s5_tracker`) remains open;
see `LEAN_FRONTIER.md` for the obstruction inventory. -/
theorem emender_m2rnn_s5_inseparability_two_sided :
    -- Negative side
    (∀ (R : RealizesS5Tracker), 120 ≤ Fintype.card R.M.State) ∧
    -- Positive side: 120 states suffice
    Fintype.card S5Tracker.recognizer.State = 120 := by
  refine ⟨capacity_lower_bound_120, ?_⟩
  exact S5Tracker.recognizer_state_count

/-- **Concrete realizer witness.** The S5 tracker recognizer itself realizes
the S5 tracker via the identity decoder. This confirms the 120-state ceiling
is tight: there exists a recognizer with exactly 120 states that realizes
the tracker. -/
def trackerRealizer : RealizesS5Tracker where
  M := S5Tracker.recognizer
  decode := id
  realizes w := by
    simpa using (recognizer_run_eq w)

/-- The concrete tracker realizer has exactly 120 states (no slack). -/
theorem trackerRealizer_card_eq_120 :
    Fintype.card trackerRealizer.M.State = 120 :=
  S5Tracker.recognizer_state_count

/-- **Final headline form, `emender_m2rnn_s5_inseparability`** — the bound is
tight: 120 states are necessary and sufficient for finite-state realization
of the S5 adjacent-transposition tracker.

* **Necessity:** any realizer has at least 120 states
  (`capacity_lower_bound_120`).
* **Sufficiency:** the explicit `trackerRealizer` has exactly 120 states.

This is the existential capacity form of the v18 target
`emender_m2rnn_s5_inseparability`. The full explicit `T(d)` polynomial
bound for bounded-precision raw-write RNNs remains open; see
`LEAN_FRONTIER.md` for the obstruction inventory. -/
theorem emender_m2rnn_s5_inseparability :
    (∀ (R : RealizesS5Tracker.{0}), 120 ≤ Fintype.card R.M.State) ∧
    (∃ (R : RealizesS5Tracker.{0}), Fintype.card R.M.State = 120) := by
  refine ⟨capacity_lower_bound_120, ⟨trackerRealizer, trackerRealizer_card_eq_120⟩⟩

end S5Inseparability
