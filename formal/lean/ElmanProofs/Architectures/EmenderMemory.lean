import ElmanProofs.Architectures.DiagonalMemory
import Mathlib.Tactic.DeriveFintype

/-!
# Emender slot-store operators over exact reals

The numeric store includes keys, values, strengths and ages. Boolean liveness
is separate architectural bookkeeping. Death freezes, rather than zeroes,
the numeric row. Ages are represented in ℝ; integer/overflow refinement and
floating-point execution are outside this model.

This is the X30 slot store, not the E88 tanh recurrence named `emender` in the
older resource taxonomy. Gates, selections and live masks are fixed inputs.
-/

namespace EmenderMemory

open MemoryAlgebra BigOperators

inductive Field (K V : ℕ) where
  | key : Fin K → Field K V
  | value : Fin V → Field K V
  | strength : Field K V
  | age : Field K V
  deriving DecidableEq, Fintype

abbrev Store (B K V : ℕ) := (Fin B × Field K V) → ℝ
abbrev Live (B : ℕ) := Fin B → Bool

structure State (B K V : ℕ) where
  data : Store B K V
  live : Live B

variable {B K V : ℕ}

/-- Fixed elementary decisions. POP's absent target means no pop fires.
The decay mask is liveness after pop and write, before decay death. -/
inductive Op (B K V : ℕ) where
  | append (target : Fin B) (key : Fin K → ℝ) (value : Fin V → ℝ) (gate : ℝ)
  | correct (target : Fin B) (enabled : Bool) (value : Fin V → ℝ)
  | merge (target : Fin B) (beta : ℝ) (value : Fin V → ℝ)
  | pop (target : Option (Fin B)) (magnitude : ℝ)
  | decayAge (live : Live B) (rate : ℝ)

/-- Direct arithmetic semantics, independently of the affine encoding. -/
def step (op : Op B K V) (s : Store B K V) : Store B K V := fun (i, f) =>
  match op with
  | .append j key value gate =>
    if i = j then
      match f with
      | .key k => key k
      | .value v => gate * value v
      | .strength => 1
      | .age => 0
    else s (i, f)
  | .correct j enabled value =>
    match f with
    | .value v => if i = j ∧ enabled = true then value v else s (i, f)
    | _ => s (i, f)
  | .merge j beta value =>
    match f with
    | .value v => if i = j then s (i, f) + beta * (value v - s (i, f)) else s (i, f)
    | _ => s (i, f)
  | .pop target magnitude =>
    match f with
    | .strength => if target = some i then s (i, f) * (1 - magnitude) else s (i, f)
    | _ => s (i, f)
  | .decayAge live rate =>
    match f with
    | .strength => if live i then s (i, f) * rate else s (i, f)
    | .age => if live i then s (i, f) + 1 else s (i, f)
    | _ => s (i, f)

def scale (op : Op B K V) : Store B K V := fun (i, f) =>
  match op with
  | .append j _ _ _ => if i = j then 0 else 1
  | .correct j enabled _ =>
    match f with
    | .value _ => if i = j ∧ enabled = true then 0 else 1
    | _ => 1
  | .merge j beta _ =>
    match f with
    | .value _ => if i = j then 1 - beta else 1
    | _ => 1
  | .pop target magnitude =>
    match f with
    | .strength => if target = some i then 1 - magnitude else 1
    | _ => 1
  | .decayAge live rate =>
    match f with
    | .strength => if live i then rate else 1
    | _ => 1

def injection (op : Op B K V) : Store B K V := fun (i, f) =>
  match op with
  | .append j key value gate =>
    if i = j then
      match f with
      | .key k => key k
      | .value v => gate * value v
      | .strength => 1
      | .age => 0
    else 0
  | .correct j enabled value =>
    match f with
    | .value v => if i = j ∧ enabled = true then value v else 0
    | _ => 0
  | .merge j beta value =>
    match f with
    | .value v => if i = j then beta * value v else 0
    | _ => 0
  | .pop _ _ => 0
  | .decayAge live _ =>
    match f with
    | .age => if live i then 1 else 0
    | _ => 0

def diagonal (op : Op B K V) : DiagonalOperator (Fin B × Field K V) :=
  ⟨scale op, injection op⟩

theorem step_eq_diagonal (op : Op B K V) (s : Store B K V) :
    step op s = (diagonal op).toAffine.apply s := by
  funext q
  rcases q with ⟨i, f⟩
  cases op <;> cases f <;>
    simp only [step, DiagonalOperator.toAffine_apply, diagonal, scale, injection] <;>
    (try split_ifs) <;> ring

instance : MemoryOperator (Op B K V) (Store B K V) where
  step := step
  embed := fun op => (diagonal op).toAffine
  correct := step_eq_diagonal

/-- Required decomposition: diagonal carry plus one-hot slot injections. The
activity set includes age increments; content-only activity is sharper below. -/
theorem step_decomposition (op : Op B K V) (s : Store B K V) :
    step op s = (fun q => scale op q * s q) +
      ∑ j ∈ slotSupport (injection op), slotInjection j (fun f => injection op (j, f)) := by
  rw [step_eq_diagonal]
  exact diagonal_slot_decomposition (diagonal op) s

theorem append_overwrites (j : Fin B) (key : Fin K → ℝ) (value : Fin V → ℝ)
    (gate : ℝ) (s : Store B K V) :
    (∀ k, step (.append j key value gate) s (j, .key k) = key k) ∧
    (∀ v, step (.append j key value gate) s (j, .value v) = gate * value v) ∧
    step (.append j key value gate) s (j, .strength) = 1 ∧
    step (.append j key value gate) s (j, .age) = 0 := by
  simp [step]

theorem correct_full_overwrite (j : Fin B) (value : Fin V → ℝ)
    (s : Store B K V) (v : Fin V) :
    step (.correct j true value) s (j, .value v) = value v := by simp [step]

theorem correct_disabled (j : Fin B) (value : Fin V → ℝ) (s : Store B K V) :
    step (.correct j false value) s = s := by
  funext q
  rcases q with ⟨i, f⟩
  cases f <;> simp [step]

theorem merge_delta_rule (j : Fin B) (beta : ℝ) (value : Fin V → ℝ)
    (s : Store B K V) (v : Fin V) :
    step (.merge j beta value) s (j, .value v) =
      s (j, .value v) + beta * (value v - s (j, .value v)) := by simp [step]

/-- Module entry points supply a binary beta. A fired MERGE then also overwrites. -/
theorem merge_one_eq_correct (j : Fin B) (value : Fin V → ℝ) (s : Store B K V) :
    step (.merge j 1 value) s = step (.correct j true value) s := by
  funext q
  rcases q with ⟨i, f⟩
  cases f <;> simp [step]

theorem correct_preserves_key (j i : Fin B) (enabled : Bool) (value : Fin V → ℝ)
    (s : Store B K V) (k : Fin K) :
    step (.correct j enabled value) s (i, .key k) = s (i, .key k) := rfl

theorem merge_preserves_key (j i : Fin B) (beta : ℝ) (value : Fin V → ℝ)
    (s : Store B K V) (k : Fin K) :
    step (.merge j beta value) s (i, .key k) = s (i, .key k) := rfl

theorem pop_strength (j : Fin B) (magnitude : ℝ) (s : Store B K V) :
    step (.pop (some j) magnitude) s (j, .strength) =
      s (j, .strength) * (1 - magnitude) := by simp [step]

theorem decay_live_strength_age (live : Live B) (rate : ℝ) (s : Store B K V)
    (i : Fin B) (hi : live i = true) :
    step (.decayAge live rate) s (i, .strength) = s (i, .strength) * rate ∧
    step (.decayAge live rate) s (i, .age) = s (i, .age) + 1 := by simp [step, hi]

theorem decay_dead_frozen (live : Live B) (rate : ℝ) (s : Store B K V)
    (i : Fin B) (hi : live i = false) (f : Field K V) :
    step (.decayAge live rate) s (i, f) = s (i, f) := by cases f <;> simp [step, hi]

/-! ## Allocation and threshold death: Boolean bookkeeping is explicit. -/

def isAppend : Op B K V → Prop
  | .append _ _ _ _ => True
  | _ => False

/-- Death masks are fixed replay inputs. They do not zero numeric data. -/
def liveAfter (op : Op B K V) (live freed : Live B) : Live B := fun i =>
  match op with
  | .append j _ _ _ => if i = j then true else live i
  | .pop _ _ | .decayAge _ _ => live i && !freed i
  | _ => live i

theorem only_append_allocates (op : Op B K V) (live freed : Live B)
    (h : ¬ isAppend op) (i : Fin B) :
    liveAfter op live freed i = true → live i = true := by
  cases op <;> simp_all [isAppend, liveAfter]

theorem append_allocates_only_selected (j : Fin B) (key : Fin K → ℝ)
    (value : Fin V → ℝ) (gate : ℝ) (live freed : Live B) (i : Fin B) :
    liveAfter (.append j key value gate) live freed i = true ↔ i = j ∨ live i = true := by
  by_cases h : i = j <;> simp [liveAfter, h]

/-- Native threshold check, including death at equality. -/
noncomputable def freeMask (threshold : ℝ) (live : Live B) (s : Store B K V) : Live B := by
  classical
  exact fun i => live i && decide (s (i, .strength) ≤ threshold)

theorem freeMask_iff (threshold : ℝ) (live : Live B) (s : Store B K V) (i : Fin B) :
    freeMask threshold live s i = true ↔ live i = true ∧ s (i, .strength) ≤ threshold := by
  classical
  simp [freeMask]

theorem pop_death_iff (j : Fin B) (magnitude threshold : ℝ) (st : State B K V) :
    liveAfter (K := K) (V := V) (.pop (some j) magnitude) st.live
      (freeMask threshold st.live (step (.pop (some j) magnitude) st.data)) j = true ↔
    st.live j = true ∧ threshold < st.data (j, .strength) * (1 - magnitude) := by
  classical
  cases h : st.live j <;> simp [liveAfter, freeMask, h, step, not_le]

theorem decay_death_iff (rate threshold : ℝ) (st : State B K V) (i : Fin B) :
    liveAfter (K := K) (V := V) (.decayAge st.live rate) st.live
      (freeMask threshold st.live (step (.decayAge st.live rate) st.data)) i = true ↔
    st.live i = true ∧ threshold < st.data (i, .strength) * rate := by
  classical
  cases h : st.live i <;> simp [liveAfter, freeMask, h, step, not_le]

/-! ## Injection activity -/

def activeSlots : Op B K V → Finset (Fin B)
  | .append j _ _ _ | .correct j _ _ | .merge j _ _ => {j}
  | .pop _ _ => ∅
  | .decayAge live _ => Finset.univ.filter (fun i => live i = true)

theorem injection_support_subset (op : Op B K V) :
    slotSupport (injection op) ⊆ activeSlots op := by
  classical
  intro i hi
  obtain ⟨f, hf⟩ := (Finset.mem_filter.mp hi).2
  cases op <;> cases f <;> simp_all [injection, activeSlots]

theorem injection_activity_bound (op : Op B K V) :
    (slotSupport (injection op)).card ≤ (activeSlots op).card ∧
    (activeSlots op).card ≤ B := by
  exact ⟨Finset.card_le_card (injection_support_subset op),
    (Finset.card_le_card (Finset.subset_univ _)).trans_eq (by simp)⟩

/-- Project away age and strength; include both keys and values as content. -/
def contentInjection (op : Op B K V) : (Fin B × (Fin K ⊕ Fin V)) → ℝ :=
  fun (i, f) => match f with
    | .inl k => injection op (i, .key k)
    | .inr v => injection op (i, .value v)

theorem content_injection_support_le_one (op : Op B K V) :
    (slotSupport (contentInjection op)).card ≤ 1 := by
  classical
  cases op with
  | append j key value gate =>
    have h : contentInjection (Op.append j key value gate) =
        slotInjection j (fun f => match f with
          | .inl k => key k | .inr v => gate * value v) := by
      funext q; rcases q with ⟨i, f⟩; cases f <;> rfl
    rw [h]; exact slotSupport_injection_card _ _
  | correct j enabled value =>
    have h : contentInjection (Op.correct (K := K) j enabled value) =
        slotInjection j (fun f => match f with
          | .inl _ => 0 | .inr v => if enabled then value v else 0) := by
      funext q; rcases q with ⟨i, f⟩
      cases f <;> simp [contentInjection, injection, slotInjection]
      split_ifs <;> simp_all
    rw [h]; exact slotSupport_injection_card _ _
  | merge j beta value =>
    have h : contentInjection (Op.merge (K := K) j beta value) =
        slotInjection j (fun f => match f with
          | .inl _ => 0 | .inr v => beta * value v) := by
      funext q; rcases q with ⟨i, f⟩; cases f <;> simp [contentInjection, injection, slotInjection]
    rw [h]; exact slotSupport_injection_card _ _
  | pop target magnitude => simp [slotSupport, contentInjection, injection]
  | decayAge live rate =>
    have h : contentInjection (Op.decayAge (K := K) (V := V) live rate) = 0 := by
      funext q; rcases q with ⟨i, f⟩; cases f <;> rfl
    simp [h, slotSupport]

theorem sparse_step (op : Op B K V) (s : Store B K V) :
    step op s = (fun q => scale op q * s q) +
      ∑ j ∈ activeSlots op, slotInjection j (fun f => injection op (j, f)) := by
  rw [← sparse_reconstruction (injection op) (activeSlots op) (injection_support_subset op)]
  exact step_eq_diagonal op s

/-- The age injection really can occupy every slot; a one-slot full-state
claim would be false. This is not a content-write lower bound. -/
theorem age_injection_dense (rate : ℝ) :
    slotSupport (injection (Op.decayAge (B := B) (K := K) (V := V) (fun _ => true) rate)) =
      Finset.univ := by
  classical
  ext i
  simp only [slotSupport, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
  exact ⟨.age, by norm_num [injection]⟩

theorem age_activity_not_one_slot :
    ¬ (slotSupport (injection (Op.decayAge (B := 2) (K := 1) (V := 1)
      (fun _ => true) 1))).card ≤ 1 := by
  rw [age_injection_dense]
  norm_num

end EmenderMemory
