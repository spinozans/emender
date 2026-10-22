/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.WeightedAutomata

/-!
# R2 — The counter-machine simulation, re-proved in the `StreamModel` framework

Merrill et al. (ICML 2026) prove (their Lemma 3) that a log-precision **MLP** RNN simulates
any real-time counter machine, and hence (their Thm. 2) that such an RNN solves an
`L`-complete problem. This file proves an exact embedding of the defined additive-update
machine in our `StreamModel` framework, not their complete Lemma-3 closed-form simulation.
`cmRealStep` retains discrete control, Boolean equality decisions, and arbitrary lookup
functions. Only `zeroMaskForm` has an arithmetic-expression correctness theorem; complete
transition assembly is unproved, including for the one-counter witness (O14).

## What is proved

* `CounterMachine` — a defined additive-update machine: finite control `Fin m`, `c`
  integer counters, arbitrary lookups from `(control, token, zero-masks)`. Reset updates
  allowed by the paper's Lemma 3 are not modelled.
* `cmStep` — its exact one-step semantics.
* `zeroMaskR` / `zeroMaskForm` / `eval_zeroMaskForm` — the **zero test is a closed form**
  in their Def. 2 sense: `max 0 (1 − |r|)`, built from `max`, `−`, `√` and `·`.
* `zeroMaskR_int` / `isZeroR_cast` — on integer-valued reals it is **exactly** the zero
  indicator, so the zero test is exact — no tolerance, no rounding, no `ε`.
* `cmRealStep`, `cmModel` — the real-channel simulation: state `Fin m × (Fin c → ℝ)`
  (control × real counters), one step per token.
* `cmModel_fold` / `cmModel_simulates` — **the simulation theorem**: the real model's state
  after any word is the image of the counter machine's state after that word. Hence a
  `StreamModel` exactly embeds the defined additive-update machine, forever; this is
  not closed-form-arithmetic completeness.
* `balancedCM`, `cmModel_fold_blocks`, `cmModel_separates_balanced` — the concrete
  one-counter machine with **balanced-block discrimination**, not exact language
  recognition (it also accepts `abab`): the positive counterpart of F3(a)'s
  impossibility (no bounded-precision channel can do this at any dimension).
* `cmModel_reachable_card` — **the log-precision content**: after a word of length `≤ n`
  the machine's reachable state set has at most `3 · (n + 1)` elements, i.e. `O(log n)`
  bits of state. This is what "log precision suffices" means in our semantic model.

## Divergences from their Lemma 3 — flagged

1. **No complete closed-form or MLP transition is synthesized.** Their Lemma 3 builds
   ReLU networks for zero masks and finite lookup tables. Our arithmetic gadget
   `max 0 (1 − |r|)` is exact on integer-valued reals, but `cmRealStep` retains
   Boolean equality decisions, discrete control, and arbitrary lookups. The gadget
   theorem does not establish a Def.-2 expression for the complete step, even for
   one counter (O14).
2. **No configured-activation realization is proved.** `cmModel` is a `StreamModel`
   with an unrestricted per-token step, not a proved Def.-3 closed-form RNN or a
   `tanh`-state model. Its unbounded counter coordinates do not prove impossibility
   of other encodings in bounded states. E97 ReLU/softplus are candidate hosts only:
   a configured elementwise activation is not a ReLU MLP transition, and the paper
   supplies no softplus construction.
3. **Precision is modelled as cardinality, not as bits.** "Log precision" is
   `cmModel_reachable_card`: a reachable-set *cardinality* bound `3 · (n+1)`. We do not
   model bit-lengths, rounding, or their Def. 10 `p_h(n)` as such.
4. **Real-time and unbounded counters.** We do not impose their `O(n)` counter bound as a
   hypothesis; the cardinality bound is proved for the concrete machine, whose counters are
   `≤ |w|` by construction. For a general machine whose counters can exceed the input
   length the bound would not hold, and we flag that rather than assume it.
-/

namespace CounterMachineSim

open Function
open LinTanhExpressivity
open PrecisionRegime
open StreamingExpressivity
open MemoryBoundary
open ArmMapping

/-! ## Real-time counter machines -/

/-- An additive-update counter machine: finite control, `c` integer counters, and
arbitrary lookups from control, token, and zero masks. Unlike the paper's Lemma 3,
this definition does not include reset updates. -/
structure CounterMachine (α : Type*) (m c : ℕ) where
  /-- Next control state. -/
  next : Fin m → α → (Fin c → Bool) → Fin m
  /-- Counter increments. -/
  delta : Fin m → α → (Fin c → Bool) → Fin c → ℤ

/-- A counter-machine configuration. -/
abbrev CMState (m c : ℕ) := Fin m × (Fin c → ℤ)

/-- The zero masks of the counters. -/
def zeroMasks {c : ℕ} (cs : Fin c → ℤ) : Fin c → Bool := fun i => decide (cs i = 0)

/-- One step of the counter machine. -/
def cmStep {α : Type*} {m c : ℕ} (CM : CounterMachine α m c) (s : CMState m c) (x : α) :
    CMState m c :=
  (CM.next s.1 x (zeroMasks s.2), fun i => s.2 i + CM.delta s.1 x (zeroMasks s.2) i)

/-- The counter machine's configuration after a word. -/
def cmRun {α : Type*} {m c : ℕ} (CM : CounterMachine α m c) (s : CMState m c) (w : List α) :
    CMState m c := w.foldl (cmStep CM) s

/-! ## The zero test is an exact closed form -/

/-- The zero test, as a real function: `1` at `0`, `0` off it, and in between only for
non-integers. This is their "zero-mask" gadget (their Lemma 3's `ε = 1/3` equality check),
here in exact form. -/
noncomputable def zeroMaskR (r : ℝ) : ℝ := max 0 (1 - |r|)

/-- **The zero mask is an arithmetic closed form** in their Def. 2 sense:
`max 0 (1 − √(r²))`. -/
def zeroMaskForm : ArithClosedForm 1 :=
  ArithClosedForm.max (ArithClosedForm.const 0)
    (ArithClosedForm.sub (ArithClosedForm.const 1)
      (ArithClosedForm.sqrt (ArithClosedForm.pow (ArithClosedForm.var 0) 2)))

theorem eval_zeroMaskForm (v : Fin 1 → ℝ) : zeroMaskForm.eval v = zeroMaskR (v 0) := by
  simp [zeroMaskForm, zeroMaskR, ArithClosedForm.eval, Real.sqrt_sq_eq_abs]

/-- **Exactness on integers.** On an integer-valued real, the closed-form zero mask returns
exactly `1` or `0` — no tolerance, no error term. Their ReLU gadget needs `|c| ≤ ε` with
`ε = 1/3`; ours is exact. -/
theorem zeroMaskR_int (n : ℤ) : zeroMaskR (n : ℝ) = if n = 0 then 1 else 0 := by
  rcases eq_or_ne n 0 with h | h
  · subst h; simp [zeroMaskR]
  · have h1 : (1 : ℝ) ≤ |(n : ℝ)| := by
      have : (1 : ℤ) ≤ |n| := by
        have := abs_pos.mpr h
        omega
      calc (1 : ℝ) = ((1 : ℤ) : ℝ) := by norm_num
        _ ≤ ((|n| : ℤ) : ℝ) := by exact_mod_cast this
        _ = |(n : ℝ)| := by rw [Int.cast_abs]
    simp only [zeroMaskR, h, if_false]
    rw [max_eq_left]
    linarith

/-- The Boolean zero test read off the real state. -/
noncomputable def isZeroR (r : ℝ) : Bool := decide (zeroMaskR r = 1)

theorem isZeroR_int (n : ℤ) : isZeroR (n : ℝ) = decide (n = 0) := by
  rw [isZeroR, zeroMaskR_int]
  by_cases h : n = 0 <;> simp [h]

theorem isZeroR_natCast (n : ℕ) : isZeroR ((n : ℝ)) = decide (n = 0) := by
  have h : ((n : ℝ)) = (((n : ℤ)) : ℝ) := by push_cast; ring
  rw [h, isZeroR_int]
  simp

/-! ## The real-channel simulation -/

/-- Embed a counter-machine configuration into the real channel: same control, counters as
reals. -/
noncomputable def cmEmbed {m c : ℕ} (s : CMState m c) : Fin m × (Fin c → ℝ) :=
  (s.1, fun i => (s.2 i : ℝ))

/-- The embedding step retains discrete control, Boolean equality decisions, and
arbitrary `next`/`delta` lookups. Only `zeroMaskForm` has an arithmetic-expression
correctness theorem; a closed form for this whole step is not proved. -/
noncomputable def cmRealStep {α : Type*} {m c : ℕ} (CM : CounterMachine α m c)
    (s : Fin m × (Fin c → ℝ)) (x : α) : Fin m × (Fin c → ℝ) :=
  (CM.next s.1 x (fun i => isZeroR (s.2 i)),
   fun i => s.2 i + (CM.delta s.1 x (fun i => isZeroR (s.2 i)) i : ℝ))

/-- **The real model's step is the counter machine's step**, under the embedding. This is the
whole simulation: integer counters stay integer-valued, and the closed-form zero test agrees
with the counter machine's exact zero test. -/
theorem cmRealStep_embed {α : Type*} {m c : ℕ} (CM : CounterMachine α m c)
    (s : CMState m c) (x : α) : cmRealStep CM (cmEmbed s) x = cmEmbed (cmStep CM s x) := by
  obtain ⟨q, cs⟩ := s
  have hz : ∀ i, isZeroR ((cs i : ℝ)) = zeroMasks cs i := by
    intro i; rw [isZeroR_int]; rfl
  simp only [cmRealStep, cmEmbed, cmStep, hz]
  refine congrArg₂ Prod.mk rfl ?_
  funext i
  push_cast
  ring

/-- The `StreamModel` with discrete `Fin m` control and `c` real counter coordinates.
The readout is arbitrary; no fully real-coded arithmetic transition is assembled. -/
noncomputable def cmModel {α β : Type*} {m c : ℕ} (CM : CounterMachine α m c)
    (q₀ : Fin m) (readout : Fin m × (Fin c → ℝ) → β) : StreamModel (Fin m × (Fin c → ℝ)) α β where
  init := cmEmbed (q₀, fun _ => 0)
  step := cmRealStep CM
  out := readout

/-- The fold of the real model's step from an embedded configuration is the embedding of the
counter machine's fold. -/
theorem fold_embed {α : Type*} {m c : ℕ} (CM : CounterMachine α m c) :
    ∀ (w : List α) (s : CMState m c),
      w.foldl (cmRealStep CM) (cmEmbed s) = cmEmbed (w.foldl (cmStep CM) s) := by
  intro w
  induction w with
  | nil => intro s; rfl
  | cons x xs ih =>
      intro s
      rw [List.foldl_cons, List.foldl_cons, cmRealStep_embed, ih]

/-- **Exact additive-update embedding, not closed-form completeness.** After any
word, the model's state is the embedded configuration of the defined machine —
for every word, hence forever on an `ω`-stream (the `ω` state is the same fold,
`StreamModel.omegaState`). -/
theorem cmModel_fold {α β : Type*} {m c : ℕ} (CM : CounterMachine α m c) (q₀ : Fin m)
    (out : Fin m × (Fin c → ℝ) → β) (w : List α) :
    (cmModel CM q₀ out).stateAfter w = cmEmbed (cmRun CM (q₀, fun _ => (0 : ℤ)) w) := by
  have hinit : (cmModel CM q₀ out).init = cmEmbed (q₀, fun _ => (0 : ℤ)) := rfl
  have hstep : (cmModel CM q₀ out).step = cmRealStep CM := rfl
  have h : (cmModel CM q₀ out).stateAfter w
      = w.foldl (cmRealStep CM) (cmEmbed (q₀, fun _ => (0 : ℤ))) := by
    rw [StreamModel.stateAfter, hinit, hstep]
  rw [h, fold_embed CM w]
  rfl

/-! ## The concrete one-counter machine for balanced-block discrimination -/

/-- A one-counter machine for balanced-block discrimination (`SeparatesBalanced`),
not exact `aᵏbᵏ` recognition: it also accepts `abab`. Controls: `0` pushing, `1` popping,
`2` underflow. Updates ignore the prior phase; `a` increments and `b` decrements unless zero. -/
def balancedCM : CounterMachine Bool 3 1 where
  next := fun _ x z => match x with
    | true => 0
    | false => if z 0 then 2 else 1
  delta := fun _ x z _ => match x with
    | true => 1
    | false => if z 0 then 0 else -1

/-- The real-channel readout: "the popping phase with an empty stack". -/
noncomputable def outR (s : Fin 3 × (Fin 1 → ℝ)) : Bool := decide (s.1 = 1 ∧ s.2 0 = 0)

/-- The control index of a stack phase (matching `balancedCM`'s control numbering). -/
def phaseIdx : StackPhase → Fin 3
  | StackPhase.pushing => 0
  | StackPhase.popping => 1
  | StackPhase.underflow => 2

/-- The state map from the X22T stack model's counter form into the real channel. -/
noncomputable def toReal (p : StackPhase × ℕ) : Fin 3 × (Fin 1 → ℝ) :=
  (phaseIdx p.1, fun _ => (p.2 : ℝ))

theorem cmRealStep_balanced_toReal (p : StackPhase × ℕ) (x : Bool) :
    cmRealStep balancedCM (toReal p) x = toReal (counterStep p x) := by
  obtain ⟨ph, n⟩ := p
  cases x with
  | true => simp [cmRealStep, balancedCM, toReal, counterStep, phaseIdx]
  | false =>
      rcases n with _ | n
      · simp [cmRealStep, balancedCM, toReal, counterStep, phaseIdx, isZeroR, zeroMaskR]
      · simp only [cmRealStep, balancedCM, toReal, counterStep, isZeroR_natCast, phaseIdx]
        norm_num

/-- The counter-machine fold tracks the stack model's depth. -/
theorem fold_counterStep_eq_stack : ∀ (w : List Bool) (p : StackPhase × List Unit),
    w.foldl counterStep (p.1, p.2.length)
      = (fun q : StackPhase × List Unit => (q.1, q.2.length)) (w.foldl stackStep p)
  | [], _ => rfl
  | x :: xs, p => by
      rw [List.foldl_cons, List.foldl_cons, ← counterStep_depth p x,
        fold_counterStep_eq_stack xs (stackStep p x)]

/-- The counter fold from the initial configuration. -/
theorem fold_counterStep_init (w : List Bool) :
    w.foldl counterStep (StackPhase.pushing, 0)
      = (fun q : StackPhase × List Unit => (q.1, q.2.length))
          (w.foldl stackStep (StackPhase.pushing, [])) := by
  have h := fold_counterStep_eq_stack w (StackPhase.pushing, [])
  simpa using h

/-- The readout on an embedded configuration. -/
theorem outR_cmEmbed (q : Fin 3) (m : ℤ) :
    outR (cmEmbed (q, fun _ => m)) = decide (q = 1 ∧ m = 0) := by
  simp [outR, cmEmbed, Int.cast_eq_zero]

/-- **The counter machine step tracks the X22T counter step** under the phase index. -/
theorem cmStep_balanced_toCounter (ph : StackPhase) (m : ℕ) (x : Bool) :
    cmStep balancedCM (phaseIdx ph, fun _ => (m : ℤ)) x
      = (phaseIdx (counterStep (ph, m) x).1,
          fun _ => (((counterStep (ph, m) x).2 : ℕ) : ℤ)) := by
  cases x with
  | true => simp [cmStep, balancedCM, counterStep, phaseIdx]
  | false =>
      rcases m with _ | n
      · simp [cmStep, balancedCM, counterStep, phaseIdx, zeroMasks]
      · have hz : decide (((n : ℤ) + 1) = 0) = false := by
          simp only [decide_eq_false_iff_not]; omega
        refine Prod.ext ?_ ?_
        · simp [cmStep, balancedCM, counterStep, phaseIdx, zeroMasks, hz]
        · funext i
          simp only [cmStep, balancedCM, counterStep, phaseIdx, zeroMasks, decide_eq_true_eq]
          rw [if_neg (by omega : ¬ ((↑(n + 1) : ℤ) = 0))]
          push_cast
          ring

/-- The counter machine's configuration is the phase-indexed image of the X22T counter. -/
theorem cmRun_balanced_aux : ∀ (w : List Bool) (ph : StackPhase) (m : ℕ),
    cmRun balancedCM (phaseIdx ph, fun _ => (m : ℤ)) w
      = (phaseIdx (w.foldl counterStep (ph, m)).1,
          fun _ => (((w.foldl counterStep (ph, m)).2 : ℕ) : ℤ))
  | [], ph, m => by simp [cmRun]
  | x :: xs, ph, m => by
      show xs.foldl (cmStep balancedCM) (cmStep balancedCM (phaseIdx ph, fun _ => (m : ℤ)) x) = _
      rw [cmStep_balanced_toCounter ph m x]
      exact cmRun_balanced_aux xs (counterStep (ph, m) x).1 (counterStep (ph, m) x).2

/-- **The configuration after `aᵏ' bᵏ`** — the counter machine's state, with the control and
counter depth from X22T's `fold_blocks`. -/
theorem cmRun_blocks (k k' : ℕ) :
    cmRun balancedCM (0, fun _ => (0 : ℤ)) (List.replicate k' true ++ List.replicate k false)
      = (if k ≤ k' then (if k = 0 then phaseIdx StackPhase.pushing
            else phaseIdx StackPhase.popping) else phaseIdx StackPhase.underflow,
          fun _ => (if k ≤ k' then ((k' - k : ℕ) : ℤ) else 0)) := by
  have hbridge := cmRun_balanced_aux (List.replicate k' true ++ List.replicate k false)
    StackPhase.pushing 0
  rw [show (phaseIdx StackPhase.pushing, fun _ => ((0 : ℕ) : ℤ))
      = ((0 : Fin 3), fun _ => (0 : ℤ)) from rfl] at hbridge
  rw [hbridge, fold_counterStep_init, fold_blocks k k']
  by_cases hle : k ≤ k'
  · rw [if_pos hle, if_pos hle]
    refine Prod.ext ?_ ?_
    · by_cases hk : k = 0 <;> simp [hk]
    · funext i
      simp only [List.length_replicate]
      omega
  · rw [if_neg hle, if_neg hle]
    refine Prod.ext ?_ ?_
    · simp
    · funext i
      simp [hle]

theorem cmModel_separates_balanced :
    SeparatesBalanced (cmModel balancedCM 0 outR) true false where
  hits := by
    intro k hk
    have hk0 : k ≠ 0 := by omega
    have hstate : (cmModel balancedCM 0 outR).stateAfter
        (List.replicate k true ++ List.replicate k false)
          = cmEmbed (phaseIdx StackPhase.popping, fun _ => (0 : ℤ)) := by
      rw [cmModel_fold, cmRun_blocks k k, if_pos (le_refl k), if_neg hk0]
      simp
    rw [hstate]
    simp [cmModel, outR, cmEmbed, phaseIdx]
  misses := by
    intro k k' _ _ hne
    simp only [StreamModel.stateAfter]
    rw [show List.foldl (cmModel balancedCM 0 outR).step (cmModel balancedCM 0 outR).init
        (List.replicate k' true ++ List.replicate k false)
          = (cmModel balancedCM 0 outR).stateAfter
            (List.replicate k' true ++ List.replicate k false) from rfl]
    rw [cmModel_fold, cmRun_blocks k k']
    by_cases hle : k ≤ k'
    · have hk0 : k ≠ 0 := by omega
      have hsub : k' - k ≠ 0 := by omega
      rw [if_pos hle, if_neg hk0]
      have hdec : decide (((((k' - k : ℕ)) : ℤ) : ℝ) = 0) = false := by
        simp only [decide_eq_false_iff_not]
        intro hz
        have h1 : ((k' - k : ℕ) : ℤ) = 0 := by exact_mod_cast hz
        exact hsub (by exact_mod_cast h1)
      simp only [cmModel, outR, cmEmbed, phaseIdx, if_pos hle, true_and]
      exact hdec
    · rw [if_neg hle]
      simp [cmModel, outR, cmEmbed, phaseIdx]

/-- The stack model's counter form never grows by more than one per token. -/
theorem counterStep_snd_le (p : StackPhase × ℕ) (x : Bool) : (counterStep p x).2 ≤ p.2 + 1 := by
  obtain ⟨ph, n⟩ := p
  cases x with
  | true => simp [counterStep]
  | false =>
      rcases n with _ | n
      · simp [counterStep]
      · simp only [counterStep]
        omega

theorem counterFold_snd_le (w : List Bool) (p : StackPhase × ℕ) :
    (w.foldl counterStep p).2 ≤ p.2 + w.length := by
  induction w generalizing p with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, List.length_cons]
      have h1 := ih (counterStep p x)
      have h2 := counterStep_snd_le p x
      omega

/-- **The log-precision content, as cardinality**: after a word of length `≤ n` the reachable
set of the real-channel model has at most `3 · (n + 1)` elements — three controls times
`n + 1` counter values. That is `O(log n)` bits of state, the resource their Lemma 3
accounts for. -/
theorem cmModel_reachable_card (n : ℕ) :
    ∃ S : Finset (Fin 3 × (Fin 1 → ℝ)),
      (∀ w : List Bool, w.length ≤ n → (cmModel balancedCM 0 outR).stateAfter w ∈ S) ∧
      S.card ≤ 3 * (n + 1) := by
  classical
  refine ⟨((Finset.univ : Finset (Fin 3)).product (Finset.range (n + 1))).image
      (fun p : Fin 3 × ℕ => cmEmbed (p.1, fun _ => (p.2 : ℤ))), ?_, ?_⟩
  · intro w hw
    have hbridge := cmRun_balanced_aux w StackPhase.pushing 0
    rw [show (phaseIdx StackPhase.pushing, fun _ => ((0 : ℕ) : ℤ))
        = ((0 : Fin 3), fun _ => (0 : ℤ)) from rfl] at hbridge
    have hbound : (w.foldl counterStep (StackPhase.pushing, 0)).2 ≤ n := by
      have h := counterFold_snd_le w counterModel.init
      simp only [counterModel] at h
      omega
    refine Finset.mem_image.mpr
      ⟨(phaseIdx (w.foldl counterStep (StackPhase.pushing, 0)).1,
        (w.foldl counterStep (StackPhase.pushing, 0)).2), ?_, ?_⟩
    · exact Finset.mem_product.mpr
        ⟨Finset.mem_univ _, Finset.mem_range.mpr (by omega)⟩
    · rw [cmModel_fold, hbridge]
  · refine le_trans Finset.card_image_le ?_
    calc ((Finset.univ : Finset (Fin 3)).product (Finset.range (n + 1))).card
        = (Finset.univ : Finset (Fin 3)).card * (Finset.range (n + 1)).card :=
          Finset.card_product _ _
      _ = 3 * (n + 1) := by simp
      _ ≤ 3 * (n + 1) := le_refl _

end CounterMachineSim
