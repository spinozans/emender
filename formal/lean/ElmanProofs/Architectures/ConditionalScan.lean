import ElmanProofs.Architectures.EmenderToken
import ElmanProofs.Architectures.GDNMemoryOperator
import ElmanProofs.Architectures.RecurrentResourceFormalism

/-! # Semantic scan refinement and named hard-decision obstructions -/

namespace MemoryAlgebra

variable {S I Op : Type*} [AddCommGroup S] [Module ℝ S] [MemoryOperator Op S]

/-- A state-dependent selector is allowed in the original run. -/
def routedRun (choose : I → S → Op) (inputs : List I) (s : S) : S :=
  inputs.foldl (fun x i => MemoryOperator.step (choose i x) x) s

/-- The supplied decisions must match the particular trajectory being replayed. -/
def TraceMatches (choose : I → S → Op) : List I → List Op → S → Prop
  | [], [], _ => True
  | i :: inputs, op :: ops, s =>
      choose i s = op ∧ TraceMatches choose inputs ops (MemoryOperator.step op s)
  | _, _, _ => False

theorem routedRun_eq_replay (choose : I → S → Op) (inputs : List I)
    (ops : List Op) (s : S) (h : TraceMatches choose inputs ops s) :
    routedRun choose inputs s = ops.foldl (fun x op => MemoryOperator.step op x) s := by
  induction inputs generalizing ops s with
  | nil => cases ops <;> simp_all [TraceMatches, routedRun]
  | cons i inputs ih =>
    cases ops with
    | nil => simp [TraceMatches] at h
    | cons op ops =>
      obtain ⟨heq, hrest⟩ := h
      simp only [routedRun, List.foldl_cons, heq]
      exact ih ops (MemoryOperator.step op s) hrest

theorem routedRun_eq_scan_of_trace (choose : I → S → Op) (inputs : List I)
    (ops : List Op) (s : S) (h : TraceMatches choose inputs ops s) :
    routedRun choose inputs s =
      (AffineOperator.composeAll (ops.map MemoryOperator.embed)).apply s := by
  rw [routedRun_eq_replay choose inputs ops s h, memoryOperator_fold]

end MemoryAlgebra

namespace RecurrentResourceFormalism

open MemoryAlgebra

/-- Semantic companion to the legacy taxonomy. A Boolean field is not evidence
of this schedule-indexed property; the supplied update equation is essential. -/
structure SemanticArchitectureSignature (D S : Type*) where
  taxonomy : ArchitectureSignature
  update : D → S → S

def SemanticArchitectureSignature.scanCompatibleRelativeTo {D S : Type*}
    [AddCommGroup S] [Module ℝ S] (a : SemanticArchitectureSignature D S)
    (schedule : List D) : Prop := ScanCompatibleRelativeTo a.update schedule

/-- X30's slot store is separate from the legacy `emender` E88/NDM signature.
The numeric count excludes Boolean live flags and usage/read bookkeeping. -/
def slotStoreSignature (layers heads B K V : ℕ) : ArchitectureSignature where
  name := "Emender X30 slot store (unconditioned routing)"
  stateGeometry := .externalTape
  temporalNonlinearity := .fullState
  transitionSide := .elementwise
  transitionControl := .inputAndStateDependent
  writeRule := .deltaCorrecting
  carryPlacement := .outsideNonlinearity
  memorySemantics := .errorCorrectingAssociativeTable
  implementationMode := .sequentialManyProgram
  totalLayers := layers
  nonlinearTemporalLayers := layers
  headsPerLayer := heads
  stateScalarsPerHead := B * (K + V + 2)
  pureRecurrentStack := true
  scanCompatible := false

def slotStoreSemantics (layers heads B K V : ℕ) :
    SemanticArchitectureSignature (EmenderMemory.TokenDecision B K V) (EmenderMemory.Store B K V) :=
  ⟨slotStoreSignature layers heads B K V, EmenderMemory.TokenDecision.step⟩

theorem slotStore_scan_relative (layers heads B K V : ℕ)
    (schedule : List (EmenderMemory.TokenDecision B K V)) :
    (slotStoreSemantics layers heads B K V).scanCompatibleRelativeTo schedule :=
  EmenderMemory.token_conditional_scan schedule

/-- Independent-program shape now quantifies over the update family, P1's
missing predicate. The affine hypothesis is not needed for program independence. -/
theorem slotStore_multiProgrammed {P : Type*} [Fintype P] (B K V : ℕ)
    (many : 2 ≤ Fintype.card P) :
    IsMultiProgrammedRecurrentStack
      (fun p (ts : P → EmenderMemory.TokenDecision B K V) s => (ts p).step s)
      (independentUpdate (fun p (ts : P → EmenderMemory.TokenDecision B K V) s => (ts p).step s)) :=
  every_update_family_multiProgrammed _ many

theorem gdn2_multiProgrammed {P : Type*} [Fintype P] (K V : ℕ)
    (many : 2 ≤ Fintype.card P) :
    IsMultiProgrammedRecurrentStack
      (fun p (ts : P → CompositionalDepthDuality.GdnStepParams K V) s =>
        CompositionalDepthDuality.gdnStep (ts p) s)
      (independentUpdate (fun p (ts : P → CompositionalDepthDuality.GdnStepParams K V) s =>
        CompositionalDepthDuality.gdnStep (ts p) s)) :=
  every_update_family_multiProgrammed _ many

end RecurrentResourceFormalism

namespace MemoryAlgebra.Obstructions

noncomputable section

/-- Full overwrite of old value 0 with new value 1, enabled by a hard gate. -/
def hardGate (g : ℝ) : ℝ := if (1 : ℝ) / 2 < g then 1 else 0

theorem hard_gate_witness : hardGate 1 + hardGate 0 ≠ hardGate (1/2) + hardGate (1/2) := by
  norm_num [hardGate]

/-- Even restricting the gate to [0,1] does not make thresholding affine. -/
theorem hard_gate_not_affine_on_unit_interval :
    ¬ ∃ a : AffineOperator ℝ, ∀ g, 0 ≤ g → g ≤ 1 → a.apply g = hardGate g := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity (1/2) (1/2)
  rw [show (1/2 : ℝ) + 1/2 = 1 by norm_num,
    ha 1 (by norm_num) (by norm_num), ha 0 (by norm_num) (by norm_num),
    ha (1/2) (by norm_num) (by norm_num)] at h
  exact hard_gate_witness h

def hardCorrect (g : ℝ) : EmenderMemory.Op 1 0 1 :=
  .correct 0 (decide ((1 : ℝ) / 2 < g)) (fun _ => 1)

/-- The gate counterexample is a literal CORRECT operation on the zero store. -/
theorem hard_correct_eq_gate (g : ℝ) :
    EmenderMemory.step (hardCorrect g) 0 (0, .value 0) = hardGate g := by
  classical
  simp [hardCorrect, EmenderMemory.step, hardGate]

def strengthStore (x : ℝ) : EmenderMemory.Store 1 0 0 :=
  fun (_, f) => match f with | .strength => x | _ => 0

/-- A live strength is checked for death, then the next decay uses the
resulting live mask. Dead strength freezes, exactly as in the reference. -/
def deathThenDecay (x : ℝ) : ℝ :=
  EmenderMemory.step
    (.decayAge
      (fun i => !(EmenderMemory.freeMask (1/2) (fun _ => true) (strengthStore x) i)) (1/2))
    (strengthStore x) (0, .strength)

theorem deathThenDecay_formula (x : ℝ) :
    deathThenDecay x = if x ≤ (1 : ℝ) / 2 then x else x / 2 := by
  classical
  simp [deathThenDecay, EmenderMemory.step, EmenderMemory.freeMask, strengthStore]
  split_ifs <;> linarith

theorem death_threshold_witness :
    deathThenDecay 1 + deathThenDecay 0 ≠ deathThenDecay (1/2) + deathThenDecay (1/2) := by
  norm_num [deathThenDecay_formula]

/-- Fixed pop/decay magnitudes alone are insufficient: recomputing death
masks from the state breaks affinity even of the numeric strength channel. -/
theorem state_dependent_death_not_affine :
    ¬ ∃ a : AffineOperator ℝ, ∀ x, a.apply x = deathThenDecay x := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity (1/2) (1/2)
  rw [show (1/2 : ℝ) + 1/2 = 1 by norm_num, ha, ha, ha] at h
  exact death_threshold_witness h

end

end MemoryAlgebra.Obstructions
