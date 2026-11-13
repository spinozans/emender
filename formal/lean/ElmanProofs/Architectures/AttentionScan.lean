import ElmanProofs.Architectures.AttentionMixture
import ElmanProofs.Architectures.ConditionalScan
import ElmanProofs.Expressivity.MemoryScan

/-!
# One scheduled algebra for store writes and index reads

A read's values are an affine function of the current numeric state. This
covers mutable store values, a carried context/world value cache, and supplied
constant payloads. It does not cover nonlinear value production or search.
The accumulator can itself be a product of output coordinates to retain each
read separately. This theorem is algebraic, not a work/depth bound.
-/

namespace MemoryAlgebra.Attention

noncomputable section

variable {S V : Type*} [AddCommGroup S] [Module ℝ S] [AddCommGroup V] [Module ℝ V]

/-- A closed block triangular family on a fixed source state and residual. -/
structure ReadWriteSummary (S V : Type*) [AddCommGroup S] [Module ℝ S]
    [AddCommGroup V] [Module ℝ V] where
  store : AffineOperator S
  readLinear : S →ₗ[ℝ] V
  readBias : V

namespace ReadWriteSummary

def apply (a : ReadWriteSummary S V) (s : S × V) : S × V :=
  (a.store.apply s.1, s.2 + a.readLinear s.1 + a.readBias)

def toAffine (a : ReadWriteSummary S V) : AffineOperator (S × V) where
  linear :=
    { toFun := fun s => (a.store.linear s.1, s.2 + a.readLinear s.1)
      map_add' := by
        intro x y
        apply Prod.ext
        · exact map_add a.store.linear _ _
        · change x.2 + y.2 + a.readLinear (x.1 + y.1) =
            (x.2 + a.readLinear x.1) + (y.2 + a.readLinear y.1)
          rw [map_add]; abel
      map_smul' := by
        intro c x
        apply Prod.ext
        · exact map_smul a.store.linear _ _
        · change c • x.2 + a.readLinear (c • x.1) = c • (x.2 + a.readLinear x.1)
          rw [map_smul, smul_add] }
  injection := (a.store.injection, a.readBias)

theorem toAffine_apply (a : ReadWriteSummary S V) (s : S × V) :
    a.toAffine.apply s = a.apply s := rfl

def identity : ReadWriteSummary S V := ⟨1, 0, 0⟩

def thenOp (a b : ReadWriteSummary S V) : ReadWriteSummary S V :=
  ⟨a.store * b.store, a.readLinear + b.readLinear.comp a.store.linear,
    a.readBias + b.readLinear a.store.injection + b.readBias⟩

/-- Reads make an off-diagonal block, but a fixed-dimensional triangular
summary remains closed. A blanket nonclosure claim would be false. -/
theorem toAffine_thenOp (a b : ReadWriteSummary S V) :
    (a.thenOp b).toAffine = a.toAffine * b.toAffine := by
  apply AffineOperator.ext
  · apply LinearMap.ext
    intro s
    apply Prod.ext
    · rfl
    · simp only [toAffine, thenOp, AffineOperator.linear_mul, LinearMap.comp_apply,
        LinearMap.coe_mk, AddHom.coe_mk, LinearMap.add_apply]
      abel
  · apply Prod.ext
    · rfl
    · rfl

theorem toAffine_identity : (identity : ReadWriteSummary S V).toAffine = 1 := by
  apply AffineOperator.ext
  · apply LinearMap.ext
    intro s
    change (s.1, s.2 + (0 : S →ₗ[ℝ] V) s.1) = s
    simp
  · rfl

def composeAll : List (ReadWriteSummary S V) → ReadWriteSummary S V
  | [] => identity
  | a :: as => a.thenOp (composeAll as)

theorem composeAll_toAffine (ops : List (ReadWriteSummary S V)) :
    (composeAll ops).toAffine = AffineOperator.composeAll (ops.map toAffine) := by
  induction ops with
  | nil => exact toAffine_identity
  | cons a as ih => simp [composeAll, toAffine_thenOp, ih, AffineOperator.composeAll]

end ReadWriteSummary

/-- All search, exclusion, chaining, fan-out and numerical weights have already
been decided. `source` and `offset` must also be fixed for this event. -/
inductive Event (W U R S V : Type*) [AddCommGroup S] [Module ℝ S]
    [AddCommGroup V] [Module ℝ V] where
  | write (op : W)
  | attend (op : AttentionOperator U R V) (source : S →ₗ[ℝ] (U → V)) (offset : U → V)

namespace Event

variable {W U R : Type*} [MemoryOperator W S]

def step : Event W U R S V → S × V → S × V
  | .write w, s => (MemoryOperator.step w s.1, s.2)
  | .attend a source offset, s => (s.1, s.2 + a.readout (source s.1 + offset))

def summary : Event W U R S V → ReadWriteSummary S V
  | .write w => ⟨MemoryOperator.embed w, 0, 0⟩
  | .attend a source offset => ⟨1, a.readoutLinear.comp source, a.readout offset⟩

theorem step_eq_summary (op : Event W U R S V) (s : S × V) :
    op.step s = op.summary.toAffine.apply s := by
  rw [ReadWriteSummary.toAffine_apply]
  cases op with
  | write w => simp [step, summary, ReadWriteSummary.apply, MemoryOperator.correct]
  | attend a source offset =>
    change (s.1, s.2 + a.readoutLinear (source s.1 + offset)) =
      ((1 : AffineOperator S).apply s.1, s.2 + a.readoutLinear (source s.1) + a.readout offset)
    simp [map_add, add_assoc, AttentionOperator.readoutLinear]

instance : MemoryOperator (Event W U R S V) (S × V) where
  step := step
  embed := fun op => op.summary.toAffine
  correct := step_eq_summary

theorem combined_scan_compatible (ops : List (Event W U R S V)) :
    ScanCompatibleRelativeTo step ops := memoryOperator_scanCompatible ops

/-- The payoff: any fixed interleaving of writes and reads is one affine scan. -/
theorem combined_fold_scan (ops : List (Event W U R S V)) (s : S × V) :
    ops.foldl (fun x op => op.step x) s =
      (AffineOperator.composeAll (ops.map (fun op => op.summary.toAffine))).apply s :=
  memoryOperator_fold ops s

theorem combined_closed_summary (ops : List (Event W U R S V)) (s : S × V) :
    ops.foldl (fun x op => op.step x) s =
      (ReadWriteSummary.composeAll (ops.map summary)).apply s := by
  rw [← ReadWriteSummary.toAffine_apply, ReadWriteSummary.composeAll_toAffine,
    List.map_map, combined_fold_scan]
  rfl

theorem combined_chunked_scan (chunks : List (List (Event W U R S V))) (s : S × V) :
    chunks.flatten.foldl (fun x op => op.step x) s =
      (AffineOperator.composeAll (chunks.map (fun c =>
        AffineOperator.composeAll (c.map (fun op => op.summary.toAffine))))).apply s := by
  rw [combined_fold_scan, List.map_flatten, AffineOperator.composeAll_flatten, List.map_map]
  rfl

theorem combined_prefix_scan (ops : List (Event W U R S V)) (n : ℕ) (s : S × V) :
    (ops.take n).foldl (fun x op => op.step x) s =
      (AffineOperator.composeAll ((ops.take n).map (fun op => op.summary.toAffine))).apply s :=
  combined_fold_scan _ _

theorem combined_ordered_tree (ops : List (Event W U R S V)) (tree : ScanTree (S × V))
    (h : tree.leaves = ops.map (fun op => op.summary.toAffine)) (s : S × V) :
    ops.foldl (fun x op => op.step x) s = tree.summary.apply s := by
  rw [combined_fold_scan, ScanTree.summary_eq_composeAll, h]

/-- Replaying state-dependent decisions requires evidence that the supplied
trace matches the original trajectory. This does not compute that trace. -/
theorem combined_routed_scan {I : Type*} (choose : I → S × V → Event W U R S V)
    (inputs : List I) (ops : List (Event W U R S V)) (s : S × V)
    (h : TraceMatches choose inputs ops s) :
    routedRun choose inputs s =
      (AffineOperator.composeAll (ops.map (fun op => op.summary.toAffine))).apply s :=
  routedRun_eq_scan_of_trace choose inputs ops s h

end Event

/-! ## Concrete SELF / WORLD / STORE value sources and actual five-op writes -/

namespace Emender

variable {C W : Type*} {B K D : ℕ}

abbrev Position (C W : Type*) (B : ℕ) := C ⊕ (W ⊕ Fin B)
abbrev Value (D : ℕ) := Fin D → ℝ
abbrev SourceState (C W : Type*) (B K D : ℕ) :=
  EmenderMemory.Store B K D × ((C → Value D) × (W → Value D))

/-- Slot store plus explicitly carried context/corpus value tables. This is
larger than the original store; the tables are not hidden in the index. -/
def values : SourceState C W B K D →ₗ[ℝ] (Position C W B → Value D) where
  toFun := fun s j => match j with
    | .inl c => s.2.1 c
    | .inr (.inl w) => s.2.2 w
    | .inr (.inr b) => fun d => s.1 (b, .value d)
  map_add' := by
    intro x y
    funext j d
    cases j with
    | inl c => rfl
    | inr j => cases j <;> rfl
  map_smul' := by
    intro c x
    funext j d
    cases j with
    | inl c => rfl
    | inr j => cases j <;> rfl

theorem attend_self_source (s : SourceState C W B K D) (c : C) :
    values s (.inl c) = s.2.1 c := rfl

theorem attend_world_source (s : SourceState C W B K D) (w : W) :
    values s (.inr (.inl w)) = s.2.2 w := rfl

theorem attend_store_source (s : SourceState C W B K D) (b : Fin B) (d : Fin D) :
    values s (.inr (.inr b)) d = s.1 (b, .value d) := rfl

/-- Type carries table universes so MemoryOperator's state out-parameter is
unambiguous. The actual token semantics come from phase one. -/
structure Write (C W : Type*) (B K D : ℕ) where
  token : EmenderMemory.TokenDecision B K D

def Write.summary (w : Write C W B K D) :
    ReadWriteSummary (EmenderMemory.Store B K D) ((C → Value D) × (W → Value D)) :=
  ⟨w.token.summary.toAffine, 0, 0⟩

instance : MemoryOperator (Write C W B K D) (SourceState C W B K D) where
  step := fun w s => (w.token.step s.1, s.2)
  embed := fun w => w.summary.toAffine
  correct := by
    intro w s
    simp [Write.summary, ReadWriteSummary.toAffine_apply, ReadWriteSummary.apply,
      EmenderMemory.token_summary_correct]

abbrev ScheduledEvent (C W R : Type*) (B K D : ℕ) :=
  Event (Write C W B K D) (Position C W B) R (SourceState C W B K D) (Value D)

def attend {R : Type*} (a : AttentionOperator (Position C W B) R (Value D)) :
    ScheduledEvent C W R B K D := .attend a values 0

theorem attend_current_values {R : Type*}
    (a : AttentionOperator (Position C W B) R (Value D))
    (s : SourceState C W B K D × Value D) :
    (attend a).step s = (s.1, s.2 + a.readout (values s.1)) := by
  simp [attend, Event.step]

theorem store_index_single_scan {R : Type*}
    (ops : List (ScheduledEvent C W R B K D)) (s : SourceState C W B K D × Value D) :
    ops.foldl (fun x op => op.step x) s =
      (AffineOperator.composeAll (ops.map (fun op => op.summary.toAffine))).apply s :=
  Event.combined_fold_scan ops s

end Emender

end

end MemoryAlgebra.Attention
