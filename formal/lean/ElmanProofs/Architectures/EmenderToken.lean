import ElmanProofs.Architectures.EmenderMemory

/-! # Token ordering, fixed death traces, and token-level scan summaries -/

namespace EmenderMemory

open MemoryAlgebra BigOperators

variable {B K V : ℕ}

/-- One write at most per token. -/
inductive Write (B K V : ℕ) where
  | none
  | append (target : Fin B) (key : Fin K → ℝ) (value : Fin V → ℝ) (gate : ℝ)
  | correct (target : Fin B) (enabled : Bool) (value : Fin V → ℝ)
  | merge (target : Fin B) (beta : ℝ) (value : Fin V → ℝ)

def Write.ops : Write B K V → List (Op B K V)
  | .none => []
  | .append j k v g => [.append j k v g]
  | .correct j e v => [.correct j e v]
  | .merge j b v => [.merge j b v]

def Write.run (w : Write B K V) (s : Store B K V) : Store B K V :=
  w.ops.foldl (fun x op => step op x) s

def Write.live (w : Write B K V) (l : Live B) : Live B :=
  w.ops.foldl (fun live op => liveAfter op live (fun _ => false)) l

/-- All numbers here, including the append gate computed using detached
residuals, are held fixed during the state scan. The mask is post-write live. -/
structure TokenDecision (B K V : ℕ) where
  popTarget : Option (Fin B)
  popMagnitude : ℝ
  freedPop : Live B
  write : Write B K V
  liveForDecay : Live B
  decayRate : ℝ
  freedDecay : Live B

def TokenDecision.ops (t : TokenDecision B K V) : List (Op B K V) :=
  [.pop t.popTarget t.popMagnitude] ++ t.write.ops ++ [.decayAge t.liveForDecay t.decayRate]

/-- READ precedes this numeric update and is not included in the affine map. -/
def TokenDecision.step (t : TokenDecision B K V) (s : Store B K V) : Store B K V :=
  EmenderMemory.step (.decayAge t.liveForDecay t.decayRate)
    (t.write.run (EmenderMemory.step (.pop t.popTarget t.popMagnitude) s))

theorem token_step_eq_fold (t : TokenDecision B K V) (s : Store B K V) :
    t.step s = t.ops.foldl (fun x op => step op x) s := by
  simp [TokenDecision.step, TokenDecision.ops, Write.run, List.foldl_append]

def TokenDecision.summary (t : TokenDecision B K V) :
    DiagonalOperator (Fin B × Field K V) :=
  DiagonalOperator.composeAll (t.ops.map diagonal)

theorem token_summary_correct (t : TokenDecision B K V) (s : Store B K V) :
    t.step s = t.summary.toAffine.apply s := by
  rw [token_step_eq_fold]
  calc
    _ = (AffineOperator.composeAll (t.ops.map (fun op => (diagonal op).toAffine))).apply s :=
      memoryOperator_fold t.ops s
    _ = t.summary.toAffine.apply s := by
      simp only [TokenDecision.summary, DiagonalOperator.toAffine_composeAll, List.map_map]
      rfl

instance : MemoryOperator (TokenDecision B K V) (Store B K V) where
  step := TokenDecision.step
  embed := fun t => t.summary.toAffine
  correct := token_summary_correct

theorem token_conditional_scan (ts : List (TokenDecision B K V)) :
    ScanCompatibleRelativeTo TokenDecision.step ts := memoryOperator_scanCompatible ts

theorem token_fold_scan (ts : List (TokenDecision B K V)) (s : Store B K V) :
    ts.foldl (fun x t => t.step x) s =
      (AffineOperator.composeAll (ts.map (fun t => t.summary.toAffine))).apply s :=
  memoryOperator_fold ts s

theorem token_chunked_scan (chunks : List (List (TokenDecision B K V))) (s : Store B K V) :
    chunks.flatten.foldl (fun x t => t.step x) s =
      (AffineOperator.composeAll
        (chunks.map (fun c =>
          AffineOperator.composeAll (c.map (fun t => t.summary.toAffine))))).apply s := by
  rw [token_fold_scan, List.map_flatten, AffineOperator.composeAll_flatten, List.map_map]
  rfl

theorem token_decomposition (t : TokenDecision B K V) (s : Store B K V) :
    t.step s = (fun q => t.summary.scale q * s q) +
      ∑ j ∈ slotSupport t.summary.shift, slotInjection j (fun f => t.summary.shift (j, f)) := by
  rw [token_summary_correct]
  exact diagonal_slot_decomposition t.summary s

theorem token_activity_bound (t : TokenDecision B K V) :
    (slotSupport t.summary.shift).card ≤ B := slotSupport_card_le _

/-- Token-level content injections. POP and DECAY do not modify keys/values. -/
def TokenDecision.contentShift (t : TokenDecision B K V) :
    (Fin B × (Fin K ⊕ Fin V)) → ℝ := fun (i, f) =>
  match f with
  | .inl k => t.summary.shift (i, .key k)
  | .inr v => t.summary.shift (i, .value v)

theorem token_content_support_le_one (t : TokenDecision B K V) :
    (slotSupport t.contentShift).card ≤ 1 := by
  classical
  have heval (q : Fin B × Field K V) : t.summary.shift q = t.step 0 q := by
    rw [token_summary_correct, DiagonalOperator.toAffine_apply]
    simp
  have h : t.contentShift =
      match t.write with
      | .none => 0
      | .append j k v g => contentInjection (Op.append j k v g)
      | .correct j e v => contentInjection (Op.correct j e v)
      | .merge j b v => contentInjection (Op.merge j b v) := by
    funext q
    rcases q with ⟨i, f⟩
    cases f <;> simp only [TokenDecision.contentShift, heval]
    all_goals
      cases hw : t.write <;>
        simp [TokenDecision.step, Write.run, Write.ops, step, hw, contentInjection, injection]
  rw [h]
  cases t.write with
  | none => simp [slotSupport]
  | append j k v g => exact content_injection_support_le_one (.append j k v g)
  | correct j e v => exact content_injection_support_le_one (.correct j e v)
  | merge j b v => exact content_injection_support_le_one (.merge j b v)

/-! The affine state scan does not compute thresholds. The next two definitions
make the contract with architectural Boolean bookkeeping explicit. -/

def TokenDecision.replay (t : TokenDecision B K V) (s : State B K V) : State B K V :=
  ⟨t.step s.data,
    fun i => t.write.live (fun j => s.live j && !t.freedPop j) i && !t.freedDecay i⟩

/-- Fresh death checks with supplied write/pop selections; this does not model
the selector, read softmax, usage counter, or append-gate network. -/
noncomputable def TokenDecision.checked (t : TokenDecision B K V) (threshold : ℝ)
    (s : State B K V) : State B K V :=
  let popped := EmenderMemory.step (.pop t.popTarget t.popMagnitude) s.data
  let freed := freeMask threshold s.live popped
  let live := t.write.live (fun i => s.live i && !freed i)
  let written := t.write.run popped
  let decayed := EmenderMemory.step (.decayAge live t.decayRate) written
  ⟨decayed, fun i => live i && !freeMask threshold live decayed i⟩

/-- The trace must supply the two actual death masks and post-write live mask.
This is a hypothesis about the original run, not a theorem that detaching
its decisions makes them independent of state. -/
def TokenDecision.ValidFor (t : TokenDecision B K V) (threshold : ℝ)
    (s : State B K V) : Prop :=
  t.freedPop = freeMask threshold s.live
    (EmenderMemory.step (.pop t.popTarget t.popMagnitude) s.data) ∧
  t.liveForDecay = t.write.live (fun i => s.live i && !t.freedPop i) ∧
  t.freedDecay = freeMask threshold t.liveForDecay (t.step s.data)

theorem replay_eq_checked (t : TokenDecision B K V) (threshold : ℝ)
    (s : State B K V) (h : t.ValidFor threshold s) :
    t.replay s = t.checked threshold s := by
  rcases h with ⟨hp, hl, hd⟩
  dsimp only [TokenDecision.checked]
  rw [← hp, ← hl]
  change t.replay s = ⟨t.step s.data,
    fun i => t.liveForDecay i && !freeMask threshold t.liveForDecay (t.step s.data) i⟩
  rw [← hd, hl]
  rfl

end EmenderMemory
