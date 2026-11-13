import ElmanProofs.Architectures.MemoryOperator
import Mathlib.LinearAlgebra.Finsupp.Defs

/-!
# Arbitrary operators have a free-vector-space embedding

This is an algebraic existence result, not a compact scan implementation.
The basis is indexed by complete states, not by the state's scalar coordinates.
For finite X its dimension is |X|; for infinite X it is not finite-dimensional
in general. The compact Emender instance instead operates on the original store.
-/

namespace MemoryAlgebra.LiftedMemory

noncomputable section

variable {X : Type*}

def encode (x : X) : X →₀ ℝ := Finsupp.single x 1

/-- Linear push-forward of a finitely supported state distribution. -/
def lift (f : X → X) : AffineOperator (X →₀ ℝ) :=
  ⟨Finsupp.lmapDomain ℝ ℝ f, 0⟩

theorem lift_correct (f : X → X) (x : X) :
    (lift f).apply (encode x) = encode (f x) := by
  simp [lift, encode, AffineOperator.apply, Finsupp.lmapDomain_apply]

theorem encode_injective : Function.Injective (encode (X := X)) := by
  intro x y h
  by_contra hne
  have h' := congrArg (fun v : X →₀ ℝ => v x) h
  simp [encode, hne] at h'

/-- Push-forward respects chronological composition. -/
theorem lift_comp (f g : X → X) : lift (g ∘ f) = lift f * lift g := by
  apply AffineOperator.ext
  · exact Finsupp.lmapDomain_comp ℝ ℝ f g
  · simp [lift]

/-- Any update family, including state-dependent branches, embeds on the lifted
state space. This asserts no useful representation-size or work bound. -/
theorem arbitrary_family_scan {Op : Type*} (step : Op → X → X)
    (ops : List Op) (x : X) :
    (AffineOperator.composeAll (ops.map (fun op => lift (step op)))).apply (encode x) =
      encode (ops.foldl (fun s op => step op s) x) := by
  induction ops generalizing x with
  | nil => simp [AffineOperator.composeAll]
  | cons op ops ih =>
    simpa [AffineOperator.composeAll, lift_correct] using ih (step op x)

end

end MemoryAlgebra.LiftedMemory
