import ElmanProofs.Expressivity.CompositionalDepthDuality
import Mathlib.Algebra.Module.LinearMap.End

/-!
# Update-family-parametric memory operators

An operator is a linear endomorphism and an injection, not just a function
composition label. Multiplication is chronological: `a * b` applies `a` first.
The elementary update family need not itself be closed under composition.
Its embeddings live in a monoid of affine summaries. Non-affine families do
not automatically have a `MemoryOperator` instance.
-/

namespace MemoryAlgebra

variable (S : Type*) [AddCommGroup S] [Module ℝ S]

/-- A state-affine map, represented by its linear part and constant injection. -/
structure AffineOperator where
  linear : S →ₗ[ℝ] S
  injection : S

namespace AffineOperator

variable {S}

def apply (a : AffineOperator S) (s : S) : S := a.linear s + a.injection

@[ext] theorem ext {a b : AffineOperator S}
    (hl : a.linear = b.linear) (hi : a.injection = b.injection) : a = b := by
  cases a
  cases b
  simp_all

def identity : AffineOperator S := ⟨LinearMap.id, 0⟩

/-- Composition in time order; this is the affine semidirect-product law. -/
def thenOp (a b : AffineOperator S) : AffineOperator S :=
  ⟨b.linear.comp a.linear, b.linear a.injection + b.injection⟩

theorem thenOp_assoc (a b c : AffineOperator S) :
    thenOp (thenOp a b) c = thenOp a (thenOp b c) := by
  apply ext
  · rfl
  · simp [thenOp, map_add, add_assoc]

instance : Monoid (AffineOperator S) where
  mul := thenOp
  one := identity
  mul_assoc := thenOp_assoc
  one_mul a := by
    change thenOp identity a = a
    apply ext
    · ext s; rfl
    · change a.linear 0 + a.injection = a.injection
      simp
  mul_one a := by
    change thenOp a identity = a
    apply ext
    · ext s; rfl
    · exact add_zero _

@[simp] theorem apply_one (s : S) : (1 : AffineOperator S).apply s = s := by
  change s + 0 = s
  exact add_zero _

@[simp] theorem apply_mul (a b : AffineOperator S) (s : S) :
    (a * b).apply s = b.apply (a.apply s) := by
  simp [HMul.hMul, Mul.mul, thenOp, apply, map_add, add_assoc]

@[simp] theorem linear_mul (a b : AffineOperator S) :
    (a * b).linear = b.linear.comp a.linear := rfl

@[simp] theorem injection_mul (a b : AffineOperator S) :
    (a * b).injection = b.linear a.injection + b.injection := rfl

/-- A necessary affine identity, useful for proving obstructions. -/
theorem affine_identity (a : AffineOperator S) (x y : S) :
    a.apply (x + y) + a.apply 0 = a.apply x + a.apply y := by
  simp only [apply, map_add, map_zero, zero_add]
  abel

/-- Summary of a chronological list. -/
def composeAll : List (AffineOperator S) → AffineOperator S
  | [] => 1
  | a :: as => a * composeAll as

theorem foldl_eq_composeAll (ops : List (AffineOperator S)) (s : S) :
    ops.foldl (fun x a => a.apply x) s = (composeAll ops).apply s := by
  induction ops generalizing s with
  | nil => simp [composeAll]
  | cons a as ih => simpa [composeAll] using ih (a.apply s)

/-- Connection to the existing function-composition theorem; not a scan claim
on its own. The algebra above supplies the closed-size summaries. -/
theorem composeAll_eq_function_composition (ops : List (AffineOperator S)) (s : S) :
    (composeAll ops).apply s =
      CompositionalDepthDuality.composeAll (ops.map apply) s := by
  rw [← foldl_eq_composeAll]
  exact CompositionalDepthDuality.foldl_eq_composeAll (fun x a => a.apply x) ops s

theorem composeAll_append (xs ys : List (AffineOperator S)) :
    composeAll (xs ++ ys) = composeAll xs * composeAll ys := by
  induction xs with
  | nil => simp [composeAll]
  | cons a as ih => simp only [List.cons_append, composeAll, ih, mul_assoc]

theorem composeAll_flatten (chunks : List (List (AffineOperator S))) :
    composeAll chunks.flatten = composeAll (chunks.map composeAll) := by
  induction chunks with
  | nil => rfl
  | cons c cs ih => simp [List.flatten_cons, composeAll_append, ih, composeAll]

/-- Intra-chunk sequential execution equals execution of the chunk summaries. -/
theorem chunked_fold_equiv (chunks : List (List (AffineOperator S))) (s : S) :
    chunks.foldl (fun x c => c.foldl (fun y a => a.apply y) x) s =
      (composeAll (chunks.map composeAll)).apply s := by
  induction chunks generalizing s with
  | nil => simp [composeAll]
  | cons c cs ih =>
    simpa [composeAll, foldl_eq_composeAll] using
      ih (c.foldl (fun y a => a.apply y) s)

/-- Every prefix, including empty and oversized prefixes, is a summary. -/
theorem prefix_scan_equiv (ops : List (AffineOperator S)) (n : ℕ) (s : S) :
    (ops.take n).foldl (fun x a => a.apply x) s =
      (composeAll (ops.take n)).apply s := foldl_eq_composeAll _ _

/-- Prefix inside a chunk, after any number of complete preceding chunks. -/
theorem chunk_prefix_equiv (before : List (List (AffineOperator S)))
    (current : List (AffineOperator S)) (n : ℕ) (s : S) :
    (before.flatten ++ current.take n).foldl (fun x a => a.apply x) s =
      (composeAll (current.take n)).apply
        ((composeAll (before.map composeAll)).apply s) := by
  rw [foldl_eq_composeAll, composeAll_append, composeAll_flatten, apply_mul]

/-- Product of linear parts: the last temporal map acts on the left. -/
def linearProduct : List (AffineOperator S) → (S →ₗ[ℝ] S)
  | [] => LinearMap.id
  | a :: as => (linearProduct as).comp a.linear

/-- Each injection is propagated through all subsequent linear parts.
This recursion is the ordered suffix-product / sum formula, with no inverses. -/
def propagatedInjections : List (AffineOperator S) → List S
  | [] => []
  | a :: as => linearProduct as a.injection :: propagatedInjections as

theorem composeAll_linear (ops : List (AffineOperator S)) :
    (composeAll ops).linear = linearProduct ops := by
  induction ops with
  | nil => rfl
  | cons a as ih => simp only [composeAll, linear_mul, ih, linearProduct]

theorem composeAll_injection (ops : List (AffineOperator S)) :
    (composeAll ops).injection = (propagatedInjections ops).sum := by
  induction ops with
  | nil => rfl
  | cons a as ih =>
    simp only [composeAll, injection_mul, composeAll_linear, ih,
      propagatedInjections, List.sum_cons]

theorem injection_propagation (ops : List (AffineOperator S)) (s : S) :
    ops.foldl (fun x a => a.apply x) s =
      linearProduct ops s + (propagatedInjections ops).sum := by
  rw [foldl_eq_composeAll, apply, composeAll_linear, composeAll_injection]

theorem prefix_injection_propagation (ops : List (AffineOperator S))
    (n : ℕ) (s : S) :
    (ops.take n).foldl (fun x a => a.apply x) s =
      linearProduct (ops.take n) s + (propagatedInjections (ops.take n)).sum :=
  injection_propagation _ _

end AffineOperator

/-- A semantic interface for an elementary update family. `correct` is an
equation on every state, not a Boolean architecture label. -/
class MemoryOperator (Op : Type*) (S : outParam Type*)
    [AddCommGroup S] [Module ℝ S] where
  step : Op → S → S
  embed : Op → AffineOperator S
  correct : ∀ op s, step op s = (embed op).apply s

variable {S}

/-- Scan compatibility relative to a supplied, fixed decision schedule.
Every scheduled step must have a state-independent affine representation. -/
def ScanCompatibleRelativeTo {D : Type*} (step : D → S → S) (schedule : List D) : Prop :=
  ∀ d ∈ schedule, ∃ a : AffineOperator S, ∀ s, step d s = a.apply s

theorem memoryOperator_scanCompatible {Op : Type*} [MemoryOperator Op S]
    (schedule : List Op) :
    ScanCompatibleRelativeTo (MemoryOperator.step (S := S)) schedule := by
  intro d _
  exact ⟨MemoryOperator.embed d, MemoryOperator.correct d⟩

theorem conditional_scan_exists {D : Type*} (step : D → S → S) (schedule : List D)
    (h : ScanCompatibleRelativeTo step schedule) :
    ∃ a : AffineOperator S, ∀ s, schedule.foldl (fun x d => step d x) s = a.apply s := by
  induction schedule with
  | nil => exact ⟨1, fun s => (AffineOperator.apply_one s).symm⟩
  | cons d ds ih =>
    obtain ⟨a, ha⟩ := h d (by simp)
    obtain ⟨b, hb⟩ := ih (fun e he => h e (by simp [he]))
    exact ⟨a * b, fun s => by simp [List.foldl_cons, hb, ha]⟩

theorem memoryOperator_fold {Op : Type*} [MemoryOperator Op S]
    (ops : List Op) (s : S) :
    ops.foldl (fun x op => MemoryOperator.step op x) s =
      (AffineOperator.composeAll (ops.map MemoryOperator.embed)).apply s := by
  induction ops generalizing s with
  | nil => simp [AffineOperator.composeAll]
  | cons op ops ih =>
    simpa [AffineOperator.composeAll, MemoryOperator.correct] using
      ih (MemoryOperator.step op s)

/-! ## P1: independent programs, parameterized over arbitrary update families

This property does not require affinity or scan compatibility. Inputs to the
programs are supplied; inter-layer input generation is not asserted independent.
-/

def IsMultiProgrammedRecurrentStack {P I X : Type*} [Fintype P]
    (family : P → I → X → X) (update : I → (P → X) → (P → X)) : Prop :=
  2 ≤ Fintype.card P ∧ ∀ input state p, update input state p = family p input (state p)

def independentUpdate {P I X : Type*} (family : P → I → X → X)
    (input : I) (state : P → X) : P → X := fun p => family p input (state p)

theorem every_update_family_multiProgrammed {P I X : Type*} [Fintype P]
    (family : P → I → X → X) (many : 2 ≤ Fintype.card P) :
    IsMultiProgrammedRecurrentStack family (independentUpdate family) :=
  ⟨many, fun _ _ _ => rfl⟩

theorem independent_fold_projection {P I X : Type*} (family : P → I → X → X)
    (inputs : List I) (state : P → X) (p : P) :
    inputs.foldl (fun s i => independentUpdate family i s) state p =
      inputs.foldl (fun s i => family p i s) (state p) := by
  induction inputs generalizing state with
  | nil => rfl
  | cons i inputs ih => exact ih (independentUpdate family i state)

end MemoryAlgebra
