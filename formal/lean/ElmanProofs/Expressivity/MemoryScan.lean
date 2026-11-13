import ElmanProofs.Architectures.MemoryOperator

/-! # Ordered scan trees and explicit suffix-product injection propagation -/

namespace MemoryAlgebra

open BigOperators

variable {S : Type*} [AddCommGroup S] [Module ℝ S]

namespace AffineOperator

/-- Index form: injection i is acted on by the linear product strictly after i.
The default operator is irrelevant because the sum only ranges over valid indices. -/
theorem injection_eq_suffix_sum (ops : List (AffineOperator S)) :
    (composeAll ops).injection =
      ∑ i ∈ Finset.range ops.length,
        linearProduct (ops.drop (i + 1)) ((ops[i]?.getD 1).injection) := by
  induction ops with
  | nil => simp [composeAll]; rfl
  | cons a as ih =>
    simp only [composeAll, injection_mul, composeAll_linear, List.length_cons,
      Finset.sum_range_succ']
    simp only [List.getElem?_cons_succ, List.drop_succ_cons, List.getElem?_cons_zero,
      Option.getD_some, List.drop_zero]
    rw [← ih, add_comm]

theorem prefix_product_injection_formula (ops : List (AffineOperator S)) (n : ℕ) (s : S) :
    (ops.take n).foldl (fun x a => a.apply x) s =
      linearProduct (ops.take n) s +
        ∑ i ∈ Finset.range (ops.take n).length,
          linearProduct ((ops.take n).drop (i + 1)) (((ops.take n)[i]?.getD 1).injection) := by
  rw [foldl_eq_composeAll, apply, composeAll_linear, injection_eq_suffix_sum]

end AffineOperator

/-- Any ordered reduction tree, including balanced trees and empty chunks. -/
inductive ScanTree (S : Type*) [AddCommGroup S] [Module ℝ S] where
  | empty
  | leaf (op : AffineOperator S)
  | node (left right : ScanTree S)

namespace ScanTree

def leaves : ScanTree S → List (AffineOperator S)
  | .empty => []
  | .leaf a => [a]
  | .node l r => l.leaves ++ r.leaves

def summary : ScanTree S → AffineOperator S
  | .empty => 1
  | .leaf a => a
  | .node l r => l.summary * r.summary

theorem summary_eq_composeAll (tree : ScanTree S) :
    tree.summary = AffineOperator.composeAll tree.leaves := by
  induction tree with
  | empty => rfl
  | leaf a => simp [summary, leaves, AffineOperator.composeAll]
  | node l r hl hr => simp [summary, leaves, AffineOperator.composeAll_append, hl, hr]

/-- Parallel reduction may change parenthesization, but never temporal order. -/
theorem ordered_tree_equiv (tree : ScanTree S) (s : S) :
    tree.summary.apply s = tree.leaves.foldl (fun x a => a.apply x) s := by
  rw [summary_eq_composeAll, AffineOperator.foldl_eq_composeAll]

end ScanTree

end MemoryAlgebra
