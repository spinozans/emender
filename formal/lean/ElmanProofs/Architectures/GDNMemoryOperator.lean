import ElmanProofs.Architectures.MemoryOperator
import Mathlib.Data.Matrix.Bilinear

/-! # GDN2 as an instance of the same semantic operator interface -/

namespace GDNMemoryOperator

open MemoryAlgebra CompositionalDepthDuality M2RNNComparison

variable {K V : ℕ}

def embed (p : GdnStepParams K V) : AffineOperator (MatState K V) where
  linear := mulLeftLinearMap (Fin V) ℝ
    (SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay)
  injection := outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v)

theorem embed_correct (p : GdnStepParams K V) (s : MatState K V) :
    gdnStep p s = (embed p).apply s := gdnStep_eq_affine p s

instance : MemoryOperator (GdnStepParams K V) (MatState K V) where
  step := gdnStep
  embed := embed
  correct := embed_correct

theorem gdn2_conditional_scan (schedule : List (GdnStepParams K V)) :
    ScanCompatibleRelativeTo gdnStep schedule := memoryOperator_scanCompatible schedule

theorem gdn2_fold_scan (schedule : List (GdnStepParams K V)) (s : MatState K V) :
    schedule.foldl (fun x p => gdnStep p x) s =
      (AffineOperator.composeAll (schedule.map embed)).apply s := memoryOperator_fold schedule s

end GDNMemoryOperator
