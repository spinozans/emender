/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Nat.Basic
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Architectures.M2RNNComparison

/-!
# Resource-Bounded Formalism for Recurrent Architectures

This module records the comparison frame we need for E88/NDM, M2RNN, Gated
DeltaNet, and Mamba2.

The point is not to argue over the broad phrase "matrix state". Matrix state is
common. The useful questions are:

* Does temporal nonlinearity occur inside the recurrent state path?
* Is the stack pure recurrent or hybridized with attention/linear layers?
* What memory semantics does the update implement?
* How many independent recurrent programs are exposed to the machine?
* Are two architectures the same one-step update family, or merely in the same
  broad nonlinear temporal class?

The definitions here are deliberately lightweight. They are intended as a
stable vocabulary for later theorems about one-step transition separations,
resource-bounded expressivity, and empirical protocol obligations.
-/

namespace RecurrentResourceFormalism

/-! ## Structural Axes -/

/-- Shape of persistent recurrent state. -/
inductive StateGeometry where
  | vector
  | matrix
  | externalTape
  | attentionKV
  deriving Repr, DecidableEq

/-- Where, if anywhere, the temporal nonlinearity is applied. -/
inductive TemporalNonlinearity where
  | none
  | candidateOnly
  | fullState
  | gatedFullState
  deriving Repr, DecidableEq

/-- Side on which the previous state is transformed before the update. -/
inductive TransitionSide where
  | none
  | left
  | right
  | bilateral
  | elementwise
  deriving Repr, DecidableEq

/-- Whether the state transition is fixed, input-dependent, or state-dependent. -/
inductive TransitionControl where
  | fixedLearned
  | inputDependent
  | stateDependent
  | inputAndStateDependent
  deriving Repr, DecidableEq

/-- How new information enters memory. -/
inductive WriteRule where
  | none
  | rawOuterProduct
  | selectiveEraseWrite
  | deltaCorrecting
  | attentionCopy
  deriving Repr, DecidableEq

/-- Where linear carry/highway paths sit relative to the nonlinear candidate. -/
inductive CarryPlacement where
  | none
  | insideNonlinearity
  | outsideNonlinearity
  | residualReadout
  deriving Repr, DecidableEq

/-- Operational memory semantics, distinct from raw tensor shape. -/
inductive MemorySemantics where
  | fadingAccumulator
  | rawAssociativeTable
  | errorCorrectingAssociativeTable
  | latchAttractor
  | copyBuffer
  deriving Repr, DecidableEq

/-- Hardware-level execution model for the recurrent part. -/
inductive ImplementationMode where
  | parallelScan
  | sequentialSingleProgram
  | sequentialManyProgram
  | hybrid
  | quadraticAttention
  deriving Repr, DecidableEq

/-- Boolean version of "has recurrent temporal nonlinearity". -/
def hasTemporalNonlinearity : TemporalNonlinearity → Bool
  | .none => false
  | .candidateOnly => true
  | .fullState => true
  | .gatedFullState => true

/-- Boolean version of "uses delta/error-correcting memory writes". -/
def hasDeltaCorrectingWrite : WriteRule → Bool
  | .deltaCorrecting => true
  | _ => false

/-- Boolean version of the broader GDN/E88 delta-family write axis. GDN uses a
linear selective erase/write delta rule; E88/NDM uses a nonlinear
delta-correcting version. -/
def hasDeltaStyleWrite : WriteRule → Bool
  | .selectiveEraseWrite => true
  | .deltaCorrecting => true
  | _ => false

/-- Boolean version of "uses matrix-valued persistent state". -/
def hasMatrixState : StateGeometry → Bool
  | .matrix => true
  | _ => false

/-- Architecture feature signature for resource-bounded comparisons. -/
structure ArchitectureSignature where
  name : String
  stateGeometry : StateGeometry
  temporalNonlinearity : TemporalNonlinearity
  transitionSide : TransitionSide
  transitionControl : TransitionControl
  writeRule : WriteRule
  carryPlacement : CarryPlacement
  memorySemantics : MemorySemantics
  implementationMode : ImplementationMode
  totalLayers : Nat
  nonlinearTemporalLayers : Nat
  headsPerLayer : Nat
  stateScalarsPerHead : Nat
  pureRecurrentStack : Bool
  scanCompatible : Bool
  deriving Repr, DecidableEq

/-! ## Resource Measures -/

/-- Number of independent recurrent programs available per token and batch item. -/
def programsPerBatchToken (a : ArchitectureSignature) (batch : Nat) : Nat :=
  a.totalLayers * a.headsPerLayer * batch

/-- Total recurrent state scalars per layer, ignoring batch. -/
def stateScalarsPerLayer (a : ArchitectureSignature) : Nat :=
  a.headsPerLayer * a.stateScalarsPerHead

/-- Total recurrent state scalars in the stack, ignoring batch. -/
def stateScalarsPerStack (a : ArchitectureSignature) : Nat :=
  a.totalLayers * stateScalarsPerLayer a

/-- Sequential recurrent steps exposed by a full training chunk. -/
def sequentialProgramSteps (a : ArchitectureSignature) (batch seqLen : Nat) : Nat :=
  programsPerBatchToken a batch * seqLen

/-- All recurrent layers are temporally nonlinear. -/
def allLayersTemporallyNonlinear (a : ArchitectureSignature) : Bool :=
  a.nonlinearTemporalLayers == a.totalLayers

/-- Pure recurrent stack with temporal nonlinearity in every layer. -/
def isPureNonlinearRecurrentStack (a : ArchitectureSignature) : Bool :=
  a.pureRecurrentStack &&
  allLayersTemporallyNonlinear a &&
  hasTemporalNonlinearity a.temporalNonlinearity

/-- The resource-bounded comparison mode: not broad computability, but capability
under fixed wallclock/memory/precision/training protocol. -/
inductive ComparisonMode where
  | broadComputabilityClass
  | oneStepTransitionFamily
  | resourceBoundedCapability
  | trainabilityAndStability
  | hardwareUtilization
  deriving Repr, DecidableEq

/-! ## Canonical Signatures -/

/-- NDM: Nonlinear Delta Memory.

This is the paper-facing model family: a pure nonlinear recurrent stack with
matrix state, input-dependent delta correction, and many independent recurrent
programs per token. E88 is the current production implementation lineage of
this family.
-/
def ndm (layers heads nState : Nat) : ArchitectureSignature where
  name := "E88/NDM"
  stateGeometry := .matrix
  temporalNonlinearity := .fullState
  transitionSide := .left
  transitionControl := .inputDependent
  writeRule := .deltaCorrecting
  carryPlacement := .insideNonlinearity
  memorySemantics := .errorCorrectingAssociativeTable
  implementationMode := .sequentialManyProgram
  totalLayers := layers
  nonlinearTemporalLayers := layers
  headsPerLayer := heads
  stateScalarsPerHead := nState * nState
  pureRecurrentStack := true
  scanCompatible := false

/-- Backward-compatible name for the E88 implementation lineage of NDM. -/
def e88NDM (layers heads nState : Nat) : ArchitectureSignature :=
  ndm layers heads nState

/-- Homogeneous/pure M2RNN: nonlinear matrix-state recurrence, but raw writes and
a fixed learned right transition. -/
def m2rnnPure (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "M2RNN pure"
  stateGeometry := .matrix
  temporalNonlinearity := .candidateOnly
  transitionSide := .right
  transitionControl := .fixedLearned
  writeRule := .rawOuterProduct
  carryPlacement := .outsideNonlinearity
  memorySemantics := .rawAssociativeTable
  implementationMode := .sequentialManyProgram
  totalLayers := layers
  nonlinearTemporalLayers := layers
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := false

/-- Hybrid M2RNN/attention or M2RNN/GDN stack. It may contain nonlinear recurrent
layers, but the stack is not pure recurrent. -/
def m2rnnHybrid
    (layers nonlinearLayers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "M2RNN hybrid"
  stateGeometry := .matrix
  temporalNonlinearity := .candidateOnly
  transitionSide := .right
  transitionControl := .fixedLearned
  writeRule := .rawOuterProduct
  carryPlacement := .outsideNonlinearity
  memorySemantics := .rawAssociativeTable
  implementationMode := .hybrid
  totalLayers := layers
  nonlinearTemporalLayers := nonlinearLayers
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := false
  scanCompatible := false

/-- Gated DeltaNet / FLA-GDN: matrix state and delta-style associative memory,
but linear/affine in the recurrent state for fixed inputs. -/
def gatedDeltaNet (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "Gated DeltaNet"
  stateGeometry := .matrix
  temporalNonlinearity := .none
  transitionSide := .left
  transitionControl := .inputDependent
  writeRule := .selectiveEraseWrite
  carryPlacement := .none
  memorySemantics := .rawAssociativeTable
  implementationMode := .parallelScan
  totalLayers := layers
  nonlinearTemporalLayers := 0
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := true

/-- Mamba2-like selective SSM: vector/diagonal state, input-dependent but linear
in the temporal state. -/
def mamba2SSM (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "Mamba2"
  stateGeometry := .vector
  temporalNonlinearity := .none
  transitionSide := .elementwise
  transitionControl := .inputDependent
  writeRule := .none
  carryPlacement := .none
  memorySemantics := .fadingAccumulator
  implementationMode := .parallelScan
  totalLayers := layers
  nonlinearTemporalLayers := 0
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := true

/-- Concrete current production NDM geometry: 12 layers, 370 heads per layer,
and 32×32 state per head. -/
def ndm_1p27B : ArchitectureSignature :=
  ndm 12 370 32

/-- Backward-compatible name for the current E88 production geometry. -/
def e88NDM_1p27B : ArchitectureSignature :=
  ndm_1p27B

/-! ## Basic Theorems -/

theorem ndm_1p27B_is_pure_nonlinear_recurrent_stack :
    isPureNonlinearRecurrentStack ndm_1p27B = true := by
  rfl

theorem ndm_1p27B_has_delta_memory :
    hasDeltaCorrectingWrite ndm_1p27B.writeRule = true := by
  rfl

theorem ndm_1p27B_has_matrix_state :
    hasMatrixState ndm_1p27B.stateGeometry = true := by
  rfl

theorem ndm_1p27B_programs_per_batch_token (batch : Nat) :
    programsPerBatchToken ndm_1p27B batch = 12 * 370 * batch := by
  rfl

theorem ndm_1p27B_programs_per_batch_token_bs5 :
    programsPerBatchToken ndm_1p27B 5 = 22200 := by
  rfl

theorem ndm_1p27B_state_scalars_per_layer :
    stateScalarsPerLayer ndm_1p27B = 370 * (32 * 32) := by
  rfl

theorem e88NDM_1p27B_is_pure_nonlinear_recurrent_stack :
    isPureNonlinearRecurrentStack e88NDM_1p27B = true := by
  rfl

theorem e88NDM_1p27B_has_delta_memory :
    hasDeltaCorrectingWrite e88NDM_1p27B.writeRule = true := by
  rfl

theorem e88NDM_1p27B_has_matrix_state :
    hasMatrixState e88NDM_1p27B.stateGeometry = true := by
  rfl

theorem e88NDM_1p27B_programs_per_batch_token (batch : Nat) :
    programsPerBatchToken e88NDM_1p27B batch = 12 * 370 * batch := by
  rfl

theorem e88NDM_1p27B_programs_per_batch_token_bs5 :
    programsPerBatchToken e88NDM_1p27B 5 = 22200 := by
  rfl

theorem e88NDM_1p27B_state_scalars_per_layer :
    stateScalarsPerLayer e88NDM_1p27B = 370 * (32 * 32) := by
  rfl

theorem pure_m2rnn_is_nonlinear_matrix_recurrent
    (layers heads state : Nat) :
    hasMatrixState (m2rnnPure layers heads state).stateGeometry = true ∧
    hasTemporalNonlinearity (m2rnnPure layers heads state).temporalNonlinearity = true := by
  constructor <;> rfl

theorem pure_m2rnn_is_not_delta_correcting
    (layers heads state : Nat) :
    hasDeltaCorrectingWrite (m2rnnPure layers heads state).writeRule = false := by
  rfl

theorem hybrid_m2rnn_is_not_pure_recurrent_stack
    (layers nonlinearLayers heads state : Nat) :
    (m2rnnHybrid layers nonlinearLayers heads state).pureRecurrentStack = false := by
  rfl

theorem gated_delta_net_has_matrix_state_but_no_temporal_nonlinearity
    (layers heads state : Nat) :
    hasMatrixState (gatedDeltaNet layers heads state).stateGeometry = true ∧
    hasTemporalNonlinearity (gatedDeltaNet layers heads state).temporalNonlinearity = false := by
  constructor <;> rfl

theorem e88_and_gdn_share_delta_style_write
    (layers heads state : Nat) :
    hasDeltaStyleWrite e88NDM_1p27B.writeRule = true ∧
    hasDeltaStyleWrite (gatedDeltaNet layers heads state).writeRule = true := by
  constructor <;> rfl

theorem ndm_and_gdn_share_delta_style_write
    (layers heads state : Nat) :
    hasDeltaStyleWrite ndm_1p27B.writeRule = true ∧
    hasDeltaStyleWrite (gatedDeltaNet layers heads state).writeRule = true := by
  constructor <;> rfl

theorem e88_and_gdn_split_on_temporal_nonlinearity
    (layers heads state : Nat) :
    hasTemporalNonlinearity e88NDM_1p27B.temporalNonlinearity = true ∧
    hasTemporalNonlinearity (gatedDeltaNet layers heads state).temporalNonlinearity = false := by
  constructor <;> rfl

theorem ndm_and_gdn_split_on_temporal_nonlinearity
    (layers heads state : Nat) :
    hasTemporalNonlinearity ndm_1p27B.temporalNonlinearity = true ∧
    hasTemporalNonlinearity (gatedDeltaNet layers heads state).temporalNonlinearity = false := by
  constructor <;> rfl

theorem mamba2_has_no_temporal_nonlinearity
    (layers heads state : Nat) :
    hasTemporalNonlinearity (mamba2SSM layers heads state).temporalNonlinearity = false := by
  rfl

/-- E88/NDM and homogeneous M2RNN share the broad nonlinear matrix-state family. -/
theorem e88_and_m2rnn_share_broad_nonlinear_matrix_family
    (m2Layers m2Heads m2State : Nat) :
    hasMatrixState e88NDM_1p27B.stateGeometry = true ∧
    hasTemporalNonlinearity e88NDM_1p27B.temporalNonlinearity = true ∧
    hasMatrixState (m2rnnPure m2Layers m2Heads m2State).stateGeometry = true ∧
    hasTemporalNonlinearity (m2rnnPure m2Layers m2Heads m2State).temporalNonlinearity = true := by
  constructor
  · rfl
  constructor
  · rfl
  constructor <;> rfl

/-- NDM and homogeneous M2RNN share the broad nonlinear matrix-state family. -/
theorem ndm_and_m2rnn_share_broad_nonlinear_matrix_family
    (m2Layers m2Heads m2State : Nat) :
    hasMatrixState ndm_1p27B.stateGeometry = true ∧
    hasTemporalNonlinearity ndm_1p27B.temporalNonlinearity = true ∧
    hasMatrixState (m2rnnPure m2Layers m2Heads m2State).stateGeometry = true ∧
    hasTemporalNonlinearity (m2rnnPure m2Layers m2Heads m2State).temporalNonlinearity = true := by
  constructor
  · rfl
  constructor
  · rfl
  constructor <;> rfl

/-- But E88/NDM and M2RNN are not the same one-step transition family. -/
theorem e88_and_m2rnn_differ_as_one_step_transition_families
    (m2Layers m2Heads m2State : Nat) :
    e88NDM_1p27B.transitionSide ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionSide ∧
    e88NDM_1p27B.writeRule ≠
      (m2rnnPure m2Layers m2Heads m2State).writeRule ∧
    e88NDM_1p27B.transitionControl ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionControl := by
  constructor
  · intro h
    cases h
  constructor
  · intro h
    cases h
  · intro h
    cases h

/-- NDM and M2RNN are not the same one-step transition family. -/
theorem ndm_and_m2rnn_differ_as_one_step_transition_families
    (m2Layers m2Heads m2State : Nat) :
    ndm_1p27B.transitionSide ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionSide ∧
    ndm_1p27B.writeRule ≠
      (m2rnnPure m2Layers m2Heads m2State).writeRule ∧
    ndm_1p27B.transitionControl ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionControl := by
  constructor
  · intro h
    cases h
  constructor
  · intro h
    cases h
  · intro h
    cases h

/-- The M2RNNComparison scaffold and this resource formalism agree on the delta
write axis for E88. -/
theorem agrees_with_m2rnn_comparison_on_e88_delta_axis :
    M2RNNComparison.e88Features.deltaWrite = true ∧
    hasDeltaCorrectingWrite e88NDM_1p27B.writeRule = true := by
  constructor <;> rfl

/-- The M2RNNComparison scaffold and this resource formalism agree that M2RNN is
not a delta-write architecture. -/
theorem agrees_with_m2rnn_comparison_on_m2rnn_raw_write_axis :
    M2RNNComparison.m2rnnFeatures.deltaWrite = false ∧
    hasDeltaCorrectingWrite (m2rnnPure 1 1 1).writeRule = false := by
  constructor <;> rfl

/-! ## Concrete One-Step Transition Separation -/

abbrev TwoVec := M2RNNComparison.Vec 2
abbrev TwoMat := Matrix (Fin 2) (Fin 2) Real

/-- First basis key in a two-dimensional key space. -/
def key0 : TwoVec :=
  fun i => if i = 0 then 1 else 0

/-- Second basis key in a two-dimensional key space. -/
def key1 : TwoVec :=
  fun i => if i = 1 then 1 else 0

/-- E88's expanded delta transition is genuinely key-dependent: two different
keys induce two different left-transition matrices.

This is the concrete core of the one-step transition-family separation. M2RNN's
learned right transition `W` is fixed across keys inside a layer; E88's delta
rule induces `lambda I - k k^T`, which changes with the current key. -/
theorem e88_two_keys_induce_distinct_left_transitions :
    M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 ≠
      M2RNNComparison.e88DeltaTransition (K := 2) 1 key1 := by
  intro h
  have h00 := congrArg (fun M : TwoMat => M 0 0) h
  simp [M2RNNComparison.e88DeltaTransition, M2RNNComparison.outerKV, key0, key1] at h00

/-- No fixed two-dimensional transition matrix can equal E88's key-dependent
transition for both basis keys. -/
theorem no_fixed_transition_matches_e88_two_basis_keys
    (A : TwoMat) :
    ¬ (A = M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 ∧
       A = M2RNNComparison.e88DeltaTransition (K := 2) 1 key1) := by
  intro h
  exact e88_two_keys_induce_distinct_left_transitions (h.1.symm.trans h.2)

/-- M2RNN preactivation in the 2×2 witness setting:
`H W + k vᵀ`. -/
def m2rnnPreactivation2
    (W H : TwoMat) (k v : TwoVec) : TwoMat :=
  H * W + M2RNNComparison.outerKV k v

/-- E88/NDM expanded delta preactivation in the 2×2 witness setting:
`(I - k kᵀ) H + k vᵀ`. -/
def e88DeltaPreactivation2
    (H : TwoMat) (k v : TwoVec) : TwoMat :=
  M2RNNComparison.e88DeltaTransition (K := 2) 1 k * H +
    M2RNNComparison.outerKV k v

/-- No fixed M2RNN right-transition matrix can make the M2RNN preactivation
family match E88's key-dependent delta preactivation for two basis keys.

This is the one-step transition-family version of the separation. M2RNN's
candidate path has a fixed right transition `W`; E88/NDM's expanded delta rule
has an input-dependent left transition `I - k kᵀ`. If the two families matched
for all states and values at both `key0` and `key1`, then testing on `H = I`
and `v = 0` would force the same fixed `W` to equal two distinct E88
transitions. -/
theorem no_fixed_m2rnn_preactivation_matches_e88_two_basis_keys
    (W : TwoMat) :
    ¬ ((∀ H v,
          m2rnnPreactivation2 W H key0 v =
            e88DeltaPreactivation2 H key0 v) ∧
       (∀ H v,
          m2rnnPreactivation2 W H key1 v =
            e88DeltaPreactivation2 H key1 v)) := by
  intro h
  have hw0 :
      W = M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 := by
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.outerKV] using h.1 (1 : TwoMat) (0 : TwoVec)
  have hw1 :
      W = M2RNNComparison.e88DeltaTransition (K := 2) 1 key1 := by
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.outerKV] using h.2 (1 : TwoMat) (0 : TwoVec)
  exact no_fixed_transition_matches_e88_two_basis_keys W ⟨hw0, hw1⟩

/-- The same fixed-transition separation lifts through the `tanh` candidate
path. Since `tanh` is injective over reals, uniformly matching the M2RNN
candidate `tanh(H W + k vᵀ)` to the E88 candidate
`tanh((I - k kᵀ) H + k vᵀ)` would imply equality of the preactivations, which
the previous theorem rules out for two basis keys.

This still isolates the candidate path. The external M2RNN forget interpolation
is a further resource axis handled by later theorems. -/
theorem no_fixed_m2rnn_candidate_matches_e88_two_basis_keys
    (W : TwoMat) :
    ¬ ((∀ H v,
          M2RNNComparison.m2rnnCandidate W H key0 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key0 v) ∧
       (∀ H v,
          M2RNNComparison.m2rnnCandidate W H key1 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key1 v)) := by
  intro h
  apply no_fixed_m2rnn_preactivation_matches_e88_two_basis_keys W
  constructor
  · intro H v
    ext i j
    have hij := congrArg (fun M : TwoMat => M i j) (h.1 H v)
    have hpre := Activation.tanh_injective hij
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.m2rnnCandidate, M2RNNComparison.e88DeltaUpdateExpanded,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap] using hpre
  · intro H v
    ext i j
    have hij := congrArg (fun M : TwoMat => M i j) (h.2 H v)
    have hpre := Activation.tanh_injective hij
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.m2rnnCandidate, M2RNNComparison.e88DeltaUpdateExpanded,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap] using hpre

/-- The two basis keys still induce distinct E88/NDM updates after the
elementwise `tanh` nonlinearity.

The witness fixes `H = I` and `v = 0`. The preactivations are just the two
key-dependent E88 transition matrices, and elementwise `tanh` is injective. -/
theorem e88_two_keys_induce_distinct_updates_at_identity_zero :
    M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key0 (0 : TwoVec) ≠
      M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key1 (0 : TwoVec) := by
  intro h
  apply e88_two_keys_induce_distinct_left_transitions
  ext i j
  have hij := congrArg (fun M : TwoMat => M i j) h
  have hpre := Activation.tanh_injective hij
  simpa [M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.matrixTanh,
    M2RNNComparison.matrixMap, M2RNNComparison.outerKV] using hpre

/-- Adding M2RNN's external scalar forget carry does not remove the
fixed-transition separation.

For any fixed right-transition matrix `W` and scalar forget gate `f`, the full
M2RNN update

`f H + (1 - f) tanh(H W + k vᵀ)`

cannot uniformly match E88/NDM's expanded delta update for both basis keys. The
witness again sets `H = I` and `v = 0`: M2RNN's full update is identical for
`key0` and `key1`, because the key only appears through the zero raw-write term,
while E88's update remains key-dependent through `(I - k kᵀ) H`. -/
theorem no_fixed_m2rnn_update_matches_e88_two_basis_keys
    (W : TwoMat) (f : Real) :
    ¬ ((∀ H v,
          M2RNNComparison.m2rnnUpdate W f H key0 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key0 v) ∧
       (∀ H v,
          M2RNNComparison.m2rnnUpdate W f H key1 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key1 v)) := by
  intro h
  have hm2 :
      M2RNNComparison.m2rnnUpdate W f (1 : TwoMat) key0 (0 : TwoVec) =
        M2RNNComparison.m2rnnUpdate W f (1 : TwoMat) key1 (0 : TwoVec) := by
    simp [M2RNNComparison.m2rnnUpdate, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.outerKV]
  have he88 :
      M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key0 (0 : TwoVec) =
        M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key1 (0 : TwoVec) := by
    calc
      M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key0 (0 : TwoVec)
          = M2RNNComparison.m2rnnUpdate W f (1 : TwoMat) key0 (0 : TwoVec) :=
            (h.1 (1 : TwoMat) (0 : TwoVec)).symm
      _ = M2RNNComparison.m2rnnUpdate W f (1 : TwoMat) key1 (0 : TwoVec) := hm2
      _ = M2RNNComparison.e88DeltaUpdateExpanded 1 (1 : TwoMat) key1 (0 : TwoVec) :=
            h.2 (1 : TwoMat) (0 : TwoVec)
  exact e88_two_keys_induce_distinct_updates_at_identity_zero he88

/-! ## Input-Dependent Forget Gates -/

/-- `tanh(1)` is nonzero. -/
theorem tanh_one_ne_zero : Real.tanh (1 : Real) ≠ 0 := by
  intro h
  have hzero : Real.tanh (1 : Real) = Real.tanh 0 := by
    simpa [Real.tanh_zero] using h
  have hone : (1 : Real) = 0 := Activation.tanh_injective hzero
  norm_num at hone

/-- `tanh(-1)` is nonzero. -/
theorem tanh_neg_one_ne_zero : Real.tanh (-1 : Real) ≠ 0 := by
  intro h
  have hzero : Real.tanh (-1 : Real) = Real.tanh 0 := by
    simpa [Real.tanh_zero] using h
  have hneg : (-1 : Real) = 0 := Activation.tanh_injective hzero
  norm_num at hneg

/-- Even if the scalar M2RNN forget gate is allowed to depend on the input key,
the fixed-transition/scalar-carry update cannot uniformly match E88/NDM on the
two basis keys.

The key point is that matching writes from an empty state forces the scalar
forget gate to be `0` for each basis key: with `H = 0`, E88 writes
`tanh(k vᵀ)`, while M2RNN writes `(1 - f(k)) tanh(k vᵀ)`. Since `tanh(1) ≠ 0`,
uniform equality forces `f(key0) = f(key1) = 0`, reducing the claim to the
candidate-path separation above. -/
theorem no_fixed_m2rnn_update_with_key_scalar_forget_matches_e88_two_basis_keys
    (W : TwoMat) (f0 f1 : Real) :
    ¬ ((∀ H v,
          M2RNNComparison.m2rnnUpdate W f0 H key0 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key0 v) ∧
       (∀ H v,
          M2RNNComparison.m2rnnUpdate W f1 H key1 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key1 v)) := by
  intro h
  have hf0 : f0 = 0 := by
    have hentry := congrArg (fun M : TwoMat => M 0 0) (h.1 (0 : TwoMat) key0)
    have hentry' : (1 - f0) * Real.tanh (1 : Real) = Real.tanh (1 : Real) := by
      simpa [M2RNNComparison.m2rnnUpdate, M2RNNComparison.m2rnnCandidate,
        M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
        M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
        M2RNNComparison.outerKV, key0] using hentry
    have hmul : f0 * Real.tanh (1 : Real) = 0 := by nlinarith
    exact (mul_eq_zero.mp hmul).resolve_right tanh_one_ne_zero
  have hf1 : f1 = 0 := by
    have hentry := congrArg (fun M : TwoMat => M 1 1) (h.2 (0 : TwoMat) key1)
    have hentry' : (1 - f1) * Real.tanh (1 : Real) = Real.tanh (1 : Real) := by
      simpa [M2RNNComparison.m2rnnUpdate, M2RNNComparison.m2rnnCandidate,
        M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
        M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
        M2RNNComparison.outerKV, key1] using hentry
    have hmul : f1 * Real.tanh (1 : Real) = 0 := by nlinarith
    exact (mul_eq_zero.mp hmul).resolve_right tanh_one_ne_zero
  apply no_fixed_m2rnn_candidate_matches_e88_two_basis_keys W
  constructor
  · intro H v
    simpa [hf0, M2RNNComparison.m2rnnUpdate] using h.1 H v
  · intro H v
    simpa [hf1, M2RNNComparison.m2rnnUpdate] using h.2 H v

/-! ## Vector/Row/Column Forget-Gate Characterization -/

/-- A mixed key that exposes the cross-row part of E88's `k kᵀ H` correction. -/
def mixedKey : TwoVec :=
  fun _ => 1

/-- A state whose first row is zero and whose second row has one active cell. -/
def lowerLeftState : TwoMat :=
  Matrix.of fun i j => if i = 1 then if j = 0 then 1 else 0 else 0

/-- Row-vector external carry: each row has its own forget value. -/
def rowForgetCarry2 (r : TwoVec) (H Z : TwoMat) : TwoMat :=
  Matrix.of fun i j => r i * H i j + (1 - r i) * Z i j

/-- Column-vector external carry: each column has its own forget value. -/
def columnForgetCarry2 (c : TwoVec) (H Z : TwoMat) : TwoMat :=
  Matrix.of fun i j => c j * H i j + (1 - c j) * Z i j

/-- Cellwise external carry: every matrix cell has its own forget value. -/
def cellForgetCarry2 (g : TwoMat) (H Z : TwoMat) : TwoMat :=
  Matrix.of fun i j => g i j * H i j + (1 - g i j) * Z i j

/-- M2RNN candidate followed by row-vector external forget carry. -/
noncomputable def m2rnnRowForgetUpdate2
    (W : TwoMat) (r : TwoVec) (H : TwoMat) (k v : TwoVec) : TwoMat :=
  rowForgetCarry2 r H (M2RNNComparison.m2rnnCandidate W H k v)

/-- M2RNN candidate followed by column-vector external forget carry. -/
noncomputable def m2rnnColumnForgetUpdate2
    (W : TwoMat) (c : TwoVec) (H : TwoMat) (k v : TwoVec) : TwoMat :=
  columnForgetCarry2 c H (M2RNNComparison.m2rnnCandidate W H k v)

/-- M2RNN candidate followed by cellwise external forget carry. -/
noncomputable def m2rnnCellForgetUpdate2
    (W : TwoMat) (g : TwoMat) (H : TwoMat) (k v : TwoVec) : TwoMat :=
  cellForgetCarry2 g H (M2RNNComparison.m2rnnCandidate W H k v)

/-- A row-gated M2RNN external carry still cannot match E88/NDM's mixed-key
delta correction.

The witness fixes `k = (1, 1)`, `v = 0`, and a state whose first row is zero
but second row is nonzero. A fixed right transition `H W` is row-local: the
first output row depends only on the first input row, so the row-gated M2RNN
entry `(0,0)` is zero. E88's `(I - k kᵀ) H` subtracts the second row from the
first row, so the same entry is `tanh(-1)`. -/
theorem row_forget_m2rnn_fails_mixed_key_delta_correction
    (W : TwoMat) (r : TwoVec) :
    m2rnnRowForgetUpdate2 W r lowerLeftState mixedKey (0 : TwoVec) ≠
      M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec) := by
  intro h
  have hentry := congrArg (fun M : TwoMat => M 0 0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [m2rnnRowForgetUpdate2, rowForgetCarry2, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, lowerLeftState, mixedKey, Matrix.mul_apply] using hentry
  exact tanh_one_ne_zero hbad

/-- A column-gated M2RNN external carry still cannot match E88/NDM's mixed-key
delta correction, for the same row-locality reason as the row-gated case. -/
theorem column_forget_m2rnn_fails_mixed_key_delta_correction
    (W : TwoMat) (c : TwoVec) :
    m2rnnColumnForgetUpdate2 W c lowerLeftState mixedKey (0 : TwoVec) ≠
      M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec) := by
  intro h
  have hentry := congrArg (fun M : TwoMat => M 0 0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [m2rnnColumnForgetUpdate2, columnForgetCarry2, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, lowerLeftState, mixedKey, Matrix.mul_apply] using hentry
  exact tanh_one_ne_zero hbad

/-- Even cellwise external carry cannot repair the missing mixed-key delta
correction if the candidate path remains a fixed right transition plus raw
outer-product write. Cell gates can retain or replace cells, but they do not
create the cross-row term `-k kᵀ H` that E88/NDM applies before `tanh`. -/
theorem cell_forget_m2rnn_fails_mixed_key_delta_correction
    (W : TwoMat) (g : TwoMat) :
    m2rnnCellForgetUpdate2 W g lowerLeftState mixedKey (0 : TwoVec) ≠
      M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec) := by
  intro h
  have hentry := congrArg (fun M : TwoMat => M 0 0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [m2rnnCellForgetUpdate2, cellForgetCarry2, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, lowerLeftState, mixedKey, Matrix.mul_apply] using hentry
  exact tanh_one_ne_zero hbad

/-! ## One-Step Resource Separation Target -/

/-- The one-step mixed-key delta-correction target.

This target isolates the operation that E88/NDM gets from its input-dependent
left transition:

`H ↦ tanh((I - k kᵀ) H + k vᵀ)`.

At the witness state/key/value below, the correction moves information across
rows. A fixed-right-transition M2RNN candidate can mix columns within each row,
but cannot create this cross-row correction in one step; external row/column/cell
forget gates can only retain or replace already-computed cells. -/
def ImplementsMixedKeyDeltaCorrection
    (update : TwoMat → TwoVec → TwoVec → TwoMat) : Prop :=
  update lowerLeftState mixedKey (0 : TwoVec) =
    M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec)

/-- E88/NDM implements the mixed-key delta-correction target by definition. -/
theorem e88_implements_mixed_key_delta_correction :
    ImplementsMixedKeyDeltaCorrection (M2RNNComparison.e88DeltaUpdateExpanded 1) := by
  rfl

/-- A small resource class for M2RNN-style one-step updates with a fixed right
transition, raw outer-product candidate, and external forget carry.

The row/column/cell cases cover scalar and input-dependent variants at this
witness: once the state/key/value are fixed, any such gate has some concrete
row, column, or cell values. -/
inductive FixedRightRawExternalForget2 where
  | row (W : TwoMat) (r : TwoVec)
  | column (W : TwoMat) (c : TwoVec)
  | cell (W : TwoMat) (g : TwoMat)

/-- Interpret a fixed-right/raw-write external-forget resource as an update
function. -/
noncomputable def FixedRightRawExternalForget2.update :
    FixedRightRawExternalForget2 → TwoMat → TwoVec → TwoVec → TwoMat
  | .row W r => m2rnnRowForgetUpdate2 W r
  | .column W c => m2rnnColumnForgetUpdate2 W c
  | .cell W g => m2rnnCellForgetUpdate2 W g

/-- Row-gated fixed-right M2RNN resources do not implement the mixed-key delta
correction target. -/
theorem row_forget_m2rnn_not_implements_mixed_key_delta_correction
    (W : TwoMat) (r : TwoVec) :
    ¬ ImplementsMixedKeyDeltaCorrection (m2rnnRowForgetUpdate2 W r) := by
  intro h
  exact row_forget_m2rnn_fails_mixed_key_delta_correction W r h

/-- Column-gated fixed-right M2RNN resources do not implement the mixed-key
delta correction target. -/
theorem column_forget_m2rnn_not_implements_mixed_key_delta_correction
    (W : TwoMat) (c : TwoVec) :
    ¬ ImplementsMixedKeyDeltaCorrection (m2rnnColumnForgetUpdate2 W c) := by
  intro h
  exact column_forget_m2rnn_fails_mixed_key_delta_correction W c h

/-- Cellwise-gated fixed-right M2RNN resources do not implement the mixed-key
delta correction target. -/
theorem cell_forget_m2rnn_not_implements_mixed_key_delta_correction
    (W : TwoMat) (g : TwoMat) :
    ¬ ImplementsMixedKeyDeltaCorrection (m2rnnCellForgetUpdate2 W g) := by
  intro h
  exact cell_forget_m2rnn_fails_mixed_key_delta_correction W g h

/-- Main one-step resource separation:

E88/NDM implements the mixed-key delta correction in one recurrent step. The
M2RNN-style resource class with fixed right transition, raw outer write, and
external row/column/cell forget carry cannot implement that same target in one
step.

This is deliberately a one-step/resource theorem, not a broad computability
separation. It identifies the extra mechanism M2RNN would need to simulate E88:
an input-dependent left transition or additional resources/steps that recreate
the missing `-k kᵀ H` correction. -/
theorem ndm_m2rnn_one_step_resource_separation :
    ImplementsMixedKeyDeltaCorrection (M2RNNComparison.e88DeltaUpdateExpanded 1) ∧
    ∀ resource : FixedRightRawExternalForget2,
      ¬ ImplementsMixedKeyDeltaCorrection resource.update := by
  constructor
  · exact e88_implements_mixed_key_delta_correction
  · intro resource
    cases resource with
    | row W r =>
        exact row_forget_m2rnn_not_implements_mixed_key_delta_correction W r
    | column W c =>
        exact column_forget_m2rnn_not_implements_mixed_key_delta_correction W c
    | cell W g =>
        exact cell_forget_m2rnn_not_implements_mixed_key_delta_correction W g

/-! ## General-Dimension Embedding -/

/-- Row-vector external carry for arbitrary key/value dimensions. -/
def rowForgetCarryKV {K V : Nat}
    (r : M2RNNComparison.Vec K)
    (H Z : M2RNNComparison.MatState K V) : M2RNNComparison.MatState K V :=
  Matrix.of fun i j => r i * H i j + (1 - r i) * Z i j

/-- Column-vector external carry for arbitrary key/value dimensions. -/
def columnForgetCarryKV {K V : Nat}
    (c : M2RNNComparison.Vec V)
    (H Z : M2RNNComparison.MatState K V) : M2RNNComparison.MatState K V :=
  Matrix.of fun i j => c j * H i j + (1 - c j) * Z i j

/-- Cellwise external carry for arbitrary key/value dimensions. -/
def cellForgetCarryKV {K V : Nat}
    (g H Z : M2RNNComparison.MatState K V) : M2RNNComparison.MatState K V :=
  Matrix.of fun i j => g i j * H i j + (1 - g i j) * Z i j

/-- M2RNN candidate followed by row-vector external forget carry. -/
noncomputable def m2rnnRowForgetUpdateKV {K V : Nat}
    (W : Matrix (Fin V) (Fin V) Real) (r : M2RNNComparison.Vec K)
    (H : M2RNNComparison.MatState K V)
    (k : M2RNNComparison.Vec K) (v : M2RNNComparison.Vec V) :
    M2RNNComparison.MatState K V :=
  rowForgetCarryKV r H (M2RNNComparison.m2rnnCandidate W H k v)

/-- M2RNN candidate followed by column-vector external forget carry. -/
noncomputable def m2rnnColumnForgetUpdateKV {K V : Nat}
    (W : Matrix (Fin V) (Fin V) Real) (c : M2RNNComparison.Vec V)
    (H : M2RNNComparison.MatState K V)
    (k : M2RNNComparison.Vec K) (v : M2RNNComparison.Vec V) :
    M2RNNComparison.MatState K V :=
  columnForgetCarryKV c H (M2RNNComparison.m2rnnCandidate W H k v)

/-- M2RNN candidate followed by cellwise external forget carry. -/
noncomputable def m2rnnCellForgetUpdateKV {K V : Nat}
    (W : Matrix (Fin V) (Fin V) Real)
    (g H : M2RNNComparison.MatState K V)
    (k : M2RNNComparison.Vec K) (v : M2RNNComparison.Vec V) :
    M2RNNComparison.MatState K V :=
  cellForgetCarryKV g H (M2RNNComparison.m2rnnCandidate W H k v)

/-- The embedded two-direction mixed key in any key space of size at least two. -/
def embeddedMixedKey {K : Nat} (hK : 1 < K) : M2RNNComparison.Vec K :=
  fun i =>
    if i = (⟨0, Nat.lt_trans Nat.zero_lt_one hK⟩ : Fin K) then 1
    else if i = (⟨1, hK⟩ : Fin K) then 1
    else 0

/-- The embedded witness state: row 0 is zero; row 1 stores one active value in
column 0; all other coordinates are zero. -/
def embeddedLowerLeftState {K V : Nat} (hK : 1 < K) (hV : 0 < V) :
    M2RNNComparison.MatState K V :=
  Matrix.of fun i j =>
    if i = (⟨1, hK⟩ : Fin K) then
      if j = (⟨0, hV⟩ : Fin V) then 1 else 0
    else 0

/-- The arbitrary-dimensional mixed-key delta-correction target. It is the same
2D obstruction embedded into any `K ≥ 2, V ≥ 1` state space. -/
def ImplementsEmbeddedMixedKeyDeltaCorrection {K V : Nat} (hK : 1 < K) (hV : 0 < V)
    (update :
      M2RNNComparison.MatState K V →
      M2RNNComparison.Vec K →
      M2RNNComparison.Vec V →
      M2RNNComparison.MatState K V) : Prop :=
  update (embeddedLowerLeftState hK hV) (embeddedMixedKey hK)
      (0 : M2RNNComparison.Vec V) =
    M2RNNComparison.e88DeltaUpdateExpanded 1
      (embeddedLowerLeftState hK hV) (embeddedMixedKey hK)
      (0 : M2RNNComparison.Vec V)

/-- E88/NDM implements the embedded mixed-key delta-correction target in any
dimension with at least two key coordinates and one value coordinate. -/
theorem e88_implements_embedded_mixed_key_delta_correction
    {K V : Nat} (hK : 1 < K) (hV : 0 < V) :
    ImplementsEmbeddedMixedKeyDeltaCorrection hK hV
      (M2RNNComparison.e88DeltaUpdateExpanded 1) := by
  rfl

/-- Row-gated fixed-right M2RNN cannot implement the embedded mixed-key
delta-correction target in any larger state space. -/
theorem row_forget_m2rnn_fails_embedded_mixed_key_delta_correction
    {K V : Nat} (hK : 1 < K) (hV : 0 < V)
    (W : Matrix (Fin V) (Fin V) Real) (r : M2RNNComparison.Vec K) :
    ¬ ImplementsEmbeddedMixedKeyDeltaCorrection hK hV
      (m2rnnRowForgetUpdateKV W r) := by
  intro h
  let i0 : Fin K := ⟨0, Nat.lt_trans Nat.zero_lt_one hK⟩
  let j0 : Fin V := ⟨0, hV⟩
  have hentry := congrArg (fun M : M2RNNComparison.MatState K V => M i0 j0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [ImplementsEmbeddedMixedKeyDeltaCorrection, m2rnnRowForgetUpdateKV,
      rowForgetCarryKV, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, embeddedLowerLeftState, embeddedMixedKey,
      Matrix.mul_apply, i0, j0] using hentry
  exact tanh_one_ne_zero hbad

/-- Column-gated fixed-right M2RNN cannot implement the embedded mixed-key
delta-correction target in any larger state space. -/
theorem column_forget_m2rnn_fails_embedded_mixed_key_delta_correction
    {K V : Nat} (hK : 1 < K) (hV : 0 < V)
    (W : Matrix (Fin V) (Fin V) Real) (c : M2RNNComparison.Vec V) :
    ¬ ImplementsEmbeddedMixedKeyDeltaCorrection hK hV
      (m2rnnColumnForgetUpdateKV W c) := by
  intro h
  let i0 : Fin K := ⟨0, Nat.lt_trans Nat.zero_lt_one hK⟩
  let j0 : Fin V := ⟨0, hV⟩
  have hentry := congrArg (fun M : M2RNNComparison.MatState K V => M i0 j0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [ImplementsEmbeddedMixedKeyDeltaCorrection, m2rnnColumnForgetUpdateKV,
      columnForgetCarryKV, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, embeddedLowerLeftState, embeddedMixedKey,
      Matrix.mul_apply, i0, j0] using hentry
  exact tanh_one_ne_zero hbad

/-- Cellwise-gated fixed-right M2RNN cannot implement the embedded mixed-key
delta-correction target in any larger state space. -/
theorem cell_forget_m2rnn_fails_embedded_mixed_key_delta_correction
    {K V : Nat} (hK : 1 < K) (hV : 0 < V)
    (W : Matrix (Fin V) (Fin V) Real)
    (g : M2RNNComparison.MatState K V) :
    ¬ ImplementsEmbeddedMixedKeyDeltaCorrection hK hV
      (m2rnnCellForgetUpdateKV W g) := by
  intro h
  let i0 : Fin K := ⟨0, Nat.lt_trans Nat.zero_lt_one hK⟩
  let j0 : Fin V := ⟨0, hV⟩
  have hentry := congrArg (fun M : M2RNNComparison.MatState K V => M i0 j0) h
  have hbad : Real.tanh (1 : Real) = 0 := by
    simpa [ImplementsEmbeddedMixedKeyDeltaCorrection, m2rnnCellForgetUpdateKV,
      cellForgetCarryKV, M2RNNComparison.m2rnnCandidate,
      M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, embeddedLowerLeftState, embeddedMixedKey,
      Matrix.mul_apply, i0, j0] using hentry
  exact tanh_one_ne_zero hbad

/-- Fixed-right/raw-write external-forget resources in arbitrary dimension. -/
inductive FixedRightRawExternalForgetKV (K V : Nat) where
  | row (W : Matrix (Fin V) (Fin V) Real) (r : M2RNNComparison.Vec K)
  | column (W : Matrix (Fin V) (Fin V) Real) (c : M2RNNComparison.Vec V)
  | cell (W : Matrix (Fin V) (Fin V) Real) (g : M2RNNComparison.MatState K V)

/-- Interpret an arbitrary-dimensional fixed-right/raw-write resource as an
update function. -/
noncomputable def FixedRightRawExternalForgetKV.update {K V : Nat} :
    FixedRightRawExternalForgetKV K V →
      M2RNNComparison.MatState K V →
      M2RNNComparison.Vec K →
      M2RNNComparison.Vec V →
      M2RNNComparison.MatState K V
  | .row W r => m2rnnRowForgetUpdateKV W r
  | .column W c => m2rnnColumnForgetUpdateKV W c
  | .cell W g => m2rnnCellForgetUpdateKV W g

/-- General embedding theorem: the one-step separation is not a 2D artifact.

For every key dimension `K ≥ 2` and value dimension `V ≥ 1`, E88/NDM implements
the embedded mixed-key delta correction in one recurrent step, while every
fixed-right/raw-write M2RNN-style resource with external row/column/cell forget
fails on the embedded witness.
-/
theorem ndm_m2rnn_one_step_resource_separation_embeds
    {K V : Nat} (hK : 1 < K) (hV : 0 < V) :
    ImplementsEmbeddedMixedKeyDeltaCorrection hK hV
      (M2RNNComparison.e88DeltaUpdateExpanded 1) ∧
    ∀ resource : FixedRightRawExternalForgetKV K V,
      ¬ ImplementsEmbeddedMixedKeyDeltaCorrection hK hV resource.update := by
  constructor
  · exact e88_implements_embedded_mixed_key_delta_correction hK hV
  · intro resource
    cases resource with
    | row W r =>
        exact row_forget_m2rnn_fails_embedded_mixed_key_delta_correction hK hV W r
    | column W c =>
        exact column_forget_m2rnn_fails_embedded_mixed_key_delta_correction hK hV W c
    | cell W g =>
        exact cell_forget_m2rnn_fails_embedded_mixed_key_delta_correction hK hV W g

/-! ## Interpretation Hooks

These are not capability theorems yet. They are hooks for the theorems we want:

* broad nonlinear temporal class: E88/NDM and M2RNN are likely not separated
  by classical computability class alone;
* one-step transition family: E88/NDM and M2RNN are structurally separated;
* resource-bounded capability: compare at fixed params, wallclock, memory,
  precision, optimizer, tokens, and context length;
* hardware utilization: E88/NDM exposes many independent small recurrent
  programs, making pure nonlinear recurrence trainable at scale.
-/

end RecurrentResourceFormalism
