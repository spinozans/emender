/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import ElmanProofs.Architectures.M2RNNComparison

/-!
# Online Keyed Memory

This module factors out the part shared by Gated DeltaNet and E88/NDM:
delta-correcting associative memory.

The important point for the architecture comparison is that exact overwrite is
not E88-specific. It is already present in the ideal delta rule used by GDN:

`S' = S + k (v - S^T k)^T`.

E88/NDM uses the same error-correcting pre-activation, then applies a nonlinear
full-state map (tanh in the current implementation). The later separation target
is therefore not "delta memory versus GDN"; it is linear/scan-compatible delta
memory versus nonlinear recurrent delta memory, and raw-write/fixed-transition
matrix RNNs versus delta-correcting memory.

M2RNN's matrix-state recurrence contains a raw association term `k v^T`. The
theorems below separate that raw association primitive from the GDN/E88 delta
primitive as an online overwrite memory. This is intentionally a one-step memory
semantics theorem, not yet a full lower bound for the complete M2RNN recurrence
with `tanh`, forget interpolation, and learned state transition.
-/

namespace OnlineMemory

open Matrix BigOperators

abbrev KeyVec (K : Nat) := M2RNNComparison.Vec K
abbrev ValueVec (V : Nat) := M2RNNComparison.Vec V
abbrev Memory (K V : Nat) := M2RNNComparison.MatState K V

variable {K V : Nat}

/-! ## Abstract Online Memory Spec -/

/-- Inner product on key vectors. -/
def keyDot (k q : KeyVec K) : Real :=
  Finset.univ.sum fun i => k i * q i

/-- Read a value vector from memory by query key. -/
def read (S : Memory K V) (q : KeyVec K) : ValueVec V :=
  M2RNNComparison.queryReadout S q

/-- Retrieval error at key `k`: the correction needed to make `read S k = v`. -/
def correction (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : ValueVec V :=
  v - read S k

/-- Ideal delta-correcting write:

`S' = S + k (v - read S k)^T`.
-/
def linearDeltaWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  S + M2RNNComparison.outerKV k (correction S k v)

/-- Raw outer-product write:

`S' = S + k v^T`.

This is the common raw associative-memory core. It can add an association, but
without state-dependent correction it does not implement overwrite for arbitrary
existing memory.
-/
def rawOuterWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  S + M2RNNComparison.outerKV k v

/-- General state-independent additive write. The added matrix may depend on the
current key and desired value, but not on the old memory `S`. -/
def stateIndependentAdditiveWrite
    (term : KeyVec K → ValueVec V → Memory K V)
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  S + term k v

/-- Gated DeltaNet's ideal write with unit write gate and no global decay. -/
def gdnIdealWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  linearDeltaWrite S k v

/-- E88/NDM pre-nonlinearity write with unit decay. -/
def ndmPreTanhWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  linearDeltaWrite S k v

/-- E88/NDM's nonlinear full-state version of the delta write. -/
noncomputable def ndmNonlinearWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  M2RNNComparison.matrixTanh (linearDeltaWrite S k v)

/-! ## Delta-Memory Semantics -/

/-- Reads distribute over matrix addition. -/
theorem read_add
    (A B : Memory K V) (q : KeyVec K) :
    read (A + B) q = read A q + read B q := by
  ext j
  simp [read, M2RNNComparison.queryReadout, Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum (fun x : Fin K => (A x j + B x j) * q x)
        =
      Finset.univ.sum (fun x : Fin K => A x j * q x + B x j * q x) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
    _ =
      Finset.univ.sum (fun x : Fin K => A x j * q x) +
      Finset.univ.sum (fun x : Fin K => B x j * q x) := by
        rw [Finset.sum_add_distrib]

/-- Reading an outer-product write returns the written value scaled by key
overlap. -/
theorem read_outerKV
    (k q : KeyVec K) (e : ValueVec V) :
    read (M2RNNComparison.outerKV k e) q =
      fun j => keyDot k q * e j := by
  ext j
  simp [read, M2RNNComparison.queryReadout, M2RNNComparison.outerKV,
    keyDot, Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum (fun x : Fin K => k x * e j * q x)
        = Finset.univ.sum (fun x : Fin K => (k x * q x) * e j) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = (Finset.univ.sum fun i : Fin K => k i * q i) * e j := by
          rw [Finset.sum_mul]
    _ = (Finset.univ.sum fun i : Fin K => k i * q i) * e j := rfl

/-- Delta write changes a query in proportion to its overlap with the write key. -/
theorem read_linearDeltaWrite
    (S : Memory K V) (k q : KeyVec K) (v : ValueVec V) :
    read (linearDeltaWrite S k v) q =
      read S q + fun j => keyDot k q * correction S k v j := by
  ext j
  simp [linearDeltaWrite, read, correction, M2RNNComparison.queryReadout,
    M2RNNComparison.outerKV, keyDot, Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum
        (fun x : Fin K => (S x j + k x * (v j - Finset.univ.sum (fun x => S x j * k x))) * q x)
        =
      Finset.univ.sum
        (fun x : Fin K =>
          S x j * q x + (k x * q x) * (v j - Finset.univ.sum (fun x => S x j * k x))) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ =
      Finset.univ.sum (fun x : Fin K => S x j * q x) +
      Finset.univ.sum
        (fun x : Fin K => (k x * q x) * (v j - Finset.univ.sum (fun x => S x j * k x))) := by
          rw [Finset.sum_add_distrib]
    _ =
      Finset.univ.sum (fun x : Fin K => S x j * q x) +
      (Finset.univ.sum fun i : Fin K => k i * q i) *
        (v j - Finset.univ.sum (fun x => S x j * k x)) := by
          rw [Finset.sum_mul]

/-! ## Raw Writes Versus Delta Writes -/

/-- Raw outer-product write changes a query in proportion to key overlap, but it
does not subtract existing retrieved content. -/
theorem read_rawOuterWrite
    (S : Memory K V) (k q : KeyVec K) (v : ValueVec V) :
    read (rawOuterWrite S k v) q =
      read S q + fun j => keyDot k q * v j := by
  rw [rawOuterWrite, read_add, read_outerKV]

/-- With a unit key, raw writing reads back "old content plus new value". -/
theorem rawOuterWrite_reads_existing_plus_value_unit
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) :
    read (rawOuterWrite S k v) k = read S k + v := by
  rw [read_rawOuterWrite]
  ext j
  simp [hk]

/-- A raw outer-product write cannot be an exact overwrite when the addressed
slot already contains nonzero content. -/
theorem rawOuterWrite_not_exact_overwrite_with_existing_content
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) (j : Fin V)
    (hprev : read S k j ≠ 0) :
    read (rawOuterWrite S k v) k ≠ v := by
  intro h
  rw [rawOuterWrite_reads_existing_plus_value_unit S k v hk] at h
  have hj := congrFun h j
  change read S k j + v j = v j at hj
  have hz : read S k j = 0 := by
    linarith
  exact hprev hz

/-- More generally, no state-independent additive write can overwrite two old
memories that disagree at the addressed key in one step.

This is the resource-separation hook: to implement overwrite uniformly, the
write term must either depend on the old state, add extra machinery, or use
additional steps/resources to infer and cancel the old content. -/
theorem stateIndependentAdditiveWrite_cannot_exact_overwrite_two_memories
    (term : KeyVec K → ValueVec V → Memory K V)
    (S₁ S₂ : Memory K V) (k : KeyVec K) (v : ValueVec V) (j : Fin V)
    (hread : read S₁ k j ≠ read S₂ k j) :
    ¬ (read (stateIndependentAdditiveWrite term S₁ k v) k = v ∧
       read (stateIndependentAdditiveWrite term S₂ k v) k = v) := by
  intro h
  simp [stateIndependentAdditiveWrite, read_add] at h
  have h₁ := congrFun h.1 j
  have h₂ := congrFun h.2 j
  change read S₁ k j + read (term k v) k j = v j at h₁
  change read S₂ k j + read (term k v) k j = v j at h₂
  have heq : read S₁ k j = read S₂ k j := by
    linarith
  exact hread heq

/-- Unit-key delta write exactly overwrites the addressed readout. -/
theorem linearDeltaWrite_exact_overwrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) :
    read (linearDeltaWrite S k v) k = v := by
  rw [read_linearDeltaWrite]
  ext j
  simp [correction, hk]

/-- Orthogonal queries are preserved by a delta write. -/
theorem linearDeltaWrite_preserves_orthogonal_query
    (S : Memory K V) (k q : KeyVec K) (v : ValueVec V)
    (horth : keyDot k q = 0) :
    read (linearDeltaWrite S k v) q = read S q := by
  rw [read_linearDeltaWrite]
  ext j
  simp [horth]

/-- In the idealized no-decay, unit-write setting, GDN and E88/NDM share the
same delta-correcting pre-nonlinearity. -/
theorem gdn_and_ndm_share_ideal_delta_write
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) :
    gdnIdealWrite S k v = ndmPreTanhWrite S k v := by
  rfl

/-- The ideal overwrite theorem applies to both GDN's delta core and E88/NDM's
pre-tanh delta core. -/
theorem shared_delta_core_exact_overwrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) :
    read (gdnIdealWrite S k v) k = v ∧
    read (ndmPreTanhWrite S k v) k = v := by
  constructor <;> exact linearDeltaWrite_exact_overwrite S k v hk

/-! ## Multi-Key Orthogonal Capacity -/

/-- A finite family of keys is orthonormal when self-overlap is one and
cross-overlap is zero. -/
def OrthonormalKeys {N : Nat} (keys : Fin N → KeyVec K) : Prop :=
  ∀ i j, keyDot (keys i) (keys j) = if i = j then 1 else 0

/-- The explicit associative-memory table for a finite family of key/value
pairs. -/
def memoryTable {N : Nat}
    (keys : Fin N → KeyVec K) (values : Fin N → ValueVec V) : Memory K V :=
  Finset.univ.sum fun i => M2RNNComparison.outerKV (keys i) (values i)

/-- Orthonormal keys support exact finite-table retrieval. This is the capacity
upper bound shared by ideal GDN-style delta memory and E88/NDM's pre-tanh delta
core: `N` orthogonal keys can store `N` independent value vectors in a matrix
state. -/
theorem memoryTable_retrieves_orthonormal {N : Nat}
    (keys : Fin N → KeyVec K) (values : Fin N → ValueVec V)
    (horth : OrthonormalKeys keys) :
    ∀ j, read (memoryTable keys values) (keys j) = values j := by
  intro j
  ext dim
  simp [memoryTable, read, M2RNNComparison.queryReadout,
    M2RNNComparison.outerKV, Matrix.sum_apply, Matrix.of_apply,
    Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum
        (fun x : Fin K => (Finset.univ.sum fun i : Fin N =>
          keys i x * values i dim) * keys j x)
        =
      Finset.univ.sum
        (fun x : Fin K => Finset.univ.sum fun i : Fin N =>
          values i dim * keys i x * keys j x) := by
          apply Finset.sum_congr rfl
          intro x _
          rw [Finset.sum_mul]
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ =
      Finset.univ.sum
        (fun i : Fin N => Finset.univ.sum fun x : Fin K =>
          values i dim * keys i x * keys j x) := by
          rw [Finset.sum_comm]
    _ =
      Finset.univ.sum
        (fun i : Fin N => values i dim *
          Finset.univ.sum (fun x : Fin K => keys i x * keys j x)) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro x _
          ring
    _ =
      Finset.univ.sum
        (fun i : Fin N => values i dim * keyDot (keys i) (keys j)) := by
          rfl
    _ =
      Finset.univ.sum
        (fun i : Fin N => values i dim * (if i = j then 1 else 0)) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [horth i j]
    _ = values j dim := by
          rw [Fintype.sum_eq_single j]
          · simp
          · intro i hi
            simp [hi]

/-- Online extension step: if `S` already stores an orthogonal family of keys,
a delta write to a new orthogonal unit key stores the new value and preserves
all existing readouts. This is the induction step behind online associative
memory. -/
theorem linearDeltaWrite_extends_orthogonal_readouts {N : Nat}
    (S : Memory K V) (keys : Fin N → KeyVec K) (values : Fin N → ValueVec V)
    (newKey : KeyVec K) (newValue : ValueVec V)
    (hunit : keyDot newKey newKey = 1)
    (horth : ∀ i, keyDot newKey (keys i) = 0)
    (hstored : ∀ i, read S (keys i) = values i) :
    read (linearDeltaWrite S newKey newValue) newKey = newValue ∧
    ∀ i, read (linearDeltaWrite S newKey newValue) (keys i) = values i := by
  constructor
  · exact linearDeltaWrite_exact_overwrite S newKey newValue hunit
  · intro i
    rw [linearDeltaWrite_preserves_orthogonal_query S newKey (keys i) newValue (horth i)]
    exact hstored i

/-- Online overwrite step for an orthonormal table: rewriting one key replaces
only that key and preserves every other key. -/
theorem linearDeltaWrite_overwrites_one_preserves_others {N : Nat}
    (keys : Fin N → KeyVec K) (values : Fin N → ValueVec V)
    (horth : OrthonormalKeys keys) (target : Fin N) (newValue : ValueVec V) :
    read (linearDeltaWrite (memoryTable keys values) (keys target) newValue)
        (keys target) = newValue ∧
    ∀ i, i ≠ target →
      read (linearDeltaWrite (memoryTable keys values) (keys target) newValue)
        (keys i) = values i := by
  have hunit : keyDot (keys target) (keys target) = 1 := by
    have h := horth target target
    simpa using h
  constructor
  · exact linearDeltaWrite_exact_overwrite (memoryTable keys values) (keys target) newValue hunit
  · intro i hi
    have horth_i : keyDot (keys target) (keys i) = 0 := by
      have h := horth target i
      simp [hi.symm] at h
      exact h
    rw [linearDeltaWrite_preserves_orthogonal_query
      (memoryTable keys values) (keys target) (keys i) newValue horth_i]
    exact memoryTable_retrieves_orthonormal keys values horth i

/-! ## Resource Interpretation Hooks -/

/-- Uniform one-step overwrite: for every old memory, every unit key, and every
desired value, one write makes that key read as the desired value. -/
def UniformOneStepOverwrite
    (write : Memory K V → KeyVec K → ValueVec V → Memory K V) : Prop :=
  ∀ (S : Memory K V) (k : KeyVec K) (v : ValueVec V),
    keyDot k k = 1 → read (write S k v) k = v

/-- Uniform orthogonal preservation: a write at key `k` leaves any orthogonal
query `q` unchanged. -/
def UniformOrthogonalPreservation
    (write : Memory K V → KeyVec K → ValueVec V → Memory K V) : Prop :=
  ∀ (S : Memory K V) (k q : KeyVec K) (v : ValueVec V),
    keyDot k q = 0 → read (write S k v) q = read S q

/-- A model family has one-step exact overwrite semantics for orthogonal keys if
its write operation satisfies exact overwrite and non-target preservation.

This is intentionally an operational spec, not a full architecture definition.
It lets later modules compare which update families implement the spec directly
and which need extra heads, dimensions, layers, or recurrent steps. -/
structure OneStepOverwriteSpec (K V : Nat) where
  write : Memory K V → KeyVec K → ValueVec V → Memory K V
  exactOverwrite :
    ∀ (S : Memory K V) (k : KeyVec K) (v : ValueVec V),
      keyDot k k = 1 → read (write S k v) k = v
  preservesOrthogonal :
    ∀ (S : Memory K V) (k q : KeyVec K) (v : ValueVec V),
      keyDot k q = 0 → read (write S k v) q = read S q

/-- The ideal delta rule directly satisfies one-step overwrite semantics. -/
def linearDeltaOneStepOverwriteSpec (K V : Nat) : OneStepOverwriteSpec K V where
  write := linearDeltaWrite
  exactOverwrite := by
    intro S k v hk
    exact linearDeltaWrite_exact_overwrite S k v hk
  preservesOrthogonal := by
    intro S k q v horth
    exact linearDeltaWrite_preserves_orthogonal_query S k q v horth

/-- The ideal delta rule satisfies the uniform one-step overwrite target. -/
theorem linearDeltaWrite_uniformOneStepOverwrite :
    UniformOneStepOverwrite (K := K) (V := V) linearDeltaWrite := by
  intro S k v hk
  exact linearDeltaWrite_exact_overwrite S k v hk

/-- The ideal delta rule satisfies the uniform non-target preservation target. -/
theorem linearDeltaWrite_uniformOrthogonalPreservation :
    UniformOrthogonalPreservation (K := K) (V := V) linearDeltaWrite := by
  intro S k q v horth
  exact linearDeltaWrite_preserves_orthogonal_query S k q v horth

/-- Gated DeltaNet's ideal delta core satisfies the uniform overwrite target. -/
theorem gdnIdealWrite_uniformOneStepOverwrite :
    UniformOneStepOverwrite (K := K) (V := V) gdnIdealWrite := by
  intro S k v hk
  exact linearDeltaWrite_exact_overwrite S k v hk

/-- E88/NDM's pre-nonlinearity delta core satisfies the uniform overwrite target. -/
theorem ndmPreTanhWrite_uniformOneStepOverwrite :
    UniformOneStepOverwrite (K := K) (V := V) ndmPreTanhWrite := by
  intro S k v hk
  exact linearDeltaWrite_exact_overwrite S k v hk

/-- A structure-level spec entails the two uniform operational obligations for
its write function. -/
theorem oneStepOverwriteSpec_has_uniform_obligations
    (spec : OneStepOverwriteSpec K V) :
    UniformOneStepOverwrite spec.write ∧
    UniformOrthogonalPreservation spec.write := by
  constructor
  · intro S k v hk
    exact spec.exactOverwrite S k v hk
  · intro S k q v horth
    exact spec.preservesOrthogonal S k q v horth

/-- The two uniform obligations are exactly the data needed to package a write
function as a one-step overwrite spec. -/
theorem exists_oneStepOverwriteSpec_iff_uniform_obligations
    (write : Memory K V → KeyVec K → ValueVec V → Memory K V) :
    (∃ spec : OneStepOverwriteSpec K V, spec.write = write) ↔
    UniformOneStepOverwrite write ∧
    UniformOrthogonalPreservation write := by
  constructor
  · rintro ⟨spec, hwrite⟩
    rw [← hwrite]
    exact oneStepOverwriteSpec_has_uniform_obligations spec
  · intro h
    refine ⟨{
      write := write
      exactOverwrite := ?_
      preservesOrthogonal := ?_
    }, rfl⟩
    · exact h.1
    · exact h.2

/-- A state-independent additive write cannot satisfy the same uniform overwrite
target when there exist two old memories with different content at the addressed
key.

This is the spec-level separation: delta memory satisfies the uniform one-step
overwrite obligation; raw/state-independent additive writes cannot satisfy it
uniformly unless the old addressed content is somehow already fixed or canceled
by extra machinery. -/
theorem stateIndependentAdditiveWrite_not_uniformOneStepOverwrite
    (term : KeyVec K → ValueVec V → Memory K V)
    (S₁ S₂ : Memory K V) (k : KeyVec K) (v : ValueVec V) (j : Fin V)
    (hunit : keyDot k k = 1)
    (hread : read S₁ k j ≠ read S₂ k j) :
    ¬ UniformOneStepOverwrite (K := K) (V := V)
      (stateIndependentAdditiveWrite term) := by
  intro hspec
  exact stateIndependentAdditiveWrite_cannot_exact_overwrite_two_memories
    term S₁ S₂ k v j hread ⟨hspec S₁ k v hunit, hspec S₂ k v hunit⟩

/-- The raw outer-product association primitive used by M2RNN-style matrix
state updates cannot be a uniform one-step overwrite memory when two old
memories disagree at the addressed key. -/
theorem rawOuterWrite_not_uniformOneStepOverwrite
    (S₁ S₂ : Memory K V) (k : KeyVec K) (v : ValueVec V) (j : Fin V)
    (hunit : keyDot k k = 1)
    (hread : read S₁ k j ≠ read S₂ k j) :
    ¬ UniformOneStepOverwrite (K := K) (V := V) rawOuterWrite := by
  intro hspec
  let term : KeyVec K → ValueVec V → Memory K V :=
    fun k v => M2RNNComparison.outerKV k v
  have h₁ :
      read (stateIndependentAdditiveWrite term S₁ k v) k = v := by
    simpa [term, stateIndependentAdditiveWrite, rawOuterWrite]
      using hspec S₁ k v hunit
  have h₂ :
      read (stateIndependentAdditiveWrite term S₂ k v) k = v := by
    simpa [term, stateIndependentAdditiveWrite, rawOuterWrite]
      using hspec S₂ k v hunit
  exact stateIndependentAdditiveWrite_cannot_exact_overwrite_two_memories
    term S₁ S₂ k v j hread ⟨h₁, h₂⟩

/-- Memory-semantics separation for the direct comparison with M2RNN:
GDN's ideal delta core and E88/NDM's pre-tanh delta core satisfy uniform
one-step overwrite, while M2RNN's raw association primitive cannot satisfy that
same target without extra state-dependent correction or other machinery. -/
theorem delta_core_separates_gdn_ndm_from_m2rnn_raw_write
    (S₁ S₂ : Memory K V) (k : KeyVec K) (v : ValueVec V) (j : Fin V)
    (hunit : keyDot k k = 1)
    (hread : read S₁ k j ≠ read S₂ k j) :
    UniformOneStepOverwrite (K := K) (V := V) gdnIdealWrite ∧
    UniformOneStepOverwrite (K := K) (V := V) ndmPreTanhWrite ∧
    ¬ UniformOneStepOverwrite (K := K) (V := V) rawOuterWrite := by
  constructor
  · exact gdnIdealWrite_uniformOneStepOverwrite
  constructor
  · exact ndmPreTanhWrite_uniformOneStepOverwrite
  · exact rawOuterWrite_not_uniformOneStepOverwrite S₁ S₂ k v j hunit hread

end OnlineMemory
