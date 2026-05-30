# E97 / GDN-2 Formal Proof Plan

Task: `e97-gdn-2-formal`

Date: 2026-05-30

This is a formal target audit only. It does not modify Lean, Python, model,
checkpoint, HuggingFace, or generated artifacts.

## Source Boundary

Use these sources as the local specification for proof work:

- `docs/HANDOFF_E97_GDN2_CMAES_20260528.md:75-104` gives the E88 and proposed
  E97 update rules, and says E97 keeps E88's scalar per-head decay, output path,
  and residual wrapper while adding key-axis erase/read and value-axis write
  gates.
- `docs/GDN2_E97_NOTES.md:16-28` records the GDN-2 split erase/write recurrence;
  `docs/GDN2_E97_NOTES.md:41-65` records the E97 candidate and implementation
  status.
- `ndm/models/e88_fla_hybrid.py:1003-1008` defines the E97 split-edit gate
  projections; `ndm/models/e88_fla_hybrid.py:1687-1712` is the PyTorch reference
  recurrence.
- `/home/erikg/GatedDeltaNet-2/README.md:31-39` and
  `/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py:10-22` give the
  published GDN-2 linear recurrence. Treat these as specification sources only;
  do not vendor code.
- `/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2_ops/fused_recurrent_gdn2.py:18-28`
  and `:218-242` are useful for decay placement: the recurrent path decays the
  state first, then reads with `b * k`, writes `w * v`, and adds `k * v_new^T`.

## Current E88 Target

The currently trusted Lean core formalizes the per-head matrix-state delta
update, not the whole language-model block:

```text
read(H, k)  = H^T k
delta       = v - read(H, k)
E88_direct  = tanh(lambda H + k delta^T)
E88_expand  = tanh((lambda I - k k^T) H + k v^T)
```

Relevant existing Lean:

- `ElmanProofs.Architectures.M2RNNComparison`
  - `e88Delta` at `formal/lean/ElmanProofs/Architectures/M2RNNComparison.lean:87-92`
  - `e88DeltaUpdateDirect` at `:94-100`
  - `e88DeltaTransition` and `e88DeltaUpdateExpanded` at `:102-124`
  - `e88DeltaUpdateDirect_eq_expanded` at `:258-303`
- `ElmanProofs.Architectures.OnlineMemory`
  - `linearDeltaWrite` at `formal/lean/ElmanProofs/Architectures/OnlineMemory.lean:57-63`
  - `gdn_and_emender_share_ideal_delta_write` and
    `shared_delta_core_exact_overwrite` at `:241-255`
- `ElmanProofs.Architectures.RecurrentResourceFormalism`
  - `emender` / `e88NDM` signatures at
    `formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean:186-212`
  - the current per-head cost model and theorem
    `emender_m2rnn_flop_class_equiv` at `:1100-1174`

This is the correct starting point. Do not re-formalize the full residual stack,
output projection, output gate, short convolution, or optimizer behavior for
this comparison.

## E97 Formal Target

Use the per-head variables:

```text
H : K x V matrix state
k : K key
q : K query
v : V value
b : K erase/read gate
w : V write gate
lambda : scalar decay
```

The proposed E97 direct update is:

```text
read_key     = b * k
write_value  = w * v
retrieved    = H^T read_key
delta        = write_value - retrieved
E97_direct   = tanh(lambda H + k delta^T)
readout      = E97_direct^T q
```

The expanded split-gated left-transition form is:

```text
E97_expanded =
  tanh((lambda I - k (b * k)^T) H + k (w * v)^T)
```

This is the main algebraic theorem target. It follows the existing
`e88DeltaUpdateDirect_eq_expanded` proof almost exactly, replacing `k k^T` with
`k (b * k)^T` and replacing `v` with `w * v`.

When `b = 1_K` and `w = 1_V`, E97 specializes to E88:

```text
E97_direct lambda H k 1_K 1_V v = e88DeltaUpdateDirect lambda H k v
E97_expanded lambda H k 1_K 1_V v = e88DeltaUpdateExpanded lambda H k v
```

That specialization is the useful Lean form of "E97 is at least as expressive
as E88." It is constructive and should be proved now. A strict expressivity
theorem should not be attempted unless there is a small finite witness whose
statement does not smuggle in trainability, optimization, or distributional
assumptions.

## GDN-2 Formal Target

The GDN-2 source recurrence is linear in the recurrent state:

```text
GDN2_linear =
  (I - k (b * k)^T) D H + k (w * v)^T
```

The local token-serial kernel spells the same update as:

```text
H0          = D H
v_new       = (w * v) - H0^T (b * k)
GDN2_linear = H0 + k v_new^T
```

Important caveat: current E97 reads from `H` and separately adds
`lambda H`, while GDN-2 decays first and then performs the gated read from
`D H`. Therefore the exact GDN-2/E97 theorem should not claim unconditional
equality between current E97 and published GDN-2 when `D` is non-identity.

The useful formal relationship is:

```text
GDN2_linear(D, H, k, b, w, v)
  = split_gated_linear_core(1, D H, k, b, w, v)

E97_linear(lambda, H, k, b, w, v)
  = split_gated_linear_core(lambda, H, k, b, w, v)
```

So E97 and GDN-2 share the same split-gated read/write core; E97 adds an
elementwise nonlinear state activation, and current E97 keeps E88's scalar
decay placement. The identity-decay theorem is exact:

```text
GDN2_linear(I, H, k, b, w, v)
  = e97_linear_core(1, H, k, b, w, v)
```

If downstream work wants exact equality for non-identity decay, the architecture
must choose one of these formally distinct targets:

- Current E97: `lambda H + k((w*v) - H^T(b*k))^T`.
- GDN-2-aligned E97-channel-decay: `(I - k(b*k)^T) D H + k(w*v)^T`, followed
  by optional `tanh`.

## Lean Modules And Theorems

Add a new module first:

```text
formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean
```

Recommended definitions:

```lean
namespace SplitGatedDelta

def hadamard {n : Nat} (a b : M2RNNComparison.Vec n) : M2RNNComparison.Vec n
def onesVec (n : Nat) : M2RNNComparison.Vec n

def splitGatedDelta
    (H : M2RNNComparison.MatState K V)
    (k b : M2RNNComparison.Vec K)
    (w v : M2RNNComparison.Vec V) :
    M2RNNComparison.Vec V

def splitGatedTransition
    (lambda : Real) (k b : M2RNNComparison.Vec K) :
    Matrix (Fin K) (Fin K) Real

def e97LinearCore
    (lambda : Real) (H : M2RNNComparison.MatState K V)
    (k b : M2RNNComparison.Vec K)
    (w v : M2RNNComparison.Vec V) :
    M2RNNComparison.MatState K V

noncomputable def e97UpdateDirect ...
noncomputable def e97UpdateExpanded ...

def gdn2LinearCore
    (D : Matrix (Fin K) (Fin K) Real)
    (H : M2RNNComparison.MatState K V)
    (k b : M2RNNComparison.Vec K)
    (w v : M2RNNComparison.Vec V) :
    M2RNNComparison.MatState K V
```

Recommended theorem names for the first Lean pass:

```lean
theorem e97UpdateDirect_eq_expanded
theorem e97LinearCore_eq_expanded
theorem e97_specializes_to_e88_all_one_gates_direct
theorem e97_specializes_to_e88_all_one_gates_expanded
theorem gdn2LinearCore_eq_e97LinearCore_on_decayed_state
theorem gdn2LinearCore_identity_decay_eq_e97LinearCore_one
theorem e97_and_gdn2_share_split_gated_linear_core
theorem e97_expresses_e88_by_specialization
```

Only after `SplitGatedDelta.lean` compiles without `sorry`, update:

```text
formal/lean/ElmanProofs/PaperCore.lean
```

to import `ElmanProofs.Architectures.SplitGatedDelta`. If downstream wants a
resource-signature theorem, add it in:

```text
formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean
```

with a new `gdn2` signature rather than overloading the existing
`gatedDeltaNet` signature, because the existing one represents the older scalar
gate family.

## Cost And Efficiency Claims

Lean should prove only algebraic and combinatorial cost claims here.

Under the current cost model, E88 per-head update cost is `6 * S`, where
`S = K * V` state scalars. With precomputed gates, E97 has the same rank-one
state read, rank-one correction, outer write, activation, and state write costs,
plus lower-order gate applications:

```text
read_key    = b * k       costs K scalar multiplications
write_value = w * v       costs V scalar multiplications
```

For square heads `K = V = d`, the recommended cost definitions are:

```lean
def e88UpdateCost (d : Nat) : Nat := 6 * (d * d)
def e97UpdateCostPrecomputedGates (d : Nat) : Nat :=
  6 * (d * d) + 2 * d
```

Recommended theorem names:

```lean
theorem e97_precomputed_gate_cost_eq_e88_plus_linear_overhead
theorem e97_precomputed_gate_cost_le_quadratic_bound
theorem e97_same_leading_update_cost_class_as_e88
```

For the bound theorem, use a concrete finite statement such as:

```text
2 <= d -> e97UpdateCostPrecomputedGates d <= 8 * (d * d)
```

This proves "same leading per-head update cost class" in the existing style.
It does not prove speed, wallclock throughput, memory traffic, kernel occupancy,
or optimizer efficiency.

Lean cannot answer "more efficient than delta alone" unless the phrase is
reduced to an explicit formal metric. It can answer:

- exact equality of direct and expanded algebraic forms;
- exact specialization of E97 to E88 with all-one gates;
- equality between GDN-2 and the shared split-gated linear core under the
  correct decay convention;
- finite operation-count upper bounds under a declared cost model.

Lean cannot answer:

- whether E97 trains better than E88;
- whether split gates improve loss, sample efficiency, or state tracking;
- whether the extra gate projections are worth their parameter or bandwidth
  cost in a real kernel;
- whether a Triton implementation is faster than the E88 kernel;
- whether "more efficient than delta alone" is true as an empirical claim.

Those are CMA/probe/kernel-validation claims, not theorem claims.

## Recommendation

Formal work should proceed, but only on the staged algebra and cost targets
above. The first useful Lean artifact is `SplitGatedDelta.lean` with the direct
vs expanded E97 theorem and all-one specialization theorem. The second useful
artifact is a small cost theorem showing E97 has only linear gate-application
overhead when gates are already computed. Defer strict expressivity separation
until a clean finite witness exists.
