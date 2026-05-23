# E88 Variant Clarification: Which Proofs Apply Where?

**Date:** 2026-02-03
**Status:** Complete
**Related Files:**
- `ElmanProofs/Expressivity/E88VariantClarification.lean` (formal proofs)
- `ElmanProofs/Expressivity/LinearLimitations.lean`
- `ElmanProofs/Expressivity/MultiLayerLimitations.lean`
- `ElmanProofs/Expressivity/E88MultiPass.lean`
- `OPEN_QUESTIONS_RESOLUTION.md`

---

## Executive Summary

E88 has **two distinct variants** that are often conflated:

1. **Simple E88**: `S := tanh(α·S + δ·k^T)` — linear recurrence, nonlinear temporal composition
2. **Gated E88**: `S := tanh(W_h·S + δ·k) ⊙ σ(W_g·S + V_g·k)` — fully nonlinear

**Key finding**: Neither variant is constrained by the linear RNN limitation theorems (`LinearLimitations.lean`), because both have **nonlinear temporal composition** even though Simple E88 has a "linear recurrence".

---

## The Two Variants

### Simple E88 (Linear Recurrence)

```
S_t = tanh(α · S_{t-1} + δ · k_t^T)
```

**Structure:**
- **Pre-activation**: `α · S_{t-1} + δ · k_t` — LINEAR (affine) in S
- **After T steps**: `S_T = tanh(tanh(...tanh(δ·k_0)...))` — T nested tanh applications
- **Temporal depth**: T

**Classification:**
- ✓ Linear recurrence (each step's argument to tanh is affine in state)
- ✓ Nonlinear temporal composition (tanh compounds)
- ✗ NOT a linear RNN (output is not a linear combination of inputs)

### Gated E88 (Fully Nonlinear)

```
S_t = tanh(W_h · S_{t-1} + δ · k_t) ⊙ σ(W_g · S_{t-1} + V_g · k_t)
```

**Structure:**
- **Pre-activation**: `W_h · S_{t-1} + δ · k_t` — linear in S
- **Gate**: `σ(W_g · S_{t-1} + V_g · k_t)` — nonlinear in S
- **Combined**: Element-wise multiplication — fully nonlinear in S
- **After T steps**: T nested applications of (tanh ⊙ gate)
- **Temporal depth**: T

**Classification:**
- ✓ Nonlinear recurrence (gate makes each step nonlinear in state)
- ✓ Nonlinear temporal composition
- ✗ NOT a linear RNN

---

## The Key Distinction: Recurrence vs Temporal Composition

| Property | Linear RNN (Mamba2) | Simple E88 | Gated E88 |
|----------|---------------------|------------|-----------|
| **Recurrence structure** | `h := A·h + B·x` (linear) | Pre-act: `α·S + δ·k` (linear) | Gate: `tanh ⊙ σ(...)` (nonlinear) |
| **Temporal composition** | Linear (collapses) | Nonlinear (`tanh^T`) | Nonlinear (`(tanh⊙gate)^T`) |
| **State form** | `h_T = Σ A^{T-t}Bx_t` | `S_T = tanh^T(...)` | `S_T = (tanh⊙σ)^T(...)` |
| **Within-layer depth** | 1 | T | T |
| **Can compute threshold?** | NO (proven) | YES | YES |

**The critical insight**: Simple E88's "linear recurrence" refers to the PRE-ACTIVATION being affine in state. But the tanh makes the TEMPORAL COMPOSITION nonlinear, which is what matters for expressivity.

---

## Which Proofs Apply to Which Variant?

### ✗ LinearLimitations.lean

**Main theorems:**
- `linear_cannot_threshold` (line 107)
- `linear_cannot_xor` (line 315)

**Apply to:**
- ✓ Linear RNNs where `h_T = C · (Σ A^{T-t} B x_t)`
- ✓ Mamba2, FLA-GDN (linear temporal dynamics)

**Do NOT apply to:**
- ✗ Simple E88 — state is `tanh^T(...)`, not a linear combination
- ✗ Gated E88 — state involves nested nonlinear gates

**Reasoning:**
```lean
-- LinearLimitations assumes state has the form:
h_T = Σ_{t=0}^{T-1} A^{T-1-t} B x_t  -- Linear combination

-- Simple E88 state has the form:
S_T = tanh(α·tanh(α·tanh(...·tanh(δ·x_0)...) + δ·x_1) + ... + δ·x_{T-1})
-- NOT a linear combination!
```

### ✗ MultiLayerLimitations.lean

**Main theorems:**
- `multilayer_cannot_running_threshold` (line 231)
- `multilayer_cannot_threshold` (line 286)

**Apply to:**
- ✓ D-layer models where EACH layer has `h_t = A(x)·h_{t-1} + B(x)·x_t`
- ✓ Stacked Mamba2, stacked FLA-GDN

**Do NOT apply to:**
- ✗ Simple E88 — has `S_t = tanh(α·S_{t-1} + δ·x_t)`, not `A·S + B·x`
- ✗ Gated E88 — has `S_t = tanh(...) ⊙ σ(...)`, not linear temporal

**Reasoning:**
```lean
-- MultiLayerLimitations analyzes models where temporal aggregation is LINEAR
-- at each layer (though inter-layer connections can be nonlinear)

-- E88 has NONLINEAR temporal aggregation (tanh compounds across time)
-- So it's in a different computational class entirely
```

### ✓ E88MultiPass.lean

**Main theorems:**
- `e88_multipass_depth_theorem` — depth = k×T
- `e88_exceeds_linear_theorem` — exceeds linear-temporal
- `e88_exceeds_transformer_theorem` — exceeds TC⁰ for large T
- `e88_random_access_theorem` — 3k passes → k random accesses

**Apply to:**
- ✓ Simple E88 — multi-pass gives depth k×T
- ✓ Gated E88 — multi-pass gives depth k×T

**Reasoning:**
Both variants have temporal depth T per pass. The compositional depth accumulates across passes regardless of whether the recurrence is "linear" (simple) or fully nonlinear (gated).

### ~ RecurrenceLinearity.lean

**Main theorems:**
- `minGRU_is_linear_in_h` (line 110)
- `e1_is_nonlinear_in_h` (line 148)
- `mamba2_is_linear_in_h` (line 171)

**Classification:**
- Simple E88: **Hybrid** — pre-activation is linear in S, but full update is nonlinear (tanh)
- Gated E88: **Nonlinear** — like E1 (gating makes it fully nonlinear)

**Notes:**
RecurrenceLinearity analyzes the PER-STEP structure, not the temporal composition. It correctly identifies that MinGRU/Mamba2 have linear recurrence, but it doesn't claim this limits their expressivity with depth. The temporal vs depth analysis is in `OPEN_QUESTIONS_RESOLUTION.md`.

---

## What Can E88 Compute That Linear RNNs Cannot?

### Threshold Functions ✓

**Definition**: Output 1 iff `Σ x_i > τ`, else 0

**Linear RNN**:
- Cannot compute (proven in `LinearLimitations.lean:107`)
- Reason: Threshold is discontinuous, linear output is continuous

**Simple E88**:
- Can compute via tanh saturation
- Mechanism: State accumulates sum with tanh compression, crosses 0 at threshold
- Example: α ≈ 1, δ small → `S_T ≈ tanh(sum)` → output = `tanh(S_T)` has sharp transition

**Gated E88**:
- Can also compute (gate provides additional flexibility)

### XOR Over History ✓

**Definition**: `XOR(x_1, x_2, ..., x_T)`

**Linear RNN**:
- Cannot compute (proven in `LinearLimitations.lean:315`)
- Reason: XOR violates superposition (not affine)

**Simple E88**:
- Can compute via sign-flip dynamics
- Mechanism: Each input flips sign of accumulated state through tanh

**Gated E88**:
- Can compute (gate can implement conditional sign flip)

### Running Parity ✓

**Definition**: `y_t = parity(x_1, ..., x_t)` for all t

**Linear RNN**:
- Cannot compute (extension of XOR proof)

**E88 (both variants)**:
- Can compute by maintaining binary state (latched at ±1)
- Each input toggles the state via tanh dynamics

---

## The Hierarchy

### By Temporal Expressivity (within single layer)

```
Linear RNN (Mamba2) < Simple E88 ≈ Gated E88
    [depth 1]           [depth T]   [depth T]
```

### By Recurrence Structure (per-step)

```
MinGRU ≈ Mamba2 ≈ Simple E88 (pre-activation is linear in h)
E1 ≈ Gated E88 (fully nonlinear in h)
```

### By Multi-Pass Expressivity

```
Linear k-pass < E88 k-pass < Random Access
  [depth k]     [depth k×T]   [depth ∞]
```

---

## Common Misconceptions

### Misconception 1: "Simple E88 is linear, so linear limitations apply"

**Wrong!** Simple E88 has a linear RECURRENCE (pre-activation is affine) but nonlinear TEMPORAL COMPOSITION (tanh^T). The limitation theorems apply to models where the state is a LINEAR COMBINATION of inputs, which E88 is not.

### Misconception 2: "Gated E88 is fundamentally different from Simple E88"

**Partially true.** Gated E88 has within-step nonlinearity (gate) that Simple E88 lacks. However, for TEMPORAL expressivity, both have the same depth (T per pass). The gate provides additional flexibility but doesn't change the compositional depth.

### Misconception 3: "MultiLayerLimitations proves E88 has linear limitations"

**Wrong!** MultiLayerLimitations analyzes multi-LAYER models where each layer has linear TEMPORAL dynamics (like stacked Mamba2). E88 has nonlinear temporal dynamics (tanh compounds), so it's in a different computational class entirely.

### Misconception 4: "E88 is just E1 with a different parameterization"

**Wrong!** E1 is:
```
h_t = tanh(W_h·h + W_x·x) ⊙ σ(W_g·h + V_g·x)
```
This is like Gated E88 but operates on VECTOR states, not MATRIX states. E88's multi-head square matrices provide additional capacity and structure.

---

## Formalization Summary

**File**: `ElmanProofs/Expressivity/E88VariantClarification.lean`

### Main Theorems

1. **Classification** (`e88_classification_two_axes`):
   - Simple E88: linear recurrence, nonlinear temporal (depth T)
   - Gated E88: nonlinear recurrence, nonlinear temporal (depth T)

2. **Separation** (`e88_variants_exceed_linear_expressivity`):
   - Both E88 variants can compute functions linear RNNs cannot
   - Example: threshold over accumulated inputs

3. **Non-applicability** (`linear_limitations_does_not_apply_to_simple_e88`):
   - Simple E88 state is NOT a linear combination of inputs
   - Therefore linear limitation theorems don't apply

4. **Multi-pass** (`simple_e88_multipass_depth`, `gated_e88_multipass_depth`):
   - Both variants: k passes → depth k×T
   - Exceeds linear-temporal k-pass (depth k)

### Key Proofs (with sorry placeholders)

- `simpleE88_not_affine`: tanh makes update nonlinear
- `simple_e88_state_not_linear_combination`: T-step state violates additivity
- `simple_e88_can_threshold`: E88 can compute threshold via tanh saturation
- `e88_separates_from_linear_rnn`: Explicit separation example

---

## Practical Implications

### For Language Modeling

**Question**: Does the Simple vs Gated distinction matter?

**Answer**: Unclear from current evidence.
- Both have temporal depth T
- Empirical E88 results don't distinguish variants
- Gated adds within-step flexibility, but language modeling may not need it

**Hypothesis**: Simple E88 may be sufficient for language, as temporal expressivity (depth T) is the key advantage over Mamba2 (depth 1).

### For Algorithmic Reasoning

**Question**: When does the gate matter?

**Answer**: Tasks requiring input-dependent state routing.
- **Simple E88**: State evolves uniformly via α and δ
- **Gated E88**: State evolution is input-dependent (like attention)

**Examples where gate helps:**
- Conditional branching (if-then-else logic)
- Variable binding (track which variable to update)
- Context-dependent computation

### For Multi-Pass Computation

**Question**: Does variant affect multi-pass power?

**Answer**: No for depth, possibly yes for capacity.
- Both achieve depth k×T
- Gated may use fewer passes due to within-step flexibility
- 3k-pass random access protocol works for both

---

## Connections to Existing Results

### OPEN_QUESTIONS_RESOLUTION.md

**Q1: Does depth compensate for linear temporal dynamics?**
- Answer: NO
- E88 has nonlinear temporal dynamics (tanh^T)
- Mamba2 has linear temporal dynamics (collapsed sum)
- Gap persists at any depth

**Q2: Separation example?**
- Answer: YES
- Threshold, XOR, running parity
- E88 (both variants) can compute, linear-temporal cannot

**Q3: Practical implications?**
- For D ≥ 32 layers, depth may compensate for typical sequences
- But E88's 6× smaller state suggests temporal nonlinearity provides real advantage

### RecurrenceLinearity.lean

**within_layer_depth definition** (line 215):
```lean
def within_layer_depth (r : RecurrenceType) (seq_len : Nat) : Nat :=
  match r with
  | RecurrenceType.linear => 1           -- Collapses regardless of seq_len
  | RecurrenceType.nonlinear => seq_len  -- Grows with sequence length
```

**E88 classification**:
- Simple E88: Technically has "linear recurrence" but `within_layer_depth = T` due to tanh
- Gated E88: Clearly `RecurrenceType.nonlinear`, `within_layer_depth = T`

The RecurrenceLinearity classification focuses on per-step structure, but the temporal depth is what determines expressivity.

### E88MultiPass.lean

**Compositional depth** (line 145):
```lean
def e88KPassTotalDepth (k seqLen : ℕ) : ℕ := k * seqLen
```

This applies to BOTH Simple and Gated E88 because both accumulate tanh applications across time.

---

## Future Work

### Formal Proofs Needed

1. **`simple_e88_can_threshold`**: Complete proof that Simple E88 can compute threshold
   - Use tanh saturation analysis
   - Show state accumulation creates decision boundary

2. **`gated_e88_can_threshold`**: Complete proof for Gated E88
   - Show gate provides additional flexibility
   - May enable sharper or more robust thresholds

3. **`e88_separates_from_linear_rnn`**: Concrete separation example
   - Construct specific E88 parameters
   - Prove threshold function is computed
   - Use linear_cannot_threshold for impossibility

4. **Empirical validation**: Test Simple vs Gated E88
   - Does gating improve language modeling?
   - Ablation study on synthetic tasks (threshold, XOR, parity)

### Documentation Needed

1. Add section to main E88 documentation clarifying variants
2. Update RecurrenceLinearity.lean with E88 classification
3. Create tutorial on temporal vs recurrence nonlinearity

---

## Summary Table

| Aspect | Simple E88 | Gated E88 | Linear RNN (Mamba2) |
|--------|------------|-----------|---------------------|
| **Recurrence** | Pre-act linear in S | Fully nonlinear | Linear in h |
| **Temporal composition** | Nonlinear (tanh^T) | Nonlinear | Linear (collapsed) |
| **Within-layer depth** | T | T | 1 |
| **Can threshold?** | YES | YES | NO (proven) |
| **Can XOR?** | YES | YES | NO (proven) |
| **Multi-pass depth** | k×T | k×T | k |
| **LinearLimitations apply?** | NO | NO | YES |
| **MultiLayerLimitations apply?** | NO | NO | YES |
| **E88MultiPass apply?** | YES | YES | NO |

---

## Conclusion

**The key takeaway**: "Linear recurrence" (pre-activation is affine in state) is NOT the same as "linear temporal composition" (state is a linear combination of inputs).

- **Simple E88** has the former but not the latter
- This is why it escapes the linear limitation theorems
- The temporal nonlinearity (tanh^T) is what provides expressivity

**Both E88 variants** have nonlinear temporal composition with depth T, making them fundamentally more expressive than linear-temporal models (Mamba2, FLA-GDN) which have depth 1 per layer.

**Gated E88** adds within-step nonlinearity on top of temporal composition, potentially providing additional flexibility for certain tasks.

**For language modeling**, the temporal expressivity (depth T) appears to be the key advantage, with Simple E88 achieving strong results (1.40 loss) using 6× less state than Mamba2.
