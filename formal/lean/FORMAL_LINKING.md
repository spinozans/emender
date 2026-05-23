# Formal Linking: Connecting the Spectral Theory to the Existing Edifice

## The Current State

We have **isolated islands** of proven results that need **bridges**:

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   LinearCapacity.lean   │     │    SpectralLowRank.lean │
│                         │     │                         │
│ • State = Σ A^k B x     │     │ • Power law spectrum    │
│ • dim(reachable) ≤ n    │     │ • Condition grows: r^α  │
│ • Same state → same out │     │ • Manifold dimension    │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            │    GAP: How does rank of W    │
            │    affect reachable states?   │
            │                               │
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  LowRankCapacity.lean   │     │      Flow.lean          │
│                         │     │                         │
│ • E5 has 3x capacity    │     │ • O(1/k) convergence    │
│ • Efficiency ordering   │     │ • L-smooth bounds       │
│ • SNR increases         │     │ • Strong convex interp  │
└─────────────────────────┘     └─────────────────────────┘
            │                               │
            │    GAP: How does condition    │
            │    number affect convergence? │
            │                               │
            └───────────────┬───────────────┘
                            ▼
                    OPTIMAL RANK THEORY
                    (needs all bridges)
```

---

## Critical Distinction: Two Regimes

**We must be precise about what we're formalizing.**

### Regime 1: Low-Rank Approximation (Compression)

Given a dense matrix W with spectrum σ₁ ≥ σ₂ ≥ ... ≥ σ_d:
- Truncate to rank r: keep top r singular values
- W_r = Σᵢ≤r σᵢ uᵢ vᵢᵀ (best rank-r approximation)
- Approximation error: ‖W - W_r‖²_F = Σᵢ>r σᵢ²

**This is pure linear algebra. We can prove everything about it.**

### Regime 2: Low-Rank Training (Optimization)

Constrain W = U·V where U ∈ ℝ^{d×r}, V ∈ ℝ^{r×d}:
- Gradient descent finds optimal rank-r W for the task
- The resulting spectrum is NOT predetermined
- It emerges from the optimization process

**This involves optimization dynamics. Much harder to formalize.**

### The Implicit Assumption

Our theory assumes: "Regime 2 result ≈ Regime 1 result"

That is: The learned low-rank W is approximately the best rank-r approximation of what a full-rank W would have learned.

**This assumption needs empirical validation, not formal proof.**

---

## What We Can Prove Without Empirics

### Bridge 1: Spectrum → Variance (Pure Analysis)

**Theorem to prove:**
```lean
/-- For power law spectrum, variance in top r is approximately r^{1-2α}/(1-2α) -/
theorem powerLaw_variance_partial_sum (α : ℝ) (hα : α > 1/2) (r : ℕ) (hr : r > 0) :
    let variance_r := ∑ i in Finset.range r, powerLawSigma i α ^ 2
    ∃ C₁ C₂ : ℝ, C₁ > 0 ∧ C₂ > 0 ∧
      C₁ * (r : ℝ)^(1 - 2*α) ≤ variance_r ∧ variance_r ≤ C₂ * (r : ℝ)^(1 - 2*α)
```

**Proof approach:**
1. Compare sum to integral: Σᵢ≤r i^{-2α} ≈ ∫₁ʳ x^{-2α} dx
2. Compute integral: [x^{1-2α}/(1-2α)]₁ʳ = (r^{1-2α} - 1)/(1-2α)
3. For α > 1/2: exponent 1-2α < 0, so integral converges

**Dependencies:** Only needs Mathlib analysis (integrals, series bounds).

---

### Bridge 2: Variance → Optimal Rank (Pure Algebra)

**Theorem to prove:**
```lean
/-- The rank needed to capture (1-ε) fraction of variance -/
theorem optimal_rank_for_variance (α ε : ℝ) (hα : α > 1/2) (hε : 0 < ε ∧ ε < 1)
    (d : ℕ) (hd : d > 0) :
    let r_opt := (ε : ℝ)^(1/(2*α - 1)) * d
    let variance_ratio := (∑ i in Finset.range ⌊r_opt⌋₊, powerLawSigma i α ^ 2) /
                          (∑ i in Finset.range d, powerLawSigma i α ^ 2)
    variance_ratio ≥ 1 - ε
```

**Proof approach:**
1. Use Bridge 1 for numerator and denominator
2. Take ratio
3. Solve for r such that ratio ≥ 1 - ε

**Dependencies:** Bridge 1.

---

### Bridge 3: Rank → State Capacity (Linear Algebra)

**Key insight:** For LINEAR RNNs, state capacity = dim(reachable subspace), which depends on the SPAN of {B, AB, A²B, ...}, NOT on the rank of A.

**Theorem to prove:**
```lean
/-- Low-rank recurrence matrix doesn't reduce reachable dimension -/
theorem lowRank_preserves_reachable (A : Matrix (Fin d) (Fin d) ℝ) (B : Matrix (Fin d) (Fin m) ℝ)
    (U : Matrix (Fin d) (Fin r) ℝ) (V : Matrix (Fin r) (Fin d) ℝ)
    (T : ℕ) :
    -- The reachable subspace with A replaced by U*V has dimension
    -- that depends on the controllability matrix [B, (UV)B, (UV)²B, ...]
    -- which can still span all of ℝ^d if B is rich enough
    True  -- Placeholder for precise statement
```

**The precise statement is subtle:**
- If A = U·V has rank r, then A^k also has rank ≤ r
- BUT the span of {B, AB, A²B, ..., A^{d-1}B} can still be d-dimensional
- The controllability matrix can have rank d even if A has rank r

**Example:**
- d = 3, r = 1
- A = u·vᵀ where u = [1,0,0]ᵀ, v = [0,1,0]
- B = [1,0,0]ᵀ
- Then: B = [1,0,0]ᵀ, AB = [0,0,0]ᵀ, A²B = [0,0,0]ᵀ
- Reachable dimension = 1 (reduced!)

**Counterexample:**
- Same A, but B = [1,1,1]ᵀ
- Then reachable might be larger

**Conclusion:** Low-rank A CAN reduce reachable dimension. The theory needs refinement.

---

### Bridge 4: Condition Number → Convergence Rate

**Theorem to prove:**
```lean
/-- Convergence rate depends on condition number of Hessian -/
theorem condition_number_convergence (f : E → ℝ) (μ L : ℝ)
    (hμ : μ > 0) (hL : L > 0) (hμL : μ ≤ L)
    (hStrong : IsStronglyConvex f μ) (hSmooth : IsLSmooth f L)
    (x_star : E) (hMin : gradient f x_star = 0)
    (η : ℝ) (hη : η = 1/L) (x₀ : E) (k : ℕ) :
    let κ := L / μ  -- condition number
    ‖gdSequence f η x₀ k - x_star‖ ≤ (1 - 1/κ)^k * ‖x₀ - x_star‖
```

**This is the LINEAR convergence rate for strongly convex functions.**

**Current state:** We have `convex_convergence_rate` which gives O(1/k) for convex.
For strongly convex, we need to prove (1-μ/L)^k contraction.

**Proof approach:**
1. Use `strong_smooth_interpolation` (already proven)
2. Show ‖x_{k+1} - x*‖² ≤ (1-μ/L)² ‖x_k - x*‖²
3. Induct to get (1-μ/L)^{2k}

**Dependencies:** Existing Flow.lean theorems.

---

### Bridge 5: Low-Rank → Condition Number

**Theorem to prove:**
```lean
/-- For power law spectrum, rank-r approximation has condition number r^α -/
theorem lowRank_condition_number (α : ℝ) (hα : α > 0) (r : ℕ) (hr : r > 0) :
    let κ_r := powerLawCondition r α / powerLawCondition 0 α
    κ_r = ((r + 1) : ℝ)^α
```

**This is essentially already proven as `condition_grows`.**

The connection: If we use rank-r, the condition number is r^α. Combined with Bridge 4:
- Convergence rate ≈ (1 - 1/r^α)^k
- Lower r → faster convergence

---

## The Complete Chain (What We're Building)

```
Power law spectrum (ASSUMPTION about task structure)
         │
         ▼ [Bridge 1: variance sum formula]
Variance in top-r components
         │
         ▼ [Bridge 2: solve for r]
Optimal rank r* = ε^{1/(2α-1)} × d
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼ [Bridge 3: controllability]          ▼ [Bridge 5: condition number]
State capacity                            Condition number κ = r^α
(depends on controllability,                    │
 NOT just rank of A)                            ▼ [Bridge 4: convergence]
         │                                Learning rate ≈ (1 - 1/r^α)^k
         │                                      │
         └───────────────────┬──────────────────┘
                             ▼
                    TRADEOFF: Lower r means
                    - Possibly lower capacity (Bridge 3)
                    - Faster learning (Bridge 4-5)

                    Optimal r balances these.
```

---

## Immediate Next Steps (No Empirics Required)

### Step 1: Prove Variance Sum Formula

Create new file or extend SpectralLowRank.lean:

```lean
/-- Integral bound for power law sum -/
theorem powerLaw_sum_integral_bound (α : ℝ) (hα : α > 1/2) (r : ℕ) (hr : r > 0) :
    ∫ x in Set.Icc 1 r, x^(-2*α) ≤ ∑ i in Finset.range r, (i+1 : ℝ)^(-2*α) ∧
    ∑ i in Finset.range r, (i+1 : ℝ)^(-2*α) ≤ 1 + ∫ x in Set.Icc 1 r, x^(-2*α)
```

This is the **Integral Test** from analysis.

### Step 2: Compute the Integral

```lean
/-- Closed form for power law integral -/
theorem powerLaw_integral (α : ℝ) (hα : α ≠ 1/2) (a b : ℝ) (hab : 0 < a ∧ a < b) :
    ∫ x in Set.Icc a b, x^(-2*α) = (b^(1-2*α) - a^(1-2*α)) / (1 - 2*α)
```

### Step 3: Derive Variance Ratio

```lean
/-- Fraction of variance in top r components -/
theorem variance_fraction (α : ℝ) (hα : α > 1/2) (r d : ℕ) (hrd : r < d) :
    ∃ f : ℝ, f ≥ 1 - ((r : ℝ)/d)^(2*α - 1) ∧
    (∑ i in Finset.range r, powerLawSigma i α ^ 2) /
    (∑ i in Finset.range d, powerLawSigma i α ^ 2) ≥ f
```

### Step 4: Prove Linear Convergence

Extend Flow.lean:

```lean
/-- Linear convergence for strongly convex + smooth -/
theorem strong_convex_linear_convergence (f : E → ℝ) (μ L : ℝ)
    ... (existing hypotheses) ... :
    ‖x_k - x_star‖² ≤ (1 - μ/L)^k * ‖x₀ - x_star‖²
```

### Step 5: Connect Condition Number to Convergence

```lean
/-- Lower rank → faster convergence (up to capacity limit) -/
theorem rank_convergence_tradeoff (α : ℝ) (hα : α > 0) (r₁ r₂ : ℕ) (hr : r₁ < r₂) :
    -- Convergence factor (1 - 1/κ) is smaller (better) for smaller r
    (1 - 1 / powerLawCondition r₁ α) < (1 - 1 / powerLawCondition r₂ α)
```

---

## What CANNOT Be Proven Formally

1. **α ≈ 1.35 for language models** — Empirical observation
2. **ε = 0.05 is the right threshold** — Task-dependent
3. **Regime 2 ≈ Regime 1** — Optimization dynamics assumption
4. **Low-rank doesn't hurt capacity much** — Depends on task structure

These are the gaps that require experiments.

---

## Summary: The Formal Program

| Step | What to Prove | Dependencies | Difficulty |
|------|---------------|--------------|------------|
| 1 | Integral test for series | Mathlib.Analysis | Medium |
| 2 | Closed form for ∫x^{-2α} | Step 1 | Easy |
| 3 | Variance ratio formula | Steps 1-2 | Medium |
| 4 | Linear convergence rate | Flow.lean | Hard |
| 5 | Condition-convergence link | Steps 3-4 | Medium |
| 6 | Optimal rank formula | Steps 3, 5 | Medium |

**Total: ~5 new theorems, building on existing foundations.**

The result: A complete formal proof that **given** power law spectrum with exponent α, the optimal rank ratio is ε^{1/(2α-1)} where ε is determined by the capacity-convergence tradeoff.

The empirical question becomes: What is α for language models?
