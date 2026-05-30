# Scientific Rigor Review: "Emending Nonlinear Recurrence"

**Reviewer role:** Evaluator (calibrated rubric-driven analysis)  
**Date:** 2026-05-30  
**Source:** `paper/main.typ` (read-only analysis)  
**Task:** review-scientific-rigor  

---

## Summary

The paper makes three principal claims: (1) a pure-nonlinear-recurrent language model can reach sub-1-bpb at 1.3B scale on a single workstation GPU (*viability*); (2) the delta-correcting update is formally and empirically more expressive than the raw-write update at matched FLOP (*power separation*); and (3) E88 and GDN occupy the same loss-vs-wallclock band at 1.3B (*supporting comparison*). The paper is unusually transparent about its limitations and explicitly demarcates what the Lean core does and does not prove. Most overclaims are hedged or acknowledged. The main rigor risks are: (a) a single-seed racer at 1.3B with unequal training tokens across architectures; (b) the "matched no-tuning" condition for expressivity probes concealing a selection-history asymmetry that favors the Emender; (c) the QA panel being far too small for the claims resting on it; and (d) the ablation being elimination-by-architecture rather than controlled variable isolation.

---

## Section-by-Section Claim Analysis

### §1 Introduction & Abstract

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| "E88 reaches 0.977 bpb on The Pile after about 21 stitched wall-clock days on a single workstation-class GPU" | Abstract, §1 opening | **SUPPORTED** | Confirmed by AS_OF.md snapshot (100K-step trailing BPB 0.976819). Single-seed caveat is acknowledged. |
| "22,200 independent programs per token (370 × batch 5 × depth 12)" | Abstract, §1, §3 | **SUPPORTED** | Arithmetic checks out; Lean-verified. |
| "The recurrent-language-modelling literature treated this regime as out of reach" | §1 | **SUPPORTED (with qualification)** | Substantiated by M²RNN choosing hybrid form, ParaRNN requiring Newton iteration. However, the claim that the field's operating verdict was a hard impossibility is slightly overstated — it was more of a practical gap than a proven ceiling. |
| "Lean 4 trusted core machine-verifies that delta-correcting update > raw-write at matched FLOP" | Abstract, §1 | **SUPPORTED** | Lean proofs exist; non-claims explicitly listed in §7. |
| Emender "reaches 0.79 on the S₅ word problem against 0.22 for the raw-write baseline" | Abstract | **SUPPORTED** | Verified against ndmpapernotes.md data (mean over 3 seeds; min/max: Emender 0.62–0.89, M²RNN 0.19–0.23). |

**Flag:** The abstract says "about 21 stitched wall-clock days." The AS_OF.md shows: E88 512.9 stitched GPU-hours (≈21.4 days), GDN 517.8 h (≈21.6 days), M²RNN-CMA 479.0 h (≈20.0 days). "About 21 days" for E88 is accurate. The word "stitched" is used in the abstract but not defined there; it means the run was repaired after a NaN divergence at step ~247,250 and resumed from step 231,000. This restart is material to the run trajectory and should be disclosed more prominently.

---

### §2 Background

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| Classification of linear-state vs nonlinear-state by criterion $h_t = A_t h_{t-1} + b_t$ | §2 | **SUPPORTED** | Standard and correctly applied. GDN, Mamba, RWKV etc. are all linear-state by this criterion. |
| "Asymptotically a linear-state recurrence at fixed precision and width is a regular-language recogniser inside TC⁰" | §2 | **SUPPORTED (cited)** | Merrill–Petty–Sabharwal 2024 cited correctly. |
| Barrington's theorem: S₅ is NC¹-complete | §2 | **SUPPORTED (cited)** | Correctly cited. Not formalized, as acknowledged. |

No rigor issues in §2.

---

### §3 Architecture

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| "A three-row ablation across the three candidate properties isolates the write rule by elimination" (Table 1) | §3 | **OVERCLAIMED** | See Flag below. |
| "M²RNN-CMA scores 0.31 on S₃, the solvable control... rules out a complexity-ceiling explanation" | §3 | **SUPPORTED** | Correct inference: if raw-write can't do solvable groups, complexity ceiling is not the explanation. |
| "The deficit is therefore trainability under SGD, not representability or capacity" | §3 | **SUPPORTED (with qualification)** | Capacity floor argument is valid (131,072 recurrent scalars vs 2.6-bit S₃ floor). The conclusion is reasonable but conflates parameter capacity with optimization landscape navigability. |

**Flag — Ablation by elimination (Table 1):** The claim that delta correction is isolated as the differentiator rests on a three-architecture comparison where each architecture differs from the others in multiple architectural properties, not only the claimed one. Specifically:

- GDN vs M²RNN differ in: update rule (linear delta vs raw-write nonlinear), temporal nonlinearity on state, AND model shape, head geometry, etc.
- M²RNN vs Emender differ in: write rule (raw-write vs delta-correcting), AND model shape (dim=1920/depth=21 vs dim=1664/depth=12 at 1.3B; at 8M presumably similar geometry).
- There is no architecture with delta correction but *without* temporal nonlinearity on state, which would complete the 2×2 table.

The elimination argument is valid *only if* no other architectural differences co-vary with the claimed properties. The paper acknowledges the selection-history asymmetry (§9) but does not flag the incomplete factorial design in the ablation table caption itself.

**Issue:** The 8M probe shapes — the Emender uses dim=384/depth=4/H=32/N=32, and M²RNN-CMA uses the "analogous default from its CMA-tuned reshape." The shapes are described as "parameter-matched" but the specific M²RNN 8M shape is not given in the paper. If M²RNN's 8M shape uses dim=dim₂/depth=depth₂, the comparison is not purely isolating the write rule.

---

### §4 Multi-Programming and Systems

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| Fused Triton kernel saves "approximately 50–60 ms per step" at depth 12 | §4 | **UNSUPPORTED** | No benchmark data cited or referenced. This specific claim has no supporting table or figure. |
| Sparse-checkpoint backward gives "approximately 16× shrink" in activation memory at K=16 | §4 | **SUPPORTED (arithmetic)** | T/K+1 vs T checkpoint slots: arithmetic is correct. No empirical profile cited to verify wall-clock cost. |
| "~70% of hand-tuned throughput in ~one week of porting work, versus three to six weeks to port to HIP from scratch" | §4 | **UNSUPPORTED** | No benchmark data or source cited for the throughput comparison or time estimate. These are informal characterizations with no empirical anchor. |
| Newton iteration on the tanh map was "significantly worse in throughput than the multi-programmed Triton kernels" | §4 | **UNSUPPORTED** | Mentioned as a tried alternative but no profile or comparison table is provided. |

**Flag:** §4 contains several informal performance characterizations that lack supporting data. This section is systems-descriptive rather than empirically evidenced.

---

### §5 Language-Modelling Results

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| "After about 20–22 stitched GPU-days, E88 reaches 0.977 bpb, GDN 0.970, M²RNN-CMA 0.983; all three are sub-1-bpb" | §5 | **SUPPORTED** | Confirmed by AS_OF.md. All three are indeed sub-1-bpb. |
| "E88 and Gated DeltaNet occupy the same sub-1-bpb loss-vs-wallclock band" | §5 caption | **TECHNICALLY ACCURATE BUT POTENTIALLY MISLEADING** | See Flag 1 below. |
| Per-architecture CMA-ES "at matched candidate budget" | §5 | **SUPPORTED (with qualification)** | See Flag 2 below. |
| "The plotted trajectory is one realization per architecture; the within-class ordering it illustrates is replicated across four CMA-ES sweeps (250+ configs/architecture)" | §5 caption | **SUPPORTED** | The CMA-ES replication is a meaningful robustness check, but "250+ configs" is mentioned only in caption and §9, not in the methods proper. |

**Flag 1 — "Same band" framing vs GDN lead:** At the 2026-05-29 snapshot, GDN has 0.007 bpb lower loss than E88 (0.970 vs 0.977). The paper correctly does not claim E88 beats GDN; it frames them as the "same sub-1-bpb band." However:

- **Unequal training tokens:** GDN has seen 15.131B tokens vs E88's 14.392B tokens (5.1% more) and M²RNN-CMA's 13.753B tokens (10.0% more). The paper reports "stitched GPU-days" (∼21 days each) but the models have different batch sizes (GDN: batch 4, effective 8,192 tokens/step; E88/M²RNN-CMA: batch 5, 10,240 tokens/step) and different step counts (GDN: 1,847,050 steps vs E88: 1,405,450 steps). GDN has trained for ~31% more steps than E88. This is material to the band-equivalence claim.
- **Unequal parameter counts:** E88 (1.273B), M²RNN-CMA (1.307B), GDN (1.352B). GDN has ∼6.2% more parameters than E88.
- **Training is ongoing:** The snapshot note says "training continues" — the final comparison may shift.
- **Single seed per architecture at this scale:** No variance estimate exists. The paper acknowledges this explicitly in §9 ("What remains single-seed at this scale is the multi-week trajectory itself").
- **E88 had a training divergence:** E88's run diverged at step ~247,250 (NaN gradients), was repaired from step 231,000, and resumed at step 247,500. This is disclosed in AS_OF.md but not in the main text.

The "same band" framing is defensible given all models are sub-1-bpb, but the 5–10% token-count advantage for GDN is not disclosed in the main text. A token-matched comparison would be more rigorous.

**Flag 2 — CMA-ES protocol disclosure:** The methods section states "population 16, fixed wall-clock per candidate, identical fitness rule." This discloses the key parameters. However:
- "250+ configs/architecture" appears only in the §5 caption and §9, not as a clear methods disclosure.
- "Range repositioning when limits were hit, applied identically across the three architectures" — the specific repositioned ranges and number of repositionings are not reported.
- The exact wall-clock budget per candidate is not stated.

**Flag 3 — M²RNN divergence shaping the narrative:** The M²RNN-paper shape (dim=3072, depth=10, H=759, N=16) diverged at step 8,400. The paper correctly excludes this from the racer and uses M²RNN-CMA (a CMA-reshaped variant) instead. This is transparent. However, it is notable that M²RNN-CMA is an architecture significantly modified from the published M²RNN to be stable under this setup — its paper-reported shape is unstable. The comparison is therefore Emender (designed for this setting) vs M²RNN-CMA (M²RNN, modified to survive this setting). This positions M²RNN-CMA as a weakened baseline relative to the original M²RNN paper.

---

### §6 Expressivity Results

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| Capacity is non-binding at 8M parameters for S₃/S₅ probes | §6 | **SUPPORTED** | The order-of-magnitude capacity argument is valid: 131,072 recurrent state scalars vs 6.9-bit S₅ floor = ~5 orders of magnitude. The conclusion is sound. |
| "Matched no-tuning across architectures at 8M" | §6 | **MISLEADING — UNSUPPORTED AS STATED** | See Flag 4 below. |
| Emender 0.79 on S₅ at T=128; GDN 0.36; M²RNN-CMA 0.22 | §6, Table 2 | **SUPPORTED** | Confirmed by ndmpapernotes.md lines 153–173 (E88: 0.7918 mean, min 0.6232, max 0.8880). |
| "The Emender separates from all three baselines *at training length*, not only under length extrapolation" | §6 | **SUPPORTED** | The separation is large (0.79 vs 0.36 vs 0.22); Emender min (0.62) exceeds M²RNN max (0.23). |
| "At 8M... Emender ties or wins GDN on five of six tasks" | §6 | **SUPPORTED** | Confirmed by CANONICAL_SWEEP_RESULTS.md. |
| GDN edges Emender on associative recall (0.997 vs 0.881) | §6 | **SUPPORTED** | Confirmed; characterized correctly as "attention-natural task." |
| "Emender retains 0.89 accuracy on parity at T=500 where GDN collapses to 0.55" | §6 | **SUPPORTED** | Confirmed by LENGTH_EXTRAP_RESULTS.md (E88: 0.887±0.088, FLA: 0.550±0.002). |
| Hybrid degradation: AABB underperforms either pure family | §6 | **SUPPORTED WITH SCOPE CAVEAT** | Confirmed by CANONICAL_SWEEP_RESULTS.md and LENGTH_EXTRAP_RESULTS.md. One hybrid pattern tested (AABB). |

**Flag 4 — Selection-history asymmetry in "matched no-tuning":** The paper states the 8M probes use "matched no-tuning across architectures." However, §9 acknowledges: "the Emender's defaults are the endpoint of an ablation lineage selected partly on state-tracking behaviour, whereas GDN and M²RNN's published defaults were selected by their authors on language-modelling loss."

This is not merely a caveat — it is a fundamental confound for the expressivity comparison. "Matched no-tuning" means no new probe-specific HPO was done; it does *not* mean the starting configurations were selected by the same criterion. The Emender's defaults were co-evolved with state-tracking objectives; the baselines' defaults were not. Any advantage the Emender shows at 8M could partially reflect this prior selection, not only the write rule.

The paper correctly identifies this as a limitation (§9) and proposes a matched-search experiment as the closing experiment. The concern is that the headline expressivity claim — "delta correction is the differentiating ingredient" — is derived from a comparison with a known asymmetry in favor of the Emender's defaults.

**Mitigation in the paper:** The S₃ control partially addresses this. M²RNN-CMA scoring 0.31 on S₃ (a solvable group, below the theoretical complexity ceiling of linear recurrence) indicates that the raw-write update fails at a task where non-solvability is not the explanation. This is a meaningful control that reduces (but does not eliminate) the selection-history concern.

**Flag 5 — No formal significance tests:** The expressivity comparisons use 3 seeds and report means and SEMs (via figure). No p-values or confidence intervals are reported for the architecture comparisons. For the headline S₅ result (0.79 vs 0.36 vs 0.22), the gap is large enough that 3-seed inference is plausible, but the Emender's within-seed variance is substantial (0.62–0.89 range), and no formal test confirms separation. For the six-task sweep with 3 seeds (dim=384, 10K steps), the claim "Emender ties or wins on 5/6 tasks" is descriptive, not tested.

**Flag 6 — Canonical sweep step count:** CANONICAL_SWEEP_RESULTS.md notes "10K steps is enough for grokking on these tasks at dim=384, but is short; longer runs would tighten the ± bands." The length-extrapolation results use 5K steps. The paper does not disclose that 10K steps was used for the canonical sweep or that this choice was made without probe-specific tuning.

**Flag 7 — QA panel sample sizes:** The QA evaluation (§6 QA and reasoning panel) uses:
- Knowledge panel: 300 items total (50 per task × 6 tasks), SE ≈ 6 pp per task.
- Reasoning panel: Evaluated at step ~957K for E88, ~1,272K for GDN, ~879K for M²RNN — different training stages.

With SE ≈ 6 pp at 50 items/task, differences of less than ~12 pp are below the 2-SE threshold. The paper correctly states "all three sit within one standard error of one another" and that "none has crossed the threshold where reasoning benchmarks differentiate." However:
- The reasoning panel (Table 2 in section_draft.md) shows GDN overall 0.350 vs E88 0.319 — a 0.031 difference that may not be within 1 SE of the *overall* score.
- The paper says "E88's overall reasoning accuracy (0.319) is within one standard error of M²RNN-CMA (0.336)" but does not explicitly compare E88 to GDN (0.350) in this framing.
- 50 items/task is simply too few to draw reliable conclusions. The paper's hedging ("none has crossed the threshold") is appropriate, but the QA panel should not be presented as evidence for or against the main claims.

---

### §7 Formal Results

| Claim | Location | Verdict | Notes |
|---|---|---|---|
| Lean 4 trusted core contains no sorry/admit/axiom/opaque/native_decide | §7, abstract | **SUPPORTED** | Stated clearly; verifiable by inspection. |
| k-step separation (`emender_m2rnn_k_step_separation`) holds for all k≥1 | §7 | **SUPPORTED** | Proof exists in trusted core. |
| S₅ tracker realised by orthonormal-key Emender (`emender_realizes_s5_tracker`) | §7 | **SUPPORTED** | Proof exists; correctly scoped to realisability, not empirical weight recovery. |
| "Barrington's theorem (cited; not formalized in this work)" | §7 NC¹ paragraph | **CORRECTLY DISCLOSED** | Non-claim is explicit. |
| "Parameter-efficiency corollary (informal; follows from sets C and D)" | §7 | **CORRECTLY LABELED** | Explicitly labeled informal. |
| "Not proved: that gap compounds on the S₅ generator alphabet specifically with explicit length bound T(d)" | §7 | **CORRECTLY DISCLOSED** | The frontier is clearly named. |

§7 is the most rigorously scoped section in the paper. The explicit non-claims list and frontier description are exemplary transparency.

**Minor flag:** The "NC¹ paragraph" states: "The Emender therefore reaches the top of NC¹ in the canonical regular-language witness." This is technically correct (the Emender *can realise* the S₅ tracker) but reads as a strong capability claim. The full logic chain is: fixed-precision Emender is finite-state (provable) → within that ceiling it can realise the S₅ tracker (proved) → S₅ is NC¹-complete (Barrington, cited) → therefore the Emender reaches the top of NC¹ in the canonical witness. This chain is valid but depends on Barrington's theorem being cited-not-proved, which the paper discloses.

---

### §8 Related Work

No rigor issues. The paper correctly situates M²RNN as concurrent prior art for nonlinear matrix-state recurrence (§8, M²RNN paragraph). The xLSTM, Titans, and classical LSTM/GRU comparisons are appropriately scoped.

---

### §9 Limitations

The limitations section is unusually thorough. It explicitly acknowledges:
- Single-seed racer at 1.3B
- Selection-history asymmetry in 8M defaults
- The k-step separation running on the constructed 2D alphabet, not S₅ generators
- Ongoing training (results are a snapshot)
- No transformer comparison

This section is well-calibrated.

---

## Summary of Findings by Category

### Overclaims

| ID | Claim | Location | Severity |
|---|---|---|---|
| OC-1 | "A three-row ablation isolates the write rule by elimination" — presented as decisive, but ablation is elimination-by-architecture, not controlled variable isolation | §3 Table 1 caption | Medium |
| OC-2 | "Matched no-tuning across architectures at 8M" — Emender's defaults were co-evolved with state-tracking; baseline defaults were not | §6 | High |
| OC-3 | Fused kernel saves "approximately 50–60 ms per step"; Triton achieves "~70% of hand-tuned throughput" — no data cited | §4 | Low-Medium |

### Unsupported Assertions

| ID | Claim | Location | Notes |
|---|---|---|---|
| UA-1 | Triton kernel savings: "50–60 ms per step" | §4 | No benchmark table or figure |
| UA-2 | "~70% of hand-tuned throughput in ~one week of porting work" | §4 | No source or measurement cited |
| UA-3 | Newton iteration on tanh map "significantly worse in throughput" | §4 | No profile shown |

### Missing Controls / Baselines

| ID | Issue | Severity |
|---|---|---|
| MC-1 | No architecture combining delta correction + raw-write state (would complete the ablation table's 2×2 design) | Medium |
| MC-2 | No token-matched comparison between E88, GDN, M²RNN-CMA in the 1.3B racer (GDN has 5–10% more training tokens) | Medium |
| MC-3 | No transformer baseline at 1.3B (paper acknowledges this explicitly in §9) | Low (acknowledged) |
| MC-4 | Hybrid degradation tests only one pattern (AABB); other patterns mentioned only informally | Low |
| MC-5 | QA/reasoning panel evaluated at different training steps per architecture | Medium |

### Statistical Issues

| ID | Issue | Severity |
|---|---|---|
| SI-1 | No formal significance tests for S₅/S₃/six-task accuracy comparisons | Medium |
| SI-2 | 1.3B racer: single seed per architecture, no variance estimate | High (acknowledged) |
| SI-3 | QA panel: 50 items/task, SE ≈ 6 pp — insufficient for capability claims | High |
| SI-4 | 1.3B racer: unequal token counts (E88 14.4B, GDN 15.1B, M²RNN-CMA 13.8B) — not disclosed in main text | Medium |
| SI-5 | Emender's 8M seed variance is substantial (S₅: 0.62–0.89 range over 3 seeds) — correct to show SEM in figure, but noted |  Low |

### Results Matching Figures

| Claim | Figure/Table | Match |
|---|---|---|
| E88 0.977 BPB, GDN 0.970 BPB, M²RNN-CMA 0.983 BPB | Fig. 2 caption + AS_OF.md | **CONFIRMED** |
| Emender S₅ 0.79 (T=128); GDN 0.36; M²RNN-CMA 0.22 | Table 2 (tab_s5) + ndmpapernotes.md | **CONFIRMED** |
| Emender S₃ 1.00; GDN 0.72; M²RNN-CMA 0.31 | Table 2 + ndmpapernotes.md | **CONFIRMED** |
| Parity at T=500: E88 0.89, GDN 0.55 | §6 + LENGTH_EXTRAP_RESULTS.md | **CONFIRMED** (0.887 vs 0.550) |
| FSM tracking at T=500: E88 0.59, GDN 0.39 | §6 + LENGTH_EXTRAP_RESULTS.md | **CONFIRMED** (0.591 vs 0.387) |
| Canonical sweep: E88 wins/ties 5/6 tasks | §6 + CANONICAL_SWEEP_RESULTS.md | **CONFIRMED** |
| GDN edges on associative recall (0.997 vs 0.881) | §6 + CANONICAL_SWEEP_RESULTS.md | **CONFIRMED** |
| M²RNN-paper diverged at step 8,400 | §5, Appendix + AS_OF.md | **CONFIRMED** |
| Emender S₅ degradation: 0.79→0.42→0.22→0.11 at T=128/256/512/1024 | §6, §9 + ndmpapernotes.md | **CONFIRMED** |

---

## Issues Ranked by Priority for Revision

1. **[HIGH] Token-count asymmetry in 1.3B racer (§5):** GDN's ~5–10% token advantage and ~31% step advantage over E88 should be disclosed in the main text alongside the wallclock comparison. A token-normalized comparison (BPB vs tokens seen) would strengthen the "same band" claim.

2. **[HIGH] Selection-history asymmetry (§6):** The "matched no-tuning" condition needs a clearer label in §6 itself (not only §9) that the Emender's 8M defaults were partly co-selected on state-tracking. The S₃ control partially addresses this but is not sufficient to fully absorb it.

3. **[HIGH] QA panel sample sizes (§6):** Either expand to ≥500 items per task or soften the QA comparison to "preliminary evidence" and move it to an appendix. 50 items/task at SE ~6pp cannot support architecture capability comparisons.

4. **[MEDIUM] Ablation by elimination (§3 Table 1):** The caption should clarify this is an elimination argument across architectures, not a controlled single-variable manipulation. The missing cell (delta correction without temporal nonlinearity) should be noted.

5. **[MEDIUM] Unsupported systems claims (§4):** The 50–60 ms saving, the 70% throughput claim, and the Newton-iteration comparison should either be backed by benchmark tables or softened to "estimated" / "anecdotal."

6. **[MEDIUM] E88 training divergence (§5):** The NaN divergence at step ~247,250 and the repair/resume from step 231,000 should be disclosed in the main text. It is in AS_OF.md (a supplementary data file) but not in the paper itself.

7. **[MEDIUM] Missing 8M shape details (§6):** The exact M²RNN-CMA shape at 8M scale (dim, depth, H, N) is not given. Since the claim is "parameter-matched," both shapes should be tabulated.

8. **[LOW] CMA-ES protocol quantification:** "250+ configs/architecture" appears in the caption and §9 but not in the §5 methods block. Adding it to methods strengthens reproducibility.

9. **[LOW] Hybrid degradation scope:** The AABB pattern finding should note that other patterns give "similar conclusions" (per CANONICAL_SWEEP_RESULTS.md caveats).

---

## Assessment of Main Claims

### Claim 1: Viability (E88 reaches sub-1-bpb at 1.3B)
**Verdict: SUPPORTED.** E88's 0.977 BPB is confirmed by data, and the result is explicitly an in-progress snapshot. The "~21 stitched wall-clock days" framing is accurate. The training divergence and repair should be disclosed.

### Claim 2: Power Separation (delta-correcting > raw-write formally and empirically)
**Verdict: FORMALLY SUPPORTED, EMPIRICALLY SUPPORTED WITH CAVEATS.** The Lean proofs are robust and properly scoped. The empirical S₅/S₃ gaps are large and confirmed. The primary caveat is the selection-history asymmetry, which the paper acknowledges but does not fully absorb in the main claim framing.

### Claim 3: Supporting Comparison (E88 in same band as GDN)
**Verdict: SUPPORTED WITH IMPORTANT QUALIFICATIONS.** The "same sub-1-bpb band" framing is defensible, but the unequal training tokens (GDN +5–10%) and unequal parameter counts (GDN +6.2%) are not disclosed in the main text. A token-matched comparison would either strengthen or clarify the claim.

---

## Validation Checklist

- [x] Every section's central claims enumerated with supported/unsupported/overclaimed verdict
- [x] Each flagged issue cites the specific claim location and the missing/weak evidence
- [x] Findings written to a markdown artifact (paper/review/rigor.md)
