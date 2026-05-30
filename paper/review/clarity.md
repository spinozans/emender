# Clarity, Structure & Narrative Review — paper/main.typ

**Reviewer:** Default Evaluator (agent-635)  
**Task:** review-clarity-structure  
**Date:** 2026-05-30  
**Status:** READ-ONLY review; no edits to main.typ

---

## Executive Summary

The paper presents three results clearly and with appropriate detail. The core story — that width-axis multi-programming makes pure nonlinear recurrence trainable at 1.3 B scale, and that delta-correction is strictly more expressive than raw-write at matched FLOP cost — is legible and compelling. The major structural weaknesses are: (1) heavy redundancy of the core claims across abstract / intro / contributions / section bodies / conclusion, (2) unnumbered subsection headings throughout creating inconsistent section hierarchy, (3) one extremely long figure caption that embeds methodology, (4) the "Frontier and unproven targets" subsection breaking §7 flow by appearing mid-section, and (5) undefined abbreviations (CMA-ES, bpb, PNR) used before formal introduction. No section is incomprehensible, but §7 and §9 are the most demanding reads and would benefit from restructuring.

---

## Section-by-Section Clarity Assessment

### Abstract (lines 71–106)

**Clarity rating: B (good but dense)**

- **Strengths:** Result-first structure is correct and effective. The three contributions (viability, formal separation, comparison) are all present.
- **Issue 1 — density and order:** The abstract gives the Lean 4 machine-verification results before the reader has any context for what "delta-correcting update" means or why "raw-write" is the alternative. The jump to `emender_m2rnn_one_step_resource_separation_embeds` (an identifier) in the abstract is premature.
- **Issue 2 — flow break:** The long middle sentence beginning "A Lean 4 trusted core machine-verifies..." is a run-on that breaks narrative flow. It tries to pack three theorem names into one sentence.
- **Issue 3 — "bits per byte" undefined:** "bits per byte" is used without definition; readers unfamiliar with this metric will not have a baseline for whether 0.977 is good.
- **Issue 4 — three draft alternatives left in source:** Lines 13–59 contain three commented-out alternate abstracts. These are not rendered but should be removed before submission to avoid confusion.
- **Recommendation:** The active abstract is the correct framing; trim the Lean theorem names to one representative per result and move them to §7.

---

### §1 Introduction (lines 157–321)

**Clarity rating: B+ (strong framing, minor structural issues)**

- **Strengths:** The opening paragraph is excellent — E88's performance is stated first, then the prevailing assumption it challenges, then the mechanism that makes it work. This is the right structure.
- **Strengths:** The "Delta correction is one response; hybrids are another" unnumbered subsection correctly repositions the paper's scope relative to hybrid alternatives.
- **Issue 1 — unnumbered heading inconsistency (lines 208, 250):** `#heading(level: 2, numbering: none)` is used for "Delta correction is one response..." and "Contributions". These look like subsections typographically but have no numbers, making cross-reference impossible and breaking the hierarchy. This pattern recurs throughout the paper (§3, §4, §5, §6, §7, §8, §9, §12).
- **Issue 2 — numbering mode switch mid-contributions (lines 254–288):** The contributions are numbered `1.`/`2.`/`3.`, then the paragraph introducing the three trained models (`(a)` E88, `(b)` M²RNN-CMA, `(c)` GDN) also uses `(a)`-style enumeration — but without labeling items (a), (b), (c) explicitly in the text. The `#set enum(numbering: "(a)")` on line 288 is set *after* the contributions list but before the cohort description. This creates ambiguity: does "(a)" list the contributions or the three models?
- **Issue 3 — "pure nonlinear recurrent (PNR)" first use:** "PNR" is introduced on line 226 but "pure nonlinear recurrent" is used in the first paragraph (line 163) without the abbreviation. The abbreviation is later used without reinstatement in §3 and §4.
- **Issue 4 — "CMA-ES" abbreviation:** Used in Contribution 3 (line 283) without expansion. First expanded as "per-architecture CMA-ES @cmaes2003 hyperparameter search" in §3, but the citation arrives in §5. Expand on first use.
- **Issue 5 — URL block (lines 312–320):** The block of GitHub/HuggingFace/hypervolu.me URLs in the intro is verbose for a section intro. These are appropriate in §10 (Conclusion) but disrupt the §1 narrative.
- **Issue 6 — "bpb" undefined:** "0.977 bits per byte" is used in the abstract and §1 without definition. The formal definition (nats/token × log₂(e) / bytes/token) appears only in §5 (line 916).

---

### §2 Background (lines 323–437)

**Clarity rating: B (good core exposition, unusual opener)**

- **Issue 1 — "work began against a workload" opener (lines 325–335):** This pangenomic motivation paragraph is the most unusual piece of framing in the paper. It provides genuine motivation but the connection between pangenomics and billion-parameter language model expressivity is not developed. After §2, pangenomics is never mentioned again. Either develop it (connecting to why S5 matters for biological sequence modelling) or remove it.
- **Strengths:** The linear-state vs nonlinear-state classification (lines 354–374) is precise and well-stated. The criterion is explicit: "$h_t = A_t h_{t-1} + b_t$ with $A_t$ and $b_t$ depending on $x_t$ only."
- **Issue 2 — TC⁰ complexity digression (lines 370–382):** The paragraph ending "...lie in DLOGTIME-uniform TC⁰ @chiang2025uniform_tc0" is technically correct but assumes the reader knows what TC⁰ means relative to NC¹ means relative to regular languages. This is the most audience-unfriendly passage in §2. A one-sentence gloss ("TC⁰ and NC¹ are sub-polynomial circuit complexity classes...") would help.
- **Issue 3 — matrix state subsection (lines 413–425):** The claim that matrix state is "treated here as a precondition shared by the Emender and its baselines, not as a contribution" is important and well-placed, but the subsection is very short. The listing of architectures that use matrix state (mLSTM, RWKV, etc.) overlaps with similar lists in §8 (Related Work).

---

### §3 Architecture (lines 439–703)

**Clarity rating: A- (strong, minor flow issues)**

- **Strengths:** The per-head update block (lines 461–467) is clean, well-labeled, and correctly introduced.
- **Strengths:** The ablation table (@tab_ablation, lines 657–681) is the single clearest exposition in the paper. Three rows, three properties, decisive reasoning. This table should be referenced earlier (e.g., in §1) to give readers a preview of the argument structure.
- **Issue 1 — parameterisation details break architectural narrative (lines 614–624):** The "Parameterisation choices" subsection includes float32 precision, cast-back to storage dtype, numerically stable tanh form, and the rationale for removing output RMSNorm. These are implementation details that belong in an appendix or §4 (Systems), not in the architecture description.
- **Issue 2 — "Ablation by architecture" subsection placement (lines 634–703):** This subsection functions as analytical results (it argues that delta-correction is the key differentiator) but sits in §3 (Architecture) where a reader expects description, not argument. Consider moving to §6 or making it a named numbered section.
- **Issue 3 — Emender footnote (line 443):** "The name arrived as a correction to a prior internal handle; the architecture emended its own name by the same process it performs on memory." This is charming but the footnote placement in the middle of the architectural definition may distract.
- **Issue 4 — "1.3B production stack" subsection (lines 626–632):** This very short subsection reiterates the E88 shape and refers to §4; it could be merged with the parameterisation choices subsection.

---

### §4 Multi-Programming and Systems (lines 704–792)

**Clarity rating: A- (clear and well-organized)**

- **Strengths:** The core idea (width-axis parallelism instead of time-axis parallelism) is explained correctly and repeatedly reinforced.
- **Strengths:** The "Fused Triton recurrence kernel" and "Sparse-checkpoint backward" subsections are technically specific in a helpful way.
- **Issue 1 — "Distributed training" subsection contains Newton iteration comparison (lines 785–792):** The mention of ParaRNN @pararnn2025 and the finding that Newton iteration was "significantly worse in throughput" is a significant claim that appears without a table or number to back it. Either add data or move to §9 (Limitations).
- **Issue 2 — "The 1.3B Emender shape under multi-programming" subsection (lines 731–741):** This repeats facts already stated in §3 ("22,200 independent programs per token," "32×32 state tile that fits in registers"). Some repetition is appropriate, but this subsection could be reduced to a single bridge sentence.

---

### §5 Language-Modelling Results (lines 793–923)

**Clarity rating: B+ (mostly clear, one major figure caption problem)**

- **Strengths:** The CMA-ES protocol is well-explained and the "fairness anchor" framing is compelling and appropriate.
- **Strengths:** The CMA-ES as instrument narrative ("The search, configured for fairness across architectures, independently voted for the architecture's central thesis") is elegant and should be preserved.
- **Issue 1 — Figure 2 caption is far too long (lines 873–904):** The @fig_lm_racers caption exceeds 200 words. It includes: methodology explanation (bytes-per-token conversion), hedging about "training continues," endpoint values with step numbers, seed counts, notes about color conventions, and a timestamp. Most of this belongs in the main text or methods appendix. A figure caption should support the figure, not substitute for a methods section.
- **Issue 2 — "Gradient conditioning is a third recipe property" subsection heading vs. "A fourth run" opening sentence (lines 849–865):** The heading says "third recipe property" but the first sentence says "A fourth run." These are counting different things (recipe ingredients vs. training runs), but side-by-side the discrepancy is jarring.
- **Issue 3 — "bpb" conversion formula in main text (line 916):** The conversion formula appears in §5 body text; it was needed already in §1 and the abstract. This is appropriate placement for the formula, but a forward reference or brief parenthetical earlier would help.

---

### §6 Expressivity Results (lines 924–1109)

**Clarity rating: B+ (results are clear; some structural concerns)**

- **Strengths:** The "capacity is non-binding" argument is explained rigorously and the floor calculations are provided.
- **Strengths:** The S3 solvable-group control is well-motivated and the conclusion drawn from M²RNN-CMA's failure on S3 is correctly stated.
- **Issue 1 — "Hybrid degradation: purity matters" subsection (lines 1058–1088):** This finding (AABB hybrid underperforms both pure families) is interesting but its placement in §6 (Expressivity Results) is awkward. It is neither the S5/S3 probe result nor the six-task sweep — it's a separate experiment. Consider moving to §9 (Limitations) or a separate section.
- **Issue 2 — QA and reasoning panel subsection (lines 1090–1109):** This subsection is important for completeness but the results are null (all three models within standard error). The framing is appropriate ("None has crossed the threshold where reasoning benchmarks differentiate"). However, it reads as a placeholder result — readers may wonder why it is in §6 rather than §9.
- **Issue 3 — six-task sweep subsection transition (lines 1032–1057):** The transition from the S5/S3 results to the six-task sweep lacks an explicit bridge sentence explaining why this additional sweep is needed ("To verify that the S5 result is not a single-task artefact..." is on line 1033, which is good, but the length-extrapolation results within this subsection are not visually separated from the headline results).

---

### §7 Formal Results (lines 1111–1398)

**Clarity rating: B (technically correct; mid-section placement of limitations breaks flow)**

- **Strengths:** Organizing by theorem sets A through F provides clear structure. Each theorem name is given and can be verified.
- **Strengths:** The "Explicit non-claims" at the end (lines 1388–1397) are clear and appropriately calibrated.
- **Issue 1 — "Frontier and unproven targets" appears between Theorem set C′ and Theorem set D (lines 1232–1276):** This is the most significant structural problem in §7. The subsection lists what the paper has not proved and belongs either at the *end* of §7 (after all theorem sets are stated) or in §9 (Limitations). Breaking the theorem-set sequence to insert a limitations block undermines the cumulative effect of the formal results.
- **Issue 2 — "NC¹ paragraph (verbatim)" (lines 1372–1386):** The framing "We adopt the audit-recommended wording from the formalisation gap analysis" signals that this text was written by someone else or was externally reviewed. This is appropriate for transparency but may raise questions for readers about who the "audit" was. If this is an internal review, rephrase; if external, identify the source.
- **Issue 3 — Theorem set E (lines 1306–1319):** The multi-programming predicate theorem (`multiProgrammed_admits_m2rnn_and_emender`) is described here, but multi-programming was the subject of §4. The logical flow would be cleaner if this theorem appeared in §4 or was cross-referenced there.
- **Issue 4 — Theorem set F (Latching) subsection (lines 1321–1369):** The latching theorems are described with good precision, but the notation is introduced mid-section without a table or figure. A small schematic of the slot-wise state before/after saturation would help visualization.

---

### §8 Related Work (lines 1399–1508)

**Clarity rating: A- (well-organized with clear ancestry narrative)**

- **Strengths:** The ancestry section (Widrow-Hoff → fast-weight programmers → DeltaNet → Emender) is the clearest narrative in the paper and accurately positions the contribution.
- **Issue 1 — M²RNN paragraph is partially redundant with §1 and §2 (lines 1464–1485):** The facts about M²RNN (concurrent prior art, hybrid form at 7B MoE, Mishra et al. §5.2 uniform protocol) are stated at least partially in §1, §2, §3, and §5. The §8 treatment adds the fair-comparison context, which is appropriate, but some compression is possible.
- **Issue 2 — xLSTM classification note (lines 1487–1494):** The calculation "87.5% of its blocks are linear-state" is correct but the arithmetic (7:1 ratio → 87.5%) is left implicit. Spell it out: "7 mLSTM (linear-state) blocks for every 1 sLSTM (nonlinear-state) block."

---

### §9 Limitations (lines 1510–1659)

**Clarity rating: B (admirably honest; too long; some items belong elsewhere)**

- **Strengths:** The limitations section is thorough and calibrated. The "Evidence structure" subsection accurately summarizes what rests on what and is well-placed.
- **Issue 1 — "Formal scope" subsection largely repeats §7 (lines 1513–1536):** The trusted Lean core coverage and explicit non-claims are already stated in §7. The §9 version adds little beyond a condensed restatement. Either refer to §7 or remove.
- **Issue 2 — "The opposite architectural bet: hybrids" (lines 1617–1633):** This subsection reads as a related-work discussion (comparing the Emender's pure-recurrent bet against OLMo-Hybrid's hybrid bet). It is not a limitation of this paper — it is a framing of two different research bets. Move to §8.
- **Issue 3 — Limitations section is disproportionately long:** At approximately 150 lines, §9 is nearly the length of §5 (Language-Modelling Results) and §6 (Expressivity Results) combined. Some consolidation is warranted; items already stated in §7, §8, or §12 should be cut or cross-referenced.

---

### §10 Conclusion (lines 1660–1709)

**Clarity rating: A (clean and appropriately brief)**

- **Strengths:** "Two results." opener is bold and appropriate. The conclusion correctly summarizes both results without overclaiming.
- **Minor issue — URLs in conclusion (lines 1696–1708):** Three raw URLs appear in the conclusion. These are appropriate for an arXiv preprint; no change needed, but they visually interrupt the conclusion flow. Consider a release footnote instead.

---

### §11 Testable Predictions (lines 1710–1784)

**Clarity rating: A (unusual but effective)**

- **Strengths:** Falsifiability criteria are stated explicitly for each prediction. This is commendable.
- **Issue — Prediction 5 (Emender + attention hybrid) appears to contradict §6 hybrid-degradation finding without explicit reconciliation.** The §6 result shows Emender+GDN AABB *underperforms* either pure family. Prediction 5 predicts Emender+attention *outperforms* both. The distinction (GDN vs. attention as the mixing partner) should be stated explicitly in the prediction itself, not left for the reader to infer.

---

### §12 Future Work (lines 1785–1865)

**Clarity rating: B+ (specific and well-organized; partial overlap with §9)**

- **Issue — Overlap with §9:** The S5-generator-specific capacity bound (§12 lines 1789–1802) and the broader CMA-ES follow-up (§12 lines 1838–1846) are already described in §9. Rather than two near-identical descriptions, use a forward reference in §9 ("for the next-step experiments, see §12").

---

## Cross-Cutting Issues

### 1. Redundancy of core claims
The claims "E88 reaches 0.977 bpb after ~21 GPU-days," "22,200 programs per token," and "delta-correcting update is strictly more expressive than raw-write at matched FLOP cost" appear verbatim or near-verbatim in: abstract, §1 intro paragraph, §1 Contributions items, §5 results, §10 Conclusion, and in multiple table captions. Each section-by-section encounter is appropriate in isolation, but the density of repetition makes the paper feel longer than it is. Consider tightening the section bodies to refer back to prior statements rather than restating them.

### 2. Unnumbered subsection headings throughout
The `#heading(level: 2, numbering: none)` pattern is used for nearly every subsection in the paper. This creates a visual two-tier structure (numbered §N, unnumbered subsections) that looks professional but breaks cross-referencing. Readers cannot say "as in §3.2" — they must use the heading title, which is cumbersome. Either number all subsections, or adopt numbered sections at the section level and use bold run-in headers for sub-subsections.

### 3. Undefined abbreviations at first use
- **CMA-ES:** First used in §1 (Contribution 3, line 283); first expanded in §5 (line 818). Expand on first use.
- **bpb (bits per byte):** Used in abstract and §1 without definition; defined in §5 (line 916). Add a parenthetical at first use.
- **PNR:** Introduced as "pure nonlinear recurrent (PNR)" in §1 (line 226), but used before this without the abbreviation; then dropped in §4 and §5.

### 4. Figure caption length
@fig_lm_racers caption (lines 873–904) exceeds 200 words and includes methodology details, hedging statements, endpoint values, step numbers, timestamps, and color conventions. The caption should summarize what is shown; methodology details belong in the main text or a methods appendix. Target: ≤ 80 words.

### 5. "Frontier and unproven targets" subsection placement in §7
This subsection (lines 1232–1276) appears between Theorem set C′ and Theorem set D. It interrupts the cumulative theorem-set sequence. Move to end of §7 or merge with §9 (Limitations).

### 6. Pangenomic motivation in §2 is vestigial
Lines 325–335 introduce pangenomic sequences as the motivating application ("terabases per study," "trillions of tokens"). This context is never returned to, and none of the experiments use genomic data. Either develop this thread (explaining concretely why NL expressivity matters for DNA) or remove it and begin §2 directly with the linear-vs-nonlinear classification.

---

## Concrete List of Confusing or Mis-Ordered Passages

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | Abstract, lines 71–106 | Lean theorem names in abstract before architecture is defined; run-on Lean sentence | Moderate |
| 2 | §1 line 283 | "CMA-ES" used without expansion | Minor |
| 3 | §1 lines 288–302 | Numbering switch from "1." to "(a)" context is ambiguous | Moderate |
| 4 | §1 lines 312–320 | URL block disrupts intro narrative | Minor |
| 5 | §2 lines 325–335 | Pangenomic motivation paragraph: orphaned context never returned to | Moderate |
| 6 | §2 lines 370–382 | TC⁰ complexity paragraph: no gloss for applied-ML readers | Moderate |
| 7 | §3 lines 614–624 | Implementation details (float32, cast-back) in architecture section | Minor |
| 8 | §3 lines 634–703 | Ablation-by-elimination material in §3 rather than §6 | Moderate |
| 9 | §5 lines 849–865 | "third recipe property" heading + "A fourth run" body: number clash | Minor |
| 10 | §5 lines 873–904 | @fig_lm_racers caption: 200+ words with methodology, step counts, timestamps | High |
| 11 | §6 lines 1058–1088 | Hybrid-degradation result in §6 rather than §9 or its own section | Moderate |
| 12 | §7 lines 1232–1276 | "Frontier and unproven targets" mid-§7, between theorem sets C′ and D | High |
| 13 | §7 lines 1372–1386 | "Audit-recommended wording" framing raises unresolved provenance question | Minor |
| 14 | §9 lines 1513–1536 | Formal scope limitations repeat §7 with little added value | Moderate |
| 15 | §9 lines 1617–1633 | "The opposite architectural bet: hybrids" is a related-work discussion, not a limitation | Moderate |
| 16 | §11 Prediction 5 | Emender+attention prediction appears to contradict §6 Emender+GDN result; distinction not explicit | Moderate |
| 17 | Throughout | Core claims restated verbatim in abstract, §1, §5, §10, captions; excessive redundancy | Moderate |
| 18 | Throughout | Unnumbered `level: 2` headings prevent cross-referencing | Moderate |

---

## Validation Checklist

- [x] Section-by-section clarity assessment (§1 through §12, Abstract, Appendix)
- [x] Concrete list of confusing/redundant/mis-ordered passages with locations
- [x] Findings written to paper/review/clarity.md
