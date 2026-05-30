# Citation Review: Related Work & Citation Completeness
## paper/main.typ vs paper/refs.bib
**Reviewer:** Evaluator agent (review-related-work)
**Date:** 2026-05-30

---

## Executive Summary

The citation coverage is **overall solid**: all 44 cite keys used in `main.typ` resolve to entries in `refs.bib` (zero broken keys), and the bibliography is well-matched to the paper's claims. The principal gaps are (1) two orphaned bib entries never cited in the text, (2) first-mention citation omissions for several architectures in §2's enumeration list, (3) three missing foundational works (GRU, Ba et al. 2016 fast weights, and arguably Vaswani et al. 2017), (4) uncited evaluation tasks in §6, and (5) two bib entries with incomplete/corporate authorship.

---

## 1. Claims Missing Citations

### 1.1 First-mention gaps in §2 (lines 366–369)

The sentence *"Mamba, Mamba2 (SSD), RetNet, GLA, DeltaNet, Gated DeltaNet, RWKV-4/5/6/7, HGRN2, mLSTM, MinGRU/MinLSTM @mingru_2024 and Griffin's RG-LRU @griffin2024 are all linear-state by this criterion"* omits inline citations for:

| Architecture | First inline cite | Location of that cite |
|---|---|---|
| RetNet | @retnet2023 | §8 Related Work, line 1431 |
| RWKV-4/5/6/7 | @rwkv4_2023 / @rwkv7_2025 | §3 line 700 / §8 line 1434 |
| HGRN2 | @hgrn2_2024 | §8 Related Work, line 1436 |
| mLSTM | @xlstm7b2025 | §8 Related Work, line 1438 |

All four have bib entries; the citation is simply deferred to §8. For a reader following the §2 list, these look like uncited claims. **Recommended fix:** add inline cites at first mention in §2.

### 1.2 GRU cited only via LSTM key (lines 240, 384, 1504–1505)

"LSTM and GRU" appears three times. The citation at line 1504 ("*Classical LSTM/GRU* @lstm1997") attaches only the LSTM paper to both. GRU has no independent citation. The canonical references (Cho et al. 2014, "Learning Phrase Representations using RNN Encoder-Decoder…", or Chung et al. 2014, "Empirical Evaluation of Gated Recurrent Neural Networks…") are absent from `refs.bib`. **Gap:** GRU is a first-class comparison target (§9 Limitations cites ParaRNN training "LSTM and GRU at 7 B") and should be cited.

### 1.3 Uncited evaluation tasks in §6 (lines 1035–1042)

The six-task canonical sweep — parity, modular counter (K=5), FSM tracking (K=4), Dyck-1, associative recall, selective copy — is described as a standard benchmark battery with no citation for task provenance or prior use. These tasks have published prior uses (e.g., parity and state-tracking tasks appear in Merrill et al. 2024, Liu et al. 2024, and others). The absence of any citation leaves the reader unable to calibrate the tasks or verify that the paper's setup matches canonical definitions. **Recommended:** cite at least one prior work that defines or benchmarks this exact set.

### 1.4 Missing step: "no published classical LSTM/GRU model has reached ≥500 M parameters on a Pile-class corpus" (lines 1504–1507)

This empirical claim is stated without a citation to a survey or negative result paper. It functions as a literature claim and readers may want a pointer. @lstm1997 (the original LSTM paper) is the only reference here, which does not support the scaling claim.

### 1.5 Transformer comparison omits foundational citation (§9 Limitations, line 1653–1658)

The Transformer comparison section acknowledges the omission of an attention baseline but never cites the transformer architecture itself. Given repeated mentions of "attention", "hybrid architectures", and the field's "architectural bets" on attention, the absence of Vaswani et al. 2017 ("Attention Is All You Need") from the bibliography is a notable gap. The paper discusses the transformer as the dominant baseline but never formally introduces it via citation.

---

## 2. Notes on Missing / Weak / Duplicate References vs refs.bib

### 2.1 Orphaned bib entries (in refs.bib, never cited in main.typ)

| Key | Title | Status |
|---|---|---|
| `ndm2026` | "Nonlinear Delta Memory..." (self-reference) | Placeholder; arXiv ID "to be assigned"; never cited in text. **Remove or assign a real ID.** |
| `s5_2022` | "Simplified State Space Layers for Sequence Modeling" (Smith et al. 2022) | Never cited. Notably, `s5_2022` abbreviates the SSM paper "S5" — this conflicts silently with the paper's extensive use of $S_5$ (symmetric group). The presence in the bib with this key could confuse maintainers. **Remove or add a clear name distinction.** |

### 2.2 Incomplete / corporate-author bib entries

| Key | Issue |
|---|---|
| `olmohybrid2026` | `author = {{Allen Institute for AI}}` with no individual names, no arXiv ID/eprint, `note = {Technical report}`. The actual report has named authors (Soldaini et al. or similar) and an arXiv identifier. **Should be completed.** |
| `pararnn2025` | `author = {{Apple Machine Learning Research}}` — corporate attribution, no individual names. The actual paper has individual authors. **Should be completed.** |
| `ndm2026` | Self-citation placeholder; `note = {arXiv preprint; identifier to be assigned}`. This should be updated when the arXiv ID is assigned, or removed if not yet submittable. |

### 2.3 Missing key prior work in the fast-weights ancestry (§8)

§8 traces the delta-correction line as: Widrow–Hoff (1960) → Schmidhuber (1992) → Schlag, Irie & Schmidhuber (2021) → DeltaNet (2024). A major link in the modern revival is missing:

- **Ba et al. 2016, "Using Fast Weights to Attend to the Recent Past"** (NeurIPS 2016): This paper introduced fast-weight programmers to the modern deep-learning context and is the direct precursor that Schlag et al. 2021 explicitly extend. Its absence leaves a seven-year gap (1992–2021) in the ancestry that the paper explicitly claims to document. **Should be added to §8.**

### 2.4 Missing S4 citation in the SSM lineage

§2 and §8 cite Mamba (@mamba2024) and Mamba2 (@mamba2_2024) as modern SSMs but do not cite **S4** (Gu et al. 2022, "Efficiently Modeling Long Sequences with Structured State Spaces"), the foundational work that Mamba directly supersedes. For a paper positioning itself within the SSM lineage, S4 is a standard background citation. **Recommended:** add S4 to the §2 background paragraph on state-space models.

### 2.5 No duplicate keys detected

All keys in `refs.bib` are unique. No duplicate entries found.

---

## 3. Relevance Check for Cited Works

All 44 cited works are directly relevant to their call sites. No gratuitous or tangential citations were found. Specific checks:

- **Pangenomics citations** (@hprc2023, @guarracino2023acrocentric, @pggb2024): Used in exactly one sentence of §2 to motivate the workload. Appropriate; not padding.
- **Eval benchmarks** (@arc2018, @hellaswag2019, @sciq2017, @openbookqa2018, @boolq2019, @bbh2022, @reclor2020, @folio2022): All cited at the specific §6 QA/reasoning panel where results from those benchmarks are reported.
- **Lean 4 / Mathlib** (@lean42021, @mathlib4): Cited in §7 where the formal core is described. @mathlib4 is correctly cited at line 1114 despite spanning a line break in source.
- **CMA-ES** (@cmaes2003): Cited where the optimization method is first used. Correct.
- **Triton** (@triton2019): Cited in §4 Systems. Correct.
- **DiLoCo** (@diloco2023): Cited once in §4 for the distributed training strategy. Correct.

---

## 4. Summary Findings Table

| Severity | Issue | Location | Action |
|---|---|---|---|
| **High** | GRU has no independent citation; @lstm1997 is applied to both | Lines 240, 384, 1504 | Add Cho et al. 2014 or Chung et al. 2014 to bib |
| **High** | `ndm2026` in bib but never cited; placeholder arXiv ID | refs.bib line 5 | Remove placeholder or assign real arXiv ID |
| **High** | `s5_2022` in bib but never cited; naming conflicts with $S_5$ symmetric group notation | refs.bib line 177 | Remove from bib; re-add only if cited |
| **Medium** | olmohybrid2026: corporate author, no arXiv ID | refs.bib line 439 | Complete with individual authors and eprint |
| **Medium** | pararnn2025: corporate author, no individual names | refs.bib line 313 | Complete with individual authors |
| **Medium** | Ba et al. 2016 fast-weights missing from fast-weight ancestry (§8) | §8 lines 1405–1425 | Add @ba2016fastweights to bib and §8 |
| **Medium** | First-mention citation gaps for RetNet, RWKV, HGRN2, mLSTM in §2 | Lines 366–369 | Add inline cites at §2 first mention |
| **Medium** | Uncited evaluation tasks in §6 canonical sweep | Lines 1035–1042 | Add provenance citation for task suite |
| **Low** | Vaswani et al. 2017 (Attention Is All You Need) absent from bib | §9 Limitations, §1 | Add and cite for transformer background |
| **Low** | S4 (Gu et al. 2022) not cited despite SSM lineage discussion | §2, §8 | Add S4 citation in §2 SSM background |
| **Low** | Claim that LSTM/GRU never reached 500 M on Pile-class corpus has no supporting citation | Line 1504–1507 | Add a scaling survey or empirical-record citation |

---

## 5. No Broken or Mismatched Cite Keys

Every `@key` in `main.typ` (excluding Typst-internal refs `@fig_*`, `@tab_*`, `@eq`, `@preview`, `@gmail`) resolves to an entry in `refs.bib`. **Zero broken references.**

