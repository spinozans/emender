# Consolidated Evaluation & Revision Plan — *Emending Nonlinear Recurrence* (`paper/main.typ`)

**Synthesizer:** Evaluator (agent-646), task `synthesize-review-into`
**Date:** 2026-05-31
**Inputs:** `paper/review/rigor.md`, `clarity.md`, `figures.md`, `citations.md`
**Mode:** READ-ONLY synthesis. No edits to `main.typ`. This document is the evaluation artifact only.

---

## 1. Overall Assessment

The paper advances three claims: **(1) viability** — a pure-nonlinear-recurrent (PNR) LM reaches sub-1-bpb at 1.3 B on a single workstation GPU; **(2) power separation** — the delta-correcting update is formally (Lean-verified) and empirically more expressive than raw-write at matched FLOP; **(3) supporting comparison** — E88 and Gated DeltaNet occupy the same sub-1-bpb loss-vs-wallclock band at 1.3 B.

**State of the paper: strong core, submission-blocked by disclosure and evidence-sufficiency gaps, not by flawed claims.** Across all four reviews the picture is consistent:

- **Formal results (§7) are exemplary** — properly scoped, explicit non-claims, no `sorry`/`axiom` in the trusted core. This is the paper's strongest asset and is essentially submission-ready (modulo placement of the "Frontier" subsection).
- **The empirical claims are real and the numbers reproduce** against the underlying data files (rigor.md cross-checked every headline number: all CONFIRMED). The problem is not fabrication or overreach in the results — it is **(a)** material experimental asymmetries disclosed only in supplementary files, not the main text, and **(b)** one comparison (QA/reasoning) too underpowered to carry any weight.
- **The single biggest acceptance risk** is the *selection-history asymmetry* behind "matched no-tuning across architectures at 8 M": the Emender's defaults were co-evolved on state-tracking objectives while the baselines' defaults were not. This confounds the headline expressivity claim and is currently confined to §9.
- **Presentation is professional but not yet submission-grade**: the Figure 2 caption is a methods section in disguise, the figure is sourced from an in-progress run with a `_draft` filename, several labeled floats are never cross-referenced, and two key result sets (QA panel, six-task sweep) are prose-only despite tabular data existing.
- **Citation coverage is solid** (0 broken keys, 44/44 resolve) with bounded, fixable gaps (GRU, Vaswani, Ba 2016, S4; two orphan bib keys).

**Recommendation:** Not ready to submit as-is. The work is publishable after a focused revision: the **blockers below are all addressable without new large-scale training** — they are disclosure additions, one comparison demotion, caption/table surgery, and citation fills. If the authors are willing to either expand the QA panel or demote it, and to disclose the racer asymmetries and the selection-history confound in the main text, the paper moves from "borderline / major-revision" to "accept with minor revision."

---

## 2. Severity Model

- **BLOCKER** — a reviewer can reasonably reject or demand major revision on this alone; the claim's support or the paper's integrity depends on it.
- **MAJOR** — a real weakness a careful reviewer will flag; should be fixed before submission but does not by itself sink the paper.
- **MINOR** — polish, navigability, and consistency; fix opportunistically.

Each consolidated issue lists its **source reviews** (R=rigor, C=clarity, F=figures, K=citations) so overlaps are visible.

---

## 3. Blockers

### BL-1 — "Matched no-tuning" conceals a selection-history asymmetry favoring the Emender
**Sources:** R (OC-2 / Flag 4, *High*) · C (§3 ablation placement, §6) · F (tab_ablation)
**Sections:** §6 (`Matched no-tuning across architectures at 8 M`, line 941); §3 (`Ablation by architecture`, line 634); §9 (line 1600, where it is currently disclosed)
**Issue:** The 8 M expressivity probes claim "matched no-tuning across architectures," but the Emender's defaults are the endpoint of an ablation lineage *selected partly on state-tracking behaviour*, whereas GDN's and M²RNN's published defaults were selected on language-modelling loss. "Matched no-tuning" means no *new* probe-specific HPO — it does **not** mean the starting configurations were chosen by a common criterion. Any 8 M Emender advantage may partly reflect this prior selection rather than the write rule itself. This is the load-bearing comparison for Claim 2 (empirical side).
**Why it is a blocker:** The headline conclusion "delta correction is *the* differentiating ingredient" is derived from a comparison with a known, directional confound. It is honestly disclosed — but only in §9, far from the claim.
**Fix:** Move (or mirror) the selection-history caveat into §6 *at the point the claim is made*, relabel "matched no-tuning" to make the asymmetry explicit (e.g., "no probe-specific tuning; note defaults were selected under different criteria — see §9"). Foreground the S₃ solvable-group control (M²RNN-CMA scores 0.31 on S₃, where non-solvability cannot be the explanation) as the partial mitigation it is. Best resolution is the matched-search experiment the paper already proposes; absent that, the disclosure must be co-located with the claim.

### BL-2 — QA/reasoning panel is too underpowered to support comparison, yet is presented as evidence
**Sources:** R (Flag 7 / SI-3, *High*) · F (ISSUE-08, *High*) · C (§6 Issue 2)
**Sections:** §6 (`QA and reasoning panel at 1.3 B`, line 1090)
**Issue:** The knowledge panel uses 50 items/task (SE ≈ 6 pp); differences below ~12 pp are within 2 SE. The reasoning panel is additionally evaluated at *different training steps per architecture* (E88 ~957K, GDN ~1,272K, M²RNN-CMA ~879K — rigor MC-5). At this resolution the panel cannot discriminate architectures. Separately (figures ISSUE-08), the numbers are reported in prose only, though full tables exist in `results/qa_reasoning/section_draft.md`.
**Why it is a blocker:** Two reviewers independently flagged this from opposite directions (under-powered *and* under-presented). A panel that cannot support a comparison should not read as if it does.
**Fix (choose one):**
  1. **Demote** — soften to "preliminary parity-rate evidence," state explicitly it cannot differentiate at current N, and move to an appendix; **or**
  2. **Strengthen** — expand to ≥500 items/task *and* evaluate all architectures at a matched step, then present as a proper labeled table.
  Either way, add the table (resolves ISSUE-08) and note the unequal-step caveat.

### BL-3 — 1.3 B racer asymmetries (single seed, unequal tokens/steps/params) are not disclosed in the main text
**Sources:** R (SI-2 *High*, SI-4 *Medium*, MC-2; Flag 1) · F (ISSUE-05)
**Sections:** §5 (`Loss-vs-wallclock racer`, line 866; Fig. 2 caption, lines 873–904)
**Issue:** The "same sub-1-bpb band" claim (Claim 3) rests on a comparison where, per AS_OF.md but **not** the paper: GDN has seen ~5–10% more tokens (15.13 B vs E88 14.39 B vs M²RNN-CMA 13.75 B), ~31% more steps, and has ~6.2% more parameters; each architecture is a **single seed**; training is **ongoing**; and E88 suffered a **NaN divergence at step ~247,250, repaired/resumed from step 231,000** (the meaning of "stitched"). All material facts live in a supplementary data file.
**Why it is a blocker:** A comparison claim whose key confounds are only in supplementary files is a reviewer magnet, especially when the favored-direction asymmetry (GDN's token/step lead) cuts *against* the paper's own model — making non-disclosure look incautious rather than self-serving, but still a disclosure failure.
**Fix:** In §5 main text: (a) state the per-architecture token/step/param counts (or add them to the setup table — see MA-3); (b) define "stitched" and disclose the divergence-and-repair at first use, not only in AS_OF.md; (c) keep the honest "same band, GDN nominally ahead" framing but add a token-normalized (bpb-vs-tokens) view, or explicitly note the comparison is wallclock-matched, not token-matched. The single-seed caveat is already in §9 — cross-reference it from §5.

### BL-4 — Figure 2 is sourced from an in-progress run flagged "do not cite"; filename and caption signal provisional status
**Sources:** F (ISSUE-05, *High*; ISSUE-04, *High*) · R (Flag 1) · C (§5 Issue 1)
**Sections:** §5, `<fig_lm_racers>`, image `results/figure_2/figure_2_draft.png`; caption lines 873–904
**Issue:** The figure underpinning Claims 1 and 3 is named `figure_2_draft.png`, and `results/figure_2/AS_OF.md` states "Training is **in progress**" and "Do not cite these results without re-running `smooth.py`." The caption simultaneously admits "training continues." For a submission, the central empirical figure cannot be a draft-named snapshot of an explicitly not-to-be-cited run.
**Why it is a blocker:** This is a publish-readiness gate: the source data carries an explicit do-not-cite warning.
**Fix:** Before submission, finalize the run (or freeze a defensible checkpoint), re-run `smooth.py`, rename the asset away from `_draft`, and update the caption's "training continues" hedge to a fixed as-of statement. If the snapshot is intentionally kept, the paper must justify citing data its own provenance file says not to cite. This issue is entangled with the BL-3 disclosure and the MA-1 caption fix (do all three together).

---

## 4. Major Issues

### MA-1 — Figure 2 caption is a methods section (~200–300 words / ~30 lines)
**Sources:** C (§5 Issue 1, cross-cutting #4, *High*) · F (ISSUE-04, *High*)
**Sections:** §5, `<fig_lm_racers>` caption, lines 873–904
**Issue:** The caption embeds bytes/token derivation, tokenizer name, script paths, CMA-ES sweep counts, the delta-off ablation, active-training disclosure, a color legend, and a timestamp. A caption should identify what is shown and the takeaway (target ≤ 80 words).
**Fix:** Move methodology to §5 body / methods; reduce caption to what-is-shown + endpoint values + key caveat pointer. Coordinate with BL-3 (the disclosure text relocates *into* §5) and BL-4 (the as-of wording).

### MA-2 — Ablation is elimination-by-architecture, not controlled single-variable isolation; and it sits in §3 as argument, not description
**Sources:** R (OC-1 *Medium*, MC-1 *Medium*) · C (§3 Issue 2, *Moderate*) · F (ISSUE-02 cross-ref)
**Sections:** §3 (`Ablation by architecture`, line 634; `<tab_ablation>`)
**Issue:** The three compared architectures differ in multiple properties at once (update rule *and* model shape/geometry). No architecture has delta-correction *without* temporal state nonlinearity, so the 2×2 is incomplete; the elimination argument is valid only if no other property co-varies. The 8 M M²RNN-CMA shape is also not tabulated, so "parameter-matched" is unverifiable (see MA-3). Clarity additionally notes this analytical subsection is misplaced in the *Architecture* section.
**Fix:** In the `<tab_ablation>` caption, state explicitly that this is elimination across architectures, not single-variable manipulation, and name the missing cell. Tabulate the 8 M shapes for all three architectures. Consider relocating the analytical subsection to §6, or numbering it so it reads as results.

### MA-3 — §5 model-setup table (U1) and the 8 M probe shapes are unlabeled / unspecified
**Sources:** F (ISSUE-07, *Moderate*) · R (Flag 3 / "Issue", missing 8 M shapes, *Medium*)
**Sections:** §5 setup table (lines ~804–815, plain `#align(center)[#table(...)]`, no `#figure`/caption/label); §6 / §3 8 M shapes
**Issue:** The table defining the three primary models has no caption, label, or number — it cannot be cross-referenced and does not enter a list of tables. Separately, the M²RNN-CMA 8 M shape (dim/depth/H/N) is never given, so "parameter-matched" expressivity probes are unverifiable.
**Fix:** Wrap U1 in `#figure()` with caption + `<tab_setup>`; add the per-architecture token/step/param columns BL-3 needs here. Add a small table of the 8 M probe shapes for all architectures.

### MA-4 — Unsupported systems claims in §4 (no benchmark data)
**Sources:** R (UA-1/2/3, OC-3, *Low–Medium*) · C (§4 Issue 1)
**Sections:** §4 — fused kernel "≈50–60 ms/step" saving; Triton "~70% of hand-tuned throughput in ~one week"; Newton-iteration "significantly worse in throughput" (line ~785)
**Issue:** Three quantitative/comparative performance claims appear with no table, profile, or source.
**Fix:** Back each with a benchmark table, or soften to clearly-labeled estimates ("estimated," "anecdotal") and/or move the Newton-iteration comparison to §9. Low scientific stakes but easy reviewer ammunition.

### MA-5 — "Frontier and unproven targets" breaks the §7 theorem sequence
**Sources:** C (§7 Issue 1, cross-cutting #5, *High*)
**Sections:** §7, lines 1232–1276 (between Theorem set C′ and set D)
**Issue:** A limitations/frontier block is inserted mid-proof-sequence, undercutting the cumulative force of the formal results — the paper's strongest section.
**Fix:** Move to the end of §7 (after all theorem sets) or merge into §9. Pure restructuring, no content change.

### MA-6 — Missing foundational citations: GRU, Vaswani 2017, Ba 2016, S4
**Sources:** K (§4 table: GRU *High*, Vaswani *Low*, Ba 2016 *Medium*, S4 *Low*)
**Sections:** §2 (SSM lineage), §8 (fast-weight ancestry, lines 1402–1425), §9 (transformer comparison, line 1651)
**Issue:** GRU is a first-class comparison target but is cited only via the LSTM key (`@lstm1997`). Ba et al. 2016 (fast-weights) is the missing 1992→2021 link in the ancestry the paper explicitly documents. Vaswani 2017 is absent despite extensive transformer discussion. S4 is uncited despite SSM-lineage framing.
**Fix:** Add Cho 2014 / Chung 2014 (GRU), `@ba2016fastweights` (§8), Vaswani 2017, and S4 (Gu 2022, §2) to `refs.bib` and cite at first mention.

### MA-7 — Two orphan bib entries, one with a notation collision
**Sources:** K (orphans, *High*)
**Sections:** `refs.bib`
**Issue:** `ndm2026` (self-reference, placeholder "arXiv ID to be assigned") and `s5_2022` (the SSM paper "S5") are in the bib but never cited. `s5_2022` is especially hazardous: the key silently collides with the paper's pervasive $S_5$ symmetric-group notation.
**Fix:** Remove both unless cited; if `ndm2026` is the intended self-cite, assign a real ID. Rename or annotate any retained `s5_2022` to avoid the $S_5$ collision.

---

## 5. Minor Issues

| ID | Issue | Sources | Section / line | Fix |
|----|-------|---------|----------------|-----|
| MI-1 | Three labeled floats never cross-referenced (`@fig_arch`, `@tab_ablation`, `@fig_hybrid`) | F (ISSUE-01/02/03) | §3 (440, 634), §6 (1058) | Add `@`-citations at the natural prose points (§3 ×2; §6 + §9 for hybrid) |
| MI-2 | Hybrid-degradation result placement + §11 Prediction 5 apparent contradiction | C (§6 Issue 1, §11), R (MC-4), F (ISSUE-03) | §6 (1058), §11 | Consider moving the experiment to §9/own section; in Prediction 5 state explicitly that *attention* (not GDN) is the mixing partner, reconciling with the §6 AABB result |
| MI-3 | Six-task canonical sweep reported in prose only | F (ISSUE-09), C (§6 Issue 3) | §6 (1032) | Add a labeled table for the six tasks × architectures (+ length-extrap) |
| MI-4 | Source citations point to internal dev notes (`ndmpapernotes.md`) | F (ISSUE-06) | `<tab_s5>`, `<fig_s5_bars>` captions | Repoint to a stable CSV/script before submission |
| MI-5 | Undefined abbreviations at first use: bpb, CMA-ES, PNR | C (cross-cutting #3) | Abstract, §1 (226, 283) | Expand/parenthesize on first use; `bpb` formula is in §5 (line 916) — forward-reference it |
| MI-6 | Pervasive unnumbered `level: 2` headings prevent cross-referencing | C (cross-cutting #2) | Throughout | Number subsections or use run-in bold sub-subheads |
| MI-7 | Core claims restated near-verbatim across abstract/§1/§5/§10/captions | C (cross-cutting #1) | Throughout | Tighten section bodies to refer back rather than restate |
| MI-8 | Three commented-out draft abstracts left in source (lines 13–59) | C (Abstract Issue 4) | Source head | Delete before submission |
| MI-9 | Lean theorem identifiers in the abstract before architecture is defined; run-on sentence | C (Abstract Issues 1–2) | Abstract (71–106) | Keep one representative theorem per result; move identifiers to §7 |
| MI-10 | Pangenomic motivation in §2 is never returned to | C (§2 Issue 1) | §2 (325–335) | Develop the DNA-expressivity link or cut it |
| MI-11 | TC⁰/NC¹ digression lacks a one-line gloss for applied-ML readers | C (§2 Issue 2) | §2 (370–382) | Add a one-sentence gloss |
| MI-12 | Implementation details (float32, cast-back, tanh form) inside §3 Architecture | C (§3 Issue 1) | §3 (612–624) | Move to §4 or an appendix |
| MI-13 | "third recipe property" heading vs "A fourth run" body — counting clash | C (§5 Issue 2) | §5 (847–865) | Align the wording |
| MI-14 | "Audit-recommended wording" provenance unresolved for reader | C (§7 Issue 2) | §7 (1372–1386) | Identify the audit source or rephrase |
| MI-15 | §9 over-long; "Formal scope" repeats §7; "opposite architectural bet: hybrids" is related-work | C (§9 Issues 1–3) | §9 (1513–1536, 1617–1633) | Cross-reference §7; move the hybrids bet to §8; trim |
| MI-16 | §12 overlaps §9 (S₅ bound, CMA-ES follow-up) | C (§12) | §12 (1789–1846) | Replace duplicate text with a forward reference |
| MI-17 | First-mention citation gaps in §2 list (RetNet, RWKV, HGRN2, mLSTM) | K (1.1) | §2 (366–369) | Add inline cites at first mention (entries already exist) |
| MI-18 | Uncited evaluation-task provenance for the six-task suite | K (1.3) | §6 (1035–1042) | Cite a prior work defining/benchmarking the suite |
| MI-19 | Two corporate-author bib entries lack individual names / eprint | K (olmohybrid2026, pararnn2025) | `refs.bib` | Complete authorship + arXiv IDs |
| MI-20 | "No classical LSTM/GRU ≥500 M on Pile-class corpus" claim uncited | K (1.4) | §8 (1504–1507) | Add a scaling-survey / empirical-record citation |
| MI-21 | Orphan figure assets | F (§3) | `figures/figure_2_placeholder.svg`, `results/figure_3/combined.csv`, `results/cmaes_burst_v2/`, `results/cma_flop_rate/` | Delete placeholders; consider `cma_flop_rate` FLOPs-per-bit figure as an appendix supporting §5's "same band" claim |

---

## 6. Deduplication Map (overlaps across reviews)

These issues were raised by more than one reviewer and are consolidated above rather than double-counted:

| Consolidated | rigor.md | clarity.md | figures.md | citations.md |
|---|---|---|---|---|
| **BL-1** selection-history / "matched no-tuning" | OC-2, Flag 4 | §3 ablation, §6 | tab_ablation | — |
| **BL-2** QA panel underpowered + no table | Flag 7, SI-3, MC-5 | §6 Issue 2 | ISSUE-08 | — |
| **BL-3** racer asymmetries undisclosed | SI-2, SI-4, MC-2, Flag 1 | — | ISSUE-05 | — |
| **BL-4** Fig 2 draft/in-progress source | Flag 1 | §5 Issue 1 | ISSUE-04/05 | — |
| **MA-1** Fig 2 caption too long | — | §5 Issue 1, #4 | ISSUE-04 | — |
| **MA-2** ablation by elimination | OC-1, MC-1 | §3 Issue 2 | ISSUE-02 | — |
| **MA-3** unlabeled setup table + 8M shapes | Flag 3, "Issue" | — | ISSUE-07 | — |
| **MA-4** §4 systems claims | UA-1/2/3, OC-3 | §4 Issue 1 | — | — |
| **MA-5** §7 Frontier placement | — | §7 Issue 1, #5 | — | — |
| **MA-6/7** citations | — | — | — | §1–2, table |
| **MI-1** missing cross-refs | — | — | ISSUE-01/02/03 | — |
| **MI-2** hybrid placement / Pred-5 | MC-4 | §6 Issue 1, §11 | ISSUE-03 | — |
| **MI-3** six-task table | — | §6 Issue 3 | ISSUE-09 | — |

Items appearing in only one review (e.g., R's formal-results praise, C's heading/redundancy/abbreviation set, K's bib completeness, F's orphan inventory) are carried through at their original severity.

---

## 7. Ordered Revision Plan

Ordered by *acceptance impact per unit effort*. Phases 1–2 are the must-do gate for submission.

**Phase 1 — Disclosure & evidence integrity (blockers; no new large training runs required)**
1. **BL-3 + MA-3:** Add per-architecture token/step/param counts to §5 (via a labeled setup table); define "stitched" and disclose the E88 divergence/repair in §5; cross-reference the single-seed caveat from §9.
2. **BL-1:** Co-locate the selection-history caveat in §6 at the claim; relabel "matched no-tuning"; foreground the S₃ control as partial mitigation.
3. **BL-2:** Decide QA-panel fate — demote to appendix as "preliminary, non-differentiating," *or* expand to ≥500 items/task at matched step; either way add the table.
4. **BL-4 + MA-1:** Finalize/freeze the Fig 2 run, re-run `smooth.py`, rename off `_draft`, fix the as-of wording, and cut the caption to ≤80 words (methods text moves into §5).

**Phase 2 — Scientific framing & evidence presentation (major)**
5. **MA-2:** Reframe the ablation caption as elimination-not-isolation; tabulate 8 M shapes; consider relocating the analytical subsection.
6. **MA-4:** Back or soften the three §4 systems claims.
7. **MA-5:** Move §7 "Frontier" block to section end (or §9).
8. **MA-6 + MA-7:** Add GRU / Vaswani / Ba 2016 / S4; remove or fix the two orphan bib keys (esp. the `s5_2022` ↔ $S_5$ collision).

**Phase 3 — Navigability & evidence completeness (minor, batchable)**
9. **MI-1, MI-3:** Add the three missing cross-references; add the six-task sweep table.
10. **MI-2:** Resolve the §11 Prediction-5 / §6 hybrid apparent contradiction (name attention vs GDN); settle hybrid-result placement.
11. **MI-4, MI-17–MI-20:** Repoint informal data sources to stable files; fill §2 first-mention cites, task provenance, the scaling claim, and corporate-author bib entries.

**Phase 4 — Polish (minor, low-risk, do last)**
12. **MI-5/MI-6/MI-7/MI-8/MI-9:** Abbreviations on first use; subsection numbering; trim redundancy; delete commented draft abstracts; thin the abstract's theorem identifiers.
13. **MI-10–MI-16, MI-21:** §2 pangenomics decision; TC⁰ gloss; move §3 implementation details; fix §5 counting clash; §7 audit provenance; §9/§12 de-duplication; orphan-asset cleanup (and consider promoting `cma_flop_rate` as an appendix figure).

---

## 8. Claim-Level Verdict (carried from rigor.md, contextualized)

| Claim | Verdict | Gating issues |
|---|---|---|
| **1. Viability** (E88 sub-1-bpb @1.3 B) | **Supported** — number confirmed | BL-3 (divergence/stitch disclosure), BL-4 (figure provenance) |
| **2. Power separation** (delta-correct > raw-write) | **Formally supported (exemplary); empirically supported with caveats** | BL-1 (selection-history confound) is the live risk; formal side (§7) is clean |
| **3. Supporting comparison** (E88 ≈ GDN band) | **Supported with important qualifications** | BL-3 (unequal tokens/steps/params, single seed) — disclosure, not correctness |

No claim was found *unsupported by the data*; every headline number reproduces. The gating issues are about **honest disclosure** and **evidence sufficiency**, which is why the paper is revisable-to-accept rather than fundamentally flawed.

---

## 9. Validation Checklist (this task)

- [x] Single consolidated report at `paper/review/EVALUATION.md`
- [x] Issues deduped and ranked (blocker / major / minor) with section refs — see §3–§6
- [x] Concrete, ordered revision recommendations — see §7
- [x] Overall assessment of the paper's current state — see §1 and §8
