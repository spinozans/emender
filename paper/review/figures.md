# Figures & Tables Review — Emender Paper

**Reviewer:** Evaluator agent (review-figures-tables)
**Date:** 2026-05-30
**Scope:** `paper/main.typ`, `paper/figures/`, `paper/results/`
**Mode:** READ-ONLY. No edits to main.typ.

---

## 1. Complete Inventory

### 1.1 Labeled figures and tables (with `<label>` in main.typ)

| ID | Type | Label | Source | Section | Caption present | Explicit cross-ref |
|----|------|-------|--------|---------|-----------------|-------------------|
| F1 | Figure (Typst inline) | `<fig_arch>` | Inline code | §3 | ✓ | **MISSING** |
| T1 | Table (Typst inline) | `<tab_ablation>` | Inline code | §3 | ✓ | **MISSING** |
| F2 | Figure (image) | `<fig_lm_racers>` | `results/figure_2/figure_2_draft.png` | §5 | ✓ | ✓ (§9 ×2) |
| T2 | Table (Typst inline) | `<tab_s5>` | Inline code | §6 | ✓ | ✓ (§3, §7) |
| F3 | Figure (image) | `<fig_s5_bars>` | `figures/s5_expressivity_seeds.png` | §6 | ✓ | ✓ (§7) |
| F4 | Figure (image) | `<fig_hybrid>` | `figures/hybrid_degradation_seeds.png` | §6 | ✓ | **MISSING** |

### 1.2 Unlabeled inline structures

| ID | Type | Location | Caption | Label |
|----|------|----------|---------|-------|
| U1 | Table (plain `#align(center)[#table(...)]`) | §5, lines 804–815 | None | None |

### 1.3 Files referenced by figures

| File path | Referenced by | Exists? |
|-----------|--------------|---------|
| `results/figure_2/figure_2_draft.png` | `<fig_lm_racers>` | ✓ |
| `figures/s5_expressivity_seeds.png` | `<fig_s5_bars>` | ✓ |
| `figures/hybrid_degradation_seeds.png` | `<fig_hybrid>` | ✓ |
| `paper/results/figure_4_hybrid/` (dir) | Caption of `<fig_hybrid>` | ✓ |
| `paper/figures/plot_hybrid_degradation.py` | Caption of `<fig_hybrid>` | ✓ |
| `paper/figures/plot_expressivity_seeds.py` | Caption of `<fig_s5_bars>` | ✓ |
| `paper/ndmpapernotes.md` | Captions of `<tab_s5>` and `<fig_s5_bars>` | ✓ |

---

## 2. Issues by Figure/Table

### ISSUE-01 — `<fig_arch>`: No explicit cross-reference
**Figure ID:** `<fig_arch>` (Figure 1 — Emender architecture)
**Problem:** The figure has a label `<fig_arch>` but there is no `@fig_arch` citation anywhere in the body text. The figure is introduced only by the surrounding prose ("The figure and §4 use the same trusted geometry"), which could leave a reader unclear which figure is meant, especially since the figure immediately follows the prose and no formal pointer exists. All other named figures/tables should be explicitly cited.
**Severity:** Moderate — cross-references are standard academic practice and allow readers to navigate non-linear reads.

---

### ISSUE-02 — `<tab_ablation>`: No explicit cross-reference
**Figure ID:** `<tab_ablation>` (Table — ablation by elimination)
**Problem:** `<tab_ablation>` has a label but no `@tab_ablation` citation in the body. The table appears in §3 surrounded by prose that implies it, but there is no formal pointer. Standard practice is to cite every labeled float.
**Severity:** Low-moderate — context makes the reference clear, but formal citation is missing.

---

### ISSUE-03 — `<fig_hybrid>`: No explicit cross-reference
**Figure ID:** `<fig_hybrid>` (Figure 4 — hybrid degradation)
**Problem:** `<fig_hybrid>` has a label but no `@fig_hybrid` anywhere in the body. The §6 prose ("We test the pattern…") and §9 ("The hybrid-degradation finding in §6") discuss the result but neither cites the figure directly. §11 prediction 5 also references the result in prose without citing the figure.
**Severity:** Moderate — same issue as ISSUE-01/02. More noticeable here because §9 and §11 reference the result at a distance from the figure's location.

---

### ISSUE-04 — `<fig_lm_racers>`: Caption is extremely long (~30 lines)
**Figure ID:** `<fig_lm_racers>` (Figure 2 — 1.3 B loss-vs-wallclock racer)
**Problem:** The caption is approximately 300 words across ~30 lines. A standard academic caption is 1–4 sentences that identify what is shown and highlight the key takeaway. The current caption contains:
- Full methodology text (bytes/token derivation, tokenizer name, script paths)
- Experimental caveats (number of CMA-ES sweeps, delta-off ablation)
- Active-training status disclosure ("Recorded from a 2026-05-29T18:04:51Z active-log snapshot; training continues")
- Color legend that could be placed in the body or a figure note

Much of this belongs in the §5 body text, not in the caption. Readers scanning figures are overwhelmed; readers reading linearly find the same information repeated. 
**Severity:** High — caption length is a significant presentation deficiency for a journal/preprint submission.

---

### ISSUE-05 — `<fig_lm_racers>`: "Draft" filename signals provisional status
**Figure ID:** `<fig_lm_racers>`
**Problem:** The image file is named `results/figure_2/figure_2_draft.png`. The word "draft" in a published figure filename is unprofessional and signals the figure is not finalized. Additionally, `paper/results/figure_2/AS_OF.md` explicitly states "Training is **in progress**" and "Do not cite these results without re-running `smooth.py`."
**Severity:** High — the figure itself is sourced from an active training snapshot. If paper submission occurs before training is complete, the caption wording "training continues" and the filename both need updating.

---

### ISSUE-06 — `<tab_s5>` and `<fig_s5_bars>`: Source cited as informal dev notes
**Figure IDs:** `<tab_s5>` (Table 2) and `<fig_s5_bars>` (Figure 3)
**Problem:** Both captions cite `paper/ndmpapernotes.md lines 153–173` as the data source. This is an internal development notebook, not an appropriate archival data source for a published paper. A published figure should cite a stable, reproducible data source — e.g., a CSV in `paper/results/`, a script, or a DOI-linked dataset.
**Severity:** Moderate — fine for a preprint in active development, but must be resolved before formal submission.

---

### ISSUE-07 — `U1`: §5 setup table has no caption or label
**Figure ID:** U1 (§5 model setup table, lines 804–815)
**Problem:** The model/params/batch/shape table in §5 is placed as `#align(center)[#table(...)]` with no `#figure()` wrapper, no caption, and no label. This means:
1. The table cannot be cross-referenced by number in body or supplementary text.
2. It does not appear in a table of contents or list of tables.
3. Readers cannot distinguish it as an official numbered table vs. informal display.

The table is important (it defines the three primary models used throughout) and should be a properly captioned, labeled table.
**Severity:** Moderate — affects citability and navigability of a key reference table.

---

### ISSUE-08 — QA/reasoning panel: no figure or table despite data existing
**Section:** §6 "QA and reasoning panel at 1.3 B"
**Problem:** §6 reports detailed numerical results for a 300-item QA panel and a reasoning panel (BIG-Bench Hard, ReCLor, FOLIO), but presents them only in prose. `paper/results/qa_reasoning/` contains `fact_panel_latest.csv`, `knowledge_probe_40item.csv`, `racer_panel_300item_progression.csv`, `reasoning_panel_latest.csv`, and a full `section_draft.md` with two complete tables — none of which appears in the paper. Readers have no way to verify the reported numbers without the underlying tables.
**Severity:** High — quantitative claims (e.g., "E88 reaches 0.367", "GDN 0.380", "E88's overall reasoning accuracy (0.319) is within one standard error of M²RNN-CMA (0.336)") are better supported by a visible table than prose alone, and a table already exists in the supporting data.

---

### ISSUE-09 — Six-task canonical sweep: no figure or table
**Section:** §6 "The six-task canonical sweep"
**Problem:** §6 reports results for six tasks (parity, modular counter, FSM tracking, Dyck-1, associative recall, selective copy) and length-extrapolation results entirely in prose with many specific accuracy numbers. No table or figure is provided. The data appears to reside in `paper/results/figure_4_hybrid/` (per-seed JSON files cover FSM tracking and modular counter at least), and `paper/figures/plot_hybrid_degradation.py` already generates Figure F4.
**Severity:** Moderate — prose-only reporting of a multi-task multi-architecture sweep is harder to read and verify than a table.

---

## 3. Orphan Files (exist but not referenced in main.typ)

| File / Directory | Description | Disposition |
|-----------------|-------------|-------------|
| `figures/figure_2_placeholder.svg` | Old placeholder for Figure 2, superseded by `results/figure_2/figure_2_draft.png` | Can be deleted; no paper reference |
| `results/figure_3/combined.csv` | Raw data file; purpose unclear from filename alone; no plot exists; not referenced | Needs labeling or deletion |
| `results/cmaes_burst_v2/cmaes_burst_v2.png/pdf` | CMA-ES burst diagnostic figure; FIGURE_NOTE.md explicitly calls it "a prototype supplement diagnostic, not an integrated main-paper figure" | Supplement candidate; not in main text |
| `results/cma_flop_rate/convergence.png/pdf` | FLOPs-per-bit convergence figure (4-architecture comparison at 480M); README describes finding supporting "architectural option" framing | Potentially relevant; not referenced in paper |
| `results/qa_reasoning/` (data) | QA and reasoning panel data; `section_draft.md` has full tables | Referenced in §6 prose but not visualized |

### Notes on orphans:
- **`results/cma_flop_rate/`**: The README describes a finding directly relevant to §5's "supporting comparison" claim (that E88 lands in the same loss-vs-wallclock band as GDN). The FLOPs-per-bit convergence figure could strengthen §5 or serve as an appendix figure. Its absence is not an error but is a missed presentation opportunity.
- **`results/figure_3/`**: The directory name implies it was intended as Figure 3 at some point; the lone `combined.csv` without a corresponding plot or README suggests an abandoned figure pipeline. Recommend either adding a plot and caption or renaming/deleting the directory.

---

## 4. Caption Quality Assessment

| Figure | Caption informative? | Claims supported? | Length | Issues |
|--------|---------------------|------------------|--------|--------|
| `<fig_arch>` | ✓ Clear two-panel description with mathematical notation | ✓ Architecture diagram matches §3 equations | Appropriate (~12 lines) | None |
| `<tab_ablation>` | ✓ Explains ablation logic concisely | ✓ Directly supports §3 elimination argument | Appropriate | None |
| `<fig_lm_racers>` | ✓ but overloaded | ✓ Endpoint values stated | **Too long** (~30 lines) | See ISSUE-04, ISSUE-05 |
| `<tab_s5>` | ✓ Clear column descriptions | ✓ Numbers match §6 text | Appropriate | Informal source citation (ISSUE-06) |
| `<fig_s5_bars>` | ✓ SEM noted, individual seeds noted | ✓ Matches `<tab_s5>` values | Appropriate | Informal source citation (ISSUE-06) |
| `<fig_hybrid>` | ✓ Dashed random-baseline line explained; seed points noted | ✓ Per-seed data visible | Appropriate | No body cross-reference (ISSUE-03) |

---

## 5. Readability Assessment

| Figure | Readable? | Notes |
|--------|-----------|-------|
| `<fig_arch>` | Likely — rendered in Typst | Blue grid cells and flow boxes; actual readability depends on PDF rendering; color is purely decorative (blue = generic). No axis labels needed (schematic). |
| `<tab_ablation>` | ✓ | 3-row, 5-column table; clear; bolded header |
| `<fig_lm_racers>` | ✓ | `figure_2_draft.png` is a real plot with labeled curves; 95% width; likely readable at standard page width |
| `<tab_s5>` | ✓ | 5-row, 5-column; aligned right for numbers; clear header |
| `<fig_s5_bars>` | ✓ | Bar chart with error bars and seed overlays; 95% width |
| `<fig_hybrid>` | ✓ | Per-seed bars with SEM; dashed random baseline; 95% width |

---

## 6. Summary of Issues by Priority

| Priority | Issue | Figure(s) | Action |
|----------|-------|-----------|--------|
| High | ISSUE-04: Caption far too long | `<fig_lm_racers>` | Move methods text to §5 body; reduce caption to ≤5 sentences |
| High | ISSUE-05: "Draft" filename; data from active training run | `<fig_lm_racers>` | Rename file on finalization; update caption "training continues" wording before submission |
| High | ISSUE-08: QA/reasoning panel has no table despite data existing | §6 prose | Add table from `results/qa_reasoning/section_draft.md` |
| Moderate | ISSUE-01: Missing `@fig_arch` cross-reference | `<fig_arch>` | Add `@fig_arch` to §3 prose |
| Moderate | ISSUE-02: Missing `@tab_ablation` cross-reference | `<tab_ablation>` | Add `@tab_ablation` to §3 prose |
| Moderate | ISSUE-03: Missing `@fig_hybrid` cross-reference | `<fig_hybrid>` | Add `@fig_hybrid` to §6 and §9 prose |
| Moderate | ISSUE-06: Source cited as internal dev notes | `<tab_s5>`, `<fig_s5_bars>` | Update source citation to CSV/script path before submission |
| Moderate | ISSUE-07: §5 setup table has no label/caption | U1 | Wrap in `#figure()` with caption and label |
| Moderate | ISSUE-09: Six-task sweep reported in prose only | §6 | Add table for the six-task sweep |
| Low | Orphan: `figures/figure_2_placeholder.svg` | — | Delete or document purpose |
| Low | Orphan: `results/figure_3/combined.csv` | — | Add plot or delete/rename directory |

---

## 7. Validation Checklist

- [x] Inventory of all figures/tables with reference + caption check (§1, §2)
- [x] Each issue notes figure id and the problem (§2, §6)
- [x] Findings written to `paper/review/figures.md`
