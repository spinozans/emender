# P23 — expressivity surgery from the scoped X-series findings

## Contract and source authority

Worktree `/home/erikg/emender-paper`, branch `paper-1p3b-spine`, base
**5cc34e8cc771bfa57fba6a4b98218fc9b33b2b09** (remote equality checked).
Authority: `/home/erikg/emender-scratch/fix-fanout/taskP23-expressivity-surgery.md`
and the ACCEPTED X4R review. X3/X4 reports, summaries and earlier GDN2
selected-test receipts were read from `/home/erikg/emender` without modification.
`paper/results/invertible_quadratic/provenance.json` binds all ten imported
files to their original paths and SHA-256 hashes. The copies retain exact
source bytes; they are self-contained paper evidence, not new experiments.

The supervisor confirmed that P20's already-retired contributions list and
claim table must stay retired. Their surviving abstract, introduction,
conclusion and prediction prose is updated instead. No training, evaluation,
scheduler submission, native/Python build, production code, CI, or resilient
behavior changes occur. Resilient-DiLoCo requirement IDs: **not applicable**.

Delivery: one commit, `P23: replace scalar-map headline with scoped invertible-map findings`,
pushed to `origin/paper-1p3b-spine`. The commit hash, push, remote-equality and
clean-index receipts are recorded externally at
`/home/erikg/emender-scratch/fix-fanout/P23-push.txt`, avoiding a self-referential
commit hash inside this report. Independent reviewer acceptance remains required.

## Retire / keep / replace map

| Site or finding | Disposition and reason |
|---|---|
| Old mod-256 headline: 99.96/76.09% trained length and 76.07/53.36% at 8× | **Retired from every active manuscript variant.** The task admits exact parity-summary-plus-seven-driver recovery. The section gives the task-property argument and explicitly leaves the trained models' shortcut use unidentified. |
| Abstract expressivity sentences | **Replaced** by matched-1e-3 three-seed parity, the per-seed extrapolation-CE observation and the retained S5 finding. No architecture-superiority inference. |
| Contributions list | **Already retired by P20, not reintroduced.** The introduction's empirical paragraphs now state paired uncertainty and the task-dependent findings. |
| Claim table / expressivity row | **Already retired by P20, not reintroduced.** Conclusion and section interpretation carry the scoped empirical summary; the existing retired-structure guard remains. |
| Old scalar-map display `<tab_depth_wall>` / `wall-table` | **Retired**, together with its length-transfer caption and obsolete scalar fine-tune Algorithm 4. New `<tab_invertible>` reads the frozen X4 summary and exposes seeds 0/1/2, means/sample SDs and paired differences. |
| Old scalar and unused contractive display definitions | **Removed from the active figure module.** Historical result JSONs, scripts and records remain unchanged for provenance. |
| New map | **Added**: two-coordinate p=61 map, explicit inverse, no state merging, typed initialization, running full-state prediction, 128/512/1024 drivers. |
| Suffix result | **Added with exact observer scope**: initialization plus bounded suffix, at least two omitted drivers, 1/p² = 0.0269%; excludes extra prefix summaries and early positions. Marginal-optimal loss is named as ln(3721) = 8.221748 nats. |
| X4 headline comparison | **Added**: 0.268 ± 0.091% E97 vs 0.227 ± 0.291% GDN2; paired 0.041 ± 0.201 pp, all sample SDs over three seeds. No robust trained-depth separation. Fresh AdamW, matched 1e-3, procedural training, frozen validation-only selection and unequal stop counts are explicit. |
| Per-seed structure / extrapolation CE | **Added**: E97 ahead on seeds 0/1, GDN2 on seed 2 including CE. E97's CE is lower in every seed at 4×/8×; this remains an observation rather than a mechanism. |
| Matched-5e-5 regime | **Added as observational**: GDN2 mean 1.73% and E97 loss freeze, with changed stopping thresholds (1% versus 0.1%) stated beside the comparison. No LR-only causal interpretation. |
| Nominal 8 M grid | **Added with exact scope**: measured 9,900,800/9,887,056 parameters, real GDN2/negative eigenvalues, symmetric three-LR/three-seed grid, both validation-selected at 1e-3. GDN2 ≥ E97 in all nine primary matched test-1× cells, not every metric/post-seven cell. Best-grid and other matched-LR means are both reported. Scratch/pretrained experiment-class confounds are explicit. |
| p13/p31 ladder | **Added**: seed-0 matched-1e-3 scores; p13 GDN2's best 3e-4 score is separately identified, with best-rate scope for both arms. |
| One-step diagnostics | **Added**: 500/512 = 97.66% and 512/512 = 100%; local arithmetic learnability does not certify earlier long-task mastery or novel-input generalization. Robust 128-deep composition remains unlearned under the tested protocols/scales. |
| Marginal-like loss behavior | **Added as behavior plus limited signal**, not an identified internal marginal-solution mechanism: selected 1× CE 8.188895/8.223262, extrapolation near/above the reference, floor-like post-seven accuracy and zero X4 terminal accuracy. |
| Linear-state scope | **Kept/clarified**: every finite-state task has a sufficiently large one-hot linear-state realization; the benchmark concerns learned efficiency at practical state sizes. No impossibility or universal expressivity separation. |
| S5 section, exact linear-state realization and display | **Kept.** S5 table definition is byte-identical to base; current-pair finding, OOM boundary, and small-model companion remain. Only the adjacent obsolete scalar-task contrast and final interpretation are revised. |
| Formal results, language modeling, systems, Frontier/loss figures, bibliography | **Kept.** Existing evidence/formal assets and all five unaffected empirical table bodies retain their bytes/values. |
| Open questions / sampling | **Updated** to three invertible-map fine-tuning seeds, retaining one LM seed and the one-seed current-pair S5 scope. |
| Next tests for the current pair | **Re-derived**: cross rate with common stop/budget controls, track arithmetic and post-suffix depth signal, repeat per-seed CE, condition nonlinearity ablation on successful arithmetic/depth controls, feasible S5 repeats and held-out LM tests. Old map-advantage predictions leave with the old headline. |
| Compact register | **Kept**, with the current-pair category pointing to the new evidence snapshots as well as historical probe records. |

## Packaging and guard updates

The complete source/table delta is `paper/tmlr/validation/p23-expressivity-revision.diff`.
`canonical-baseline.json` preserves its historical lineage and pins the P23
source and exact symbol inventory. Conversion now has 66 headings, seven tables,
ten captions, five numbered algorithms and 13 amsthm statements. The old
four-column/eight-row scalar display becomes the four-column/five-row
per-seed display; all six other table structures are unchanged.

`integration-audit.mjs` now replays the exact P23 delta against 5cc34e8c, checks
all pre-existing evidence/formal bytes, preserves S5's definition and the five
unaffected empirical table bodies, and constrains scope to paper files plus
this report. Historical P21/P22 receipts retain their own original scope.
Live-scope mutation guards replace only scalar-specific boundaries with the
new selection, shortcut-attribution, protocol-confound, grid-scope and
marginal-like-behavior boundaries; all unrelated formal/measured guards stay.
The source and bibliography one-byte drift tests and independent symbol oracle
remain active. Three inherited math-expression occurrences are removed and
20 added (net +17), enumerated in `validation/p23-math-delta.json`;
attachments change from 285 to 293; eight state primes remain.

`expressivity.test.mjs` independently replays the X4 counts, means, sample SDs,
paired differences, seed ordering and extrapolation CE; the earlier 1.73%
mean; exactly nine primary grid cells; ladder and one-step counts; source-byte
hashes; old-headline retirement; map argument scope; and uncertainty glyphs.
The prior depthwall pseudocode test is replaced by the current Algorithm 4
selection/plateau test. The recorded-loss caption regression is corrected to
P22b's existing plain caption (its earlier expectation still described P22a's
superseded caption); figure/source bytes are unchanged.

Development findings fixed before delivery: Unicode ± in ordinary TeX text
rendered as ś in the official font. The converter now emits a mathematical
plus-minus, with all-PDF regression coverage. Visual inspection also caught a
bare line-leading `256.` parsed as a list item and the ungrouped new F subscript
absorbing its arguments; source grouping/wrapping fixes both. The final built
PDFs display the parity proof, F_c(x,y), inverse and uncertainty correctly.

## Validation receipts and results

All paths in this table are under `paper/tmlr/validation/` unless qualified.
Environment: Node v24.21.0, Typst 0.14.2, `/tmp/tectonic`, Poppler under
`/tmp/p9-poppler`. The chain exports `TECTONIC_BIN=/tmp/tectonic`,
`LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`, and `PDFTOTEXT` /
`PDFINFO` from `/tmp/p9-poppler/usr/bin`.

| Check | Result / evidence |
|---|---|
| Canonical pin → Typst compile → vector export → convert → build → verify | **PASS**, `p23-chain.txt`, `p23-build.txt`, `p23-verify.txt`; **35 / 30 / 30 pages** canonical/submission/preprint. Two vector assets freshly exported; both existing PNGs unchanged. |
| PDF content / independent math | **PASS**, 20 samples × three PDFs, **1,215** long source fragments, **352** complete math expressions, **293** attachments, **eight** state primes in each PDF. Exact numeric strings, source/bibliography pin, all-prose audit retained. |
| `node --test paper/tmlr/*.test.mjs` | **77/77 PASS**, `p23-tests.txt`, including numerical receipt replay and negative scope/source/symbol-loss regressions. |
| `node paper/figures/validate_assembly.js` | **PASS**, `p23-assembly.txt`: 13 body sections + A, labels, current boundaries, 17-file Lean closure and all 61 bibliography entries. |
| `node paper/tmlr/integration-audit.mjs` | **PASS**, `p23-integration.txt`, `integration-audit.json`; exact source delta, preservation and scope checks. |
| Anonymous-PDF audit | **PASS**, `p23-chain.txt`, refreshed `p21-anonymity-details.txt`: 30 pages, anonymous visible author/metadata, empty XMP/attachments, 11 approved URI annotations (seven unique), 26 embedded fonts. |
| References immediately before Appendix A | **PASS**, `p16b-appendix-order.json`: canonical last reference p.34 → A p.35; both TMLR variants p.29 → A p.30. Register category, destination and pointer checks pass. |
| Source archive / fresh extraction / Tectonic build / raw-text `cmp` | **PASS**, `p23-archive.txt`, `p23-arxiv-build.txt`; one identified master, four assets, 30 pages, raw text byte-identical to preprint. |
| Visual spot checks | **PASS**, canonical pp.17–18, submission pp.15–17: parity argument, correctly scoped map/inverse, Algorithm 4, paired table/±, confound, ladder, marginal-like paragraph and unchanged S5 display legible and unclipped. Raster probes are in `/tmp/p23-{canonical,submission}-final-*` plus the earlier p.17 S5 probe. This is a spot check, not a full-page visual audit. |
| Active manifest | **PASS**, `SHA256SUMS` refreshed for current deliverables and new evidence/guards/receipts; `p23-checksums.txt`. |
| Whitespace / delivery | `git diff --check`, staged diff check, one commit/push and remote equality retained in external `P23-push.txt`; no staged files after delivery. |

## Residual scope

Independent reviewer acceptance remains required. The empirical statements
inherit the X4R-scoped evidence, not a new training or convergence certificate.
Selected checkpoints are not equal-step comparisons. The cross-rate protocol
confound, scratch/pretrained differences, practical state-size scope and
one-step diagnostic limits remain explicit. Existing nonfatal underfull TeX
spacing warnings and Poppler's Typst-PDF `Suspects object is wrong type (boolean)`
diagnostic remain. The bibliography's pre-existing flagged/pending metadata
review is unchanged. An inherited applications display uses
`F_theta(S_t, u_t)`, whose Typst AST absorbs the argument list into the theta
subscript; it remains unchanged outside this expressivity revision and is
flagged for separate notation review. No new experimental or
architecture-superiority claim is made; no additional experiment is authorized
by this revision.
