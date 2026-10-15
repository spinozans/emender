# P22a — E97 recorded-training-loss figure

## Scope and source binding

Base: `9c0810a1414fe558b71014d13753a25e4d3d6a7a`, branch `paper-1p3b-spine`.
The manuscript change is exactly the insertion in
`p22a-lossfigure-revision.diff`: one figure with a forward-voiced caption and
two introductory sentences in Extended training on Frontier. The matched-pair
claims, all six empirical table bodies and all seven table structures retain
their bytes/evaluated content. S5 source, bibliography, scientific data,
formal sources, production code and CI remain unchanged. Resilient-DiLoCo
requirement IDs are not applicable: this task plots retained measurements
and changes publication artifacts only; it runs no training or scheduler jobs.

`paper/figures/plot_recorded_loss.py` reads the three named public-export
tails through their `/tmp/e97-150b-logs/SOURCES.sha256` receipt, enumerates
only checkpoint names in the three specified workstation directories, and
uses the manuscript's exact Frontier accepted-token anchors and GDN2 retained
checkpoint reference. It never opens a checkpoint payload. The committed
`paper/figures/e97_recorded_loss.json` retains every plotted value, checkpoint
path, source digest, token conversion and metric identity.

The actual raw windows contain **869 / 870 / 838** records, respectively:

| Export | Optimizer steps | Accepted tokens (B) |
|---|---|---|
| tail_1282500.txt | 1,260,950–1,282,650 | 82.6376192–84.0597504 |
| tail_1525000.txt | 1,503,550–1,525,275 | 98.5366528–99.9604224 |
| tail_2300930.txt | 2,280,000–2,300,925 | 149.42208–150.7934208 |

All 2,577 raw records appear unchanged, at 65,536 aggregate tokens/step.
Upper panels expand each window; the lower panel spans the full record.
Only raw points within the same window are connected. The 28 workstation
checkpoint anchors and three post-seed Frontier anchors are marker-only;
the seed is the common workstation/Frontier boundary at 150,793,748,480 tokens.
The orange star marks GDN2's single retained checkpoint at 152,280,498,176
tokens, filename/last-window loss 2.4034. The Pile/CommaPile boundary and
Frontier background are marked. The caption directs matched-token comparison
to the unchanged table, separately from this single-checkpoint reference.

## Validation

Environment: lambda01 (not Frontier), system Python 3.12.3 / Matplotlib 3.10.8,
Typst 0.14.2, Node v24.21.0, `/tmp/tectonic`, Poppler under `/tmp/p9-poppler`.
The Frontier activation script is specific to Frontier; no Frontier Python,
native build, submission preflight or training behavior was exercised here.

Full successful chain: `p22a-chain.txt`, with
`TECTONIC_BIN=/tmp/tectonic`,
`LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`,
`PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext`, and
`PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo`.

- Figure generation and four Python unittest checks: PASS. Tests pin source
  digests, raw record counts and cadence, exact clocks and boundaries, visible
  y range, filename-only checkpoint parsing, rejection of source corruption
  and logged gaps, and all six raw plotting calls (three windows twice).
- Canonical compile and vector-asset export: PASS.
- Assembly audit: PASS, 13 numbered sections, one register appendix, 61
  bibliography entries, 17-file Lean closure.
- Convert → build → verify: PASS; **34 / 29 / 29 pages** for canonical,
  anonymous submission and identified preprint. Ten captions, seven tables,
  13 statements; 20 samples across all three PDFs, 1,155 long source fragments,
  335 expressions, 285 attachments, eight state primes per PDF.
- All **67 packaging tests PASS**. The figure asset, label, caption contents,
  forward voice and matched-table reference extend the conversion regression.
  One-byte source/bibliography rejection and symbol-loss tests remain active.
- Integration audit: PASS, exact approved S5 source plus pinned P22a insertion;
  all table bodies/structures, evidence, formal sources and production/CI retained.
- Anonymous-PDF audit: PASS, 26 embedded fonts and approved URI inventory.
- Bundle and fresh extraction/Tectonic rebuild: PASS, **29 pages**, raw PDF
  text byte-identical to identified preprint; `p22a-archive.txt` and
  `p22a-arxiv-build.txt` retain the receipts.
- Visual inspection: direct PNG, canonical p.15 and submission p.13 inspected.
  Dense windows, individual anchors, GDN2 star/annotation, corpus rule, legend,
  axes and caption are visible and unclipped.
- Active `SHA256SUMS` refreshed; checksum and whitespace checks pass.

Development-only test corrections: the first fixture assumed equal window
lengths instead of their actual source counts; the first caption assertion
assumed the evaluated plain-text inventory preserved whitespace. Both tests
now use source-accurate values/normalization; the final full chain is green.

## Remaining scope

Independent reviewer acceptance remains required. Figure regeneration reads
external raw tails and local checkpoint directory names at the documented
paths; the committed JSON retains the complete plotted measurements and
source hashes for review. Existing nonfatal underfull TeX spacing warnings
and Typst-PDF Poppler's `Suspects object is wrong type (boolean)` diagnostic
remain. Publication artifacts provide recorded training loss rather than
held-out evaluation or an additional matched-pair trajectory.
