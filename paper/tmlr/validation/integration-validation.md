# Integrated paper validation, 2026-10-15

## Inputs and scope

- Main: `baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e` (the merged CI repair).
- Author paper: `5b16e273518f7ab20463e5a4826bd78851c4f6bf`.
- Approved S5 clarification: `7bc278704d56f2f8f6a0d1ca7ec944e0eaaeb0af`.
- The author branch and S5 change merged cleanly. The canonical manuscript,
  bibliography, and S5 figure source exactly match the approved S5 commit.
- All six empirical table bodies and all seven converted table structures are
  preserved. Data, results, formal sources, production code, and CI are unchanged.
- The separate Frontier-planning branch was not merged: the author manuscript
  already incorporates that transfer, and the old plan's raw-log recovery
  proposal was superseded by the author's no-raw-logs correction.

## Packaging

The canonical PDF has 33 pages; identified and anonymous TMLR PDFs each have
29 pages. All are rebuilt from the same Typst source, including the S5
permutation-matrix construction, the necessary-condition status of the Lean
bridge, and the explicit production-realization question G3.

The only layout-logic change allocates one-third of the notation table to
symbols and two-thirds to definitions, rather than the inherited inverse
allocation. This prevents a mostly blank page introduced when the expanded
introduction pushed the unnecessarily tall table onto a new page. Content,
font sizes, empirical tables, and official TMLR style files are unchanged.
A new regression checks the allocation.

The source hash and symbol counts were repinned for the approved S5 revision:
335 expressions, 285 attachments, eight PDF state primes. This adds ten
expressions and nineteen attachments to the author's P20 inventory, with no
prime-count change. The independent complete-expression oracle is unchanged.

P21's source-specific provenance replay remains historical. Its inherited
release checksum manifest is preserved as `p21-release-SHA256SUMS`. The active
`SHA256SUMS` covers current deliverables. The old accepted PDF was an external,
untracked artifact and is not available in this checkout; its historical digest
is retained, not asserted as a freshly checked file. Current source/style guards
remain fail-closed.

## Checks

- Typst 0.14.2 canonical build: passed.
- Assembly audit: 13 numbered body sections, 33 pages, one register appendix,
  17-file Lean import closure, and all 61 bibliography entries checked.
- TeX Live 2025/pdfTeX TMLR builds: passed with no missing glyphs, unresolved
  citations/references, overfull boxes, oversized floats, or errors. Ordinary
  underfull-spacing warnings remain.
- Packaging regression tests: 67 passed, zero failed/skipped.
- Drift checks: 20 load-bearing samples across all three PDFs; 1,143 long source
  fragments; all 335 complete expressions and 285 attachments; eight primes in
  every PDF. Bibliography bytes are identical.
- Anonymous-PDF check: anonymous text and author metadata, no XMP or embedded
  files, 11 approved literature/documentation links (seven unique), and all 26
  fonts embedded. Identified preprint retains the author and release links.
- Integration audit: exact source commits, six unchanged empirical table
  bodies, seven table structures, unchanged evidence/formal/code/CI.
- Source bundle: fresh extraction rebuilds to 29 pages, with raw PDF text
  byte-identical to the identified preprint.
- Visual inspection: canonical pages and both TMLR variants checked for table,
  figure, equation, heading, and bibliography layout; notation allocation
  corrected and reinspected. No clipped or missing content observed.
- All active artifact hashes verified; whitespace check passed.

Repository CI and post-merge CI are separate checks recorded by GitHub on the
integration PR and eventual merge commit.
