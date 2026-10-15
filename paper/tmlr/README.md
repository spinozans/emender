# TMLR packaging of the integrated manuscript

`paper/main.typ` and `paper/figures/expressivity_1p3b.typ` exactly match the
approved S5 clarification **7bc278704d56f2f8f6a0d1ca7ec944e0eaaeb0af**, based on
the author revision **5b16e273518f7ab20463e5a4826bd78851c4f6bf**. The author's
no-raw-logs correction is retained. `paper/current.pdf` is the rebuilt
**33-page** canonical PDF; the two TMLR variants and source bundle are generated
from that same source. `canonical-baseline.json` retains the historical accepted
snapshot and revision lineage while pinning the current approved source hash.

## Deliverables

- `submission.pdf`: anonymous, **29 pages**, official TMLR submission style.
- `preprint.pdf`: identified, **29 pages**, official TMLR preprint style.
- `arxiv-source.tar.gz`: self-contained identified source, one `main.tex`
  master, explicitly selecting preprint mode. A fresh extraction rebuilds to
  **29 pages** with PDF raw text byte-identical to `preprint.pdf`.

Upload **submission.pdf** to OpenReview. The preprint and source archive are
identified artifacts. Shared source conditionally selects author details and
release links for the appropriate variant.

## Regenerate, build, verify

Dependencies: Node.js, Typst, Poppler, and Tectonic or TeX Live with latexmk.
Tectonic and Poppler paths accept `TECTONIC_BIN`, `PDFTOTEXT`, and `PDFINFO`.
Run from the repository root:

```bash
typst compile --root . paper/main.typ paper/current.pdf
typst compile --root . --pdf-standard 1.5 --no-pdf-tags --pages 1,2 \
  paper/tmlr/figures.typ /tmp/emender-packaging-figures.pdf
pdfseparate /tmp/emender-packaging-figures.pdf paper/tmlr/assets/figure-%d.pdf
node paper/figures/validate_assembly.js
node paper/tmlr/convert.mjs
bash paper/tmlr/build.sh
node paper/tmlr/verify.mjs
node --test paper/tmlr/*.test.mjs
node paper/tmlr/integration-audit.mjs
node paper/tmlr/validation/p21-anonymity.mjs
bash paper/tmlr/bundle-arxiv.sh
(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)
```

`serialize.typ` exports the evaluated content tree, including dynamic tables
and math. `convert.mjs` translates it directly and fails on unsupported nodes.
The structure is 13 numbered body sections, 63 converted headings, seven LaTeX
tables (including notation), nine independently typeset captions, five numbered
pseudocode blocks, and 13 unnumbered amsthm statements. Formal blocks follow
the current subsection names and stable section labels. The two vector drawings
are freshly exported; the Frontier PNG is a byte-identical copy. The standalone
asset tree contains exactly these three figures.

Appendix A is the compact development/source register: one heading, four source
categories. `verify.mjs` checks evidence inside each category, the current-pair
scope, built label A and its physical PDF destination, and every rendered
appendix pointer. Ruling 6 checks the References heading and the actual final
bibliography entry on the physical page immediately preceding Appendix A in
all three PDFs. The current receipts retain the established paths
`validation/p16b-appendix-order.json` and `validation/p17c-register-appendix.json`.

## Provenance and guards

Official style files retain upstream revision
**7bf90efe3a0debbba703c05c43f3ff7e4d4a2992**, recorded in `STYLE-COMMIT`.
`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `LICENSE`, retain their frozen hashes. The historical accepted PDF is an external,
untracked artifact; its recorded digest remains in the baseline/history, while
the active checksum manifest covers the current repository deliverables.

`../../P21-report.md` documents every regenerated baseline and its reason.
`validation/p20-transfer-revision.diff` enumerates the approved source delta;
`validation/p21-guard-delta.json` retains full old/new guard definitions;
`validation/p21-inputs.json` binds the transferred inputs. `p21-audit.mjs`
is the historical P21-only replay, not a validator for later source revisions.
`integration-audit.mjs` checks the exact approved S5 source, all six empirical
table bodies, all seven table structures, unchanged data/results/formal sources,
and unchanged production code and CI. Earlier P13–P21 receipts remain evidence
for their own revisions; `validation/p21-release-SHA256SUMS` preserves the
inherited release manifest.

`validation/drift-check.json` checks 20 load-bearing current-manuscript samples
across all three PDFs, including Frontier findings and current-pair scopes,
and 1,143 long source fragments. The independent symbol oracle checks every
complete expression and attachment: **335 expressions, 285 attachments, eight
PDF state primes** (two in drawings). The historical P20 delta remains in `validation/p21-math-delta.json`;
S5 adds ten expressions and nineteen attachments, with no prime-count change,
as recorded in `validation/integration-audit.json`. Negative regressions retain
prime/attachment loss and actual P10 conversion-defect coverage and add all
current live-scope and source-register loss tests. The source remains exactly the approved S5 revision; packaging does not
paraphrase it. The 67 packaging tests retain negative conversion, prime-loss,
source-drift, scope-loss, appendix-order and anonymization checks.

Three release/mirror URLs are variant-gated. The anonymous-PDF receipt screens
visible text, author metadata, XMP, embedded files, annotation destinations and
embedded fonts; scholarly third-person citations are retained. The identified
preprint retains its author and release links. Canonical bibliography bytes are
copied exactly; narrow Unicode mappings preserve Ž and superscript ² in the
official style font. P3's ledger records all 61 entries; the four P20 intake rows
retain their pending independent metadata-review scope.

Builds pass the existing fail-closed error, glyph, reference, citation, overflow
and oversized-float checks. Underfull spacing warnings remain. Receipts, logs,
PDF extractions, bundle rebuild comparison and SHA-256 digests are retained in
`validation/`. The integrated source and regenerated packaging receive independent review
before merge. Local builds use Typst 0.14.2 and TeX Live 2025/pdfTeX; the source
bundle rebuild must match the preprint raw text exactly.
