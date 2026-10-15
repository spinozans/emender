# TMLR packaging of the transferred manuscript

`paper/main.typ` at **883131ae** is the P20 canonical manuscript;
`paper/current.pdf` is its rebuilt **33-page** PDF. P21 regenerates this
packaging tree from that exact source. `canonical-baseline.json` retains the
accepted snapshot `eb2b96363e3ff800d7eefe2ba7386e7cf673d1cf`, its frozen PDF,
and the complete P13–P21 lineage. Source and bibliography SHA-256 guards bind
the current transfer. The four P20 bibliography additions and the earlier
Mathlib access-year amendment are retained byte-for-byte.

## Deliverables

- `submission.pdf`: anonymous, **28 pages**, official TMLR submission style.
- `preprint.pdf`: identified, **28 pages**, official TMLR preprint style.
- `arxiv-source.tar.gz`: self-contained identified source, one `main.tex`
  master, explicitly selecting preprint mode. A fresh extraction rebuilds to
  **28 pages** with PDF raw text byte-identical to `preprint.pdf`.

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
node paper/tmlr/validation/p21-audit.mjs
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
`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `LICENSE`, and the original accepted
PDF retain their frozen hashes.

`../../P21-report.md` documents every regenerated baseline and its reason.
`validation/p20-transfer-revision.diff` enumerates the approved source delta;
`validation/p21-guard-delta.json` retains full old/new guard definitions;
`validation/p21-inputs.json` binds the transferred inputs. `p21-audit.mjs`
checks exact provenance replay and the narrow packaging scope. Earlier P13–P18
audits remain historical receipts for their own source revisions.

`validation/drift-check.json` checks 20 load-bearing current-manuscript samples
across all three PDFs, including Frontier findings and current-pair scopes,
and 1,099 long source fragments. The independent symbol oracle checks every
complete expression and attachment: **325 expressions, 266 attachments, eight
PDF state primes** (two in drawings). All old/new expression multiplicities are
enumerated in `validation/p21-math-delta.json`. Negative regressions retain
prime/attachment loss and actual P10 conversion-defect coverage and add all
current live-scope and source-register loss tests. The source remains exactly
P20's forward-narrated text; packaging reproduces its tone.

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
`validation/`. Independent acceptance review is the delivery gate.
