# TMLR packaging of the integrated manuscript

`paper/main.typ` includes the operator-directed P23 expressivity surgery on
**5cc34e8c**, following the S5 clarification and P22 recorded-loss figure.
`../../P23-report.md` gives the retire/keep map and validation evidence.
The exact manuscript/table delta is `validation/p23-expressivity-revision.diff`;
`canonical-baseline.json` pins the source and preserves the historical lineage.

## Deliverables

- `paper/current.pdf`: canonical Typst PDF, **35 pages**.
- `submission.pdf`: anonymous, **30 pages**, official TMLR submission style.
- `preprint.pdf`: identified, **30 pages**, official TMLR preprint style.
- `arxiv-source.tar.gz`: self-contained identified source with one `main.tex`
  master; fresh extraction rebuilds to **30 pages**, with raw PDF text identical
  to `preprint.pdf`.

Upload `submission.pdf` to OpenReview. The preprint and archive are identified.
Shared source conditionally selects author details and release links.

## Regenerate, build, verify

Run from the repository root with Node.js, Typst, Poppler and Tectonic:

```bash
export TECTONIC_BIN=/tmp/tectonic
export LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu
export PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext
export PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo
typst compile --root . paper/main.typ paper/current.pdf
typst compile --root . --pdf-standard 1.5 --no-pdf-tags --pages 1,2 \
  paper/tmlr/figures.typ /tmp/emender-packaging-figures.pdf
/tmp/p9-poppler/usr/bin/pdfseparate /tmp/emender-packaging-figures.pdf \
  paper/tmlr/assets/figure-%d.pdf
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

`serialize.typ` exports the evaluated content tree. `convert.mjs` translates it
and fails on unsupported nodes. Current inventory: 13 numbered body sections,
66 headings, seven tables including notation, ten captions, five numbered
algorithms and 13 amsthm statements. The scalar-map display is retired; the
new table reads the matched-rate three-seed JSON directly. The S5 table
retains its exact definition. Two vector drawings are freshly exported; the
Frontier and recorded-loss PNGs retain their bytes. These are the four assets.
P23 adds a mathematical plus-minus text mapping, with PDF glyph regression
coverage for the uncertainty statistics.

## Provenance and guards

`paper/results/invertible_quadratic/provenance.json` binds byte-identical
read-only copies of the X-series summaries, task proof, reports, scoped X4R
review, and earlier GDN2 selections to their original paths and SHA-256 hashes.
`expressivity.test.mjs` independently replays paired counts/statistics, seed
ordering, extrapolation CE, the nine primary grid cells, ladder/one-step counts,
and the 1.73% regime observation with its stopping-rule confound.

`integration-audit.mjs` checks the exact P23 delta, unchanged S5 definition,
five unaffected empirical table bodies, six unaffected table structures,
all pre-existing evidence/formal bytes, bibliography, official styles, and
production/CI scope. Historical P13–P22 receipts retain their original scope;
`validation/p21-release-SHA256SUMS` preserves the inherited release manifest.

`verify.mjs` checks 20 samples across all three PDFs and 1,215 long source
fragments. The independent symbol oracle checks all **352 expressions,
293 attachments and eight PDF state primes**. Negative tests retain source
and bibliography drift, complete-expression/prime loss, live-scope loss,
appendix-order and anonymity regressions. The full current suite has **77 tests**.

Appendix A is one compact register with four source categories. Built-PDF
checks cover category-local evidence, label/destination, every appendix pointer,
and the actual final reference on the physical page immediately before A.
Three release/mirror URLs are variant-gated. The anonymous-PDF audit screens
visible text, metadata, XMP, embedded files, URI annotations and font embedding.
The 61-entry bibliography and official style revision
**7bf90efe3a0debbba703c05c43f3ff7e4d4a2992** retain their exact bytes.

The active checksum manifest binds current deliverables and receipts. Builds
fail on errors, missing glyphs, unresolved references/citations, overflow and
oversized floats. Existing underfull TeX spacing warnings and the Typst-PDF
Poppler diagnostic remain nonfatal. Independent reviewer acceptance remains
required; packaging evidence adds no training or mechanism certification.
