# TMLR packaging of the accepted manuscript

`paper/main.typ` at `eb2b96363e3ff800d7eefe2ba7386e7cf673d1cf` remains the
canonical manuscript. This directory is a formatting/packaging layer, not a
scientific revision. The original Typst build and frozen 62-page accepted PDF
are untouched.
Only the explicit supervisor-approved mathlib4 access-year bibliography amendment
may differ from the accepted canonical bibliography; `canonical-guard.mjs`
checks that exact change and forbids any other canonical source drift.

## Deliverables

- `submission.pdf`: anonymous, **53 pages**, default official TMLR submission style.
- `preprint.pdf`: identified, **53 pages**, official TMLR `preprint` option.
- `arxiv-source.tar.gz`: self-contained identified source archive, with exactly
  one master (`main.tex`), explicitly selecting preprint mode. It was unpacked
  and rebuilt separately; its PDF text is identical to `preprint.pdf`.

For OpenReview double-blind submission upload **submission.pdf**, not
`preprint.pdf` or the source archive. Shared source contains conditionally
excluded author details/URLs; it is not an anonymized-source artifact.

## Build / verify

Install TeX Live with latexmk (pdflatex/BibTeX), or Tectonic; Node.js; Typst;
and Poppler (`pdftotext`, `pdfinfo`, `pdfseparate`). No Python is needed.

```bash
cd paper/tmlr
bash build.sh
node --test verify-symbols.test.mjs
node verify.mjs
bash bundle-arxiv.sh
```

Tectonic can be specified with `TECTONIC_BIN=/path/to/tectonic`.
Verification accepts `PDFTOTEXT` and `PDFINFO` overrides.
The standalone LaTeX builds do not depend on Typst or the repository's data
artifacts; only regeneration and the canonical-content audit need them.

To regenerate the packaging layer from the frozen canonical source:

```bash
typst compile --root ../.. --pdf-standard 1.5 --no-pdf-tags --pages 1,2 \
  figures.typ /tmp/emender-packaging-figures.pdf
pdfseparate /tmp/emender-packaging-figures.pdf assets/figure-%d.pdf
node convert.mjs
bash build.sh
node --test verify-symbols.test.mjs
node verify.mjs
bash bundle-arxiv.sh
```

`serialize.typ` exports Typst's **evaluated content tree**, retaining exact
text/math and evaluating artifact-derived tables/numbers. `convert.mjs` maps
that tree to LaTeX, failing on unsupported nodes. It does not paraphrase prose.
The two original vector diagrams are exported as figure assets, not manuscript
page images. The other five figures are byte-identical copies of the canonical
PNG assets. All 20 captions are independently typeset in LaTeX; all 14 body
tables are actual LaTeX tables. The theorem descriptions are in 21 unnumbered
amsthm environments (13 original bullet statements, seven prose statements,
one definition), without adding theorem labels, numbers, or statement text.

`refs.bib` is byte-identical to the canonical bibliography; official TMLR
natbib author-year citations and `tmlr.bst` replace IEEE numeric formatting.
Six occurrences of three identifying release/mirror URLs and two public
repository mentions are gated to "the artifact repository (anonymized)" or
"artifact repository (anonymized)" in submission mode. No acknowledgments or
funding statement exists in the accepted body. Normal third-person scholarly
self-citations are retained, not marked as the anonymous author's work.

## Provenance / evidence

The official https://github.com/JmlrOrg/tmlr-style-file repository was cloned
and vendored at **7bf90efe3a0debbba703c05c43f3ff7e4d4a2992**. `tmlr.sty`,
`tmlr.bst`, `fancyhdr.sty`, and `LICENSE` are unchanged upstream files;
`upstream/` retains its sample sources. There is no nested Git repository or
submodule dependency. `STYLE-COMMIT` records the revision.

`validation/drift-check.json` records 20 load-bearing samples across the paper,
checked in the accepted PDF and both new PDFs, plus a 2,500-fragment all-prose
audit. Decimal digits/percentages in samples match exactly (only whitespace,
ligatures, mathematical letter styling, and punctuation formatting normalize).
The independent symbol oracle also checks 643 complete LaTeX math expressions,
including all 417 converted attachments and all three primes in each PDF.
It preserves operators and script boundaries instead of using prose normalization.
The 28 attachments in the two imported vector drawings are not converted; those
assets remain hash-identical. Regression tests reject the actual P10 TeX and
removal of any one of the three primes, missing attachment features, and PDF
prime loss. Explicit caption/table font-size overrides are forbidden.
PDF text, final engine logs, archive-build logs, and SHA-256 digests are retained
in `validation/`; the sample-by-sample baseline report is `../../P9-report.md`
and the repair validation/push receipt is `../../P11-report.md`.

Builds have no errors, missing glyphs, undefined citations/references,
overfull boxes, or oversized floats. Nonfatal underfull-box spacing warnings
remain. The supervisor-approved canonical `mathlib4` entry uses 2026 explicitly as
an access year, not a publication year, with that basis labeled in its note;
the bibliography copies remain byte-identical and no empty-year warning remains.
The canonical conclusion's Arkheion/compliance caveat and existing TODO-NEEDS-EVIDENCE text
are preserved verbatim, not silently revised during packaging.
