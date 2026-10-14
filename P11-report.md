# P11 — packaging repair report

## Result and delivery

Implemented P10 findings **B1, B2, and C1** in `/home/erikg/emender-paper`
on `paper-1p3b-spine`. This is packaging repair, not a scientific rewrite.
Independent acceptance review is still required; no venue upload was performed.

- Separate canonical bibliography ruling commit: **77103b1e**
  (`paper: record supervisor-approved mathlib4 access-year convention`).
- Converter/table/regeneration commit: **bd24fc6122a20e97e5b056a3a84feb9c8383cfc2**
  (`paper: preserve math attachments and restore official TMLR font sizes`).
- Both commits pushed to `origin` (`git@github.com:spinozans/emender.git`),
  branch `paper-1p3b-spine`. After push, `git ls-remote origin
  refs/heads/paper-1p3b-spine` exactly matched `bd24fc6122a20e97e5b056a3a84feb9c8383cfc2`.
  This report is committed/pushed separately after that verified repair receipt.
- No staged files remain. Pre-existing untracked P4/P6/P7/P8/P10 review files
  are untouched and excluded from all repair commits.

| Deliverable | Path relative to worktree | Pages |
|---|---|---:|
| Anonymous TMLR submission | `paper/tmlr/submission.pdf` | 53 |
| Identified preprint | `paper/tmlr/preprint.pdf` | 53 |
| Independently rebuilt arXiv master | scratch `P11-validation/arxiv/main.pdf` | 53 |
| Frozen accepted canonical reference PDF | `paper/Garrison_2026_Emender-eb2b9636.pdf` | 62 |

The regenerated `paper/tmlr/arxiv-source.tar.gz` builds standalone; its raw
PDF text is byte-identical to the delivered preprint. The frozen accepted
canonical PDF, `paper/main.typ`, `paper/build.sh`, all seven imported assets,
and all vendored style files remain unchanged.

## B1 — primes and attachment audit

`convert.mjs` now renders `attach.tr` primes before the subscript, producing
`{S }'` in both state updates and `{S }'_{i , j }` in the saturation hypothesis.
The regenerated submission visibly retains the application prime on p.24 and
the latching/saturation primes on p.31. All three PDFs extract **three S′**.

The complete evaluated AST attachment sweep found **445 attachments**:

| Attachment fields (excluding `func`) | Count | Disposition |
|---|---:|---|
| `base,b` | 364 | Existing subscript conversion retained |
| `base,t` | 63 | Existing superscript conversion retained |
| `base,b,t` | 14 | Both scripts retained |
| `base,tr` | 2 | Prime repaired |
| `base` | 1 | Base retained |
| `base,b,tr` | 1 | Prime plus subscript repaired |

No other attachment fields/variants occur; `tr` primes were the only previously
dropped attachment feature. The converter now rejects unknown attachment
fields and unsupported top-right combinations instead of silently dropping them.

Added an **independent** `verify-symbols.mjs` oracle (no converter imports).
It checks the exact multiset of **643 complete converted math expressions**,
preserving operators, prime tokens, and script boundaries, with whitespace
normalization only. This covers **417 converted attachments**. The other
33 expressions/28 attachments are inside imported vector drawing bodies,
not converted LaTeX; their asset hashes are unchanged. Figure captions remain
in the math audit. The full census is retained separately.

Added **12 passing tests** in `verify-symbols.test.mjs`: all six observed
attachment shapes plus repeated primes; omitted features; unknown-field
fail-closed behavior; PDF prime loss independently of TeX; full current-source
math matching; individual deletion of each actual prime; and rejection of the
actual defective P10 generated TeX fetched from commit `13115501`.
`verify.mjs` now runs this symbol audit alongside the original full drift audit.
This is notation-preservation evidence, not a claim of machine-proved scientific
or mathematical equivalence of every PDF glyph.

## B2 — official sizing restored

Removed `captionsetup{font=small}` and all **14** table
`fontsize{8}{10}/selectfont` overrides from the generator and generated TeX.
No replacement font-size commands, margin changes, or vendored style edits.
The verifier now fails on explicit caption/body font-size overrides.
Both PDFs naturally reflow from 51 to **53 pages**. Build checks find no overfull
boxes, oversized floats, missing glyphs, unresolved references/citations, or
errors. Visually inspected submission pp.15/17/23/24/31/53: reflowed tables,
all repaired primes, and the access-year bibliography note are readable and
unclipped. Underfull spacing warnings remain nonfatal.

## C1 — supervisor access-year ruling receipted

Applied the supplied ruling to **canonical `paper/refs.bib`**, identically to
`paper/tmlr/refs.bib`: `mathlib4` now has `year = {2026}` and retains its
access date `2026-10-14`. Its note explicitly says **“citation year is the
access year, not a publication year.”** This is an access-year convention,
not an invented publication date. Both BibTeX logs have no empty-year warning;
`submission.bbl` has `The Mathlib Community}(2026)`, and the PDF citation is
**(The Mathlib Community, 2026)**. The explicit explanatory note renders in the
bibliography on p.53. The copies compare byte-identical.

`canonical-guard.mjs` checks the original canonical manuscript/build against
`eb2b9636` and permits only this exact bibliography amendment; any other
bibliography drift fails. The frozen canonical reference PDF is intentionally
retained as the accepted-content baseline, not overwritten by the access-year
metadata repair. README and drift JSON describe this exception honestly.

## Validation commands and evidence

Executed on lambda01, not Frontier. No Python, native build, Slurm, training,
WorksGood service, or additional worker was used. Resilient-DiLoCo requirement
IDs are **not applicable** to this packaging-only change.

1. `node paper/tmlr/convert.mjs` — PASS: 96 headings, 20 figure/table captions,
   21 amsthm statements; bibliography copied byte-for-byte.
2. `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` — PASS: both 53-page PDFs;
   clean mandatory layout/glyph/reference checks.
3. `node --test paper/tmlr/verify-symbols.test.mjs` — PASS: 12/12 tests.
4. With `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`,
   `PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext
   PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo node paper/tmlr/verify.mjs` — PASS:
   20 samples × 3 PDFs, exact numeric strings, 2,500 long source fragments,
   canonical/amendment guards, identity guards, bibliography identity,
   643 complete math expressions, 417 attachments, three primes per PDF.
5. `bash paper/tmlr/bundle-arxiv.sh`; fresh `tar -xzf` extraction;
   `/tmp/tectonic --keep-intermediates --keep-logs main.tex` — PASS standalone;
   `pdftotext -raw` and `cmp` against delivered preprint text — PASS.
6. Reran the five independent P10 artifact-derived table values (99.14%, 50.53%,
   0.685, 0.877, 0.2146) against all three PDFs — all PASS.
7. Re-extracted anonymous PDF text/metadata/XMP/URI annotations/attachments/fonts:
   all **53 pages** screened for identity-bearing strings; Anonymous authors and
   Anonymous PDF author; no identity URLs/home paths/affiliation/email/ORCID;
   no XMP or embedded files; all fonts embedded; US Letter. All **10** URI
   destinations exactly match P10's ordinary literature/documentation links.
   Third-person literature self-citation remains unchanged and disclosed.
8. `cmp paper/refs.bib paper/tmlr/refs.bib`, no-font-override grep,
   canonical source/build/asset/style Git comparisons, official style clone
   revision and byte comparisons — PASS. Vendored official commit remains
   `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`.
9. `sha256sum -c validation/SHA256SUMS` in `paper/tmlr` — all 13 entries PASS;
   updated only changed deliverable/bibliography digests. `git diff --check` — PASS.
10. Explicit-file Git commits, push, remote-hash equality, empty-index check — PASS.

Retained tracked evidence: `paper/tmlr/validation/` contains refreshed raw text,
full engine logs, standalone build logs, drift JSON, 12-test output, symbol
verification output, full attachment census, anonymity/citation screening
summary, and SHA-256 manifest. Detailed independent metadata/URI/font extraction,
rendered pages, source AST, standalone build, and five-number replay are in
`/home/erikg/emender-scratch/fix-fanout/P11-validation/`; build/test/verifier
stdout is also retained under scratch as `P11-*-output.txt`.

Diagnostic attempts: the initial whole-AST equation check correctly detected
that vector bodies are imported rather than typeset; the final oracle explicitly
excludes those bodies and guards their unchanged assets. An initial style `cmp`
used the sample-source directory (which lacks style files); rerunning against
the retained official clone `/tmp/p9-tmlr-style` passes. Neither diagnostic
failure was bypassed silently.

## Residual risks and next step

Nonfatal underfull-box warnings remain. The frozen Typst PDF retains its
pre-existing Poppler “Suspects object is wrong type” diagnostic; extraction and
page counting succeed. Shared preprint/arXiv source intentionally identifies
the author and is **not** an anonymous-source artifact: upload only the anonymous
submission PDF for double-blind review. No scientific qualifications, provenance,
training results, or conclusion caveats were rewritten.

Next step: independent packaging acceptance review of the repair commits and
this report before submission. No remaining B1/B2/C1 issue found in self-checks.
