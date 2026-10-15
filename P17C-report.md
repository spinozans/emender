# P17C — register subsection-level conversion repair

## Scope and delivery

Base: **4106fb41**, branch `paper-1p3b-spine`. Authority: scratch
`taskP17C-repair.md` and `P17-review.md`; **C5 is the sole condition**.
Delivery: one commit named **P17C: register subsection-level conversion repair**,
pushed to `origin/paper-1p3b-spine`. The final handoff records the actual commit
and remote equality; the push receipt is retained outside the worktree at
`/home/erikg/emender-scratch/fix-fanout/P17C-push.txt`.

This is conversion/packaging only. `paper/main.typ`, bibliography, canonical
baseline, imported manuscript sources, figure assets, and scientific evidence
are unchanged. Canonical source SHA-256 remains
`9150b69fd7dfb8837a3608e3c5dd963bf4127335f3809fd1b45c652e3618d0b8`.
No training, evaluation, native/Python build, scheduler submission, or WorksGood
machinery was run. Resilient-DiLoCo requirement IDs are not applicable.

## Cause and minimal repair

Typst's evaluated tree serializes markup `==` headings with **depth: 2**, while
explicit `heading(level: ...)` calls carry **level**. The converter read only
`level`, defaulting missing values to 1. It therefore promoted all five register
subsections to top-level appendices, despite the canonical source being correct.
`heading-level.mjs` now resolves `level ?? depth ?? 1`; `convert.mjs` uses it.
Explicit unnumbered level-3 internal headings retain their existing treatment.

An exact base/current generated-TeX comparison passes: **the only five changes
are `\section` → `\subsection` for these entries**. No prose, numeric result,
label, caption, table, mathematical expression, or numbering policy was edited.
Receipt: `paper/tmlr/validation/p17c-generated-delta.txt`.

## Before / after in the built submission PDF

| Register entry | Before at 4106fb41 | Rebuilt |
|---|---|---|
| Development history and source register | A | **A** |
| Predecessor-cohort systems measurements | B | **A.1** |
| Downstream capability panels: a null, on the predecessor cohort | C | **A.2** |
| Lineage of the E63 → E88 → E97 experimental program | D | **A.3** |
| The predecessor 1.3 B cohort (E88 era) | E | **A.4** |
| Held-out bits-per-byte measurement (predecessor cohort) | F | **A.5** |

Before evidence: `validation/p17c-before-structure.txt` (base PDF extraction and
base TeX). After evidence: newly extracted `validation/submission-text.txt` and
`validation/preprint-text.txt`, plus `validation/p17c-register-appendix.json`.
The inventory now records one level-1 register, five level-2 subsections and
unchanged level-3 internal headings, all under the register's section name.

### Evidence-pointer resolution, not merely command counting

`register-appendix.mjs`, called by `verify.mjs`, checks both rebuilt TMLR PDFs:

- One unique Appendix A heading and unique ordered A.1–A.5 headings; generated
  TeX has exactly one register section and five subsections.
- Evidence lies **inside the corresponding subsection span**: A.1 count,
  occupancy/MFU/kernel numbers (22,200; 7,492; 15.7%; 8,248); A.2 downstream
  panels and Table 10 values; A.3 E63/E88/E97 lineage and step-8,400 record;
  A.4 predecessor held-out/train-loss measurements; A.5 Common-Pile bpb values.
- All three main-text **Section A** pointers are present, including Systems'
  occupancy/MFU/kernel pointer and LM's downstream-panel pointer. Their evidence
  is now contained within Appendix A (A.1/A.2), not separate B/C appendices.
- Built labels resolve to A p.48, **A.4 p.50**, **A.5 p.58**, and each destination
  page actually contains the corresponding heading in extracted PDF text.
- Every register reference number/count agrees with the generated TeX:
  **3 × A, 8 × A.4, 7 × A.5 = 18 references per PDF**. No stale B–F reference
  remains. Cohort and held-out references now resolve within the same register.

## Full packaging validation

Executed from the worktree; Poppler used
`LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`,
`PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext`, and
`PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo`; TeX used
`TECTONIC_BIN=/tmp/tectonic`. Versions are in `validation/p17c-tools.txt`.

| Command / check | Result and receipt under `paper/tmlr/validation/` |
|---|---|
| `node paper/tmlr/validation/p16b-audit.mjs` | PASS; all existing preservation/voice dispositions; `p17c-preservation.txt`. |
| `node paper/figures/validate_assembly.js` | PASS; 13 body sections + one appendix and existing scope guards; `assembly-check.txt`. |
| `typst compile --root . paper/main.typ paper/current.pdf` | PASS; **67 pages**; `p17c-canonical-build.txt` (empty successful log). Canonical extracted text is unchanged. |
| `node paper/tmlr/convert.mjs` | PASS; 102 headings, 20 captions, 21 formal environments; byte-identical bibliography; `convert.txt`. |
| `bash paper/tmlr/build.sh` | PASS; submission/preprint **60/60 pages**; no missing glyphs, undefined citations/references, overflows, oversized floats, or build errors; `build-output.txt`, `submission-build.txt`, `preprint-build.txt`. |
| `node --test paper/tmlr/{verify-symbols,notation-audit,author-rulings,appendix-order,register-appendix}.test.mjs` | **31/31 PASS** (seven new regressions); `p17c-tests.txt`. |
| `node paper/tmlr/verify.mjs` | PASS; 20 original samples × three PDFs, 2,635 long source fragments, 718 math expressions / 449 attachments, eight S primes per PDF; source/bibliography guards, anonymity, identity and appendix order; `verify-output.txt`, `drift-check.json`, `p17c-register-appendix.json`. |
| `bash paper/tmlr/bundle-arxiv.sh`; fresh untar; `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; `pdftotext -raw`; `cmp` | PASS; fresh **60-page** archive rebuild, raw text byte-identical to identified preprint; `arxiv-build-output.txt`, `arxiv-build.txt`, `arxiv-compare.txt`, `p17c-arxiv-pdfinfo.txt`. |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS, all 15 existing paths; only rebuilt deliverable hashes updated; `p17c-checksums.txt`. |
| Exact generated-TeX comparison against 4106fb41; canonical-source diff; `git diff --check` | PASS; only five hierarchy commands change in TeX; no manuscript edit; `p17c-generated-delta.txt`, `p17c-final-checks.txt`. |

All page counts are unchanged from base: **67 canonical / 60 submission / 60
preprint / 60 fresh archive build**. References still begin on p.44, end p.47,
and the sole register Appendix A begins p.48 in both TMLR PDFs (canonical:
References p.51, register p.55). Anonymous visible text and PDF author metadata
pass the existing checks; identified preprint retains its author/release links.
The packaging README now documents the corrected hierarchy/checks and current
page counts.

Seven new regressions cover depth/level precedence, the actual generated
inventory, old separate appendices, missing/duplicate subsections, evidence
outside its target span, stale/missing/misnumbered pointers, and wrong built
label numbers/destination pages. Existing symbol-loss and appendix-order
negative tests remain intact. No guard, sample, or baseline pin was relaxed.

## Residual risks / acceptance

The longstanding Typst/Poppler `Suspects object is wrong type (boolean)` diagnostic
and nonfatal TeX underfull-box warnings remain; all fail-closed checks pass.
No new scientific/editorial claim is certified. No fresh visual page inspection
was performed; PDF verification is extraction, built-label and packaging based.
Independent reviewer acceptance remains the next gate. The index is checked empty
before explicit task staging and again after the single commit/push; delivery
receipts are kept in scratch so they require no second commit.
