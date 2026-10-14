# P9 — TMLR packaging report

## Result

The accepted manuscript at **eb2b96363e3ff800d7eefe2ba7386e7cf673d1cf** is
packaged in the official TMLR LaTeX style, with no scientific edits.
Worktree: `/home/erikg/emender-paper`; branch: `paper-1p3b-spine`.
Delivery remote: `origin` (`git@github.com:spinozans/emender.git`), same branch.
Packaging commit: **13115501ebf85aad3a46938269d41d1ff1e5e292**, pushed successfully;
`git ls-remote origin refs/heads/paper-1p3b-spine` matched the local packaging
commit after push. This delivery receipt is committed separately. The index is
empty; the pre-existing untracked review artifacts are the only unrelated files.

| Deliverable | Absolute path | Pages |
|---|---|---:|
| Anonymous TMLR submission | /home/erikg/emender-paper/paper/tmlr/submission.pdf | 51 |
| Identified arXiv preprint | /home/erikg/emender-paper/paper/tmlr/preprint.pdf | 51 |
| Canonical accepted Typst PDF, unchanged | /home/erikg/emender-paper/paper/Garrison_2026_Emender-eb2b9636.pdf | 62 |

The self-contained arXiv source bundle is
`/home/erikg/emender-paper/paper/tmlr/arxiv-source.tar.gz`. It contains exactly
one master `main.tex`, explicitly selecting preprint mode, and was unpacked
and independently rebuilt into a 51-page identified PDF. Its extracted text
is byte-identical to the delivered preprint's text.

## Conversion / anonymity

- Read the official sample and cloned the official style repository. Vendored
  revision: **7bf90efe3a0debbba703c05c43f3ff7e4d4a2992**. The style, bibliography
  style, fancyhdr dependency, and license are unmodified; sample sources are
  retained under `paper/tmlr/upstream/`. No nested Git repository is required.
- Used Typst's evaluated content tree, not PDF reconstruction or paraphrasing,
  to preserve exact prose, math operands, artifact-derived numbers, and captions.
  Converted all 96 body headings and all 20 figure/table captions; 14 body tables
  are native LaTeX tabularx tables. The two vector drawings reuse their canonical
  Typst bodies as cropped assets; the five PNG figures are unchanged copies.
- The formal statement descriptions use 21 unnumbered amsthm environments
  (13 bullet statements, seven prose statements, one definition). Original
  statement headings and text are retained, with no new theorem numbering or
  assertion text. Equations remain unnumbered. Section/figure/table cross-references
  resolve; appendix references use the LaTeX appendix lettering.
- `refs.bib` is byte-identical to the canonical bibliography. Official TMLR
  natbib author-year citations replace the canonical IEEE numeric presentation.
- Anonymous PDF: official default submission mode, Anonymous authors, Anonymous
  PDF author metadata; six occurrences of three identifying release/mirror URLs
  and two public repository mentions are replaced by anonymized artifact wording.
  Identifying affiliation/email/ORCID are absent. No acknowledgments or funding
  statement exists to redact. Normal third-person literature self-citations remain.
- Preprint: official `preprint` option, original author/email/affiliation and ORCID
  hyperlink, original release/mirror URLs and repository wording.
- `paper/main.typ`, `paper/refs.bib`, `paper/build.sh`, and the original Typst PDF
  were not changed. The packaging source is shared/conditional, **not anonymous
  source**: upload only `submission.pdf` for double-blind review.

## Mandatory 20-sample drift verification

Each sample below matches the accepted Typst PDF **and both LaTeX PDFs**.
Only whitespace, ligatures, math-letter styling, and presentation punctuation
normalize; decimal digits and percentages match exactly. Source and generated
PDF text are retained for independent review in `paper/tmlr/validation/`.

| # | Section / load-bearing sample | Result |
|---:|---|---|
| 1 | **Abstract** — modular-quadratic terminal accuracy is 99.96% versus 76.09% at trained length, and 76.07% versus 53.36% at eight times that length. | PASS × 3 |
| 2 | **Introduction** — at the matched 150.79-billion-token cutoff the smoothed loss gap between the two arms is 0.010 nats/token | PASS × 3 |
| 3 | **Background** — At fixed precision and width, both state regimes are finite-state recognizers. | PASS × 3 |
| 4 | **Architecture** — 1,286,589,072 parameters in total | PASS × 3 |
| 5 | **Systems** — All throughput measurements and throughput-budget smokes reported here are CUDA-only; none establishes correctness, performance, or parity on ROCm. | PASS × 3 |
| 6 | **Language modeling** — exact common logged steps through step 2,300,925 (150,793,420,800 tokens): 92,037 aligned records, zero E97-only and zero GDN2-only records through the cutoff, no interpolation. | PASS × 3 |
| 7 | **Expressivity / contractive control** — gdn-8m has 7.94 M parameters versus E97’s 7.38 M (7.6% more, favoring GDN). | PASS × 3 |
| 8 | **Expressivity / pre-registered budget** — The throughput-derived common cap of 18269 steps was fixed before accuracy was seen (budget.json, 1989656c). | PASS × 3 |
| 9 | **Expressivity / terminal sweep** — At the terminal sweep E97 reaches 99.96% at trained length, retaining 90.16% at four times and 76.07% at eight times that length. | PASS × 3 |
| 10 | **Applications** — A learned surrogate would additionally need stability and safety validation; a state-tracking result is not a control guarantee. | PASS × 3 |
| 11 | **Formal / trust inventory** — 857 named declarations, 723 user-written, 17 files; extraction source commit e58ae937 | PASS × 3 |
| 12 | **Formal / recognizer floor** — Any finite-state recognizer with a state decoder that reproduces the tracker on every input word needs at least 120 states | PASS × 3 |
| 13 | **Formal / latching boundary** — the hold theorem assumes λ ≥ 1, whereas production decay is below one | PASS × 3 |
| 14 | **Related work** — with 87.5% of its blocks linear-state, it sits closer to the linear-state class than to the pure-nonlinear-recurrent one. | PASS × 3 |
| 15 | **Limitations** — The current modular-quadratic comparison shares a cap and plateau rule but realizes 2,500 versus 2,200 steps | PASS × 3 |
| 16 | **Conclusion / negative transfer** — The 8M S5 separation does NOT replicate at 1.3B on this pair. | PASS × 3 |
| 17 | **Testable predictions** — Falsified if per-token throughput degrades faster than the parameter count grows, beyond the kernel-arithmetic baseline. | PASS × 3 |
| 18 | **Future work** — the single-arm E97 513 B-token Frontier continuation that exists cannot answer a paired question | PASS × 3 |
| 19 | **Appendix B / predecessor cohort** — E88 0.973 BPB, GDN 0.973 BPB, and M²RNN-CMA 0.979 BPB. | PASS × 3 |
| 20 | **Appendix C / Common Pile** — Our three models on the same Common-Pile slice measure Emender 0.981, GDN 0.963, M²RNN-CMA 0.973 | PASS × 3 |

Additional whole-prose check: **2,500 canonical text/raw fragments of at least
30 characters**, including tables, captions, and all appendices, match both
PDFs. The two float-interrupted fragments per LaTeX PDF are checked as exact
halves in the original order and recorded explicitly in `drift-check.json`;
the canonical PDF has three analogous reading-order interruptions. This is
an additional prose audit, not a claimed machine proof of mathematical
equivalence or a substitute for the mandatory samples.

## Validation commands / retained evidence

Environment: Ubuntu 24.04 on lambda01 (not Frontier); no Python, Slurm, native
builds, training changes, or resilience testing were performed. Resilient-DiLoCo
requirement IDs therefore are not applicable to this packaging-only task.
No WorksGood service or additional worker was started.

1. `node paper/tmlr/convert.mjs` — evaluated canonical source; strict supported-node
   conversion, 96 headings / 20 figures and tables / 21 amsthm descriptions.
2. `typst compile --root ../.. --pdf-standard 1.5 --no-pdf-tags --pages 1,2 figures.typ /tmp/p9-figures.pdf`,
   then `pdfseparate` — canonical vector drawing assets only, not the Typst manuscript build.
3. `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` — Tectonic 0.17.0, both PDFs
   build with no errors, missing glyphs, unresolved references/citations, overfull
   boxes, or oversized floats; bibliography identity check passes.
4. `PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo node paper/tmlr/verify.mjs`
   (Poppler library path supplied) — 20 samples × 3 PDFs, 2,500 source fragments,
   PDF authors/anonymity guards, and canonical-source guards all PASS.
5. `bash paper/tmlr/bundle-arxiv.sh`; unpacked archive in a fresh directory and
   `/tmp/tectonic --keep-intermediates --keep-logs main.tex` — clean standalone
   build, 51 pages, Erik Garrison author metadata. `cmp` of raw PDF text against
   the delivered preprint passes.
6. `git diff --exit-code eb2b9636 -- paper/main.typ paper/refs.bib paper/build.sh`
   and SHA-256 manifest — canonical source/build unchanged.
7. Final PDF pages visually inspected: title/abstract, claims table, architecture
   diagram, training identities/loss table, CMA sweep table, and formal statements.
   No clipping or missing scientific content observed.

Evidence: `paper/tmlr/validation/drift-check.json`, extracted text for all three
PDFs, final submission/preprint logs, standalone arXiv build log, and
`paper/tmlr/validation/SHA256SUMS`. Conversion inventory and reproducible build,
conversion, verification, and bundle scripts are committed with the PDFs.

## Residual notes / scope boundaries

- Nonfatal underfull spacing warnings remain. BibTeX warns that the canonical
  `mathlib4` entry has no year. No year was invented, and the official style
  was not modified to hide the warning.
- Poppler reports the accepted Typst PDF's pre-existing malformed Suspects
  metadata type; text extraction/page counting succeeds. The new PDFs do not
  have that diagnostic.
- The accepted conclusion's Arkheion/compliance caveat and its existing
  TODO-NEEDS-EVIDENCE scientific qualifications remain verbatim: they were not
  silently revised as part of packaging.
- No OpenReview upload or arXiv submission was performed. Venue review and an
  actual anonymous artifact-hosting decision remain outside this packaging task.
- Existing untracked P4/P6/P7/P8 review artifacts were left untouched and are
  not included in the packaging commit.
