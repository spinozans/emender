# P21 — TMLR packaging regenerated for the transferred manuscript

## Contract and delivery

Base **883131ae**, worktree `/home/erikg/emender-paper`, branch
`paper-1p3b-spine`. Authority: `/home/erikg/emender-scratch/fix-fanout/taskP21-packaging.md`
and P20-report.md, “TMLR packaging follow-up needs (P21)”. Delivery is one
commit named **P21: TMLR packaging regenerated for the transferred manuscript**,
pushed to `origin/paper-1p3b-spine`. Commit, remote-equality, and clean-index
receipts are retained outside the commit at
`/home/erikg/emender-scratch/fix-fanout/P21-push.txt`; the handoff names the hash.

The P20 manuscript and scientific inputs retain their exact bytes. Source
SHA-256: **d098a4a63f849aada480b1906ee14c762a44d4225d62c0f2dffb102cc7e10d5b**.
Bibliography SHA-256: **0121d5d77d5ec901ee9cdee22120a71f79da9fe3260dfd02a7d14eceba29cb67**.
`p21-inputs.json` binds seven transferred inputs and eight frozen source/style
assets. The accepted PDF is bound by its established manifest digest; the
other inputs are compared byte-for-byte with their committed P20 blobs.
Training, evaluation, scheduler submissions, Python/native builds, and
resilient-training behavior remain outside this packaging revision.
Resilient-DiLoCo requirement IDs are **not applicable** to this task.

## Deliverables and physical page counts

| Artifact | Previous packaging | Fresh P21 build |
|---|---:|---:|
| `paper/current.pdf` | P20 already 33 | **33** |
| `paper/tmlr/submission.pdf` (anonymous) | 59 | **28** |
| `paper/tmlr/preprint.pdf` (identified) | 59 | **28** |
| Freshly extracted `arxiv-source.tar.gz` → `main.pdf` | 59 | **28** |

The identified archive has exactly one `main.tex` master, official style files,
the byte-identical bibliography and built bbl, two freshly exported vector PDFs,
and the byte-identical Frontier PNG. Five unused predecessor PNG copies are
retired from the standalone asset tree. Fresh archive-build raw PDF text is
byte-identical to the identified preprint (`arxiv-compare.txt`).

## Regenerated-baseline table

All receipt paths below are under `paper/tmlr/validation/` unless qualified.
`p20-transfer-revision.diff` is the exact zero-context source/bibliography diff
from 0325d925 to 883131ae. `p21-guard-delta.json` retains **20 complete old/new
records**, each with its reason. `node paper/tmlr/validation/p21-audit.mjs`
checks this exact replay, immutable inputs, math delta, PDF receipts and narrow
file scope. Thus each guard move is bound to the enumerated transfer, rather
than a permissive source range.

| Baseline / layer | Previous → current | What changed and why |
|---|---|---|
| `paper/figures/validate_assembly.js` | formal before systems → formal after expressivity; still 13 body sections + A | P20 section labels/order and forward-voiced O1/G1–G3/U3.1 scopes; current findings and Frontier measurements replace removed historical/set-D apparatus. Counters, all label targets, overstatement bans, checkpoint hashes and recursive 17-file closure stay enforced. |
| `canonical-baseline.json` | P18 source SHA → exact P20 source and bibliography SHAs | Full P13–P21 lineage and accepted snapshot retained; current symbol pins recorded. |
| `canonical-guard.mjs` | original bibliography + sole access-year amendment → exact approved P20 bibliography | Four P20 additions now belong to the approved source. Exact SHA-256 **and** byte equality with committed P20 bibliography replace the old reconstruction; source SHA and original build-script guard remain. One-byte source/bibliography drift tests fail closed. |
| `convert.mjs` | obsolete Formal Results/set-letter recognition → stable formal section label/current subsection names | Five prose statements and eight bullet statements retain their full current text in 13 amsthm environments. Unsupported content still fails. Narrow Ž/² Unicode mappings preserve bibliography glyphs in the official font. |
| `conversion-inventory.json` | 102 → 63 headings; 20 → 9 captions; 15 → 7 actual tables; 21 → 13 amsthm statements | Direct evaluated P20 content. The older table total includes notation; its 14 captioned tables become six, plus notation = seven. Five algorithms replace eight. Frontier anchor table and communication figure are included. |
| `serialize.typ`, `figures.typ` | retained byte-for-byte | Their evaluated-tree export already supports the current content. Both drawings are freshly exported from P20; new PNG copied exactly. |
| `verify.mjs` sample inventory | 20 old samples → 20 current load-bearing samples | Eight sample sentences remain exact; revoiced/removed samples follow current claims and scopes. The two historical-appendix slots now pin Frontier endpoint loss and measured communication. Complete source-fragment audit and its normalizers retain their existing constraints. |
| `verify-symbols.mjs` / `verify-symbols.test.mjs` | oracle unchanged; exact pins 706/444 → **325/266** | P20 removes/replaces large sections. Every complete current expression and attachment is still independently compared. Six converted primes + two vector primes remain **eight per PDF**; actual P10 defect and every prime-removal regression retained. All expression multiplicity deltas are in `p21-math-delta.json`. |
| `notation-audit.test.mjs` | eight algorithms / removed EXHIBIT taxonomy → five algorithms / current formal prose | Keep returned-state, split-gate, driver-indexing, terminal-weight and vector-prime checks; add operator tone check. |
| `author-rulings.test.mjs` | Limitations / 14 O1 mentions → Open questions / three O1 mentions | Preserve exact production normalization, source location, full-stack G2 binding, tracker-realization target and distinct saturation Theorem F1. Whitespace-only wrapping is normalized. |
| `appendix-order.mjs` + tests | old final DOI marker → actual final current references | Unique physical References/Appendix A headings and actual final entry on the immediately preceding page; original reversed/same-page/missing/spilled/multiple-appendix tests retained. |
| `register-appendix.mjs` + tests | A.1–A.5 benchmark appendix → one compact A with four source categories | Category-local source paths/evidence, current-pair scope, built A label/destination and all rendered appendix pointers remain checked. Historical cohort labels, stale pointers, relocated evidence and scope loss fail. |
| `armor-audit.mjs` + tests | claim-row scopes / removed apparatus → **21 current prose scopes + five caption scopes** | P20 explicitly removes the claim table, predecessor benchmarks and set-D synthetic-cost apparatus. Current seed/budget/OOM/selection boundaries, S5 finding, TC0/NC1, fixed-weight real-valued witness, G1–G3/O1/U3.1, production decay, control stability and Frontier measurement scopes remain pinned. Each scope has its own loss mutation test; retired-framing bans remain. |
| `conversion.test.mjs` | new | Exact section/table/caption/statement/algorithm/figure inventory, copied bibliography/PNG, current Frontier values, three-asset tree, 50 distinct citations and source/bibliography drift regressions. |
| `attachment-census.json` | 472 → **297** full-AST attachments | Recompute whole evaluated AST **including vector drawings**; distinguishes this census from 266 converted attachments. Eight prime nodes retained. |
| `paper/review/P3-reference-audit.md` | 57 → **61** ledger rows | Four P20 intake rows retain canonical locators and pending independent metadata-review scope; original P3 statuses remain intact. |
| `bundle-arxiv.sh` / `README.md` | old shape/delivery text → current shape and affirmative archive text | Describe the compact register, three assets, fresh pages, provenance and complete regeneration commands. Archive README follows the operator tone ruling. |
| `SHA256SUMS` / PDF and extraction receipts | old rebuilt assets → **22** current checked paths | Refresh source/baseline/build outputs, exported drawings, PDFs, archive and primary provenance; frozen style, build-script and accepted-PDF digests retain their values. |

### Sample dispositions

Eight exact sentences survive from P18: fixed-precision recognizer scope,
parameter count, exact matched-token alignment, pre-registered cap, terminal
map sweep, trust inventory, 120-state recognizer floor and related-work
linear-state fraction. The other twelve slots pin:

1. Abstract's current 99.96/76.09 and 76.07/53.36 map finding.
2. Introduction's parameter match within 0.01%.
3. Actual-geometry throughput ratio 0.976 and 28,942/35,715 MB memory.
4. Comparable training loss with window-dependent sign.
5. Application trajectory error, latency and closed-loop stability test.
6. Hold-theorem λ ≥ 1 versus production decay below one.
7. One language-model seed and one seed per computational probe.
8. GDN2 leading every reported S5 length with both curves declining.
9. Mechanism persistence/disappearance under budget/selection controls.
10. Matched 3–7 B training/memory/computation study.
11. Frontier rank-0 training-loss average 2.2798 over 100 logging windows.
12. 73 K40 merges, median duration 6.320 s and interval 72 s.

Every sentence matches the canonical PDF and both TMLR PDFs with exact numeric
strings; `drift-check.json` stores all 20 sentences and per-PDF matches.
The all-source audit checks **1,099 long source fragments**, with three recorded
layout splits and four existing identifying front-matter/release exclusions.

## Ruling 6 — verified in the rebuilt PDFs

| PDF | References begins | Actual last reference | Appendix A begins |
|---|---:|---:|---:|
| canonical | 29 | 32 | 33 |
| anonymous submission | **24** | **27** | **28** |
| identified preprint | **24** | **27** | **28** |

Canonical final entry: Chaubard/Kochenderfer, arXiv:2505.17852. TMLR final
entry: Yang et al., arXiv:2406.06484. The check uses physical form-feed PDF
pages and the actual final entry, together with unique headings and TeX order.
Receipts: `p16b-appendix-order.json`, `p17c-register-appendix.json`, and the three
current `*-text.txt` extractions. This is built-PDF evidence, beyond source order.

## Anonymous submission receipt

`p21-anonymity.mjs` screens the new 28-page submission's visible text,
anonymous author metadata, XMP, embedded-file inventory, URI annotations and
font embedding. **PASS:** Anonymous authors/Anonymous metadata; empty XMP and
embedded-file inventories; **11 approved literature/documentation destinations
(seven unique); all 26 fonts embedded**. The two P20 scratchpad/chain URLs
extend the literature-link whitelist. Public model and mirror URLs remain
variant-gated; third-person scholarly citations remain. Details are in
`anonymization-output.txt` and `p21-anonymity-details.txt`.

## Validation chain

Environment: Node v24.21.0, Typst 0.14.2, `/tmp/tectonic`, Poppler under
`/tmp/p9-poppler`. Full versions: `p21-tools.txt`. Set
`TECTONIC_BIN=/tmp/tectonic`, `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`,
`PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext`,
`PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo`. All commands start at repository root.

| Command | Result / receipt |
|---|---|
| `typst compile --root . paper/main.typ paper/current.pdf` | PASS, 33 pages; `p21-canonical-build.txt`. |
| `typst compile --root . --pdf-standard 1.5 --no-pdf-tags --pages 1,2 paper/tmlr/figures.typ /tmp/p21-packaging-figures.pdf`; `pdfseparate ... assets/figure-%d.pdf` | PASS, two regenerated vector assets; manifest digests. |
| `node paper/figures/validate_assembly.js` | PASS, all 13 body sections + A, live scopes, label/checkpoint hashes, 17-file recursive closure and all 61 bibliography rows; `assembly-check.txt`. |
| `node paper/tmlr/convert.mjs` | PASS, 63 headings, nine figures/tables, 13 statements; copied bibliography; `convert.txt`. |
| `bash paper/tmlr/build.sh` | PASS, both 28-page PDFs; existing fail-closed glyph/error/reference/citation/overflow/oversized-float checks; `build-output.txt`, `{submission,preprint}-build.txt`. Text log copies trim trailing whitespace. |
| `node paper/tmlr/verify.mjs` | PASS, 20 × three samples, 1,099 fragments, 325 complete math expressions, 266 attachments, eight primes per PDF; identity, exact source/bibliography, register and physical references order; `verify-output.txt`, `drift-check.json`. |
| `node --test paper/tmlr/*.test.mjs` | **66/66 PASS**, full current TMLR suite; `p21-tests.txt`. |
| `node paper/tmlr/validation/p21-audit.mjs` | PASS, exact source/provenance/guard replay and packaging scope; `p21-preservation.txt`. |
| `node paper/tmlr/validation/p21-anonymity.mjs` | PASS, new PDF anonymity receipt; `anonymization-output.txt`. |
| `bash paper/tmlr/bundle-arxiv.sh`; fresh untar; `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; `pdftotext -raw`; `cmp` | PASS, fresh 28-page rebuild, raw text byte-identical to preprint; `bundle.txt`, `arxiv-build-output.txt`, `arxiv-compare.txt`, `p21-arxiv-{contents,pdfinfo}.txt`. |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS, all 22 paths; `p21-checksums.txt`. |
| `pdftoppm` spot checks | PASS, submission pp.1/13 and preprint pp.1/28 inspected; legible title/identity, new table/figure and compact register; `p21-visual-spotcheck.md`. |
| `git diff --check`; staged diff check; commit/push and remote equality | PASS receipts: `p21-final-checks.txt`, external `P21-push.txt`. |

## Tone, review gate and remaining scope

The exact P20 source, imported captions and scientific artifacts retain their
bytes, preserving the plain forward-narrated manuscript. Packaging translates
the evaluated content directly; the archive README uses affirmative delivery
language. The existing technical negations in mathematical statements and
algorithms retain their meaning. Guards for removed predecessor/set-D text
are retired specifically because P20 removes those claims, as enumerated above.

Independent acceptance review remains required. Visual checking is a four-page
spot check, rather than an exhaustive page-by-page audit. The four P20 intake
bibliography entries retain pending independent metadata review; existing
publication-metadata flags remain visible. Nonfatal underfull TeX spacing
warnings and Typst/Poppler's “Suspects object is wrong type (boolean)” diagnostic
remain; all fail-closed packaging checks pass. Development test failures in
`p21-tests-initial.txt` were a malformed test-string edit and an appendix
fixture without its own heading line; both fixtures were corrected and the
full 66-test suite passes. An initial missing Ž glyph triggered the existing
build guard, leading to explicit Ž/² mappings and built-PDF glyph checks.
The implementation evidence supports the reviewer gate; it adds no scientific
or experimental certification beyond the transferred manuscript.
