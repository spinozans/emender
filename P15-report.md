# P15 — three supervisor author rulings

## Contract and ruling-to-site map

Started directly at P14 commit `c8e5e69b` on `paper-1p3b-spine`; HEAD already
matched the requested base, so no rebase was needed. Authority:
`/home/erikg/emender-scratch/fix-fanout/taskP15-author-rulings.md` and the
**Author calls** section of `P13-report.md` in this worktree.

| Ruling | Applied site in delivered main.typ | Change |
|---|---|---|
| 1 / call #1 | 839, after the per-head display and before Algorithm 1 | One sentence distinguishes idealized nonzero-address normalization from production `norm+1e-6`, citing `elman/models/e88_fused.py:239-242` in the manuscript's inline-code reference form. |
| 2 / call #2 | 1035, immediately before Algorithm 2 | One sentence identifies Algorithms 1/2 as abstract cores and the released projection/normalization/gate-domain/residual machinery as a full-stack binding carried by limitation G2, cross-referencing the Limitations section. |
| 3 / call #5 | 179, 1699, 1733, 2424, 2426, 2772, 2923, 2954, 2959, 2964, 3083, 3317 | All twelve tracker-gap references change F1 → O1, including the defining Open item, limitation item/heading, and introduction/expressivity/formal/conclusion/future-work references. |

The live label sweep also updates `paper/OUTLINE.md` (two references),
`paper/results/RESULTS_INVENTORY.md` (one current-authority pointer), and
`paper/figures/validate_assembly.js` (expected label and status message).
Generated TeX, conversion inventory, PDF texts and guard receipts inherit O1.
Historical reports/review notes, frozen PDFs, and unrelated F1 audit/figure
identifiers remain historical evidence, not current tracker-gap references.
The saturation **Theorem F1** in `paper/data/math-notation-spec.md` is unchanged.

No science was changed for calls #3/#4. The tracker remains an open item,
with identical realization/trajectory requirements. Abstract/production
domain gaps, protocol boundaries and scoped naming overloads remain as
disclosed. A source replay reverses exactly the two added sentences and
twelve relabels to recover the entire P14 manuscript byte-for-byte; all
**eight algorithms and 743 literal math spans** are identical. No training,
Lean source, bibliography, experimental data, or figure source was edited.

The cited local source was read and checked at
`/home/erikg/elman/elman/models/e88_fused.py:239-242`: both q and k divide by
`norm + 1e-6`. Its inspected SHA-256 is
`95adda05d9488f637aae6a950d1013e4f21bf3239389508cc320a73602468000`.
This citation does not supply a new full-stack equivalence or trajectory proof.

## Enumerated guard regeneration

The unchanged canonical guard first rejected the edited source against P14's
digest (expected negative check). The authorized baseline advance retains
all accepted/P13/P14 provenance and changes only:

- `mainTypSha256`: `c899544128ebd08948edf721e86f1b15d249ecfbbfee00166b3a9e6b7314f7ad`
  → `2841f1432eec22ac4357829dbfef583006da65f921335dc9a57d565964aad59c`.
- Retains the former digest as `p14MainTypSha256`; adds
  `p15BaseCommit: c8e5e69b`; appends this directive/report and the exact
  `validation/author-rulings-revision.diff` path to `revision`.

The source/live-label diff is retained with zero context. Guard implementation,
accepted snapshot `eb2b96363e3ff800d7eefe2ba7386e7cf673d1cf`, frozen PDF,
bibliography amendment and style provenance are unchanged. All 20 fixed
numeric/scope samples and the full symbol audit are **deep-identical to P14**.
Source-derived coverage changes **2,567 → 2,571** long fragments; every current
fragment passes in all three PDFs. Conversion stays at 98 headings, 20 captions
and 21 formal statements; the sole conversion-inventory change is the
Limitations subsection title's F1 → O1.

Every layout-split record change is enumerated below. This is the existing
split policy, not a new exception; Algorithm 5's complete raw block remains
byte-identical and now matches contiguously. Full strings are also retained
in `validation/p15-checks.txt`.

| Disposition | PDF | Fragment | Split index |
|---|---|---|---:|
| Removed | submission | Algorithm 5 raw block, `Input: arm, seed set, task generator, recorded study recipe` through `Return: accuracy-by-length records and protocol/failure metadata` | 246 |
| Removed | preprint | Same complete Algorithm 5 block | 246 |
| Removed | submission | `kernel choice. The reported successful-seed subsets nevertheless` | 15 |
| Removed | preprint | `kernel choice. The reported successful-seed subsets nevertheless` | 15 |
| Added | canonical | `(the M²RNN family) accumulates without` | 8 |
| Added | submission | `theorem, not a per-token circuit-depth lower bound` | 11 |
| Added | preprint | `theorem, not a per-token circuit-depth lower bound` | 11 |

SHA256SUMS keeps exactly the same 15 paths. Six digests change: main.typ,
current.pdf, submission.pdf, preprint.pdf, arxiv-source.tar.gz and
canonical-baseline.json. The other nine, including both vector drawings,
bibliography, build wrapper, frozen PDF and vendored style, are unchanged.
Generated files were rebuilt, not hand-edited; no acceptance check was weakened.

## Validation

Executed on lambda01, manuscript-only. No Python, Slurm/native build,
resilient training or WorksGood machinery was used; resilient-DiLoCo
requirement IDs are not applicable. Full logs/check scripts are in
`/home/erikg/emender-scratch/fix-fanout/P15-validation/`; current durable
receipts are under `paper/tmlr/validation/`, including `p15-checks.txt`.

| Command | Result |
|---|---|
| `node .../P15-validation/preservation.mjs` | PASS: exact source replay, eight unchanged algorithms, 743 unchanged math spans, calls #3/#4 and overload preservation, cited code lines, live-label sweep, identical fixed samples/symbol oracle and retained provenance. |
| `typst compile paper/main.typ paper/current.pdf` | PASS: **65 pages**. |
| `node paper/tmlr/convert.mjs` | PASS: 98 headings, 20 figure/table captions, 21 formal statements; byte-identical bibliography. |
| `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` | PASS: anonymous/identified PDFs **58 / 58 pages**; no missing glyphs, unresolved references, overflows or errors. |
| `bash paper/tmlr/bundle-arxiv.sh`; fresh extraction; `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; Poppler `pdftotext -raw`; `cmp` | PASS: standalone archive rebuild has byte-identical raw text to preprint. |
| `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo node paper/tmlr/verify.mjs` | PASS: 20 × 3 fixed samples, 2,571 long fragments, 707 complete math expressions, 441 attachments, eight S primes per PDF including two drawing primes; source/bibliography/identity guards. |
| `node --test paper/tmlr/verify-symbols.test.mjs paper/tmlr/notation-audit.test.mjs paper/tmlr/author-rulings.test.mjs` | PASS: **20/20**; three new source-named tests cover regularization versus exact-norm math, abstract cores/G2 reference, and O1 versus saturation F1. Existing 17 tests unchanged. |
| `node paper/figures/validate_assembly.js` | PASS: 13 body sections, three appendices, 57 references, G1–G3/O1/U3.1, 17-file trust inventory and checkpoint bindings. |
| `bash formal/lean/scripts/check_paper_core.sh` | PASS: 17 local project sources; no placeholders, added axioms, opaque or native_decide. |
| `(cd formal/lean && lake build ElmanProofs.PaperCore)` | PASS: 2,196 jobs, cached build; existing linter warnings only. |
| `node formal/lean/scripts/test_local_import_closure.js` | PASS: recursive imports, external dependency exclusion, five forbidden tokens and compiler-error fail-closed fixtures. |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS: all 15 paths. |
| `git diff --check` | PASS. |

## Diagnostics, risks and delivery

The initial scratch edit expected thirteen F1 occurrences; the actual count
was twelve and the assertion stopped before writing. A direct strong-text G2
label was not referenceable in Typst, so the final sentence uses the existing
Limitations section reference and names G2 explicitly. An initial en dash in
the raw code-line citation lacked a TeX monospace glyph; the unchanged build
guard rejected it, and the final citation uses ASCII `239-242`. Final builds
and all checks pass. Existing underfull TeX spacing warnings and Typst PDF's
Poppler `Suspects object is wrong type (boolean)` diagnostic remain non-fatal.

Delivery is one commit, `author rulings: normalization note, core-definition
sentence, open-item relabel`, including this report and explicit task files,
then push to `origin/paper-1p3b-spine`. The final hash, remote comparison,
empty-index check and untouched pre-existing untracked-file list are retained
in the scratch `delivery.txt` receipt after commit/push; no second report-only
commit is planned. Independent reviewer acceptance is still required.
Calls #3/#4 remain disclosed open research/protocol items, not newly solved
claims. Next: independent review of the three ruling-to-site mappings.
