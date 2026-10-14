# P13 — language, notation/protocol audit, and durability inventory

## Part 1 — operator-directed language sweep

Four occurrences removed from the active manuscript; no method or quantitative
content changed. EXHIBIT/ANALOGY/OPEN already used neutral scope definitions
(main.typ:1987–1990 before Part 2); names retained. No `carefully` or `rigorously`
occurrences in the manuscript or its caption module. Historical review/outline
notes, data source quotations, and frozen PDFs are not manuscript variants
and are retained as provenance, not rewritten.

Every changed source line (old/new line numbers in diff headers):

```diff
diff --git a/paper/main.typ b/paper/main.typ
index 3d2956a1..4794e91b 100644
--- a/paper/main.typ
+++ b/paper/main.typ
@@ -1367 +1367 @@ the window-dependent sign of the $Delta$ in @tab_matched_loss is the
-honest summary of a tie at this precision, not evidence for either
+summary of a tie at this precision, not evidence for either
@@ -1385 +1385 @@ Within that band E97-MLP is ahead of GDN2-MLP on *both* primary metrics
-but the honest statement is descriptive closeness with the reported
+but the statement is descriptive closeness with the reported
@@ -1529 +1529 @@ surprising, and why the paper's positive claims live in #ref(<sec:expressivity>)
-Downstream capability panels: an honest null, on the predecessor cohort]
+Downstream capability panels: a null, on the predecessor cohort]
@@ -1534,2 +1534,2 @@ from the predecessor 1.3 B racer cohort (E88, FLA-GDN, M²RNN, Mamba2 —
-#ref(<sec:appendix_cohort>)), and it is reported here without spin as the honest
-expectation to beat: at $approx 1 M$-step training extents, on a
+#ref(<sec:appendix_cohort>)), and it provides the
+reference comparison: at $approx 1 M$-step training extents, on a
```

Supervisor approved moving the guard to the P13 source SHA-256, retaining
the eb2b9636 accepted source snapshot/history and frozen PDF. New source digest
and lineage: `paper/tmlr/canonical-baseline.json`. The rebuilt `paper/current.pdf`
is the verification authority; no old fragment mismatch was bypassed.
Part 1: current PDF 62 pages, both TMLR PDFs 53 pages; 20 samples × 3 PDFs,
2,499 source fragments, 643 math expressions, 417 attachments, three state
primes per PDF, 12/12 symbol regression tests passed. Bundle rebuilt.

## Part 2 — notation and pseudocode audit

Audit authority: the current manuscript, evaluated Typst math, and the cited
Lean definitions (SplitGatedDelta, CompositionalDepthDuality, S5Tracker).
No Lean theorem, experimental result, training implementation, or release
was changed. One manuscript-wide symbol table is at main.typ:490–529.
Paper state S maps explicitly to Lean H; paper erase gate e maps to Lean b.
H remains the M²RNN comparator state and, explicitly as a count, the number
of heads, respecting the pre-existing notation spec. Application-specific
state coordinates and locally overloaded scalars are explicitly scoped.
This is a source/presentation audit, not a proof of mathematical or numerical
equivalence of the full production stacks.

### Every identified symbol inconsistency and its disposition

Old lines are from Part 1 commit 96aad6f9; new lines are the Part 2 source.

| Old main.typ lines | Issue | Fix / current location |
|---|---|---|
| 511, 513, 521, 1076 | Generic affine state h vs architecture/formal state S | S in vector coordinates; 552, 556, 564, 1168 |
| 661–663, 2573 | E97/GDN abstract H confused with head count and M²RNN H | S in both adjacent cores and affine-fold theorem; 704, 2759; Lean alias explicitly mapped |
| 661–662, 2560 | Erase gate b vs production e; generic offset b has a different role | e in printed core and schedule, Lean b mapped; 704, 2746 |
| 796–799 | Vector gate products used a scalar/inner-product dot | Hadamard operator in retrieval, delta and output; 839–842 |
| 862, 865, 868, 871, 915, 943–945 | Same operator drift in schematic, caption, ingredient list | Figure regenerated; 921, 1002 and adjacent caption use Hadamard |
| 798–799, 868, 871 | In-place S update obscured old-vs-new state read | S′ explicit, output reads returned S′; 841, 842; Algorithm 1 returns S_next |
| 802 | Vector SiLU product written dot | Componentwise product; 845 |
| 1951, 1955 | Generic family state s vs S | S in repeated-map definition; 2123, 2127 |
| 2512 | Decay-times-state notation differed from application form | Same scalar juxtaposition; 2698 |
| 2513, 2532, 2543 | W used for scalar slot write and learned matrix | omega for latch contribution; 2699, 2718, 2729 |

The historical math-notation spec also had stale §7 formal-section references
and a normalized witness key (old lines 114–120), contradicting the actual
C/C′ witness (1,1), its -1 distinguishing entry, and the manuscript. Fixed
references to §8, witness to (1,1), the split-gate delta explanation, and
slot-write/source mappings in paper/data/math-notation-spec.md. Deferred
blueprint examples remain deferred, not a second manuscript symbol authority.

### Algorithm / formal-block coverage

All eight added blocks use explicit Input, numbered execution order, Return,
and adjacent plain-language descriptions. The converter preserves multiline
raw blocks as verbatim pseudocode (no body font-size override).

- main.typ:862: Algorithm 1 — E97 per-head step.
- main.typ:1041: Algorithm 2 — GDN-2 split-gated linear core.
- main.typ:1108: Algorithm 3 — raw-write nonlinear comparator step.
- main.typ:1406: Algorithm 4 — pair selection, training, and loss alignment.
- main.typ:1826: Algorithm 5 — probe-study train/evaluate loop.
- main.typ:2005: Algorithm 6 — matched-cap fine-tune and terminal evaluation.
- main.typ:2146: Algorithm 7 — repeated-map fold.
- main.typ:3935: Algorithm 8 — tokenizer-invariant held-out evaluation.

The application one-step formulas instantiate Algorithm 7 (the repeated-map
fold); each retains its own state, injection, map, scope and negative cases.
The formal A–H blocks are declarative results with accompanying English
statements, hypotheses and exact Lean names, not algorithms requiring invented
proof procedures. The generic probe loop is a protocol skeleton: it does not
pretend disparate studies had identical optimizers, successful-seed subsets,
curricula, or stops.

Every standalone display block was inspected in context (current line ranges):

- 834–843: Per-head update; adjacent prose defines the variables and operation.
- 1098–1101: Architectural contrasts: delta correction and state nonlinearity; adjacent prose defines the variables and operation.
- 2122–2124: Repeated nonlinear maps in the wild <sec:applications>; adjacent prose defines the variables and operation.
- 2133–2135: Repeated nonlinear maps in the wild <sec:applications>; adjacent prose defines the variables and operation.
- 2176–2179: Sigma–delta conversion: quantization inside the loop; adjacent prose defines the variables and operation.
- 2201–2204: Adaptive filtering and automatic gain control; adjacent prose defines the variables and operation.
- 2213–2217: Adaptive filtering and automatic gain control; adjacent prose defines the variables and operation.
- 2233–2236: Proximal iterations and unrolled optimization; adjacent prose defines the variables and operation.
- 2242–2245: Proximal iterations and unrolled optimization; adjacent prose defines the variables and operation.
- 2259–2261: Reservoirs and echo-state networks; adjacent prose defines the variables and operation.
- 2282–2284: Discrete ecology: the logistic map; adjacent prose defines the variables and operation.
- 2304–2307: Feedback control with limited actuators; adjacent prose defines the variables and operation.

### Author calls — left open rather than changing the science

1. **Normalization boundary / production fidelity:** main.typ:835–836 and Algorithm 1 divide by the exact norm. The old formula is undefined at a zero projected address; no epsilon or zero-vector convention is supplied. The neighboring local E88 code uses norm+1e-6 (external /home/erikg/elman/elman/models/e88_fused.py:239–242; e88_fla_hybrid.py has multiple fused/unfused normalization routes). An author must bind the exact released E97 path and decide whether to print an idealized nonzero-domain step or its numerical regularization; no guessed epsilon was inserted. Likewise the displayed gate projections omit possible biases/processing.
2. **Core versus full GDN2-MLP definition:** main.typ:1031–1041 defines the cited abstract core, not every projection, normalization, gate-domain transformation and residual wrapper of the released baseline. Adding a fully specified production-stack algorithm requires a release-to-source binding and an author scope decision. G2 already records state-dependent upstream gates as unmodeled. Algorithms 1/2 do not assert full-stack equality.
3. **Abstract versus production domains:** main.typ:2433 (tracker realization), 2650–2664 (unnormalized key / boundary and out-of-range gates), and 2717 (lambda>=1 vs production decay<1) remain substantive gaps, not notation fixes. No parameter remapping or trajectory theorem was invented.
4. **Protocol boundaries / absent data:** successful-seed conditioning, unequal terminal update counts, single 1.3 B seed, S5 OOM curriculum stage, missing first-generation winner fitness U3.1 and missing historical 8 M per-seed records remain explicit. Algorithm 6 follows the implemented zero-based stop indices (driver scripts/expressivity_1p3b_depthwall.py:150–162), not a silently corrected minimum-step policy. Generic Algorithm 5 does not supply missing trial recipes.
5. **Legacy naming collisions:** H count vs raw-write state, scalar delta perturbation vs vector correction, T length vs latch threshold, lambda decay vs Lyapunov exponent, and local application x/w/g/h states are scoped in the common table. These are intentional local roles rather than algebraic contradictions; completely eliminating overloading would require an author-wide renaming. The deferred blueprint spec uses Theorem F1 for saturation while the manuscript Open item F1 is tracker realization; the companion's label scheme needs an author ruling before publication.

### Rebuild / regression evidence

Current Typst PDF: 65 pages; anonymous and identified TMLR PDFs: 58 each.
20 samples × 3 PDFs and all 2,587 long current-source fragments pass.
707 complete math expressions / 441 converted attachments pass the independent
oracle; all eight S primes (six converted, two in imported drawings) survive
in every PDF. Added five notation/protocol tests and updated the twelve symbol
regressions: 17/17 pass, including dropping each converted prime and a missing
drawing prime. No number or qualification from the pre-sweep twenty-sample
ledger was rebased away. The engine found no missing glyphs, references,
errors, overfull boxes or oversized floats; underfull spacing warnings remain.
Fresh standalone archive build has byte-identical raw text to preprint.pdf.
CPU assembly audit passes: 13 sections, three appendices, all 57 references,
17-file trust inventory, theorem scopes and both checkpoint SHA bindings.

Diagnostic attempts retained in scratch: a replacement-string interpolation
error was caught before build and the source was regenerated from Part 1;
new ell glyph lacked a TeX mapping, fixed in both converter and oracle;
prime census initially omitted the regenerated drawing's two primes, fixed
with an independent vector-prime count and regression test; stale test counts
were updated to the source-derived census. No acceptance check was disabled.
Git initially treated the new Typst PDF as text and flagged its PDF xref CRLF
as whitespace; paper/.gitattributes now marks current.pdf binary.

## Part 3 — committed results durability inventory

`paper/results/RESULTS_INVENTORY.md` contains 152 TRACKED artifacts, each
with exact byte size, SHA-256, and role. Its 40KB manifest includes all
paper/results CSV/JSON inputs, depth-wall sweeps and training records, the
matched throughput budget, 8 M control aggregate plus 15 per-seed sources,
S5 capacity and crash records, CMA winners/search records, trust inventory,
evidence/source ledgers, release identity receipts, and key-table sources.
Fourteen tracked artifacts exceed 100,000 bytes, including the 3.27MB trust
inventory and historical multi-MB loss CSVs; every exception is marked.
No scientific input artifact was changed or copied into Git for this part.

Findings: the E97 winner is tracked as RESULT/results.json, not a separate
best.json (GDN2's best.json is tracked); five external metadata files are
SCRATCH-ONLY (both runs' args/run_manifest plus GDN2 launch_manifest), with
hashes; four old /tmp probes and the requested E97 best.json spelling are
MISSING. The historical exact 8 M S3/S5 seed records remain unlocated.
The evidence dossier is a stale historical snapshot, not a current claim
oracle. Current-pair aligned-loss CSV/JSON and full QA per-item manifest
snapshots are not in paper/results; committed summary receipts/CSVs are
not represented as complete raw-data preservation. Existing unrelated
untracked review files are untouched. No checkpoint-scale payload was hashed
or added, and no remote S3 durability claim was inferred from Git status.

Independent verification parses the delivered Markdown (not its generator),
uses sha256sum, checks sizes and >100KB marks, requires exact git ls-files
membership, and compares every tracked input byte-for-byte to git show HEAD.
Result: 152 tracked checks, five external hashes, five explicit missing
paths, fourteen size exceptions, all PASS. Scratch scripts/output are under
/home/erikg/emender-scratch/fix-fanout/P13-validation/. An initial whole-index
listing exceeded Node's default stdout buffer; the explicit buffer was
raised before generating any inventory. No digest or membership was skipped.

## Validation commands and delivery

Executed on lambda01. No Python, Slurm/native build, training, or WorksGood
service was used. Resilient-DiLoCo requirement IDs are not applicable to
this manuscript-only audit; protocol pseudocode documents existing recipes
without changing resilient training behavior.

- `typst compile paper/main.typ paper/current.pdf` — PASS (65 pages).
- `node paper/tmlr/convert.mjs` — PASS (98 headings, 20 figure/table captions,
  21 amsthm statements; bibliography identity retained).
- `typst compile --root . --pdf-standard 1.5 --no-pdf-tags --pages 1,2
  paper/tmlr/figures.typ ...`; Poppler `pdfseparate` — PASS, both vector
  drawings regenerated; Figure 2 carries the fixed gate/returned-state notation.
- `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` — PASS (58/58 pages).
- `bash paper/tmlr/bundle-arxiv.sh`; fresh tar extraction and
  `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; raw PDF text
  `cmp` against preprint — PASS standalone.
- With `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`,
  `PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext`,
  `PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo`, `node paper/tmlr/verify.mjs`
  — PASS (20×3 samples, 2,587 source fragments, 707 math / 441 attachments,
  eight primes in each PDF, identity/source guards).
- `node --test paper/tmlr/verify-symbols.test.mjs
  paper/tmlr/notation-audit.test.mjs` — PASS (17/17).
- `node paper/figures/validate_assembly.js` — PASS.
- `sha256sum -c validation/SHA256SUMS` in paper/tmlr — PASS (15 entries).
- `node .../P13-validation/verify-inventory.mjs` — PASS as detailed above.
- `git diff --check`, explicit-path per-part commits, empty-index check,
  push and remote-branch hash comparison — required delivery checks below.

Next: independent review of these three parts, then author disposition of
the five structural/naming/protocol calls above. No venue submission occurred.

### Verified delivery receipt

- Part 1 commit: `96aad6f907cf24508f6205fdfd742e0c576bdc5a`.
- Part 2 commit: `738236d6b7ad4517d35815c18b8ea0c1ce92e367`.
- Part 3 commit: `fe108b211674f3c245f82fd17002b85bb1d83959`.
- All three pushed to `origin/paper-1p3b-spine`; origin is
  `git@github.com:spinozans/emender.git`. After push, `git ls-remote origin
  refs/heads/paper-1p3b-spine` exactly equaled Part 3 HEAD
  `fe108b211674f3c245f82fd17002b85bb1d83959`.
- Final 17-test replay, 152 tracked-result digest/blob checks, five external
  hashes, five missing-path checks, and all 15 PDF/bundle/source digest checks
  passed. `git diff --check` passed; index empty. Only the seven pre-existing
  untracked review files remain. This receipt is committed/pushed separately
  after the per-part push, following the previous packaging report convention.
- Self-review found no unreported implementation blocker. Independent review
  is still required; normalization/full-stack fidelity and the other author
  calls are not certified as resolved. Underfull TeX spacing and Typst's
  pre-existing Poppler “Suspects object is wrong type” diagnostic persist.
