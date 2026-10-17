# P25 — growth curriculum and the learned composition frontier

## Authority and scope

Worktree `/home/erikg/emender-paper`, branch `paper-1p3b-spine`, base
**610fa8a2dacf79fd2413318cf73ce14caaa3beb6**; remote equality checked before
editing. Contract: `/home/erikg/emender-scratch/fix-fanout/taskP25-growth-integration.md`.
Read the X5c launch and positional reports, the scoped X5cR and X4R reviews,
and the existing X3/X4 manuscript evidence. The supervisor confirmed that
X5cR's stagnation evidence is ACCEPTED and its sole composition-depth
condition is satisfied by X5cP (commit 75800435). The original review's
conditional verdict is preserved, not rewritten as unconditional acceptance.

The only manuscript change is the new subsection after flat-training and
marginal analysis, with a small six-run positional table. This is paper
integration, not training, model evaluation, or a change to stopping behavior.
No Python/native builds, scheduler submission, resilient behavior, production,
CI, formal source or bibliography changes. Resilient-DiLoCo requirement IDs:
**not applicable**. Independent P25 reviewer acceptance remains required.

Delivery is one commit, `P25: integrate growth curriculum and positional composition frontier`,
pushed to `origin/paper-1p3b-spine`. The commit, push, remote equality and
clean-index receipts are external at
`/home/erikg/emender-scratch/fix-fanout/P25-push.txt` to avoid a self-referential
hash in this report.

## Retire / keep / add map

| Site / interpretation | Disposition |
|---|---|
| Existing X3/X4 flat-training and marginal-like analysis | **Keep, byte-for-byte.** No rewriting limited predictive signal as an identified internal marginal-solution mechanism. |
| A short or early plateau as sufficient evidence of failure | **Retire that interpretation.** Add measured one-step E97 signature: 2.93% at 20K, S-curve to 97.85% at 57K; archived rule would stop at 10K. This motivates protocol calibration, not a learning mechanism. |
| Open-horizon curriculum | **Add.** Threshold-only advancement (90%), registered stagnation with replay-calibrated 100K floor, streak reset, two eligible post-floor 2K-window comparisons, both strict 0.1% criteria, earliest stop 104K. Safety reasons stay separate; none fired. |
| Growth versus stage progression | **Add exact scope.** Planned 4→8→16→32→64→128; all six stop at the first four-driver stage. No claim of reaching later stages or of successful multi-stage growth. Scratch nominal-8M, seeds 0/1/2, matched 1e-3, Schedule-Free optimizer, procedural training and frozen validation are explicit. |
| Final versus selected checkpoint scores | **Keep distinct.** Final E97 24.95–25.15%, GDN2 50.05–75.00%; every final terminal 0/512. Per-seed percentages and stop updates printed. Positional results use existing best aggregate-validation checkpoints, accuracy then loss tie-break, no position-based re-selection. E97 best seeds 1/2 each have one terminal hit. |
| Aggregate fractions alone as proof of depth | **Retire that inference.** Add `<tab_growth_positions>` reading the exact frozen positional receipts. Each position has 512 examples; four drivers follow two unscored initialization tokens, causally processed without target feedback; position 1 is one map application. |
| Composition frontier | **Add in learned-solution scope.** E97 depth 1 across all seeds (511–512 at P1, 0–10 at P2); GDN2 depth 3 in seeds 0/1 (512 and 508 at P3), depth 2 in seed 2 (4 at P3). P4 unsolved by all (0–1). “Tracks” is ≥90%, not literal perfection. Partial-coordinate signals 338/512 and 105/512 remain visible. |
| Integrated flat-versus-curriculum reading | **Add.** Flat full-depth training remains near the marginal reference; the progressive-depth protocol exposes a frontier at its first stage: learned GDN2 solutions chain three/two applications, learned E97 one, neither four. Not an architectural capacity limit, impossibility, convergence certificate, or identified internal mechanism. |
| Suffix bound and marginal reference | **Keep and repeat beside new result.** 0.026874% only for initialization plus proper suffix with at least two omitted drivers, not a recurrent-model bound; 8.221748 nats uniform marginal reference. |
| S5 companion and all prior empirical tables | **Keep.** S5 text and table definition unchanged; no broader cross-task superiority claim added. All six prior empirical table bodies and seven prior table structures replay unchanged. |
| Abstract, introduction, conclusion, formal results, language modeling, systems, next tests, compact register | **Keep.** P25 is the expressivity section's new nonlinear-map culmination, not a whole-paper repositioning. Existing register points into the evidence root containing the new subdirectory. Scalar headline and claim table remain retired. |

## Frozen evidence and packaging changes

`paper/results/invertible_quadratic/growth/provenance.json` binds **17
byte-identical copies** to original paths and SHA-256 hashes: six positional
receipts, six terminal result JSONs, X5c-launch/X5cP/X5cR reports, the archived
rule replay, and the original E97 one-step validation curve. Existing X3/X4
evidence and provenance files retain their exact bytes. No checkpoints or
training logs are imported. Positional receipts retain checkpoint hashes,
paired frozen-pool hashes, joint/x/y counts, losses and best/final accounting.
This is a post-training validation diagnostic, not an independent test pool.

The exact source/table delta is `paper/tmlr/validation/p25-growth-revision.diff`.
The canonical pin preserves historical lineage and updates the approved source
hash. No math expression is added or removed: **352 complete expressions,
293 attachments, eight state primes** remain. Conversion adds one unnumbered
subheading and one six-column/seven-row table: **67 headings, eight tables,
11 captions, five algorithms and 13 amsthm statements**. The two vector
assets are re-exported; PNG bytes are unchanged. `README.md` describes the
current release. `integration-audit.mjs` now replays the exact P25 delta
against 610fa8a2, preserves all prior evidence/formal bytes, six empirical
table bodies, seven table structures, S5 definition, bibliography and styles,
and limits scope to paper files plus this report.

Five new `growth-curriculum.test.mjs` tests independently replay source hashes,
the measured delayed one-step curve and old/fixed replay outcomes, all six
stops/final ranges, all 24 positional counts and their generated TeX rows,
paired validation hashes, aggregate/best/final reconciliation, coordinate
exceptions, and new claim scope in every PDF. Five new live-scope guards plus
one caption guard have negative mutation tests. Conversion pins update only
for the new subsection/table. All inherited symbol/source drift and unrelated
scope guards remain. The first test run was **87 pass / 1 fail**: a new PDF
phrase check retained the canonical page number inserted mid-sentence. Its
normalizer now removes running furniture using the existing verifier's rule;
no manuscript content was missing and no numerical receipt changed. The
initial output remains in `p25-tests-initial.txt`; final result is **88/88**.

## Validation receipts

All receipt paths below are under `paper/tmlr/validation/` unless qualified.
Environment: Node v24.21.0, Typst 0.14.2, `TECTONIC_BIN=/tmp/tectonic`,
`LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`, `PDFTOTEXT` and
`PDFINFO` from `/tmp/p9-poppler/usr/bin`. No Python invoked.

| Check | Result / receipt |
|---|---|
| Canonical pin → Typst compile → vector export → conversion → Tectonic build → verify | **PASS**, `p25-chain.txt`, `p25-build.txt`, `p25-verify.txt`; **36 / 31 / 31 pages** canonical/submission/preprint. No missing glyphs, unresolved references, overflow or errors. |
| Source/PDF drift and independent math oracle | **PASS**: 20 samples × three PDFs, **1,290** long source fragments, 352 expressions, 293 attachments and eight primes in each PDF. New signature, depth scope and checkpoint caveat additionally tested in all three PDFs. |
| `node --test paper/tmlr/*.test.mjs` | **88/88 PASS**, `p25-tests.txt`. Includes five source-named numerical/packaging tests and six new negative scope-loss regressions. |
| `node paper/figures/validate_assembly.js` | **PASS**, `p25-assembly.txt`: 13 body sections + A, labels, findings/scopes, 17-file Lean closure, all 61 bibliography entries. |
| `node paper/tmlr/integration-audit.mjs` | **PASS**, `p25-integration.txt`, `integration-audit.json`: exact P25 delta and preservation checks. |
| Anonymous-PDF audit | **PASS**, `p25-chain.txt`, refreshed `p21-anonymity-details.txt`: 31 pages, anonymous visible author/metadata, empty XMP/embedded-file inventories, 11 approved URI annotations (seven unique), 26 embedded fonts. |
| References immediately before Appendix A | **PASS**, `p16b-appendix-order.json`: canonical p.35 → A p.36; both TMLR variants p.30 → A p.31. Register categories/destinations/pointers pass. |
| Source archive → fresh extraction → Tectonic → raw-text `cmp` | **PASS**, `p25-archive.txt`, `p25-arxiv-build.txt`: one identified master, four assets, 31 pages, text byte-identical to preprint. |
| Visual spot checks | **PASS**: canonical pp.20–21 and submission pp.17–18 raster-read: signature/protocol, numeric ranges, all six table rows, best/final caveat, scoped integration and unchanged S5 table legible/unclipped. Images in `/tmp/p25-{canonical,submission}-*.png`; not a full-document visual audit. |
| Active manifest | **PASS**, `SHA256SUMS` refreshed for current deliverables, new evidence, tests, report and receipts; `p25-checksums.txt`. |
| Whitespace / delivery | `git diff --check`, staged diff check, one commit/push, remote equality and no staged files recorded in external `P25-push.txt`. |

## Residual risks

Independent P25 reviewer acceptance remains required. Validation-selected
checkpoints and different stop times are not equal-budget or held-out-test
comparisons. All runs stop at T4; later stages remain unattempted. The floor
is replay-calibrated, not a guarantee against all delayed learning. Depth is
an observed ≥90% tracking frontier for these learned solutions and frozen
validation inputs, not exact computational depth or architecture capacity.
No new experiment or mechanism claim is authorized. Existing nonfatal
underfull TeX spacing warnings and Poppler's Typst-PDF `Suspects object is
wrong type (boolean)` diagnostic remain. P23's previously flagged unrelated
applications notation and bibliography metadata questions remain untouched.
