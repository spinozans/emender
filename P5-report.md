# P5 — P4 blocker repairs and textual punchlist

## Result and authority

Implemented B1–B5 and the textual/claim-scope portions of P1–P8 in
`/home/erikg/emender-paper`, branch `paper-1p3b-spine`. No experiments,
training, native/GPU builds, Slurm submissions, or resilient-training behavior
were changed or run. No new proof claim, production inclusion, or positive
experimental result was manufactured. The frozen 1.3 B negatives, one-seed
boundaries, F1/G1–G3 and U3.1 remain. **Independent reviewer acceptance is
still required**; successful regressions are not submission acceptance.

Read the supplied `taskP5-blocker-repair.md` and all of `P4-review.md` first.
The baseline is `d923265a9a406bd388fa53ceb9ae65fe04f2708a`. Original P4
review deliverables were left untouched and untracked.

The supervisor explicitly approved limiting P7 to lineage/textual fixes and
leaving journal-specific template/anonymization packaging open: no current
TMLR template was supplied. The manuscript is an identified Arkheion preprint,
not an accepted or format-certified TMLR submission.

## Blocker-to-location repair ledger

Locations prefixed **P4** are the referee's original coordinates, not shifted
line numbers in the repaired manuscript. Current source is commit `a09cf845`.

### B1 — synthetic accounting, not actual FLOP equivalence

- **P4:** `paper/main.typ:292–305,552–561,2357–2382`; proof locations
  `RecurrentResourceFormalism.lean:1164–1204,1219–1257,1259–1340` and
  `M2RNNComparison.lean:59–65`.
- Renamed/scoped the contribution, background argument, and set D to a
  synthetic/coarse state-pass accounting model. Each argumentative site
  discloses excluded actual dense right-multiplication `H W` and
  feature-network costs, and rejects an actual matched-empirical-compute
  inference. The repeated introduction argument at original `main.typ:243`
  was also corrected in the punchlist audit.
- Current locations: `paper/main.typ:243–248,294–315,557–570,2390–2420`.
  The dense square multiplication's cubic arithmetic versus the model's
  state-sized charge is explicit. Empirical budgets remain empirical claims,
  not consequences of the synthetic theorem.
- Preserved the separately valid E97/E88 result, naming the precomputed-gate
  recurrent-update accounting model, `d >= 2`, `6 d²` versus `7 d²`, and
  excluded gate/feature generation (`main.typ:2403–2413`). No theorem body
  or theorem name was changed.

### B2 — conditional containment; symmetric-group minimality

- **P4:** `paper/main.typ:474–477,531–532,577–585`; adjacent caveat
  `515–522`; `S5Witness.lean:75–86` proves order/non-solvability only.
- Replaced the unconditional “too weak” impossibility with conditional
  restricted-SSM containment under transition/arithmetic assumptions; no
  TC0/NC1 separation is inferred. Both minimality statements now say
  “smallest non-solvable symmetric group.”
- Current locations: `main.typ:479–486,540,587`. The surrounding S3
  interpretation is now “a nonabelian solvable symmetric group,” avoiding
  the additional incorrect smallest-nontrivial-group phrasing. Existing
  affinity, all-words recognition, unproved separation, and G3 caveats remain.

### B3 — positive, non-monotone measured gaps

- **P4:** `paper/main.typ:1699–1709,2287–2291,3030–3034`;
  `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md:14–17,23–26,33–36`.
- All three repeated monotonic-widening claims were replaced. Parity peaks
  near 0.359 at length 320 then narrows to 0.337 at 500. FSM also peaks at
  320; counter shrinks throughout. Positive ordering persists on these
  measured grids, but neither exact finite-k non-equality nor these curves
  establishes an increasing gap or the proposed S5 capacity bound.
- Current locations: `main.typ:1712–1729,2320–2326,3092–3098`.
- Updated the Section 6 regression to independently parse all three source
  tables and require the complete gap sequences:
  parity `[.003,.156,.311,.359,.337]`, counter
  `[.496,.455,.249,.116,.075]`, FSM `[.012,.076,.226,.238,.204]`.

### B4 — recursive local trust closure and reproducible extraction

- **P4:** `paper/main.typ:2133–2146`;
  `formal/lean/scripts/check_paper_core.sh:12–20`,
  `check_trusted_no_placeholders.sh:14–21`,
  `extract_theorems.lean:33–47`; inventory `:4–21`.
- Added `formal/lean/scripts/local_import_closure.sh`: visits the compiler's
  direct source dependencies recursively, includes the root, excludes Lake
  dependency sources under `.lake/`, sorts/deduplicates the local closure,
  and aborts on compiler failure or missing local sources. Both gates and
  the extraction wrapper use this enumerator, with checked command
  substitution rather than hiding failures inside process substitution.
- Replaced the eleven-module extraction allowlist with filtering of the
  elaborated recursive imported-module environment. Thus ordinary direct
  invocation no longer silently omits project modules. The wrapper accepts
  an explicit `PYTHON_BIN` rather than using a guessed login-shell Python.
- Regenerated inventory: **17 local source files, 857 named declarations,
  723 user-written**, including Lipschitz and E1HDefinition. Its extraction
  commit is `e58ae937`; no source under `formal/lean/ElmanProofs/` changed.
  Current manuscript scope/count is at `main.typ:2152–2172`, with project
  versus Mathlib/Lean trust boundaries explicitly distinguished.
- Final trust gate: all **17** project sources have no `sorry`/`admit`,
  explicit axiom/opaque declaration, or `native_decide` token matches.
  `lake build ElmanProofs.PaperCore` passes, 2,196 jobs, existing linter
  warnings only. This is a cached elaboration build, not a clean Mathlib
  bootstrap or an audit of foundational library axioms.
- Added source-named `scripts/test_local_import_closure.js`: recursive
  fixture imports, external exclusion, all five forbidden tokens in an
  indirect module, and compiler-error fail-closed cases. The assembly
  regression also independently parses actual local import statements and
  matches that closure against the compiler enumeration and inventory.

### B5 — reconstructed points/SEM, not seed records

- **P4:** `paper/main.typ:1653–1667`;
  `paper/figures/plot_expressivity_seeds.py:3–7,19–39,75–78,98–104`.
- **Checked before relabeling:** searched worktree results, paper/results,
  experiments, logs, staging, provenance, scratch filenames/archives, and
  primary checkout results, paper/results, experiments, benchmark_results,
  and logs. Inspected the archive member listing for
  `experiments/e99_1p3b_cma/artifacts/raw_eval_records.tar.gz`; it contains
  evaluation receipts, no S3/S5/permutation-named members. Parsed **1,774**
  candidate JSONs across the checked trees; no historical S5 final-extrema
  match or bound three-seed cohort was found. Numerical substring hits and
  newer S5 cohorts were not reused as provenance. This is a bounded local
  search, not a claim that inaccessible/off-machine archives cannot exist.
- Search receipts are in `P5-validation/B5-{candidates,primary-candidates,
  summary-matches,primary-matches,archive-members}.txt` and
  `B5-search-summary.json` in the scratch output directory.
- Caption, script docstrings/comments/title, and reproducibility discussion
  now label points and SEM as approximate reconstruction from rounded
  summaries, using the **3mean-min-max midpoint method**, clamped to the
  reported extrema; the rounded bar mean remains the published summary.
  Rounding/clamping, ceiling effects, lost seed order/identity, and inability
  to reanalyze exact seed-level uncertainty are explicit.
- Regenerated `paper/figures/s5_expressivity_seeds.png`; no numeric summary
  or historical result table was altered. Current caption is at
  `main.typ:1664–1680`; reproducibility limit at `2938–2948`.

## Named punchlist ledger

| Item | P4 original coordinates | Repair / current evidence | Remaining test/packaging boundary |
|---|---|---|---|
| P1 | `main.typ:647–659,2408–2410`; accurate identity `2435–2439` | Replaced boxed exact-restriction proposition and set G summary with adjacent/shared split cores, different decay placement; displayed both cores and limited exact equality to pre-decayed-state/unit-decay identity. Figure shading/caption audited too. Current `main.typ:655–666,730–750,2442–2446,2475–2495`. | No production-domain parameter mapping or whole-stack inclusion theorem added. |
| P2 | `main.typ:2425–2433,782–785,2249–2281` | Production-step caveat plus sets C/C′/G explicitly name unrestricted abstract real gate/key/state domains: unnormalized `(1,1)` keys, zero fillers, nonzero initial state, boundary all-one/zero gates, write gate 2 outside sigmoid range. Transition-factor comparison is not arbitrary reparameterized full-network separation. Current `main.typ:783–789,2254–2263,2300–2307,2442–2473`. | Production reachable-trajectory/gate-domain results remain unproved; no new production theorem requested. |
| P3 | `main.typ:2590–2595` | Related work now says saturation motivates persistent binding under conditional slot-level hold (`lambda >= 1`, write budget), not achieved production-default binding; set F summary aligned. Current `main.typ:2499–2500,2629–2636`. | Production-range hold and whole-layer lift remain open. |
| P4 | `main.typ:1085–1091,2390–2404`; source `:1368–1373,1431–1447` | Names actual `m2rnnPure 22 70 256` proxy; identifies trained depth-21/H370/N16 shape separately; describes its four field checks as inspection/arithmetic, not this exact-instance theorem or scheduling verification. Current `main.typ:1095–1103,2422–2439`. | No new exact trained-instance proof, kernel-independence verification, or cross-layer concurrency theorem added. |
| P5 | `main.typ:1747–1757,1774–1785,3204–3217`; selection caveat `1762–1764` | Retained tanh/identity flag intervention, removed every-confound/algorithm-recovery/required-condition rhetoric, qualified post-selected curves, described mechanism as tested hypothesis. Appendix names width/depth/head-count differences; addressing explanation is a hypothesis. Intro claim table, lineage paragraph and repeated interpretation audited. Current `main.typ:415–422,1762–1808,2490–2496,3183–3197,3269–3283`. | Unconditional causal estimate, exact algorithm recovery, and single-factor geometry ablation remain open. |
| P6 | `refs.bib:451–455`; P4 section 6 nine-key audit | Fetched real arXiv report: **Olmo Hybrid: From Theory to Practice and Back**, all 22 authors, 2604.03444. Removed unused ndm2026 placeholder; cited Mathlib as project documentation; added Widrow and Jaeger locators/corrected-copy note. Preserved independently supported Candy/Temes 1992 print metadata, Haykin, Lyons copyright 2011, Khalil. Updated all nine named reference-ledger flags with P4/P5 evidence boundaries. | Representative book equations not verified page-by-page; other references retain P3's stated metadata-only boundary. |
| P7 | `main.typ:4–8,17–22,2597–2620,482–488`; `refs.bib:177–184` | Added cited S5 architecture/SSM lineage explicitly distinct from group S5; tempered universal historical expressive-power loss; explicitly disclaimed format/anonymization compliance. Current `main.typ:492–499,2640–2647,2905–2907`. | Actual current TMLR template/anonymization packaging is **OPEN**, per supervisor scope ruling; no template invented. Length/compression editing remains separate. |
| P8 | `main.typ:3038–3041,3347–3364,2870–2886` | Fixed “is the a”; noncontainment cites set C's `emender_m2rnn_one_step_resource_separation_embeds`, not set D. All manuscript/figure-caption statistical-tie repeats now descriptive closeness, explicitly not seed-level/paired equivalence. Hash table/prose labels source `.pt` SHA; immutable HF revisions and converted-digest locator distinguished. Current `main.typ:1277,2913–2925,3100–3107,3390–3427,3819–3820`. | No equivalence analysis or multi-GB converted-weight binary readback performed; tiny release metadata is not weight validation. |

## Validation and artifacts

Full logs: `/home/erikg/emender-scratch/fix-fanout/P5-validation/`.
All final commands below passed from the worktree root unless noted.

| Command | Final result / receipt |
|---|---|
| `bash formal/lean/scripts/check_paper_core.sh` | PASS, 17 sources; `trust-final.log`. |
| `(cd formal/lean && lake build ElmanProofs.PaperCore)` | PASS, 2,196 jobs; existing flexible-tactic/unused-variable warnings; `lean-final.log`. |
| `node formal/lean/scripts/test_local_import_closure.js` | PASS recursive/exclusion/forbidden-token/compiler-failure fixtures; `closure-regression-final.log`. |
| `PYTHON_BIN=/home/erikg/emender/.venv/bin/python bash formal/lean/scripts/extract_theorems.sh` | PASS 857/723 declarations, 17 files; `extraction-final.log`; interpreter is existing Python 3.12.3. |
| `node paper/figures/validate_section6.js` | PASS all 7 wall, 5 control, 6 S5, 14 CMA rows plus two U3.1 placeholder rows and all three non-monotone gap sequences; `section6-final.log`. |
| `node paper/figures/validate_assembly.js` | PASS 13 body sections, three lettered appendices, 62 pages, 57 reference entries, new B1–B5/punchlist scope guards, independent recursive closure/inventory check and rendered source hashes; `assembly-final.log`. |
| `/tmp/emender-paper-p5-plot-env/bin/python paper/figures/plot_expressivity_seeds.py` | PASS regenerated disclosed approximate figure; `B5-figure.log`. |
| `bash paper/build.sh` | PASS `paper/Garrison_2026_Emender-a09cf845.pdf`, 62 pages; `pdf-build.log`. PDF is intentionally git-ignored by existing policy, not forcibly added. |
| `PDFJS_MODULE=/home/erikg/emender-scratch/fix-fanout/P3-tools/node_modules/pdfjs-dist/legacy/build/pdf.mjs node /home/erikg/emender-scratch/fix-fanout/P5-verification.mjs --pdf --refs` | PASS adapted independent P4 arithmetic/pooled-control/CMA/import/PDF checks: zero omitted modules, zero suspect tokens, zero out-of-page text, 31,766 words, all numeric cells retained; ten live reference locators HTTP 200. `independent-pdf-refs.log`. |
| `node /home/erikg/emender-scratch/fix-fanout/P5-render.mjs` | PASS rendered all 62 actual-PDF pages; seven contact sheets `contact-{1,10,19,28,37,46,55}.png`, manually inspected for figures/tables and gross clipping/overlap. No such defect found at contact-sheet scale; not a full-resolution typography certification. |
| `git diff --check`; `git diff --exit-code -- formal/lean/ElmanProofs`; `git diff --cached --name-only` | PASS whitespace, no changed theorem sources, nothing staged after commits. |

The P5 verification adaptation preserves the original P4 checker untouched;
only expected omission count (2 -> 0), current PDF path/page count (61 -> 62),
and associated descriptive messages change. Numerical and PDF value checks
are otherwise the referee's checks. The supplied original P4 script hardcodes
the *old* coverage failure and old PDF, so it is not a valid final artifact
acceptance command unchanged. Both versions remain discoverable.

### Environment/iteration notes (not hidden final failures)

- Host is `lambda01`, not Frontier. Attempted canonical activation returned
  69 because the Frontier module command is absent. No bare or guessed
  Python was used: extraction used the existing named Python 3.12.3 venv.
- That existing venv lacked matplotlib; the first plot attempt failed. A
  separate temporary venv based on that interpreter was created with `uv`
  and numpy/matplotlib installed; final generation passed. Repository
  dependency files and the shared environment were not altered.
- Initial recursive enumeration inadvertently traversed `.lake` dependencies
  and timed out (300/600 seconds). It was corrected to exclude third-party
  Lake sources; final actual closure takes about 24 seconds. Regression
  fixtures explicitly test this exclusion and fail-closed behavior.
- A transient syntax error in new text guards and a guard-detected residual
  statistical-tie caption were corrected before committing. All final
  regression commands pass. No timed-out test was reported as green.

### Artifact identity

Source/PDF commit: `a09cf845` (the report-only follow-up does not alter it).

- `paper/main.typ` SHA-256:
  `c1b11cdf2b87a409b8abc053f6c3208efc18090bd80da1a96169a71892c54810`.
- `paper/refs.bib` SHA-256:
  `65e0c27a8f2fd1c2073de036781a27ce9f14056ab198cb49dcb7305491049e15`.
- `paper/Garrison_2026_Emender-a09cf845.pdf` SHA-256:
  `61d288e4ff85875ac95013d151ba3793f7aab12e9f13377c5695c12123d9660f`.
- `paper/data/lean-trust-gate-theorems.json` SHA-256:
  `2e3ed61744c20b6f3d7056a63ef69e8c9b83795a2a3d1abbd3b3e87ffee263b3`.

## Commit/push ledger

Each blocker was committed separately and immediately pushed to
`origin/paper-1p3b-spine`; two subsequent audit/regression commits were likewise
pushed. No force push, staging of P4 deliverables, or unrelated edits.

1. `3f8c5f7f` — B1 synthetic state-pass scope.
2. `ed77c7ea` — B2 restricted complexity/symmetric-group scope.
3. `f7182e8b` — B3 non-monotone extrapolation.
4. `a197de82` — B4 recursive gate/extractor/inventory and fixtures.
5. `2a181db2` — B5 reconstruction disclosure and regenerated figure.
6. `e58ae937` — P1–P8 textual punchlist/repeated-interpretation audit.
7. `a09cf845` — new scope/gap regressions and refreshed extraction receipt.
8. This report is a separate report-only commit/push.

## Changed files and added/updated tests

- `formal/lean/scripts/{check_paper_core.sh,check_trusted_no_placeholders.sh,
  extract_theorems.lean,extract_theorems.sh,local_import_closure.sh,
  test_local_import_closure.js}`.
- `paper/main.typ`, `paper/refs.bib`, `paper/review/P3-reference-audit.md`.
- `paper/data/lean-trust-gate-theorems.json`.
- `paper/figures/{plot_expressivity_seeds.py,s5_expressivity_seeds.png,
  validate_assembly.js,validate_section6.js}`.
- `P5-report.md` (also copied to the supplied scratch output directory).

Tests added/updated: `formal/lean/scripts/test_local_import_closure.js`,
`paper/figures/validate_assembly.js`, `paper/figures/validate_section6.js`.
Scratch-only verification/rendering/search receipts are not product source.

## Open boundaries and next step

Historical exact 8 M seed records remain unbound; reconstruction is now
honest, not raw-observation evidence. Actual compute matching, production
key/gate/decay transfer, exact algorithm recovery and an unconditional causal
estimate are not proved by these repairs. F1/G1–G3/U3.1, single-seed 1.3 B
replication, held-out/current-pair evaluation, feasible matched curricula,
application deployments, scale/backend validation, a single-factor stability
geometry ablation, and a paired/seed-level equivalence analysis remain open.
Journal-specific current-template/anonymization packaging is an explicitly
separate task. No resilient DiLoCo design requirement is claimed validated by
this manuscript/proof-artifact-only pass.

Recommend an independent repaired claim-to-proof/data/PDF review, then a
separately authorized journal packaging pass. Nothing here asserts reviewer
approval or TMLR administrative compliance. Final no-staged-files check is
required after the report-only commit; only pre-existing untracked P4 review
files should remain.
