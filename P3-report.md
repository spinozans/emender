# P3 — whole-manuscript assembly and correctness pass

Worktree `/home/erikg/emender-paper`, branch `paper-1p3b-spine`.
Read `taskP3-assembly.md`, P1/P2 reports, all of `paper/main.typ`, GAPS-D2,
and the actual load-bearing Lean statements before assembly. No training,
Slurm submission, Python invocation, native kernel build, or resilient-training
behavior change was performed. Lean proof sources and frozen result files are
unchanged. Independent reviewer acceptance is still required.

## Milestones and delivery

1. **`5a90482c`, pushed:** numbered applications and label-based body references;
   lettered appendices; duality-first TMLR abstract; exact-scope formal spine;
   15-file/790-declaration refreshed inventory; current OUTLINE authority.
2. **`da64357d`, pushed:** honest limitations, with significance and closure
   criteria for G1–G3, F1, single-seed evidence, U3.1, applications, evaluation,
   scale/comparators, and backend scope.
3. **Commit containing this report, pushed:** bibliography metadata fixes and
   all-entry audit; final scope/cutoff/seed-selection corrections; checkpoint
   hash table overflow repair; durable assembly regression; complete PDF checks.

Build command: `bash paper/build.sh`. The final clean, commit-stamped PDF is
`paper/Garrison_2026_Emender-$(git rev-parse --short=8 HEAD).pdf` (substitute the
hash in the filename, rather than treating this as a literal filename).
PDFs are intentionally gitignored, not committed. The validation scripts and
bibliography ledger are committed. External receipts/logs/renders are under
`/home/erikg/emender-scratch/fix-fanout/P3-*`.

## Section map and size

Final Typst counters checked, not inferred from source order:

| Section | Title | Starts on page |
|---|---|---:|
| 1 | Introduction | 1 |
| 2 | Background | 6 |
| 3 | Architecture | 10 |
| 4 | Multi-Programming and Systems | 14 |
| 5 | Language-Modeling Results | 17 |
| 6 | Expressivity Results | 22 |
| 7 | Repeated nonlinear maps in the wild | 29 |
| 8 | Formal Results | 32 |
| 9 | Related Work | 38 |
| 10 | Limitations | 40 |
| 11 | Conclusion | 42 |
| 12 | Testable Predictions | 43 |
| 13 | Future Work | 45 |
| A | Lineage of E63 → E88 → E97 | 47 |
| B | Predecessor 1.3 B cohort | 48 |
| C | Held-out bits-per-byte measurement (predecessor cohort) | 55 |
| — | References, unnumbered | 57 |

**61 PDF pages; 31,168 rendered whitespace tokens.** Word-count method:
PDF.js text extraction, joining text items with spaces and counting nonempty
whitespace-delimited tokens, including headings, captions, tables, math
fragments, bibliography, and page numbers. This is a reproducible rendered-word
estimate, not an exact prose-only count. PDF.js and Typst agree on page count.
There are 7 image figures and 13 numbered table figures, plus the uncaptioned
predecessor setup table. The existing Arkheion template is retained; TMLR
positioning does not claim TMLR house-format compliance.

## Claim-versus-artifact spot checks (complete list for this pass)

1. **D1/executed depth.** Read `CompositionalDepthDuality.lean`:
   `e97Step_eq_tanh_comp`, `e97StateAfter_eq_compose`, and
   `e97_compositional_depth_grows_in_time` describe appending the actual update,
   not minimal required circuit depth. `gdn_state_channel_depth_bounded_in_time`
   gives `A,C` for every initial state conditional on the schedule. Set H,
   abstract, results, and applications now retain this exact scope.
2. **Applications family.** Matched `E97Step` `(k,b,w,v)` at fixed lambda to
   the repeated-map prose. The coupled specialization explicitly includes
   lambda in its input tuple; production input-dependent decay is distinguished
   from the fixed-lambda formal fold. Autonomous logistic dynamics have no
   injection; ordinary normalized LMS remains affine; FISTA needs augmented
   momentum/schedule state. All six surrogate predictions remain OPEN.
3. **D2/routing.** Read `S5GDNBridge.lean` and GAPS-D2. The headline
   `gdn_s5_realization_requires_per_token_action` requires a token-only schedule
   and a reachable-state-separating decoder. Its conclusion is factorization,
   right multiplication on the image, and affinity; only a token-independent
   schedule is ruled out. The manuscript does not infer a circuit-size/depth
   bound or impossibility for the whole GDN2-MLP stack.
4. **F1/lookup overstated as recurrent realization.** Read the full statement
   of `NDMRealizesS5.emender_realizes_s5_tracker`: it queries a preloaded
   orthonormal generator table at d=12, decodes a generator, then externally
   multiplies a supplied group state. No tanh recurrent trajectory appears in
   the theorem. Corrected every manuscript realization interpretation and
   added numbered Open item F1. Proof sources remain unchanged.
5. **Finite-k separation.** Read `MultiStepSeparation.lean`, especially
   `iterUpdate_emender_kStep_entry`, `tanhIter_ne_zero_of_ne_zero`, and
   `emender_m2rnn_k_step_separation`. The distinguishing entry is tanh^k(-1)
   versus zero from `lowerLeftState` in the 2D external-forget resource class.
   Nonzero for finite k is not compounding, a uniform margin, finite-precision
   robustness, or an S5-generator theorem. Removed those interpretations.
6. **Resource relation.** Checked the one-step theorem's K≥2/V≥1 embedding
   scope and the separately provided read-then-delta embedding. A
   noncontainment witness does not prove strict inclusion of entire update
   families, minimal parameter count, or that extra state is the only remedy.
   Narrowed the contribution, removed the informal parameter-efficiency
   corollary, and qualified the future partial-order proposal.
7. **Recognizer floor/complexity.** Checked `S5Inseparability`'s named 120-state
   floor against its use. It is architecture-agnostic, not a learned-map
   capacity proof. Barrington completeness does not prove TC0≠NC1, and affinity
   alone does not rule out group actions. Removed blanket linear-state
   impossibility throughout the body and predecessor interpretations;
   reconciled the discussion with G3. No unproved circuit-class separation is
   assumed.
8. **Latching/overwrite transfer.** Checked conditional set F against production
   decay: `emender_latch_holds_by_default` assumes lambda≥1, production decay is
   below one. Scoped introductory and ingredient-level statements accordingly.
   Exact orthonormal overwrite belongs to `linearDeltaWrite` before tanh with
   unit decay/all-one gates, not the general nonlinear production step.
9. **Architecture terminology.** Corrected the ablation table's false claim
   that GDN lacks delta writes; GDN and Emender share delta correction, while
   M²RNN and Emender share temporal nonlinearity. Removed causal elimination
   language. Corrected stale E88 no-write-gate prose for E97's split gates;
   SiLU values are activated, not inherently bounded. Production geometry and
   input processing are not silently erased by the shared-core comparison.
10. **Frozen Section 6 grids.** `validate_section6.js` independently parses
    committed JSON/CSV and compares rendered Typst cells: 7 modular-quadratic
    rows, 5 control arms, 6 S5 rows, 14 CMA snapshots, and 2 U3.1 winner rows.
    Checked common cap/recipe versus unequal realized steps, terminal versus
    interim GDN2 accuracy, non-chance plateau, CUDA-only scope, and both verbatim
    S5 negatives. No empirical cell or result artifact was edited.
11. **8 M S5 numbers.** Read `paper/ndmpapernotes.md` lines 153–180: the 0.7918 /
    0.3552 / 0.2157 / 0.1698 training-length means and 0.4158 / 0.2150 E88
    extrapolation entries agree with the manuscript. State-count headroom is
    now only motivation for an inductive-bias hypothesis, not a capacity or
    recurrent-realizability theorem.
12. **Small-scale map selection.** Read `grok_symmetric_width/RESULTS.md`:
    the p=256/dim1024 length curves (0.982→0.981, 0.619, 0.560) condition on
    `final_test>=0.9`, with n=2/1/4, not all four seeds in each arm. Added that
    disclosure without changing the numeric finding or claiming all confounds
    disappear. The broader confirmation report remains cited, not rerun.
13. **Exact logged-loss cutoff.** Read `GDN2_OPS_SNAPSHOT_20260728.md`: 92,037
    common records end at step 2,300,925 / 150,793,420,800 tokens, five steps
    before the E97 checkpoint at 2,300,930 / 150,793,748,480. Corrected prose
    and caption from exact-terminal readout to last-common-log readout. The
    rounded 150.79 B table values and window-dependent deltas are unchanged.
14. **Winner provenance/U3.1.** Read `docs/repro/lb_gdn2_mlp_20260612/best.json`:
    104 evaluations, 13 generations, winner eval86, fitness 5.894941..., and
    1,286,713,448 parameters agree. Generation-zero fitness is absent; no
    earlier sweep was substituted for the carried-in winner.
15. **Checkpoint identity/table fit.** Rendered wrapped SHA-256 cells are
    compared byte-for-byte to the two published authorities by the new
    assembly regression. PDF.js initially found a right-edge overflow on
    page18; fixed fractional columns and wrapped hashes, then verified zero
    out-of-page text and visually inspected the repaired table.
16. **All citation entries.** Audited all 58 BibTeX keys: 49 independent metadata
    receipts; 9 individually flagged in `paper/review/P3-reference-audit.md`.
    Corrected Schedule-Free's title to *The Road Less Scheduled* from arXiv
    2405.15682, ParaRNN's named authors from arXiv 2510.21450, and added verified
    DOI/arXiv locators for classical/benchmark entries. No invented metadata
    filled an unresolved entry. Metadata retrieval is not verification of all
    attributed scientific claims or every edition/year/author detail.

## Validation and rendering evidence

- `bash formal/lean/scripts/check_paper_core.sh` — PASS, 15 project sources,
  no placeholder proofs/explicit axiom/opaque/native_decide. The task's script
  lives here, not at the repository-root `scripts/` path.
- `(cd formal/lean && lake build ElmanProofs.PaperCore)` — PASS, 2,196 jobs;
  existing tactic/linter warnings, no proof errors (`P3-lean-build.log`).
- Lean inventory extraction — PASS, 790 declarations, all 15 project files.
  Existing `extract_theorems.lean` hardcodes 11 modules; running it unchanged
  omits four current modules. A temporary external copy replaced only
  `trustGateModules` with the `--src-deps`-enumerated list, then ran
  `EXTRACT_OUTPUT=... EXTRACT_SRC_ROOT="$PWD" lake env lean <copy>`.
  Node wrapped its JSON with the original unchanged-proof source commit
  `b643a544`. Copy, inner JSON, file list, and log remain in scratch as
  `P3-extract-theorems.lean`, `P3-lean-declarations.json`,
  `P3-trusted-files.txt`, and `P3-lean-extract.log`.
- `node paper/figures/validate_section6.js` — PASS (all result cells/caveats).
- `node paper/figures/validate_assembly.js` — PASS: 13 consecutive body
  counters, 3 reset appendix counters, label references, exact theorem scopes,
  G1–G3/F1/U3.1, corrected cutoff/seed-selection disclosures, 15-file inventory,
  rendered checkpoint hashes, and all bibliography ledger entries.
- `bash paper/build.sh` — PASS, full PDF, no Typst diagnostics/citation errors.
- `typst query paper/main.typ figure` — PASS: 7 images and 13 table figures.
- `typst compile paper/main.typ '.../P3-page-{p}.png' --format png --ppi 60`
  — PASS, all 61 pages rendered. Inspected whole-document contact sheets and
  individual result/figure/table pages: 6, 18, 21, 24–28, 50, 53, 54, 57;
  architecture/taxonomy and other tables also inspected in the contact sheets.
  Figures present, table grids fit, no clipping observed after hash repair.
- `node .../P3-pdf-check.mjs <PDF>` — PASS: 61 pages, 31,168 rendered tokens,
  zero out-of-page text. Text bounding boxes alone do not prove absence of
  internal cell overlap; the visual inspections supply that separate check.
- `git diff --check` — PASS. Final commits pushed; no staged files remain.
  Scratch-only checks had initial probe-label/parser/module-list and utility
  path errors; corrected before final PASS. No failed experiment was hidden.

## Independent review focus / things not resolved

1. **F1:** actual recurrent tanh-tracker realization remains unproved; audit the
   narrowed lookup statement and every downstream interpretation.
2. **G1–G3:** no circuit size/depth bound, no state-dependent gate model, and no
   constrained affine/reflection realizability decision. In particular the
   formal arbitrary decay matrix is broader than production scalar decay.
3. **Finite-k and resource scope:** no uniform margin, no general KV k-step
   Emender-side proof, no S5-specific T(d), no strict family-inclusion or
   minimal-parameter theorem; production latching transfer remains open.
4. **Empirical inference:** single-seed 1.3 B pair/probes, unequal plateau stops,
   interim/terminal variation, curriculum OOM, conditional successful-seed
   small-scale length curves, and no causal attribution of S5 decay to capacity
   or non-solvability. Recurrent learning failure is a hypothesis, not proved.
5. **U3.1 / evaluation:** missing winner generation series, 150 B-pair held-out
   bpb and downstream panels; training-log pairing is not an equivalence test.
   Verify the newly explicit last-common-log versus checkpoint cutoff.
6. **Applications/scale/backends:** six separation predictions untested;
   energy, deployment and safety unmeasured; no matched 3–7 B pair, current
   raw-write/transformer long-run arm, or paired ROCm evidence.
7. **Bibliography:** unresolved keys are `ndm2026` (unused/unassigned),
   `widrow_hoff_1960`, `mathlib4` (documentation page retrieved, publication
   metadata not established), `olmohybrid2026` (no persistent locator),
   `candy_temes1992`, `haykin2002`, `lyons2011`, `jaeger2001`, `khalil2002`.
   These are explicit metadata-review flags, not evidence of fabrication;
   resolve before submission, especially OLMo's in-text attribution.
8. **Tooling/length:** repository extraction helper still has an obsolete
   module allowlist; the temporary corrected extraction is fully retained but
   proof/tool source edits were not authorized in this pass. The 61-page draft
   also merits an editorial length review; no unrequested manuscript pruning
   or journal-template replacement was performed.

No unresolved trust-gate or PDF-build blocker. Research/evidence/citation gaps
are disclosed, not closed by assembly. Recommended next step: independent
whole-manuscript claim-vs-proof/data review, with F1/G3/finite-k scope and
bibliography flags prioritized before TMLR submission.
