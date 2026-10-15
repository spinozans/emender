# P16 — merge reader-first voice, positive assertions, genre reorder

## Contract and scope

Started at the worktree's own base `f17b3be2` on `paper-1p3b-spine` (HEAD
already equaled it), then fetched the operator's branch
`origin/editorial/reader-first-paper-20261014` (tip `fba15ce5`) and worked
from that tip as the reader-first source. Three operator rulings of
2026-10-15 govern this pass:

1. **Positive assertions** — negation-framed claims go affirmative, scope
   attached in the same sentence; the one exception is a negation that *is*
   the finding (kept verbatim).
2. **Genre reorder** — theorems before experiments, per the target skeleton.
3. **Reader-first voice import** — narrative framing and application
   grounding imported only where the canonical's own evidence supports it.

Constraints honored: every P8-accepted claim scope is preserved (the pass
changes how and where a claim is voiced, never what it claims); equations,
tables, figures and bibliography bytes are untouched; Lean citations remain
bound to their exact theorem names; the Testable Predictions and Limitations
sections survive. One commit, pushed to `origin/paper-1p3b-spine`.

Task: `/home/erikg/emender-scratch/fix-fanout/taskP16-merge-reader-first.md`.

## Ruling 1 — positive-assertion table

Each row is one P14 judgment-call entry (J1–J32, the primary input list) or one
additional negation-framed claim caught in the sweep (E). "Ruling" is the
hedge-vs-finding distinction the task requires. All old line numbers are
`f17b3be2:paper/main.typ`; the exact replay is
`paper/tmlr/validation/reader-first-merge-revision.diff`.

| ID | Section | Old negation → new affirmative | Ruling |
|---|---|---|---|
| J1 | Abstract | `not invariant extrapolation` → `over the tested lengths` | hedge-negation → converted |
| J2 | Abstract | `they do not lower-bound minimal circuit depth or prove a full-stack impossibility` → `minimal circuit-depth lower bounds and a full-stack impossibility remain open` | hedge-negation → converted |
| J3 | Intro | `under the addressing-bottleneck hypothesis … may benefit` | hypothesis modality / P8 scope → **kept** |
| J4 | Intro | `not a proof that the tanh recurrence implements the full tracker` → `proves the lookup bridge; … still open as Open item O1` | hedge-negation → converted |
| J5 | Intro | `not the production decay below one` / `not a proved default behavior` → `while production decay sits below one` / `remains open` | hedge-negation → converted |
| J6 | Intro | heading `What we claim, and what we do not` → `Claims, evidence, and scope`; `what the same result does *not* license` → `the boundary the same result observes` | hedge-negation → converted |
| J7 | Background | `is not a universal historical claim of declining expressive power` → `describes how an update parallelizes at fixed width and precision, a per-architecture property distinct from any lineage-wide historical trend` | hedge-negation → converted |
| J8 | Background | `alone does not prove inability … affinity alone does not prohibit` → `places restricted linear SSMs in that class; an inability … would require the still-unproved separation …; Affinity admits a finite-group action` | hedge-negation → converted |
| J9 | Background | `not finite-length accuracy … does not locate a model in a smaller circuit class` → `The complexity-theoretic question is exact recognition over all words; training-length learning failure is an empirical observation about SGD, separate from circuit-class membership` | hedge-negation → converted |
| J10 | Background | `within the restricted transition analyses motivating this grid, not from affinity alone` → `within the restricted transition analyses motivating this grid` | hedge-negation → converted |
| J11 | Architecture | `It is not a full-stack implementation.` → `It is the per-head core, bound to the full stack by limitation G2` | hedge-negation → converted |
| J12 | Architecture | `motivate revisable memory rather than guarantee persistent bindings` → `motivate revisable memory; persistent bindings in production E97 remain a conditional property` | hedge-negation → converted |
| J13 | Systems | `a source-portability route rather than evidence of cross-backend parity` → `a source-portability choice; cross-backend parity evidence would require ROCm measurement` | hedge-negation → converted |
| J14 | Systems | `a conservative lower bound` → `a lower bound` | hedge word → converted |
| J15 | Language Modeling | `literally identical recipe` | precision intensifier, no negation → **kept** |
| J16 | Expressivity | `not a consequence of the lookup theorem` → `supported by this headroom …; deriving it from the lookup theorem remains Open item O1` | hedge-negation → converted |
| J17 | Expressivity | `not an unconditional four-seed mean` → `the reported mean is conditional on that subset` | hedge-negation → converted |
| J18 | Expressivity | `reflects inductive bias rather than raw state count` → `reflects a difference in inductive bias; raw state count is controlled by that headroom` | hedge-negation → converted |
| J19 | Applications | `a structural family, not a claim that the systems below share E97's weights…` → `a structural family: the systems below share its update form, while each keeps its own weights, learning rule, and computational difficulty` | hedge-negation → converted |
| J20 | Applications | `*OPEN* a prediction not yet established` → `*OPEN* a prediction open for future test` | hedge-negation → converted |
| J21 | Formal Results | `not a missing proof … but a stronger, currently unproved architecture claim` → `a stronger, currently unproved architecture claim, beyond the already-proved lookup identity` | hedge-negation → converted |
| J22 | Formal Results | `the *ceiling reachable from …*`; `what it does and does not establish` → `the *strongest separation reachable from …*; Its scope is named below` | hedge-negation → converted |
| J23 | Formal Results | heading `Resource scope, not a parameter-efficiency lower bound` → `The resource scope of the separation` | hedge-negation → converted |
| J24 | Related Work | `not an achieved production-default property` → `a conditional production property` | hedge-negation → converted |
| J25 | Limitations | heading `Formal gaps: G1–G3 and O1` → `Formal scope: G1–G3 and O1`; the G1–G3/O1 body statements are the disclosed gaps | heading converted; body **kept** as scope disclosures |
| J26 | Conclusion | `rather than priority, universal nonlinear superiority, or an unproved circuit-class separation` → `priority, universal nonlinear superiority, and an unproved circuit-class separation lie outside it` | hedge-negation → converted |
| J27 | Testable Predictions | `conjecture` | predictions modality; section survives as-is → **kept** |
| J28 | Future Work | `the gap could be … either … or` → `the gap reflects either a genuine difference … or the residue …` | hedge-negation → converted |
| J29 | Appendix A | `a stability hypothesis, not an isolated cause because width, depth, and head count also changed` → `a stability hypothesis under the tested width, depth, and head count, which also changed` | hedge-negation → converted |
| J30 | Appendix B | `neither "E88 length-generalizes on $S_5$" nor "E88 fails $S_5$"` → `the delta update learns $S_5$ at the trained length and leads at every length on that $S_5$ task` | hedge-negation → converted |
| J31 | Appendix B | `does not establish exact algorithm recovery or exclude finite-context heuristics … not a theorem that non-solvability causes the difference` → `exact algorithm recovery and finite-context heuristics remain open. The solvable controls are useful diagnostics for the non-solvability difference` | hedge-negation → converted |
| J32 | Appendix C | `— *not* the fixed $bytes/token = 3.92$ constant` → `whereas the train-loss curve uses the fixed $bytes/token = 3.92$ constant` | hedge-negation → converted |
| E1 | Abstract | `define the applications scope, not demonstrated deployments` → `define the applications scope as structural families, with no demonstrated deployments` | hedge-negation → converted |
| E2 | Formal Results | heading `Theorem set B: generator-table lookup, not a recurrent tracker` → `generator-table lookup and the recurrent-tracker target` | hedge-negation → converted |
| E3 | Formal Results | `The core proves neither … nor …`; `is not robustness at finite precision`; `not a default persistence guarantee` → `… remain open targets`; `holds at exact precision`; `a conditional persistence property` | hedge-negation → converted |
| E4 | Architecture | `Two checks motivate, rather than prove, a learning-efficiency reading` → `Two checks support a learning-efficiency reading` | hedge-negation → converted |

**Finding-negations kept verbatim** (the exception — the negative *is* the
result), distinguished from the hedges above:

- Abstract: `The negative transfer is: "The 8M S5 separation does NOT replicate
  at 1.3B on this pair."`
- Introduction / Expressivity Results: `The $S_5$ separation does *not*
  replicate: GDN2-MLP is ahead in the long-horizon probe` (@tab_s5_boundary).
- Conclusion: `The negative transfer is:` + the verbatim S5 failure and GDN2
  ordering.
- Background: `a failure to learn at training length` phrased as the
  observed 8 M decay; the GDN2 OOM stage; the measured loss null.

**Negations reviewed and retained as technical contrast or mandatory scope**
(not hedge-negation; enumerated so the reviewer can confirm the sweep): the
literal axis/type contrasts (`parallel across width rather than across time`,
`an outer product $k v^T$ rather than a delta correction`, `graceful rather
than invariant`, `occupancy, not peak arithmetic`) and the P8-ceiling
non-claims that `validate_assembly.js` and `verify.mjs` pin as load-bearing
(`No strict separation of TC#super[0] and NC#super[1] is known or assumed
here.`, `No minimal parameter-count corollary is claimed.`, `G1 — no
per-token circuit-size or depth lower bound.`, `it is not a causal
ablation`). Removing these would either damage precision or breach the P8
scope ceiling, so they are kept and listed rather than inverted.

## Ruling 2 — reorder map (old → new section order)

| Old | New | Section |
|---|---|---|
| 1 | 1 | Introduction |
| 2 | 2 | Background |
| 3 | 3 | Architecture |
| 4 | 5 | Multi-Programming and Systems |
| 5 | 6 | Language-Modeling Results |
| 6 | 7 | Expressivity Results |
| 7 | 8 | Repeated nonlinear maps in the wild |
| 8 | 4 | **Formal Results** (moved before the experiments) |
| 9 | 9 | Related Work |
| 10 | 10 | Limitations |
| 11 | 11 | Conclusion |
| 12 | 12 | Testable Predictions |
| 13 | 13 | Future Work |
| A–C | A–C | Appendices (unchanged) |

The Formal Results block (its `// ── 8.` banner through the blank line before
Related Work) is moved whole before `= Multi-Programming and Systems`; no
content is deleted. Content edits from the move are limited to: the intro's
"paper proceeds" pointer paragraph, the five in-file `// ── N.` banner
renumbers, and two transition sentences (`The architecture fixes the
mechanism; this section states what it provably computes.` opening Formal
Results; `The formal results establish the mechanism; the next question is
whether a serial nonlinear recurrence can be trained to competitive scale at
all.` opening Systems). Cross-references use stable `#ref(<sec:…>)` labels and
were unaffected; `validate_assembly.js`'s expected label order was advanced to
the new genre order. Auto-numbering follows: Formal Results is §4, Systems
§5, Language-Modeling §6, Expressivity §7, Applications §8.

## Ruling 3 — reader-first import inventory (fact-checked)

Source: `origin/editorial/reader-first-paper-20261014`, tip `fba15ce5`
(`paper/main.reader-first.typ`).

| Import | Site | Fact-check against canonical |
|---|---|---|
| (a) Motivation arc: balance/motion/feedback, conversation and perception, per-input time budget, carried-memory path | new opening paragraph of Introduction | Canonical already establishes fixed-width/fixed-depth nonlinear recurrence with bounded per-update work (#ref(<sec:arch>), #ref(<sec:systems>)); re-voiced, no new claim, **no new citation** |
| (b) Application grounding: `Balance, motion tracking, and movement correction offer an intuitive starting point: each new observation revises an estimate that guides the next action.` | Applications opening | Supported by the canonical applications family `S_{t+1}=F(S_t,u_t)` and the feedback-control EXHIBIT; motivation only |
| (c) Connective tissue | Formal-Results and Systems opening transitions; the reverbalized Ph.D.-level prose above | Re-voices canonical transitions; no scope change |

**Not imported** (canonical evidence or bibliography does not support it, so
per the ruling it is excluded): the reader-first scratchpad / chain-of-thought
citations (`@nye2021scratchpads @wei2022chain`), the Transformer citation, and
the Enformer genomic-model citation — none exist in the canonical
`paper/refs.bib`, and the bibliography is byte-frozen. The reader-first
"Conversation and perception" and "Biological dynamics" subsections are
therefore folded into the applications motivation sentence rather than added
as new cited subsections; the canonical Applications section's EXHIBIT /
ANALOGY / OPEN structure is preserved. The canonical's claim scopes remain the
ceiling; the imported language only re-voices.

## Ruling 4 — E97 focus and the demoted predecessor material

Operator ruling 2026-10-15 (relayed by the supervisor, superseding any narrower
reading of Ruling 3): the merged manuscript focuses on the current E97 with
GDN2-MLP as the matched control, and the E63→E88 lineage and predecessor-1.3B-
cohort material demote out of the main line into a compact development-history /
source register appendix. No P8-accepted claim loses scope; load-bearing
predecessor-cohort evidence keeps a minimal citation; the paper remains the
referee-accepted scope set.

After escalation, the supervisor chose **Option A** (relocate, no evidence
loss): deleting accepted evidence would change what the acceptance covers and
would require re-fitting the acceptance guards to permit the deletion, neither
of which this programme does. So the demotion is a relocation inside the
manuscript, not a deletion — every predecessor-cohort table and number remains,
now in the register. `verify.mjs`'s hardcoded samples are untouched.

**Register consolidation.** The three former level-1 appendices —
`Lineage of the E63 $arrow$ E88 $arrow$ E97 experimental program`,
`The predecessor 1.3 B cohort (E88 era)`, and `Held-out bits-per-byte
measurement (predecessor cohort)` — become level-2 subsections of one level-1
appendix `Development history and source register <sec:appendix>`. The register
opens with the reader-first draft's framing text: “The project developed
through successive recurrent-memory designs, implementation changes, and
increasingly controlled comparisons. This register records that process and
points to its source material. The empirical study in this paper is the current
E97/GDN2 pair; the subsections below preserve the development trail.” Labels
`sec:appendix_cohort` and `sec:appendix_bpb` are retained on the subsections, so
every in-body `#ref` still resolves.

**Body demotions (relocated into the register).** Two predecessor-cohort
blocks that had stood in the main line move into the register under new
level-2 subsections, leaving one-line pointers to `#ref(<sec:appendix>)`:

| Former site | New site | Pointer left in body |
|---|---|---|
| Systems, `Measured throughput and utilization (CUDA only)`: the E88-predecessor occupancy paragraph (median 100% util, 7,492 tokens/s) | register `== Predecessor-cohort systems measurements` | `Predecessor-cohort occupancy, MFU, and the linear-scan kernel comparison are recorded in #ref(<sec:appendix>).` |
| Systems, same subsection: the MFU (15.7% lower bound) and FLA linear-scan (8,248 tokens/s, ≈91%) paragraphs | register `== Predecessor-cohort systems measurements` | (same pointer) |
| Language Modeling, `Downstream capability panels: a null, on the predecessor cohort` (with @tab_downstream) | register `== Downstream capability panels: a null, on the predecessor cohort` | `Predecessor-cohort downstream capability panels are recorded in #ref(<sec:appendix>).` |

The E97/GDN2 pair-geometry throughput paragraph stays in Systems as the main-line
measurement; the register retains the predecessor evidence it once sat beside.

**Main-line claims load-bearing on predecessor-cohort evidence (kept with a
minimal citation, per Ruling 4c).** (1) The claim-table row “Group-state
tracking is protocol-dependent” cites “8 M and predecessor GDN probes decay at
length; the current GDN2-MLP pair reverses the predicted ordering” — the
predecessor GDN probe evidence now lives in the register (`#ref(<sec:appendix_cohort>)`).
(2) The Systems width-axis throughput claim now points to the register instead
of restating the E88 occupancy/MFU/kernel measurements. (3) The Language
Modeling downstream-capability null now points to the register. (4) Related Work
and Future Work retain their predecessor-cohort pointers by label. No claim text
or number was removed.

**Guard update.** `paper/figures/validate_assembly.js` was advanced for the new
appendix count (13 numbered body sections + 1 register appendix) and the
section-label list; its phrase list and `verify.mjs`'s sample set are unchanged.

## Validation evidence

Durable receipts are under `paper/tmlr/validation/`; the exact source diff is
`reader-first-merge-revision.diff` (zero context) and the guard delta is
`p16-guard-delta.json`. The canonical guard first rejected the edited source
against P15's digest (expected negative check); it was not disabled or
relaxed. `canonical-baseline.json` retains the full P13/P14/P15 provenance,
records the former digest as `p15MainTypSha256`, adds `p16BaseCommit:
f17b3be2`, and appends the P16 directive/report/diff to `revision`.

| Command | Result / receipt |
|---|---|
| `node paper/tmlr/convert.mjs` | PASS: 98 headings, 20 figure/table captions, 21 amsthm statements; bibliography byte-identical (`convert.txt`) |
| `typst compile paper/main.typ paper/current.pdf` | PASS: **66 pages** (was 65) |
| `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` | PASS: anonymous/identified PDFs **58 / 58** pages; no missing glyphs, unresolved refs, overflows or errors (`build-output.txt`, `submission-build.txt`, `preprint-build.txt`) |
| `bundle-arxiv.sh`; fresh extraction; tectonic; Poppler `pdftotext -raw`; `cmp` | PASS: standalone archive rebuild byte-identical to preprint (`arxiv-build.txt`, `arxiv-compare.txt`) |
| `verify.mjs` (Poppler env) | PASS: 20 samples × 3 PDFs, **2600** long source fragments, **707** math expressions, **441** attachments, 8 S primes per PDF; source/bibliography/identity guards (`verify-output.txt`, `drift-check.json`) |
| `node --test verify-symbols notation-audit author-rulings` | PASS: **20/20** (`symbol-tests-output.txt`) |
| `node paper/figures/validate_assembly.js` | PASS: 13 body sections, 1 register appendix, 57 references, G1–G3/O1/U3.1, 17-file inventory and checkpoint bindings (`assembly-check.txt`) |
| `bash formal/lean/scripts/check_paper_core.sh` | PASS: 17 local sources; no placeholder/axiom/opaque/native_decide (`p16-lean.txt`) |
| `(cd formal/lean && lake build ElmanProofs.PaperCore)` | PASS: 2,196 jobs, cached build; existing linter warnings only |
| `test_local_import_closure.js` | PASS: recursive imports, external exclusion, forbidden tokens, fail-closed fixtures |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS: all 15 paths (`p16-checksums.txt`) |
| `git diff --check` | PASS |

**Anonymity build**: `submission.pdf` is the anonymous variant; `verify.mjs`
asserts `Anonymous authors`, an anonymous author field, and the absence of any
identity token (`spinozans`, `hypervolu.me`, the author email, `Poietic`,
`UTHSC`, ORCID) while the identified preprint retains them. It still passes.

**Math/tables/figures/bib**: the only source math-span change is **zero** —
`verify-symbols` reports the identical **707** equations and **441**
attachments as P15 after the three inline `$S_5$` spans removed with their
negation clauses (J2, J30) were re-placed into the affirmative sentences.
Tables, figures, bibliography, and Lean theorem names are byte-unchanged.

## Diagnostics, risks and delivery

The reorder makes the zero-context source diff large (195 hunks) because Git
realigns the shifted region around the moved block; heading counts (98) and
body-section counts (13 + 3) confirm no content was duplicated or dropped, and
the 2,591-fragment audit confirms every prose leaf is present in all three
PDFs. The `validate_assembly.js` section-order list and one asserted phrase
(`not an unconditional four-seed mean` → `the reported mean is conditional on
that subset`) were advanced to match the operator-directed order and wording,
not relaxed. Existing underfull-TeX spacing warnings and the pre-existing
Poppler `Suspects object is wrong type (boolean)` diagnostic remain non-fatal.

Residual author decisions: the 32-site hedge/finding classification above; the
retained technical/scope negations listed under Ruling 1; the stale Appendix B
prose pointer (J30); and P13's normalization/protocol/open-research calls. No
new experimental, formal, or deployment claim is certified. Independent
reviewer acceptance is still required.

Delivery: the first commit `609ce671` (`merge reader-first voice, positive
assertions, genre reorder per operator rulings`) carried Rulings 1–3; this
follow-up commit carries Ruling 4 (E97 focus + the consolidated development
history / source register appendix), rebuilding every generated artifact and
receipt, then pushes to `origin/paper-1p3b-spine`. The seven pre-existing
untracked review files are not staged; no unrelated file is staged.
