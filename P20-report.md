# P20 — the transfer revision

## Contract and delivery

Base **0325d925** (`paper-1p3b-spine`), worktree `/home/erikg/emender-paper`.
Authority: `taskP20-transfer.md` (operator directive 2026-10-15), with a
live supervisor scope ruling recorded below. The revision replaces the
65-page canonical structure with the reader-first organizing text and ports
the verified correctness state plus the Frontier training result. The result
is the new canonical `paper/main.typ`. One commit, pushed to
`origin/paper-1p3b-spine`.

Sources fetched and read: `origin/editorial/reader-first-paper-20261014`
(tip `fba15ce5`: `paper/main.reader-first.typ`,
`editorial/READER_FIRST_ROADMAP.md`, `editorial/EXCLUDED_HISTORICAL_EVIDENCE.md`)
and `origin/research/frontier-training-integration-20261015` (tip `9964224d`:
`editorial/READER_FIRST_TRANSFER_REVIEW.md`,
`editorial/FRONTIER_TRAINING_INTEGRATION.md`). The canonical at `0325d925`
is the correctness authority ported FROM.

## Supervisor scope ruling (recorded)

The P20 checklist names facts ("synthetic-cost ceilings", "TC0!=NC1") that
live only in the raw-write / set-C/D apparatus the reader-first base
deliberately dropped. The supervisor ruled a scoped middle, decided by the
P18 armor logic (a guard exists only for a live claim):

1. Reader-first structure and organization win; no set C/C'/D/E resource-class
   apparatus and no 8 M "state-count headroom" section apparatus.
2. Two compact retentions (flagged as supervisor judgment calls for the
   operator): (a) the 8 M parameter-matched sweep as one paragraph plus its
   small table inside *What the models learn to compute*; (b) the Lean
   one-step/k-step raw-write separation as one paragraph in *What the update
   rules establish*, theorem names and one-sentence scope only.
3. TC#super[0] $!=$ NC#super[1] stated once, plainly.
4. Set-D synthetic-cost ceilings dropped entirely.
5. Two-phase: P20 rebuilds `paper/current.pdf` and runs typst-level validation
   only; the TMLR packaging subsystem is deferred to P21.

## Passage map (old -> new)

| Canonical `0325d925` (old) | New `paper/main.typ` (P20) |
|---|---|
| Abstract, lines 23–50 (inventory of unproved propositions) | Abstract = review's proposed abstract, verbatim, with the operator tone ruling; mechanism, training result, both findings, one formal statement |
| Introduction, 109–459 (+ contributions list + Claim/Evidence/Scope table) | §1 Introduction: reader-first ~90-line arc + ONE Frontier sentence; contributions list and claim table dropped (reader-first prose carries the claims) |
| Background, 460 | §2 Background (incl. unified notation table, linear/nonlinear criterion, $S_5$ probe, dynamics taxonomy + taxonomy figure) |
| Architecture, 786 | §3 Computing with a nonlinear memory |
| Formal Results, 1152 | §7 What the update rules establish (moved after the empirical sections) |
| Multi-Programming and Systems, 1558 | §4 Making sequential updates practical |
| Language-Modeling Results, 1698 | §5 Language modeling at 1.3 billion parameters + NEW *Extended training on Frontier* |
| Expressivity Results, 2006 | §6 What the current models learn to compute + NEW *Parameter-matched probes at 8 M* |
| Repeated nonlinear maps in the wild, 2435 | §8 Repeated nonlinear maps in the wild |
| Related Work, 2648 | §9 Related Work |
| Limitations, 2768 | §10 Open questions |
| Conclusion, 2925 | §11 Conclusion (three paragraphs + Reproducibility subsection) |
| Testable Predictions, 2992 | §12 Next tests for the current pair |
| Future Work, 3073 | §13 Future Work |
| Appendix, 3189 (predecessor systems / downstream null / lineage / predecessor 1.3 B cohort / held-out bpb tables) | Appendix A Development history and source register (brief register + repository pointers only; benchmark tables removed) |
| References after appendix | References physically before Appendix A (Ruling 6 at source level) |

Built `paper/current.pdf`: **33 pages** (was 30 for `reader-first.pdf`, 65 for
the canonical). 13 numbered body sections, Appendix A.

## Ported-correctness checklist (each item -> new location)

| Item | New location | Status |
|---|---|---|
| Unified notation table (P13) | §2 Background, `[Notation]` | present (reader-first form; the dropped M²RNN-comparator `$H$` row is omitted, the head-count note is absent because the compact M²RNN sentence needs no `$H$` state symbol) |
| Production `norm+1e-6` regularization note beside the idealized step | §3 *Per-head update*, immediately after the display | present, byte-identical text |
| Core-vs-full-stack G2 scope sentence | §3 *Per-head update* end + §3 *GDN2 state channel* end; G2 open item in §10 *Recurrent realization* | ported from canonical 846 / 1019 |
| O1 open-item label | §7 *Generator memory and recurrent realization*; §10 *Recurrent realization* | present |
| S5 at 1.3 B stated as a finding | Abstract, §6 *S5 group-state tracking*, §10, §11 | present (both curves decline with length; GDN2 leads every tested length) |
| Single-seed scopes | §5 (one training seed), §6 (one seed per probe), §10 | present |
| TC#super[0] $!=$ NC#super[1] stated once, plainly | §7 *Temporal composition and routing* (one sentence) | ported |
| Synthetic-cost ceilings as facts of the accounting model | — | **dropped** per supervisor ruling 4; no live claim in the new structure requires the set-D guard |
| 150.79 B matched-token null as a measured result | §5 *Training loss at matched tokens* (table + window-dependent sign) | present |

Additional P18 findings retained: S5 curves decline toward random and GDN2
leads; unequal realized steps; better intermediate GDN2 checkpoint; failed
length-512 GDN2 curriculum (OOM after one completed step); G1–G3 and O1 open
items; production-range persistence gap (`lambda >= 1` vs production decay).

## Historical-evidence repair list (citation re-anchor / trim)

The EXCLUDED_HISTORICAL_EVIDENCE decision is implemented by using the
reader-first base, which had already removed the predecessor-cohort material.
Every canonical main-line site the task names is resolved:

1. **Appendix benchmark tables.** Canonical 3219–3862 (predecessor-cohort
   systems, downstream capability null, E63→E88→E97 lineage, predecessor 1.3 B
   cohort leaderboards, held-out bpb landscape) are removed from the
   manuscript. Appendix A keeps the short source register and repository
   pointers. No main-line claim cites these tables.
2. **Parity-null "same kind the predecessor cohort measured" framing.** Not
   present in the reader-first base; §5 states the matched-token result from
   the current pair's logs only.
3. **Protocol-dependent group-state row's predecessor legs.** Replaced by the
   §6/§10 current-pair reading; the new 8 M compact paragraph uses
   current-architecture probes (Emender / GDN / M²RNN) only, never the
   predecessor cohort.
4. **Systems occupancy references.** Canonical E88 occupancy / MFU and
   linear-scan kernel comparisons (1609, 1634, 2915, appendix) are absent;
   §4 reports only the current E97/GDN2 preflight.
5. **Related-work M²RNN predecessor comparisons.** Canonical 1587–1601
   ("both pure-nonlinear instances of the predecessor cohort") is absent; §9
   describes M²RNN's own published results and the current pair's focus.
6. **New P20 content.** The Frontier subsection cites current validation
   receipts and the corrected 513 B audit; the 8 M paragraph cites
   `paper/ndmpapernotes.md` and the current probe figure script; the raw-write
   Lean paragraph cites only theorem names. None cite excluded evidence.

## Frontier subsection text as built

Heading **Extended training on Frontier** (§5), after the matched-token
result and before model-selection provenance. Lead paragraph:

> The 1.2866-billion-parameter E97-MLP model was also continued on Frontier
> from the released 150.794-billion-token workstation checkpoint to a retained
> checkpoint at 513.014 billion accepted training tokens. The continuation ran
> 256 nodes and 2,048 accelerator ranks, per-rank batch size 4 and context
> length 2,048, schedule-free optimization, and hierarchical DiLoCo averaging
> every 40 local updates. The configuration strip reads: 256 nodes / 2,048
> ranks, batch 4 / context 2,048, $K = 40$ DiLoCo, final segment CommaPile
> mainmix; the paired workstation study trained on The Pile. Its final recorded
> rank-0 training-loss average is 2.2798 nats/token over the last 100 logging
> windows. Continued training reduced the recorded loss stage by stage across
> the retained checkpoints.

Anchor table `<tab_frontier_anchors>`: seed 2,300,930 / 150,793,748,480 /
2.4365; first 256-node continuation 2,303,840 / 199,615,447,040 / 2.3178;
intermediate milestone 2,315,840 / 400,942,039,040 / 2.2928; retained endpoint
2,322,520 / 513,013,841,920 / 2.2798.

Production segment paragraph: 73 hierarchical K40 merges, median merge 6.320 s,
median inter-merge interval 72 s, last-ten median aggregate throughput 9.410
million tokens/s. Figure B `<fig_frontier_segment>` (new
`paper/figures/frontier_segment_5134243.png`, script
`paper/figures/plot_frontier_segment.py`) plots the 73 merge durations and 72
cadence intervals from the committed job-5134243 verdict. One sentence closes:
"A continuous loss curve over the retained lineage will be published with the
recovered raw logs." No interpolated curve, no disclaimer paragraph.

Intro sentence (added once): "The same E97-MLP architecture was then continued
on Frontier from the 150.79-billion-token checkpoint to 513.0 billion
cumulative accepted tokens, and continued training reduced the recorded rank-0
loss from 2.4365 to 2.2798 nats/token, with the final segment on CommaPile
mainmix (§5)." Conclusion updated with the same continuation in one clause.
Abstract carries "An extended Frontier run reaches 513.0 billion cumulative
tokens, with its final segment trained on CommaPile."

## Compact retentions (supervisor judgment calls)

- **8 M parameter-matched sweep** (§6 *Parameter-matched probes at 8 M*, one
  paragraph + `<tab_s5>`): three seeds, Emender 0.7918 vs GDN 0.3552 vs
  raw-write M²RNN-CMA 0.2157 at $S_5$ $T=128$; Emender 1.0000 vs GDN 0.7185 on
  the solvable $S_3$ control with both raw-write variants in 0.31–0.38.
- **Lean raw-write separation** (§7 *Matched-signature delta-versus-raw-write
  separation*, one paragraph):
  `RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation_embeds`
  and `emender_m2rnn_k_step_separation`; one-sentence scope (fixed-weight
  raw-write resource class, constructed real-valued witness, each finite $k$).

## Tone audit

Grep over `paper/main.typ` for banned negative constructions:

| Pattern | Hits in new/ported text |
|---|---|
| `this is not` | 0 |
| `must not be described` | 0 |
| `do not call it` / `do not describe` | 0 |
| `is not a` / `not merely` | 0 |
| `rather than` as framing | 0 (2 technical uses: algorithm step "record an error rather than a score"; extractor "traverse the recursive local imports rather than a selected module allowlist") |
| `not X but Y` | 0 |
| Frontier banned phrases (`outperforms`, `scales linearly`, `513B matched comparison`, `held-out 2.2798`, `from scratch on Frontier`) | 0 |

Frontier is narrated forward: the continuation extended training and reduced
the recorded loss; the corpus transition (final segment CommaPile mainmix) is
stated once as a configuration fact in the strip. Banned-construction checks on
the extracted PDF report `??`=0 (all refs and citations resolved).

## Validation

Environment: Typst 0.x (`/home/erikg/.cargo/bin/typst`), Poppler pdfinfo/pdftotext
(`/tmp/p9-poppler`), Lean (`.elan`). Commands from the repository root.

| Command | Result |
|---|---|
| `typst compile --root . paper/main.typ paper/current.pdf` | PASS, no warnings; `paper/current.pdf` **33 pages** |
| `bash formal/lean/scripts/check_paper_core.sh` | PASS: 17-file trusted closure, no `native_decide`; no Lean source changed (`git status` shows no `formal/` diff) |
| `pdftotext -raw paper/current.pdf` + greps | PASS: Frontier anchor/loss/corpus text present; References (p.1243 of text) before `APPENDIX A` (p.1362); 0 `??` |
| `node paper/figures/validate_assembly.js` | FAIL (expected): hard-codes the old 13-section canonical order (`sec:formal` before `sec:systems`). Deferred to P21. |
| `git diff --check` | PASS, clean |

## TMLR packaging follow-up needs (P21)

The TMLR packaging subsystem is bound to the old 65-page canonical and was
not touched in P20. P21 must regenerate:

1. **Structure baselines.** `paper/figures/validate_assembly.js` (section label
   list, 13-section counters, hard-coded scope phrases such as "Open item O1 —
   recurrent $tanh$-tracker realization unavailable.") and
   `paper/tmlr/canonical-baseline.json` / `canonical-guard.mjs` source SHA-256.
2. **Conversion layer.** `paper/tmlr/convert.mjs` + `serialize.typ` expectations
   (13 body sections, 14 LaTeX tables, 20 captions, 8 pseudocode blocks, 21
   amsthm environments, register as one Appendix A with five subsections
   A.1–A.5). The new structure has 13 numbered sections but a different
   numbering/order, a new `<tab_frontier_anchors>` table and figure
   `<fig_frontier_segment>` to export as a packaging figure.
3. **Verification layer.** `paper/tmlr/verify.mjs`, `verify-symbols.mjs`,
   `notation-audit.test.mjs`, `author-rulings.test.mjs`, `appendix-order.mjs`,
   `register-appendix.mjs`, `armor-audit.mjs` and their `.test.mjs` samples and
   expression counts (718/706 evaluated pins etc.).
4. **Ruling 6 in the built submission/preprint PDFs.** Confirm references
   physically precede Appendix A in `submission.pdf` (anonymous) and
   `preprint.pdf` (identified); `paper/current.pdf` already orders references
   before the appendix at source level.
5. **Anonymity + bundle.** Re-run the submission anonymity receipt and
   `bundle-arxiv.sh`; rebuild `arxiv-source.tar.gz` against the new structure.
6. **New artifacts to package.** `paper/figures/frontier_segment_5134243.png`,
   the anchor table, and the updated `paper/refs.bib` (four added entries:
   `nye2021scratchpads`, `wei2022chain`, `vaswani2017attention`,
   `avsec2021enformer`).
7. **Page-count receipts.** Re-record built page counts for `current.pdf`,
   `submission.pdf`, `preprint.pdf`, and the fresh archive rebuild.

No training, evaluation, scheduler submission, Python/native build, WorksGood
service, or resilient-training behavior changed. Resilient-DiLoCo requirement
IDs are not applicable to this editorial/transfer task.
