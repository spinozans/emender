# P16 — final audit and completion of all seven rulings

## Contract, base, and delivery

P16b starts directly on **3f3f2134**, branch `paper-1p3b-spine`, after
`609ce671` (R1–R3), `c8256c8d` (R4), and `3f3f2134` (R5).
Reader-first authority is **fba15ce5**,
`origin/editorial/reader-first-paper-20261014:paper/main.reader-first.typ`.
The task is `/home/erikg/emender-scratch/fix-fanout/taskP16b-finish.md`,
with R1–R5 inherited from `taskP16-merge-reader-first.md` and the late rulings.
This report supersedes the prior report's validation/delivery paragraphs;
its inherited disposition tables below remain the history of R1–R5.

| Ruling | Final disposition and implementation evidence |
|---|---|
| 1 — positive assertions | Inherited J1–J32/E table below, plus the P16b positive/voice edit table and exact `p16b-edit-ledger.json`. Finding negatives remain findings. Every retained negative/contrast line and its context is inventoried in `p16b-negation-audit.md`. |
| 2 — genre reorder | Introduction → Background → Architecture → **Formal Results** → Systems → Language Modeling → Expressivity → Applications → Related Work → Limitations → Conclusion → Testable Predictions → Future Work; original reorder map below. Stable labels resolve; assembly checks 13 body sections. |
| 3 — reader-first imports | Original motivation/import inventory below is extended by R7's paragraph-complete audit; actual sentences, not just ideas, are adopted within canonical scope. |
| 4 — E97 focus / Option A | Current E97/GDN2 contrast is foregrounded in Architecture, Systems, LM and probe framing. A single `sec:appendix` retains lineage, predecessor cohort and held-out bpb as level-2 subsections, with internal cohort/bpb headings corrected to level 3. Predecessor count evidence also relocates to the register without deletion. |
| 5 — GDN2 name | Prior sweep's **56** sites verified (50 main source + 6 imported figure source); zero uppercase GDN2-MLP remains in manuscript sources. Infrastructure/release/run identifiers retain their real spelling. |
| 6 — appendices after references | Canonical bibliography moved before appendix activation; generated TeX now emits bibliography before its single appendix. **Built submission.pdf:** References p.44, last reference p.47, Appendix A p.48. Physical PDF checks and negative regressions added. |
| 7 — default reader-first sentences | Complete 218-paragraph source inventory: **129 as-is, 24 scoped adoptions, 63 scoped deviations, 2 packaging-preserved**. This includes unchanged shared paragraphs as well as imports. Every departure is located and justified, with full old/new edits and the exact final source diff retained. |

No training, evaluation, native/Python build, Slurm, WorksGood machinery, or mirror
update was run. This is a paper/packaging task; resilient-DiLoCo requirement IDs
are not applicable. All scientific ceilings remain the P8-accepted ones.

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

## Ruling 5 — GDN2 naming sweep (drop the `-MLP` suffix)

Operator ruling 2026-10-15: the control is named **GDN2**; the `-MLP` suffix is
dropped everywhere in the manuscript (the feedforward is an MLP by
construction). Applied as a whole-manuscript naming sweep.

**Rename sites: 56** — 50 in `paper/main.typ` (abstract; introduction; the
claim/scope table and its caption; the 1.3 B pairing description; the pair
identities table header; running prose; the `GDN2 state channel` feature
heading and its cross-references; related work; conclusion; future work) and 6
in `paper/figures/expressivity_1p3b.typ` (the `tab_depth_wall` and
`tab_s5_boundary` column headers `[*GDN2-MLP*]`, and the caption text). Every
`GDN2-MLP` occurrence became `GDN2`; the figure-source table columns and
captions are manuscript-side labels and were renamed in place.

**First use (clean).** Per the existing expansion convention (E97 is likewise
left unexpanded in the abstract and defined in the introduction), the
introduction's first definitional mention now reads “*GDN2* (Gated DeltaNet-2
@gated_deltanet2_2026), a mixer followed by the same SwiGLU MLP”. The abstract
keeps the bare abbreviation, matching how E97 is presented.

**Infrastructure identifiers NOT renamed** (paper-side naming only): the six
lowercase `gdn2-mlp` sites inside `paper/main.typ` — the Hugging Face release
URL `…/gdn2-mlp-1.3b`, the checkpoint run id `gdn2_gdn2-mlp_1.3B_20260722_083444`,
and the leaderboard keys `` `gdn2-mlp` `` / `` `lb-gdn2-mlp` `` — plus the
`paper/results/**` summaries and `RESULTS_INVENTORY.md` artifact keys.
`paper/figures/validate_section6.js` still contains one `GDN2-MLP` phrase in an
already-stale caption regression that is not part of the acceptance chain; it
is not manuscript-side and was left untouched rather than re-fitted.


## Ruling 1 — P16b positive/voice completion table

The earlier J/E entries above describe the inherited changes relative to
`f17b3be2`. The following P16b sites additionally remove or revoice
negative/contrast framing. The exact complete old/new paragraphs and scope
reasons are in `paper/tmlr/validation/p16b-edit-ledger.json` (124 operations,
including voice imports, reference move and register hierarchy).
Long paragraphs are abbreviated here only; the machine ledger is unabridged.

| Edit | Old negative/contrast anchor | New affirmative/reader-first anchor | Scope / ruling |
|---|---|---|---|
| B4 | Neither had been tested directly, because the pure setting had been regarded as too expensive to measure. | The resulting program joins a practical architecture to a testable computational hypothesis. Feedback systems, iterative optimization, and driven dynamics provide further settings in which a | Keep genre-order roadmap at its canonical site; reader-first final sentence puts theorems after experiments. |
| B9 | not across scale: | The 1.3 B result is therefore a replication across the *update rule and tuning regime* at a single scale: | Positive assertion, preserving the stated scope. |
| B11 | not a derivative. Matrix products and scalar scaling use juxtaposition; $dot.o$ denotes componentwise multiplication, and $T$ in a superscript denotes | The same state-channel symbols are used in the architecture, applications, and formal sections. Head and time subscripts may be omitted for a single step. A prime denotes the returned state. | Reader-first wording within the same accepted scope. |
| B12 | does not fit the #ref(<sec:lm>) matched protocol, so it is left to future work. | The recurrent language model lineage runs from Elman networks @elman1990 and the LSTM @lstm1997 through the modern linear-recurrent variants: state-space models such as Mamba and Mamba2 @mam | Retain 8 M comparator identity and Mamba-3 protocol exclusion after reader-first lineage. |
| B19 | rather than a raw addition; | a gated magnitude; | Positive assertion, preserving the stated scope. |
| B22 | does not impose E97's projection processing on GDN2. In the trained stack the mixer is followed by the SwiGLU MLP of #ref(<sec:lm>), not by a state activ | Algorithm 2 defines the adjacent GDN-2 linear core used by the formal comparison: decay is applied *before* retrieval and correction, and no $tanh$ is applied to the returned state. Its inpu | Keep the exact P15 abstract-core/full-stack limitation sentence and projection-processing boundary. |
| B23 | does not pull the learned decay toward a default. The recurrence is computed in float32 and cast back to the storage precision, and the $tanh$ on the sta | E97 parameterizes the per-head decay $d_h$ in log space, following Mamba2 @mamba2_2024: each head learns a scalar $A_(log)$ and a bias $delta_(text("bias"))$ — the code names \`A_log\` and \`dt | E88 no-write-gate history is already preserved in the lineage subsection. |
| B24 | rather than one large matrix — and the numbers follow from it: each head's $32 times 32$ tile is small enough to live in registers (#ref(<sec:systems>)), th | Ingredient (d) sets the 1.3 B stack’s shape — many small heads — and the numbers follow from it: each head's $32 times 32$ tile is small enough to live in registers (#ref(<sec:systems>)), th | Reader-first wording within the same accepted scope. |
| B26 | rather than a selected module allowlist. This check covers project sources, not the axioms or source files of Lean and Mathlib dependencies. | A trusted Lean 4 @lean42021 core built on Mathlib @mathlib4 carries the formal spine: each result is stated as a theorem and compiled together with a proof of it. The import closure of the \` | Keep Lean/Mathlib dependency trust boundary. |
| B27 | does not prove strict containment below NC#super[1]. | #set list(indent: 1em) - *Finite-state ceiling at fixed precision.* At fixed width and precision, every online recognizer of this kind has only finitely many distinct states (\`S5Witness. | Finite-state scope is retained in the explicit complexity non-claims below. |
| B30 | Neither theorem states that iterating the nonlinear cell from zero tracks every input word. | The exact statement of \`NDMRealizesS5.emender_realizes_s5_tracker\` supplies $d=12$, four orthonormal keys and one-hot values, queries, a decoder, and $lambda=1$. Reading a *preloaded* table  | Retain the accepted specification: preloaded generator lookup plus external group multiplication. |
| B33 | not a proof that the resulting function has no shallower representation. | The concrete repeated-map family is the schedule of \`CompositionalDepthDuality.E97Step\` values $(k,e,w,v)$ at fixed $lambda$ (Lean field \`b\` is the paper's erase gate $e$), acting through \`e | Keep minimal-representation-depth boundary. |
| B35 | does not rule out affine group representations or lower-bound the network generating their parameters. G1–G3 (#ref(<sec:limitations>)) record those gaps. | \`S5GDNBridge.gdn_s5_realization_requires_per_token_action\` assumes a token-only schedule and a decoder that recovers \`S5Tracker.run\` on every word and separates reachable states. It then giv | Keep parameter-generator resource boundary and named gap pointer. |
| B36 | Rather than parallelize one recurrence along time, it replicates the recurrent computation across many independent heads — each with its own small bounded m | Width-axis multi-programming supplies throughput while each head’s time loop remains serial. A linear recurrence earns its throughput on the *time axis*: because $S_t = A_t S_(t-1) + b_t$ un | Reader-first wording within the same accepted scope. |
| B40 | not a portability benchmark for the E97/GDN2 pair. | The recurrence is implemented in Triton @triton2019. The paired production runs, throughput measurements, and expressivity smokes use NVIDIA CUDA. Porting the pair to ROCm would add a direct | Retain single-arm Frontier continuation exclusion. |
| B41 | rather than along time, it does not need sequence parallelism to stay competitive at the 1.3 B-class — a simplification relative to the chunked-scan impleme | The 1.3 B-class result reported here was produced on a single eight-GPU workstation, with no cluster and no sequence parallelism — both 150 B-token arms of #ref(<sec:lm>) ran the same *local | Reader-first wording within the same accepted scope. |
| B44 | rather than specifying a new search or training run; the historical CMA sweep table below identifies retained and missing winner-search records. The averagi | Algorithm 4 separates selection, training, and comparison. It summarizes the recorded pair protocol; the selection account below identifies the retained winner configurations and the missing | Algorithm number follows retained raw-write algorithm; search evidence remains in place. |
| B46 | not an unlogged terminal-step measurement. | The primary read-out is the training-log loss of both arms aligned on exact common logged steps through step 2,300,925 (150,793,420,800 tokens): 92,037 aligned records, zero E97-only and zer | Reader-first wording within the same accepted scope. |
| B47 | does not distinguish the nonlinear-state from the linear-state recurrence. Under the run's recorded byte normalization (\`bytes/token\` $= 3.783$ from the  | The two current models reach *comparable training loss* after each family's geometry and learning rate have been selected by search. The nonlinear recurrence trains competitively under this  | Reader-first wording within the same accepted scope. |
| B52 | rather than interpreted as a new training recipe. | The protocol below summarizes the retained driver, \`scripts/expressivity_1p3b_depthwall.py\`, including its zero-based probe indices. Evaluation uses the weights at each arm's plateau stop. T | Retain explicit terminal-versus-best checkpoint distinction. |
| B53 | do not trigger it. The cap, stopping criterion and recipe are common; realized training lengths in updates are not equal. | Relative gain divides the loss decrease by the earlier loss magnitude (floored at $10^(-12)$); both trailing windows must pass. The implemented minimum uses the zero-based index $j$. Both ar | The zero-based implemented minimum remains explicit. |
| B54 | not a hard ceiling: intermediate GDN2 trajectory probes were higher. One seed, unequal plateau-stop lengths, and validation fluctuations limit the inf | At the terminal sweep E97 reaches #expressivity.percent(expressivity.wall-acc(expressivity.wall-e97, 128)) at trained length, retaining #expressivity.percent(expressivity.wall-acc(expressivi | Retain terminal-versus-intermediate checkpoint and interpretation scope. |
| B55 | rather than learn length-invariant tracking; held-out transitions remain weaker than seen ones. GDN2's attempted length-512 curriculum stage was OOM-infeasi | Both capacity curves decline with length; held-out transitions remain weaker than seen ones. GDN2's attempted length-512 curriculum stage was OOM-infeasible even at sub-batch 1, after one co | Retain hardware limitation and single-seed scope. |
| B60 | not merely placed in an output readout. | Its discontinuous output changes the residue used at the next step. | Reader-first feedback description. |
| B61 | not a nonlinear-state example. | For fixed inputs this update is affine in $w_t$. | Reader-first affirmative LMS classification. |
| B66 | rather than to refill. The earliest formulation is Widrow and Hoff's least-mean-square rule @widrow_hoff_1960, which writes the error between target and cur | The Emender update belongs to a long line that treats memory as a structure updated by prediction error. The earliest formulation is Widrow and Hoff's least-mean-square rule @widrow_hoff_196 | Keep conditional hold range and budget (canonical inline math). |
| B67 | not a universal complexity classification. The empirical panel in #ref(<sec:expressivity>) directly tests Gated DeltaNet as the matched representative | The dominant comparison cohort is the family of linear-state billion-parameter recurrent LMs surveyed in #ref(<sec:background>): Mamba @mamba2024 and Mamba2 @mamba2_2024 (selective SSMs); Re | Predecessor GDN evidence remains in expressivity and the register; current-pair comparison foregrounded. |
| B68 | does not require a new time-parallel scan. | A newer linear-state result, Gated DeltaNet-2 @gated_deltanet2_2026, decouples channel-wise erase and write gates and is reported by its authors to outperform GDN. It extends retrieval and s | Exact theorem set-G cross-reference retained. |
| B75 | does NOT replicate at 1.3B on this pair.” The 1.3 B GDN2 model is ahead on that group-state protocol. Real-valued update-family separation and the 120-st | The iterated nonlinear-map experiment connects that practicality to the architectural motivation. E97 learns the recurrence more accurately and retains a larger fraction of its accuracy as s | Retain negative transfer verbatim, seed/protocol scope, and lookup/affine-channel ceilings. |
| B76 | does not establish TMLR template or anonymization compliance; journal-specific submission packaging remains a separate step. | These findings support studying nonlinear memory as a computational resource. The next step is to test that resource across training budgets, scales, and tasks in which an evolving state mus | Retain applications/deployment, contribution and journal packaging scope. |
| B80 | does not add cross-token mixing; it changes what the recurrent state writes | It changes what the recurrent state writes | Positive assertion, preserving the stated scope. |
| B81 | not an actual matched-compute certificate. | actual dense $H W$ and feature-network costs; actual matched-compute certification remains open. | Positive assertion, preserving the stated scope. |
| B82 | rather than simply many value heads | the pure setting may benefit from many independent *address programs* at a given value-head count | Positive assertion, preserving the stated scope. |
| B83 | does not uniquely isolate the cause of divergence, since width, depth, and head count also changed. | width, depth, and head count also changed, leaving the cause of divergence unresolved. | Positive assertion, preserving the stated scope. |
| B84 | does not certify matched empirical compute. | feature-network costs; empirical compute matching requires separate measurement. | Positive assertion, preserving the stated scope. |
| B85 | not a claim of general or learned-weight superiority. | raw-write resource class; general or learned-weight superiority remains open. | Positive assertion, preserving the stated scope. |
| B86 | rather than a categorical impossibility: | matched-budget learning-efficiency gap: | Positive assertion, preserving the stated scope. |
| B87 | not a guarantee of learning; the two findings jointly | Realizability describes representational scope; the two learning findings jointly | Positive assertion, preserving the stated scope. |
| B88 | not a theorem that every pangenomic workload requires them; | updates as an empirical research direction; | Positive assertion, preserving the stated scope. |
| B89 | does not rule out non-solvable state tracking without an unproved complexity-class separation. | A non-solvable-state-tracking impossibility would additionally require the open complexity-class separation. | Positive assertion, preserving the stated scope. |
| B90 | does not depend on whether the non-linearity is a $tanh$, an exponential, a $tanh$-of-product, or a sigmoid. | distinction is structural across activation choices: a $tanh$, an exponential, a $tanh$-of-product, or a sigmoid. | Positive assertion, preserving the stated scope. |
| B92 | does not separate NC#super[1] from TC#super[0]. | NC#super[1]-complete witness; separation of NC#super[1] from TC#super[0] remains open. | Positive assertion, preserving the stated scope. |
| B93 | not a general learnability theorem. | budgets and per-architecture CMA-ES; a general learnability theorem remains open. | Positive assertion, preserving the stated scope. |
| B94 | does not certify actual matched empirical compute. | Actual matched empirical compute requires separate measurement. | Positive assertion, preserving the stated scope. |
| B98 | rather than blending an external carry with a candidate. | The Emender carry sits inside the bounded delta-correcting update, and E97's split gates act inside the delta path. | Positive assertion, preserving the stated scope. |
| B100 | not a verified actual FLOP count. | coarse state-pass accounting model; actual FLOP counts require measurement. | Positive assertion, preserving the stated scope. |
| B101 | not a kernel concurrency certificate.* | *A signature-field predicate is shared by E88 and a declared M²RNN proxy; kernel concurrency requires a separate certificate.* | Positive assertion, preserving the stated scope. |
| B102 | rather than E97's smooth saturation. | nonlinearity, distinct from E97's smooth saturation. | Positive assertion, preserving the stated scope. |
| B105 | not a single fixed map on $z_j$ alone. | its clock), yielding a fixed map on that augmented state beyond $z_j$ alone. | Positive assertion, preserving the stated scope. |
| B106 | does NOT replicate at 1.3B on this pair.” Repeated nonlinear maps in feedback, optimization, reservoirs, and dynamics define the applications sco |  abstract: [ Conversation, perception, and feedback call for ongoing computation with predictable per-input cost. Emender explores a complementary path through efficient nonline | Reader-first abstract voice, retaining fixed numeric sample, negative transfer, single-seed/unequal-step, fixed-schedule, formal-resource and deployment ceilings. Scratchpad framing excluded: absent canonical citations. |
| B108 | does NOT replicate at 1.3B on this pair.” These are single-seed, common-cap results with unequal realized training and graceful length transfer | The negative transfer is: “The 8M S5 separation does NOT replicate at 1.3B on this pair.” The modular-quadratic comparison uses one seed, a common cap, unequal realized training, and gracefu | Bind common-cap/unequal-duration qualification specifically to the modular-quadratic protocol. |
| B111 | does not identify them: E97 uses $lambda S - k(e dot.o k)^T S + k(w dot.o v)^T$, whereas GDN-2 uses $(I - k(e dot.o k)^T) D S + k(w dot.o v)^T$. The  | #block( width: 100%, inset: 8pt, radius: 3pt, stroke: 0.6pt + rgb("#e08a2e"), fill: rgb("#fdf4e8"), )[ *Shared split-gated cores, with different decay placement.* Gated DeltaNe | Reader-first adjacent-core wording with production-domain inclusion ceiling retained. |
| B116 | does not replicate the predicted separation] | #heading(level: 2, numbering: none)[S5 group-state tracking on the same pair] The $S_5$ word problem tests repeated composition of permutation actions. Finite-group actions admit affine rep | Reader-first group-action question; original negative result and exact source quotes remain. |

The companion `p16b-negation-audit.md` enumerates **315 retained source
lines**, including exact neighboring context: negative findings/recorded absences,
technical coordinate/axis/metric distinctions, and accepted scope/protocol/fixed
content boundaries. It is an inventory, not 315 new scientific findings.
In particular the original 20 hardcoded guard samples are byte-identical,
including the CUDA/ROCm boundary, control-safety boundary, negative transfer,
and former Appendix B/C measurement samples. No sample was deleted or rewritten.

## Ruling 4 — final register and evidence audit

The Option A decision is **relocate, not delete**. The register has five
level-2 entries: predecessor systems measurements, predecessor downstream
panels, lineage, predecessor 1.3 B cohort (`sec:appendix_cohort`), and
held-out bpb (`sec:appendix_bpb`). Nested cohort/bpb headings are level 3.
The extra predecessor arithmetic from Architecture (12 layers, 370 heads,
32×32 tiles, 22,200 programs at batch 5) now lives in the systems entry;
its exact Lean definition and both inline math expressions are preserved.
The current E97 count remains in Architecture. All tables/figures remain
byte-identical to 3f3f2134, including the certified predecessor annotation;
those constrained figure annotations and load-bearing formal comparisons
are explicit exceptions to the prose demotion, not a hidden deletion.

`validate_assembly.js` already received the 13-body-plus-one-appendix
expectation in c8256c8d; no further relaxation is needed. Its original phrase
and scope oracles remain. It passes on the final tree. `verify.mjs` still
finds, in **all three PDFs**, exactly the old Appendix B sample
“E88 0.973 BPB, GDN 0.973 BPB, and M²RNN-CMA 0.979 BPB.” and Appendix C
sample “Our three models on the same Common-Pile slice measure Emender
0.981, GDN 0.963, M²RNN-CMA 0.973”. The labels survive consolidation.

## Ruling 5 — audited rename count

The historical rename commit changes 50 main-source and 6 imported-table-source
`GDN2-MLP` sites, **56 total**. Final checks find none in
`paper/main.typ` or `paper/figures/expressivity_1p3b.typ`.
The six lowercase main-source `gdn2-mlp` infrastructure occurrences remain
real artifact identifiers. This finishing commit introduces no further rename.
An early terminal inspection was misread as a residual caption alias; direct
source/commit comparison confirmed the prior rename was complete before edits.

## Ruling 6 — in-PDF appendix-order verification

The bibliography directive precedes `#show: arkheion-appendices` in the
canonical source. The converter needs no special reordering hack: it renders
that evaluated order, so `\bibliography{refs}` precedes `\clearpage\appendix`.
The canonical artifact follows the same reader order. Poppler extracts the
**newly rebuilt PDFs themselves**, and the verification code splits physical
pages, finds the References heading and register heading, and requires the
last TMLR bibliography entry (HellaSwag DOI `10.18653/v1/P19-1472`) before
the appendix. This verifies the bibliography's end, not just its start.

| PDF | References begins | Final reference | Appendix A begins | Total pages |
|---|---:|---:|---:|---:|
| canonical | 51 | IEEE ordering; heading verified | 55 | 67 |
| submission | 44 | 47 | 48 | 60 |
| preprint | 44 | 47 | 48 | 60 |

Receipts: `p16b-appendix-order.json`, `p16b-appendix-pages.txt`, and
`{canonical,submission,preprint}-text.txt` under `paper/tmlr/validation/`.
New `appendix-order.test.mjs` has four tests covering physical order,
old appendix-before-references TeX/PDF order, absent headings, same-page
appendix, bibliography spill/missing final entry, and duplicate commands.
`verify.mjs` invokes the physical-order check for all three artifacts.

## Ruling 7 — paragraph-complete adoption/deviation and import inventory

**Default is the reader-first sentences as written.** The complete inventory
is `paper/tmlr/validation/p16b-reader-voice-audit.md` (and JSON): every one
of the 218 source paragraphs, its original line/anchor, whether as-is,
scoped adoption or deviation, and its scope reason. Actual imports are also
retained unabridged in `p16b-reader-imports.json` (98 explicitly recorded
source paragraphs); as-is shared paragraphs include pre-existing identical
material. Counts are paragraphs, not claims or newly imported sentences.

| Section | Adoption / preserved precision |
|---|---|
| Abstract | Reader-first per-input/correct/transform/practical-training arc and computational-bias conclusion; fixed numeric sample, single seed, common cap, unequal realized training, S5 negative, fixed schedule, formal scope and untested deployments retained. |
| Introduction | Actual balance/conversation/perception opening; sequence-as-instructions framing; memory read/correct/transform and layer naming; systems problem; practical and computational questions; formal synopsis. Scratchpad/CoT sentences excluded because their citations are absent from frozen canonical bibliography. |
| Background | Actual location-of-nonlinearity opening, lineage, criterion, matrix-state opening, S5 probe and taxonomy sentences. Retain detailed precision/uniformity/class-separation conditions, production core non-inclusion and unchanged unified notation table. |
| Architecture | Actual E97 head description, parameterization, many-small-head framing, matched E97/GDN2 contrast and core description. Preserve all-one-gate specialization, raw-write resource comparison, finite-logit/reachable-domain and G2 boundaries. |
| Formal Results | Actual trusted-artifact paragraph, finite-state/lookup descriptions, recurrent-realization target, split-core relations, latching synopsis and temporal-composition/routing sentences. Keep theorem-set names and all accepted C/C′/D/E/G/F/H hypotheses/targets. Alternate experiments-above framing is inapplicable after R2. |
| Systems | Actual serial-loop/width-axis explanation, programmable-body framing, E97 occupancy-pending sentence, DDP preflight measurements, production/preflight distinction, Triton platform and distributed-training connective tissue, ParaRNN contrast. The 7-GPU DDP numbers were checked against experiments/preflight_100b/RESULTS.md; CUDA-only sample, proxy theorem and local attempted-solver evidence remain. |
| Language Modeling | Actual workload motivation, shared-recipe/separate-selection description, algorithm connective tissue, matched-loss description and pending-evaluation paragraph. Full eleven-family/CMA winner/provisional-search evidence remains; alternative's shorter account omits load-bearing source records. |
| Expressivity | Actual current-pair computational questions, map-study protocol language, terminal/intermediate checkpoint wording, S5 group-action question, feasible curriculum and hypothesis interpretation. Keep exact negative artifact quotations, 8 M/grok/contractive controls, successful-seed conditions, fixed-cap/unequal-step and terminal-not-best boundaries. |
| Applications | Actual evolving-state opening and selected feedback, converter, affine LMS, FISTA, reservoir and closing sentences. Original equations/coordinates and every EXHIBIT/ANALOGY/OPEN, fixed-support/contraction/precision/alternative-encoding/stability condition remain. New scratchpad/Enformer cited subsections exceed canonical bibliography/evidence and are excluded. |
| Related Work | Actual prediction-error ancestry, current GDN2 comparison, split-gate relation and M²RNN synopsis. Keep conditional-hold range/budget, predecessor fairness/geometry evidence and the classical negative empirical record. |
| Limitations | Actual O1 target, seed/budget/checkpoint/curriculum and evaluation targets, G2 extension target, scale/occupancy/ROCm and citation-resolution prose. All exact G1–G3/O1/U3.1 disclosures survive. |
| Conclusion | Actual ongoing-computation opening, practical-training statement, iterated-map/S5 contrast, formal fixed-schedule summary and next-study conclusion. Preserve negative transfer verbatim, common-cap scope specifically on the modular-quadratic study, pending held-out evaluation and realization/deployment boundaries. |
| Testable Predictions | Canonical section byte-identical, as explicitly required; alternate replacement is excluded and its current-pair tests appear elsewhere. |
| Future Work / register | Actual scale/task/formal-target synopsis and reader-first register opening/source inventory; all canonical detailed future targets and Option A evidence remain. |

### Approved inline-math mention delta and notation consistency

The supervisor expressly approved the intended **inline-mention** delta under
the P13/P14 provenance mechanism, conditional on a complete delta inventory,
exact pins and notation-table consistency. **All 12 display equations, all
17 figure/table blocks, both standalone tables, all eight pseudocode blocks,
and every theorem/source name are byte-unchanged** (exact names remain present;
116 unique source identifiers checked). The unified P13 notation table is
byte-identical. The criterion now uses `S_(t-1)` for the previous carried
state, consistent with its sanctioned time-indexed `S_t`; it no longer
uses `x_t` (the table's token/layer input) in that explanatory clause.
No new overload or changed mathematical assertion is introduced.

Net repeated inline-mention changes: **+3 tanh, +8 S5, −1 x_t, +1 S_(t-1)**.
Consequently exact evaluated counts are **718 math expressions / 449
attachments**, from 707/441, with **eight S primes per PDF unchanged**.
`canonical-baseline.json` and the symbol regression pin these exact counts,
not a range. The verifier additionally compares the rendered inventory with
these baseline pins. All older missing-prime/unsupported-attachment and P10
conversion-defect negative tests remain active.

Every per-edit inline addition/removal (including relocations that cancel in
the multiset) follows; full expressions, source anchors and final occurrence
lines are in `p16b-math-delta.json`. Locations are final paragraph-start lines;
removed clauses use their replaced paragraph's site or the exact diff.

| Edit | Final site / anchor | Reader-first source / reason | Added inline expressions | Removed inline expressions |
|---|---|---|---|---|
| B5 | 195: Emender uses a nonlinear matrix memory. At each step, a head reads its current p | R13 | $tanh$ | — |
| B7 | 218: The formal analysis describes the state computations behind this comparison. Lea | R18 | $S_5$ | — |
| B22 | 1026: Algorithm 2 defines the adjacent GDN-2 linear core used by the formal comparison | R63 | $tanh$ | — |
| B25 | 1078: #heading(level: 2, numbering: none)[The matched architectural contrast] E97 and  | R69; R70 | $tanh$ | — |
| B49 | 2065: Comparable language-modeling loss leaves open what the carried state can learn t | R107 | $S_5$ | — |
| B55 | 2469: Both capacity curves decline with length; held-out transitions remain weaker tha | R120 | — | $S_5$ |
| B56 | 2475: #heading(level: 2, numbering: none)[Interpreting the current-pair evidence] The  | R121; R122 | $S_5$ | — |
| B70 | 2855: *O1: realizing the group tracker.* Find an E97 update configuration, initial sta | R193 | $S_5$ | — |
| B71 | 2918: #heading(level: 2, numbering: none)[Single-seed 1.3 B evidence and protocol boun | R190 | $S_5$ | — |
| B75 | 3033: The iterated nonlinear-map experiment connects that practicality to the architec | R203 | $S_5$ | — |
| B77 | 3176: A matched 3–7 B E97/GDN2 study would test training viability, memory cost, and c | R210 | $tanh$ | — |
| B106 | 23: abstract: [ Conversation, perception, and feedback call for ongoing computation  | R3 | $S_5$ | — |
| B109 | 558: A single explicit criterion classifies recurrent architectures. A recurrent laye | R28 | $S_(t-1)$ | $x_t$ |
| B111 | 693: #block( width: 100%, inset: 8pt, radius: 3pt, stroke: 0.6pt + rgb("#e08a2e"), fi | R42 | — | $tanh$ |
| B112 | 880: For E97, | R54 + canonical predecessor count | — | $32 times 32$; $370 times (32 times 32)$ |
| B113 | 3320: == Predecessor-cohort systems measurements A Lean 4 definition (\`emender_1p27B\`) | R54 + canonical predecessor count | $32 times 32$; $370 times (32 times 32)$ | — |
| B116 | 2449: #heading(level: 2, numbering: none)[S5 group-state tracking on the same pair] Th | R116; R117 | $S_5$ | — |
| B119 | 183: The second question is what these trained models can learn to compute. An iterat | R17 | $S_5$ | — |

The matrix-state opening additionally moves its existing matrix-coordinate
mention before its existing vector-coordinate mention (R32); both expressions
are identical and counts unchanged. The R54 predecessor geometry expressions
move to the register verbatim. No table/figure/display equation or numerical
result was rebased away to obtain these counts.

## Cleanup, provenance, and guard delta

All **seven** pre-existing loose review copies (including P8-review.md, beyond
the six named in the task's abbreviated list) matched their scratch originals
with `cmp`, then were removed from the worktree: P4/P6/P10/P12-review.md,
P4-verification.mjs, P7-report.md, P8-review.md. Hashes are retained in
`p16b-cleanup.txt`; originals remain in scratch. None was staged.

The old source digest guard first **rejected** the edited source as unapproved
(`p16b-unapproved-source.txt`); it was not disabled. The baseline preserves
P13/P14/P15/P16 history and the old digest as `p16MainTypSha256`, adds
`p16bBaseCommit: 3f3f2134`, and identifies this report, exact source diff and
full disposition inventory. Final source SHA-256:
`9150b69fd7dfb8837a3608e3c5dd963bf4127335f3809fd1b45c652e3618d0b8`.
`p16b-guard-delta.json` enumerates the approved categories, old/new source
and symbol counts, unchanged fixed samples, and every old/new layout split.
The original one-insertion prose-layout rule is unchanged; fragments are
2600 → **2635**, all checked in every PDF.
The accepted eb2b9636 PDF, bibliography amendment, vendored style, image
assets, build wrapper and Lean source artifact stay frozen. SHA256SUMS retains
exactly its original 15 paths; all verify.

## Full final validation chain

| Command | Final result / retained evidence |
|---|---|
| `node paper/tmlr/validation/p16b-audit.mjs` | PASS: 12 displays, 17 figures/tables, two standalone tables, eight pseudocode blocks, 116 exact source identifiers, notation table, predictions, all 20 samples, bibliography/imported figure source and one-register structure; `p16b-preservation.{txt,json}`. |
| `typst compile paper/main.typ paper/current.pdf` | PASS: **67 pages**. |
| `node paper/tmlr/convert.mjs` | PASS: **102 headings, 20 captions, 21 formal environments**, byte-identical bibliography; `convert.txt`. |
| `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` | PASS: submission/preprint **60 / 60 pages**, no missing glyphs, unresolved refs, overflows or errors; `build-output.txt`, `submission-build.txt`, `preprint-build.txt`. |
| `bash paper/tmlr/bundle-arxiv.sh`; fresh extraction; `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; Poppler raw text; `cmp` | PASS: fresh archive rebuild, raw PDF text byte-identical to preprint; `arxiv-build-output.txt`, `arxiv-build.txt`, `arxiv-compare.txt`. |
| `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo node paper/tmlr/verify.mjs` | PASS: 20 original samples × three PDFs, **2,635 long source fragments**, **718 math expressions / 449 attachments**, eight S primes per PDF, source/bibliography/identity guards, physical appendix order; `verify-output.txt`, `drift-check.json`. |
| `node --test paper/tmlr/{verify-symbols,notation-audit,author-rulings,appendix-order}.test.mjs` | PASS: **24/24**, including four new appendix-order tests; `symbol-tests-output.txt`. |
| `node paper/figures/validate_assembly.js` | PASS: 13 body sections + one appendix, all 57 bibliography entries, original scope phrases, 17-file trust inventory and checkpoint bindings; `assembly-check.txt`. |
| `bash formal/lean/scripts/check_paper_core.sh` | PASS: 17 project source files, no placeholder/axiom/opaque/native_decide; `p16b-lean-gate.txt`. |
| `(cd formal/lean && lake build ElmanProofs.PaperCore)` | PASS: **2,196 jobs**, cached build, pre-existing linter warnings only; `p16b-lean-build.txt`. |
| `node formal/lean/scripts/test_local_import_closure.js` | PASS: recursive imports, external exclusion, five forbidden-token and compiler-error fail-closed fixtures; `p16b-import-closure.txt`. |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS: all 15 paths; `p16b-checksums.txt`. |
| `git diff --check` | PASS after trailing-space normalization of copied TeX log receipts (raw generated logs retained by the build). |

**Anonymity:** rebuilt submission.pdf contains Anonymous authors and anonymous
PDF metadata, excludes all identity tokens, and retains anonymized artifact
links. Identified preprint retains author and release identities. verify.mjs
checks both. Page counts change from 66/58/58 at 3f3f2134 to 67/60/60; the
references-first placement and full scoped sentence imports account for layout.

## Diagnostics, residual risks, and delivery gate

Initial exact-text editing assertions aborted before writing when the already
completed GDN2 rename was mistaken for a residual alias and when a transition
anchor did not match; corrected. The first regression run intentionally exposed
stale TeX and old 707/441 / O1-count pins. TeX was rebuilt; symbol deltas received
explicit supervisor approval and enumeration; the O1 test pins 14 occurrences
(the added reader-first realization target and future-work synopsis) and checks
both new anchors. No sample, math comparison or negative regression was removed.
New PDF-order tests reject the old order, missing references and bibliography
spill, not merely inspect source commands. The longstanding Typst-Poppler
“Suspects object is wrong type (boolean)” diagnostic and non-fatal TeX
underfull-box warnings remain; final checks pass.

No new experimental, theorem, deployment, scale-law or circuit-class claim is
certified. O1/G1–G3/U3.1, single seeds, unequal realized budgets, pending held-out
pair evaluation, metadata flags and the inherited stale predecessor plateau
pointer remain disclosed research/editorial limitations. Independent reviewer
acceptance is required. The requested delivery is **one P16b commit on top of
3f3f2134**, including this report and explicit task artifacts, pushed to
origin/paper-1p3b-spine; final response/scratch receipt records its actual hash
and remote equality. No loose review copies or unrelated files are included.
