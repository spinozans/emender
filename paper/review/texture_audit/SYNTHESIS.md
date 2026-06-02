# Texture & Flow Audit — SYNTHESIS (fan-in over 18 auditor nodes)

**Subject:** Garrison, *Emending Nonlinear Recurrence* (`paper/main.typ`)
**Mode:** AUDIT-ONLY. Diagnosis, not edits. No fix recommendations. `paper/main.typ` untouched. The editing phase is separate and gated on human review of this document.
**Inputs aggregated:** `paper/review/texture_audit/{abstract,s1a,s1b,s1c,s2,s3a,s3b,s4,s5a,s5b,s6a,s6b,s6c,s7,s9,s10,s11_14}.md`

> **Coverage note (read first).** 17 of the 18 commissioned registers exist. **`s8.md` (§8 Related Work) was never written** — the `texaudit-s8` node was marked done but produced no artifact (no file on disk, no blob in git on any branch). §8 is therefore **un-audited**. This is a real hole, not a rounding error: two auditors (s6a, s2) explicitly punt the M²RNN-rivalry / competitor-framing relitigation *into* §8 ("belongs in §8 related work, not woven into the S₅ headline"), so §8 is precisely where one class of pre-litigation defect is expected to concentrate and where no one looked. The gap is carried through every section below and is the first item of §6 (The Validation Test). A follow-up audit task (`texaudit-s8-fill`) has been filed so the graph self-heals before the editing phase.

---

## 1. Density / clustering map

### 1.1 Counts

Defect rows, excluding OK-anchor rows (which auditors logged only to mark the clean baseline):

| unit | rows | UNCANNY | predominant category | skew |
|------|-----:|--------:|----------------------|------|
| abstract | 4 | 0 | residual INSIDE-OUT / UNPAID-WORD | the paper's **cleanest** unit |
| s1a (intro opener + contributions) | 8 | 2 | INSIDE-OUT / RESULT-AS-SYSTEM | clustered in sentence 1 |
| s1b (claims table) | 7 | 2 | PRE-LITIGATION | clustered rows 2 & 4 |
| s1c (claims passage) | 5 | 1 | flow / posture | best-behaved §1 sub-unit |
| s2 (background) | 11 | 3 | INSIDE-OUT (+ autobiography) | clustered at head & tail |
| s3a (per-head update) | 6 | 2 | RESULT-AS-SYSTEM / ANTHROPOMORPHIC | clustered at non-equation moments |
| s3b (ablation) | 6 | 1 | PRE-LITIGATION / redundancy | clustered at back |
| s4 (systems) | 10 | 2 | PRE-LITIGATION (+ INSIDE-OUT) | clustered L818–851 |
| s5a (CMA-ES setup) | 9 | 2 | PRE-LITIGATION / ANTHROPOMORPHIC | clustered in one narrative ¶ |
| s5b (racer / tie) | 6 | 2 | PRE-LITIGATION | clustered in "FLOP-locked" |
| s6a (8 M probes) | 6 | 1 | PRE-LITIGATION (+ RESULT-AS-SYSTEM) | mildly clustered |
| s6b (deployed 1.3 B) | 10 | 3 | PRE-LITIGATION + RESULT-AS-SYSTEM | densest "deployed/production" cluster |
| s6c (micro/mega) | 10 | 1 | RESULT-AS-SYSTEM | clustered around one ¶ |
| s7 (formal) | 6 | 2 | INSIDE-OUT (process leak) | tightly clustered at one seam |
| **s8 (related work)** | **—** | **—** | **NOT AUDITED** | **coverage gap** |
| s9 (limitations) | 9 | 2 | INSIDE-OUT + RESULT-AS-SYSTEM | clustered at edges |
| s10 (conclusion) | 5 | 2 | INSIDE-OUT | clustered at seams |
| s11_14 (predictions/future/appendix) | 8 | 0 | PRE-LITIGATION (+ "production") | two pockets; §11 is clean |
| **total** | **126** | **~28** | | |

### 1.2 Category × frame-rule distribution (qualitative)

- **PRE-LITIGATION (R3)** is the single largest category — present in 15 of 17 audited units, the clear plurality of all 126 rows. The "honest"/"genuinely"/"directly rather than assert it"/"not seed-luck"/"not a buried caveat"/"a critic can always ask"/"fairness anchor"/"moat"/repeated-caveat family.
- **RESULT-AS-SYSTEM (R2)** — concentrated but uniformly recurring: the "production"/"deployed"/"released v0.3"/"production stack"/"production architecture" thread. Lower row-count than R3 but the **single most cross-flagged token** in the entire audit (see §3).
- **INSIDE-OUT (R1)** — pervasive: undefined codenames and insider terms ("E88"/"the Emender" at first use, "stitched," "the racer," "the campaign," "comma-pile control," "on the rack," "v0.3," bare Lean lemma ids, snake_case task slugs, `ElmanProofs`).
- **ANTHROPOMORPHIC (R2)** — a minority thread but the source of several UNCANNY peaks (the architecture that "emended its own name," the optimizer that "kept pushing"/"independently voted," memory that can "change its mind"/"stand firm," E88 that "invites," the latch that "holds comfortably," the result that "retains the lead"/"consumed" a stream).
- **UNNAMED** — 12+ rows that fit no category; these are the §4 deliverable and the source of the proposed new rule.
- **Pure FLOW-BREAK / CONCLUSION-FIRST** (ordering faults with no stance source) are a genuine minority: the three-vs-four load-bearing count mismatch (s3a), "Two results" vs three contributions (s10), the conclusion trailing into file paths (s10), the deferral chain "see Honest mirror below" (s6b), the Frontier digression interrupting the theorem-set spine (s7).

### 1.3 Verdict: **FEW DEEP SOURCES, not many shallow defects**

The hypothesis is **confirmed with high confidence**. Roughly **three stance defects generate the great majority of the 126 surface symptoms.** Named:

- **STANCE-A — Vantage collapse (R1).** The paper writes *from inside its own project*, treating its private world as shared furniture. This one stance generates: the undefined intro proper nouns, every "racer"/"stitched"/"campaign"/"comma-pile"/"on the rack" hit, the bare Lean ids, the snake_case slugs, the Elman/Emender naming gap, the "v0.3" release bookkeeping. **Symptom count: dozens, across ~15 units.** The abstract proves this is a *stance choice and not a constraint*: for ~150 words it holds the disciplined outside-in voice perfectly (no E88, no "production," "a machine-checked proof" rather than "the Lean 4 trusted core"), then the body's first sentence drops it. The target voice already exists in the paper.

- **STANCE-B — Result-as-system (R2, the deep error).** The paper treats E88 — *one training run demonstrating a possibility* — as a deployed, versioned, maintained product with a biography. This single ontology error generates the entire "production/deployed/released-production-weights/production-stack/production-architecture/production-Emender" thread (≥11 units), the lineage-as-autobiography framing ("endpoint of a multi-year ablation lineage," the E63/E70–E75 milestone bullets), and the anthropomorphism cluster (an architecture/optimizer/memory with agency and temperament). The paper **contradicts itself**: §1c row 1 and §1's body call E88 "one artifact shown to exist" / "the 1.3 B *instance*," directly against the opener's "the 1.273 B-class *production instance*." The disciplined ontology, like the disciplined vantage, is present in the same document one column-width away.

- **STANCE-C — Pre-litigation (R3).** The paper argues with an absent skeptic inside the prose that makes each claim, rather than confining defense to the explicit scope/non-claim structure it built for exactly that purpose. This generates the largest raw category: the honesty-protesting, the relitigated nulls and caveats (the loss-tie diagnostic restated 3×, the plateau concession restated ≥5×, the "not a ranking" caveat 4× in one caption, the occupancy≠peak point 3× in one paragraph), and the personified-critic constructions.

These three are not independent: B drives part of A (the "production"/codename world is the same insider world), and C is what makes A and B *feel* defensive rather than merely sloppy. But they are separable diagnoses, and **fixing the three stances would dissolve most of the 126 rows.** A residual ~15–20 rows are genuinely independent local faults (ordering, count mismatches, the table-contract self-contradiction, the register/process leaks) — those are the subject of §4.

---

## 2. The Uncanny inventory (ranked)

Ranked by reported confidence × depth-of-stance-violation. Every UNCANNY span across all 17 registers. **The intro's first sentence appears at #1 — it was not missed (s1a flagged it at 0.9).**

| rank | span (verbatim, abridged) | unit · loc | rule | why it parses fine yet feels wrong | conf |
|----:|----------------------------|------------|------|-------------------------------------|-----:|
| 1 | **"E88, the 1.273 B-class production instance of the Emender, reaches 0.973 bits per byte…"** | s1a · §1 ¶1 s1 (L162) | R1+R2 | Grammatical as a result sentence; but it is the **canonical specimen** — opens inside the project's world with two undefined proper nouns (defined ~30 lines later) and the unpaid word "production." Manufactures no context; assumes it. | 0.9 |
| 2 | "in an actual deployed artifact — the released v0.3 production weights, loaded strictly" | s6c · L1534 | R2 | Reads as a *methodological strength* ("we tested the real weights"), which is what hides the smuggled ontology — three product words (deployed/released/production) asserting a deployment that does not exist. | 0.85 |
| 3 | "the production architecture keeps the conservative settings… not a clean ablation at 1.3 B." | s9 · L2100 | R2 | Idiomatic ML-engineering speech; globally there is no production architecture, and "keeps" casts a result as an ongoing system holding a config. | 0.83 |
| 4 | "Driving the production 1.273 B E88 … through the racer's own training path on a free NVIDIA RTX 6000 Ada" | s4 · L819–821 | R1+R2 | Ordinary methods prose; **three insider tokens stacked in one subordinate clause** ("production," "the racer's own," "free"), none manufactured for an outsider. | 0.82 |
| 5 | "E88 at 0.973 bpb after about 23 **stitched** wall-clock days." | s10 · L… ; s5b · L1002/L1025 | R1(R2) | Quantitatively precise; "stitched" silently imports interrupted/resumed-run logistics the reader cannot decode and the paper never defines. | 0.82 / 0.80 |
| 6 | "the separation survives at the *deployed* 1.3 B scale, on the actual released production weights" | s6b · L1292–1293 | R2 | "actual" intensifies an unearned claim; weights were *released*, nothing was *deployed*. | 0.80 |
| 7 | **"We adopt the audit-recommended wording from the formalization gap analysis:"** | s7 · L1830–1831 | R1/R2 | Reads as scrupulous provenance — which is why it is uncanny; it is **addressed to the wrong audience**. Names an internal QA artifact the reader cannot see; "audit-recommended" imports unverifiable authority. (Process leak — see §4.) | 0.8 |
| 8 | "the same ratio the Emender uses at **production**, $H=370$" | s5a · L971–972 | R2 | The canonical unpaid word as a fixed epithet; nothing local contrasts "production" against any non-production run. | 0.78 |
| 9 | "The 1.3 B wallclock **racer** is one continuously-trained seed per architecture…" | s9 · L1994 | R1 | A project codename used with the definite article as established common ground for a controlled comparison. | 0.78 |
| 10 | "This is the honest scope of the contribution, **not a buried caveat**." | s5b · L1103 | R3 | Locally well-formed but **speaks at the wrong level** — a sentence about the authors' candor, not the result; redundant with the Caveat block below it. (Meta-level slip — see §4.) | 0.78 |
| 11 | "the search, configured for fairness… **independently voted for the architecture's central thesis**" | s5a · L953–956 | R2/R3 | An appealing flourish; an optimizer cannot endorse a thesis, and "independently" pre-empts the "you tuned it to win" charge — reception-management dressed as corroboration. | 0.72 |
| 12 | "the default configuration **carried down from its 1.3 B production stack**" | s6a · L1142–1143 | R2 | Routine methods prose; "carried down from" lends one run a heritable lineage; "production stack" a tier that does not exist. | 0.8 |
| 13 | "…ordering survives in an **actual deployed artifact — the released v0.3 production weights**" | s6b · L1532 | R2 | Three product-ontology words in one clause — the deep R2 error in concentrated form. | 0.75 |
| 14 | "The work began against a workload." / "Pangenomic sequence data runs to terabases…" / "The Emender is the construction that did." | s2 · L399–409 | R1/R2 | Opens Background **inside the project's biography**; the pangenomics motivation is paid off **nowhere** in the paper (orphaned arc), and one architecture is cast as the hero that fulfilled a quest never demonstrated. | 0.80 / 0.78 / 0.70 |
| 15 | "began as a fairness **doctrine, motivated by frustration at undisclosed HPO budgets in nearby papers**" | s5a · L946–948 | R1/R3 | Parses as motivation prose; a *results* section narrating the author's grievance with unnamed rivals is insider autobiography addressed to a reader party to the grudge. | 0.7 |
| 16 | "The **production stack** at 1.3 B parameters exposes 370 small independent heads…" | s3a · L639 | R2 | Routine ML caption; silently upgrades a result into a serving stack. | 0.7 |
| 17 | "NC#super[1] paragraph **(verbatim)**" | s7 · L1828 | R1 | A *heading* — the most reader-facing furniture — carrying a typesetting/collaboration instruction ("do not re-edit") that points at a document the reader cannot see. (Process leak — see §4.) | 0.7 |
| 18 | "a complexity class that **the substrate of the actual world** does not" | s2 · L… | R2 | A precise complexity-theory sentence suddenly waxes cosmic; an unpaid metaphysical connotation clashing with the clinical register everywhere else. (Register leak — see §4.) | 0.62 |
| 19 | the "Claim" column of `tab_claims` holding propositions the paper **denies** (rows 2 & 4: "Bulk LM loss distinguishes…" / "Raw-write *cannot* do $S_5$" → "*No.*" / "*Not claimed.*") | s1b · #2 ; s1c · #1 | R1/flow | Each cell is locally admirable; the **table breaks its own announced contract** ("each load-bearing claim"). A reader scanning the "Claim" column reads two *rejected* propositions as asserted. (Instrument-contract fault — see §4.) | 0.6 / 0.6 |
| 20 | "the architecture **emended its own name** by the same process it performs on memory" | s3a · L518–520 | R2/R1 | Reads as charming wit; instantiates the frame's named anti-pattern (architecture naming itself) and grants a math object a biography. | 0.6 |
| 21 | quadruple null-restatement in `tab_claims` row 2 ("…distinguishes…" / "*No.*" / "the loss tie is real" / "it does not") | s1b · #1 | R3 | The same null asserted four times in one row; the trailing "— it does not" adds zero information — the relitigation tell. | 0.7 |
| 22 | "A three-row ablation… **isolates the write rule by elimination**" | s3b · L729/L708 | R2/R3 | Parses as a standard ablation sentence; the three rows are whole distinct architectures, not one model with a component toggled — the truth-table format lends an accuracy-gap argument the finality of a logical proof. | 0.45 |

Lower-confidence UNCANNY / borderline (logged, not ranked): "What surfaced was an instrument." (s5a, narrative beat, 0.5); "the latch holds comfortably" (s7, register break, 0.4).

**Observation on the inventory:** the UNCANNY peaks cluster at **structural seams** — section openings (intro s1, §2 opening, §6 "deployed" heading), headings (s7 "(verbatim)"), and the result-object's first naming. They are dense where the text *introduces* something (a section, a run, a result), which is exactly where vantage and ontology choices are forced.

---

## 3. Cross-lens overlaps (≥2 auditors — the most reliable defects)

Spans/threads flagged independently by two or more auditor nodes. Treat these as the highest-confidence findings in the entire audit; independent convergence is the strongest signal available.

| thread | flagged by | reliability |
|--------|-----------|-------------|
| **"production" / "deployed" / "production stack/instance/architecture" / "released production weights"** | s1a, s2(voice), s3a, s3b, s4, s5a, s6a, s6b, s6c, s9, s11_14 — **11 units** | **Maximal.** The single most-corroborated defect in the paper. Every auditor who met it independently flagged it as R2 RESULT-AS-SYSTEM and most explicitly named it the paper's canonical unpaid word. |
| **"the racer"** (codename for the loss-vs-wallclock comparison, used before §5 defines it) | s2, s4, s5a, s5b, s9, s10, s11_14 — **7 units** | Very high. Used bare in §1/§2/§4 before its operational introduction in §5; every downstream unit inherits it with the definite article. |
| **"stitched" wall-clock/GPU-days** (never defined paper-wide) | s1a, s5b, s9, s10 (+ s2 voice) | High. Seeded in the intro's first sentence; recurs to the conclusion; defined nowhere. |
| **"honest" / honesty-protesting register** | s1b, s1c, s4, s6a(voice), s7(flags paper-wide) | High. "keep the reading honest," "Honest null," "Honest mirror," "the honest claim," "the honest profile" — a paper-wide self-vouching budget. |
| **"Claim" column of `tab_claims` holds disowned propositions** (rows 2 & 4) | s1b, s1c — same span, 2 independent auditors | High (same span, independent convergence on the table-contract self-contradiction). |
| **"the comma-pile control"** (definite article, defined only in Appendix §14) | s1b, s1c — same span | High. Used-before-defined in the passage whose job is "scope legible up front." |
| **Duplicated capacity-floor argument** (131,072 scalars ≈ 5 orders above the bit-floor) | s3b (L762–768), s6a (L1130–1137) — near-verbatim in two sections | High. Two homes for one computation; both auditors flag for dedup ownership. |
| **`H = 370` / "many small heads" thesis** ("throughput from width") relitigated across §4 and §5 | s4 (anchor), s5a (cross-flag) | Medium-high. |
| **Anthropomorphic optimizer / "the search voted"** | s5a (finding), s5b (cross-flag correlates the thread) | Medium. |
| **`S₅` = 120 elements / smallest non-solvable / Barrington-NC¹** over-defined | s2 (twice within §2, + notes §1 and §7) | Medium. A paper-level redundancy spanning §1/§2/§7. |
| **"trusted" core + `sorry`/`admit`/`axiom` checklist** recited across sections | s7, s10, s9(notes) | Medium. Legitimate per-use; the defect is cumulative incantation. |
| **M²RNN vs M²RNN-CMA name-variant slippage** | s3b (within @tab_ablation), s1/s5 (cross-flag) | Medium. |

**Note:** the "production" and "racer" threads are so heavily cross-flagged that every auditor independently warned the synthesizer *not to double-count* them per-section. They are counted here as **one systemic defect each**, surfacing in 11 and 7 places respectively.

---

## 4. The UNNAMED synthesis (PRIMARY discovery deliverable)

### 4.1 The collected UNNAMED findings

| span | unit | the discomfort, precisely |
|------|------|---------------------------|
| "the enabling direction" | abstract #1 | self-referential label for a two-direction dichotomy the reader has not been given |
| "matching strong … baselines" | abstract #3 | register runs *prouder* than the body's own scoped framing ("parity-class, not superiority") |
| "Claim" column holds rejected propositions | s1b #2, s1c #1 | a structured device contradicts the contract it announced |
| "the substrate of the actual world" | s2 #4 | clinical prose suddenly waxes cosmic/metaphysical |
| three-vs-four load-bearing count | s3a #4 | figure says 3 choices, prose says 4, and the members differ |
| "fairness doctrine, motivated by frustration…" | s5a #3 | author's emotional/biographical state offered as method provenance |
| "What surfaced was an instrument." | s5a #4 | a methods observation narrated as a literary reveal / story beat |
| "not a buried caveat" / "this is the honest scope" | s5b #4 | a sentence about how to judge the authors' candor, not about the result |
| "not two anecdotes but one mechanism… the culmination" | s6c #4 | the author scoring their own work's importance for the reader |
| "parity-rate evidence" (heading) | s6c #7 | a heading names a metric ("rate") the prose never delivers |
| "(verbatim)" + "audit-recommended wording from the formalization gap analysis" | s7 #1, #2 | the project's internal QA/editorial process leaked onto the published surface |
| "the latch holds comfortably" | s7 #3 | conversational reassurance dropped into tight theorem prose |
| "moat around the trusted surface" | s9 #5 | a siege metaphor positions the reader as a besieger |
| "retains the lead" | s9 #8 | a race metaphor personifies the result as a contestant |

### 4.2 Do they share a structure? **Yes — and it is the audit's central discovery.**

Strip away the four findings that are really *instrument-contract* faults (the "Claim" column, the three-vs-four count, the "parity-rate" heading — see §4.4) and the remaining UNNAMED findings collapse onto one shared mechanism:

> **Each is a sentence that is locally well-formed but is operating on a *different plane* than the one a results paper occupies — it changes what *kind of utterance* it is, without warrant.**

This is exactly why they are UNCANNY rather than merely MESSY: each plane is internally coherent (the prose parses, the metaphor is apt, the provenance is scrupulous), so the reader cannot point to a grammatical fault — but the *floor shifts* under them. Three sub-modes recur:

- **(a) Process / build leak** — the text stops addressing the reader and addresses the *project's own collaborators*: "(verbatim)," "audit-recommended wording from the formalization gap analysis," the conclusion trailing off into measurement-script file paths, "v0.3" release bookkeeping, the E63/E70–E75 milestone bullets. The reader is suddenly standing inside the build system and the QA loop.
- **(b) Meta-level slip** — the text stops talking about the *result* and talks about *how the reader should judge the authors*: "not a buried caveat," "this is the honest scope," "not two anecdotes," "the culmination," "independently voted for the thesis," the whole honesty-protesting register. The sentence is about the paper's reception and the authors' credibility, not its content.
- **(c) Register / affect leak** — the clinical, measured baseline register is broken by a register the surrounding prose does not occupy: cosmic ("the substrate of the actual world"), colloquial ("holds comfortably"), emotional ("frustration"), or narrative ("what surfaced was an instrument," the race/siege/quest metaphors, "the construction that did").

The common denominator: **a stable utterance-type — measured report, addressed to the outside reader, about the result — silently swapped for another utterance-type.** The existing frame does not name this. R1 governs *vantage* (does the text manufacture context), R2 governs the *ontology of the result-object*, R3 governs *defensive posture*. None of them catches "the text just changed what sort of thing it is saying, and to whom." That is the gap the UNNAMED pile exposes.

### 4.3 Proposed missing frame rule

> **R4 — PLANE / REGISTER DISCIPLINE.** The paper speaks on one plane: a *measured report*, *addressed to the outside reader*, *about the result*. It must not slip the plane without warrant. **Violations:** (a) **process leak** — addressing the project's collaborators, build system, or QA loop ("verbatim," "audit-recommended," internal artifact names, file paths and version strings as reader-facing content); (b) **meta-level slip** — addressing the reader about the authors' own credibility, honesty, or the work's importance, rather than about the result ("the honest scope," "not a buried caveat," "the culmination," reception-management); (c) **register/affect leak** — breaking the surrounding clinical register into the cosmic, colloquial, emotional, or narrative ("the substrate of the actual world," "holds comfortably," "frustration," "what surfaced was an instrument," race/siege/quest metaphor). **Diagnostic:** the span is locally grammatical but is a *different kind of utterance* than the one a results paper makes; the reader feels the floor shift while unable to name a grammatical fault.

R4 cleanly absorbs the UNCANNY findings at ranks #7, #10, #11, #17, #18 in §2, and most of the §4.1 list. It also sharpens the diagnosis of the existing anthropomorphism thread: "the search voted," "memory can change its mind," "E88 invites" are R4(c) register leaks *and* R2 ontology errors — the two rules co-fire, which is why those spans feel doubly wrong.

### 4.4 Secondary residue → candidate R5 (lower confidence)

Three UNNAMED findings do **not** fit R4: the "Claim" column holding disowned propositions (s1b/s1c), the three-vs-four load-bearing count (s3a), and the "parity-rate evidence" heading promising an undelivered metric (s6c). These share a *different* structure: **a structured device (table, figure, enumerated list, heading) violates the contract it announced about itself.** The "Claim" column should hold claims; a figure asserting "three design choices" should match prose enumerating "four"; a heading naming a "rate" should be followed by a rate. Proposed, held at lower confidence because only ~3 instances:

> **R5 — INSTRUMENT INTEGRITY.** A structured device must obey the contract it announces. A column labelled "Claim" must contain claims; a figure's count must match the prose's; a heading must name what the body delivers. **Violation:** the reader builds a model from the device's announced structure and is then contradicted by its contents.

R5 is offered as a candidate, not asserted; two of its three instances are the same span (the table) seen by two auditors, so the support is thinner than R4's.

---

## 5. Frame revision

**Are R1–R3 sufficient?** No. They capture STANCE-A (R1), STANCE-B (R2), and STANCE-C (R3) — and those three account for ~106 of 126 rows — but they leave the entire UNNAMED pile (the source of five of the highest-confidence UNCANNY findings) unnamed. R4 is necessary. R5 is a candidate.

### Revised frame (held as hypothesis, not law)

- **R1 — Vantage.** Written for a reader *outside* the project who knows nothing of an Emender. Manufacture context, never assume it. *Violations:* codenames as shared history; undefined insider terms ("stitched," "the racer," "the campaign," "comma-pile," "on the rack"); results stated before the reader has reason to care. *(Sharpened: the abstract proves the disciplined vantage is achievable — R1 violations are lapses from a voice the paper already owns, not an unavoidable cost of technical density.)*
- **R2 — Ontology.** E88 is a RESULT (one training run demonstrating a possibility), not a deployed system/creature/entity with a biography. *Violations:* "production"/"deployed"/"released production weights" (there is no production); architecture naming itself; memory/optimizer with will or temperament; lineage as autobiography. *(Sharpened: the paper contradicts itself — §1c calls E88 "one artifact shown to exist"; the opener calls it "the production instance." The corrective ontology is in the document already.)*
- **R3 — Posture.** State claim, state scope, move on. Defense belongs in the explicit scope/non-claim structure, not woven into every sentence. *Violations:* arguing with an absent skeptic mid-paragraph; the same point relitigated; hedging exceeding its claim; honesty-protesting. *(Sharpened: §11 Predictions is the paper's own positive R3 anchor — claim + scope + explicit falsifier, then stop — and shows the structured form works.)*
- **R4 — Plane / register discipline. (NEW.)** Speak on one plane: a measured report, to the outside reader, about the result. Do not slip to (a) addressing collaborators/the build/QA loop, (b) addressing the reader about the authors' credibility or the work's importance, or (c) a register (cosmic/colloquial/emotional/narrative) the surrounding prose does not occupy. *Diagnostic:* locally grammatical, but a different kind of utterance than a results paper makes.
- **R5 — Instrument integrity. (CANDIDATE, lower confidence.)** A structured device obeys the contract it announces: a "Claim" column holds claims; a figure's count matches the prose's; a heading names what the body delivers.

**Categories** are unchanged except: add **PLANE-SLIP** (R4) — process leak / meta-level slip / register leak — and **CONTRACT-BREAK** (R5) as a candidate. The previously catch-all UNNAMED tag should, going forward, resolve mostly into PLANE-SLIP.

---

## 6. The Validation Test

**Question:** reading ONLY this synthesis (not the paper), can a reader predict where the discomfort lives and reconstruct the "uncanny and fucked up" reaction?

**Honest answer: mostly yes for *location* and *source*; partially no for *cumulative intensity* and for §8.**

What this synthesis **does** let a reader reconstruct:
- **Where.** The discomfort concentrates at structural seams — the intro's first sentence, §2's opening, the §6 "deployed" heading and its captions, §7's "(verbatim)" heading, the conclusion's trailing file paths. A reader of §2 above could point at those spots blind.
- **Why.** Three stance sources (A vantage, B ontology, C pre-litigation) plus one register fault (R4 plane-slip). The "uncanny" feeling is explained: the spans parse because each plane/ontology is internally coherent, so the wrongness is at the level of *who is speaking, to whom, about what* — exactly the "locally grammatical, globally off" signature. A reader could reconstruct the core reaction: *the paper is talking past me, from inside a world I'm not in, about a creature that does not exist, while arguing with someone who isn't me.*

What it **misses** — the layers an additional pass is needed for:
1. **§8 Related Work is un-audited (coverage hole).** No register exists. Two auditors punt competitor-framing pre-litigation *into* §8, so it is the predicted home of a defect class no one logged. A reader of this synthesis would wrongly infer §8 is clean. **Additional pass needed: `texaudit-s8-fill` (filed).**
2. **Cumulative / compounding intensity is under-captured.** The audit is span-local with per-span confidences. But the "fucked up" reaction is partly *additive* — meeting "production" for the 11th time, "the racer" for the 7th, "honest" yet again, is a compounding dread that no single 0.8-confidence row conveys. This synthesis can state the *density* (§1, §3) but cannot reproduce the *felt accumulation*. **Additional pass needed: a whole-read "compounding" pass that scores the experience of recurrence, not individual spans.**
3. **Prosody / syntax / cadence was never measured.** Every auditor worked at word- and sentence-stance level; none scored the hedge-heavy, clause-stacked *rhythm* of the prose. Part of "feels wrong" may live in syntax (the relentless subordinate-clause qualification) that no register row captures. **Additional pass needed: a read-aloud cadence/syntax pass.**

So: the synthesis is a faithful *map* of the discomfort's location and cause, sufficient to predict where a reader will flinch and to explain the uncanniness — but it is not a full *reproduction* of the reaction's intensity, and it has one literal blind spot (§8). Those three passes would close the gap.

---

## 7. Section verdict table

| unit | verdict | one-line justification |
|------|---------|------------------------|
| abstract | **MESSY** | Cleanest unit; 4 residual MESSY rows, **no UNCANNY** — the disciplined outside-in voice the body abandons (the target stance, proven achievable). |
| s1a (intro opener + contributions) | **UNCANNY-bearing** | The canonical first-sentence specimen (#1): INSIDE-OUT + RESULT-AS-SYSTEM stacked in sentence 1; the hedged contribution label leaks scope-defense into a headline. |
| s1b (claims table) | **UNCANNY-bearing** | The "Claim" column holds disowned propositions; the row-2 null is restated four ways — table contract self-contradiction + relitigation. |
| s1c (claims passage) | **UNCANNY-bearing** | Same table-contract self-contradiction; otherwise the best-behaved §1 sub-unit (row 1 gets the R2 ontology *right*). |
| s2 (background) | **UNCANNY-bearing** | Three UNCANNY: an orphaned pangenomics autobiography paid off nowhere, "the substrate of the actual world," "the Emender is the construction that did." |
| s3a (per-head update) | **UNCANNY-bearing** | The architecture "emended its own name" (anthropomorphic self-naming) + "production stack" caption; the equation prose itself is clean. |
| s3b (ablation) | **UNCANNY-bearing** | "Isolates by elimination" lends a confounded 3-model comparison the finality of a proof; otherwise MESSY relitigation (conclusion stated 3×). |
| s4 (systems) | **UNCANNY-bearing** | The L819 methods sentence stacks three insider tokens ("production"/"racer's own"/"free"); occupancy≠peak relitigated 3×. |
| s5a (CMA-ES setup) | **UNCANNY-bearing** | "Independently voted for the architecture's central thesis," "fairness doctrine motivated by frustration," "at production H=370" — all in one narrative paragraph. |
| s5b (racer / tie) | **UNCANNY-bearing** | "stitched" + "this is the honest scope … not a buried caveat" (meta-level slip); the loss-tie caveat relitigated 3×. |
| s6a (8 M probes) | **UNCANNY-bearing** | "Default config carried down from its 1.3 B production stack"; capacity-defense front-loaded before the result that provokes it. |
| s6b (deployed 1.3 B scale) | **UNCANNY-bearing** | **Densest** "deployed/production/released-production-weights" cluster (3 UNCANNY); the plateau concession relitigated ≥5×. |
| s6c (micro/mega) | **UNCANNY-bearing** | "Actual deployed artifact — the released v0.3 production weights, loaded strictly"; "a critic can always ask" twice in one paragraph. |
| s7 (formal results) | **UNCANNY-bearing** | Internal-process leak: "(verbatim)" heading + "audit-recommended wording from the formalization gap analysis" — yet otherwise the most R2-clean section. |
| **s8 (related work)** | **NOT AUDITED** | **No register written; coverage gap. Predicted home of competitor-framing pre-litigation (punted here by s2, s6a). Audit pending (`texaudit-s8-fill`).** |
| s9 (limitations) | **UNCANNY-bearing** | "Production architecture keeps the conservative settings" + "the 1.3 B wallclock racer"; subsections argue instead of scoping — but the close ("scope, not result") is the clean R3 anchor. |
| s10 (conclusion) | **UNCANNY-bearing** | "stitched" + "§5 has the racer in full"; the conclusion has no conclusion — it trails into repo file paths (R4 process leak). |
| s11_14 (predictions / future / appendices) | **MESSY** | **No UNCANNY** — §11 Predictions is the paper's positive R3 anchor (claim + scope + falsifier); residual "production" in the §13 lineage appendix and a "not a ranking" caption relitigated 4×. |

**No unit is fully OK.** The two MESSY-only units (abstract, s11_14) are the paper's cleanest and double as the proof that the disciplined voice/posture exists within the document. Eleven of seventeen audited units are UNCANNY-bearing — the uncanny is not localized, it tracks the three stance sources wherever the text introduces a section, a run, or a result.

---

*End of diagnosis. No edits recommended; the editing phase is separate and gated on human review of this document. One follow-up audit task (`texaudit-s8-fill`) and two additional-pass recommendations (compounding-read, cadence/syntax) are recorded in §6 as gaps for the human reviewer to schedule, not as edits to the paper.*
