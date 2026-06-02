# CROSS-TRACK EXECUTIVE SUMMARY

**Subject:** Garrison, *Emending Nonlinear Recurrence* (`paper/main.typ`)
**Mode:** AUDIT / DIAGNOSIS ONLY. No edits to `main.typ`. This document aggregates
**two independent review tracks** and ends at *strategy* — it recommends the **scope**
of an eventual, separate, human-gated editing phase, not specific edits.
**Inputs aggregated:**
- **Track A — Texture & flow:** `texture_audit/SYNTHESIS.md` (master fan-in over 18 nodes)
  + the per-section registers (`abstract`, `s1a/b/c`, `s2`, `s3a/b`, `s4`, `s5a/b`,
  `s6a/b/c`, `s7`, `s8`, `s9`, `s10`, `s11_14`). §8 (`s8.md`) was the one register the
  master synthesis could not aggregate (original node failed); it has since been filled
  (`texaudit-s8-fill`) and is folded in here.
- **Track B — Claim-vs-evidence & experiments:** `BL1_adjudication`, `BL1_provenance_recheck`,
  `BPB_CONTEXT`, `BPB_FULL_TABLE`, `PILE_BPB_MEASURED`, `HELDOUT_MULTISLICE`, `COMMA_PILE_BPB`,
  `COMPRESSION_BPB`, `S3_S5_FINETUNE`, `S3_S5_FINETUNE_V03`, `S5_SYMMETRIC_BUDGET`, `THROUGHPUT`,
  `HF_V03_FIX`, `HF_V03_REPUBLISH`, `PINNED_CHECKPOINTS`, `E88_HELDOUT_HF`, `E88_HELDOUT_HARNESS`,
  `V3_GATE`, `V3_NUMBERS` (19 files; `S3_S5_HF_TEST` does not exist on disk).

---

## 1. TOP-LINE

**The science is sound and, once scoped, conservative; the paper's *framing* is one notch
stronger than the work it sits on, and that gap is the whole problem — in both tracks.** The
underlying contribution is real: a purely nonlinear (delta-correcting) recurrent LM trained to
1.27 B parameters that (a) lands in the **same sub-1-bpb, parity-class band** as strong
linear-recurrent baselines on held-out Pile, (b) shows a **clean, theory-backed expressivity
separation** from linear recurrence on the non-solvable S₅ word problem that is **robust to a
10× training-budget swing**, and (c) recovers **linear-scan-class throughput on the width axis**
without a sequential time scan. Every load-bearing experimental worry the reviews raised has now
been *settled in the evidence* — the held-out tie, the S₅ ceiling-vs-undertraining question, the
throughput claim, the baseline-selection asymmetry, the broken public release — and in **every
case the honest result is more modest than the paper's current surface**: parity not superiority,
efficiency not impossibility, occupancy not peak-FLOPs, a disclosure asymmetry not a clean
matched comparison, a release that was actually broken-then-fixed. Track A independently reaches
the mirror-image conclusion about the prose: ~126 texture defects collapse onto **three deep
stances** plus a register fault, and the paper's own abstract proves the disciplined,
outside-facing voice already exists in the document — the body just abandons it. **The mess is
concentrated, diagnosable, and not load-bearing on the results. This calls for a stance-level
revoice plus a bounded set of mandatory factual reconciliations — not a rewrite.**

---

## 2. THE AGGREGATE — ONE LEDGER OF WHAT IS REAL

### 2.1 Track A (texture & flow) — what the prose audit found

- **Density.** 126 defect rows across 17 audited units; ~28 rated UNCANNY ("parses fine, feels
  wrong"). **11 of 17 units are UNCANNY-bearing.** Only two units are MESSY-only — the **abstract**
  and **§11–14** — and those double as the paper's *positive anchors* (proof the target voice/posture
  exist in-document). [`SYNTHESIS.md` §1, §7]
- **Three stance sources generate the great majority of symptoms** (confirmed with high confidence,
  "few deep sources, not many shallow defects"):
  - **STANCE-A — Vantage collapse (R1):** the paper writes *from inside its own project* — undefined
    codenames ("E88"/"the Emender" at first use, "the racer," "stitched," "comma-pile control," "on
    the rack," bare Lean ids, snake_case slugs). Dozens of symptoms across ~15 units.
  - **STANCE-B — Result-as-system (R2, the deep error):** E88 — *one training run demonstrating a
    possibility* — is repeatedly handed a **product biography it does not have**: "production
    instance / production stack / production architecture / deployed / released production weights."
    This is the **single most cross-flagged token in the entire audit** (11 units, every auditor who
    met it flagged it independently). [`SYNTHESIS.md` §3]
  - **STANCE-C — Pre-litigation (R3):** the paper argues with an **absent skeptic inside the prose**
    rather than confining defense to the explicit scope/non-claim structure it already built — the
    honesty-protesting register ("honest null," "honest scope," "not a buried caveat"), nulls and
    caveats relitigated 3–5× each.
- **Discovered frame rules.** The "unnamed" residue resolves into a new rule the audit proposes:
  **R4 — Plane/register discipline** (process/QA leaks like "(verbatim)" and "audit-recommended
  wording"; meta-slips about the authors' credibility; register leaks — cosmic/colloquial/emotional/
  narrative), plus a lower-confidence candidate **R5 — Instrument integrity** (a "Claim" column that
  holds *disowned* propositions; a figure count that contradicts the prose). [`SYNTHESIS.md` §4–5]
- **The central qualitative finding:** *the body is more honest than the framing.* §1c calls E88
  "one artifact shown to exist"; the opener one column-width away calls it "the 1.273 B-class
  production instance." The abstract holds a strict outside-in voice for ~150 words, then the body's
  first sentence drops it. **The discipline is a choice the paper can already make, not a cost of
  technical density.** [`abstract.md` cross-section flags; `SYNTHESIS.md` §1.3]
- **§8 (filled):** confirmed the predicted competitor-framing **pre-litigation** cluster — rivals
  "escorted out" by §1 criteria via adversarial verbs ("violate"/"fail"), a GDN-2 timeline alibi,
  "concurrent prior art" importing a priority-dispute register. Notably §8 is **R2-clean** (no
  "production" leak) — a second clean island alongside §7. [`s8.md`]
- **Two axes the audit admits it under-measured** (honest severity caveat): **compounding intensity**
  (the felt dread of meeting "production" for the 11th time is additive and not captured by
  per-span 0.8 confidences) and **prosody/cadence** (the clause-stacked, hedge-heavy *rhythm* was
  never scored). [`SYNTHESIS.md` §6]

### 2.2 Track B (claims & experiments) — what the experiments actually SETTLED

| Worry the reviews raised | What the evidence settled | Honest disposition |
|---|---|---|
| Does the train-loss bpb ordering (E88 < GDN < M²RNN-CMA) hold **held-out**? | **No — TIE.** Across 5 independent Pile slices the three sit in a **0.0058-BPB band < cross-slice noise (0.0254)**; E88/GDN tie to 0.00002 BPB; M²RNN-CMA is marginally *lowest* (4/5 slices). | **Parity, not superiority.** [`HELDOUT_MULTISLICE`, `E88_HELDOUT_HARNESS`, `BPB_FULL_TABLE`] |
| Is the bpb result **contamination-inflated** (Pythia/Neo saw the Pile)? | **No.** On a clean Common-Pile slice the Pile-trained refs move ≤0.0028 BPB. | Worry closed. [`COMMA_PILE_BPB`] |
| Is "LM-as-compression" backed by a fair anchor? | Classical coders floor at **2.19–2.80 BPB** (xz -9 = 2.19) on the identical bytes — neural is well clear. | Confirmed as **anchor, not leaderboard**. [`COMPRESSION_BPB`, `BPB_CONTEXT`] |
| Was E88's S₅ length shortfall (0.162 @ T=512) **under-training**? | **No — a real capacity/expressivity wall.** At a symmetric, a-priori, 2× budget E88's T=512 accuracy **plateaus at ≈0.14, flat from step 12k**; three recipes converge to ≈0.14. | **Efficiency, not impossibility.** [`S5_SYMMETRIC_BUDGET`] |
| Does the delta-vs-linear separation survive at trained length & at scale? | **Yes, robustly.** E88(delta) > M²RNN(raw-write) > GDN(linear) at **every** length to T=1024, robust to a 10× budget swing and a recipe change. GDN is a *converged* failure (competent on solvable S₃, fails non-solvable S₅) — clean expressivity test. | Confirmed; the **clean** pure-expressivity contrast is **E88-vs-GDN** (M²RNN's shortfall is entangled with optimization). [`S3_S5_FINETUNE`, `S3_S5_FINETUNE_V03`, `S5_SYMMETRIC_BUDGET`] |
| Is "saturates the GPU / full utilization" true? | **Yes in the occupancy sense** (median 100% util, 97% of power cap) — **but MFU is only ≈15.7%.** E88 sustains 7,492 tok/s ≈ **91% of a real linear-scan FLA-GDN kernel** at matched budget. | **Occupancy, not peak-FLOPs; parity-class, not superiority.** [`THROUGHPUT`] |
| Is §6's "matched no-tuning across architectures" honest? | **No — a real disclosure asymmetry (BL-1 SUBSTANTIATED).** The load-bearing `tanh` was tie-broken on a **state-tracking** proxy when LM loss was indifferent; no baseline used a state-tracking criterion. The honest caveat exists but is mislocated in §9. S₃ control + NC¹ theorem neutralize the **direction/mechanism**, not the **magnitude** of the within-class S₅ gap. | **Disclosure/labeling blocker** (not results-invalidity); verdict re-confirmed on provenance re-check, citation sharpened. [`BL1_adjudication`, `BL1_provenance_recheck`] |
| Does the **released** public artifact reproduce the paper? | **It did not — it was catastrophically broken** (18–102 nats/token, worse than random) because the v0.3 export froze schedule-free **x-mode** weights with **zero optimizer state**. Root-caused to the *weights* (not config/graph), fixed by a y-mode re-export that reproduces the harness to **≤2×10⁻⁵ nats**, and the public v0.3 repos were overwritten with working weights. | **Broken → fixed.** Cite the corrected revision. [`HF_V03_FIX`, `E88_HELDOUT_HF`, `HF_V03_REPUBLISH`, `PINNED_CHECKPOINTS`] |
| Are the Figure-2 numbers current/traceable? | Recomputed; the leaderboard **flipped late** (previously GDN led at 0.970 / E88 second; now E88 leads at 0.974, driven by *more training*, not a constant change). Constant corrected (×0.368, not the old ×0.391). V3 consistency gate: **GO** (8/8). | Numbers consistent; **still-live, do-not-cite-as-final** retained. [`V3_NUMBERS`, `V3_GATE`] |

**The one-sentence Track B ledger:** every headline survives *as a scoped, relative, parity-class
claim*; **none survives as the stronger absolute claim the surface currently implies**, and the
one literally-deployed artifact had to be repaired.

---

## 3. THE PATTERN, AND WHY IT FAILS

**Hypothesis under test (from the task):** texture's stance defects and the claim-track's
overclaiming are the *same* underlying pattern — **framing/claims running ahead of what an outside
reader and the evidence support**, the author's deep immersion in a long solo research process
leaking onto the surface.

**Verdict: CONFIRMED, with one refinement.** This is one mechanism wearing two costumes, and the
cross-track evidence is unusually direct:

1. **The most-flagged texture defect is literally an overclaim that the claim track proved false.**
   Track A's single most cross-corroborated defect is the R2 "production / deployed / released
   production weights" thread (11 units). Track B then found that the **released v0.3 artifact was
   in fact non-functional** — worse than random — until repaired [`HF_V03_FIX`, `E88_HELDOUT_HF`].
   The prose's "deployed product" stance was not merely a register lapse; it was **framing that had
   outrun the artifact**. Same error, surfaced independently by both tracks.

2. **Pre-litigation (R3) and overclaim-then-scope-down are the same gap viewed from two angles.**
   The texture defect is *defending each claim mid-sentence against an absent skeptic*; the claim
   defect is *a headline that an outside reader/reviewer would over-read* (bpb "win," S₅ "solves,"
   "full utilization," "matched no-tuning"). Both are the distance between **what is asserted** and
   **what an outsider/the evidence grants** — pre-litigation tries to *talk that gap closed*; the
   overclaim *opens* it. Every Track B file is a record of that gap being measured and the framing
   pulled back: "should not over-claim the train-loss ordering as an architecture verdict"
   [`BPB_FULL_TABLE`], "efficiency-not-impossibility" [`S5_SYMMETRIC_BUDGET`], "occupancy, not
   peak-FLOPs … parity-class, not superiority" [`THROUGHPUT`].

3. **BL-1 is the pattern in a single object.** The label "matched no-tuning across architectures"
   is *simultaneously* a Track-A framing defect (a confident surface claim) and a Track-B claims
   defect (it conceals a real selection asymmetry). And the honest caveat **exists** — it is just
   **mislocated in §9 instead of §6** [`BL1_adjudication`]. That is the texture finding "*the body
   is more honest than the framing*" expressed as a claim-track blocker.

4. **The "self-immersion" reading is corroborated structurally, not just asserted.** §8 shows the
   product ontology **drops exactly where the paper faces external literature** (R2-clean, like §7)
   and **reappears when it narrates its own run** [`s8.md`]. The overclaim is tied to *self-narration
   moments*, which is the signature of an author writing from inside a world the reader is not in.

**The refinement — why this is fixable rather than fatal.** The pattern is **not that the claims
are false.** The science is sound and the author has *already done the honest work* — the entire
Track B corpus exists because the author commissioned/ran the adversarial checks, and the body
prose already states the scoped truths. So the defensive register (R3) is the **residue of real
self-checking**, and the honesty is present but **distributed wrong**: defense smeared into every
sentence instead of localized to the scope structure; headline pride not yet reconciled with the
already-scoped body. The precise failure is: **the authoring surface (prose stance *and* headline
framing) renders the work one notch more finished / more deployed / more superior / more
adversarially-settled than the calm evidence underneath — and the body and experiments are both
more honest than the surface that introduces them.**

**Why it recurs:** a long solo research process builds an interior world — E88 *is* "the production
Emender," rivals *have been* litigated, each null *has been* argued to rest — and that interior
runs ahead onto the page at exactly the moments the text introduces a section, names the result,
or meets a competitor. The defect clusters at structural *seams* (Track A §2) for the same reason
the overclaims cluster at *headlines* (Track B): both are where the author's settled internal
conviction is forced into a sentence before the outside reader has been given the same grounding.

---

## 4. FIXED vs OPEN — LEDGER ACROSS BOTH TRACKS

> **Critical scoping note.** "FIXED" below almost always means **the evidence/diagnosis now
> exists** — the experiment was run, the blocker adjudicated, the release repaired. Because both
> tracks were audit-only, **`main.typ` itself has not yet been updated** to reflect most of these.
> Absorbing the settled evidence into the paper surface *is* the open editing phase.

### FIXED (settled in the evidence / artifacts)
- **Held-out bpb measured + multi-slice robustness** → three-way tie established. [`E88_HELDOUT_HARNESS`, `HELDOUT_MULTISLICE`, `BPB_FULL_TABLE`]
- **Contamination ruled out** (Common-Pile cross-check). [`COMMA_PILE_BPB`]
- **Classical-compression floor anchored** on identical bytes. [`COMPRESSION_BPB`]
- **S₅ symmetric-budget experiment run** → under-training worry closed; efficiency-not-impossibility; ordering robust to 10× budget. [`S5_SYMMETRIC_BUDGET`, `S3_S5_FINETUNE_V03`]
- **Throughput / MFU measured** → occupancy vs peak-FLOPs distinction quantified; width-axis parity with a real linear-scan kernel. [`THROUGHPUT`]
- **BL-1 baseline-selection asymmetry adjudicated** (SUBSTANTIATED, disclosure-scope) + **provenance re-check** (verdict STANDS, citation sharpened from the superseded ablation notes to the production-scale records). [`BL1_adjudication`, `BL1_provenance_recheck`]
- **Broken HF v0.3 release root-caused and repaired** (x-mode→y-mode re-export, public repos overwritten, reproduces harness ≤2×10⁻⁵ nats); **checkpoints pinned** so y-mode recovery survives. [`HF_V03_FIX`, `HF_V03_REPUBLISH`, `PINNED_CHECKPOINTS`]
- **Figure-2 numbers recomputed + consistency gate GO** (constant corrected; ranking re-attributed to E88-leads). [`V3_NUMBERS`, `V3_GATE`]
- **Track-A positive anchors confirmed to exist in-document:** the abstract (clean outside-in voice) and §11 (claim+scope+falsifier posture). [`abstract.md`, `SYNTHESIS.md` §5]

### OPEN (not yet reconciled onto the paper surface, or not yet measured)
*Claim track — mandatory surface reconciliations:*
- Restate the bpb result as a **held-out parity/tie**, and stop presenting the train-loss ordering as an architecture verdict. [`BPB_FULL_TABLE`, `HELDOUT_MULTISLICE`]
- **Relabel §6** ("matched no-tuning") and **co-locate the §9 selection caveat into §6**, foregrounding the S₃ control. [`BL1_adjudication`]
- State throughput as **occupancy + MFU ≈15.7%**; state S₅ as **efficiency-not-impossibility**. [`THROUGHPUT`, `S5_SYMMETRIC_BUDGET`]
- Update the checkpoint **citation to the corrected revision**; retain do-not-cite-as-final on still-live numbers. [`HF_V03_FIX`, `V3_NUMBERS`]
- **v0.1 / v0.2 releases** were published from the same x-mode pipeline and are *likely also broken* — untouched; any claim referencing them is unverified. [`HF_V03_REPUBLISH`]
- *Optional:* a **stronger raw-write (M²RNN) baseline** trained to clean competence would close the one entangled contrast (no clean pure-expressivity wall is claimable for raw-write today). [`S3_S5_FINETUNE`, `S3_S5_FINETUNE_V03`]

*Texture track:*
- The **three deep stances** still live in the prose: R1 vantage, R2 production-ontology, R3 pre-litigation.
- **R4 plane-slips** (process/QA leaks, meta-slips, register/affect leaks). [`SYNTHESIS.md` §4]
- **§8 competitor-framing pre-litigation** (now diagnosed, not yet addressed). [`s8.md`]
- The **two under-measured axes** — compounding intensity and prosody/cadence — should be passed before the texture audit is declared complete. [`SYNTHESIS.md` §6]

---

## 5. THE REWRITE QUESTION

**Recommendation: a STANCE-LEVEL REVOICE, executed as a small number of *global* passes, gated
behind a bounded set of *mandatory surgical* claim-reconciliations — NOT a full rewrite.** The
strategy in one line: **pull the framing back down to the body, because both tracks' root is
framing-ahead-of-substance and the already-honest body + abstract already show the target.**

The three options, weighed on the real evidence:

**(A) Full REWRITE — rejected.**
- *Against:* The science is sound and the structure works; the defects are **not** load-bearing on
  the results. Crucially, **the body is more honest than the framing** [`SYNTHESIS.md` §1.3] — a
  rewrite would put the *asset* (the calm, scoped body prose and its hard-won caveats) at risk to
  fix a problem that lives in the *surface*. The mess is concentrated in ~3 stances, not diffused
  through the content. And the texture synthesis's own warning cuts hardest here: heavy rewriting
  risks "**smoothing good prose into NEW uncanniness.**"
- *For:* essentially nothing the cheaper options don't also achieve.

**(B) Stance-level REVOICE — recommended (dominant mode).**
- *For:* The defects are a **voice property, not a content property** — three stances plus a register
  fault, each with a known mechanism and, decisively, an **in-document exemplar of the correct
  voice** (the abstract for vantage/ontology; §11 for posture). A revoice keeps every number,
  result, and structural choice and changes only *who is speaking, to whom, about what*: purge the
  product-ontology, lift the insider vantage, confine defense to the existing scope structure. This
  attacks the **root** the §3 analysis identifies, and it is the only option that addresses the
  **compounding-intensity** axis — you cannot fix the felt accumulation of "production ×11" by
  patching spans one at a time; you fix it by removing the *stance* that generates them.
- *Against:* it is judgment-heavy and must be **global, not span-by-span** (see the smoothing
  warning); it should be done by one hand in a few coherent passes, and re-checked against the two
  under-measured axes so the revoice does not introduce new cadence problems.

**(C) Pure SURGICAL edits — necessary but insufficient on their own.**
- *For:* Some findings genuinely **are** point fixes and **must** happen regardless of (B): the
  claim-track reconciliations in §4 (bpb-tie restatement, §6 relabel + §9-caveat co-location,
  throughput/S₅ rewording, citation bump). These are factual corrections an outside reviewer would
  otherwise catch; they are not optional and not stance.
- *Against:* As a *strategy for the texture problem* it fails. The synthesis warns span-patching
  risks new uncanniness; whack-a-mole on individual "production"/"racer" hits leaves the generating
  stance intact and cannot touch the compounding axis. Surgical is the **floor**, not the plan.

**The synthesis, therefore, is hybrid but unambiguous in scope:** the **mandatory factual
reconciliations are surgical** and gate everything (a reviewer rejects on those); the **texture
cleanup is a global stance revoice** anchored on the abstract and §11; a **full rewrite is not
warranted** and is actively counter-indicated by the "body is more honest than the framing" and
"smoothing into new uncanniness" findings. The editing phase should be **one editor, a few global
stance passes, gated on the claim reconciliations landing first, and re-audited on the two axes the
texture pass under-measured.**

---

## 6. PRIORITY ORDER (highest-leverage first, worst-first)

1. **Reconcile the headline factual claims to the settled evidence (mandatory, surgical).** This is
   where framing-ahead-of-substance does the most damage — a reviewer rejects here. Held-out bpb =
   **parity/tie** (retire the train-loss ordering as a verdict); **relabel §6** + co-locate the
   BL-1 caveat; throughput = **occupancy + MFU**; S₅ = **efficiency-not-impossibility**; **citation
   → corrected revision** (and flag v0.1/v0.2). [`BPB_FULL_TABLE`, `BL1_adjudication`, `THROUGHPUT`,
   `S5_SYMMETRIC_BUDGET`, `HF_V03_FIX`]
2. **Purge the result-as-system / "production-deployed" ontology globally (STANCE-B / R2).** The
   single most cross-flagged texture defect *and* the one the broken release made literally false —
   highest-leverage stance move. One global pass; anchor on the abstract's "one artifact shown to
   exist" voice. [`SYNTHESIS.md` §3; `s8.md`]
3. **Lift the vantage (STANCE-A / R1).** Define-or-drop the insider codenames ("the racer,"
   "stitched," "comma-pile," E88/Emender at first use); re-anchor the §1 opener on the abstract's
   outside-in voice. [`SYNTHESIS.md` §3]
4. **Confine the defensive posture (STANCE-C / R3) to the scope structure.** Stop relitigating
   nulls/caveats sentence-by-sentence; clean the §8 competitor-framing. [`SYNTHESIS.md` §1.2; `s8.md`]
5. **Sweep the R4 plane-slips** (process/QA leaks like "(verbatim)" and "audit-recommended wording,"
   meta-slips, register leaks) — lower volume, targeted, after the three stances are pulled. [`SYNTHESIS.md` §4]
6. **Close the measurement gaps before declaring done:** commission the **compounding-intensity** and
   **prosody/cadence** texture passes; optionally train the **stronger raw-write baseline**. [`SYNTHESIS.md` §6; `S3_S5_FINETUNE_V03`]

---

*End of diagnosis. Diagnosis only — no edits to `paper/main.typ`; this document recommends the
**strategy and scope** of the eventual, separate, human-gated editing phase, not specific edits.
Every claim above is traceable to a named review file in `paper/review/`.*
