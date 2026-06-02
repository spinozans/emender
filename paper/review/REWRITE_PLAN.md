# REWRITE_PLAN.md — the spine of the structural rewrite

**Subject:** Garrison, *Emending Nonlinear Recurrence* (`paper/main.typ`)
**Purpose:** This is the per-section serial rewrite plan that follows the Phase-1
Introduction rewrite (`rewrite-intro`). Sections are listed in **reading order, top
to bottom** (every `=` heading and its `==`/`#heading(level: 2)` subheadings). For
each unit: **(a)** what it must accomplish *for a reader who does not already live
inside this project*, and **(b)** its known voice/structure debt to repair.

**Seed crystal:** the **abstract** is the one unit both prior audits rated clean. It
already does reader-first, outside-in onboarding in the target voice. Grow every
section outward from its sensibility — do not invent a new voice.

**Three deep stances to purge everywhere** (from `EXECUTIVE_SUMMARY.md` / texture
synthesis), referenced below as the recurring debt:
- **R1 — Vantage collapse:** writing from inside the project; undefined codenames at
  first use ("E88"/"the Emender"/"the racer"/"stitched"/"comma-pile"/bare Lean ids/
  snake_case slugs).
- **R2 — Result-as-system:** handing one training run a product biography it does not
  have ("production / deployed / released production weights"). The single
  most-flagged defect; also literally false (the v0.3 release was broken-then-fixed).
- **R3 — Pre-litigation:** arguing with an absent skeptic mid-prose; honesty-protesting
  register ("honest null", "not a buried caveat"); nulls/caveats relitigated 3–5× each.
- **R4 — Plane/register slips:** process/QA leaks ("(verbatim)", "audit-recommended
  wording"), meta-slips about author credibility, cosmic/colloquial/emotional leaks.

**Four mandatory factual reconciliations** (must land regardless of voice work):
parity-not-superiority on held-out bpb; relabel §6 "matched no-tuning" + co-locate the
§9 selection caveat into §6; throughput = occupancy + MFU ≈15.7% (not peak-FLOPs); S₅ =
efficiency-not-impossibility; cite the corrected v0.3 revision (flag v0.1/v0.2).

---

## Reading order

### Abstract — *the seed crystal (DONE / anchor; do not disturb)*
- **Reader purpose:** onboard a non-insider in ~150 words: what recurrent sequence
  models are, why the field linearized, what is conceded, what this work shows, and the
  loss-vs-state-tracking split. Sets the arc the body must honor.
- **Debt:** none load-bearing — rated clean by both tracks. It is the voice exemplar.
  *Action: leave as-is; use as the register reference for every section below.*

### § Introduction `<sec:intro>` — *PHASE 1, rewritten in `rewrite-intro`*
- **Reader purpose:** build the door — space → tension/stakes → open question →
  meaning → contributions. First sentence situates the reader; codenames arrive late as
  an instance.
- **Debt (now repaired in this pass):** the old §1 opened in medias res ("The instance
  reported here, E88…") with no on-ramp. Rewritten to grow from the abstract.
  - `== Delta correction is one response; hybrids are another` — **purpose:** position
    delta-correction vs the hybrid bet as *complementary, not rival*. **Debt:** light
    R3 (the "complementary not rival" framing pre-empts a skeptic); keep, but do not let
    it tip into litigating the hybrid camp.
  - `== Contributions` — **purpose:** let the three results fall out of the motivated
    arc. **Debt:** verify each contribution states the *scoped* claim (viability =
    one artifact; separation = within formalized resource class; comparison = same
    band, parity). Already scoped post-revoice; keep aligned with the claims table.
  - `== What we claim, and what we do not` (claims table) — **purpose:** make scope
    legible up front; the reader's contract. **Debt:** strong asset; keep. Ensure the
    bpb row stays a *defended null/tie* and the S₅ rows stay *efficiency/computability*,
    never blanket can/can't.

### § Background `<sec:background>`
- **Reader purpose:** give the reader the linear-vs-nonlinear-state classification, the
  complexity ceiling (TC⁰ vs NC¹), and the S₅ probe — the conceptual toolkit the
  results sections assume.
- **Debt:** **R1 — the opening sentence jumps into "pangenomic sequence data … terabases
  per study"**, the author's private motivating workload, before the reader has any
  footing. Re-anchor on the general sequence-modeling stakes (as the new §1/abstract do)
  and demote the pangenomics motivation to a concrete *example*, or move it later.
  "the Emender is the construction introduced here" at line ~406 is an early codename
  drop — fine *if* §1 has already named it, else soften.
  - `== Linear-state and nonlinear-state recurrence` — **purpose:** the formal
    classifier + why linear-state dominates. **Debt:** clean and well-cited; minor — the
    Mamba-3 "does not fit our protocol cleanly" aside risks R3; state plainly.
  - `== Matrix state` — **purpose:** establish matrix state as a *shared precondition*,
    not a contribution. **Debt:** clean; keep the "not a contribution" honesty.
  - `== The S₅ state-tracking probe` — **purpose:** define the word problem + the S₃
    control for a reader who has never seen Barrington. **Debt:** clean; pedagogically
    load-bearing — keep.

### § Architecture `<sec:arch>`
- **Reader purpose:** show *how* the delta-correcting + tanh-latching update works,
  concretely enough to believe the expressivity story, and define the cohort instances.
- **Debt:** R1 codename density (per-head slugs, E88-specific shape numbers as if the
  reader knows them). Define each name at first use; keep equations but gloss them in
  the abstract's plain register.
  - `== Per-head update` — mechanism; **debt:** notation-heavy, ensure each symbol is
    introduced.
  - `== Parameterization choices` — **debt:** justify, don't relitigate.
  - `== The 1.3 B stack` — **debt:** R1 — shape numbers (370 heads, depth 12) need a
    sentence of "why these".
  - `== Ablation by architecture: isolating the write rule` — **purpose:** the controlled
    contrast; **debt:** ensure it reads as a clean isolation, not a verdict.

### § Multi-Programming and Systems `<sec:systems>`
- **Reader purpose:** explain the width-axis throughput mechanism that makes serial
  nonlinear recurrence practical — the systems crux of the viability claim.
- **Debt:** **R2 epicenter** ("production stack/architecture", "saturates the GPU").
  Mandatory reconciliation: state throughput as **occupancy (median 100% util) + MFU
  ≈15.7%**, width-axis *parity* with a real linear-scan kernel (≈91% of FLA-GDN), not
  peak-FLOPs or "full utilization".
  - `== Multi-programming: the throughput-enabling design choice` — purpose: the core
    idea; **debt:** R2 "production".
  - `== The 1.3 B Emender shape under multi-programming` — debt: R1 shape slugs.
  - `== Measured throughput and utilization` — debt: **the occupancy-vs-MFU
    reconciliation lands here.**
  - `== Fused Triton recurrence kernel` / `== Sparse-checkpoint backward` /
    `== Portable kernels` / `== Distributed training` — debt: R1/R4 implementation
    shorthand; keep claims matched to what was actually run (single-GPU; no cluster).

### § Language-Modeling Results `<sec:lm>`
- **Reader purpose:** the parity result — three pure-recurrent models in the same
  loss-vs-wallclock band; the held-out tie that says loss does not separate them.
- **Debt:** **mandatory bpb reconciliation** — present held-out bpb as a **statistical
  tie / parity**, and **retire the train-loss ordering as an architecture verdict**.
  R3 around the "defended null". R1 ("the racer", "stitched", "comma-pile control").
  - `== Setup` / `== Per-architecture CMA-ES protocol` — purpose: the matched protocol;
    debt: R1 slugs, R4 process leaks.
  - `== Gradient conditioning is a third recipe property` — debt: keep as recipe fact.
  - `== Loss-vs-wallclock comparison` — purpose: the headline parity figure; **debt:
    "same band", not "wins".**
  - `== Held-out bits-per-byte: the three are a statistical tie` — **the bpb
    reconciliation anchor; this framing is correct — propagate it backward to §1 and
    the abstract's claims.**
  - `== The loss tie reproduces under matched compute` — debt: R3, keep as robustness.

### § Expressivity Results `<sec:expressivity>`
- **Reader purpose:** the state-tracking separation the loss tie hands off to — delta >
  raw-write > linear on S₅, at 8 M and at 1.3 B, budget-robust.
- **Debt:** **mandatory S₅ reconciliation = efficiency-not-impossibility**; never
  "delta solves S₅" (both nonlinear updates plateau below ceiling at length).
  **BL-1: the §9 baseline-selection caveat must be co-located here / in §6** and the
  "matched no-tuning" label corrected. R3 throughout.
  - `== Capacity is non-binding at 8 M parameters for these probes` — purpose: the floor
    argument; keep — it is what makes the probe fair.
  - `== Matched no-tuning across architectures at 8 M` — **debt: RELABEL; this is the
    BL-1 disclosure asymmetry (tanh tie-broken on a state-tracking proxy). Foreground the
    S₃ control and the asymmetry here, do not bury it in §9.**
  - `== Headline: S₅ permutation composition` — debt: "headline" register; state
    efficiency-gap, not solve.
  - `== The six-task canonical sweep` / `== Hybrid degradation: purity matters` — debt:
    R3 ("purity matters" can tip adversarial); keep as measured effect.
  - `== The same separation at 1.3 B scale` — purpose: scale-up of the ordering;
    debt: budget-robust *ordering*, not impossibility.

### § Formal Results `<sec:formal>`
- **Reader purpose:** the machine-checked spine — what is *proved* (one-step and k-step
  separation, S₅ realization, FLOP-class equivalence, latching) vs what is empirical.
- **Debt:** **R1 — bare Lean identifiers** (`emender_m2rnn_one_step_…`) dropped without
  gloss; give each theorem a plain-language statement first, lemma name in parentheses.
  R5 (instrument integrity) — keep the "Claim" column free of disowned propositions.
  - Theorem sets A, B, C, C′, D, E, F + `== Frontier and unproven targets` +
    `== The NC¹ statement` + `== Explicit non-claims` — purpose: each set needs a
    one-line "what a reader should take from this". **Debt:** `== Explicit non-claims`
    is an R3 asset *if* it is the single place defense lives — make it carry the load so
    earlier sections stop relitigating.

### § Related Work `<sec:related>`
- **Reader purpose:** place the delta-correction line in its lineage (Widrow–Hoff →
  fast-weight programmers → DeltaNet) and against linear-recurrent + nonlinear-state
  peers — fairly.
- **Debt:** **R3 competitor-framing pre-litigation** — rivals "escorted out" via §1
  criteria with adversarial verbs ("violate"/"fail"), a GDN-2 timeline alibi,
  "concurrent prior art" importing a priority-dispute register. Describe peers
  neutrally; this section is otherwise R2-clean (a positive island).
  - `== Ancestry: the fast-weight line on delta correction` — purpose/asset: keep.
  - `== Linear-state recurrent language models` / `== Pure-nonlinear-recurrent peers and
    adjacent nonlinear-state work` — debt: neutralize adversarial verbs and timeline
    alibis.

### § Limitations `<sec:limitations>`
- **Reader purpose:** state honestly what the work does *not* establish — the reader's
  trust anchor.
- **Debt:** **R3 — honesty-protesting register** is densest here; and **the BL-1
  selection caveat currently lives here but belongs in §6** (move it). Single-seed,
  training-duration, geometry-sensitivity, hybrid bet — keep, stated plainly once.
  - Subsections (`== Formal scope`, `== Evidence structure`, `== Length extrapolation…`,
    `== Per-architecture CMA-ES is best-effort matched…`, `== Geometry-sensitivity…`,
    `== Design-space asymmetry of the 8 M defaults`, `== The opposite architectural bet:
    hybrids`, `== Training duration and result scope`, `== Open architectural choices`,
    `== Transformer comparison`) — purpose: one honest limitation each; **debt:** say it
    once, drop the protest, and ensure none silently contradicts the reconciled claims.

### § Conclusion `<sec:conclusion>`
- **Reader purpose:** restate the take-home in the abstract's voice: pure nonlinear
  recurrence is practical at scale, loss is a poor guide to compute, the design space is
  open.
- **Debt:** R2 ("production") risk; keep it scoped to "one artifact shown to exist" +
  the open-design-space framing the abstract already lands.

### § Testable Predictions `<sec:predictions>`
- **Reader purpose:** give the reader falsifiable, dated predictions — the §11 posture
  the texture audit rated as a *positive anchor*.
- **Debt:** clean (claim+scope+falsifier). Keep as a second voice exemplar; ensure
  predictions are consistent with the reconciled (parity/efficiency) claims.
  - Subsections (`== S₅-generator-specific capacity bound`, `== A partial order on PNR
    update rules`, `== Additional seeds and architecture-internal revalidation`,
    `== Scale beyond 1.3 B`, `== Cleanest within-class HPO follow-up`) — purpose:
    one crisp prediction each; debt: minimal.

### § Future Work `<sec:future_work>`
- **Reader purpose:** signal the open directions (stronger raw-write baseline, scale,
  length extrapolation, HPO breadth) without over-promising.
- **Debt:** light R3; keep modest. The "stronger raw-write baseline" item should note it
  would close the one entangled (optimization-vs-expressivity) contrast.

### Appendix: Lineage of the E63 → E88 experimental program `<sec:appendix>`
- **Reader purpose:** provenance for how the studied instance was arrived at.
- **Debt:** **R1 saturated** — bare experiment slugs (E63, E88) as a private lab
  notebook. Frame as a reproducibility/provenance record for an outsider; gloss the
  lineage rather than assuming it.

### Appendix: Held-out bits-per-byte measurement `<sec:appendix_bpb>`
- **Reader purpose:** the exact harness/protocol behind the held-out tie — the
  evidentiary backbone of the parity claim.
- **Debt:** R4 process-leak risk; keep as a clean methods appendix. **Cite the corrected
  v0.3 revision** and pin checkpoints; this is where the broken-then-fixed release is
  made reproducible.

---

## Sequencing note for the serial chain
1. Land the **four mandatory factual reconciliations** first (bpb-parity, §6 relabel +
   §9-caveat move, throughput occupancy/MFU, S₅ efficiency, corrected citation) — a
   reviewer rejects on these and they gate the voice work.
2. Then the **global stance passes** (R2 purge → R1 vantage lift → R3 confinement → R4
   plane-slips), each anchored on the abstract and §11, done by one hand to avoid
   "smoothing good prose into new uncanniness".
3. Re-audit the two under-measured axes (compounding intensity, prosody/cadence) before
   declaring the rewrite complete.
</content>
</invoke>
