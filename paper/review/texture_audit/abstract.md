# Texture & Flow Audit — Unit: ABSTRACT (id=abstract)

Source: `paper/main.typ`, `abstract:` field passed to `arkheion.with(...)` (lines 71–100).
Auditor node: blind fan-out, audit-only. No fixes proposed.

## Global voice (whole-paper stance, written first to calibrate)

The paper speaks as a careful empiricist reporting one demonstration, not as the
proprietor of a system. Its dominant, well-controlled move is scope discipline:
an explicit Claim/Evidence/Scope/Non-claim table, repeated "supporting
comparison", "parity-class, not superiority", "honest null", "single seed". The
intended reader is an outsider to recurrent-LM systems work who must be handed
context, and the body mostly does manufacture it. BUT the voice is not uniform:
the **Introduction** drops this discipline and speaks from *inside the project's
world* — it opens with the bare codename "E88", calls it a "production instance",
narrates "23 stitched wall-clock days", and lets the architecture name itself.
The **abstract is the discipline kept**: it is the cleanest unit in the paper,
written strictly outside-in (no E88, no "production", no insider codenames; "a
machine-checked proof" rather than "the Lean 4 trusted core"). So for this unit
the audit's job is the inverse of the intro's: the violations are few, residual,
and small — mostly late-paragraph jargon/connotation imports and a slight
register that runs prouder than the body's own hedged framing.

## Register

schema: section · location · span · category · severity · frame-rule · presupposes/imports · why-defect-not-deliberate · confidence

| # | section | location | span (verbatim ≤25w) | category | severity | frame | presupposes / imports | why it's a defect, not deliberate | conf |
|---|---------|----------|----------------------|----------|----------|-------|------------------------|------------------------------------|------|
| 1 | Abstract | para 1, sent. 7 (the "machine-checked proof" sentence) | "A machine-checked proof establishes the enabling direction:" | INSIDE-OUT / UNNAMED | MESSY | R1 | Presupposes a two-direction framing (a positive "enabling"/realizability result vs. a negative "impossibility" result) that the cold reader has not been given a name for yet. "the enabling direction" is internal shorthand. | The cold reader meets "the enabling direction" with no antecedent; it is only repaired by the colon-expansion and by the *next* sentence's "matching impossibility". A self-referential label for a dichotomy the abstract never set up reads as project-internal vocabulary, not reader-facing prose. Not a mere preference: the phrase names a structure (two directions) before the structure exists for the reader. | 0.55 |
| 2 | Abstract | para 1, final sentence | "massively parallel nonlinear recurrence is a practical and largely open design space" | UNPAID-WORD | MESSY | R2/R1 | Imports the HPC/SIMD connotation of "massively parallel" (massively-parallel processor arrays, thousands of lanes). The mechanism delivered earlier in the same abstract is "hundreds of small recurrent programs … across the network's width". | The body never uses "massively parallel"; its term of art is "multi-programming" / "width-axis". The closing sentence swaps in a grander, hardware-paradigm adjective at the editorializing moment. The unpaid connotation (a hardware regime) is doing rhetorical lift the demonstrated mechanism ("hundreds … across width") did not pay for. | 0.40 |
| 3 | Abstract | para 1, sent. 5 ("Here we show…") | "matching strong linear-recurrent baselines" | UNNAMED (register/stance gap) | MESSY | R3 | Imports "strong" as an unscoped quality on the baselines and headlines the comparison as a co-result. | The body deliberately *demotes* this same comparison: it is labeled "Supporting comparison", scoped to "one E88 run, single seed; viability demonstration", and §4 states "the honest claim is parity-class, not superiority." GDN is never called "strong" in the body. The abstract's register here runs prouder and less scoped than the paper's own governing scope-discipline voice — a stance inconsistency with the body, not a deliberate compression (the parity *fact* could be stated without the unscoped "strong"). | 0.45 |
| 4 | Abstract | para 1, sent. 4 (the "It has been unclear…" gap sentence) | "whether a purely nonlinear recurrence can be trained to competitive scale on its own" | INSIDE-OUT | MESSY | R1 | "on its own" presupposes the pure-vs-hybrid framing — that prior scaling of nonlinear recurrence happened only *with help* (attention hybridization) — which the abstract has not introduced. | At this point the abstract has told the reader only that nonlinear recurrence "must run serially, which has been assumed too slow". It has not mentioned hybridization, so "on its own" gestures at an absent competitor framing (the hybrid concession) that lives only in the Introduction. The cold reader cannot resolve what "on its own" is contrasted against; the contrast is project-internal context not yet manufactured. | 0.40 |

## One-line unit rating

Predominant category: residual INSIDE-OUT / unpaid-connotation imports; severity skew strongly toward MESSY with **no UNCANNY findings** (this unit is the paper's cleanest); clustered, not uniform — all four sit in the **second half of the single paragraph**, in the formal-results and closing-editorial sentences, while the opening context/gap/result arc is essentially defect-free.

## Cross-section flags for the synthesizer

- **Inverse-of-intro signal (highest-value cross-section note).** The abstract and the Introduction adopt *opposite stances toward the reader*. The abstract is correctly outside-in (no codename, no "production", "a machine-checked proof"); the Introduction §1 opens inside the project's world ("E88, the 1.273 B-class production instance…", "23 stitched wall-clock days", architecture self-naming). The discontinuity is itself a finding for synthesis: the paper *can* hold the disciplined voice (it does so for ~150 words in the abstract) and then drops it at the first body sentence. Whoever audits §1 should be told the disciplined target voice already exists in this unit.
- **Forward-dependent vocabulary.** "the enabling direction" (row 1) and "the same ordering" (the delta > raw-write > linear three-way order) are only fully resolvable from §6/§7 and the Claim table; the abstract states the ordering's pieces but never names it as an ordering before invoking "the same ordering". Synthesizer should check whether the abstract assumes a structure defined sections later.
- **Register vs. scope-table (row 3).** The abstract's "matching strong … baselines" should be reconciled against `<tab_claims>`'s "Supporting comparison / single seed / parity-class, not superiority". This is a paper-wide register-consistency question, not solely an abstract defect.
- **Out of scope by contract:** I did not assess whether any claim is *true* (claim-vs-evidence is the separate track); rows above concern delivery/stance only.
