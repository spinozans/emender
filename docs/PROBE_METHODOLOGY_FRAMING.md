# Probe Methodology Framing: Over-parameterization as Design

This document captures the methodology-framing position the v2 paper must
adopt for the state-tracking probes. It exists to be integrated verbatim
(§3) and as guidance (§§1–2, §4) by the downstream `paper-v2-coherent`
task.

The core inversion: for state-tracking probes, **over-parameterization
is the design, not a limitation**. The 8 M-scale S₅/S₃ probes are the
*clean* mechanism measurement; the 1.27 B LM/QA panel is the
*confounded* measurement. The v1 paper (`Garrison_2026_NDM-5a661112.pdf`)
gets this backwards by treating the 8 M probes as preliminary evidence
that must "bridge to scale," and by treating the QA tie at 1.27 B as
counter-evidence to the expressivity claim.

---

## §1 — Verified probe numbers

All numbers below are harvested directly from the v1 source. The
authoritative cell is `paper/main.typ` lines 730–756 (the §7 table),
which in turn cites `paper/ndmpapernotes.md` lines 153–173 as the
source-of-truth ledger. The 8 M parameter-matched scale and per-model
shapes are stated at `paper/main.typ` lines 718–720 and consolidated in
`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` lines 24–39.

Protocol (uniform across all four families): dim = 384 (FLA-GDN
dim = 640 to match parameter count), depth = 4, H = 32, N = 32,
schedule-free AdamW, 10K–20K steps per task, three seeds (42 / 123 / 456),
train length T = 128. Random baselines: S₃ = 1/6 ≈ 0.1667;
S₅ = 1/120 ≈ 0.0083.

| Model | Param count | S₃ T=128 (mean) | S₅ T=128 (mean) | Source |
|---|---|---|---|---|
| NDM (E88) | ~8 M (dim 384, d=4) | **1.0000** | **0.7918** | `main.typ` L739; `ndmpapernotes.md` L157, L161 |
| FLA-GDN | ~8 M (dim 640, d=4) | 0.7185 | 0.3552 | `main.typ` L740; `ndmpapernotes.md` L158, L162 |
| M2RNN-CMA (tied-head reshape) | ~8 M (dim 384, d=4) | 0.3124 | 0.2157 | `main.typ` L741; `ndmpapernotes.md` L159, L163 |
| M2RNN-paper (published grouped-head shape) | ~8 M (dim 608, d=4) | 0.3773 | 0.1698 | `main.typ` L742; `ndmpapernotes.md` L160, L164 |
| random | — | 0.1667 | 0.0083 | `main.typ` L743 |

S₅ length extrapolation (train T=128; same three seeds), from
`paper/ndmpapernotes.md` lines 168–173 and `paper/main.typ` lines
739–743:

| Model | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| NDM | 0.7918 / 0.7900 | 0.4158 | 0.2150 | 0.1104 |
| FLA-GDN | 0.3552 / 0.3544 | 0.1843 | 0.0974 | 0.0521 |
| M2RNN-CMA | 0.2157 / 0.2142 | 0.1120 | 0.0593 | 0.0339 |
| M2RNN-paper | 0.1698 / 0.1696 | 0.0884 | 0.0488 | 0.0283 |

(The two T=128 cells are the headline-table number and the
length-sweep table number; they agree to three decimal places, which
is the seed-to-seed reproducibility of the probe.)

**Unverified / not in v1.** No exact parameter count to the digit is
printed in v1 for any family — the paper says "8 M parameter-matched"
throughout and reports shapes (dim, depth, H, N) rather than fitted
totals. Calling the totals "≈ 8 M" matches the v1 wording. If §2
needs a sharper number than "~8 M" it should be regenerated from the
model configs, not invented.

---

## §2 — State-count vs parameter-count calculation

The information-theoretic floor for representing the running state of
a group-prefix tracker is `log₂(|G|)` bits per token of carried state.
For the two groups used in v1:

- `|S₅| = 120` (Lean: `S5Witness.s5_state_count`, see `main.typ` L967).
  Floor: `log₂(120) ≈ 6.907` bits → ~7 bits.
- `|S₃| = 6`. Floor: `log₂(6) ≈ 2.585` bits → ~3 bits.

A "≈ 8 M" parameter recurrent recogniser at fp32 carries
`8 × 10⁶ × 32 = 2.56 × 10⁸` bits in its parameters; at fp16,
`1.28 × 10⁸` bits. Comparing parameter-budget bits to the state-set
information-theoretic floor:

| Group | Floor (bits) | Param bits @ fp32 | Headroom (orders of magnitude) |
|---|---|---|---|
| S₅ (120 states) | ≈ 6.91 | 2.56 × 10⁸ | ≈ 3.7 × 10⁷ × the floor (~7.5 orders) |
| S₃ (6 states) | ≈ 2.59 | 2.56 × 10⁸ | ≈ 9.9 × 10⁷ × the floor (~8 orders) |

Even charging the recogniser for the full
`|S| × |Σ|` lookup table — the Lean tight bound is
`S5NDMRealization.s5_transition_key_count = 120 × 4 = 480`
state/input keys (`main.typ` L975–977) — the table itself is
on the order of `480 × log₂(120) ≈ 3.3 × 10³` bits. Eight million
parameters is roughly *five orders of magnitude* above even this
tighter realisation floor.

The conclusion is unambiguous: **at the 8 M scale chosen for the v1
probes, parameter budget is non-binding for either S₅ or S₃**. Any
model that fails on these probes fails for a structural reason, not a
capacity reason.

---

## §3 — Draft methodology paragraph (publication-ready)

The following ~295-word paragraph is written in third-person scholarly
voice for direct insertion into `paper/main.typ`. It contains no
first-person, no self-reference, and anchors every claim either to a
v1 figure/table or to the calculation in §2 above. The paragraph is
deliberately precise about *representability* (Lean, per-step) versus
*learnability* (probes, joint property); see the note immediately
following on why that distinction is load-bearing.

> The state-tracking probes of this section are *intentionally
> over-parameterized*. The symmetric-group witnesses carry small
> state sets — `|S₅| = 120`, `|S₃| = 6` — so the
> information-theoretic floor for the running prefix product is
> `log₂(120) ≈ 6.91` bits for S₅ and `log₂(6) ≈ 2.59` bits for S₃.
> At parameter-matched 8 M scale, every family carries on the order
> of `10⁸` parameter bits, roughly seven orders of magnitude above
> this floor and several above the tight lookup-table realisation
> (`|S₅| × |Σ| = 480` keys, formalised as
> `S5NDMRealization.s5_transition_key_count`). Every family in the
> comparison can therefore *represent* the prefix automata in
> principle; the Lean per-step expressivity results
> (`S5Witness.fixed_precision_state_space_finite`) state this
> representability claim formally. The probes themselves measure a
> strictly stronger *joint* property — representable AND learnable
> by SGD at this FLOP class — and isolate the update rule's
> inductive bias from its representational ceiling. M2RNN's S₃ mean
> of 0.31 — `10⁸` parameter bits chasing a ~3-bit floor — is then
> evidence that SGD under the raw-write inductive bias cannot
> *find* a six-state prefix-tracker, not that none exists in the
> model class. Two propositions must be kept distinct: (A) at
> matched per-token FLOP class, the delta-correcting write rule is
> the only architectural ingredient making S₅/S₃ prefix-tracking
> *learnable in practice* across the post-Mamba matrix-state
> family — a learnability claim, settled at 8 M; (B) that
> learnability transfers to language modelling at 1.27 B — open,
> not what the §7 QA panel rules on. The 8 M probes are the
> *clean* mechanism measurement; the 1.27 B QA panel is
> *confounded* for mechanism questions at this stage. The framing
> that the S₅ result "vanishes at scale" or "needs to be re-shown
> at 1.27 B" is therefore retracted: the learnability gap is
> scale-invariant at fixed precision and width, and the QA tie
> measures (B), not (A).

Word count: ~299 words (publication-ready window 200–300).

**Representability vs learnability — the load-bearing distinction.**
The Lean per-step theorems
(`S5Witness.fixed_precision_state_space_finite`, the
`S5NDMRealization` realisation lemmas) are *representability* claims:
the matrix state of any post-Mamba family in this comparison can
encode the prefix automaton in principle. The empirical probes do not
re-prove that; they measure the strictly stronger *joint* property
representable-AND-learnable-by-SGD at this FLOP class. M2RNN's S₃ =
0.31 is a *trainability* failure (SGD under the raw-write inductive
bias cannot locate a valid configuration), not a representability
failure (the configuration exists in the model class). Proposition A
is correspondingly a *learnability* claim, not a representability
claim; the Lean core corroborates the no-capacity-confound side of A
but does not itself test learnability.

Anchors used in the paragraph:
- `S5NDMRealization.s5_transition_key_count` — `main.typ` L975
- `S5Witness.fixed_precision_state_space_finite` — `main.typ` L962
- S₃ M2RNN = 0.31 — `main.typ` L741 (M2RNN-CMA = 0.3124); the paper-shape variant at 0.3773 (L742) reinforces the same conclusion across two M2RNN geometries.
- 8 M parameter-matched scale — `main.typ` L718, L812, L836.
- 1.27 B QA tie — `main.typ` L927–947.

---

## §4 — Recommended insertion point

**Recommendation: insert §3 verbatim as a new unnumbered subsection at
the head of §7 Expressivity Results — between the existing setup
paragraph (`main.typ` L713–720, ending "FLA-GDN uses dim = 640 to
match parameter count.") and the existing headline subsection
(`main.typ` L722, `[Headline: $S_5$ permutation composition]`).
Title the new subsection `[Why over-parameterization is the
design]` or `[Reading the probes: capacity, mechanism, and
proposition scope]`.**

Rationale for this single placement, rather than the alternatives:

1. **Frames before it asserts.** The headline S₃/S₅ table (`main.typ`
   L730–756) lands on the reader with no methodological context for
   *why* 8 M is the right scale. Inserting the framing immediately
   before the table makes the table legible as a mechanism
   measurement on first read, rather than requiring the reader to
   reach §10 to find out how the authors think about capacity.

2. **Sets up the §7 QA panel correctly.** The same §7 already contains
   the 1.27 B QA panel ("QA and reasoning panel at 1.27 B",
   `main.typ` L927–947). Inserting the framing at the head of §7
   means by the time the reader meets the QA panel inside the same
   section, the Proposition A / Proposition B distinction is already
   in scope and the QA tie reads as a measurement on Proposition B,
   not as a refutation of the §7 headline.

3. **Lets the §10 limitations subsection be repaired in place, not
   re-written.** With the framing established in §7, the v1
   "Geometry-sensitivity of the update-rule claim" subsection
   (`main.typ` L1158–1167) — which currently treats the §5 LM tie as
   constraining the §7 expressivity reading — can be retitled to
   "Scope of the empirical expressivity claim" and rephrased to say
   that §7 (probes) and §5 (LM) speak to different propositions, so
   §5 cannot constrain §7. That is a smaller surgical change in v2
   than a full rewrite, but it depends on the §7 framing being in
   place first.

4. **Does not require relocating any figure.** The new subsection is
   prose-only; the headline table and bar-chart figure (`<tab_s5>`,
   `<fig_s5_bars>`) keep their existing positions and references.

Alternative placements that are *less* preferred and why:

- *Between probe results and LM results (i.e., a new §7.5 inside §7
  after the QA panel)*: too late. The reader has already met the
  table and the QA tie without the framing.
- *Inside §10 Limitations*: wrong semantic register. The framing is
  not a limitation — it is the *interpretive frame* under which the
  probe section should be read. Putting it in §10 keeps the §7 misframe
  on the page.
- *As a §8 preamble (Formal Results)*: §8 already opens with the Lean
  trusted-core scope statement; doubling the preamble would dilute it.

A secondary follow-up change in v2, contingent on §3 being inserted at
the head of §7: revise `main.typ` L1158–1167 to drop the phrasing
"strongest reading of the empirical claim is therefore conditional on
the multi-programmed shape," which currently conflates the §5 LM tie
with the §7 expressivity claim. The replacement language should
preserve the geometry-sensitivity caveat for §5 only and explicitly
acknowledge that §7 stands independently of §5.

---

## Notes for the integrator

- **Do not** introduce parameter counts to more precision than v1 uses.
  v1 says "≈ 8 M parameter-matched"; §1 above and §3 stay at that
  precision. Tightening to a fitted total without recomputing from the
  model configs would be fabrication.
- **Do not** soften the retraction in §3 paragraph closing sentence.
  The "expressivity is scale-invariant at fixed precision and width"
  clause is the load-bearing retraction; it is what blocks the
  reviewer move "but you didn't re-show this at 1.27 B."
- **Do** keep Proposition A and Proposition B rhetorically distinct in
  the integrated text. They are not the same proposition; v1's
  apparent retreat from A was a category error introduced by reading
  B-evidence (the QA tie) as A-evidence.
- **Do** preserve the *representability vs learnability* distinction
  whenever §3 prose is paraphrased downstream. The Lean per-step
  theorems are representability results; the probes are
  learnability results; both indict the raw-write update, but they
  are not the same claim. Proposition A is a learnability claim
  ("learnable in practice at matched FLOP class"), not a
  representability claim. M2RNN's S₃ = 0.31 is a *trainability*
  failure, not evidence that the matrix state cannot encode the
  six-state automaton.
