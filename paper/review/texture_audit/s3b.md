# Texture & Flow Audit — Unit s3b

**Unit:** §3 Architecture, subsection 3b — "Ablation by architecture: isolating the write rule" (`paper/main.typ` lines 708–776, including @tab_ablation).
**Mode:** AUDIT-ONLY. Defect register, no fixes, no edits to source.

## Global-voice note (stance calibration, whole-paper read)

The paper speaks as a scope-obsessed empiricist addressing a skeptical ML-systems/theory
audience, and its dominant move is pre-emptive boundary-drawing: an explicit
"What we claim, and what we do not" table up front, repeated *null* / *scope* /
*caveat* / *honest mirror* sidebars, and inline Lean theorem names as proof-of-trust.
The voice argues with an absent reviewer almost continuously — it states a result and
then, in the same breath, fences off what the result does *not* license. Named entities
(E88, "the Emender", M²RNN-CMA, GDN, "multi-programming") are treated as established
currency; the intro even opens *inside* the project's world. The risk this stance creates
is over-defense and relitigation: the same boundary is often asserted in a sidebar, then
re-asserted in a caption, then again in body prose. My unit sits squarely in that risk
zone — it is an elimination argument, the most relitigation-prone shape in the paper.

## Register

| section | location (para/sentence) | span (verbatim, ≤25 w) | category | severity | frame-rule | what-it-presupposes-or-imports | why-it's-a-defect-not-deliberate | conf |
|---|---|---|---|---|---|---|---|---|
| §3b | S3 para (lines 758–760), sentences 1–2 | "the solvable control where non-solvability is not the obstruction … even on a solvable group, where non-solvability is not the obstruction" | FLOW-BREAK, PRE-LITIGATION | MESSY | R3 | The reader needs the qualifier once; repeating it imports the stance of a writer defending the same point twice in one breath. | The identical 6-word clause appears in two consecutive sentences with no rhetorical contrast or escalation between them; verbatim adjacent repetition is a drafting artifact, not emphasis a careful author would choose. | 0.8 |
| §3b | closing para (lines 773–776) | "State capacity is not the differentiator. … GDN has matrix state and still fails S5. By elimination, the differentiator is the write rule." | PRE-LITIGATION | MESSY | R3 | Presupposes the elimination has not already landed — but the Verdict column ("Cannot separate: GDN has it and fails S5") and the caption already state it. | The same conclusion is delivered three times within the subsection (table Verdict cells → caption → this paragraph) with no new inferential step in this paragraph beyond a citation list; the relitigation is redundancy, not a deliberate summary (it is not framed as one). | 0.6 |
| §3b | line 762–768 (capacity sub-argument) | "Nor is the shortfall a capacity ceiling. At the 8 M probe shape … the per-token recurrent state carries 131,072 scalars, about five orders of magnitude above the … floor" | PRE-LITIGATION, FLOW-BREAK | MESSY | R3 | Imports a §6-grade capacity-floor calculation into an architecture-exposition section, presupposing the reader is already adjudicating the capacity objection here. | A bit-counting floor argument is §6's job (it is restated near-verbatim at lines 1130–1137); dropping it mid-§3 interrupts the write-rule exposition to fight an objection the paper has a dedicated home for — defense woven in rather than placed in the scope structure R3 prescribes. | 0.5 |
| §3b | line 729 + heading line 708 | "A three-row ablation across the three candidate properties isolates the write rule by elimination" | UNPAID-WORD, UNCANNY | UNCANNY | R2/R3 | "ablation … isolates … by elimination" imports the connotation of controlled component-removal and deductive closure onto a comparison of three separately-trained, differently-shaped models. | Parses cleanly as a standard ablation-table sentence, yet the three rows are whole distinct architectures (different dim/depth/H, different runs), not one model with a component toggled; "isolates by elimination" lends an empirical accuracy-gap argument the finality of a logical proof — the yes/no truth-table format (@tab_ablation) reinforces the unearned deductive register. Flagged as connotation/stance, not as a truth claim. | 0.45 |
| §3b | @tab_ablation header vs Verdict cells / caption (lines 738, 743, 749–754) | header "M²RNN" vs body "M²RNN-CMA stalls at 0.22 on S5" | INSIDE-OUT, FLOW-BREAK | MESSY | R1 | Presupposes the reader is holding the M²RNN (published) vs M²RNN-CMA (this paper's CMA-reshape) distinction and silently maps the column header onto the variant the number belongs to. | One figure uses both names for the same row without reconciling them; a reader outside the project cannot tell whether the 0.22 / ablation row is the literature M²RNN or this paper's reshaped variant — the column says one thing, the verdict and caption say another. | 0.45 |
| §3b | line 723, sentence 1 | "That split is exactly what E88 lacks: its carry sits inside the bounded delta-correcting update" | RESULT-AS-SYSTEM | MESSY | R2 | Attributes an architecture-family property (the update/gate form) to E88, the single 1.273 B run, as if the run were the bearer of the design. | The contrast being drawn is M²RNN's update *form* vs the Emender's update *form* — a family-level property; pinning "lacks the split" to E88 (one training run) conflates the result-instance with the architecture, the unit's local instance of the paper's run-as-system slippage. Low-moderate because E88 does instantiate that update, so the slip is mild. | 0.4 |

## One-line unit rating

Predominant category **PRE-LITIGATION / redundancy** (the elimination conclusion is stated 3×: Verdict column → caption → closing paragraph); severity skew **MESSY** with one **UNCANNY** (the truth-table "isolates by elimination" framing dressing a confounded 3-model comparison as deduction); defects are **clustered** at the back of the subsection (table + caption + S3 paragraph + closing paragraph all relitigate the same point), not uniform.

## Cross-section dependencies for the synthesizer

- **Forward-referenced numbers not yet earned here:** the verdict/caption cells use 0.36 / 0.79 / 0.22 (and S3 0.31), all of which are §6 results (@tab_s5, lines 1175–1213). §3b states the elimination conclusion on the strength of data the reader meets only in §6 — a mild CONCLUSION-FIRST that the synthesizer should weigh against §6's own framing.
- **Duplicated capacity-floor argument:** the 131,072-scalars-vs-2.6-bit-floor computation at lines 762–768 is restated near-verbatim in §6 (lines 1130–1137). Two homes for one argument; flag for dedup ownership (which section should carry it).
- **Name-variant continuity:** the M²RNN vs M²RNN-CMA slippage in @tab_ablation depends on the distinction established in §1 (lines 240–251, 304–306) and §5 (table line 924). The synthesizer should check whether this header/body inconsistency recurs in other units' tables.
- **Formal forward-ref:** line 770 cites `emender_m2rnn_one_step_resource_separation_embeds` (§7) as the "one-step formal counterpart"; consistent with §7 set C, no defect, noted only for arc-tracking.
