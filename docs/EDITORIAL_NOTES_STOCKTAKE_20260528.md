# Editorial Notes Stocktake

Date: 2026-05-28 UTC
Task: `editorial-notes-stocktake`
Scope: stocktake only. Do not edit `paper/main.typ` from this task.

## Executive Map

`editorial_notes.md` contains five drafted insertions, one separate Section 5
Pareto/racer reframe, and a set of smaller checks. The focused paper edit should
stay to 6-7 touch points:

1. Abstract claim framing: `paper/main.typ:70-99`.
2. Introduction opening/scope note: `paper/main.typ:151-165`.
3. Background S5/TC0 motivation: `paper/main.typ:384-407` and `440-451`.
4. Systems update-rule freedom: `paper/main.typ:736-753`, with prediction mirror
   at `paper/main.typ:1785-1792`.
5. Section 5 racer framing around Figure 2: `paper/main.typ:890-943`.
6. Section 6 S5 honesty note: `paper/main.typ:987-1058`, with existing
   limitation cross-check at `paper/main.typ:1591-1600`.
7. Related-work concurrent linear-state note: `paper/main.typ:1467-1513`.

Do not turn this into a general rewrite. Preserve the current Emender/PNR
taxonomy, the 1.3B Hugging Face links and release claims, current Figure 2
values, and the proof-boundary language around what Lean proves.

## Source Facts And Guardrails

- `editorial_notes.md` is at `/home/erikg/ndm/editorial_notes.md`, outside this
  worktree checkout. It was read end to end.
- Recent release/link state: `hf-rename-127b-to-13b-link-sync` moved the three
  HF slugs to `poietic-pbc/emender-e88-1.3b`,
  `poietic-pbc/gdn-1.3b`, and `poietic-pbc/m2rnn-cma-1.3b`, with unchanged
  `v0.1` SHAs. Paper links already use 1.3B slugs at `paper/main.typ:337-340`
  and `1735-1744`. Do not change those in the focused edit except for ordinary
  surrounding prose.
- Current Figure 2 values are fixed for this editorial pass: E88 0.979 bpb,
  GDN 0.975 bpb, M2RNN-CMA 0.984 bpb in `paper/main.typ:919-943`, backed by
  `paper/results/figure_2/AS_OF.md:16-29`. The figure-polish task explicitly
  preserved those labels and changed only composition.
- Release readiness is now superseded by later public-release work: the old
  readiness report records pre-public blockers, while the HF rename task logs
  state that GitHub and the 1.3B HF repos are live. Do not update visibility,
  repo settings, tags, model cards, checkpoints, or release artifacts here.
- Primary-source spot checks done for citation-sensitive current work:
  Gated DeltaNet-2 is arXiv:2605.22791, "Gated DeltaNet-2: Decoupling Erase and
  Write in Linear Attention", Ali Hatamizadeh, Yejin Choi, Jan Kautz, submitted
  2026-05-21. The abstract states channel-wise erase/write gates, chunkwise WY,
  and strongest 1.3B results among Mamba-2, Gated DeltaNet, KDA, and Mamba-3
  variants. Mamba-3 is arXiv:2603.15569, "Mamba-3: Improved Sequence Modeling
  using State Space Principles", Aakash Lahoti et al., submitted 2026-03-16 and
  marked ICLR 2026. Current `paper/refs.bib:417-426` already has Mamba-3; it
  lacks a GDN-2 entry.
- Primary-source spot checks for the transformer/TC0 expansion: Merrill and
  Sabharwal arXiv:2210.02671 covers log-precision transformers; Chiang
  arXiv:2409.13629 covers transformers in uniform TC0 with precision caveats;
  Merrill and Sabharwal arXiv:2503.03961 covers why log-depth changes the
  picture. Current `paper/refs.bib` does not contain these entries except the
  related Merrill-Petty-Sabharwal SSM paper at `paper/refs.bib:253-260`.

## Actionable Edit Items

### 1. Frontier Scoping

Classification: must-do.

Editorial source: `editorial_notes.md:60`, with the stronger critique at
`editorial_notes.md:23-36`.

Paper locations:

- Abstract: `paper/main.typ:70-91`, especially "contemporary frontier language
  models" and "current frontier linear-recurrent learner".
- Introduction opening: `paper/main.typ:151-165`.
- Section 5 racer conclusion: `paper/main.typ:932-943`, where
  "frontier-class linear-recurrent baseline" appears.

Direction:

- Bound "frontier" to the recurrent/sub-quadratic, publicly reproducible,
  same-scale comparison class. Do not let readers infer frontier-scale
  transformers trained on trillions of tokens.
- Preserve the class-level viability claim: pure nonlinear recurrence reaches
  the same loss-vs-wallclock band as the selected linear-recurrent baseline at
  this 1.3B-class budget.
- Avoid "current frontier" as a timeless claim. Prefer "selected
  linear-recurrent baseline" or "publicly reproducible sub-quadratic recurrent
  baseline at this scale/training window".

Small phrase:

`frontier here means the strongest publicly reproducible sub-quadratic recurrent
baseline at this parameter and training-budget class, not a frontier-scale
transformer`

Risk:

- Low if kept as scoping. Medium if it reads like an apology that weakens the
  result. The goal is precision, not retreat.

Validation:

- `rg -n "frontier|frontier-scale|current frontier" paper/main.typ` after the
  focused edit should show no unbounded "frontier" use.
- Confirm the abstract still says the paper is about trainable pure nonlinear
  recurrence, not only a small-baseline comparison.

### 2. Concurrent Linear-State Work: GDN-2 And Mamba-3

Classification: must-do, citation-sensitive.

Editorial source: `editorial_notes.md:61`, `83-88`, and drafted paragraph at
`97-99`.

Paper locations:

- Related Work, after "Linear-state recurrent language models":
  `paper/main.typ:1467-1482`.
- Existing Mamba-3 note: `paper/main.typ:1509-1513`.
- Background sentence naming Mamba-3's comparison to GDN:
  `paper/main.typ:369-372`.
- `paper/refs.bib`: add GDN-2 only if the focused edit uses it.

Direction:

- Add a cautious paragraph or short subsection, not a new experiment.
- State that GDN-2 and Mamba-3 push the linear-state/sub-quadratic paradigm
  further, but neither belongs in the current racer without new matched runs.
- Use GDN-2 as a foil and a next-ablation prompt: decoupled channel-wise
  erase/write gates are exactly the kind of per-step-body variant the
  multi-programmed Emender substrate should make easy to test.
- Keep timing precise. GDN-2 was submitted 2026-05-21, after the May 7 racer
  starts and after the target baseline was selected. Mamba-3 was submitted
  2026-03-16; only say code/weights were unavailable during the training window
  if the next worker verifies that from primary project/source release state.

Small phrase:

`GDN-2 and Mamba-3 sharpen the comparison rather than replace it: they buy new
retrieval/state-tracking behavior inside the linear-state envelope, while this
paper tests the expressivity axis opened by serial nonlinear state updates.`

Risk:

- Medium. Do not rely on blog summaries. Do not claim GDN-2 "would fail S5" as
  an empirical fact without running it. It is safe to say it remains a
  linear-state/chunkwise-parallel model if the paper's update equations support
  that reading.
- The current draft in `editorial_notes.md` says "GDN-2 postdates our runs".
  Because current Figure 2 runs were still in progress as of 2026-05-27, the
  safer claim is "postdates the experimental design/selection and would require
  a new matched run".

Validation:

- Add a `gated_deltanet2_2026`-style bib entry before citing it.
- Verify the exact Mamba-3 citation in `paper/refs.bib:417-426`.
- Run `rg -n "GDN-2|Gated DeltaNet-2|Mamba-3" paper/main.typ paper/refs.bib`
  and check every claim is source-backed and time-stamped.

### 3. Update-Rule Freedom

Classification: must-do.

Editorial source: `editorial_notes.md:47-49`, `70-72`, and drafted paragraph at
`101-107`.

Paper locations:

- Abstract mirror near `paper/main.typ:87-99`.
- Systems "Multi-programming" paragraph: `paper/main.typ:736-753`.
- Prediction 4: `paper/main.typ:1785-1792`.
- Optional cross-reference from conclusion: `paper/main.typ:1706-1715`.

Direction:

- Promote update-rule freedom from "update-rule-agnostic" to a named
  consequence of multi-programming.
- Say the serial per-head loop removes the associative-scan/chunkwise-WY
  compatibility tax that shapes linear-state updates.
- Keep the claim bounded: the admissible family is bounded, register-resident
  per-step maps at the same matrix-state signature, not literally any arbitrary
  Python function with no kernel/backward work.
- Add the abstract mirror sentence only if it does not crowd the abstract:
  "per-step update rule free of parallel-scan requirement; parallelism recovered
  across width."

Small phrase:

`Multi-programming decouples throughput from scan-compatible update form.`

Risk:

- Low-to-medium. Overclaiming "any update" would create a kernel/autodiff
  objection. Make it a design-space claim scoped to per-step-body edits.

Validation:

- Ensure the existing per-head sequential cost at `paper/main.typ:747-749`
  remains visible.
- Check Prediction 4 still reads as falsifiable, not as a completed ablation.
- Do not change model taxonomy: Emender and M2RNN-CMA remain the two PNR
  instances; GDN remains linear-state.

### 4. Widened Stakes: Fixed-Depth Transformers And TC0

Classification: risky, optional high-value.

Editorial source: `editorial_notes.md:56-57`, `62`, `81`, and drafted paragraph
at `109-111`.

Paper locations:

- Best target: Background S5 motivation at `paper/main.typ:384-407` or the S5
  state-tracking probe at `paper/main.typ:440-451`.
- Secondary target if the edit needs earlier motivation: Introduction
  "Delta correction is one response; hybrids are another" at
  `paper/main.typ:197-206`.
- Related limitation context: `paper/main.typ:1679-1686`.

Direction:

- If used, state the exact assumptions: fixed depth in sequence length and
  bounded/log-precision or the stronger uniform-TC0 result being cited.
- The safe idea is that the "simple deterministic program over long input" issue
  is broader than linear recurrent models under standard theoretical models of
  fixed-depth transformers.
- Do not say current frontier deployed LLMs "provably fail S5" without the
  model assumptions. Do not claim the Emender "exceeds TC0"; current formal
  non-claims forbid that.

Small phrase:

`Under fixed-depth, limited-precision transformer models, the same TC0 ceiling
is a transformer ceiling as well, which is why nonlinear time-depth is a
structurally different bet.`

Risk:

- High. This is the easiest item to state too strongly. It also likely requires
  adding one or more bib entries beyond the current `merrill2024transformers`.
  Use only primary sources and exact assumptions.

Validation:

- Add and cite the exact transformer source(s) if this paragraph is used.
- Re-read `paper/main.typ:1273-1317`, `1412-1437`, and `1541-1564` after the
  edit; the widened-stakes paragraph must not contradict the explicit Lean
  non-claims.
- `rg -n "exceeds TC|exceeds NC|fixed-depth|log-precision|transformer" paper/main.typ`
  should show bounded, assumption-qualified language.

### 5. S5 Length-Degradation Honesty

Classification: must-do.

Editorial source: `editorial_notes.md:58`, `66`, and drafted paragraph at
`113-115`.

Paper locations:

- Drop immediately after the headline numbers at `paper/main.typ:1041-1047`,
  before the M2RNN-paper comparison at `1048-1058`.
- Cross-check existing limitations paragraph at `paper/main.typ:1591-1600`.
- Source table for T=1024 values: `paper/ndmpapernotes.md:169-176`.

Direction:

- State once, near the Section 6 S5 result, that the Emender does not solve S5
  to ceiling at length. It climbs higher and falls slower, but degrades from
  0.79 at T=128 to 0.42, 0.22, 0.11 at T=256, 512, 1024.
- Keep the claim comparative/mechanistic: architecture can realize solving
  weights, and empirically SGD finds a stronger but not length-generalized
  solution.
- The phrase "door is ajar, not open" is acceptable as an authorial phrase only
  if it fits the paper voice; otherwise use the factual sentence above.

Risk:

- Low. This is a trust-building honesty edit. The only risk is making the result
  sound weaker than necessary by omitting that baselines degrade faster.

Validation:

- Verify the values against `paper/ndmpapernotes.md:171-176`.
- Check Section 6 and Limitations do not now duplicate the same caveat too many
  times. The focused edit can add Section 6 ownership and leave the Limitations
  paragraph as the detailed follow-up.

### 6. Section 5 Pareto/Racer Reframe

Classification: must-do.

Editorial source: `editorial_notes.md:118-138`.

Paper locations:

- Section 5 "Loss-vs-wallclock racer": `paper/main.typ:890-943`.
- Figure caption must preserve labels and values: `paper/main.typ:892-917`.
- Existing single-seed limitation and CMA support: `paper/main.typ:1566-1589`.
- Current Figure 2 snapshot: `paper/results/figure_2/AS_OF.md:16-29`.

Direction:

- Demote "who won the horse race" to a two-axis surface: efficiency
  (loss-vs-wallclock) and capability (state-tracking probes).
- Figure 2 should support only: E88 is co-located with GDN in the efficiency
  band, so the capability advantage does not visibly impose a wallclock tax at
  this scale/training extent.
- Keep the within-PNR ordering, but ground it in CMA-replicated search plus
  the racer sign rather than the racer alone.
- Do not change the figure, data, or values in this task sequence unless a later
  paper synthesis task explicitly refreshes them.

Small phrase:

`The defensible reading is co-location on the efficiency surface, not a
single-seed margin of victory.`

Risk:

- Medium. This is an interpretation change, not a data change. It should make
  claims more defensible but must not imply a new plotted Pareto frontier from
  mixed-scale data.
- Avoid adding a precise 2-D scatter unless it is explicitly schematic; the
  efficiency axis is 1.3B and the S5 capability axis is 8M.

Validation:

- `rg -n "0.979|0.975|0.984|single seed|co-linear|efficiency|capability" paper/main.typ`
  after the edit should show the values unchanged and the single-seed scope
  explicit.
- Compare against `paper/results/figure_2/AS_OF.md:18-21`.
- Do not regenerate or commit PDFs, PNGs, or CSVs for this editorial reframe.

## Mechanical And Secondary Checks

| Item | Classification | Location | Direction | Validation |
| --- | --- | --- | --- | --- |
| Delta-notation mismatch | must-do | Body equation at `paper/main.typ:479`; Figure 1A box at `paper/main.typ:525`; related shorthand at `paper/main.typ:1122` | Reconcile `delta_h = silu(v_h) - r_h` with the figure's `delta_h = v_h - r_h`. Code paths indicate `v` is SiLU-activated before the delta in E88 sources, so the likely fix is the figure/shorthand, but verify against current model code before editing. | `rg -n "delta_h =|silu\\(v|v - S\\^T k|delta correction" paper/main.typ ndm/models/e88_fused.py ndm/models/e88_fla_hybrid.py` |
| BPB arithmetic | must-do check, edit only if mismatch | `paper/main.typ:919-931`; `paper/results/figure_2/AS_OF.md:23-29` | Current arithmetic appears consistent: `log2(e) / 3.918625 ~= 0.36816`, so 2.6599 nats/token gives 0.9793 bpb. Keep "3.92" and "0.368" rounded consistently. | Recompute or inspect `scripts/estimate_tokenizer_bytes_per_token.json`; confirm Figure 2 values remain 0.979/0.975/0.984. |
| Inline Lean theorem statements | optional, mostly already satisfied | Formal intro `paper/main.typ:1157-1174`; theorem statements at `1228-1251` and `1362-1408`; conclusion `1721-1727` | The current paper already surfaces the k-step, one-step, realization, FLOP, and latching statements more explicitly than the editorial critique describes. Do not add more unless the focused edit makes formal references more name-heavy elsewhere. | Re-read formal section after edits; theorem names should be accompanied by one-sentence mathematical content at first use. |
| Serial per-head cost | must-do balance if update-rule freedom is promoted | Systems `paper/main.typ:747-749`; distributed-training note `805-815`; limitations `1663-1668` | Preserve "cost is per-head sequential time" and consider one extra clause that this is most exposed at sufficiently long sequence lengths. Do not let the update-rule freedom paragraph read like free throughput. | `rg -n "per-head sequential|serial|time loop|sequence length" paper/main.typ` |
| Genomics concrete example | defer/risky | Background `paper/main.typ:348-358` | Only add a concrete pangenomics sentence if the author can provide an exact biological dependency and scale. Do not invent an "S5-like" genomics example. The current pangenomics motivation is broad but safe. | Require primary biology/source citation and exact number before editing. |
| Title/subtitle | optional/defer | Metadata `paper/main.typ:60` | A subtitle such as "Width-Axis Parallelism for Trainable Pure-Nonlinear Recurrent Language Models" may improve scanning, but it is not required for this focused edit and changes paper identity. Defer unless the author explicitly wants title work. | If changed, check generated PDF title, README/release references, and citation text. |
| Stability point | optional | Section 5 gradient conditioning `paper/main.typ:871-888` | The current paper already frames M2RNN-paper divergence as head geometry, not nonlinear instability. A light clause that linear-state models also carry stability burdens is optional, but avoid grievance tone. | Verify no new unsupported claim about GDN/Mamba training instability. |

## Suggested Focused Edit Order

1. Make the Section 5 racer reframe first. It is the structural center and
   determines how the abstract/intro should mirror the evidence.
2. Add frontier scoping in the abstract and introduction.
3. Promote update-rule freedom in Systems, with a short abstract mirror and
   Prediction 4 example if GDN-2 is cited.
4. Add the Section 6 S5 "not solved at length" honesty note.
5. Add the Related Work concurrent GDN-2/Mamba-3 note with verified bib entry.
6. Decide whether the transformer/TC0 widened-stakes paragraph is precise enough
   to include. If not, defer it rather than shipping loose complexity language.
7. Apply only must-do mechanical fixes: delta notation and BPB consistency check.

## Final Constraints For The Next Worker

- Preserve the Emender/PNR taxonomy and the distinction between class-level,
  within-class, empirical, and formal claims.
- Preserve all 1.3B HF release links and release hub/PDF claims unless a later
  release task explicitly changes them.
- Preserve Figure 2 values and current data state: E88 0.979, GDN 0.975,
  M2RNN-CMA 0.984.
- Preserve proof-boundary language. Do not claim the trusted Lean core proves a
  transformer lower bound, a linear-scan lower bound over all models, an
  S5-generator-specific T(d), empirical recovery of the lookup-table weights, or
  "Emender exceeds TC0/NC1".
- Do not change HF/GitHub visibility, repo settings, tags, model cards,
  checkpoints, generated PDFs, or large artifacts during the editorial edit.
