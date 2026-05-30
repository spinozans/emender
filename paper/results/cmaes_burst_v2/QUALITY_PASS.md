# Quality Pass: v2 CMA-ES Burst Figure Workgroup

Task: `quality-pass-v2`

Review date: 2026-05-30

Evaluator: `agent-597`

## Verdict

Ready for dispatch.

The workgroup is structured as a sequential pipeline:

1. `inventory-cma-es`
2. `normalize-cma-es`
3. `map-selected-racer`
4. `prototype-v2-cma`
5. `synthesize-v2-cma`

This satisfies the requested separation between inventory, parsing and
normalization, selected-run mapping, figure prototyping, and synthesis.

## Rubric Status

Rubric underspecified: no.

The task includes an explicit validation checklist. I applied that checklist as
the primary rubric and treated the downstream task descriptions as the artifact
under review.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Stage separation | 1.00 | The graph has distinct downstream tasks for inventory, normalization, selected-run mapping, prototype figure, and synthesis, connected in order. |
| First-15-minute hypothesis handling | 0.95 | Inventory requires verifying or rejecting the duration assumption, and prototype requires adjusting the window if logs show a different natural window. |
| Guardrails | 0.95 | Downstream tasks forbid GPU/training work, raw-log mutation, paper edits, release uploads, and public pushes. |
| Raw-log preservation and provenance | 0.95 | Inventory forbids moving/deleting logs, normalization requires source paths for each trajectory, and selected mapping requires provenance and confidence. |
| Figure scoping | 1.00 | Prototype outputs are scoped to `paper/results/cmaes_burst_v2/`, and the task explicitly says not to integrate into `paper/main.typ` yet. |
| Downstream evaluability | 0.90 | Each task has concrete validation criteria and expected artifacts. Minor residual risk remains because exact file scopes are broad and depend on inventory findings. |

Overall score: 0.96

Confidence: 0.86

## Validation Checklist

- [x] Downstream tasks separate inventory, parsing/normalization, selected-run
  mapping, figure prototype, and synthesis.
- [x] Tasks explicitly treat first-15-min focus as a hypothesis to verify from
  available timestamps/logs.
- [x] Tasks forbid GPU training and current-paper/release edits.
- [x] Tasks preserve raw logs and require provenance for every parsed
  trajectory.
- [x] Figure task is scoped to a v2 prototype output directory, not current
  Figure 2.

## Notes For Downstream Agents

- Treat `paper/results/cmaes_burst_v2/` as the v2 work area for derived
  artifacts. Do not edit current paper text or current Figure 2.
- The first 15 minutes are a hypothesis, not a fixed plotting constraint. The
  inventory and prototype stages should report the observed window and timestamp
  units.
- Selected racer mappings must be evidence-based. Use confidence labels and
  explicitly record missing or ambiguous mappings.
- Preserve raw logs and checkpoints. Any derived row should retain enough
  provenance to identify the source path and trajectory identity.
