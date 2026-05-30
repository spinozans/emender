# Evaluation: prototype-v2-cma

Task: `prototype-v2-cma`
Evaluator: `agent-615`
Date: 2026-05-30

## Grade

Overall score: 0.10 / 1.00
Confidence: 0.97
Rubric underspecified: no

The task rubric is explicit. It requires a reproducible prototype v2 CMA-ES
early-burst figure under `paper/results/cmaes_burst_v2/`, including a plotting
script, generated PNG/PDF/SVG output as appropriate, a short design note, clear
architecture/family grouping, interpretable many-trajectory or density/summary
representation, selected/best candidate highlighting only when mapping evidence
supports it, and no integration into the current paper or release surfaces.

This evaluation grades the branch state before this evaluation artifact was
added. At that point there were no commits ahead of `main`, no registered task
artifacts, and no prototype figure outputs in `paper/results/cmaes_burst_v2/`.
The directory contained normalized CMA-ES inputs from prior pipeline stages, but
not the plotting script, rendered figure, or figure-specific design note required
by this task.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Reproducible plotting script | 0.20 | 0.00 | No plot script for the burst figure is present. The only Python script in the output directory is `normalize_cmaes_trajectories.py`, which produces normalized CSV/JSON data rather than a figure. |
| Generated prototype figure | 0.25 | 0.00 | No PNG, PDF, or SVG figure exists under `paper/results/cmaes_burst_v2/`. |
| Early-burst visual design requirements | 0.25 | 0.00 | Because no figure was produced, there is no visual evidence of a first-window wallclock-minute plot, many-trajectory/density representation, architecture grouping, aggregate curves/bands, or interpretable clutter management. |
| Selected/best candidate handling | 0.10 | 0.00 | The upstream `map-selected-racer` evaluation found no selected-candidate mapping artifact. This task did not add a documented decision to omit highlighting due to insufficient mapping confidence. |
| Figure design note and limitations | 0.10 | 0.00 | Existing normalization documentation records dataset provenance and axis policy, but there is no figure-specific note documenting the chosen window length, axes, smoothing/aggregation choices, visual limitations, or selected-candidate handling. |
| Guardrails and delivery hygiene | 0.10 | 1.00 | No actor changes were present before evaluation, so there is no evidence of prohibited paper integration, release upload, HF update, public push, GPU training, or raw-log mutation. |

Weighted total:

`0.20*0.00 + 0.25*0.00 + 0.25*0.00 + 0.10*0.00 + 0.10*0.00 + 0.10*1.00 = 0.10`.

## Evidence Reviewed

- `wg show prototype-v2-cma` reported `Commits ahead: 0` and no completed
  actor work before this evaluation artifact.
- `wg artifact prototype-v2-cma` reported no produced artifacts.
- `git log --oneline main..HEAD` returned no actor commits before this
  evaluation artifact.
- `git diff --stat main...HEAD` returned no actor delta before this evaluation
  artifact.
- `rg --files paper/results/cmaes_burst_v2` listed normalized-data artifacts:
  `normalize_cmaes_trajectories.py`, `NORMALIZATION.md`, `QUALITY_PASS.md`,
  `CMAES_LOG_INVENTORY.md`, `cmaes_log_manifest.json`,
  `cmaes_normalization_manifest.json`, `cmaes_trajectory_points.csv.gz`,
  `cmaes_eval_summary.csv`, and `cmaes_generation_summary.csv`.
- A direct file search for figure outputs or plotting/design-note files in
  `paper/results/cmaes_burst_v2/` returned no `*.png`, `*.pdf`, `*.svg`,
  `*plot*.py`, `*figure*.py`, `*burst*.py`, or figure note file.
- The normalized dataset itself is nonempty: `cmaes_trajectory_points.csv.gz`
  has 47 columns and sample rows with architecture family, elapsed minutes,
  loss, and source-log provenance. The normalization manifest reports 192,147
  finite-loss trajectory rows across 1,285 eval trajectories with passing sanity
  checks. This means suitable inputs were available, but the requested figure
  layer was not delivered.
- `provenance/evaluations/map-selected-racer.md` graded the upstream selected
  mapping task at 0.10/1.00 because no selected-candidate mapping artifact was
  produced. A downstream plot should therefore either omit selected highlighting
  or explicitly document the insufficient-confidence basis for doing so; no such
  prototype note exists.

## Validation Checklist Assessment

- Plot script runs from the repository and regenerates the figure from
  normalized data: not met. No plotting script or generated figure exists.
- Generated figure shows the early CMA-ES burst and does not require
  GPU/training: not met for the figure; no GPU/training side effects observed.
- Visual inspection confirms many lines remain interpretable or are replaced by
  documented density/summary representation: not met. There is no figure to
  inspect.
- Selected/best candidates are highlighted only when mapping confidence supports
  it: not met as documented output. Upstream mapping confidence is insufficient,
  but this task did not produce a note explaining omission of highlights.
- Figure note records data coverage, window length, axes, smoothing/aggregation
  choices, and limitations: not met. Existing normalization docs are not a
  figure design note and omit figure-specific decisions.
- No current paper integration, release upload, HF update, or public push is
  performed: met by absence of actor changes.

## Final Assessment

The task output is effectively missing. The dependency-provided normalized data
appears usable for a future figure, but this task did not add the prototype
plotting script, rendered figure, interpretable early-window visualization, or
figure-specific design note. The calibrated grade is therefore limited to
guardrail credit for not modifying prohibited surfaces.
