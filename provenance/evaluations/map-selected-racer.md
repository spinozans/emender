# Evaluation: map-selected-racer

Task: `map-selected-racer`
Evaluator: `agent-612`
Date: 2026-05-30

## Grade

Overall score: 0.10 / 1.00
Confidence: 0.96
Rubric underspecified: no

The task rubric is explicit. It requires a v2 artifact mapping released/current
racer models and known best/selected candidates to CMA-ES run/config IDs where
evidence supports the mapping, with provenance, confidence, explicit ambiguity
handling, and a machine-readable selected-candidates output when mappings exist.
It also forbids paper edits, GPU jobs, release uploads, and raw-log
modifications.

This evaluation grades the branch state before this evaluation artifact was
added. At that point there were no commits ahead of `main`, no registered task
artifacts, and no selected-candidate mapping output in
`paper/results/cmaes_burst_v2/`.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Selected/best racer mapping artifact | 0.35 | 0.00 | The required `paper/results/cmaes_burst_v2/SELECTED_CANDIDATE_MAP.md` or equivalent artifact is absent. No released/current racer model is mapped to a normalized CMA-ES run/config ID. |
| Provenance and confidence per mapping | 0.25 | 0.00 | No mappings exist, so there is no provenance from manifests, checkpoint names, config IDs, training logs, HF/release notes, or prior WG artifacts, and no confidence labels. |
| Ambiguity and missing-mapping handling | 0.15 | 0.00 | The task specifically required ambiguous or missing mappings to be called out rather than guessed. No ambiguity register or missing-evidence section was produced. |
| Machine-readable selected-candidates output | 0.15 | 0.00 | No CSV/JSON/YAML selected-candidates file was produced for the downstream plot task. |
| Guardrails and delivery hygiene | 0.10 | 1.00 | No actor changes were present before evaluation, so there is no evidence of prohibited paper edits, GPU jobs, release uploads, or raw-log modifications. |

Weighted total:

`0.35*0.00 + 0.25*0.00 + 0.15*0.00 + 0.15*0.00 + 0.10*1.00 = 0.10`.

## Evidence Reviewed

- `wg show map-selected-racer` reported `Commits ahead: 0` and no completed
  work before this evaluation artifact.
- `wg artifact map-selected-racer` reported no produced artifacts.
- `git log --oneline main..HEAD` returned no actor commits before this
  evaluation artifact.
- `git diff --stat main...HEAD` returned no actor delta before this evaluation
  artifact.
- `rg --files paper/results/cmaes_burst_v2` listed only the normalized CMA-ES
  inputs and quality/inventory artifacts:
  `normalize_cmaes_trajectories.py`, `NORMALIZATION.md`,
  `cmaes_trajectory_points.csv.gz`, `cmaes_eval_summary.csv`,
  `cmaes_generation_summary.csv`, `cmaes_normalization_manifest.json`,
  `cmaes_log_manifest.json`, `CMAES_LOG_INVENTORY.md`, and `QUALITY_PASS.md`.
  No selected-candidate map or plot-consumable selected-candidate data file was
  present.

## Validation Checklist Assessment

- Artifact maps selected/best racer models to CMA-ES run/config IDs where
  evidence supports it: not met.
- Each mapping includes provenance and confidence: not met.
- Ambiguous or missing mappings are explicitly called out rather than guessed:
  not met.
- Machine-readable selected-candidates output can be consumed by the plot task
  if mappings exist: not met.
- No paper edits, GPU jobs, release uploads, or raw-log modifications are
  performed: met by absence of actor changes.

## Final Assessment

The task output is effectively missing. The normalized CMA-ES inputs from the
dependency are present, but this task did not add the selected-racer mapping,
provenance/confidence table, ambiguity notes, or machine-readable overlay file
needed by the downstream v2 CMA-ES plotting task. The calibrated grade is
therefore limited to guardrail credit for not modifying prohibited surfaces.
