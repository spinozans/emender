# Synthesis: v2 CMA-ES Burst Figure Workgroup

Task: `synthesize-v2-cma`

Generated: 2026-05-30

This note summarizes the CMA-ES burst figure workgroup for a future v2 paper
update. It is a synthesis artifact only. It does not edit `paper/main.typ`,
publish figures, upload release assets, or modify raw logs.

## Bottom Line

Observed: the workgroup successfully produced a log inventory and normalized
trajectory dataset, but it did not produce the selected-racer mapping or the
prototype figure requested by the downstream tasks.

Interpretation: the normalized CMA-ES data is strong enough for an internal
diagnostic and likely strong enough for a supplement-style v2 figure after one
focused plotting pass. It is not ready as a main v2 paper figure yet, because
there is no rendered figure to inspect and no evidence-backed mapping from
released/current racer models to CMA-ES candidates.

Recommended status now: keep this as an internal diagnostic dataset. Promote to
a supplement figure only after the follow-up task listed below creates and
visually validates a real figure. Do not make it a main paper figure unless a
selected-racer mapping artifact is produced and the figure carries a clear
paper argument beyond search provenance.

## Artifacts Produced

Observed workgroup artifacts:

- `paper/results/cmaes_burst_v2/CMAES_LOG_INVENTORY.md`
- `paper/results/cmaes_burst_v2/cmaes_log_manifest.json`
- `paper/results/cmaes_burst_v2/normalize_cmaes_trajectories.py`
- `paper/results/cmaes_burst_v2/NORMALIZATION.md`
- `paper/results/cmaes_burst_v2/cmaes_trajectory_points.csv.gz`
- `paper/results/cmaes_burst_v2/cmaes_eval_summary.csv`
- `paper/results/cmaes_burst_v2/cmaes_generation_summary.csv`
- `paper/results/cmaes_burst_v2/cmaes_normalization_manifest.json`
- `paper/results/cmaes_burst_v2/QUALITY_PASS.md`
- `provenance/evaluations/map-selected-racer.md`
- `provenance/evaluations/prototype-v2-cma.md`
- `paper/results/cmaes_burst_v2/SYNTHESIS.md`

Not produced:

- No `paper/results/cmaes_burst_v2/SELECTED_CANDIDATE_MAP.md`.
- No machine-readable selected-candidates CSV/JSON/YAML overlay file.
- No burst plotting script such as `plot_cmaes_burst.py`.
- No rendered prototype figure in PNG, PDF, or SVG form.
- No figure-specific design note from the prototype task.

The `map-selected-racer` and `prototype-v2-cma` tasks completed through
evaluation artifacts rather than through the originally requested mapping and
figure outputs. Their evaluations report the requested outputs as effectively
missing.

## Logs Found

Observed accessible CMA-ES sources:

- Fresh corrected 1.3B reruns:
  `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529`
- Primary 2K E88 delta/raw root:
  `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_e88_delta_raw_warm512_20260526`
- Primary 2K baseline root:
  `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526`
- Failed Emender E97/GDN2 attempt root:
  `/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528`
- Related long racer logs and checkpoints:
  `/tmp/pile_convergence_3arch/ctx2k`,
  `/tmp/pile_convergence_m2rnn/ctx2k`, and
  `/tmp/figure2_refresh_snapshot_20260529T180451Z`
- Historical/diagnostic CMA-ES archives under
  `/home/erikg/elman/benchmark_results/` and `/home/erikg/elman/cmaes_logs`

Observed field coverage:

- `generations.jsonl` is present for complete primary 2K sweeps, partial
  Mamba2, and corrected reruns except the failed one-eval Emender attempts.
- `results.json` is present for complete usable 2K E88 delta, E88 raw-write,
  FLA-GDN, M2RNN `m2rnn_home_xma`, Transformer, and corrected E97, E97 raw,
  GDN2, and E97 linear-state reruns.
- `.done`, `params.json`, and `stdout.txt` are present for the parsed eval
  trajectories.
- Step lines include natural-log cross-entropy loss, step/iteration,
  elapsed-hour fields, and parseable UTC `time` values in the normalized
  snapshot.
- BPB is not stored directly in the CMA-ES logs; it would need to be derived
  from loss.

## Inaccessible Or Incomplete Logs

Observed missing or incomplete artifacts recorded by the normalization
manifest:

- `/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/protocol.md`
  is missing.
- `/home/erikg/emender/benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/anchors.json`
  is missing.
- `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/mamba2/mamba2_20260527_183522/results.json`
  is missing.
- `/home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/mamba2/mamba2_20260527_183522/eval_157/.done`
  is missing.
- `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/e88/e88_20260530_151323/results.json`
  was missing at inspection time and the run should be treated as partial.
- `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/transformer/transformer_20260530_042721/results.json`
  was missing at inspection time and the run should be treated as partial.

Observed accessible but excluded background archives:

- `/home/erikg/elman/benchmark_results/cmaes_1270M_20260525`
- `/home/erikg/elman/benchmark_results/cmaes_1270M_anchored_20260525`
- `/home/erikg/elman/benchmark_results/cmaes_1270M_e88_raw_20260525`
- `/home/erikg/elman/benchmark_results/cmaes_converge`
- `/home/erikg/elman/benchmark_results/cmaes_32k`
- `/home/erikg/elman/cmaes_logs`

Interpretation: these exclusions are reasonable for the v2 burst question
because they are short-context, warm-start, legacy, long-context, or older
diagnostic archives rather than the final 2K or corrected 1.3B burst sources.

## Data Coverage

Observed normalized totals from `cmaes_normalization_manifest.json`:

- Architecture labels: 13
- Broad model families represented: E88, E97, GDN2, FLA-GDN, M2RNN, Mamba2,
  Transformer
- Inventory-subset runs/sweeps with parsed trajectories: 14
- Parsed eval trajectories/configs: 1,285
- Source stdout logs: 1,285
- Step-loss trajectory rows: 192,147
- CMA-ES generation rows: 162

Observed status split:

- Complete usable runs: 9
- Partial runs with parsed trajectories but no final `results.json`: 3
- Failed one-eval attempts: 2

Coverage by architecture label:

| Architecture label | Status represented | Runs | Eval trajectories/configs | Step-loss rows |
| --- | --- | ---: | ---: | ---: |
| E88 delta | complete usable | 1 | 128 | 15,355 |
| E88 normal rerun | partial, no final `results.json` | 1 | 34 | 4,254 |
| E88 raw-write | complete usable | 1 | 136 | 13,898 |
| E97 | failed after one eval | 1 | 1 | 38 |
| E97 linear-state | complete usable | 1 | 64 | 11,208 |
| E97 normal | complete usable | 1 | 64 | 10,875 |
| E97 raw-update | complete usable | 1 | 80 | 13,821 |
| FLA-GDN | complete usable | 1 | 160 | 23,851 |
| GDN2 | complete usable plus one failed one-eval attempt | 2 | 97 | 18,114 |
| M2RNN | complete usable | 1 | 144 | 12,195 |
| Mamba2 | partial, no final `results.json` | 1 | 157 | 20,842 |
| Transformer | complete usable | 1 | 160 | 43,997 |
| Transformer rerun | partial, no final `results.json` | 1 | 60 | 3,699 |

Observed full-sweep wallclock coverage in the normalized step data spans from
2026-05-26T03:07:34Z through 2026-05-30T20:37:22Z. Individual complete sweeps
are much longer than 15 minutes: the inventory records corrected 1.3B sweep
elapsed times of roughly 8.67 to 13.87 hours and complete primary 2K sweep
elapsed times of roughly 36.96 to 46.05 hours.

## First 15-Minute Assumption

Observed:

- All 1,285 parsed eval trajectories have `train_minutes` equal to `15`.
- The normalized candidate-level window is measured from the first parsed
  step timestamp to the last parsed step timestamp in each eval.
- Across all parsed eval trajectories, observed span statistics are:
  minimum 0.20 min, 5th percentile 13.87 min, median 14.32 min,
  95th percentile 14.53 min, maximum 14.62 min.
- Five eval trajectories have observed parsed spans under 10 minutes:
  corrected E97 eval 2, corrected E97-linear evals 13/42/45, and corrected
  GDN2 eval 50.

Interpretation:

- The first approximately 15-minute candidate window is supported as a
  configured candidate budget and as the dominant observed per-eval trajectory
  length.
- The exact plotted candidate window should be described as "first logged
  training trajectory under a configured 15-minute budget", not as exact
  wallclock start-to-stop duration.
- The complete CMA-ES searches are not 15-minute runs. A paper caption must
  not imply that a full sweep completed in 15 minutes.
- A robust figure should plot candidate elapsed minutes from 0 to 15 and either
  keep or explicitly mark short/truncated evals. The prototype task should not
  use sweep elapsed time if the intended visual is per-candidate burst behavior.

## Selected-Run Mapping

Observed:

- The selected-racer mapping artifact requested by `map-selected-racer` was not
  produced.
- No released/current racer model is mapped to a normalized `sweep_id`,
  `config_id`, or `eval_<id>` by any workgroup artifact.
- No plot-consumable selected-candidates file exists.
- The normalized CSVs do include best-candidate flags derived from
  `results.json` or observed `.done` minima. These support CMA-ES best-candidate
  overlays, not released-racer selected-model overlays.

Supported CMA-ES best-candidate overlays from normalized data include:

- Corrected E97 normal: best average-loss eval 59; best final-loss eval 45.
- Corrected E97 raw-update: best average-loss and final-loss eval 58.
- Corrected GDN2: best average-loss eval 13; best final-loss eval 90.
- Corrected E97 linear-state: best average-loss and final-loss eval 34.
- Corrected E88 partial: observed best average-loss eval 27.
- Corrected Transformer partial: observed best average/final eval 42.
- 2K E88 delta: best average-loss and final-loss eval 99.
- 2K E88 raw-write: best average-loss and final-loss eval 119.
- 2K FLA-GDN: best average-loss and final-loss eval 141.
- 2K M2RNN `m2rnn_home_xma`: best average-loss eval 44; best final-loss
  eval 125.
- 2K Mamba2 partial: observed best average/final eval 58.
- 2K Transformer: best average-loss eval 140; best final-loss eval 18.

Interpretation:

- A follow-up figure may highlight "CMA-ES best by average loss" and "best
  final loss" candidates using the existing flags.
- It should not label any point as a selected or released racer model until a
  separate mapping artifact links racer checkpoint/log provenance to a CMA-ES
  candidate with confidence.

## Prototype Figure Status

Observed:

- `prototype-v2-cma` did not produce a plotting script.
- `prototype-v2-cma` did not produce a PNG, PDF, or SVG.
- `prototype-v2-cma` did not produce a figure design note.
- A direct search under `paper/results/cmaes_burst_v2/` found no figure outputs
  and no plot or burst figure script.

Interpretation:

- There is no prototype figure to describe visually. The prototype figure shows
  nothing because it was not generated.
- The normalized data is nonempty and appears suitable for a real prototype,
  but any claim about visual clarity, architecture separation, clutter, or
  search-burst shape remains unvalidated until a rendered figure exists.

## Recommendation

Do not integrate this into the current paper or release assets.

Recommended next task: `Prototype: v2 CMA-ES burst supplement figure from normalized data`.

Scope for that task:

- Add `paper/results/cmaes_burst_v2/plot_cmaes_burst.py`.
- Generate `paper/results/cmaes_burst_v2/cmaes_burst_v2.png` and
  `paper/results/cmaes_burst_v2/cmaes_burst_v2.pdf`; SVG is optional if the
  file stays readable.
- Add `paper/results/cmaes_burst_v2/FIGURE_NOTE.md`.
- Read only the normalized files already present in this directory.
- Plot `candidate_elapsed_minutes` from 0 to 15, not full sweep elapsed time,
  for the main burst view.
- Facet or small-multiple by architecture group so 1,285 trajectories do not
  collapse into a single unreadable panel.
- Draw individual trajectories at low alpha and rasterize dense line layers
  where the backend supports it.
- Add per-family median and interquartile bands or median plus 10th/90th
  percentiles.
- Highlight only CMA-ES best candidates from existing best flags unless a new
  `SELECTED_CANDIDATE_MAP.md` and machine-readable selected overlay are
  produced.
- Visually distinguish complete, partial, and failed/diagnostic runs, or omit
  failed one-eval attempts from the paper-facing view while documenting the
  omission.
- Include a caption-ready note that says candidates were configured for
  15-minute training, while full CMA-ES sweeps span hours.
- Run the script from a clean repository checkout and verify it regenerates the
  same figure without GPU or training.
- Do not edit `paper/main.typ`, upload to a release, or publish externally.

Missing artifacts needed before main-paper promotion:

- An evidence-backed `SELECTED_CANDIDATE_MAP.md`.
- A machine-readable selected-candidates file with at least model/checkpoint
  name, racer log path, candidate `sweep_id`, candidate `config_id`/`eval_id`,
  provenance, and confidence.
- Final `results.json` files for currently partial corrected E88 and
  Transformer reruns if those partial snapshots are to be used as complete
  comparisons.
- The missing 2K Mamba2 `results.json` or an explicit decision to treat Mamba2
  as a partial observed-best-only trajectory source.
- The missing Emender failed-attempt `protocol.md` and `anchors.json` only if
  those failed one-eval attempts are to be discussed beyond diagnostics.

## Validation Against Task Checklist

- Exact artifacts/scripts/figures produced: recorded above, including the
  absence of mapping and figure outputs.
- Data coverage: recorded by architecture, runs/sweeps, configs/trajectories,
  step rows, generation rows, time window, and inaccessible logs.
- Observed data versus interpretation: separated in each major section.
- Selected racer mapping: no released/current racer mapping is supported;
  CMA-ES best-candidate flags are separately documented.
- Next v2 steps: recommended as an internal diagnostic now, supplement figure
  after a concrete plotting task, with no current paper or release changes.
