# Figure Note: v2 CMA-ES Candidate-Burst Diagnostic

Task: `prototype-v2-cma-2`

Generated: 2026-05-30

This directory now contains a supplement-oriented prototype diagnostic figure:

- `plot_cmaes_burst.py`
- `cmaes_burst_v2.png`
- `cmaes_burst_v2.pdf`

Regenerate from the repository root with:

```bash
python3 paper/results/cmaes_burst_v2/plot_cmaes_burst.py
```

The script reads only normalized files already present in
`paper/results/cmaes_burst_v2/`:

- `cmaes_trajectory_points.csv.gz`
- `cmaes_eval_summary.csv`

It does not read raw logs, mutate raw logs, launch training, use a GPU, edit
`paper/main.typ`, publish externally, or upload release assets.

## Data Coverage

The rendered figure represents all normalized candidate trajectories in the
snapshot:

| Coverage item | Count |
| --- | ---: |
| Architecture-family panels | 13 |
| Candidate trajectories/configs | 1,285 |
| Step-loss rows | 192,147 |
| Complete usable trajectories | 1,032 |
| Partial trajectories without final `results.json` | 251 |
| Failed one-eval diagnostic trajectories | 2 |

Architecture-family coverage:

| Architecture-family panel | Trajectories/configs | Status represented |
| --- | ---: | --- |
| E88 delta | 128 | complete |
| E88 normal rerun | 34 | partial |
| E88 raw-write | 136 | complete |
| E97 | 1 | failed one-eval diagnostic |
| E97 normal | 64 | complete |
| E97 raw-update | 80 | complete |
| E97 linear-state | 64 | complete |
| GDN2 | 97 | 96 complete, 1 failed one-eval diagnostic |
| FLA-GDN | 160 | complete |
| M2RNN | 144 | complete |
| Mamba2 | 157 | partial |
| Transformer | 160 | complete |
| Transformer rerun | 60 | partial |

No additional candidate/config filter is applied by the plotting script beyond
the main-view x-axis window of `candidate_elapsed_minutes` from 0 to 15. In this
normalized snapshot, all 192,147 trajectory rows fall inside that window; the
observed candidate-local range is 0.000 to 14.617 minutes.

The normalized data itself excludes historical/background CMA-ES archives as
recorded in `cmaes_normalization_manifest.json` and `NORMALIZATION.md`. Those
upstream exclusions are not changed by this figure pass.

## Axis Choices

- X-axis: `candidate_elapsed_minutes`, computed from the first parsed step
  timestamp within each eval/config.
- X range: 0 to 15 minutes, matching the configured per-candidate training
  budget in the normalized snapshot.
- Y-axis: stdout training `loss`, recorded as natural-log cross entropy.
- Y scale: log-scaled display axis from 4.6 to 30.0 nats/token.

The x-axis is candidate-local time, not `sweep_elapsed_minutes` and not total
CMA-ES sweep duration. Complete CMA-ES sweeps span hours; the synthesis records
corrected 1.3B sweeps at roughly 8.67 to 13.87 hours and primary 2K sweeps at
roughly 36.96 to 46.05 hours. The figure should therefore be read as a
per-candidate burst diagnostic, not as evidence that full sweeps lasted
15 minutes.

## Aggregation And Smoothing

Individual candidate trajectories are plotted directly from the normalized
step-loss rows as low-alpha gray lines. The dense individual line layer is
rasterized in the PDF backend to keep the supplement artifact small and
readable while retaining vector text and summary overlays.

Summary curves are computed within each architecture-family panel by
interpolating each candidate trajectory onto a fixed 0.25-minute grid over the
observed extent of that candidate. There is no temporal smoothing. The summary
overlays are:

- median trajectory across candidates at each grid point;
- 25th to 75th percentile band;
- 10th to 90th percentile band.

Grid points beyond a candidate's observed span are left missing for that
candidate rather than extrapolated. This preserves short/truncated candidate
handling in the quantiles.

## Best-Candidate Highlighting

Highlighted trajectories come only from normalized CMA-ES best flags already
present in `cmaes_eval_summary.csv` and repeated in
`cmaes_trajectory_points.csv.gz`:

- `is_results_best_loss`
- `is_results_best_final_loss`
- `is_observed_best_loss`
- `is_observed_best_final_loss`

The figure overlays 14 average-loss best-flag trajectories and 13 final-loss
best-flag trajectories. When one candidate has both average-loss and final-loss
best flags, it is drawn with the combined best style.

No released/current racer selections are labeled or implied. The workgroup did
not produce `SELECTED_CANDIDATE_MAP.md` or a machine-readable selected-candidate
mapping artifact, so this figure intentionally omits selected-racer overlays.

## Partial And Failed Runs

Partial runs are retained and labeled at the panel level because they are part
of the normalized v2 burst snapshot:

- E88 normal rerun: 34 partial trajectories.
- Mamba2: 157 partial trajectories.
- Transformer rerun: 60 partial trajectories.

For partial runs without `results.json`, existing observed-best flags derived
from normalized `.done` minima are used only as CMA-ES best diagnostics. They
should not be interpreted as final sweep winners.

The two failed one-eval attempts are also retained and labeled as failed in the
panel subtitles:

- E97: 1 failed diagnostic trajectory.
- GDN2: 1 failed diagnostic trajectory in addition to 96 complete trajectories.

These failed trajectories are useful for coverage and provenance diagnostics but
are not comparable to complete sweeps.

## Limitations

- This is a prototype supplement diagnostic, not an integrated main-paper
  figure.
- BPB is not stored directly in these normalized CMA-ES logs, so the figure
  displays natural-log cross-entropy loss rather than BPB.
- The y-axis is log-scaled to keep early loss spikes and all architecture
  panels readable in one compact figure. Exact late-training differences should
  be read from the underlying normalized CSVs rather than estimated visually.
- Candidate start/end times are based on parsed stdout step timestamps. The
  normalization notes that per-eval stdout step lines lack explicit eval
  start/end timestamps and `.done` files lack start/end timestamps.
- Complete, partial, and failed statuses are shown together for diagnostic
  coverage. Any paper-facing comparison should decide whether to exclude failed
  one-eval diagnostics or separate partial reruns.
- No selected-racer mapping exists in this workgroup output, so selected/released
  racer labels are deliberately absent.
