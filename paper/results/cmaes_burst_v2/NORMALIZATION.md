# CMA-ES Burst v2 Normalization

This directory contains a reproducible normalization pass over the CMA-ES logs
inventoried by `inventory-cma-es`. Raw logs are read in place from the absolute
paths recorded in `cmaes_log_manifest.json`; they are not copied, moved, or
modified.

## Rebuild

Run from the repository root:

```bash
python3 paper/results/cmaes_burst_v2/normalize_cmaes_trajectories.py \
  --inventory paper/results/cmaes_burst_v2/cmaes_log_manifest.json \
  --out-dir paper/results/cmaes_burst_v2
```

The script uses only the Python standard library. It writes deterministic
derived outputs for a fixed raw-log snapshot. The corrected E88 and Transformer
reruns were partial/in-flight in the upstream inventory, so rerunning after
those raw directories change may legitimately change the derived snapshot.

## Outputs

- `cmaes_trajectory_points.csv.gz`: gzip-compressed CSV with one row per parsed
  stdout step-loss point. It includes family/model, sweep/run/config ids, seed,
  UTC timestamp, sweep and per-candidate elapsed minutes, step/iteration, loss,
  CMA/final loss metadata, best flags, and source log provenance.
- `cmaes_eval_summary.csv`: one row per parsed eval trajectory, including the
  source `stdout.txt`, `.done`, `params.json`, and resolved `args.json` paths.
- `cmaes_generation_summary.csv`: one row per CMA-ES generation where
  `generations.jsonl` exists, including generation wallclock timestamps and
  best-loss fields.
- `cmaes_normalization_manifest.json`: machine-readable manifest with counts,
  axis policy, unsupported/excluded source notes, per-run exclusions, and sanity
  check results.

## Axis And Metrics

The primary trajectory axis is `wallclock_timestamp_utc`, parsed from each
`stdout.txt` step line. The script also emits:

- `sweep_elapsed_minutes`: minutes since the first parsed step timestamp in the
  sweep/run.
- `candidate_elapsed_minutes`: minutes since the first parsed step timestamp in
  that eval/config.
- `stdout_elapsed_h` and `stdout_elapsed_minutes`: the trainer-reported elapsed
  value, retained as a fallback/check.

The stored loss is natural-log cross-entropy from stdout step lines. CMA-ES BPB
is not stored in the raw artifacts; downstream plotting can derive it from loss
if needed. `cma_loss` is the `.done`/`results.json` average-loss fitness when
available, and `final_loss` is the recorded final-loss value.

## Coverage

The normalized snapshot covers all v2-relevant accessible runs listed by the
inventory: corrected 1.3B reruns, primary 2K roots, partial Mamba2/corrected
E88/corrected Transformer snapshots, and the two failed Emender one-eval
attempts. Historical or diagnostic roots listed in the inventory are excluded
from the normalized dataset and recorded in `excluded_sources` because the
inventory marks them as warm-start/background material rather than final v2
burst sources.

Per-run exclusions in `cmaes_normalization_manifest.json` record generated eval
directories that lacked a usable `stdout.txt`, `.done`, or step-loss lines at
normalization time.

## Sanity Checks

`cmaes_normalization_manifest.json` records the sanity checks run by the script:
nonempty trajectory rows, expected architecture labels, monotonic step/time
within each trajectory, finite loss values, and existing source log paths.
