# SF-DiLoCo Decision and Handoff

Task: `consolidate-and-merge`

This is the entrypoint for the current ScheduleFree-DiLoCo result set. The
supporting implementation, launchers, analyzers, and reports are now on main;
use this page to choose the operational recipe and to find the evidence trail.

## Current decision

Default recipe:

```text
inner optimizer: ScheduleFree AdamW
outer DiLoCo:    plain periodic average
export basis:    ScheduleFree eval/averaged weights, mode=x
```

Use `train.py` with:

```text
--optimizer schedulefree
--diloco
--diloco_outer_optimizer avg
--diloco_outer_lr 1.0
--diloco_outer_beta 0.0
```

This is the default because it is the simplest stable arm after the P1
ScheduleFree geometry fix and it is the best local matched-token result in the
P4/P7 evidence available so far.

## Research caveats

- `sfsgd` outer with `--diloco_export_basis y` is not the default. Keep it as a
  scale/stability canary for larger independent-island tests, especially when
  merge-shock behavior matters.
- `sfsgd` outer with `--diloco_export_basis x` is rejected by the P4 local
  evidence. It produced unacceptable held-out BPB despite smoother training
  loss and should not be scheduled except as a deliberate regression check.
- Matched-gain fixed outer momentum is stable after the P1 geometry fix, but it
  is not preferred because it did not beat plain averaging in the local
  matched-token comparison.
- The island-count evidence is local through W=8. Do not generalize it as proof
  for hundreds or thousands of islands. Treat W>=16 and true large-island runs
  as new evidence, not as interpolation from W<=8.

## What is merged

- P1 geometry fix and regression tests: `train.py`,
  `tests/test_diloco_merge.py`, and
  `scripts/launch_sf_diloco_pavg_baseline.sh`.
- P2 matched-gain momentum instrumentation and summary:
  `scripts/analyze_sf_diloco_p2.py`,
  `docs/SF_DILOCO_P2_MATCHED_GAIN_REPORT.md`.
- P3 outer `sfsgd` state machine, checkpoint/resume support, and smoke launcher:
  `train.py`, `tests/test_diloco_merge.py`,
  `scripts/launch_sf_diloco_sfsgd_smoke.sh`.
- P4 four-arm comparison launcher/analyzer/report:
  `scripts/launch_sf_diloco_p4.sh`,
  `scripts/analyze_sf_diloco_p4.py`,
  `docs/SF_DILOCO_P4_OUTER_REGIME_REPORT.md`.
- P5/P6/P7 island-count design, runner, analyzer, and analysis:
  `docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md`,
  `scripts/launch_sf_diloco_p5_island_scaling.sh`,
  `scripts/analyze_sf_diloco_p5.py`,
  `docs/experiments/SF_DILOCO_P7_ISLAND_SCALING_ANALYSIS.md`.
- Scale-test rationale:
  `docs/experiments/SF_DILOCO_SCALE_TEST_RATIONALE.md`.

Generated large run outputs, logs, checkpoints, and summaries remain outside the
repo under the run roots named by the reports.

## Evidence map

- Geometry and baseline: `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`,
  sections "P1 RESULTS" and "P3 RESULTS".
- Matched-gain momentum: `docs/SF_DILOCO_P2_MATCHED_GAIN_REPORT.md`.
- Outer-regime decision: `docs/SF_DILOCO_P4_OUTER_REGIME_REPORT.md`.
- Island-count design: `docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md`.
- Island-count analysis:
  `docs/experiments/SF_DILOCO_P7_ISLAND_SCALING_ANALYSIS.md`.
- Scale-test rationale:
  `docs/experiments/SF_DILOCO_SCALE_TEST_RATIONALE.md`.

## Frontier merge substrate

Keep two decisions separate:

- **Optimizer and parameterization:** the current recipe is still plain `avg`
  outer merging, with `sfsgd_y` carried as the paired scale/stability canary.
  The local W=2/4/8 evidence does not show that `sfsgd_y` should replace
  `avg`, and it also does not prove that flat averaging will fail at larger W.
  Flat averaging may be better than the earlier concern suggested; current
  evidence simply stops before Frontier-scale island counts.
- **Merge substrate and system shape:** the Frontier question is how island
  endpoints are aggregated when W becomes tens, hundreds, or eventually
  thousands of endpoints. That substrate choice is independent of whether the
  outer update is plain averaging or `sfsgd_y`.

Treat Gloo as local/prototype infrastructure only. It is useful for regression
tests, localhost smoke runs, and small non-Frontier experiments, but it is not
the Frontier-scale merge plan. Frontier candidates to evaluate include:

- ROCm-aware MPI / Cray MPI collectives over Slingshot.
- RCCL/NCCL-style collectives if they are stable and practical for the selected
  process layout.
- Hierarchical aggregation, for example node/local group -> global merge ->
  broadcast, if a single flat collective is brittle, slow, memory-heavy, or hard
  to recover.

DiLoCo keeps high-frequency training mostly local: the inner loop does not need
a global DDP-style collective on every step. That sparsity helps, but it does
not make the merge path optional. At each outer round the job still has to move
bulk model endpoints, compute the accepted aggregate deterministically, publish
the merged state back to islands, and leave enough audit state for checkpoint,
resume, and failure handling. A Frontier-ready merge path therefore needs
systems validation separate from optimizer quality.

Before claiming thousand-island readiness, test at least:

- Flat collective versus hierarchical aggregation at the same endpoint count and
  model size, including numeric equivalence within an agreed tolerance.
- Backend choice across ROCm-aware MPI / Cray MPI over Slingshot and any
  RCCL/NCCL-style option that is stable enough to try.
- Merge throughput, tail latency, memory pressure, and amortized sync fraction
  across K-step windows.
- Deterministic replay of the aggregate from recorded endpoint/checkpoint state.
- Checkpoint/resume behavior across a merge boundary, including restored outer
  optimizer state for `sfsgd_y` canaries.
- Fault behavior: rank/node loss, partial endpoint arrival, timeout policy,
  rejected endpoint accounting, and whether a failed round can be retried or
  cleanly aborted without corrupting the accepted checkpoint.
- Tolerance rules for flat versus hierarchical numeric differences so a
  production run does not silently change optimizer conclusions because the
  aggregation tree changed.

## Parallel GPU scheduling guidance

Future independent arms should fill all available GPUs while keeping GPU and
output ownership isolated per run.

Scheduling rules:

- Treat each independent arm/seed/world-size run as an isolated job with its own
  output directory, log file, held-out curve file, and checkpoint directory.
- Acquire a GPU lease per job, set `CUDA_VISIBLE_DEVICES` inside that job only,
  and launch `torchrun --nproc_per_node=$W` from inside the leased environment.
- For an 8-GPU host, run up to four W=2 jobs concurrently, two W=4 jobs
  concurrently, or one W=8 job. Do not mix jobs in a way that oversubscribes
  visible devices.
- Never share `--output`, log, curve, or checkpoint paths between concurrently
  running arms. Refuse to overwrite existing targets unless the run is explicitly
  a cleanup/replacement pass.
- Alternate arm order across seeds so warm-cache, queue, or transient hardware
  effects do not consistently favor one arm.
- Analyze from the output root after all jobs finish, and treat missing,
  crashed, non-finite, or incomplete logs as ineligible rather than silently
  folding them into means.

`scripts/launch_sf_diloco_p5_island_scaling.sh` is the current reference
implementation of these rules. It uses per-run labels for paths, per-job GPU
leases, W-aware concurrency, and a grow-only matrix plan before running the
analyzer.

## Next scale test

For a larger true-island run, keep the primary arm as plain average and include
`sfsgd_y` only as the paired stability canary. Promote `sfsgd_y` only if the
larger run shows no held-out BPB penalty and a material stability or throughput
benefit under the same checkpoint/resume policy.
