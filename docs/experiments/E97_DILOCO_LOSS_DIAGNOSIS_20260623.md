# E97 DiLoCo Loss-Curve Diagnosis - 2026-06-23

Task: `diagnose-e97-diloco`

This is a read-only diagnosis of the live E97/Emender 8-GPU DiLoCo training-loss curve. I did not kill, pause, restart, reconfigure, prune, move, rename, or delete anything. I did not launch training or eval. The only live-run access was metadata/log reading from the same sources used by the PNG plot:

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`

The sample below was refreshed at the log state ending with step `105925` at `2026-06-23T20:12:49+00:00`. Because `run.log` is live, later samples may have more points.

User clarification after the first pass: the actual concern was checkpoint disk-space consumption, not that the loss/model-quality curve looked bad. This report therefore treats the loss curve as incidental read-only trend context only. There may be no model-quality problem to solve here.

## Verdict

The curve does not show evidence of a sustained loss regression. The raw training loss is visually ugly because it is high-variance at the per-log-point scale and has DiLoCo merge/checkpoint sawtooth, but the smoothed trend is flat-to-down over the useful windows.

Recommendation: **continue**. Do not pause/stop or change the training recipe based on this loss curve. A real heldout/racer eval is still the right tool if an independent model-quality answer is needed later, but it is not required by this clarified disk-space concern.

## Parsed Data

The parser used the same effective-lineage rule as `scripts/plot_e97_diloco_loss.py`: sort observations by timestamp/order and keep the latest observation for duplicated steps after resume rollbacks.

| item | value |
| --- | ---: |
| raw parsed loss points | 7,152 |
| effective plotted/diagnosed points | 4,237 |
| superseded pre-resume overlap points | 2,915 |
| effective step range | 25 -> 105,925 |
| latest raw loss | 2.8697 |
| latest trailing-80-point smoothed loss | 2.8453 |

## Trend Windows

Slopes are linear-regression slopes in loss per 1,000 optimizer steps. `ma80` is the trailing 80-point moving average used as the main smoothed signal. `first80` and `last80` are raw-loss means over the first and last up-to-80 logged points in the window; they are less sensitive to single endpoint spikes than raw first/last.

| window | points | steps | raw first -> last | raw min..max | raw p05/p50/p95 | ma80 first -> last | ma80 delta | raw slope / 1k | ma80 slope / 1k | first80 -> last80 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full effective run | 4,237 | 25 -> 105,925 | 8.9395 -> 2.8697 | 2.3658..8.9395 | 2.7422 / 2.9678 / 3.9010 | 8.9395 -> 2.8453 | -6.0942 | -0.0085 | -0.0101 | 5.3006 -> 2.8453 |
| since 72,500 resume | 1,338 | 72,500 -> 105,925 | 2.9730 -> 2.8697 | 2.3658..3.1598 | 2.7468 / 2.8906 / 3.0358 | 2.9321 -> 2.8453 | -0.0868 | -0.0019 | -0.0018 | 2.8790 -> 2.8453 |
| last 10k steps | 401 | 95,925 -> 105,925 | 2.9418 -> 2.8697 | 2.5674..3.1022 | 2.7191 / 2.8639 / 3.0244 | 2.8812 -> 2.8453 | -0.0359 | -0.0053 | -0.0040 | 2.8881 -> 2.8453 |
| last 5k steps | 201 | 100,925 -> 105,925 | 2.7510 -> 2.8697 | 2.5674..3.0623 | 2.6975 / 2.8573 / 2.9950 | 2.8603 -> 2.8453 | -0.0150 | -0.0104 | -0.0036 | 2.8673 -> 2.8453 |
| latest 2k steps | 81 | 103,925 -> 105,925 | 2.8573 -> 2.8697 | 2.5680..3.0487 | 2.6934 / 2.8529 / 2.9779 | 2.8568 -> 2.8453 | -0.0115 | -0.0324 | -0.0014 | 2.8451 -> 2.8453 |
| latest 1k steps | 41 | 104,925 -> 105,925 | 2.8256 -> 2.8697 | 2.5680..3.0487 | 2.6934 / 2.8410 / 2.9501 | 2.8591 -> 2.8453 | -0.0138 | -0.0750 | -0.0222 | 2.8340 -> 2.8340 |

Interpretation:

- The full-run slope is dominated by early convergence from very high loss, so it is not the right health signal for late-stage status.
- Since the 72,500 checkpoint, the smoothed loss is down by about `0.087`; the raw linear trend is also slightly negative. That argues against sustained degradation after the resume.
- Over the last 10k and 5k steps, the smoothed loss is still mildly down. The latest 1k/2k windows are effectively flat on smoothed loss, with raw points still spanning roughly `2.57..3.05`.
- A short latest-window plateau is plausible, but there is no sustained upward trend in the observed data.

## Discontinuities And Sawtooth

Observed resume markers:

| step | source | resume checkpoint |
| ---: | --- | --- |
| 500 | `run_pre_supervisor_20260622T101450Z.log`, `run_20260623T103727Z.log` | `checkpoint_step_000500_loss_5.5811.pt` |
| 72,000 | `run_20260623T103727Z.log` | `checkpoint_step_072000_loss_2.9556.pt` |
| 72,500 | `run.log` | `checkpoint_step_072500_loss_2.9730.pt` |

The `72,000` and `72,500` resumes create the duplicated-step overlap that the plot script correctly supersedes. The post-72,500 trend does not get worse after the live `run.log` resume: the step-72,500 raw loss was `2.9730`, the trailing-80 smoothed value around that boundary was about `2.9321`, and the latest trailing-80 value was `2.8453`.

DiLoCo merge cadence is `K=250`, with plain averaging outer optimizer (`outer_lr=1.0`, `outer_beta=0.0`). Checkpoints are generally every 500 steps. Merge-local raw-loss jumps are noisy, but not directionally bad: for effective merge windows since 72,500, the mean post-minus-pre local delta was about `-0.0336`, with p05/p95 approximately `-0.1349/+0.0558`. The recent merge-local deltas remain mixed-to-negative rather than showing a persistent positive shock pattern.

Recent checkpoint raw losses also look like normal noisy training-loss samples, not a monotone failure:

| checkpoint step | raw loss |
| ---: | ---: |
| 100,000 | 2.8582 |
| 100,500 | 2.9048 |
| 101,000 | 2.9545 |
| 101,500 | 2.9315 |
| 102,000 | 2.9349 |
| 102,500 | 2.6907 |
| 103,000 | 2.8098 |
| 103,500 | 2.8982 |
| 104,000 | 2.8354 |
| 104,500 | 2.9981 |
| 105,000 | 2.9437 |

The current run's throughput also appears in the expected healthy band. From the live `run.log` segment beginning at step `72,525` (`2026-06-23T10:38:37+00:00`) through the latest sampled point at step `105,925` (`2026-06-23T20:12:49+00:00`), progress was about `3,489` optimizer steps/hour. The mean observed `global_tok/s` over that segment was about `64.0k`, with median about `65.6k`; the latest sampled lines were still around `65.5k`.

## Run-Note Expectations

The rough checkpoints from prior notes match the parsed curve:

| expected note | parsed observation | assessment |
| --- | --- | --- |
| step 72,500 around loss 2.97 | step 72,500 loss `2.9730` | matches |
| step 88,150 around loss 2.80 | step 88,150 loss `2.8005` | matches; this is a low raw sample, not a new baseline |
| step 91,475 around loss 2.93 | step 91,475 loss `2.9271` | matches; later samples recovered/continued with no sustained rise |
| projected healthy throughput | current median global throughput about `65.6k tok/s` and about `3.49k steps/hour` in live segment | consistent with healthy progress |

The apparent badness comes from comparing isolated raw samples: `2.8005` at 88,150 followed by `2.9271` at 91,475 looks like regression if read point-to-point, but the surrounding raw range is broad and the trailing average remains flat/down.

## Reference Comparison

I found an in-repo racer CSV at `paper/results/figure_3/combined.csv`. It is not a heldout eval of this live DiLoCo run and is not a same-recipe E97/DiLoCo reference, so it should be treated only as a rough training-loss sanity check.

At the current approximate token count (`105,925 * 8,192 ~= 868M tokens`), nearest racer rows show smoothed training losses around:

| reference model | nearest token-matched step | smooth_10k |
| --- | ---: | ---: |
| E88/NDM | 84,300 | 3.1133 |
| FLA-GDN | 105,350 | 3.1327 |
| M2RNN-CMA | 84,300 | 3.2114 |
| Mamba2 | 105,350 | 3.1837 |

The live E97/DiLoCo trailing-80 training loss at this sample is lower (`2.8453`), which is reassuring as a coarse sanity check. However, this is not a substitute for a real heldout/racer evaluation because data order, model recipe, smoothing, checkpoint basis, ScheduleFree evaluation basis, and DiLoCo effects may differ.

## Bottom Line

Evidence supports **continue** rather than pause/stop. The user clarified that the practical concern was checkpoint disk-space consumption, not model-quality degradation, so this loss-curve read does not justify any training intervention. I would not change the recipe mid-run based on this curve alone. If an independent model-quality answer is wanted for routine tracking, schedule a heldout/racer eval at a convenient checkpoint, but that is separate from the clarified disk-space issue.

## Validation

- Numeric trend windows reported: yes.
- Assessment distinguishes noisy raw training loss from sustained degradation: yes.
- Clear recommendation logged in this artifact: continue; heldout/racer eval only if an independent quality check is wanted later.
- No live run modified: yes; live artifacts were read only.
- No eval/training/deletion/restart launched: yes.
