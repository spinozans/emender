# E97 DiLoCo Loss-Curve Diagnosis - 2026-06-23

Task: `diagnose-e97-diloco`

This is a read-only diagnosis of the live E97/Emender 8-GPU DiLoCo training-loss curve. I did not kill, pause, restart, reconfigure, prune, move, rename, or delete anything. I did not launch training or eval. The only live-run access was metadata/log reading from the same sources used by the PNG plot:

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`

The sample below was taken at the log state ending with step `105475` at `2026-06-23T20:05:02+00:00`. Because `run.log` is live, later samples may have more points.

## Verdict

The curve does not show evidence of a sustained loss regression. The raw training loss is visually ugly because it is high-variance at the per-log-point scale and has DiLoCo merge/checkpoint sawtooth, but the smoothed trend is flat-to-down over the useful windows.

Recommendation: **continue, but schedule a real heldout/racer eval**. The training-loss evidence is good enough not to pause/stop, but training loss alone is not a final health signal for model quality.

## Parsed Data

The parser used the same effective-lineage rule as `scripts/plot_e97_diloco_loss.py`: sort observations by timestamp/order and keep the latest observation for duplicated steps after resume rollbacks.

| item | value |
| --- | ---: |
| raw parsed loss points | 7,132 |
| effective plotted/diagnosed points | 4,219 |
| superseded pre-resume overlap points | 2,915 |
| effective step range | 25 -> 105,475 |
| latest raw loss | 2.7829 |
| latest trailing-80-point smoothed loss | 2.8519 |

## Trend Windows

Slopes are linear-regression slopes in loss per 1,000 optimizer steps. `ma80` is the trailing 80-point moving average used as the main smoothed signal. `first80` and `last80` are raw-loss means over the first and last up-to-80 logged points in the window; they are less sensitive to single endpoint spikes than raw first/last.

| window | points | steps | raw first -> last | raw min..max | raw p05/p50/p95 | ma80 first -> last | ma80 delta | raw slope / 1k | ma80 slope / 1k | first80 -> last80 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full effective run | 4,219 | 25 -> 105,475 | 8.9395 -> 2.7829 | 2.3658..8.9395 | 2.7444 / 2.9684 / 3.9056 | 8.9395 -> 2.8519 | -6.0876 | -0.0085 | -0.0101 | 5.3006 -> 2.8519 |
| since 72,500 resume | 1,320 | 72,500 -> 105,475 | 2.9730 -> 2.7829 | 2.3658..3.1598 | 2.7476 / 2.8920 / 3.0363 | 2.9321 -> 2.8519 | -0.0802 | -0.0018 | -0.0018 | 2.8790 -> 2.8519 |
| last 10k steps | 401 | 95,475 -> 105,475 | 2.9587 -> 2.7829 | 2.5674..3.1022 | 2.7247 / 2.8677 / 3.0279 | 2.8802 -> 2.8519 | -0.0283 | -0.0044 | -0.0035 | 2.8851 -> 2.8519 |
| last 5k steps | 201 | 100,475 -> 105,475 | 2.8693 -> 2.7829 | 2.5674..3.0623 | 2.7191 / 2.8599 / 2.9942 | 2.8669 -> 2.8519 | -0.0150 | -0.0063 | -0.0022 | 2.8754 -> 2.8519 |
| latest 2k steps | 81 | 103,475 -> 105,475 | 2.9127 -> 2.7829 | 2.5674..3.0487 | 2.6569 / 2.8599 / 2.9783 | 2.8623 -> 2.8519 | -0.0104 | -0.0112 | +0.0021 | 2.8535 -> 2.8519 |
| latest 1k steps | 41 | 104,475 -> 105,475 | 2.8327 -> 2.7829 | 2.5680..3.0487 | 2.6975 / 2.8346 / 2.9779 | 2.8554 -> 2.8519 | -0.0035 | -0.0200 | +0.0029 | 2.8418 -> 2.8418 |

Interpretation:

- The full-run slope is dominated by early convergence from very high loss, so it is not the right health signal for late-stage status.
- Since the 72,500 checkpoint, the smoothed loss is down by about `0.08`; the raw linear trend is also slightly negative. That argues against sustained degradation after the resume.
- Over the last 10k and 5k steps, the smoothed loss is still mildly down. The latest 1k/2k windows are effectively flat on smoothed loss, with raw points still spanning roughly `2.57..3.05`.
- A short latest-window plateau is plausible, but there is no sustained upward trend in the observed data.

## Discontinuities And Sawtooth

Observed resume markers:

| step | source | resume checkpoint |
| ---: | --- | --- |
| 500 | `run_pre_supervisor_20260622T101450Z.log`, `run_20260623T103727Z.log` | `checkpoint_step_000500_loss_5.5811.pt` |
| 72,000 | `run_20260623T103727Z.log` | `checkpoint_step_072000_loss_2.9556.pt` |
| 72,500 | `run.log` | `checkpoint_step_072500_loss_2.9730.pt` |

The `72,000` and `72,500` resumes create the duplicated-step overlap that the plot script correctly supersedes. The post-72,500 trend does not get worse after the live `run.log` resume: the step-72,500 raw loss was `2.9730`, the trailing-80 smoothed value around that boundary was about `2.9321`, and the latest trailing-80 value was `2.8519`.

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

The current run's throughput also appears in the expected healthy band. From the live `run.log` segment beginning at step `72,525` (`2026-06-23T10:38:37+00:00`) through step `105,450` (`2026-06-23T20:04:37+00:00`), progress was about `3,490` optimizer steps/hour. The mean observed `global_tok/s` over that segment was about `64.0k`, with median about `65.6k`; the latest sampled lines were still around `65.5k`.

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

At the current approximate token count (`105,475 * 8,192 ~= 864M tokens`), nearest racer rows show smoothed training losses around:

| reference model | nearest token-matched step | smooth_10k |
| --- | ---: | ---: |
| E88/NDM | 84,300 | 3.1133 |
| FLA-GDN | 105,350 | 3.1327 |
| M2RNN-CMA | 84,300 | 3.2114 |
| Mamba2 | 105,350 | 3.1837 |

The live E97/DiLoCo trailing-80 training loss at this sample is lower (`2.8519`), which is reassuring as a coarse sanity check. However, this is not a substitute for a real heldout/racer evaluation because data order, model recipe, smoothing, checkpoint basis, ScheduleFree evaluation basis, and DiLoCo effects may differ.

## Bottom Line

Evidence supports **continue** rather than pause/stop. Because the latest short window is essentially flat and raw training loss is not the right final judge, the practical recommendation is **continue but schedule heldout/racer eval** at the next convenient checkpoint. I would not change the recipe mid-run based on this curve alone.

## Validation

- Numeric trend windows reported: yes.
- Assessment distinguishes noisy raw training loss from sustained degradation: yes.
- Clear recommendation logged in this artifact: continue, but schedule heldout/racer eval.
- No live run modified: yes; live artifacts were read only.
- No eval/training/deletion/restart launched: yes.
