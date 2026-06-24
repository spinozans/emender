# E97 Loss-Curve Wall-Clock Scaling

Generated: 2026-06-24T08:31Z

Scope: read-only analysis of whether the current 8-GPU E97/Emender DiLoCo run looks about 8x faster in loss-curve progress, as distinct from the already measured `8.06x` global token-throughput speedup. I did not modify `/mnt` run artifacts, change the live run, launch training, launch eval, stop or restart processes, or touch checkpoints. The only repository change is this Markdown report.

## Bottom Line

The 8-GPU run does **not** defensibly look 8x faster in training-loss curve progress. It is clearly faster than the single-GPU E97 baseline in active wall-clock time, but the comparable smoothed training-loss thresholds I could measure show roughly `3.1x-5.0x` faster progress, depending on threshold.

The previously reported `8.06x` speedup remains valid for global token throughput:

- Single-GPU baseline recent median global throughput: `8161 tok/s`.
- Current 8-GPU DiLoCo recent median global throughput: about `65620-65756 tok/s`, depending on the exact tail sample.
- Throughput ratio: about `8.0x`.

That is not the same as loss-progress or quality scaling. On the training-loss curve, the 8-GPU run reaches common smoothed-loss thresholds sooner in wall-clock time, but not by 8x. Heldout/BPB/racer eval is still needed before calling this quality scaling.

## Runs Compared

Single-GPU baseline:

- Run directory: `/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750`
- Log: `/mnt/nvme1n1/erikg/ref_emender_mlp/run.log`
- Baseline starts from step `0` and logs through step `244100` at `68.845h`.
- Latest committed comparison identified this as the clean prior single-GPU E97 baseline.

Current 8-GPU DiLoCo run family:

- Active run directory: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`
- Active log: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- Active continuation resumes from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt`
- Earlier segment logs used to reconstruct the 8-GPU loss trajectory before that resume point:
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`

The 8-GPU trajectory is segmented. I used active training elapsed time across segments, excluding calendar downtime between restarts. The chain used here is:

| Segment | Step range used | Active elapsed accounting |
| --- | ---: | --- |
| Phase 1 | `25-500` | Native elapsed from `run_phase1.log`; step `500` at `0.151h` |
| Pre-supervisor | `525-72000` | Added to phase-1 elapsed; step `72000` at segment `20.495h` |
| Recovery checkpoint segment | `72025-72500` | Added after pre-supervisor; step `72500` at segment `0.152h` |
| Current live continuation | `72525+` | Added after the reconstructed `72500` checkpoint |

This gives a reconstructed active-training time of about `20.798h` at step `72500`. The latest parsed live row while computing this report was step `148675`, loss `2.8844`, active continuation elapsed `21.821h`, reconstructed cumulative active training time `42.619h`, at `2026-06-24T08:26:48+00:00`.

## Smoothing Method

Raw train loss is noisy, and individual threshold crossings can be misleading. I therefore used a trailing median over approximately the same optimizer-step span for both logs:

- Baseline log cadence: one row per `100` steps, smoothed with `25` rows, about `2500` optimizer steps.
- 8-GPU log cadence: one row per `25` steps, smoothed with `100` rows, about `2500` optimizer steps.
- Threshold crossings below require a mature smoothing window (`25` baseline rows or `100` 8-GPU rows).

This is still training loss, not heldout loss. It is useful for a rough wall-clock progress check, not for final quality claims.

## Loss Vs Wall-Clock Time

Comparable smoothed training-loss threshold crossings:

| Smoothed train-loss threshold | Single-GPU step | Single-GPU active h | 8-GPU step | 8-GPU active h | Apparent wall-clock speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `3.50` | `42500` | `12.075h` | `10075` | `2.908h` | `4.15x` |
| `3.30` | `59100` | `16.795h` | `17950` | `5.164h` | `3.25x` |
| `3.20` | `60100` | `17.076h` | `18750` | `5.396h` | `3.16x` |
| `3.15` | `60300` | `17.132h` | `19250` | `5.539h` | `3.09x` |
| `3.10` | `60800` | `17.273h` | `19475` | `5.602h` | `3.08x` |
| `3.05` | `109200` | `30.870h` | `21325` | `6.134h` | `5.03x` |
| `3.00` | `121900` | `34.474h` | `34750` | `9.979h` | `3.45x` |
| `2.95` | `146100` | `41.331h` | `35850` | `10.294h` | `4.02x` |
| `2.90` | no mature crossing | no mature crossing | `41300` | `11.856h` | not comparable |
| `2.85` | no mature crossing | no mature crossing | `106250` | `30.481h` | not comparable |
| `2.80` | no mature crossing | no mature crossing | `131000` | `37.562h` | not comparable |

The common-threshold speedups are faster than single GPU, but they are materially below 8x. A concise reading is: the 8-GPU run appears to give about `3x-5x` active wall-clock loss-progress acceleration on this noisy train-loss measure, while token throughput is about `8x`.

The lower thresholds (`2.90`, `2.85`, `2.80`) are not comparable because the single-GPU baseline did not reach them under this smoothing rule. It had raw one-off dips below some of those values, but its mature smoothed train-loss curve did not settle there. Its final trailing-median value at the last parsed row was `3.1141`, and its best trailing-median point was `2.9089` near step `147000`.

## Loss Vs Steps

At similar optimizer steps, the reconstructed 8-GPU trajectory has lower smoothed training loss than the single-GPU baseline:

| Step neighborhood | Single-GPU active h | Single-GPU smoothed loss | 8-GPU active h | 8-GPU smoothed loss |
| ---: | ---: | ---: | ---: | ---: |
| `72500` | `20.568h` | `3.2080` | `20.798h` | `2.9510` |
| `100000` | `28.278h` | `3.2257` | `28.688h` | `2.8815` |
| `125000` | `35.346h` | `3.0042` | `35.845h` | `2.8304` |
| `148500` | `42.006h` | `2.9243` | `42.569h` | `2.7926` |

This step-normalized comparison is useful but tricky. DiLoCo steps are local optimizer steps on each rank followed by periodic averaging, not identical to a single-GPU optimizer trajectory. The 8-GPU run processes about eight times as many aggregate hardware tokens per wall-clock second, and about eight times as many aggregate tokens per nominal step if each rank sees distinct data.

## Loss Vs Tokens

There are two token conventions, and the answer changes depending on which one is meant:

1. `step * batch_size * chunk_size`, or about `step * 8192`, is the single-stream/local token convention used in several existing E97 artifacts.
2. `step * batch_size * chunk_size * world_size`, or about `step * 8192 * 8`, is the aggregate hardware-token convention for the 8-GPU DiLoCo job if ranks are consuming distinct shards.

Selected threshold crossings:

| Threshold | Single-GPU local/aggregate tokens | 8-GPU local-step tokens | 8-GPU aggregate hardware tokens |
| ---: | ---: | ---: | ---: |
| `3.20` | `0.492B` | `0.154B` | `1.229B` |
| `3.10` | `0.498B` | `0.160B` | `1.276B` |
| `3.00` | `0.999B` | `0.285B` | `2.277B` |
| `2.95` | `1.197B` | `0.294B` | `2.349B` |

Under the local-step convention, the 8-GPU run appears to reach the thresholds in fewer nominal steps/tokens. Under the aggregate hardware-token convention, it uses more total consumed tokens to get there. Neither convention alone proves quality scaling. The wall-clock question should therefore be answered directly from loss-vs-time, not inferred from token throughput.

## Throughput Cross-Check

The throughput result remains near ideal:

| Window | Mean global tok/s | Median global tok/s | Mean raw train loss |
| --- | ---: | ---: | ---: |
| Single-GPU full log | `8080.4` | `8089.0` | `3.2694` |
| Single-GPU last 100 rows | `8146.6` | `8161.0` | `3.1168` |
| Reconstructed 8-GPU chain | `64017.4` | `65475.5` | `3.0084` |
| Active 8-GPU current tail, last 100 rows | `64203.8` | `65620.0` | `2.7906` |

Recent global throughput ratio from the latest parsed tails is about `65620 / 8161 = 8.04x`, consistent with the prior `8.06x` report. The lower all-chain mean reflects merge/save and startup/recovery rows; steady rows are close to the prior report's `8.06x`.

## Caveats

- This is training loss, not heldout/BPB/racer quality. A heldout/BPB/racer eval of the current 8-GPU run is still needed for a quality-scaling claim.
- Raw train loss is very noisy and batch-order dependent. I used a trailing median over about `2500` optimizer steps, but different smoothing choices shift threshold times.
- Data order may differ between runs, and DiLoCo ranks likely see different data streams. That affects train-loss comparability.
- The 8-GPU trajectory is segmented and resumed. I reconstructed active training time from log elapsed values and excluded calendar downtime between restarts. Including downtime would make wall-clock progress slower; using only the current live continuation would make early thresholds unmeasurable because it starts from step `72500`.
- DiLoCo step semantics do not exactly match single-GPU optimizer steps. Periodic model averaging every `K=250` local steps means step-vs-step and token-vs-token comparisons are not clean controlled experiments.
- The 8-GPU run has no discovered current heldout/BPB/racer eval artifact under `/mnt/nvme1n1/erikg/diloco_8gpu/emender` in the prior comparison, so this report deliberately avoids a final quality verdict.

## Answer To The Clarified Question

No: the current 8-GPU E97/Emender DiLoCo run does not look about `8x` faster in loss-curve progress on the available training-loss evidence. It looks substantially faster, with common smoothed train-loss thresholds showing about `3.1x-5.0x` active wall-clock acceleration. The `8.06x` figure should be described as token-throughput scaling, not loss-progress or quality scaling.

## Validation

- Wall-clock loss-curve comparison reported: yes, with smoothed threshold crossing times.
- Token/step normalization caveats reported: yes, including local-step versus aggregate-token conventions and DiLoCo step semantics.
- Estimated speedup in loss progress reported where defensible: yes, `3.1x-5.0x` over common mature smoothed thresholds.
- Difference between throughput scaling and loss/quality scaling made explicit: yes.
- No live run modified or eval/training launched: yes. The analysis used read-only log inspection and parsing; the only file written was this repository report.
