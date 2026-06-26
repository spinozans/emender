# E97 Current Smoothed Loss/BPB Snapshot - 2026-06-26 14:11 UTC

Scope: read-only status computation for the active E97/Emender 8-GPU DiLoCo run. I inspected existing logs and checkpoint listings only. I did not modify, upload, plot, restart, delete, or evaluate the live training run.

## Method

- Parser/workflow: `scripts/plot_e97_diloco_loss.py`.
- Log sources:
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- Raw parsed points: `16,357`.
- Effective plotted points after resume deduplication: `13,442`.
- Superseded pre-resume points: `2,915`.
- Resume steps detected: `500`, `72000`, `72500`.
- Plot smoothing identifiable from the plotting script: moving average window `min(80, max(5, len(points)//40))`, which is `80` points for this snapshot.

## BPB Conversion

Using measured `bytes_per_token=3.783` from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/args.json:heldout_bytes_per_token`.

Formula:

```text
BPB = loss * log2(e) / bytes_per_token
```

For `bytes_per_token=3.783`, the multiplier is `0.381362685934`.

## Current Loss/BPB

Latest raw plotted/logged point:

- Step: `336050`
- Loss: `2.6518`
- Timestamp: `2026-06-26T14:11:02+00:00`
- Source: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- Estimated BPB: `1.011298`

Rolling/smoothed estimates over recent effective plotted points:

| Window | Step range | Loss | Estimated BPB |
| --- | ---: | ---: | ---: |
| Last 100 points | `333575`-`336050` | `2.659578` | `1.014264` |
| Last 500 points | `323575`-`336050` | `2.651064` | `1.011017` |
| Last 1000 points | `311075`-`336050` | `2.648610` | `1.010081` |
| Plot moving average, 80 points | recent 80-point plot smoother ending `336050` | `2.653707` | `1.012025` |

## Checkpoint Context

Latest checkpoint observed at inspection time:

- Path: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_336000_loss_2.7171.pt`
- Step: `336000`
- Checkpoint loss: `2.7171`
- Estimated checkpoint BPB: `1.036201`
- `latest.pt` resolved to this checkpoint at inspection time.

## Caveat

The latest raw loss is a single noisy training-log sample and is mixed relative to the recent rolling averages. The rolling windows and plot moving average are better short-horizon status indicators than the latest raw point alone, but they are still training-loss estimates rather than heldout evals.
