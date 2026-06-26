# E97 Loss Curve BPB Refresh - 2026-06-26 17:26 UTC

Scope: refreshed the hosted E97/Emender 8-GPU DiLoCo loss curve from the latest available logs using the existing plotting workflow. I did not stop, restart, modify, delete, rename, or launch training/eval.

## Inputs

- Plot script: `scripts/plot_e97_diloco_loss.py`
- Run root: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`
- Log sources:
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`

## Plotted Snapshot

- Raw parsed points: `16,808`
- Effective plotted points after resume deduplication: `13,893`
- Superseded pre-resume points: `2,915`
- Resume steps: `500`, `72000`, `72500`
- Latest plotted step: `347325`
- Latest raw loss: `2.5351`
- Latest plotted timestamp: `2026-06-26T17:26:07+00:00`
- Plot smoothing: moving average over `80` points
- Plot-smoothed loss: `2.640845`

## BPB

Measured tokenizer conversion:

```text
bytes_per_token=3.783
```

Source: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/args.json:heldout_bytes_per_token`

Formula:

```text
BPB = loss * log2(e) / bytes_per_token
```

For `bytes_per_token=3.783`, `log2(e) / bytes_per_token = 0.381362685934`.

- Latest raw estimated BPB: `0.966793`
- Plot-smoothed estimated BPB: `1.007120`

## Output And Remote Verification

- Local PNG: `docs/experiments/figures/e97_diloco_loss_curve_20260623.png`
- Local SHA-256: `a7f2c0377a859e612f69516ad9dd8aa39fefe082202bee97edc4dc8a9a4fe7c5`
- Remote target: `erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png`
- Public URL: `http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png`
- Remote HTTP status: `200 OK`
- Remote content type: `image/png`
- Remote SHA-256: `a7f2c0377a859e612f69516ad9dd8aa39fefe082202bee97edc4dc8a9a4fe7c5`
- Remote hash matched local: yes
