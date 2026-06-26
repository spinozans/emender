# Fresh E97 8GPU Progress Snapshot

Generated: 2026-06-26T21:49:23Z

## Live Status

- torchrun/training: alive.
- torchrun PID: 906526, elapsed 3-11:11:41 at observation.
- Training ranks: 8 `python3 -u train.py` workers alive under torchrun:
  907090, 907091, 907092, 907093, 907095, 907096, 907098, 907099.
- Supervisor: alive, PID 934892, `bash scripts/supervise_emender_8gpu_diloco.sh`.
- Checkpoint janitor: alive, PID 10772, retention guard loop every 900 seconds.
- GPU occupancy: 8 compute apps only, one training worker per GPU, each using 28204 MiB. No GPU collision observed.

## Latest Training Observation

Parsed read-only using the existing `scripts/plot_e97_diloco_loss.py` log parser and moving-average logic, with `bytes_per_token=3.783`.

- Latest observed step: 362550.
- Latest raw loss: 2.4969.
- Latest timestamp: 2026-06-26T21:48:44+00:00.
- Latest throughput: 8252 tok/s per rank, 66013 global_tok/s.
- Effective parsed points: 14502, from 17417 raw points; superseded resume points: 2915.
- Plot moving-average window: 80 points.
- Plot-smoothed loss: 2.648225.
- Raw BPB estimate: 0.952224.
- Smoothed BPB estimate: 1.009934.
- Approximate global tokens seen: 23,760,076,800 tokens, using 65536 tokens/step.

## Progress Since Prior Plot

Prior plot baseline: step 347325, raw loss 2.5351, plot-smoothed loss 2.640845, smoothed BPB 1.0071.

- Step progress: +15225.
- Token progress: +997,785,600 tokens, approximately +0.998B.
- Raw loss delta: -0.0382.
- Smoothed loss delta: +0.007380.
- Smoothed BPB delta: +0.002834.

## Checkpoints And Disk

- Latest checkpoint: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_362500_loss_2.6371.pt`.
- Latest checkpoint loss: 2.6371.
- `latest.pt` symlink points to `checkpoint_step_362500_loss_2.6371.pt`.
- Checkpoint count: 24 total under run root; 19 in the active 20260623 run directory.
- Disk: `/mnt/nvme1n1` is 14T size, 14T used, 448G available, 97% used.
- Retention guard report generated 2026-06-26T21:40:18Z: mode DELETE, checkpoint count before 25, keep count 23, deleted 2 redundant checkpoints, freed 15.44 GB, newest at that time was step 362000. Since then step 362500 was saved, giving the current count of 24.
- Retention policy state: milestone checkpoints plus latest 5 active checkpoints retained; newest and `latest.pt` target protected.

## Recent Error Scan

Scanned recent tails of:

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/e97_checkpoint_retention_guard.log`

Findings:

- No recent NaN, OOM/out-of-memory, NCCL, Gloo, save failure, hang/stuck, crash traceback, or GPU collision hits in the scanned tails.
- Supervisor log contains old 2026-06-23 adoption messages for restart supervision, with no recent restart event observed in the scan.
- Checkpoint retention guard recent log shows expected deletion of redundant checkpoints and report write; no errors in the scanned tail.

## Non-Modification Note

No training, eval, upload, stop, restart, delete, rename, or live-run modification command was launched for this snapshot. The only write was this repository artifact and WG log metadata.
