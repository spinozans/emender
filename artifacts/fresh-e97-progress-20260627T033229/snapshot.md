# Fresh E97 8GPU Progress Snapshot

Generated: 2026-06-27T03:34:41Z

## Live Status

- torchrun/training: alive.
- torchrun PID: 906526, elapsed 3-16:56:34 at observation.
- Training ranks: 8 `python3 -u train.py` workers alive under torchrun:
  907090, 907091, 907092, 907093, 907095, 907096, 907098, 907099.
- Supervisor: alive, PID 934892, `bash scripts/supervise_emender_8gpu_diloco.sh`.
- Checkpoint janitor: alive, PID 10772, retention guard loop every 900 seconds.
- GPU occupancy: 8 compute apps only, one training worker per GPU, each using 28204 MiB process memory; GPU utilization was 99-100%. No GPU collision observed.

## Latest Training Observation

Parsed read-only using the existing `scripts/plot_e97_diloco_loss.py` log parser and moving-average logic, with `bytes_per_token=3.783`.

- Latest observed step: 382625.
- Latest raw loss: 2.6431.
- Latest timestamp: 2026-06-27T03:34:10+00:00.
- Latest throughput: 8177 tok/s per rank, 65418 global_tok/s.
- Effective parsed points: 15285, from 18194 raw points; superseded resume points: 2909.
- Plot moving-average window: 80 points.
- Plot-smoothed loss: 2.619104.
- Raw BPB estimate: 1.007980.
- Smoothed BPB estimate: 0.998828.
- Approximate global tokens seen: 25,075,712,000 tokens, using 65536 tokens/step.

## Progress Since Prior Snapshot

Prior snapshot baseline: step 362550, raw loss 2.4969, plot-smoothed loss 2.648225, smoothed BPB 1.009934, tokens seen 23,760,076,800 at 2026-06-26T21:49:23Z.

- Step progress: +20075.
- Token progress: +1,315,635,200 tokens, approximately +1.316B.
- Raw loss delta: +0.1462.
- Smoothed loss delta: -0.029121.
- Smoothed BPB delta: -0.011106.

## Checkpoints And Disk

- Latest checkpoint: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_382500_loss_2.7073.pt`.
- Latest checkpoint loss: 2.7073.
- `latest.pt` symlink points to `checkpoint_step_382500_loss_2.7073.pt`.
- Checkpoint count: 24 total under run root; 19 in the active 20260623 run directory.
- Disk: `/mnt/nvme1n1` is 14T size, 14T used, 448G available, 97% used.
- Retention guard report generated 2026-06-27T03:25:21Z: mode DELETE, checkpoint count before 25, keep count 23, deleted 2 redundant checkpoints, freed 15.44 GB, newest at that time was step 382000. Since then step 382500 was saved, giving the current count of 24.
- Retention policy state: milestone checkpoints plus latest 5 active checkpoints retained; newest and `latest.pt` target protected.

## Recent Error Scan

Scanned recent tails of:

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/e97_checkpoint_retention_guard.log`

Findings:

- No recent NaN, OOM/out-of-memory, NCCL, Gloo, save failure, hang/stuck, crash traceback, exception, killed process, generic error, or GPU collision hits in the scanned training-log tail.
- Supervisor log contains only old 2026-06-23 launch/adoption messages for restart supervision; no recent restart event was observed in the scan.
- Checkpoint retention guard recent log shows expected `mode=DELETE` pruning of redundant checkpoints and report writes; no error findings in the scanned tail.

## Non-Modification Note

No training, eval, upload, stop, restart, delete, rename, or live-run modification command was launched for this snapshot. The only writes were this repository artifact and WG log metadata.
