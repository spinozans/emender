# E97 8-GPU DiLoCo Live Health Check - 2026-06-23 15:07 UTC

Task: `check-live-e97`

This was a read-only health check of the live E97/Emender 8-GPU DiLoCo continuation. No live run files, processes, checkpoints, labels, or metadata were modified.

## Summary

The run was alive and making forward progress at the time sampled. The current `run.log` had advanced to step 88150 at 2026-06-23T15:07:17+00:00 with loss 2.8005 and global throughput 65496 tok/s. All eight RTX 6000 Ada GPUs were occupied by the expected eight `train.py` worker processes only; no unrelated compute process was visible in `nvidia-smi`.

Latest observed checkpoint:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_088000_loss_2.9923.pt`

File mtime was 2026-06-23 15:04:47.543968038 +0000 and size was 7,719,673,482 bytes.

## Process State

Sample time: 2026-06-23T15:07:04Z.

- Torchrun leader: alive, PID 906526, state `S`, parent PID 1, elapsed about 4h29m.
- Supervisor: alive, PID 934892, state `S`, command `bash scripts/supervise_emender_8gpu_diloco.sh`, parent PID 1, elapsed about 4h24m.
- Watchdog: alive, PID 675799, state `S`, command `bash /home/erikg/ndm/.wg-worktrees/agent-1981/scripts/racer_watchdog.sh`, parent PID 1, elapsed about 2d1h.
- Training workers: eight live `python3 -u train.py` workers, PIDs 907090, 907091, 907092, 907093, 907095, 907096, 907098, and 907099, all children of torchrun PID 906526 and each consuming a GPU.

The launch manifest still points to PID 906526 and records `leased_gpu_ids` as `0,1,2,3,4,5,6,7`.

## GPU State

`nvidia-smi` showed all eight GPUs occupied by the expected training workers:

| GPU | Utilization | Memory Used / Total MiB | Temp C | Power W | Compute PID |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 99% | 28214 / 49140 | 81 | 295.34 | 907090 |
| 1 | 99% | 28214 / 49140 | 72 | 293.44 | 907091 |
| 2 | 99% | 28214 / 49140 | 79 | 294.51 | 907092 |
| 3 | 99% | 28214 / 49140 | 73 | 297.55 | 907093 |
| 4 | 100% | 28214 / 49140 | 78 | 297.98 | 907095 |
| 5 | 100% | 28214 / 49140 | 71 | 296.26 | 907096 |
| 6 | 100% | 28214 / 49140 | 79 | 295.31 | 907098 |
| 7 | 100% | 28214 / 49140 | 80 | 299.24 | 907099 |

Collision status: no unrelated compute processes were listed by `nvidia-smi --query-compute-apps`; each GPU had exactly one `/usr/bin/python3` worker from the E97 run.

## Training Progress

Latest log line sampled from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`:

`step  88150 | loss 2.8005 | lr 1.01e-03 | grad 0.94 | tok/s 8187 | global_tok/s 65496 | elapsed_h 4.496 | time 2026-06-23T15:07:17+00:00`

Recent checkpoint/log evidence:

- Step 87500 saved `checkpoint_step_087500_loss_2.8954.pt` at log time 2026-06-23T14:56:05+00:00; file mtime 2026-06-23T14:56:12.212431038+00:00.
- Step 88000 saved `checkpoint_step_088000_loss_2.9923.pt` at log time 2026-06-23T15:04:41+00:00; file mtime 2026-06-23T15:04:47.543968038+00:00.
- The current `run.log` file mtime was 2026-06-23T15:07:17.432875036+00:00, matching active log progress.

## Error And Hang Scan

Targeted scan of the current `run.log` and `supervisor.log` for `nan`, `oom`, `out of memory`, `nccl`, `gloo`, `hang`, `timeout`, `restart`, `traceback`, `error`, `failed`, `failure`, `could not`, and `exception` found:

- No NaN, OOM, traceback, failed save, checkpoint failure, timeout, or hang indicators in the active `run.log`.
- One normal active-log backend line: `[DiLoCo] world_size=8 backend=nccl; this is rank 0 on cuda:0`.
- Three supervisor lines at 10:39:16Z, 10:41:57Z, and 10:42:23Z saying it was adopting existing live run PID 906526 for restart supervision. These were adoption/supervisor handoff messages, not repeated live-run restarts after 10:42:23Z.

Historical log `run_20260623T103727Z.log` contains an earlier traceback around line 3435 before the currently supervised continuation. It is not present in the current `run.log`, and the current continuation is past step 88150 with fresh checkpoints.

## Path And Label Status

The launch command still resumes from the old checkpoint path:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt`

Newly created save artifacts are under the corrected E97 path:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/`

The active args metadata in that run directory records `"level": "E97"` and `"params": "100m"`. Current checkpoint filenames also use the current step/loss naming convention under the corrected run directory.

## Validation Checklist

- Current live/dead state reported with PID/supervisor/watchdog status: yes.
- GPU health and collision status reported: yes.
- Current/latest step, loss, throughput, checkpoint status reported: yes.
- Recent error/hang/restart scan summarized: yes.
- No changes made to the live training run: yes; only this repository report artifact was created.
