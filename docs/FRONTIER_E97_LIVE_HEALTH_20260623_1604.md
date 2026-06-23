# E97 8-GPU DiLoCo Live Health Check - 2026-06-23 16:04 UTC

Task: `check-live-e97-2`

This was a read-only health snapshot of the live E97/Emender 8-GPU DiLoCo continuation. No live run files, processes, checkpoints, labels, metadata, or launch state were modified.

## Summary

The run was alive and making forward progress at the time sampled. The current `run.log` had advanced to step 91475 at 2026-06-23T16:04:23+00:00 with loss 2.9271 and global throughput 65573 tok/s. Torchrun, the supervisor, and the watchdog were all alive.

All eight RTX 6000 Ada GPUs were occupied by the expected eight `train.py` worker processes only. No unrelated compute process was visible in `nvidia-smi --query-compute-apps` or `nvidia-smi pmon`.

Latest observed checkpoint:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_091000_loss_2.9588.pt`

`latest.pt` resolved to that checkpoint. File mtime was 2026-06-23 15:56:23.553199032 +0000 and size was 7,719,673,482 bytes.

## Process State

Sample time: 2026-06-23T16:04:34Z.

- Torchrun leader: alive, PID 906526, state `Ssl`, parent PID 1, elapsed about 5h27m.
- Supervisor: alive, PID 934892, state `SNs`, command `bash scripts/supervise_emender_8gpu_diloco.sh`, parent PID 1, elapsed about 5h22m.
- Watchdog: alive, PID 675799, state `S`, command `bash /home/erikg/ndm/.wg-worktrees/agent-1981/scripts/racer_watchdog.sh`, parent PID 1, elapsed about 2d2h.
- Training workers: eight live `python3 -u train.py` workers, PIDs 907090, 907091, 907092, 907093, 907095, 907096, 907098, and 907099. All were children of torchrun PID 906526 and each was using one GPU.

The launch manifest still points to PID 906526, records `gpus_requested` as 8, and records `leased_gpu_ids` as `0,1,2,3,4,5,6,7`.

## GPU State

`nvidia-smi` showed all eight GPUs occupied by the expected training workers:

| GPU | Utilization | Memory Used / Total MiB | Temp C | Power W | Compute PID |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100% | 28214 / 49140 | 80 | 291.60 | 907090 |
| 1 | 100% | 28214 / 49140 | 72 | 294.31 | 907091 |
| 2 | 100% | 28214 / 49140 | 77 | 295.15 | 907092 |
| 3 | 100% | 28214 / 49140 | 72 | 296.50 | 907093 |
| 4 | 100% | 28214 / 49140 | 74 | 295.14 | 907095 |
| 5 | 100% | 28214 / 49140 | 72 | 296.08 | 907096 |
| 6 | 100% | 28214 / 49140 | 79 | 294.84 | 907098 |
| 7 | 100% | 28214 / 49140 | 80 | 296.53 | 907099 |

`nvidia-smi pmon -c 1` reported SM occupancy of 99% on each GPU for those same eight worker PIDs. Collision status: no unrelated compute processes were listed; each GPU had exactly one `/usr/bin/python3` E97 worker.

## Training Progress

Latest log line sampled from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`:

`step  91475 | loss 2.9271 | lr 1.01e-03 | grad 1.08 | tok/s 8197 | global_tok/s 65573 | elapsed_h 5.448 | time 2026-06-23T16:04:23+00:00`

Recent checkpoint/log evidence:

- Step 90000 saved `checkpoint_step_090000_loss_2.9428.pt` at log time 2026-06-23T15:39:05+00:00; file mtime 2026-06-23T15:39:11.761561034+00:00.
- Step 90500 saved `checkpoint_step_090500_loss_2.8613.pt` at log time 2026-06-23T15:47:39+00:00; file mtime 2026-06-23T15:47:47.165134034+00:00.
- Step 91000 saved `checkpoint_step_091000_loss_2.9588.pt` at log time 2026-06-23T15:56:16+00:00; file mtime 2026-06-23T15:56:23.553199032+00:00.
- The current `run.log` file mtime was 2026-06-23T16:04:23.283944531+00:00 during sampling, matching active log progress.

## Error And Hang Scan

Targeted scan of the current `run.log` and `supervisor.log` for `nan`, `oom`, `out of memory`, `nccl`, `gloo`, `hang`, `timeout`, `restart`, `traceback`, `error`, `failed`, `failure`, `could not`, `exception`, `checkpoint`, and `save` found:

- No NaN, OOM, traceback, failed save, checkpoint failure, timeout, hang, or exception indicators in the active `run.log`.
- One expected backend line in `run.log`: `[DiLoCo] world_size=8 backend=nccl; this is rank 0 on cuda:0`.
- Normal checkpoint save lines through step 91000. No checkpoint/save error line was observed.
- Three supervisor handoff/adoption lines at 2026-06-23T10:39:16Z, 10:41:57Z, and 10:42:23Z saying it was adopting existing live run PID 906526 for restart supervision. No repeated restart loop appears after the final 10:42:23Z supervisor handoff.

## Path And Label Status

The launch command still resumes from the old checkpoint path:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt`

Newly created save artifacts are under the corrected E97 path:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/`

The active args metadata in that run directory records `"level": "E97"` and `"params": "100m"`. Current checkpoint filenames use the current `checkpoint_step_<step>_loss_<loss>.pt` naming convention under the corrected run directory. I did not observe any separate new label-fix metadata file beyond `args.json`, `launch_manifest.json`, and the corrected E97 artifact path/name.

## Validation Checklist

- Fresh timestamped health status logged: yes, sampled at 2026-06-23T16:04:34Z and logged to WG.
- PID/supervisor/watchdog status reported: yes.
- GPU/collision status reported: yes.
- Step/loss/tok/s/checkpoint reported: yes.
- Error scan summarized: yes.
- No changes made to live run: yes; only this repository report artifact was created.
