# Fresh E97/Emender DiLoCo Run Status Snapshot

Generated: 2026-06-24T07:34:55.036482234Z

Scope: read-only status snapshot for the live E97/Emender DiLoCo training run and checkpoint-retention/disk state. No training, supervisor, watchdog, janitor, GPU process, checkpoint, or run file was modified by this audit.

## Summary

- Live training is healthy: `torchrun` PID `906526` is alive, with eight `train.py` worker ranks (`907090`, `907091`, `907092`, `907093`, `907095`, `907096`, `907098`, `907099`) running at high GPU utilization.
- Supervisor PID `934892` is alive and supervising the existing live run.
- Checkpoint janitor PID `10772` is alive and running the retention guard every 900 seconds with `--delete`, `--latest-active 5`, `--milestone-every 10000`, and `--min-age-seconds 1800`.
- I did not identify a separate E97-specific watchdog process. The host has generic/kernel watchdog entries and unrelated `racer_watchdog.sh` processes, but the E97 restart/health watchdog role appears to be covered by `scripts/supervise_emender_8gpu_diloco.sh`.
- No unrelated GPU collision was observed: `nvidia-smi` reported exactly the eight E97 `python3` training rank PIDs as compute users, one per GPU 0-7.
- Latest observed training line: step `145625`, loss `2.8268`, per-rank `tok/s 8194`, `global_tok/s 65550`, timestamp `2026-06-24T07:34:34+00:00`.
- Latest checkpoint observed: `checkpoint_step_145500_loss_2.7492.pt`, with `latest.pt` pointing to it.
- Current checkpoint count under the E97 runs tree: `17`.
- Current free space on `/mnt/nvme1n1`: `892,371,099,648` bytes (`832G` shown by `df -h`, 95% used).
- Error scan found no recent NaN, OOM, Gloo, hang/stuck, traceback, exception, timeout, killed, failed, bus error, or segmentation entries. The only NCCL hit was normal initialization: `[DiLoCo] world_size=8 backend=nccl`.

## Process Health

Observed with `ps` at 2026-06-24T07:34Z:

| Component | Status | Evidence |
| --- | --- | --- |
| Torchrun launcher | Alive | PID `906526`, started 2026-06-23T10:37:27Z, elapsed about 20h57m, command includes `torchrun --standalone --nproc_per_node=8 train.py ... --level E97 ... --diloco ... --save_every 500` |
| Training ranks | Alive | Eight rank PIDs `907090` through `907099` except skipped numeric IDs, all children of `906526`, each consuming about 100% CPU and one GPU |
| Supervisor | Alive | PID `934892`, command `bash scripts/supervise_emender_8gpu_diloco.sh`, elapsed about 20h52m |
| Checkpoint janitor | Alive | PID `10772`, loop invoking `e97_checkpoint_retention_guard.py` every 900s |
| E97 watchdog | Not separately identified | No distinct E97 watchdog process found by `pgrep -af 'watchdog|racer_watchdog|supervise_emender|checkpoint_retention_guard|torchrun'`; supervisor is alive and appears to cover restart supervision |

Supervisor log showed the live run was adopted for restart supervision:

- `2026-06-23T10:39:16Z`: adopting existing live run PID `906526`.
- `2026-06-23T10:41:57Z`: adopting existing live run PID `906526`.
- `2026-06-23T10:42:23Z`: adopting existing live run PID `906526`; current supervisor PID `934892`.

## Training Progress

Run log: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`

Fresh tail at 2026-06-24T07:34:55Z:

```text
step 145525 | loss 2.7005 | lr 1.01e-03 | grad 1.00 | tok/s 6658 | global_tok/s 53261 | elapsed_h 20.923 | time 2026-06-24T07:32:54+00:00
step 145550 | loss 2.7079 | lr 1.01e-03 | grad 0.81 | tok/s 8280 | global_tok/s 66240 | elapsed_h 20.930 | time 2026-06-24T07:33:19+00:00
step 145575 | loss 2.8287 | lr 1.01e-03 | grad 0.87 | tok/s 8228 | global_tok/s 65824 | elapsed_h 20.937 | time 2026-06-24T07:33:44+00:00
step 145600 | loss 2.6609 | lr 1.01e-03 | grad 1.09 | tok/s 8202 | global_tok/s 65620 | elapsed_h 20.944 | time 2026-06-24T07:34:09+00:00
step 145625 | loss 2.8268 | lr 1.01e-03 | grad 0.96 | tok/s 8194 | global_tok/s 65550 | elapsed_h 20.951 | time 2026-06-24T07:34:34+00:00
```

Latest observed DiLoCo/checkpoint sequence:

- Merge `#292` at step `145500`.
- Step `145500` loss `2.7492`, `global_tok/s 55769`, timestamp `2026-06-24T07:32:23+00:00`.
- Saved checkpoint: `checkpoint_step_145500_loss_2.7492.pt`.

## GPU State

`nvidia-smi` showed all eight RTX 6000 Ada GPUs busy with the E97 ranks and no other compute users:

| GPU | Training PID | Memory Used | GPU Util |
| --- | ---: | ---: | ---: |
| 0 | `907090` | about 28.2 GB / 49.1 GB | 99% |
| 1 | `907091` | about 28.2 GB / 49.1 GB | 100% |
| 2 | `907092` | about 28.2 GB / 49.1 GB | 99% |
| 3 | `907093` | about 28.2 GB / 49.1 GB | 100% |
| 4 | `907095` | about 28.2 GB / 49.1 GB | 100% |
| 5 | `907096` | about 28.2 GB / 49.1 GB | 100% |
| 6 | `907098` | about 28.2 GB / 49.1 GB | 100% |
| 7 | `907099` | about 28.2 GB / 49.1 GB | 98% |

The compute-app query returned only these eight `/usr/bin/python3` processes, one per GPU.

## Checkpoints And Disk

Active run directory: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`

Latest active-run files included:

- `checkpoint_step_143000_loss_2.7344.pt`
- `checkpoint_step_143500_loss_2.8671.pt`
- `checkpoint_step_144000_loss_2.7500.pt`
- `checkpoint_step_144500_loss_2.8748.pt`
- `checkpoint_step_145000_loss_2.8329.pt`
- `checkpoint_step_145500_loss_2.7492.pt`
- `latest.pt -> checkpoint_step_145500_loss_2.7492.pt`

Current full E97 checkpoint inventory under `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs`:

- Checkpoint count: `17`
- Modal checkpoint size: `7,719,673,482` bytes (`7.19 GiB`)
- Active run directory size: `87G`
- `/mnt/nvme1n1` free space: `892,371,099,648` bytes (`832G`), 95% used

The latest janitor report was generated at `2026-06-24T07:24:47.128316+00:00` and reported:

- Mode: delete enabled
- Checkpoint count before that pass: `18`
- Keep count: `16`
- Deleted count: `2`
- Deleted bytes: `15,439,346,964`
- After-free bytes: `892,371,103,744`
- Newest checkpoint at that pass: `checkpoint_step_145000_loss_2.8329.pt`
- Rolling policy: latest 5 active checkpoints, every 10,000-step milestone, minimum age 1800 seconds, 900-second janitor interval

The inventory increased from the janitor's post-pass state to `17` because the live run saved `checkpoint_step_145500_loss_2.7492.pt` after the 07:24 janitor pass.

## Error Scan

Scanned:

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.log`
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/e97_checkpoint_retention_guard.log`

Patterns checked case-insensitively:

```text
nan|oom|out of memory|nccl|gloo|hang|stuck|timeout|restart|traceback|exception|error|failed|killed|bus error|segmentation
```

Findings:

- No NaN entries.
- No OOM or out-of-memory entries.
- No Gloo entries.
- No hang/stuck/timeout entries.
- No traceback/exception/error/failed/killed/bus error/segmentation entries.
- NCCL appeared only in normal initialization: `[DiLoCo] world_size=8 backend=nccl; this is rank 0 on cuda:0`.
- Supervisor `restart` matches were normal "adopting existing live run for restart supervision" lines, not a live restart event during this snapshot.
- Save/checkpoint scan showed regular checkpoint saves through `checkpoint_step_145500_loss_2.7492.pt`.

## Validation

- Fresh timestamped run health reported: yes, 2026-06-24T07:34:55Z.
- Fresh checkpoint/free-space/janitor status reported: yes, checkpoint count `17`, latest checkpoint `145500`, free space `892,371,099,648` bytes, janitor PID `10772` active with delete-mode retention policy.
- Error scan summarized: yes, with explicit patterns and findings.
- No live run modified: yes, only read-only inspection commands were used against the live run; the only repository change is this Markdown artifact.
