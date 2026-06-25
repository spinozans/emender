# E97 8-GPU Status Snapshot - 2026-06-25 15:30 UTC

Scope: read-only status snapshot of the active E97/Emender DiLoCo 8-GPU run. I did not stop, restart, launch, rename, delete, or modify the live training/eval run. I only inspected process state, GPU process state, existing logs, checkpoint listings, disk state, and the retention report.

## Live Status

- Snapshot command time: `2026-06-25T15:30:23Z`
- Torchrun parent: alive, PID `906526`, started `2026-06-23 10:37:27 UTC`, elapsed `2-04:52:56`.
- Training ranks: all 8 rank workers alive under PID `906526`: `907090`, `907091`, `907092`, `907093`, `907095`, `907096`, `907098`, `907099`.
- GPU occupancy: GPUs `0` through `7` each had exactly one `/usr/bin/python3` compute process, using `28204 MiB` each in `nvidia-smi --query-compute-apps`; `nvidia-smi pmon -c 1` showed the eight same worker PIDs at `99%` SM.
- Supervisor: alive, PID `934892`, command `bash scripts/supervise_emender_8gpu_diloco.sh`, started `2026-06-23 10:42:23 UTC`.
- Checkpoint janitor / retention guard: alive, PID `10772`, running the `e97_checkpoint_retention_guard.py` loop every 900 seconds with `--delete`.
- GPU collision status: none observed in `nvidia-smi`; the only compute processes listed were the eight E97 rank workers.

## Latest Training Observation

Latest observed training metric line from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`:

```text
step 257050 | loss 2.8317 | lr 1.01e-03 | grad 0.80 | tok/s 8306 | global_tok/s 66449 | elapsed_h 52.879 | time 2026-06-25T15:30:15+00:00
```

Latest checkpoint at snapshot time:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_257000_loss_2.7264.pt
```

`latest.pt` in the active run directory resolved to that checkpoint. The checkpoint file size was `7,719,673,482` bytes, and its mtime was `2026-06-25 15:29:25.978637865 +0000`.

## Progress Since Plot Sample

Comparison baseline from the latest plot-update sample:

- Plot sample: `2026-06-25T06:32:38Z`, step `225700`, loss `2.6842`.
- Fresh latest observed metric: `2026-06-25T15:30:15Z`, step `257050`, loss `2.8317`.
- Progress: `+31,350` steps over about `8h 57m 37s`, roughly `3,499 steps/hour`.
- Latest observed loss delta: `+0.1475` versus the plot sample.
- Latest checkpoint comparison: step `257000` is `+31,300` steps; checkpoint loss `2.7264` is `+0.0422` versus the plot sample.

## Checkpoints And Disk

- Current checkpoint count under `/mnt/nvme1n1/erikg/diloco_8gpu/emender`: `24`.
- Current checkpoint count in active run directory `levelE97_100m_20260623_103742`: `19`.
- Filesystem `/mnt/nvme1n1`: `14T` size, `14T` used, `791G` available, `95%` used.
- Retention is active: the guard process PID `10772` is alive and the latest report was generated at `2026-06-25T15:25:03.902528+00:00`.
- Latest retention report state: checkpoint count before `25`, keep count `23`, planned delete count `2`, deleted count `2`, deleted bytes `15,439,346,964`, after-free `848,668,336,128` bytes (`790.38 GiB`).
- A new checkpoint at step `257000` arrived after the latest retention report, explaining the current count of `24`.

## Recent Error Scan

I scanned the recent active `run.log` tail for these signatures: `NaN`, `OOM`, `out of memory`, `NCCL`, `Gloo`, save failure, hang/stuck/timeout, restart, collision, CUDA error, traceback, exception, killed, `SIGTERM`, and generic error terms.

Result for the recent active log window: no matches.

Operational notes:

- The expected DiLoCo merge lines are present and recent, including merge `#738` at step `257000`.
- Recent checkpoint save succeeded at step `257000`.
- Historical older segment logs contain expected earlier supervisor/restart/SIGTERM evidence from June 22 before the currently adopted live run; I did not find a recent restart or failure signature in the active log window sampled here.
