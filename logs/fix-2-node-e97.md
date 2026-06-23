# Fix 2-node E97 resume finalization NCCL timeout

Task: `fix-2-node-e97`
Date: 2026-06-23

## Root Cause

Job `4891784` resumed cleanly from the `4891083` checkpoint and produced finite
post-resume losses through step `205`, but shutdown/finalization failed with
NCCL watchdog timeouts:

- ranks `8-15`: `WorkNCCL SeqNum=167 OpType=ALLREDUCE NumelIn=1` timed out;
- ranks `0-7`: last enqueued/completed work remained `166`;
- the last successful logged collective was the step-`200` DiLoCo merge;
- no final-checkpoint `START` / `END` appeared before the watchdog failure.

The mismatch was a rank-local wall-clock exit decision. The training loop used
each rank's private `time.time() < train_end_time` in the `while` condition.
After the step-`200` DiLoCo merge, ranks `0-7` exited the loop for finalization
while ranks `8-15` continued training and reached the next DiLoCo merge. The
first scalar all-reduce in that merge is `Numel=1`, matching the observed
`SeqNum=167` timeout.

## Fix

Commit: `eb661972e153501ba8a64bcdb842d3e77b314a19`

Changed files:

- `train.py`
  - added `distributed_any()` for scalar consensus across the default process
    group;
  - added `consensus_final_checkpoint_stop()` so a final-checkpoint stop is
    propagated at collective-safe optimizer-step boundaries;
  - replaced rank-local distributed time-budget loop exit with a consensus stop
    after optimizer-step/checkpoint/final-checkpoint checks;
  - added consensus guards for non-finite loss and gradient-stop decisions so a
    single rank cannot break away while peers continue to a later collective.
- `tests/test_walltime_final_checkpoint.py`
  - added CPU-only fake-distributed tests for the scalar consensus helper and
    final-checkpoint peer-stop promotion.
- `scripts/frontier/diloco_scaleout_readiness.sbatch`
  - fixed the summary heredoc quoting bug by using a single-quoted Python
    heredoc and passing shell values through `sys.argv`; this prevents Markdown
    backticks in Python f-strings from being executed by the shell.

## Local Validation

Commands run:

```text
bash -n scripts/frontier/diloco_scaleout_readiness.sbatch
module load miniforge3/23.11.0-0 && python -m py_compile train.py tests/test_walltime_final_checkpoint.py
module load miniforge3/23.11.0-0 && python -m pytest tests/test_walltime_final_checkpoint.py tests/test_checkpoint_finalization.py -q
module load miniforge3/23.11.0-0 && python -m pytest tests/test_checkpoint_finalization.py tests/test_walltime_final_checkpoint.py tests/test_diloco_merge.py::test_diloco_checkpoint_roundtrip_preserves_outer_and_inner_sf_state tests/test_rocm_e97_runtime_config.py -q
```

Results:

- focused checkpoint/finalization slice: `12 passed`;
- broader checkpoint/DiLoCo/runtime-config slice: `16 passed`;
- Python syntax checks passed;
- Slurm batch syntax check passed.

The bare system `python` command is unavailable, and bare `python3` lacks
`pytest` and `torch`; the tests above used Frontier's `miniforge3/23.11.0-0`
module environment.

## Frontier Validation

Only one validation job was submitted from this follow-up. No 4-node or 8-node
jobs were submitted.

Max requested exposure was recorded before submission:

```text
2 nodes * 20 minutes = 0.666667 node-hours
```

Submitted job:

```text
Job: 4891848
Name: emender-e97-resume-canary
Partition/QOS: batch/debug
Nodes: 2
Walltime: 00:20:00
Node list: frontier[01998-01999]
Commit in manifest: eb661972e153501ba8a64bcdb842d3e77b314a19
Resume checkpoint: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
```

Accounting:

```text
4891848|emender-e97-resume-canary|COMPLETED|0:0|00:04:04|2|frontier[01998-01999]|2026-06-23T11:51:29|2026-06-23T11:55:33
4891848.0|bash|COMPLETED|0:0|00:03:44|2|frontier[01998-01999]|2026-06-23T11:51:49|2026-06-23T11:55:33
```

Run artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891848-20260623T155130Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891848-20260623T155130Z/summaries/summary.md
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891848-20260623T155130Z/logs/train.log
logs/frontier/scaleout/emender-e97-resume-canary-4891848.out
logs/frontier/scaleout/emender-e97-resume-canary-4891848.err
```

Post-resume finite losses:

```text
step    145 | loss 5.4603
step    150 | loss 4.9422
step    155 | loss 5.1172
step    160 | loss 4.2686
step    165 | loss 3.6567
step    170 | loss 2.6002
step    175 | loss 3.3575
step    180 | loss 3.2579
step    185 | loss 3.1570
step    190 | loss 2.2008
step    195 | loss 1.9489
step    200 | loss 1.3831
```

Finalization:

```text
>>> [DiLoCo] merge #6 at step 200: averaged model weights across 16 ranks in 3965 ms
>>> [DiLoCo] final merge SKIPPED at step 200: last step already merged (step % K == 0); checkpoint is already consensus
[final-checkpoint] START kind=final reason=training_complete step=200 loss=3.4459 remaining_s=970.7 model_variant=level=E97,params=100m,mlp_ratio=1.5 rank=0/16 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891848-20260623T155130Z/train/levelE97_100m_20260623_115255/checkpoint_step_000200_loss_3.4459.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891848-20260623T155130Z/train/levelE97_100m_20260623_115255/latest.pt
Training complete! Final step: 200
FINAL_LOSS_LAST100: 3.4459
DILOCO_MERGES: 6
```

Checkpoint/latest behavior:

```text
checkpoint_step_000180_loss_3.2579.pt
checkpoint_step_000190_loss_2.2008.pt
checkpoint_step_000200_loss_1.3831.pt
checkpoint_step_000200_loss_3.4459.pt
latest.pt -> checkpoint_step_000200_loss_3.4459.pt
```

No NCCL/RCCL watchdog timeout, traceback, OOM, non-finite loss, or command-not
found evidence appeared in the terminal validation. The only NCCL lines in the
successful stdout were startup warnings about device id discovery; the job
completed with exit `0:0`.

The generated summary quoting issue is fixed for this path: `summary.md`
contains literal Markdown backticks around values, and
`emender-e97-resume-canary-4891848.err` contains no line-310
`command not found` errors.
