# catchup-parallel-diloco launch harness

This directory contains the launch harness used for task `catchup-parallel-diloco`.

Primary detached run:

- durable directory: `/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp`
- recipe: emender-mlp from `docs/SCALE_PLAN.md` section 1, schedule-free AdamW, CMA LR `0.0010071509461604343`, bf16, p50k, Pile data, per-replica `bs4`
- parallelism: DiLoCo local-SGD (`--diloco --diloco_k 250 --diloco_outer_lr 1 --diloco_outer_beta 0`)
- curve artifact: `heldout_curve.csv`, populated from checkpoints by `eval_emender_mlp_checkpoint_bpb.py`

Launch pattern:

```bash
setsid bash -c 'exec nohup experiments/catchup_parallel/run_diloco_native_detached.sh > /mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp/launcher.nohup 2>&1' &
```

The GPU lease is acquired inside the detached supervisor, so the broker heartbeat
survives the launching agent's shell exit.
