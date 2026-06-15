# run-ref-gdn2 launch evidence

Task: `run-ref-gdn2`
Date: 2026-06-15 UTC
Commit: `e8a5082`

## Durable output

- Directory: `/mnt/nvme1n1/erikg/ref_gdn2_mlp`
- PID file: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.pid`
- Launch manifest: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/launch_manifest.json`
- Recipe manifest: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/recipe_manifest.json`
- Training log: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.log`
- Held-out curve: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_curve.csv`
- Held-out tensor: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_pile_tail_p50k_2048_1m.pt`

## Launch state

- Detached training PID: `3139509`
- Broker-leased GPU: `1`
- Requested GPUs: `1`
- Process status at validation: alive, parent `1`, GPU 1 at 99-100% SM utilization.

## Recipe checks

- Arm: `gdn2-mlp`
- Geometry: `dim=2176`, `n_heads=30`, `depth=12`, `expansion=1`, `gdn2_mlp_ratio=3.258732449079677`, `batch_size=4`
- Data: `/home/erikg/elman/data/pile.txt`
- Tokenizer: `p50k_base`
- Held-out tensor: copied from the `ref_emender_mlp` run, same fixed Pile tail tensor
- Optimizer: `schedulefree`
- LR: `4.74e-4`
- Warmup: `0`
- Cosine/decay: absent
- Precision: `--bf16`
- Target: `244022` steps, `2,000,000,000` tokens target using `8196` tokens/step accounting
- Held-out eval mode: `y`

The log contains the startup guard:

```text
[fused-guard] rank 0/1: level=gdn2-mlp bf16=True GDN2_PATH=/home/erikg/GatedDeltaNet-2 -> FLA chunked GDN-2 fused kernel, NO eager fallback
```

## Measured curve rows

Initial measured y-mode held-out BPB is decreasing:

| step | tokens | train_loss | heldout_ce | heldout_bpb | mode |
|---:|---:|---:|---:|---:|---|
| 500 | 4,098,000 | 5.717713 | 5.667046 | 2.072381 | y |
| 1000 | 8,196,000 | 5.235021 | 5.269750 | 1.927094 | y |

Status at handoff: the run is live and has not yet reached the `>=2B` token endpoint.
The durable CSV/log/checkpoints under `/mnt/nvme1n1/erikg/ref_gdn2_mlp` are the source of truth for completion and rollover/monotonicity checks.
