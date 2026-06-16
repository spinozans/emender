# fix-sf-diloco K=250 Smoke

Task: `fix-sf-diloco`
Date: 2026-06-16 UTC
Worktree: `/home/erikg/ndm/.wg-worktrees/agent-1495`

## Command

GPUs were leased through the project broker:

```bash
eval "$(scripts/gpu_lease.sh acquire 2 --timeout 300)"
```

The smoke used the emender-mlp family (`level E97` with the MLP branch), reduced
geometry for a short validation run, real NCCL ranks, real ScheduleFree AdamW,
and DiLoCo local-SGD with `K=250`.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --level E97 --dim 256 --depth 2 --n_heads 8 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64 \
  --lr 0.0010071509461604343 \
  --data /mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt \
  --bf16 --batch_size 1 --chunk_size 256 --steps 520 \
  --output /mnt/nvme1n1/erikg/fix_sf_diloco_smoke_20260616T053809Z/train \
  --optimizer schedulefree --seed 42 --save_every 999999 \
  --keep_checkpoints 1 --tokenizer p50k_base --log_every 25 \
  --diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0
```

Full log:
`/mnt/nvme1n1/erikg/fix_sf_diloco_smoke_20260616T053809Z/smoke.log`

## Result

The run completed 520 steps and performed DiLoCo merges at steps 250 and 500,
plus the final consensus merge at step 520.

| step | event | train loss |
|---:|---|---:|
| 225 | pre-merge #1 window | 7.8572 |
| 250 | merge #1 | 7.7425 |
| 275 | post-merge #1 | 7.8298 |
| 300 | post-merge #1 | 7.5306 |
| 475 | pre-merge #2 window | 7.3762 |
| 500 | merge #2 | 7.4374 |

Observed:

- Loss decreased from 10.5065 at step 25 to 7.4374 at step 500.
- No loss blow-up occurred after either K=250 merge.
- Merge #1 took 3 ms; merge #2 took 18 ms.
- `DILOCO_MERGES: 3`, `DILOCO_K: 250`.
- Peak GPU memory was 332 MiB per rank for this reduced smoke geometry.

