# SF DiLoCo Full-State Merge Smoke

Task: `sf-diloco-merge`

Date: 2026-06-16 UTC

## Configuration

Both measurements used leased GPUs through `scripts/gpu_lease.sh 2` and ran:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --data data/smoke_sf_diloco/train.txt \
  --level e88_fused --dim 64 --depth 2 --n_heads 4 --n_state 16 \
  --mlp_ratio 1.0 \
  --batch_size 2 --chunk_size 32 \
  --optimizer schedulefree --lr 0.001 --warmup_steps 10 \
  --steps 750 --diloco --diloco_k 250 \
  --log_every 50 --val_every 100000 --save_every 100000 \
  --keep_checkpoints 1 --final_train_eval --final_val_batches 100
```

The reset comparison was produced by temporarily reinstating the old
`weight_sum = 0; z = p.data` tail of `diloco_merge()` after the same x averaging
path, then restoring the principled implementation before committing.

## Results

| Variant | Merge count | Step 250 | Step 300 | Step 500 | Step 550 | Step 750 | Final train CE (x) | Final train BPB (x) | Last-100 loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Principled full-state merge | 3 | 0.8886 | 0.5530 | 0.1044 | 0.0852 | 0.0659 | 0.0568 | 0.0217 | 0.9159 |
| Reset band-aid comparison | 3 | 0.8886 | 0.7134 | 0.1137 | 0.1292 | 0.0713 | 0.0614 | 0.0234 | 0.9417 |

Observations:

- Loss decreased through all three `K=250` merge windows in the principled run.
- No post-merge spike was observed. After merge #1, loss improved from `0.8886`
  at step 250 to `0.5530` at step 300. After merge #2, it improved from
  `0.1044` at step 500 to `0.0852` at step 550.
- Eval-mode ScheduleFree `x` loss was better than the reset comparison:
  `FINAL_TRAIN_CE 0.0568` vs `0.0614`.

Raw logs:

- `principled_750.log`
- `bandaid_750.log`
- `principled.log` and `bandaid.log` are earlier 500-step measurements retained
  for context; the 750-step run is the acceptance measurement above.
