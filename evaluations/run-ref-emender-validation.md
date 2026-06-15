# run-ref-emender validation

Validated on 2026-06-15 after replacing the earlier contaminated held-out run.

## Live run

- Durable directory: `/mnt/nvme1n1/erikg/ref_emender_mlp`
- PID: `522358`
- Process state: detached, `PPID=1`, `PGID=522358`, `SID=522358`
- Broker lease: GPU `0`, lease owner PID `522358`
- GPU isolation: `nvidia-smi` showed exactly one compute process on GPU 0:
  `522358, GPU-5265d71b-3b75-00e3-d129-2ed12b9be786, 31680 MiB, python`
- First training line:
  `step    100 | loss 7.5712 | lr 1.01e-03 | grad 1.54 | tok/s 7785 | global_tok/s 7785 | elapsed_h 0.036 | time 2026-06-15T21:19:52+00:00`

## Command checks

The live `launch_manifest.json` command is:

```text
python train.py --data /home/erikg/elman/data/pile.txt --tokenizer p50k_base --level E97 --dim 1792 --n_heads 216 --n_state 32 --depth 11 --expansion 1.0 --use_gate 1 --gate_activation silu --mlp_ratio 2.2623 --mlp_multiple 64 --use_triton 1 --optimizer schedulefree --lr 0.001007 --bf16 --batch_size 4 --chunk_size 2048 --output /mnt/nvme1n1/erikg/ref_emender_mlp/runs --seed 42 --steps 244141 --save_every 21500 --keep_checkpoints 12 --log_every 100
```

- Uses `schedulefree` with `--lr 0.001007`.
- No `--warmup_steps`, cosine, decay, or validation flags are present.
- Uses fused E97 bf16 Triton path; `run.log` contains `NO eager fallback`.
- Uses native single-GPU `--batch_size 4`, `--chunk_size 2048`.
- Uses Pile data at `/home/erikg/elman/data/pile.txt` and tokenizer `p50k_base`.
- Checkpoint cadence is `--save_every 21500`, approximately 6 hours at the observed throughput, with `--keep_checkpoints 12`.

## Held-out exclusion

The clean run command contains none of:

- `--heldout_tensor`
- `--heldout_curve_every`
- `--final_heldout_eval`

The active `run.log` / `train.log` stream was also checked for `heldout_curve`, `Held-out curve`, and the forbidden held-out flags; no matches were present.

## Notes

The previous live run was archived under `/mnt/nvme1n1/erikg/ref_emender_mlp/archive_contaminated_20260615T211546Z` because it included inline held-out evaluation. A first direct clean attempt without the E97 builder shape defaults OOMed before the first step and was archived under `/mnt/nvme1n1/erikg/ref_emender_mlp/archive_failed_oom_20260615T211740Z`. The current live run is the clean checkpoint-only relaunch.
