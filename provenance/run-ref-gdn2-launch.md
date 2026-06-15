# run-ref-gdn2 launch evidence

Task: `run-ref-gdn2`
Date: 2026-06-15 UTC

## Durable output

- Directory: `/mnt/nvme1n1/erikg/ref_gdn2_mlp`
- PID file: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.pid`
- Launch manifest: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/launch_manifest.json`
- Training log: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.log`
- Train args: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627/args.json`

## Launch state

- Detached training PID: `584124`
- Parent/session: `PPID=1`, `SID=584124`
- Broker-leased GPU: `1`
- Requested GPUs: `1`
- Process status at validation: alive, GPU 1 at ~98-100% SM utilization.
- `nvidia-smi --query-compute-apps` showed PID `584124` on GPU bus `00000000:25:00.0`.

The first launch attempt in this retry used the same recipe but buffered stdout
behind `python train.py`; it was stopped before handoff and moved aside to
`/mnt/nvme1n1/erikg/ref_gdn2_mlp.buffered_abort_*`. The live source-path run was
restarted cleanly with `python -u train.py` so downstream monitoring can read
step lines from the durable log.

## Recipe checks

- Arm: `gdn2-mlp`
- Geometry: `dim=2176`, `n_heads=30`, `depth=12`, `expansion=1`, `gdn2_mlp_ratio=3.258732449079677`, `batch_size=4`
- Data: `/home/erikg/elman/data/pile.txt`
- Tokenizer: `p50k_base`
- Optimizer: `schedulefree`
- LR: `0.000474`
- Warmup: `0`
- Cosine/decay: absent from the launched command; train log reports schedule-free AdamW, which has no cosine scheduler path.
- Precision: `--bf16`
- Fused path: external GDN-2 FLA chunked kernel, no eager fallback.
- Checkpoint cadence: `--save_every 25000`, `--keep_checkpoints 12`
- Target: `243000` steps, which is approximately 1.99B tokens at `4 * (2048 + 1)` tokens per step.

The launch command contains none of:

- `--heldout_tensor`
- `--heldout_curve_every`
- `--final_heldout_eval`

`args.json` confirms `final_heldout_eval=false`, `heldout_tensor=null`, and
`heldout_curve_every=0`. The train log has no `heldout_curve` output.

## Log evidence

The durable log contains:

```text
gpu_lease: granted GPUs: 1 (pid 584124)
Using device: cuda
[fused-guard] rank 0/1: level=gdn2-mlp bf16=True GDN2_PATH=/home/erikg/GatedDeltaNet-2 -> FLA chunked GDN-2 fused kernel, NO eager fallback
Output directory: /mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627
Tokenizer: p50k_base, vocab_size=50281
Model: Level gdn2-mlp, 1,286,713,448 parameters
Using schedule-free AdamW (lr=0.000474, warmup_steps=0)
Starting training from step 0...
Batch size: 4, Chunk size: 2048
```

First observed training lines:

```text
step      1 | loss 11.2430 | lr 4.74e-04 | grad 27.00 | tok/s 855 | global_tok/s 855 | elapsed_h 0.008 | time 2026-06-15T21:26:51+00:00
step      2 | loss 10.7312 | lr 4.74e-04 | grad 8.69 | tok/s 5831 | global_tok/s 5831 | elapsed_h 0.009 | time 2026-06-15T21:26:52+00:00
step      3 | loss 12.4599 | lr 4.74e-04 | grad 19.12 | tok/s 10289 | global_tok/s 10289 | elapsed_h 0.009 | time 2026-06-15T21:26:53+00:00
```

At handoff, no checkpoint file is expected yet: the required checkpoint-only
cadence is configured for the first save at step `25000`, with retention `12`.
The downstream `monitor-ref-gdn2` task should verify the first checkpoint once
that cadence is reached and continue monitoring to the endpoint.
