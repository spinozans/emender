# diloco-native-k250 launch report

Task: `diloco-native-k250`
Launch time: 2026-06-15 17:58 UTC

## Detached run

- Wrapper: `scripts/launch_detached_run.sh`
- Name: `diloco_native_k250`
- Durable logdir: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250`
- Active PID: `3298829`
- Manifest: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250/launch_manifest.json`
- PID file: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250/run.pid`
- Run log: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250/run.log`
- Train output root: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250/train`
- Active train run dir: `/mnt/nvme1n1/erikg/diloco_sweep/native_k250/train/levelE97_100m_20260615_175813`

The wrapper manifest records `gpus_requested=4`, `leased_gpu_ids="2,3,4,5"`,
and the detached `torchrun --standalone --nproc_per_node=4` command. The process
is reparented to PID 1 with process group/session `3298829`, so it survives the
launcher shell exit.

## Recipe

The live manifest command uses:

- Data/tokenizer: `--data /home/erikg/elman/data/pile.txt --tokenizer p50k_base`
- Model: E97 emender-mlp, `--dim 1792 --depth 11 --n_heads 216 --n_state 32 --mlp_ratio 2.262336203876648 --mlp_multiple 64`
- Optimizer/LR: `--optimizer schedulefree --lr 0.0010071509461604343`; `args.json` confirms `warmup_steps=0`
- Precision/fused path: `--bf16`; log confirms `use_triton=1 -> fused split-edit Triton kernel, NO eager fallback`
- Native per-replica batch: `--batch_size 4 --chunk_size 2048 --grad_accum 1`
- DiLoCo local-SGD: `--diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0`
- PCIe/NCCL environment: `NCCL_P2P_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`
- Offline measurement: no `--heldout_curve_every`; `--val_every 999999999`

## Checkpoints

The corrected live run uses `--save_every 21600 --keep_checkpoints 12`.
Observed steady global throughput before/after the first merge is about
`30k-33k tok/s`. With native bs4 across 4 replicas, one step is
`4 * 4 * 2048 = 32768` tokens, so `21600` steps is approximately 5.9-6.6 hours.

## Live validation

Validation commands observed:

- `ps` showed PID `3298829` alive as detached `torchrun`, with four child
  `train.py` workers: `3299294`, `3299295`, `3299296`, `3299297`.
- `nvidia-smi` showed those four workers resident on leased GPUs 2-5, each using
  about 28.3 GiB.
- `run.log` confirmed `world_size=4 backend=nccl`, native `Batch size: 4`, and
  active training through step 300.
- `run.log` confirmed the first synchronization:
  `merge #1 at step 250: averaged model weights across 4 ranks in 1962 ms`.

An earlier launch in the same logdir used `save_every=15000`; it was stopped
after live throughput showed that cadence would be closer to four hours than the
requested six. The current manifest/PID/run log correspond to the corrected
`save_every=21600` run.
