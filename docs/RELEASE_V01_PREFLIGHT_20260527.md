# v0.1 Release Preflight

Date: 2026-05-27
Task: `release-v01-preflight`

This note records the private-release preflight before any model build, weight
conversion, or upload work. No model weights were uploaded, no repository was
made public, and no token values were printed or written to this repo.

## HuggingFace Auth And Private Write Check

Commands run:

- `hf auth whoami`
- HuggingFace Hub API `create_repo(..., private=True, exist_ok=True)` and
  `upload_file(...)` for a tiny text marker in a private scratch model repo.

Results:

- Authenticated HF account: `erikgarrison`
- Visible org from the installed session: `poietic-pbc`
- Private scratch model repo used for the write probe:
  `poietic-pbc/release-v01-preflight-agent-395-private-check`
- Repo privacy after the write probe: `private=True`
- Private write-check commit SHA: `791877baee7cba5ea5e27bdabbf4b3d171e7bab3`

The scratch upload was a small text marker only. It was not a checkpoint,
`safetensors` file, HF cache, Docker layer, generated PDF, or token-bearing
artifact. No public repo was created and no repo visibility was changed.

## Candidate Checkpoint Inventory

The active paper-model training logs live under `/tmp/pile_convergence_*`.
The latest checkpoint files below are candidate inputs for the downstream pinning
task, not a release decision by themselves. Training was still advancing during
this preflight, so `release-v01-racer-checkpoint-pin` should freeze exact paths
and hashes before conversion.

| Model | Latest checkpoint candidate | Step / loss from filename | Size | Timestamp UTC | Current log candidate |
|---|---|---:|---:|---|---|
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1278000_loss_2.6420.pt` | 1,278,000 / 2.6420 | 7,639,217,707 bytes (7.12 GB) | 2026-05-27 19:13:08 | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log` |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt` | 1,686,000 / 2.6105 | 8,114,430,987 bytes (7.56 GB) | 2026-05-27 19:59:32 | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log` |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt` | 1,212,000 / 2.6870 | 7,842,766,221 bytes (7.31 GB) | 2026-05-27 19:51:20 | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log` |

Current log tails checked during preflight:

- E88: step 1,280,950, loss 2.6403, elapsed 380.648 h,
  timestamp 2026-05-27T20:17:20+00:00; log size 2,460,585 bytes.
- GDN: step 1,687,050, loss 2.6472, elapsed 380.649 h,
  timestamp 2026-05-27T20:17:22+00:00; log size 3,187,482 bytes.
- M2RNN-CMA: step 1,213,250, loss 2.6615, elapsed 386.456 h,
  timestamp 2026-05-27T20:17:36+00:00; log size 2,594,218 bytes.

Relevant shared run metadata:

- Dataset: `/home/erikg/elman/data/pile.txt`
- Tokenizer: `p50k_base`
- Context/chunk size: 2048
- Target training steps: 10,000,000
- Save interval: every 3,000 steps
- Log interval: every 50 steps
- Checkpoint retention: 96
- Optimizer: `schedulefree`
- BF16: enabled
- Seed: 42

Per-model config metadata from checkpoint `args.json`:

| Model | Level | Params label | embed_dim | dim | depth | n_heads | n_state | Batch | LR | Triton | Resume source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E88 / NDM | `E88` | `1270M` | 1024 | 1664 | 12 | 370 | 32 | 5 | 0.000867767847776187 | yes | `/tmp/pile_convergence_3arch/ctx2k/e88_repair_from231k_ckpt/levelE88_1270M_20260511_172925/latest.pt` |
| GDN | `fla-gdn` | `1270M` | 1024 | 2688 | 21 | 44 | 64 | 4 | 0.002871 | no | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_ckpt/levelfla-gdn_1270M_20260507_180327/latest.pt` |
| M2RNN-CMA | `m2rnn` | `1270M` | 1024 | 1920 | 21 | 370 | 16 | 5 | 0.0006020919750502334 | no | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_ckpt/levelm2rnn_1270M_20260509_144653/latest.pt` |

Run directory sizes:

- E88 checkpoint directory: 43 GB
- GDN checkpoint directory: 46 GB
- M2RNN-CMA checkpoint directory: 44 GB

These directories contain large generated artifacts and must not be staged or
committed.

## Resource Check

Disk:

- `df -h / /home/erikg /tmp /home/erikg/elman`: 14 TB filesystem, 11 TB used,
  2.9 TB available, 79% full.
- This is enough for three one-for-one safetensors conversions plus temporary
  staging, provided downstream tasks avoid copying redundant checkpoint trees.

GPU:

- `nvidia-smi`: 8x NVIDIA RTX 6000 Ada Generation, 49,140 MiB each.
- Driver: 570.172.08; CUDA: 12.8.
- GPUs 0, 1, 2, 3, 6, and 7 were busy with Python training jobs at the check.
- GPUs 4 and 5 were effectively idle at the check, each showing 2 MiB memory
  use and 0% utilization.

Docker:

- `docker version`: client/server 27.2.0.
- `docker info`: daemon reachable, storage driver `overlay2`, Docker root
  `/var/lib/docker`, runtimes include `nvidia`, default runtime `runc`.
- `docker system df`: images 43.93 GB, stopped containers 1.851 GB, build cache
  15.36 GB.

Python/runtime:

- `torch` installed: 2.9.1+cu128.
- `torch.cuda.is_available()`: true.
- CUDA device count from PyTorch: 8.
- `safetensors`, `transformers`, and `huggingface_hub` are installed.

Blockers:

- HF auth/write: no blocker found.
- Private repo creation/update: no blocker found for private model repos under
  `poietic-pbc`.
- Safetensors conversion: no disk/package blocker found. Downstream still needs
  checkpoint pinning, conversion code, and output hashes.
- Docker image build: no Docker daemon or disk blocker found. Downstream should
  avoid committing Docker layers or generated images.
- CPU smoke: no package or RAM blocker identified from this preflight; actual
  model-load behavior still belongs to the smoke task.
- GPU smoke: possible scheduling blocker only. Two GPUs were idle at the check,
  but six GPUs were running active training, so smoke tasks should explicitly
  select idle devices or wait for training allocation.

## Racer Status

Recent workgraph task `v22-racer-refresh` rebuilt the 1.27B racer figure from:

- `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log`
- `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log`
- `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log`

That refresh recorded data through 2026-05-27T04:26 UTC with endpoints:

- E88: step 1,237,400, 451.2 h, 0.9785 BPB.
- GDN: step 1,631,150, 456.2 h, 0.9747 BPB.
- M2RNN-CMA: step 1,168,200, 417.4 h, 0.9933 BPB.

By this preflight at roughly 2026-05-27T20:17 UTC, those same logs had advanced
to E88 step 1,280,950, GDN step 1,687,050, and M2RNN-CMA step 1,213,250. A
figure refresh is likely needed before final paper/release sync if the release
is expected to reflect the latest racer state rather than the 04:26 UTC snapshot.

## Proceed / Block Decision

Proceed with the immediate downstream tasks:

- `release-v01-racer-checkpoint-pin`
- `release-v01-emender-repo-dry-run`

Keep build/upload tasks effectively blocked until the downstream pinning task
freezes exact checkpoint paths, records hashes, and makes an explicit racer
refresh decision. Private HF staging can proceed only after local conversion and
CPU/GPU smoke tasks pass, and HF repos must remain private until explicit user
approval to change visibility.

## Artifact Hygiene

Checked before writing this note:

- No token values were printed by the HF commands used here.
- No token-bearing files were read into this repo, staged, or committed.
- No checkpoints, safetensors, HF cache files, Docker layers, generated PDFs, or
  other large generated artifacts were staged for this task.
- The only intended repository artifact is this Markdown preflight note.
