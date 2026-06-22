# Frontier Post-Cache Debug - 2026-06-21

Task: `stage-p50k-cache`

## Shared p50k Cache

The shared Frontier-readable tiktoken cache was staged at:

```text
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
```

Login-node staging evidence:

| File | Evidence |
| --- | --- |
| tiktoken hash file | `/lustre/orion/bif148/proj-shared/tiktoken_cache/ec7223a39ce59f226a68acc30dc1af2788490e15`, 836,186 bytes, mode `664`, group `bif148` |
| named file | `/lustre/orion/bif148/proj-shared/tiktoken_cache/p50k_base.tiktoken` symlink to the hash file |
| tokenizer | `p50k_base`, `n_vocab=50281` |

Compute-node verification was added to `scripts/frontier/debug_smoke_one_node.slurm`
and ran before kernel/training steps. Both target variants recorded:

```text
"hash_path_readable": true
"named_path_readable": true
"p50k_base_n_vocab": 50281
```

Observed compute-node cache probes:

| Variant | Job | Compute node | Evidence |
| --- | ---: | --- | --- |
| `e97-MLP` | `4881258` | `frontier08338` | `logs/frontier/debug/emender-smoke-4881258.out`; `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/artifacts/env.txt` |
| `gdn2-MLP` | `4881374` | `frontier08198` | `logs/frontier/debug/emender-smoke-4881374.out`; `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/artifacts/env.txt` |

The previous tokenizer download blocker was removed: both training logs printed
`Tokenizer: p50k_base, vocab_size=50281` from compute-node context.

## Script and Code Fixes

Two post-cache launch blockers were observed and fixed during this task:

1. `LadderLM.__init__()` did not accept the parsed `use_chunked_e97` and
   `e97_chunk_size` options that `train.py` always passed. This stopped the
   first post-cache e97 attempt (`4881174`) after tokenizer resolution but
   before training loss. `ndm/models/ladder_lm.py` now accepts and forwards
   those options to the per-layer constructor.
2. The debug smoke script launched the eight training ranks without exporting
   DDP rank variables. The first fixed e97 run (`4881258`) reached 50 finite
   steps and wrote a checkpoint, but all ranks behaved as rank 0 and raced on
   `latest.pt`. The script now exports `RANK=$SLURM_PROCID`,
   `WORLD_SIZE=$SLURM_NTASKS`, and `LOCAL_RANK=0` under
   `--gpus-per-task=1 --gpu-bind=closest`, where each task sees one rank-local
   GPU as device 0.

## e97-MLP Result

Primary evidence job: `4881258`

| Field | Value |
| --- | --- |
| State | `FAILED`, exit `143:0`, after producing training/checkpoint evidence |
| Queue | `batch` partition, `debug` QOS |
| Nodes / walltime requested | 1 node / 00:30:00 |
| Elapsed / consumed node-hours | 00:14:27 / 0.240833 |
| First finite loss | `step 1 | loss 11.1347 | tok/s 2972 | global_tok/s 2972` |
| Last observed metric | `step 50 | loss 6.0547 | tok/s 3202 | global_tok/s 3202` |
| Memory | `PEAK_MEMORY_MB: 13115`, `RESERVED_MEMORY_MB: 15504` |
| Fused/no-eager guard | `[fused-guard] rank 0/1: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback` |
| Runtime path | `[e97-runtime] backend=hip path=e88-sequential-split-edit-triton use_triton=True use_chunked_e97=False e97_chunk_size=32 linear_state=False raw_write=False use_split_edit=True log_decay=True` |
| Checkpoint artifact | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/train/levelE97_100m_20260621_110302/checkpoint_step_000050_loss_7.7509.pt` |
| Failure classification | Observed script launch bug: every rank acted as rank 0 and raced to create `latest.pt`; the debug script was fixed afterward. This is not tokenizer failure and not inherited no-go language. |

Artifacts:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/artifacts/manifest.json`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/artifacts/env.txt`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/logs/kernel_smoke.log`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/logs/train.log`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/summaries/summary.md`
- `logs/frontier/debug/emender-smoke-4881258.out`
- `logs/frontier/debug/emender-smoke-4881258.err`

An earlier e97 post-cache attempt, `4881174`, verified the cache/tokenizer path
and then stopped on the observed `LadderLM.__init__()` interface mismatch. It
consumed 00:02:02, or 0.033889 node-hours.

## gdn2-MLP Result

Primary evidence job: `4881374`

| Field | Value |
| --- | --- |
| State | `COMPLETED`, exit `0:0` |
| Queue | `batch` partition, `debug` QOS |
| Nodes / walltime requested | 1 node / 00:30:00 |
| Elapsed / consumed node-hours | 00:05:32 / 0.092222 |
| First finite loss | `step 1 | loss 11.2562 | tok/s 3864 | global_tok/s 30911` |
| Last observed metric | `step 262 | loss 0.0363 | tok/s 4359 | global_tok/s 34870` |
| Memory | `PEAK_MEMORY_MB: 16847`, `RESERVED_MEMORY_MB: 17686` |
| Fused/no-eager guard | `[fused-guard] rank 3/8: level=gdn2-mlp bf16=True backend=hip torch.version.hip=6.4.43482-0f2d60242 GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2 chunk_ops=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py -> FLA chunked GDN-2 fused kernel import path, NO eager fallback` |
| Checkpoint artifact | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/checkpoint_step_000262_loss_0.0766.pt` |
| Outcome classification | Observed successful one-node post-cache training smoke: cache readable, GDN2 preflight passed, fused GDN2 path used, finite loss and throughput recorded, memory recorded, checkpoint written. |

Artifacts:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/artifacts/manifest.json`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/artifacts/env.txt`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/logs/kernel_smoke.log`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/logs/train.log`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/summaries/summary.md`
- `logs/frontier/debug/emender-smoke-4881374.out`
- `logs/frontier/debug/emender-smoke-4881374.err`

An earlier gdn2 post-cache attempt, `4881346`, verified cache and fused GDN2
guard but failed before first training loss because `LOCAL_RANK` was mapped to
`SLURM_LOCALID` while each task saw only one local GPU. It consumed 00:02:12,
or 0.036667 node-hours. The debug script fix described above resolved this.

## Accounting

Previous ledger remaining after `DEBUG-MATRIX-20260621-RETRY`:
19,999.802776 node-hours. Reserve held remains 4,928 node-hours.

| Group | Jobs | Requested node-hours | Consumed node-hours | Remaining before | Remaining after |
| --- | --- | ---: | ---: | ---: | ---: |
| e97 post-cache attempts | `4881174`, `4881258` | 1.000000 | 0.274722 | 19,999.802776 | 19,999.528054 |
| gdn2 post-cache attempts | `4881346`, `4881374` | 1.000000 | 0.128889 | 19,999.528054 | 19,999.399165 |

## Interpretation

Observed:

- The shared tiktoken cache is readable from Frontier compute-node job context.
- `e97-MLP` reached finite training losses, throughput lines, peak memory, and a
  checkpoint after the cache and constructor fixes.
- `gdn2-MLP` completed a one-node post-cache debug run with finite losses,
  throughput, peak memory, fused/no-eager guard, and checkpoint artifact.
- The debug smoke script now has explicit compute-node cache verification and
  DDP rank mapping compatible with Frontier `--gpus-per-task=1` binding.

Risks and next actions:

- A clean post-rank-fix e97 completion was attempted but blocked by the debug
  QOS user job limit after another worktree submitted `4881380`. The available
  e97 evidence is sufficient for first-loss/throughput/memory/cache evidence,
  but the final scheduler state of `4881258` remains failed because it predates
  the script rank-mapping fix.
- Before a larger e97 canary or extended launch, rerun `e97-MLP` once with the
  fixed script to confirm clean exit and checkpoint symlink behavior.
- Do not treat the earlier tokenizer no-go as current: the tokenizer blocker is
  resolved. Remaining e97 caution is based on the observed checkpoint-race
  failure in `4881258`, not inherited no-go language.
