# Frontier DiLoCo Scaleout Readiness Plan - 2026-06-21

Task: `frontier-diloco-scaleout-readiness`

## Recommendation

Decision: **NO-GO for 16/32/64-node scaleout submission today; CANARY-ONLY
after a debug retry reaches first training metrics**.

The current debug evidence is useful but not sufficient for an expensive
scaleout launch. The retry matrix proved that the staged Frontier runtime can
submit jobs, see MI250X GPUs, run the `e97-MLP` fused split-edit kernel smoke,
and run the `gdn2-MLP` external fused-path bf16 forward/backward preflight. It
did not measure training throughput, training loss, validation loss, checkpoint
behavior, DiLoCo merges, or multi-node communication overhead because
`e97-MLP` and `gdn2-MLP` stopped on compute-node `tiktoken` cache downloads
before the first training metric. `e97-linear-MLP` failed the chunked-E97 ROCm
parity/finiteness smoke and should not be in the first scaleout path until that
kernel is fixed or the variant is deliberately disabled.

The next launchable scaleout action should be an **8-node, 4-hour canary only**
if all debug gates below pass and a human records approval in WG. Treat 16, 32,
and 64 nodes as planned gates, not current submissions.

## Evidence Separation

Observed debug evidence:

- Frontier accepted and ran the one-node debug retry jobs sequentially under
  debug QOS.
- The staged runtime imported ROCm PyTorch, Triton, ScheduleFree, `tiktoken`,
  NumPy, pytest, and the GDN2 dependency path.
- `e97-MLP` passed
  `tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference` on
  one MI250X GCD, then failed before training metrics because `tiktoken` tried
  to download `p50k_base.tiktoken` from compute nodes.
- `gdn2-MLP` passed an external GDN2 bf16 forward/backward preflight with
  finite loss and gradients, then hit the same tokenizer download failure
  before training metrics.
- `e97-linear-MLP` failed the chunked-E97 ROCm parity/finiteness smoke and did
  not launch training.
- One-rank `srun` GPU binding showed rank-local `ROCR_VISIBLE_DEVICES=0` and
  one visible MI250X under `--gpus-per-task=1 --gpu-bind=closest`.

Hypotheses to test, not assumptions:

- **GPU-island hypothesis:** one Frontier node can act as one island of eight
  GCDs, with per-step DDP inside the island and DiLoCo/plain averaging between
  islands, without reproducing the observed local DDP inefficiency or causing
  RCCL/Slingshot instability.
- **Communication hypothesis:** `--diloco --diloco_island_size 8` will reduce
  cross-node communication to periodic merge boundaries enough that global
  tokens/sec scales better than flat cross-node DDP.
- **Optimizer hypothesis:** the repo's plain-average DiLoCo recipe
  (`outer_beta=0.0`, `outer_lr=1.0`) remains stable on Frontier at 8+ islands.
  Prior local results support this recipe only at much smaller island counts;
  they do not prove 64-node behavior.
- **Capacity/loss hypothesis:** short commapile loss curves are sufficient to
  reject broken launches, NaNs, tokenizer/data errors, or severe recipe
  collapse, but they are not sufficient to decide long-horizon architecture
  quality.

Prior-agent interpretations to carry as priors, not launch criteria:

- Local reports favor plain-average DiLoCo over outer momentum and warn that
  `outer_beta=0.9` diverged in prior experiments.
- The handoff reports DiLoCo matched-token parity only at small local island
  counts and explicitly leaves the Frontier large-island regime untested.
- The handoff and orientation brief warn that prior no-go statements about
  E97/GDN2 are interpretations until Frontier-specific kernel, loss, and
  throughput evidence exists.

## Required Debug Gates Before Any Scaleout

Do not submit 8+ node jobs until these are recorded as artifacts or WG logs:

1. Shared tokenizer cache is staged and readable on compute nodes:
   `TIKTOKEN_CACHE_DIR` or equivalent contains `p50k_base.tiktoken`, and a
   one-node debug job reaches at least one training log line.
2. `e97-MLP` or `gdn2-MLP` records finite training loss and global tokens/sec
   under the same data, tokenizer, batch, context, optimizer, and fused-path
   settings intended for scaleout.
3. The candidate arm has no silent eager fallback. The log must show the fused
   E97 or GDN2 runtime guard expected for that variant.
4. One-node baseline captures startup time, compile/autotune time, peak memory,
   first 10-50 optimizer steps, checkpoint write, and validation if `VAL_DATA`
   is enabled.
5. A one-node island-vs-flat control is run if the training code changes launch
   mode: flat 8-rank DDP and `--diloco --diloco_island_size 8` must both reach
   a training metric, or the missing control must be explained.
6. The allocation ledger is updated with planned requested node-hours and
   expected remaining allocation before submission.
7. A human approval record exists in WG before submitting any 8+ node canary or
   extended/production-shaped job.

Tiny 1/2-node scale tests are intentionally avoided. Use a 1-node debug retry
only to isolate tokenizer/runtime/kernel issues. Use 8 nodes as the smallest
distributed canary because the experiment question is multi-island behavior.

## Scaleout Experiment Matrix

The matrix uses the ledger remaining after the retry debug row:
`19,999.802776` node-hours, with `4,928` node-hours held in reserve. All
requested node-hours below must be debited before submission and reconciled
with scheduler-consumed hours afterward.

| ID | Status | Nodes | GCDs | Islands | Walltime | Requested node-hours | Allocation after requested debit | Queue | Prerequisites | Stopping rules |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `DILOCO-8x4H-CANARY` | Candidate after debug gates | 8 | 64 | 8 one-node islands | 04:00:00 | 32 | 19,967.802776 | regular or eligible short queue, not debug production | Debug gates 1-7; candidate arm selected; `scripts/frontier/diloco_scaleout_readiness.sbatch` reviewed; human approval recorded. | Stop and do not proceed if launch fails before first training step, any rank has nonfinite loss/grad, no DiLoCo merge occurs by K=250, global tok/s is below 4x one-node baseline after warmup, RCCL/NCCL errors occur, checkpoint/manifest missing, or requested ledger row is absent. |
| `DILOCO-8x8H-PILOT` | Deferred until canary passes | 8 | 64 | 8 one-node islands | 08:00:00 | 64 | 19,903.802776 if run after canary | regular | Canary completes with finite loss, at least one successful merge, restartable checkpoint, and interpretable throughput. | Stop ladder if loss increases monotonically for 3 log windows after a merge, merge sync is >20% of walltime, validation is nonfinite, or cost/throughput is worse than one-node extrapolation by >2x without explained compile/startup cause. |
| `DILOCO-16x12H-PILOT` | Deferred | 16 | 128 | 16 one-node islands | 12:00:00 | 192 | 19,711.802776 if run after canary+8h pilot | regular | 8-node pilot passes; ledger updated; human approval refreshed; no unresolved RCCL/DiLoCo merge issue. | Stop before 32 nodes if launch stability <100%, fewer than 2 clean merge intervals complete, global tok/s scaling <8x one-node baseline after startup, merge communication >25% of step-window time, loss/grad sanity fails, or checkpoint/restart evidence is absent. |
| `DILOCO-32x12H-PILOT` | Deferred | 32 | 256 | 32 one-node islands | 12:00:00 | 384 | 19,327.802776 if run after prior pilots | regular/extended only if policy requires | 16-node pilot passes and shows stable optimizer/loss and usable cost/throughput; explicit approval; ledger has remaining/reserve values. | Stop before 64 nodes if node/rank failures are not isolated cleanly, DiLoCo merge latency dominates (>30% of walltime or >2x 16-node merge fraction), loss shock after merge does not recover within one K-window, validation/capability signals regress severely, or budget headroom/reserve is threatened. |
| `DILOCO-64x24H-CANDIDATE` | Blocked pending evidence and approval | 64 | 512 | 64 one-node islands | 24:00:00 | 1,536 | 17,791.802776 if run after all prior rows | extended-production | 8/16/32-node scaleout evidence complete; production launch package reviewed; ledger candidate row updated; explicit human approval artifact recorded. | Abort/postmortem if startup does not reach training, any systemic RCCL/NCCL failure appears, nonfinite loss/grad occurs, checkpointing cannot keep up, merge overhead invalidates cost model, loss/capacity metrics fall outside approved thresholds, or consumed spend plus planned reruns would invade reserve without approval. |

The 64-node option is included for capacity planning but is **not launchable**
from current evidence. It becomes a candidate only after the 8/16/32-node ladder
answers launch stability, communication, optimizer/loss, checkpoint, and cost
questions.

## Metrics and Decision Thresholds

Launch stability:

- Scheduler acceptance/rejection reason, startup time, first training-step time,
  rank mapping, rank count, node list, and exit codes.
- Count of ranks reaching warmup, first optimizer step, first log, first
  DiLoCo merge, validation, and checkpoint.
- Failure classes: scheduler/QOS, module/runtime import, tokenizer/data, fused
  kernel, RCCL/NCCL, OOM, nonfinite loss, nonfinite grad, checkpoint I/O,
  timeout.

Throughput scaling:

- Per-rank tokens/sec and global tokens/sec after compile warmup.
- Startup/compile time separated from steady-state time.
- Scaling efficiency against one-node baseline:
  `global_tok_s(N) / (N * one_node_global_tok_s)`.
- Minimum canary bar: the 8-node canary must beat 4x the one-node baseline after
  warmup to justify larger jobs. This is a readiness bar, not a science result.

Optimizer and loss sanity:

- Train loss per log window, last-100 average loss, validation loss/BPB if
  enabled, grad norm, LR, DiLoCo merge count, and merge timing.
- Post-merge loss shock and recovery within each K-window.
- Nonfinite guards and outlier rank losses.
- For ScheduleFree runs, record whether the merge used the expected y/eval
  semantics and whether optimizer clocks/state were preserved.

Communication overhead:

- DiLoCo merge walltime per merge, total sync seconds, average sync seconds, and
  sync fraction of total walltime.
- Flat DDP control where practical: DDP step time/all-reduce symptoms versus
  island DiLoCo step time.
- RCCL/NCCL environment, backend, `MASTER_ADDR`, rank mapping, and any
  `aws-ofi-rccl` or Slingshot-specific settings in the job artifact.

Loss/capacity signals:

- Short commapile validation/BPB at fixed token intervals.
- Small fixed capability panel only after launch/throughput is stable; do not
  use it to justify spending if basic loss or throughput is broken.
- Keep architecture claims separate from launch readiness. A scaleout canary can
  pass systems readiness without proving E97 or GDN2 long-horizon superiority.

Cost:

- Requested node-hours before submission.
- Scheduler-consumed node-hours after completion.
- Allocation remaining before/after and reserve held.
- Cost per billion tokens from measured global tokens/sec:
  `node_hours / tokens_processed_billion`.

## Candidate Launch Shape

Primary candidate after debug gates:

```text
model arm:        gdn2-MLP first if tokenizer fix reaches training metrics;
                  e97-MLP may replace it only after equivalent first-loss and
                  throughput evidence; e97-linear-MLP deferred until ROCm
                  chunked parity/finiteness is fixed
parallelism:      --diloco --diloco_island_size 8
island shape:     1 node = 8 GCDs
outer recipe:     --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0
batch/context:    --batch_size 1 --chunk_size 2048
optimizer:        schedulefree for merge-path validation; AdamW/cosine is a
                  separate long-horizon recipe question
data:             staged commapile mainmix smoke or a documented shard derived
                  from /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
tokenizer:        p50k_base with shared pre-populated cache
```

The GPU-island pattern is the test condition, not the conclusion. The first
8-node job should record enough data to decide whether one-node islands
actually avoid the DDP communication pathologies observed elsewhere.

## Job Template and Commands

Reusable template:

```text
scripts/frontier/diloco_scaleout_readiness.sbatch
```

Pre-submit checks from a Frontier login node:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
test -r "$MEMBERWORK/emender/tiktoken/p50k_base.tiktoken"
test -r /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
wg show frontier-diloco-scaleout-readiness
wg msg read frontier-diloco-scaleout-readiness --agent "$WG_AGENT_ID"
```

8-node canary command, only after debug gates and human approval:

```bash
export HUMAN_APPROVAL_RECORD="WG message or committed approval artifact"
export SCALEOUT_VARIANT="gdn2-MLP"
export SCALEOUT_NODES=8
export SCALEOUT_WALLTIME="04:00:00"
export DATA="/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt"
export VAL_DATA="/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt"
export TIKTOKEN_CACHE_DIR="$MEMBERWORK/emender/tiktoken"
export EMENDER_CONDA_ENV="${MEMBERWORK}/emender/conda/emender-rocm711"
export GDN2_PATH="${MEMBERWORK}/emender/src/GatedDeltaNet-2"
export DILOCO_K=250
export TRAIN_MINUTES=210
sbatch -N 8 -t 04:00:00 scripts/frontier/diloco_scaleout_readiness.sbatch
```

Deferred examples, not approvals:

```bash
# 16-node pilot after 8-node evidence and approval
SCALEOUT_NODES=16 SCALEOUT_WALLTIME=12:00:00 TRAIN_MINUTES=690 \
  sbatch -N 16 -t 12:00:00 scripts/frontier/diloco_scaleout_readiness.sbatch

# 32-node pilot after 16-node evidence and approval
SCALEOUT_NODES=32 SCALEOUT_WALLTIME=12:00:00 TRAIN_MINUTES=690 \
  sbatch -N 32 -t 12:00:00 scripts/frontier/diloco_scaleout_readiness.sbatch

# 64-node production-shaped candidate after 8/16/32 evidence and explicit approval
SCALEOUT_NODES=64 SCALEOUT_WALLTIME=24:00:00 TRAIN_MINUTES=1410 \
  sbatch -N 64 -t 24:00:00 scripts/frontier/diloco_scaleout_readiness.sbatch
```

The commands above are deliberately environment-variable driven so the same
template can generate artifacts with the selected model arm, node count, and
approval record. They must not be run by an agent without recorded human
approval.

## Go/No-Go Criteria

Current state:

- **Go for 64-node:** no.
- **Go for 32-node:** no.
- **Go for 16-node:** no.
- **Go for 8-node canary:** no until the tokenizer/debug gates pass; then
  yes only with explicit human approval and a ledger row.

Move from no-go to 8-node canary when:

- A debug retry reaches finite first loss and throughput for the chosen arm.
- The tokenizer cache and data paths are proven on compute nodes.
- The one-node launch emits a checkpoint/manifest/summary with node-hour
  accounting.
- Ledger rows show the 32 node-hour requested debit and remaining allocation.
- Human approval is recorded in WG.

Move from canary to 16 nodes when:

- The 8-node canary completes at least one K=250 merge and writes a usable
  checkpoint.
- Throughput scaling, communication overhead, loss, and validation are within
  the thresholds above.
- The next 192 requested node-hours are entered in the ledger before submission.

Move from 16 to 32 nodes when:

- The 16-node pilot completes at least two clean merge windows and produces a
  restartable checkpoint.
- Communication overhead and loss shock do not grow faster than expected.
- Human approval is refreshed for the 384 requested node-hours.

Move from 32 to 64 nodes when:

- The 8/16/32-node evidence is summarized in a launch package that separates
  observations, hypotheses, risks, and prior interpretations.
- The 1,536 requested node-hours, remaining allocation, and protected reserve
  are recorded in the ledger.
- Human approval explicitly authorizes the 64-node extended-production job.
