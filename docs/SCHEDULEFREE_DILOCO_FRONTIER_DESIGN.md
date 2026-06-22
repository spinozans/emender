# ScheduleFree-DiLoCo Frontier Design

## Goal

Train E88/NDM at scale on Frontier-style allocations without forcing the
optimizer into an enormous global-batch regime. The concrete near-term target
is sustained multi-node training, for example:

- 16 nodes for 24 hours as the first realistic production shape
- 64 nodes for 24 hours once the one-node island design is trusted
- larger bursts only after the communication and optimizer behavior are clear

Short queued jobs may become useful later, but they are not the central design
point. The first question is whether hierarchical local training scales
efficiently for normal multi-node allocations.

The core problem is that plain synchronous DDP across all ranks makes the
global batch enormous. At ctx2k, even per-GPU batch size 1 gives:

```text
16 nodes  * 8 GCDs/node * 2048 tokens = ~262K tokens/update
64 nodes  * 8 GCDs/node * 2048 tokens = ~1.05M tokens/update
500 nodes * 8 GCDs/node * 2048 tokens = ~8.19M tokens/update
```

That is attractive for raw tokens/sec but risky for learning. The current
single-GPU E88 run is effectively around 10K tokens/update (`bs=5`, ctx2k).
We want high parallel throughput without collapsing update density.

## Local Smoke Results

On 2026-05-13, a 4-GPU local smoke used one process per GPU on GPUs
`0,5,6,7` with a 74.5M-parameter E88/NDM, ctx512, per-rank batch size 4, and
ScheduleFree AdamW.

The strongest result was not momentum DiLoCo. It was plain periodic model
averaging:

```text
mode:               local_sgd
sync interval H:    100 local steps
sync payload:       model weights only
steps:              50,000
tokens:             409.6M
elapsed:            1824s
effective tok/s:    224.5K
last 1K loss:       4.3737
last 1K bpb:        1.6531
final drift:        0.0199
```

Short 1K-step and 5K-step comparisons:

```text
H=50  + optimizer-state sync:  lower drift, slower, good learning
H=100 + optimizer-state sync:  best short-run loss
H=100 + model-only sync:       tied by 5K steps, faster and simpler
H=200 + optimizer-state sync:  faster, visibly more drift
H=200 + model-only sync:       worse drift/loss
DiLoCo beta=0.9:               poor short-run learning
```

So the current Frontier candidate should be described as:

```text
ScheduleFree local-SGD with periodic model averaging
```

Outer-momentum DiLoCo remains worth studying, but it should not be the first
production path until we find a stable beta/outer-lr regime.

### Scale-Matched E88 Smoke

The first scale-matched smoke used the production 1.27B E88 geometry:

```text
level:             E88 / NDM
params:            1,273,191,856
dim/depth:         1664 / 12
heads/state:       370 / 32
context:           2048
per-rank batch:    1
optimizer:         ScheduleFree AdamW
Triton:            enabled
GPUs:              0,5,6,7
```

At this scale, full model averaging across four local GPUs costs about 9s per
sync. The `H` sweep over 1000 local steps showed:

```text
H=100:  8.196M tokens, 436s, 18.8K tok/s, final drift 0.123
H=250:  8.196M tokens, 384s, 21.3K tok/s, final drift 0.275
H=500:  8.196M tokens, 368s, 22.3K tok/s, final drift 0.467
```

Short-window loss is noisy this early, but the qualitative behavior is clear:

- `H=100` is the safest setting.
- `H=250` gives most of the throughput gain with tolerable recovery.
- `H=500` is faster but produces a large post-sync loss shock.

Current scale-matched recommendation: use `H=250` as the first serious
candidate and keep `H=100` as the conservative fallback. Do not push beyond
`H=500` until longer runs show recovery without compounding drift.

## Recommended Training Shape

Use hierarchical ScheduleFree-DiLoCo:

```text
inside each island:
  normal synchronous training, ideally 1 node = 8 GCDs
  DDP every local optimizer step
  ScheduleFree AdamW as the inner optimizer

between islands:
  periodically average model weights or model deltas
  no per-step all-reduce across the whole allocation
```

For example:

```text
16-node job:
  16 islands
  8 ranks/island
  local batch: per-GPU bs=1 or 2
  inter-island averaging every K=250-1000 local steps

64-node job:
  64 islands
  8 ranks/island
  local batch: per-GPU bs=1 or 2
  inter-island averaging every K=500-2000 local steps

500-node job:
  500 islands
  8 ranks/island
  local batch: per-GPU bs=1
  only after 16/64-node behavior is understood
```

This gives each island a sane local batch:

```text
1 node island, per-GPU bs=1:
  8 * 2048 = 16,384 tokens/local update

current single-GPU E88:
  5 * 2048 = 10,240 tokens/local update
```

So the local optimizer sees a batch scale close to the current healthy run,
while the full allocation contributes many independent local updates per
DiLoCo round.

## ScheduleFree Interaction

ScheduleFree AdamW maintains internal train/eval weight behavior. In this repo,
checkpoints are currently saved after `optimizer.eval()`, so saved model
weights are the averaged/eval weights.

Initial rule:

```text
inner optimizer:
  ScheduleFree AdamW inside each island

merge target:
  model weights after optimizer.eval()

outer optimizer:
  simple DiLoCo momentum over model deltas

do not average:
  ScheduleFree optimizer internals, initially
```

This is conservative. Each island starts each DiLoCo round from the same global
eval weights, trains locally with ScheduleFree, exports eval weights, and the
outer merger averages the deltas.

The reason to avoid optimizer-state merging at first is simple: ScheduleFree
state semantics are not just Adam moments. Merging those states may work, but it
adds ambiguity before we know whether model-delta averaging works. The 4-GPU
smoke supports this conservative default: model-only averaging caught up to
state-sync by 5K steps and was faster.

## Synchronous Hierarchical DiLoCo

One Frontier job contains all islands. Every island trains locally, then all
islands synchronize at a DiLoCo boundary.

Let `W_r` be the global model at round `r`.

Each island `i` starts from `W_r`, trains for `K` local steps, and produces
`W_{r,i}`.

Compute:

```text
delta_i = W_{r,i} - W_r
delta   = weighted_mean_i(delta_i)

outer_momentum = beta * outer_momentum + delta
W_{r+1} = W_r + outer_lr * outer_momentum
```

The first sweep should start with beta 0 / local-SGD behavior and only then
reintroduce outer momentum:

```text
K:          100, 250, 500, 1000
outer_lr:   1.0 first, then 0.5/1.0
outer_beta: 0.0 first, then 0.5/0.9 only if stable
local bs:   1 per GPU, then 2 if stable
```

Token-weighted averaging is preferred if islands can process different token
counts. If all islands run the same number of steps and same batch, plain mean
is equivalent.

### Benefits

- Much lower communication than global DDP.
- Avoids huge global batch.
- Keeps local optimizer dynamics close to current single-GPU/one-node regime.
- Works naturally with short large allocations.
- Gives a clean experimental axis: DDP vs DiLoCo at matched tokens and walltime.

### Costs

- Requires custom training loop or a wrapper around `train.py`.
- Outer averaging must be robust to failed/nonfinite islands.
- Some loss of exact optimizer equivalence vs DDP.
- Hyperparameters `K`, `outer_lr`, and `outer_beta` matter.

## Independent Mode

Independent mode is the filesystem/checkpoint version. It is useful when jobs
cannot communicate during the allocation or when separate queued jobs finish at
different times. It is not the first production path because it does not answer
whether sustained multi-node training scales efficiently.

Protocol:

```text
1. Publish global checkpoint W_r with base_id.
2. Launch N independent jobs from W_r.
3. Each job trains locally for K steps or T minutes.
4. Each job writes:
   - final eval model checkpoint
   - base_id
   - worker_id
   - data_rank/data_world_size
   - tokens processed
   - loss summary
   - nonfinite status
5. Merger waits for enough workers, averages deltas, writes W_{r+1}.
6. Next wave launches from W_{r+1}.
```

This is less efficient than in-job hierarchical local-SGD/DiLoCo because each
wave pays queue/startup overhead and cannot synchronize mid-job. Treat it as a
later operational mechanism, not the core scaling strategy.

## Hogwild ScheduleFree-DiLoCo

Hogwild mode means asynchronous, non-barrier merging. At Frontier scale this
should not mean many processes literally writing shared parameters. It should
mean an append-only delta queue and an asynchronous merger.

Worker behavior:

```text
while allocation active:
  read newest global checkpoint W_r
  train locally for K steps with ScheduleFree
  write delta package against base_id=r
  optionally pull newer global checkpoint and repeat
```

Merger behavior:

```text
loop:
  scan delta queue
  group deltas by base_id
  reject invalid/nonfinite/outlier deltas
  apply enough fresh deltas to produce W_{r+1}
  optionally apply stale deltas with reduced weight
```

Staleness weighting:

```text
staleness = current_round - base_round
weight = tokens_i * stale_decay ** staleness
```

Start with:

```text
stale_decay = 0.5
max_staleness = 2
```

So a delta from one round ago counts half; older than two rounds is ignored.

### Why Hogwild Is Interesting

- It can keep huge allocations busy even when workers finish unevenly.
- It tolerates node failure and stragglers.
- It turns short allocations into continuous progress.
- It is aligned with the multiprogramming hypothesis: many learners explore
  local trajectories, then the system consolidates.

### Why Hogwild Is Risky

- Stale deltas can fight the current model.
- ScheduleFree local trajectories may depend strongly on the base point.
- Bad workers must be filtered aggressively.
- It is harder to analyze than synchronous rounds.

Hogwild should be a second-stage experiment after synchronous hierarchical
DiLoCo works.

## Delta Package Format

Each worker should write a small metadata file plus a checkpoint/delta file.

Metadata:

```json
{
  "base_id": "round_000123_sha...",
  "worker_id": "job1234_rankgroup007",
  "arch": "E88",
  "params": "1270M",
  "ctx": 2048,
  "local_steps": 500,
  "tokens": 8192000,
  "optimizer": "schedulefree_adamw",
  "inner_lr": 0.000867767847776187,
  "data_rank": 7,
  "data_world_size": 500,
  "loss_last100": 2.81,
  "nonfinite": false
}
```

Tensor payload options:

1. Full model weights `W_i`.
2. Delta weights `W_i - W_base`.

Use full weights first because they are easier to inspect and recover from.
The merger can compute deltas. Once stable, switch to delta-only storage to
reduce I/O.

For merging:

- Accumulate floating tensors in FP32 on CPU or GPU.
- Preserve integer/bool buffers from the base checkpoint.
- Reject tensors with nonfinite values.
- Reject workers whose relative delta norm is an extreme outlier.
- Keep base checkpoint, accepted worker list, rejected worker list, and merge
  config in the output checkpoint metadata.

## Data Sharding

The dataset classes already support `rank` and `world_size`, but current
`train.py` does not expose explicit CLI flags for them in the production path.

Add:

```text
--data_rank
--data_world_size
```

Then pass these into `TokenizedStreamDataset`, `FastTokenizedDataset`,
`DocumentStreamDataset`, and `BatchedStreamDataset` where applicable.

For island training:

```text
data_world_size = total number of islands or total number of workers
data_rank       = island id or global worker id
```

For DDP inside a one-node island, either:

- shard by global rank, or
- let each rank in the island have a separate stream and treat the island as
  the unit that reports tokens.

The first implementation should use global rank sharding so duplicate data is
impossible.

## Required Repo Work

### Phase 1: Local Distributed Controls

Use the 4-GPU harness to establish:

- DDP/global-batch baseline
- local-SGD at `H=100/250/500/1000`
- optimizer-state sync vs model-only sync
- fixed initialization across ranks
- same-token and same-wallclock summaries

This is the current active phase.

### Phase 2: In-Job Hierarchical Local-SGD/DiLoCo

Implement real distributed launch:

- `torchrun` over all ranks
- split ranks into islands
- DDP process group per island
- global inter-island process group for merge boundaries
- local ScheduleFree optimizer per island
- periodic model-weight or model-delta all-reduce across island leaders
- broadcast merged weights to ranks in each island

This is the main Frontier path.

### Phase 3: 16-Node Frontier Pilot

Run one production-shaped job:

```text
16 nodes
8 ranks/node
16 one-node islands
ctx2k
per-GPU bs=1 initially
H=100/250 first, then H=500 if stable
model-only averaging first
```

Compare against the current single-GPU 1.27B E88 run by wallclock, tokens, and
loss trajectory.

### Phase 4: Offline / Hogwild Modes

Add:

- delta queue directory
- async merger daemon
- stale delta weighting
- worker restart from latest checkpoint
- merge audit logs

This is for operational flexibility across separate allocations or uneven
worker completion. It should wait until the synchronous island design is
trusted.

## Initial Hyperparameter Proposal

For 1.27B ctx2k E88:

```text
island_size:       8 ranks / 1 node
per_gpu_batch:     1
local tokens/update: 16K
inner optimizer:   ScheduleFree AdamW
inner lr:          current E88 LR, then sweep 0.5x/1x
K local steps:     500
outer_lr:          1.0
outer_beta:        0.0 initially (local-SGD); sweep momentum later
delta clipping:    enabled, relative norm outlier rejection
merge interval:    every K local steps
checkpoint:        every DiLoCo round + periodic local emergency checkpoint
```

For 16 nodes x 24 hours:

```text
16 islands
8 ranks/island
K=100-500
primary first production-shaped scaling test
```

For 64 nodes x 24 hours:

```text
64 islands
8 ranks/island
K=500-2000
more stable and easier for the first production run
```

## Evaluation Protocol

Every global round should produce:

- train loss summaries from workers
- held-out validation on the merged checkpoint
- bpb estimate using the same tokenizer byte/token conversion
- generation samples from fixed prompts after major rounds
- expressivity smoke tests on small fixed tasks if cheap enough

The core comparison must be:

```text
same architecture
same tokens
same wallclock where possible
continuous single-worker baseline
DDP baseline
hierarchical DiLoCo
Hogwild DiLoCo
```

## Open Questions

- Does ScheduleFree tolerate resetting local optimizer state every DiLoCo
  round, or should local optimizer state persist within each island?
- Should the outer delta be computed from ScheduleFree eval weights or train
  weights? Start with eval weights because that is what current checkpoints
  save.
- How large can `K` be before islands drift too far?
- Does E88 benefit from island diversity, or does it need tighter
  synchronization than linear models?
- Can we use many-head E88 structure for a later head-parallel implementation
  that combines with DiLoCo?

## Current Recommendation

Build this in order:

1. Add a DDP/global-batch baseline to the local distributed harness.
2. Sweep `H=250/500/1000` with model-only local-SGD at 75M and 480M.
3. Run a reduced-batch 1.27B E88 local-SGD smoke to measure real sync cost.
4. Build the offline independent merger for short-job checkpoint waves.
5. Build one-node island DDP.
6. Build multi-island hierarchical local-SGD/DiLoCo inside a single job.
7. Add Hogwild async merging only after the synchronous version is stable.

The reason is pragmatic: the 4-GPU local-SGD smoke already proves basic
periodic model sharing. The next missing control is a central baseline, followed
by larger `H` sweeps and a 1.27B communication measurement. Hierarchical mode is
the real Frontier speed path. Hogwild mode is the high-leverage operational
layer once the synchronous math is trusted.

## P1 RESULTS — SF x/z/y geometry on the outer update (task sf-diloco-p1)

This resolves the "eval weights vs train weights" open question above and fixes
a geometry bug that made nonzero outer beta look catastrophic.

**The bug.** When a nontrivial outer update (any `outer_lr != 1` OR any outer
momentum, i.e. any time `outer_state is not None`) applied a server displacement
`s = x_new - x_bar` to the merged ScheduleFree EVAL weight `x` (held in `p.data`
after `optimizer.eval()`), it did NOT apply `s` to the base iterate `z`.
ScheduleFree's live train point is `y = beta1*x + (1-beta1)*z`, so
`optimizer.train()` then rebuilt `y+ = ybar + beta1*s` instead of `ybar + s` —
only a `beta1` (=0.9) fraction of the server update reached the next training
point — and the `x-z` gap was stretched by `-s`. This corrupted inner SF geometry
on EVERY nontrivial outer update, not just `beta>0`.

**The fix (one invariant).** Any displacement applied to `x` is applied to `z`
(hence `y`): `x<-x+s`, `z<-z+s` ⇒ `y<-y+s` (translation invariance). See
`diloco_merge` in `train.py`. Secondary: the DiLoCo anchor is captured in the SF
EVAL (x) basis (resolves the open question: **outer delta uses eval weights**),
so the first outer delta is `xbar - x`, not `xbar - y`. An env-gated runtime
assert (`NDM_DILOCO_DEBUG_ASSERT=1`) checks the `x-z` gap is preserved across the
outer rebase.

**Regression lock** (`tests/test_diloco_merge.py`, 4 tests, all REAL gloo merges):
1. translation-invariance scalar: `x=10,z=4,beta1=0.9 -> y=9.4`; `s=+3` ⇒
   `x=13,z=7,y=12.4`. FAILS on pre-fix (`z=4, y=12.1`).
2. identical-rank no-op for avg + momentum(lr1/b0, lr.7/b.9) modes.
3. anchor in x-basis: first outer delta `xbar - x == 0` with no local training.
4. state-gap preservation: `(z+s)-(x+s)==z-x` under a nonzero-`s` outer rebase.

**Plain-avg baseline reproduced** (`scripts/launch_sf_diloco_pavg_baseline.sh`,
1.3B E97 emender, `lr=1.01e-3`, bf16, fused Triton split-edit kernel NO-eager,
p50k_base, pile.txt, 2 GPUs). This is the UN-bugged averaging path (outer_state
is None, untouched by the fix); the run confirms it still descends smoothly
across merge boundaries with no post-merge spike:

```text
K=250 (4 merges @250/500/750/1000): loss 9.24(s25) -> 5.19(s1100), every merge continuous
K=100 (10 merges @100..1000):       loss 9.14(s25) -> 4.82(s1100), every merge continuous
~16k global tok/s (8k/GPU), merge ~1.0s amortized over K steps.
```

This matches the shape of the prior 20h/72k-step run that descended to ~2.96
(same recipe, early on the same trajectory).

## P3 RESULTS — separate outer ScheduleFree-SGD state (task sf-diloco-p3)

P3 adds explicit outer-optimizer routing:

```text
--diloco_outer_optimizer avg|momentum|sfsgd
--diloco_export_basis x|y
```

`avg` is the stateless periodic-average path. `momentum` is the P1/P2
geometry-fixed fixed-momentum DiLoCo update. `sfsgd` is a separate manual
ScheduleFree-SGD state machine with state `{mode, x, z, y, k, weight_sum,
lr_max}`; it does not wrap live model parameters in `schedulefree.SGDScheduleFree`.
The inner ScheduleFree AdamW state remains the normal optimizer state and is
translated at each DiLoCo boundary by rebasing both the implicit inner `x` and
inner `z` onto the new outer `y`.

Checkpointing now stores both systems when needed:

```text
optimizer_state_dict      # inner ScheduleFree AdamW x/z/y clock and moments
diloco_outer_state        # outer sfsgd or momentum state
```

Validation:

```text
python -m py_compile train.py tests/test_diloco_merge.py
python -m pytest tests/test_diloco_merge.py -q -s      # 11 passed
python -m pytest tests/test_diloco_hybrid.py -q -s     # 1 passed
```

Real 2-GPU bf16 E97 smoke, using the broker with `--no-wait`, passed fused
split-edit Triton `NO eager fallback` on both ranks:

```text
K=50 STEPS=125 OUTER_LR=1.0 OUTER_BETA=0.1 GPUS=2 \
  scripts/launch_sf_diloco_sfsgd_smoke.sh
```

Loss stayed finite and descended across sfsgd merges:

```text
step 25  loss 9.1437
step 50  merge #1, loss 7.4671
step 100 merge #2, loss 6.9988
step 125 final merge, loss 6.6703
```

The step-125 checkpoint contained `diloco_outer_state.mode=sfsgd`,
`outer_k=3`, `outer_weight_sum=3.0`, and `optimizer_state_dict`. A short resume
from that checkpoint restored the outer state and crossed the next merge:

```text
Resumed at step 125
[DiLoCo] restored outer optimizer state (sfsgd) from checkpoint
step 150 merge, loss 6.1559
final checkpoint: outer_k=4, outer_weight_sum=4.0
```

Local logs for this validation run:

```text
/tmp/sf_diloco_sfsgd_smoke_p3.log
/tmp/sf_diloco_sfsgd_resume_p3.log
```
