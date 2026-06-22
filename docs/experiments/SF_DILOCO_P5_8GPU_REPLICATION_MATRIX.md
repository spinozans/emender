# SF-DiLoCo P5 Island-Count Scaling Matrix

Task: `sf-diloco-p5`
Date: 2026-06-22
Downstream runner: `sf-diloco-p6`

## Purpose

P4 was a hypothesis-generating four-arm screen at small DiLoCo world size. It
found:

| P4 arm | Final held-out BPB | Read |
| --- | ---: | --- |
| SF inner + plain avg outer | 2.0361 | Best observed arm |
| SF inner + outer sfsgd export=y | 2.0426 | Close enough to plausibly be noise |
| SF inner + fixed momentum outer | 2.0909 | Dropped |
| SF inner + outer sfsgd export=x | 3.4744 | Dropped |

P5 should compare only the two plausible arms:

1. `avg`: ScheduleFree AdamW inner + plain average DiLoCo outer.
2. `sfsgd_y`: ScheduleFree AdamW inner + outer SF-SGD with `--diloco_export_basis y`.

The core P5 question is island-count parameterization, not only "who wins at
8 GPUs." Plain averaging looks strong at small W, but the reason to consider an
outer ScheduleFree rule is that hundreds or thousands of independent islands may
change the scale and stability of the averaged endpoint update. P5 should
therefore measure how the `sfsgd_y - avg` effect changes as DiLoCo world size W
increases on the available 8-GPU box.

## Island-Count Strategy

Run a W-scaling stress matrix at `W in {2, 4, 8}`. Here W is the number of
independent DiLoCo ranks entering each outer merge.

Use true 8-GPU utilization for throughput:

| W | GPU usage per run | Concurrent clean partitions on the 8-GPU box | Why |
| ---: | ---: | ---: | --- |
| 2 | 2 GPUs | 4 partitions | Replicate small-W P4-like behavior while using all GPUs |
| 4 | 4 GPUs | 2 partitions | Middle point for W trend |
| 8 | 8 GPUs | 1 partition | Full available scale-out point |

Each partition must have an exclusive `CUDA_VISIBLE_DEVICES`, output directory,
log file, and held-out curve path. Parallel W=2 or W=4 partitions are acceptable
because they are independent runs and answer the W-scaling question faster. Do
not accidentally run the W=8 point as serial 2-GPU jobs.

This still does not prove behavior at thousands of islands. It gives the first
local stress evidence: whether the paired BPB delta, merge shock, recovery, and
sync cost are flat, improve, or degrade as W increases from 2 to 8. P6 should
report the W trend explicitly and label extrapolation beyond W=8 as a risk
assessment, not an empirical result.

## Shared Training Configuration

Use the P4 model and optimizer surface unchanged except for W, seed, and output
roots:

```text
model:                 E97
dim:                   1792
n_heads:               216
n_state:               32
depth:                 11
expansion:             1.0
use_gate:              1
gate_activation:       silu
mlp_ratio:             2.2623
mlp_multiple:          64
tokenizer:             p50k_base
data:                  /home/erikg/elman/data/pile.txt
precision/kernel:      --bf16 --use_triton 1
inner optimizer:       --optimizer schedulefree
inner lr:              0.001007
batch_size per rank:   4
chunk_size:            2048
DiLoCo K:              250 local optimizer steps
local steps per run:   1500
log_every:             25
heldout_curve_every:   250
save_every:            100000000
keep_checkpoints:      1
heldout_eval_mode:     x
heldout tensor:        experiments/lb_compare_20260613/heldout_p50k_2048.pt
```

The held-out tensor is intentionally the P4 fixed lb-compare tensor: 64 chunks x
2049 tokens, 131072 scored tokens, `bytes_per_token=3.878128`. Keep held-out
scoring in ScheduleFree eval/averaged basis (`mode=x`) so the outcome is
directly comparable to P4 and to the intended inference-time weight basis.

Use fixed local steps rather than fixed global tokens across W. The stress
question is how the outer update behaves after the same number of local
optimizer steps and DiLoCo boundaries as the number of independent endpoints per
merge changes. Absolute held-out BPB will improve with more global tokens at
larger W, so primary inference is always based on paired arm deltas within the
same W, not raw BPB across W.

Token budget by W:

| W | Tokens per local step | Local steps | Training tokens per run | Merge windows |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 16,384 | 1500 | 24,576,000 | 6 |
| 4 | 32,768 | 1500 | 49,152,000 | 6 |
| 8 | 65,536 | 1500 | 98,304,000 | 6 |

## Replication Matrix

Use paired starts within each W. For a given `(W, seed)` cell, run both arms
with the same `--seed`, data seed behavior, model configuration, and local step
budget. `train.py` already exposes `--seed` and offsets dataset seeds by rank,
so strict paired starts are launcher-controllable.

Primary design:

| W | Seeds / starts | Runs | Partitioning |
| ---: | --- | ---: | --- |
| 2 | 7000, 7001, 7002, 7003, 7004, 7005 | 12 | up to 4 concurrent 2-GPU partitions |
| 4 | 7000, 7001, 7002, 7003, 7004, 7005 | 12 | up to 2 concurrent 4-GPU partitions |
| 8 | 7000, 7001, 7002, 7003, 7004, 7005 | 12 | one full 8-GPU run at a time |

Total primary matrix: 36 runs, all limited to the two arms. W=2 and W=4 can be
packed concurrently, so wall-clock cost is dominated by the 12 full-W=8 runs.

Launch order should alternate by seed parity within each W:

| Seed parity | First arm | Second arm |
| --- | --- | --- |
| even seeds | `avg` | `sfsgd_y` |
| odd seeds | `sfsgd_y` | `avg` |

If primary cost must be reduced, do not drop W points. Use a staged gate:

1. Run seeds `7000..7002` for W=2, 4, and 8.
2. If all launch/guardrail checks pass, run seeds `7003..7005`.

Do not make a win/tie/loss call from only the first three seeds unless the
result is a safety stop, such as repeated nonfinite failures or fused fallback.

## Arm Definitions

`avg`:

```text
--diloco
--diloco_k 250
--diloco_outer_optimizer avg
--diloco_outer_lr 1.0
--diloco_outer_beta 0.0
```

`sfsgd_y`:

```text
--diloco
--diloco_k 250
--diloco_outer_optimizer sfsgd
--diloco_export_basis y
--diloco_outer_lr 1.0
--diloco_outer_beta 0.1
```

Do not include fixed momentum, `sfsgd` export x, beta sweeps, LR sweeps, or K
sweeps in P5. Those are separate questions and would dilute the W-scaling
comparison.

## Output Layout

Use a clean run root:

```text
/mnt/nvme1n1/erikg/sf_diloco_p5_island_scaling/
  logs/
    W02_seed7000_avg.log
    W02_seed7000_sfsgd_y.log
    W04_seed7000_avg.log
    W08_seed7000_avg.log
    ...
  curves/
    W02_seed7000_avg_heldout_curve.csv
    ...
  runs/
    W02_seed7000_avg/
    W02_seed7000_sfsgd_y/
    ...
  summary.json
  paired_by_w.json
  scaling_decision.json
```

Each run log must include:

- `CUDA_VISIBLE_DEVICES` and `WORLD_SIZE=W` evidence.
- Full command line or equivalent serialized config.
- Fused guard lines for every rank in that run.
- DiLoCo outer optimizer line and export basis for `sfsgd_y`.
- Final held-out BPB, CE, token count, and mode.
- DiLoCo merge count, K, total sync seconds, and average sync seconds.

## Guardrails

These are hard eligibility filters:

- `--bf16` and `--use_triton 1` are required.
- Every rank in the run must print fused-guard `NO eager fallback`.
- No eager fallback, missing fused-guard line, missing final held-out BPB, or
  nonfinite loss makes the run ineligible.
- `WORLD_SIZE` must equal the planned W for that run.
- `avg` must not create or restore `diloco_outer_state`.
- `sfsgd_y` must print `outer optimizer: sfsgd` and `export_basis=y`.
- Held-out eval must be `mode=x` for both arms.
- W=2 and W=4 concurrent partitions must have disjoint GPU IDs and disjoint
  outputs/logs/curves.
- Output directories and logs must be under the P5 root, not under P4 or smoke
  roots.
- `NCCL_P2P_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`,
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should match P4 unless the
  runner documents a hardware-specific reason to change them.

## Shock and Recovery Metrics

Compute the P4 shock metric for every run:

```text
pre_loss:       most recent logged train loss before a DiLoCo merge step
post_loss:      first logged train loss at or after the merge step
jump:           post_loss - pre_loss
recovery_step:  first later logged step with loss <= pre_loss
recovery_steps: recovery_step - merge_step
```

Report per run:

- `mean_jump`
- `max_jump`
- `mean_positive_jump`
- `max_recovery_steps`
- number of unrecovered positive jumps by end of run
- final held-out BPB
- final curve BPB at step 1500
- per-merge sync time and sync-time fraction over the K-step window

Report paired deltas within each W:

```text
delta_bpb_W_seed = final_heldout_bpb(sfsgd_y) - final_heldout_bpb(avg)
delta_curve_bpb_W_seed = curve_bpb_1500(sfsgd_y) - curve_bpb_1500(avg)
delta_max_jump_W_seed = max_jump(sfsgd_y) - max_jump(avg)
```

Negative `delta_bpb` means `sfsgd_y` is better. Positive `delta_bpb` means
plain avg is better.

## Statistical Decision Rule

Freeze this rule before P6 results are known.

Primary endpoint: paired final held-out BPB delta within W:

```text
d_{W,i} = BPB_{W,i}(sfsgd_y) - BPB_{W,i}(avg)
```

Use only complete eligible pairs. If one arm in a pair is ineligible after at
most one infrastructure retry, exclude the whole pair for that W and report it
separately. Do not borrow the other arm into an unpaired primary analysis.

For each W:

- Estimate `mean_delta_W = mean(d_{W,i})`.
- Estimate `sd_delta_W` and paired standard error
  `se_W = sd_delta_W / sqrt(n_W)`.
- Compute a two-sided 95% paired t interval for `mean_delta_W`.
- Report paired Cohen dz: `mean_delta_W / sd_delta_W`.
- Report sign count across pairs.

Tie band:

```text
epsilon = 0.005 BPB
```

This is close to the P4 observed gap of 0.0065 BPB and prevents treating tiny
local differences as operational wins.

Per-W decision:

| Result | Rule |
| --- | --- |
| `sfsgd_y` wins at W | 95% CI upper bound < `-0.005` BPB and at least 4 of 6 deltas are negative |
| `avg` wins at W | 95% CI lower bound > `+0.005` BPB and at least 4 of 6 deltas are positive |
| practical tie at W | entire 95% CI lies within `[-0.005, +0.005]` BPB |
| inconclusive at W | anything else |

If fewer than 5 complete eligible pairs remain at a W, do not declare win/loss
for that W. Run replacement pairs or label that W inconclusive.

Scaling decision:

| System-level conclusion | Predeclared evidence |
| --- | --- |
| `avg` remains safe across local W scaling | W=8 is tie or avg win; `mean_delta_W` is not trending downward in favor of `sfsgd_y` by more than 0.005 BPB from W=2 to W=8; avg does not show worse shock/recovery trend with W |
| `sfsgd_y` is more robust to increasing island count | W=8 is `sfsgd_y` win or inconclusive with `mean_delta_8 <= -0.005`, and the fitted trend of `mean_delta_W` versus `log2(W)` is negative by at least 0.005 BPB per doubling from W=2 to W=8; shock/recovery is no worse than avg |
| practical tie across local W scaling | all complete W points are ties, or all `mean_delta_W` values lie in `[-0.005, +0.005]` with no monotone W trend and no material shock/recovery separation |
| local evidence insufficient for thousands of islands | any missing/inconclusive W point, contradictory signs across W, or trend magnitude below 0.005 BPB per doubling while confidence intervals overlap both win bands |

Trend fitting is descriptive but predeclared: fit ordinary least squares to the
three points `(log2(W), mean_delta_W)` for W=2,4,8 and report the slope in BPB
per doubling. With only three W points, do not claim precise extrapolation.
Instead, use the sign and magnitude of this local slope as stress evidence.

## Thousands-of-Islands Extrapolation

P5 cannot empirically validate W=hundreds or W=thousands on an 8-GPU box. The
runner should make one of these explicit risk statements:

- If avg is tie/win at W=8 and the W trend is flat or moves against `sfsgd_y`,
  then there is no local evidence that avg's small-W advantage is a
  parameterization artifact. Avg remains the safer next production default, but
  a Frontier-scale canary should still monitor merge shock and held-out drift.
- If `sfsgd_y` improves monotonically with W and is best or nearly best at W=8,
  then P5 supports the hypothesis that outer ScheduleFree is absorbing larger
  endpoint variance better than plain averaging. Promote `sfsgd_y` to the
  scale-out canary candidate or require a 16+ island follow-up before replacing
  avg, depending on stability.
- If W points disagree or intervals are wide, P5 does not answer the
  thousands-of-islands concern. Keep avg as the conservative default and create
  a follow-up task for a larger true-island run rather than tuning beta/LR on the
  same 8-GPU evidence.

Do not extrapolate raw BPB across W because larger W processes more tokens at
the same local step count. Extrapolate only the paired arm delta and stability
metrics within W.

## Stopping Criteria

Per-run stop/fail criteria:

- Stop immediately on nonfinite loss or gradients.
- Stop immediately on fused guard failure or eager fallback.
- Stop immediately if world size is not the planned W.
- Stop the arm and mark ineligible if no DiLoCo merge occurs by step 250.
- Stop the arm and mark ineligible if held-out scoring fails at the first curve
  checkpoint.
- Stop the current W block and inspect launcher/hardware if two consecutive
  runs fail before step 250 for the same non-model reason.

Matrix-level stopping:

- Complete W=2, W=4, and W=8 before declaring avg safe, `sfsgd_y` robust, or
  tie across W.
- After 6 pairs at each W, apply the frozen per-W and scaling decision rules.
- If exactly one W point is inconclusive and all runs were clean, run exactly 2
  additional paired starts at that W (`7006`, `7007`) before revisiting the
  scaling decision.
- Do not stop early for an apparent win before at least 5 complete eligible
  pairs at W=8 and at least 3 complete eligible pairs at W=2 and W=4.

## Launcher and Analyzer Changes Needed Before P6

P6 should add a dedicated P5 launcher instead of reusing P4 directly.

Required launcher changes:

- New script, suggested path: `scripts/launch_sf_diloco_p5_island_scaling.sh`.
- Matrix over `W=2,4,8`, seeds `7000..7005`, and arms `avg,sfsgd_y` only.
- Default `STEPS=1500`, `K=250`, `HELDOUT_EVERY=250`.
- For W=2, schedule up to four concurrent 2-GPU partitions with disjoint GPU
  IDs and output paths.
- For W=4, schedule up to two concurrent 4-GPU partitions with disjoint GPU IDs
  and output paths.
- For W=8, schedule one full 8-GPU run at a time and fail if the lease returns
  fewer than 8 GPUs.
- Alternate arm order by seed parity.
- Pass `--seed "$RUN_SEED"` for every run.
- Print the full resolved command and partition assignment before execution.
- Refuse to start if a target output directory already exists unless
  `ALLOW_OVERWRITE=1` is explicitly set.

Required analyzer changes:

- Either generalize `scripts/analyze_sf_diloco_p4.py` to group by W/seed/arm
  and accept arbitrary curve directories, or create
  `scripts/analyze_sf_diloco_p5.py`.
- Parse all 36 logs, enforce per-W fused-guard eligibility, compute shock and
  recovery metrics, and write `summary.json`, `paired_by_w.json`, and
  `scaling_decision.json`.
- Implement the frozen per-W paired decision rule and scaling decision rule
  exactly as written above.

No `train.py` optimizer or seed-plumbing changes are expected for P5.

## Validation Checklist

- Matrix compares only `avg` and `sfsgd_y`, with the P4 rationale above.
- The design is centered on island-count scaling at W=2, W=4, and W=8.
- Seed/start replication is concrete and paired within each W.
- 8-GPU usage is explicit: W=2 and W=4 use clean parallel partitions; W=8 uses
  a full 8-GPU run.
- Held-out tensor, K, LR/beta settings, token budget, shock metrics, stopping
  criteria, output layout, and guardrails are fixed before P6.
- Statistical decision criteria define per-W win/tie/loss and system-level
  avg-safe, `sfsgd_y`-robust, tie, and insufficient-evidence outcomes before
  new results are known.
- Launcher/analyzer changes needed before the run task are listed.
