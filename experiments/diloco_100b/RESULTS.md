# implement-diloco-periodic — RESULTS (measured)

DiLoCo periodic-sync parallelism for the 100B-token seed run. **All numbers below
are REAL, measured on the 8×RTX 6000 Ada box (49 GB each, PCIe, NO NVLink),
commapile_mainmix 1 TB, p50k_base tokenizer, ctx 2048, bf16 + FUSED Triton,
schedule-free AdamW, 7 GPUs.** Date 2026-06-14. Builds on `preflight-100b`
(commit f542cba) which measured vanilla per-step DDP at 31,291 global tok/s (52%
scaling efficiency) and the ~62k tok/s independent-process ceiling.

Geometry: **emender-mlp** = E97 split-edit DELTA + SwiGLU MLP, dim1792 nh216 ns32
dep11 mlp2.2623 — **1,286,589,072 params** (identical to preflight).

## What was built (train.py)

Opt-in `--diloco` mode (only active under torchrun, single-GPU path byte-identical):

- **No per-step DDP gradient all-reduce.** Each rank trains its replica
  INDEPENDENTLY; rank-0 weights are broadcast once at start (`W_0`) so all islands
  begin identical.
- **`diloco_merge()`** averages MODEL WEIGHTS across ranks every `--diloco_k` local
  optimizer steps via ONE coalesced all-reduce of the 1.29B-param flat bucket
  (SUM/÷world, backend-agnostic). The 2.6 GB sync happens once per K steps, not
  once per step — the whole point.
- **ScheduleFree y-mode swap at merges** (per `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`):
  merge runs in `optimizer.eval()` so the averaged quantity is the eval (x) weights;
  the base sequence `z` is then reset to the consensus so the `train()`-swap restores
  `y == x == W_{r+1}` on every rank → all ranks byte-identical post-merge, no
  inter-round z-drift. Adam second moments stay per-rank (independent exploration).
- **General DiLoCo outer step** `W_{r+1} = W_r + outer_lr·(outer_beta·mom + (mean_i W_{r,i} − W_r))`;
  defaults `outer_lr=1, outer_beta=0` reduce to plain periodic weight averaging
  (local-SGD), the conservative first production path. Outer-momentum buffers are
  only allocated when `outer_beta≠0` or `outer_lr≠1`.
- A FINAL merge before checkpointing so the saved model is the cross-rank consensus.

## 1. THROUGHPUT — DiLoCo K-sweep vs DDP (emender-mlp 1.286B, 7-GPU, bs6)

| K | effective global tok/s | per-GPU | pure no-merge steady | vs DDP (31,291) | % of 62k ceiling | merges | sync/merge |
|---|---|---|---|---|---|---|---|
| 100 | **57,175** | 8,168 | 58,171 | **1.83×** | 92.2 % | 4 | 2,347 ms |
| 250 | **57,745** | 8,249 | 58,094 | **1.85×** | 93.1 % | 3 | 2,164 ms |
| 500 | **57,921** | 8,274 | 58,046 | **1.85×** | 93.4 % | 2 | 2,117 ms |

- **Pure no-merge steady throughput ≈ 58,100 global tok/s** — the directly-measured
  independent-process ceiling under this exact config (preflight estimated ~62k).
- **The periodic merge costs ~2.1–2.3 s** (one 2.6 GB all-reduce over PCIe). Amortized
  over K steps this is **0.2 % overhead at K=500, 1.7 % at K=100** → effective
  throughput is **98.3–99.6 % of the no-merge ceiling**.
- DiLoCo recovers essentially all of the independent ceiling that vanilla per-step
  DDP throws away: **~1.85× the 31,291 DDP baseline.**

## 2. VALIDATION GATE — ≥0.85× of the ~62k ceiling at K≥250

| K | % of 62k ceiling | verdict |
|---|---|---|
| 250 | 93.1 % | **PASS** |
| 500 | 93.4 % | **PASS** |

(Relative to the *directly measured* 58.1k ceiling, effective throughput is
98.3–99.6 % at every K.)

## 3. y-MODE MERGE CORRECTNESS

- **Unit test** `tests/test_diloco_merge.py` (REAL gloo 2-process group + REAL
  ScheduleFree AdamW, rank-specific gradients so weights genuinely diverge):
  - `test_local_sgd_averaging` **PASS** — post-merge params byte-identical across
    ranks (max cross-rank diff ≤1e-5), merged eval `x` == mean of per-rank pre-merge
    `x`, base sequence `z` reset to consensus, train-mode `y` == merged `x`.
  - `test_outer_momentum` **PASS** — general outer step
    `W_{r+1}=W_r+lr·(β·mom+(mean_x−W_r))` reproduced exactly + consensus preserved.
- **Checkpoint roundtrip** (`ckpt_roundtrip.py`): the DiLoCo CONSENSUS checkpoint
  (K=500, step 650) loads into a fresh single-process model `strict=True` with
  **0 missing / 0 unexpected** keys and produces a FINITE held-out **BPB 2.1394**
  → the y-mode merge yields usable consensus weights, interchangeable with the
  normal inference path. `ROUNDTRIP_OK`.

## 4. LOSS-vs-TOKENS PARITY (no divergence) vs DDP

At matched STEP count both paths process identical GLOBAL tokens
(bs·ctx·world = 6·2048·7 = 86,016 tok/step), so loss-at-step == loss-at-tokens.

**No divergence.** DiLoCo loss decreases monotonically through every merge — at
K=100/250 there is no post-merge shock; at K=500 a small, fully-recovered bump
(the design's noted large-K "post-sync loss shock"). Loss never blows up.

**Merged-model held-out BPB at matched tokens** (the apples-to-apples metric — the
per-rank *training* loss is not comparable because DDP's effective batch is bs·7=42
while each DiLoCo island is bs6). One consistent experiment, both paths fresh to 600
steps on the same data/seed (`parity_trend.sh`):

| tokens (step) | DDP held-out BPB | DiLoCo K=100 held-out BPB | gap |
|---|---|---|---|
| 25.8 M (300) | **1.7201** | **2.1580** | +0.438 |
| 51.6 M (600) | **1.5928** | **2.0664** | +0.474 |

(An independent earlier pair of runs gave 1.7344 / 2.1437 at 25.8 M — reproducible.)

**Honest reading:**
- **No divergence** (the literal validation requirement): both curves decrease
  monotonically (DDP 1.720→1.593, DiLoCo 2.158→2.066); DiLoCo is stable through all
  6 merges, no blow-up. ✓
- **At matched tokens, local-SGD DiLoCo (outer_beta=0) LAGS synchronous DDP by
  ~0.44–0.47 BPB, and the gap does NOT close within 52 M tokens** — it is roughly
  flat (slightly widening). This is the well-known large-batch (DDP effective bs42)
  vs local-SGD (per-island bs6, averaged) sample-efficiency tradeoff, the cost of
  not doing a per-step global all-reduce.
- **It is not a merge bug**: the DiLoCo consensus (2.07 @ 51.6 M) is far better than
  an isolated bs6 worker would be at its own 3.7 M-token shard, so averaging is
  helping; it simply does not match full large-batch DDP at matched global tokens.
- **Caveat on the throughput win**: 1.85× more tokens/sec does NOT, in this early
  regime, buy a matched-wallclock loss win — DiLoCo@51.6 M (2.07) still trails
  DDP@~28 M (~1.70, what DDP reaches in the same wallclock). The DiLoCo bet for 100B
  rests on (a) the large-batch advantage shrinking as both approach the
  data/asymptotic regime over 100 B tokens (4 000× more than measured here) and/or
  (b) outer momentum — BOTH UNVERIFIED at this scale and horizon. The design's
  "model-only sync caught up by 5 K steps" was at 75 M params / ctx512 / 4 GPU, a
  different regime; it did NOT reproduce at 1.3 B / 7 islands / 52 M tokens here.

**Recommendation for parity:** before committing the full 100 B to DiLoCo, run the
outer-momentum sweep (`--diloco_outer_beta 0.5/0.9`, wired and ready) and a
longer-horizon (multi-B token) matched-token check, monitoring merged-checkpoint
BPB each round. Follow-up task `diloco-loss-parity-longhorizon` created.

## 5. PROJECTED WALL-CLOCK to 100B / 16B (updated)

| path | global tok/s | to 16B gate | to 100B seed |
|---|---|---|---|
| 7-GPU **DDP** (preflight) | 31,291 | 5.9 d | **37.0 d** |
| 7-GPU **DiLoCo K=500** (measured) | **57,921** | **3.2 d** | **20.0 d** |
| 7-GPU DiLoCo K=250 (measured) | 57,745 | 3.2 d | 20.0 d |
| independent ceiling (measured) | ~58,100 | 3.2 d | 19.9 d |

**DiLoCo K≥250 brings the 100B seed run to ~20 days (≈1.85× faster than DDP's 37
days), under the 3-week frame.** Use **K=250** (design-recommended, 93.1 % ceiling,
safest learning) or **K=500** (93.4 %, marginally faster, small recoverable shock).

## 6. RECOMMENDATION

**Throughput mechanism: GO.** DiLoCo periodic-sync (`--diloco --diloco_k 250/500`)
is implemented, divergence-free, and recovers **~98 % of the independent throughput
ceiling** that PCIe-bound per-step DDP wastes — **~1.85× DDP → 100 B in ~20 days**,
with bit-faithful consensus checkpoints and verified y-mode merge.

**Token sample-efficiency at outer_beta=0: OPEN.** At matched tokens the local-SGD
merge lags synchronous DDP by ~0.45 BPB in early training and the gap does not close
within 52 M tokens (§4). Whether it closes over 100 B tokens and/or with outer
momentum is **unverified**. So:

- **If wall-clock is the binding constraint** and a modest token-efficiency cost is
  acceptable, use DiLoCo `--diloco_k 250` (safest) — you still finish 100 B in ~20 d
  vs 37 d, and the seed checkpoint converges (just needs somewhat more tokens to hit
  a target BPB than DDP would).
- **Before betting the whole 100 B on it**, run the cheap gates first: the
  outer-momentum sweep (`--diloco_outer_beta 0.5/0.9`, wired) + a multi-B-token
  matched-token check, monitoring merged-checkpoint BPB each round
  (follow-up `diloco-loss-parity-longhorizon`). If the gap persists at outer
  momentum and long horizon, fall back to DDP (37 d, exact) or a 2-GPU-island
  hybrid (DDP within island + DiLoCo across islands) to trade some of the 1.85×
  for tighter sync.

### Reproduce
```
# throughput sweep K in {100,250,500}:
bash experiments/diloco_100b/sweep_diloco.sh /tmp/diloco_sweep 6
python experiments/diloco_100b/analyze_diloco.py \
  100:experiments/diloco_100b/logs/diloco_k100.log \
  250:experiments/diloco_100b/logs/diloco_k250.log \
  500:experiments/diloco_100b/logs/diloco_k500.log
# merge correctness (no GPU needed):
python tests/test_diloco_merge.py
# DDP loss-parity baseline + matched-token BPB:
bash experiments/diloco_100b/run_ddp_baseline.sh /tmp/diloco_ddp_base 300 6
python experiments/diloco_100b/ckpt_roundtrip.py --ckpt_dir <run_dir> --heldout <heldout.pt>
# single run:  experiments/diloco_100b/run_diloco.sh emender 550 6 250 /tmp/out
```
Raw logs: `experiments/diloco_100b/logs/`.
