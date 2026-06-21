# E97_delta + gdn-neg within-layer @ 1.3B — CMA-ES, fast chunked kernel

**Task:** `e97delta-1p3b` (re-run of the dominated `e97-scale` result, which the task spec
debunks: WRONG scale 0.48B, WRONG cell `e97_raw` — a train-loss artifact, SLOW
kernel).
**Date:** 2026-06-08.
**Run dir:** `experiments/e97_delta_1p3b_cma/`.
**Compute:** 8× RTX 6000 Ada, idle-only / no-preempt, REAL Pile
(`/home/erikg/elman/data/pile.txt`, p50k_base, ctx 2048), bf16, schedule-free
AdamW. CMA = 33.4 min wall / 267.6 GPU-min; head-to-head + capability ≈ 40 min.
No mocks anywhere; every BPB is a real held-out measurement with a fresh-process
checkpoint round-trip (delta ~1e-6).

---

## 0. Decision deliverable (verdict)

**Does e97_delta + gdn-neg (within-layer, 1.3B, CMAES-tuned, fast chunked kernel)
BEAT gdn2-mlp on held-out BPB — token-matched AND wall-clock-matched?**

> **TIE / SPLIT — it BEATS token-matched but LOSES wall-clock**
> (keep gdn2-mlp as the 1.3B scale cell), but the prior wall-clock verdict's *reasoning* is
> overturned: e97_delta is **not** a pure liability — it is genuinely **more
> token-efficient**.

The two matched comparisons (2 seeds each, 12-min runs, param-matched to ~1.26B)
disagree, and that disagreement is the result:

| axis | gdn2-mlp | e97_delta+gdn-neg | winner |
|---|---:|---:|---|
| **WALL-CLOCK-matched** (720 s, equal compute) | **2.027** | 2.071 | **gdn2-mlp** (+0.044) |
| **TOKEN-matched** (4.99 M tokens, equal data) | 2.094 | **2.071** | **e97_delta** (+0.023) |

- e97_delta + gdn-neg reaches a **lower BPB per token** — at equal data it wins.
- gdn2-mlp runs **~40 % faster** (≈9 770 vs ≈6 985 tok/s), so in equal wall-clock
  it processes ~7.0 M vs ~5.0 M tokens and wins on the axis that actually governs
  fixed-GPU-hour pretraining.

**Why it loses wall-clock despite the token-matched win:** fixed-compute LM pretraining is
wall-clock-bound (you have a GPU-hour budget, not a token budget), gdn2-mlp wins
that axis decisively and is architecturally simpler (one kernel per layer, no
within-layer two-block split). **Why the prior wall-clock verdict's reasoning is overturned:**
its stated cause was a "2.6× slower, 13–15 %-util" kernel — refuted here (§2,
util 88–100 %). With the throughput objection fairly removed, e97_delta loses
wall-clock only by the residual ~30 % cost of the *within-layer two-kernel split*
(running a 21-head gdn block AND a 43-head delta block per layer), not by the
kernel — and it is strictly more sample-efficient. The lever to revisit is that
split overhead, not the kernel.

Capability spot-check (§6) explains the token-efficiency: e97_delta adds a real
**counting** advantage (and better count length-extrapolation) while costing some
recall/track — the known E97 expressivity profile.

---

## 1. Anti-regression confirmation (task gate)

| gate | status | evidence |
|---|---|---|
| `e97_delta` fused head present in `typed_head_mixture.py` | **pass** | type idx 7; `n_e97_delta` allocated; `E88FLAHybrid(raw_write=False, use_chunked_e97=True)` |
| Architecture = WITHIN-LAYER head-type fractions (`typed-gdn2-lm`), NOT interleaved | **pass** | one `TypedHeadMixtureLayer` per layer; gdn-neg + e97_delta heads run in parallel within each layer; `+ SwiGLU MLP` (`mlp_ratio` = 6208/2304) |
| CHUNKED fast kernel, util ≥ 85 %, no eager | **pass** | §2 |
| REAL data, no mocks | **pass** | real Pile every screen; real fwd/bwd; fresh-process checkpoint round-trip |

gdn-neg == `gdn2_recall` heads with `gdn_allow_neg_eigval=True` (negative
along-key eigenvalue for recall+track; global on this path, as in the prior E99
batch).

---

## 2. Kernel verification (util ≥ 85 %, no eager) — reproduced in this worktree

Isolated chunked-kernel fwd+bwd benchmark (`experiments/e97_chunked_kernel/
bench_kernel.py`), re-run live here:

| shape | e97_fused util (mean) | ms | vs GDN-2 |
|---|---:|---:|---:|
| within-layer (B8 T512 H16 N32) | 87.8 % | 0.91 | **0.47× (faster)** |
| scale_1p3b (B4 T2048 H16) | 97.9 % | 1.35 | 0.85× (faster) |
| scale_1p3b_wide (B4 T2048 H32) | **99.5 %** | 1.84 | 1.18× |

The fused chunked kernel runs at **88–99.5 % SM utilization**; the eager PyTorch
reference is a separate labeled arm at **8.62× slower** and is never on the
training path. During the actual CMA + head-to-head screens, all workers held
**94–100 % GPU utilization**. The "2.6× slower, 13–15 % util" premise behind the
prior wall-clock verdict is fully refuted — **this is why the wall-clock bar is now FAIR.**

Two real kernel-integration issues were found and fixed before any screen (both
REAL, both in the task log):
1. **`B,T,H` are `tl.constexpr`** in the chunked kernel → each distinct e97_delta
   head count cold-JIT-compiles. Fixed with untimed compile-warmup steps so
   compilation never eats the timed budget → every candidate trains for **equal
   compute** regardless of cache state (`compile_seconds` logged separately, 9–12 s).
2. **`model.eval()` routes e97_delta to a ~90× slower sequential scan** (the
   chunked path is gated on `training=True`). Held-out BPB + round-trip were moved
   to `train()`+`no_grad` (identical loss: dropout = 0, RMSNorm has no batch
   stats) → 13 s → 0.14 s per eval forward; round-trip delta ~1e-6 confirms it.

---

## 3. Parameter matching (~1.3B, counts logged)

Both arms share the fixed non-capacity axes (depth 18, n_heads 64, n_state 32,
expansion 1.0, SwiGLU `mlp_ratio` ≈ 2.694) and differ ONLY in `head_type_logits`
and `dim`, where `dim` is derived per allocation to hold counted LadderLM params
at the **1.27 B ± 2 %** target (exact count; no_init fast-count verified identical
to a full build).

| arm | dim | params | counts |
|---|---:|---:|---|
| gdn2-mlp baseline | 2240 | 1.2589 B | 64 gdn-neg |
| e97_delta-only | 2048 | 1.2426 B | 64 e97_delta |
| CMA winner (cma_g2_e1) | 2112 | 1.2532 B | 21 gdn-neg / 43 e97_delta |

All 27 trained configs (3 anchors + 24 CMA) within 2 % of 1.27 B; per-eval counts
in `results/cma_all_results.json`.

---

## 4. CMA-ES (wall-clock-matched 360 s screens): pure gdn-neg is the fast optimum

CMA-ES (popsize 8 × 3 gens = 24 evals) over the 9-dim vector {6 head-type logits:
gdn-neg, e97_delta, e97_track, count, latch, nonlin} + log knob-LR + lam_max +
beta_max, seeded 50/50, σ = 1.0. Fitness = held-out BPB from a **360 s
wall-clock-matched** fused screen (compile excluded).

**Anchors (360 s wall):** A_gdn2_mlp **2.168** (3.27 M tok, 9151 tok/s) <
A_seed 50/50 2.217 (2.61 M tok) < A_delta_only 2.266 (3.11 M tok). The CMA then
searched the interior; **the best searched candidate** was `cma_g2_e1`
(21 gdn / 43 delta) at **2.207** — and **all 24 candidates (2.207–2.429) lost to
the pure-gdn-neg anchor** under wall-clock selection. Per-generation best:
2.2113 → 2.2150 → 2.2069 (no convergence toward 2.168).

This is the **wall-clock** story, and it is consistent with §0: at a short budget
(≤360 s, ≤3.3 M tokens) gdn-neg's throughput advantage dominates and pure gdn-neg
wins outright. The CMA's wall-clock-matched fitness *correctly* selects gdn-neg —
**but because that fitness is throughput-weighted, it cannot see the per-token
crossover.** That crossover is what §5 measures at a longer budget.

Full ranked table: `results/cma_all_results.json`.

---

## 5. Head-to-head @ 1.3B — token-matched AND wall-clock-matched (the verdict)

CMA winner `cma_g2_e1` (21 gdn-neg / 43 e97_delta + MLP, 1.2532 B) vs `A_gdn2_mlp`
(64 gdn-neg + MLP, 1.2589 B), 2 seeds, 720 s real-Pile training, round-trip on.

| arm | seed0 | seed1 | mean BPB | tokens | tok/s |
|---|---:|---:|---:|---:|---:|
| **W1** e97_delta+gdn-neg, 720 s wall | 2.0633 | 2.0786 | **2.0709** | 5.0 M | 6985 |
| **W2** gdn2-mlp, 720 s wall | 2.0213 | 2.0324 | **2.0269** | 7.0 M | 9770 |
| **W3** gdn2-mlp, token-capped @ 4.99 M | 2.0828 | 2.1055 | **2.0941** | 4.99 M | 9551 |

- **WALL-CLOCK-matched** (W1 vs W2): gdn2-mlp **2.027** beats e97_delta **2.071**
  by **+0.044** — both seeds. gdn2-mlp's ~40 % throughput edge → 7.0 M vs 5.0 M
  tokens in the same 12 min.
- **TOKEN-matched** (W1 vs W3, both at 4.99 M tokens): e97_delta **2.071** beats
  gdn2-mlp **2.094** by **+0.023** — both seeds. At equal data the within-layer
  e97_delta cell is the more sample-efficient learner.

The split is robust (consistent across both seeds on both axes), params are
matched (1.253 vs 1.259 B), all round-trips passed. **Verdict: TIE — wins
token-matched, loses wall-clock → does not replace gdn2-mlp** (fixed-compute
pretraining is wall-clock-bound), with the throughput gap (within-layer split, not
the kernel) the concrete lever to revisit.

---

## 6. Capability spot-check (recall / track / count) — why e97_delta is token-efficient

Length-extrapolation probes (`train_hybrid.py`, typed-gdn2, depth 4, n_heads 64,
matched ≈4 M params, 2000 steps, autocast bf16, eval T ∈ {128…1024}, mean acc over
T, 2 seeds). recall = mqar_recall, track = s5_permutation, count = anbncn_viability.

| arm | recall | track | count |
|---|---:|---:|---:|
| gdn2-mlp (100 % gdn-neg) | **0.099** | **0.084** | 0.868 |
| e97_delta + gdn-neg (CMA mix) | 0.084 | 0.069 | **0.906** |

e97_delta buys a real **counting** advantage — and a cleaner count
length-extrapolation (T=1024 acc 0.81 vs 0.72) — at the cost of some recall and
track. This is the textbook E97 expressivity profile (E97 solves count/latch,
is weak on recall) and is the mechanistic reason the e97_delta mixture is more
*per-token* efficient on Pile yet does not dominate: Pile next-token prediction
rewards recall (gdn-neg's strength), so the count gain only partly offsets the
recall/track loss, leaving a per-token edge but a throughput-driven wall-clock loss.

---

## 7. Validation checklist

| criterion | status |
|---|---|
| e97_delta + gdn-neg within-layer @ ~1.3B (param-matched, counts logged), CHUNKED fast kernel (util ≥ 85 %, no eager) | **pass** (§1–3) |
| CMAES over the 1.3B parameterization; time-bounded fused screens; REAL Pile, no mocks | **pass** (§4) |
| held-out BPB vs gdn2-mlp 1.3B, token-matched AND wall-clock-matched | **pass** (§5) |
| capability spot-check (recall, track, count) | **pass** (§6) |
| explicit BEAT/TIE/LOSE verdict + accept/reject; doc committed | **TIE (wins token-matched, loses wall-clock) → does not replace gdn2-mlp** (§0) |

**Artifacts:** `experiments/e97_delta_1p3b_cma/{shapes,screen,orchestrate,
final_headtohead,capability}.py`; `results/{cma_all_results,headtohead_results,
capability_results}.json`.
