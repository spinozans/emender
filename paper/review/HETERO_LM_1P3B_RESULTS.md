# e97-lm-1p3b — Final LM Verdict for the CMAES-best Heterogeneous E97 Cell at 1.3B

**Task:** `e97-lm-1p3b` · **Hardware:** NVIDIA RTX 6000 Ada (49 GB), GPUs 4–7
(4-GPU lane, sharing the box with `e97-cap-test` + `complex-eig-impl`) ·
torch 2.9.1+cu128, triton 3.5.1 · **Data:** REAL Pile (p50k_base), held-out slice ·
fused split-edit Triton kernel (loud no-eager guard), no mocks.

> **Headline.** On the **honest footing the task demands** (GDN-2-speed kernel,
> tight param-match, real Pile, both matching protocols), the CMAES-best
> heterogeneous cell is **a sample-efficiency TIE/slight-win but a WALL-CLOCK
> NO-GO**. At **matched tokens** the hetero cell beats gdn2-mlp by **−0.014 BPB**
> (2.063 vs 2.078 — it is more sample-efficient). At **matched wall-clock** it
> **loses by +0.033 BPB** (2.063 vs 2.030), because it pays a **real throughput
> penalty of 0.80× (LM training loop) / 0.73× (kernel microbench)** and therefore
> sees ~23 % fewer tokens in the same wall budget. **The task's "fused blended
> ≥0.95× GDN-2 → slight penalty" premise is REFUTED** by its own upstream result
> (`e97-hetero-cma`): the 0.954× head was the capability-*weak* gated-delta shell;
> the real depth-capability head (split-edit per-step-`tanh`) costs ~20–27 %
> throughput. **Verdict: gdn2-mlp wins the LM decision at honest wall-clock
> parity; the hetero cell's sample-efficiency edge does not overcome the
> throughput tax.** This is consistent with the entire e97 1.3B lineage
> (`e97delta-1p3b` TIE/split, `e97-wallclock-cma` NO-GO, `e97-scale` NO-GO).

---

## 1. What was trained

The CMAES-best heterogeneous config from `e97-hetero-cma §6`, against two matched
baselines, every arm **+/-2 % param-matched to 1.27B** (dim derived per allocation
via `shapes.derive_dim`, exact counts below):

| arch | description | dim | params | rel to 1.27B | ±2 %? |
|---|---|---:|---:|---:|:---:|
| **H** (hetero) | 48 `gdn2_recall` (gdn-neg, allow_neg_eigval) **+ 16 `e97_delta`** (split-edit, per-step `tanh` state) | 2176 | **1.2473B** | −1.79 % | ✅ |
| **G** (gdn2-mlp) | 64 `gdn2_recall` (gdn-neg) + SwiGLU MLP — the rank-1/2 1.3B reference | 2240 | **1.2589B** | −0.87 % | ✅ |
| **L** (LSTM) | gated additive cell (`level='lstm'`), reference only | 1896 | **1.2669B** | −0.24 % | ✅ |

Fixed axes for H and G (both arms): depth 18, n_heads 64, n_state 32, expansion 1.0,
SwiGLU `mlp_ratio = 6208/2304`. H and G differ **only** in head mixture and the
param-matched dim. Shared CMA-best optimizer/gate knobs (e97-hetero-cma §6):
`lr=8e-4, knob_lr_mult=11, lam_max=1.30, beta_max=3.35`, schedule-free AdamW, bf16.

**Fused kernel, no eager.** H routes its 16 `e97_delta` heads through the
**sequential split-edit fused Triton scan** with stream overlap ON
(`use_chunked_e97_delta=False`, `overlap_streams=True`). A loud guard
(`assert_fused_no_eager`) asserts every `e97_delta` head is on `use_triton=True`
at build; runs are NaN-free with stable peak memory (~20.5 GB). The split-edit
capability head is **non-chunkable** (`tanh ⊥ chunkable`, `fuse-2kernel` finding),
so "fused" here means the fused *sequential* kernel — the fastest available path
for the bounded-state capability head.

## 2. Harness (reuses the 1.3B CMA training loop verbatim)

`experiments/e97_hetero_cma/lm_verdict.py` swaps only the model builder into the
`e97_delta_1p3b_cma/screen.run()` loop (the same loop used by every prior 1.3B CMA
batch): schedule-free AdamW, bf16, REAL Pile `TokenizedStreamDataset`, untimed
compile warmup (so JIT does not eat the training wall), held-out BPB =
nats/token × tokens/byte / ln2 with **tokens/byte measured on the exact held-out
slice** (0.25942 for all arms → BPB is directly comparable). Two matching
protocols, 2 seeds (LSTM 1 seed reference):

- **WALL-CLOCK-matched:** every arm trains for the **same 720 s**. The slower H
  consumes fewer tokens; compare final held-out BPB.
- **TOKEN-matched:** every arm trains to **N_H = 5.32 M tokens** (the tokens H
  reaches in 720 s). H's wall run *is* its token run (it stops at N_H); G is
  re-run capped at N_H. Isolates sample-efficiency.

Artifacts: `experiments/e97_hetero_cma/{lm_verdict.py, aggregate_lm.py,
orchestrate_lm.sh}`, `results/lm_verdict/{H,G,L}_{wall,token}_s*.json`,
`results/lm_verdict/summary.json`.

## 3. Throughput — the penalty actually paid

Sustained fwd+bwd tok/s in the **real LM training loop** (B=2, T=2048, real Pile),
from the clean wall arm (GPUs 94–100 % util, no contention):

| arch | seed | tok/s | vs gdn2-mlp |
|---|---:|---:|---:|
| G (gdn2-mlp) | 0 | 9184 | 0.983 |
| G (gdn2-mlp) | 1 | 9495 | 1.017 |
| **H (hetero)** | 0 | **7433** | **0.796** |
| **H (hetero)** | 1 | **7528** | **0.806** |
| L (LSTM) | 0 | 286 | 0.031 |

**H/G LM-loop throughput = 0.80×.** The kernel microbench in `e97-hetero-cma` was
0.731×; the full training loop is a touch less penalized (0.80×) because the shared
MLP + embedding + data path amortize the sequential split-edit scan. **Either way
the penalty is ~20–27 %, NOT the "≥0.95× / very slight" the task spec assumed.**
The LSTM `level='lstm'` cell is an unfused per-step Python/CUDA scan and is ~32×
slower than gdn2-mlp — it is a *capability* reference, not a throughput-competitive
arm.

## 4. Held-out BPB — both protocols

**WALL-CLOCK-matched (every arm trained 720 s; faster arms see more tokens):**

| arch | seed | tokens | held-out BPB |
|---|---:|---:|---:|
| H | 0 | 5,319,204 | 2.05691 |
| H | 1 | 5,380,674 | 2.06962 |
| G | 0 | 6,544,506 | 2.02619 |
| G | 1 | 6,773,994 | 2.03472 |
| L | 0 | 208,998 | 3.16314 |

mean — **H 2.0633 · G 2.0305 · L 3.1631** → **H loses by +0.033 BPB.**

**TOKEN-matched (every arm trained to N_H = 5.32 M tokens):**

| arch | seed | tokens | held-out BPB |
|---|---:|---:|---:|
| H | 0 | 5,319,204 | 2.05691 |
| H | 1 | 5,380,674 | 2.06962 |
| G | 0 | 5,319,204 | 2.07290 |
| G | 1 | 5,319,204 | 2.08249 |
| L | 0 | 413,898 † | 2.81896 |

mean — **H 2.0633 · G 2.0777 · L (414 K tok) 2.819** → **H wins by −0.014 BPB.**

† The LSTM is so slow (286 tok/s) it could **not reach N_H** within the 2× wall
ceiling (it would need ~5.2 h); its token-arm number is at **414 K tokens**, a loose
reference only, not a matched-token result. Both real cells (H, G) are matched
exactly.

## 5. Reading the result — the sample-efficiency / wall-clock split

This is the **same split** the whole e97 1.3B lineage found, now confirmed for the
final heterogeneous cell on honest footing:

- **At equal tokens, the hetero cell is the better LM** (−0.014 BPB). The 16/64
  split-edit per-step-`tanh` heads add real modeling capacity per gradient step —
  consistent with their depth-capability signature (`e97-hetero-cma` retention
  0.886 vs linear 0.699).
- **At equal wall-clock, gdn2-mlp wins** (+0.033 BPB for H). The 0.80× throughput
  means H sees ~23 % fewer tokens per second; on a compute-bound LM that token
  deficit (5.32 M vs 6.6 M in 720 s) outweighs the per-token edge. The crossover
  is decided by the throughput tax, exactly as in `e97delta-1p3b` (TIE: wins
  token-matched, loses wall-clock) and `e97-wallclock-cma` (NO-GO).
- **The capability head cannot be made fast.** `tanh ⊥ chunkable` is fundamental:
  the bounded saturating state that *creates* the capability forbids the chunked
  tensor-core kernel, so the head is latency-bound at ~0.73–0.80× and stream
  overlap buys ~nothing (`e97-hetero-cma §3`). There is no kernel lever left that
  recovers the lost throughput while keeping the capability.

## 6. Verdict

**Does the heterogeneous cell stay LM-competitive (tie/beat gdn2-mlp on held-out
BPB) at the slight throughput penalty? — NO, on the honest footing.**

1. **Token-matched: TIE / slight win** (H −0.014 BPB). The cell is genuinely more
   sample-efficient.
2. **Wall-clock-matched: LOSS** (H +0.033 BPB). The throughput penalty is real and
   **not slight** — 0.80× LM-loop / 0.73× kernel — so at equal compute gdn2-mlp is
   the better LM.
3. **The task's enabling premise is refuted.** There is no "fused blended ≥0.95×
   GDN-2" capability kernel; that figure was the capability-weak gated-delta shell.
   Paying for the real depth capability costs ~20–27 % throughput.

**Decision: NO-GO for the heterogeneous cell as a drop-in 1.3B LM cell.** gdn2-mlp
wins the wall-clock-matched LM comparison, which is the honest decision criterion.
The hetero cell's value is *capability* (length-extrapolation / depth tasks), not
LM bits-per-byte at fixed compute — downstream consumers (`e97-prog-synth`) should
treat the 16/64 split-edit head as a **capability specialist accepted at a known
~20–27 % throughput cost**, not as an LM-competitive substitute for gdn2-mlp.

## 7. Honest limitations

- **Budget.** Each arm trained ~5–7 M tokens (720 s) — a short-horizon screen, not
  a converged 1.3B run. The token-matched edge (−0.014) and wall-clock deficit
  (+0.033) are both small in absolute BPB; the *sign* (token-win, wall-loss) is the
  robust, lineage-consistent finding, and it is stable across 2 seeds (H 2.057/2.070,
  G wall 2.026/2.035, G token 2.073/2.082 — non-overlapping H-vs-G gaps in each
  protocol).
- **Throughput from the clean wall arm only.** One token-arm G run (seed 1) was
  slowed by transient lane contention (17.2 min wall to the same 5.32 M tokens);
  its **token count is matched so its BPB is valid**, but its wall time is not used
  — the 0.80× penalty comes from the contention-free wall arm (94–100 % util).
- **LSTM is a capability reference, not a matched-throughput arm** (286 tok/s; could
  not reach N_H). It confirms both H and G are real, strong LMs (BPB ~2.03–2.06 vs
  LSTM 2.82 at 414 K tok / 3.16 at 209 K tok).
- **Shared knobs.** H and G use the CMA-best knobs found for the *hetero* cell; this
  is conservative *against* G (G is not separately tuned) yet G still wins
  wall-clock — strengthening the NO-GO.
