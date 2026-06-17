# M2 (multi-query-read E97) — CMA-ES at 1.3B, homologous leaderboard placement

**Task:** `cmaes-m2-1p3b`. Run CMA-ES on the **M2** architecture (rank-R multi-query
readout of the E97 split-edit linear-attention state, `paper/review/M2_MULTIQUERY_KERNEL.md`)
using the **exact same protocol** as the emender-mlp / gdn2-mlp / pure-E97 / m2rnn
1.3B searches, so M2's best config is directly comparable on the `docs/SCALE_PLAN.md`
§1 leaderboard — the "relatively speaking, how good is it?" question.

## TL;DR — NO-GO

**M2 places LAST on the §1 leaderboard, and the multi-query rank knob R is statistically
null at matched capacity.**

| Arm | Cell / level | Geometry | Params | **Search avg-loss** |
|---|---|---|---:|---:|
| emender-mlp | E97-delta + SwiGLU MLP | dim1792 nh216 ns32 dep11 mlp2.26 | 1286.6 M | **5.8606** |
| gdn2-mlp | GDN-2 + SwiGLU MLP | dim2176 nh30 dep12 mlp3.259 | 1286.7 M | 5.8949 |
| pure-E97 | E97 split-edit **raw-write**, no MLP | dim2432 nh416 ns16 dep10 | 1265.6 M | 5.9511 |
| m2rnn | M2RNN matrix RNN | dim3072 nh346 ns16 dep13 | 1275.0 M | 6.0636 |
| **M2 (e97-m2)** | E97 split-edit **delta**, chunked-linear, **R-query readout**, no MLP | dim2816 nh227 ns16 dep10 **R=3** | **1273.1 M** | **6.1843** |

- **R\* = 3** (the rank chosen by the CMA-best candidate).
- **Params = 1,273.1 M** (estimate; real build of the best geometry == estimate to < 1e-4).
- M2's best is **+0.121 worse** than the next-worst arm (m2rnn) and **+0.234 worse** than its
  closest sibling pure-E97.

## Protocol — homologous (identical to `scripts/repro_cmaes_1300m`)

`scripts/cmaes_search_v2.py --model e97-m2 --phase cmaes --params 1300M
--param_tolerance 0.03 --train_minutes 15 --popsize 8 --sigma 0.8 --chunk_size 2048
--tokenizer p50k_base --data .../pile.txt --min_generations 12 --use_triton_e88
--anchor_only_cmaes` (anchor = pure-E97 winner basin shrunk to ~1.3B at R=2).

The **only** difference vs the pure-E97 (`e97`) search is the model_type `e97-m2`,
whose search space adds **one axis — `multiquery_r` (R, 1..8)** — to the same E97
geometry axes (dim, n_heads, n_state{16,32}, depth, lr, batch). expansion pinned 1.0;
bf16 + FUSED throughout (E97-M2 forces the Triton chunked split-edit kernel + the
`[fused-guard] ... NO eager fallback` assert).

- **Converged:** 12 generations, **96/96 evals finite** (0 failed/NaN/inf), 13.48 h on 2
  leased GPUs (broker).
- **R explored across the full range:** R=1:14, R=2:29, R=3:30, R=4:16, R=5:4, R=6:3 evals.
- Curves/results committed: `experiments/cmaes_m2_1p3b_20260616/e97-m2/.../{generations.jsonl,results.json}`.

## The multi-query knob R is NULL (iso-geometry control, 3 seeds)

The CMA's per-R best is flat and confounded by **unequal sampling** (R=1 got 14 evals,
R=3 got 30):

| R | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| CMA best-per-R avg-loss | 6.2424 | 6.2120 | **6.1843** | 6.2242 | 6.1921 | 6.3161 |

To remove the geometry **and** sampling confounds, an iso-geometry control fixes the
basin (dim2816 / ns16 / dep10, the CMA-best shape), the optimizer (lr 7.1e-4, bs 2),
varies **only** R, adjusts n_heads per R to hold params at ~1.3B, and runs **3 seeds**
(`experiments/cmaes_m2_1p3b_20260616/iso_geometry_R_control/summary.json`):

| R | n_heads | params | mean avg-loss | std | min | vals |
|---|---:|---:|---:|---:|---:|---|
| 1 | 364 | 1299.9 M | 6.2618 | 0.0253 | 6.2268 | 6.2268 / 6.2858 / 6.2729 |
| 2 | 284 | 1301.3 M | 6.2315 | 0.0263 | 6.1996 | 6.2308 / 6.2641 / 6.1996 |
| 3 | 232 | 1298.0 M | 6.2292 | 0.0398 | 6.1968 | 6.1968 / 6.2055 / 6.2853 |
| 4 | 197 | 1301.1 M | 6.2269 | 0.0400 | 6.1887 | 6.2099 / 6.1887 / 6.2821 |

- R1→R4 mean difference = **0.035 < within-R seed std (0.025–0.040)**; distributions
  overlap heavily (R1's best 6.227 beats R3/R4's worst 6.285/6.282). Two-sample
  t-test R1-vs-R4: **t ≈ 1.3, p ≈ 0.27 — not significant.**
- Throughput is **flat across R** (1450–1530 steps / 6.9–7.3 k tok/s in 15 min): higher
  R does not cost steps at matched capacity (n_heads shrinks to offset the readout cost;
  consistent with the impl's measured sublinear-R throughput).

**Conclusion: rank-R multi-query readout provides no benefit over single-query (R=1) at
matched 1.3B capacity.** The CMA's nominal R\*=3 is within seed noise of R=1.

## NULL discipline — confound audit (every check cleared)

A NO-GO must survive the same audit as a positive. It does:

1. **Capacity** — all configs 1.27–1.30 B (real build == estimate to < 1e-4), within ±3 %
   tolerance; M2-best 1273.1 M is in the same band as the leaderboard arms (1265–1287 M).
   Not under-capacity.
2. **Eager-vs-fused** — **96/96** search evals and **12/12** control runs print the
   `[fused-guard] level=E97-M2 bf16 use_triton=1 -> ... NO eager fallback`; the guard is a
   hard assert and the forward hard-imports the Triton kernel (import failure raises), so a
   finite loss is proof the fused path ran. NON-NEGOTIABLE #1 satisfied for every run.
3. **Aggregator** — fitness is the mean loss over **all** trajectory steps
   (`parse_average_loss`), identical to the leaderboard metric (`aggregate_cmaes_leaderboard.py`).
   Same metric for search and control.
4. **Wrong substrate** *(caveat, honestly reported)* — M2 is built on the **chunked-linear
   split-edit DELTA** kernel (the throughput-viable fused path mandated by NON-NEGOTIABLE #1
   and chosen by `impl-m2-multiquery`). This differs from the leaderboard's best E97
   (pure-E97 = **raw-write, sequential-nonlinear**). So M2's base cell (R=1 ≡ single-query
   chunked-linear delta) is itself ~0.23 weaker than e97-raw at the 15-min budget — but this
   is the *defined* M2 architecture; the multi-query question is answered on that substrate.
   A tanh-state M2 arm is possible (impl note) but unbuilt; it is a separate experiment.
5. **Precision** — bf16 uniform throughout (search and control).
6. **Iso-param** — R searched jointly with geometry under the 1.3 B constraint in the CMA,
   and independently isolated by the iso-geometry control.
7. **Search-convergence / unequal-sampling** — resolved by the iso-geometry control: the
   CMA's R\*=3 (30 evals) is within seed noise of R=1 (14 evals) once geometry and sampling
   are matched.

Prior internal nulls (nlmem / refit / TTT) were **not** used as a reason to expect this
null; the verdict rests only on this run's measured data.

## Honest nuance (not a confound, reported for completeness)

M2's *trajectory-average* loss (~6.18–6.26) runs ~0.4 above its *final* last-100 loss
(~5.79–5.92). The 15-min trajectory has not plateaued, so the average is dragged up by
early high-loss steps — M2 is slightly less sample-efficient early than the faster
leaderboard cells. On the homologous **search avg-loss** metric (the one the leaderboard
uses) M2 is unambiguously last; on a final-loss basis it is closer to the pack but still
does not lead. The leaderboard verdict uses the homologous metric.

## Files

- `scripts/cmaes_search_v2.py` — `_E97_M2_SEARCH_SPACE` (E97 axes + `multiquery_r` 1..8 as a
  real CMA axis); `estimate_params_for_config`/`build_train_command` consume `params['multiquery_r']`.
- `experiments/cmaes_m2_1p3b_20260616/anchors_e97_m2.json`, `launch_e97_m2.sh` — homologous launch.
- `experiments/cmaes_m2_1p3b_20260616/e97-m2/.../{generations.jsonl,results.json}` — search curves/best.
- `experiments/cmaes_m2_1p3b_20260616/iso_geometry_R_control.py`, `.../summary.json` — R confound control.
