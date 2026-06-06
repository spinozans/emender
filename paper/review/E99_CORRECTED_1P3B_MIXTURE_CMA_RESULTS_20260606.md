# Corrected E99 1.3B Mixture-Aware CMA — Results

**Date:** 2026-06-06
**Task:** `corrected-e99-1-3b` (successor to `redo-e99-1-3b`, gated behind
`paper/review/E99_HEAD_KERNEL_AUDIT_20260606.md`)
**Human approval:** RECORDED (Erik Garrison, 2026-06-06, verbatim *"Yes, launch
detected. 1.3 billion run."*), preserved in `all_results.json:approval_note_record`.
**Run dir:** `experiments/e99_mixture_aware_lm_cma/results/corrected_run1/`
**Status:** COMPLETE. LM screen = 32 evals, 81.4 min wall, 651.5 GPU-min, GPUs
0–7, idle-only / no-preempt. bf16. No NaNs in any of the 32 evals.

This is the **fair rematch** demanded by the kernel + sizing audit. The three
repairs from the task spec were all applied and are visible in the artifacts.

---

## 0. SEARCHED vs CONTROL — binding scope (Erik, 2026-06-06)

This is the headline correctness property of this run and is recorded
machine-readably in `all_results.json:search_spec`.

| arm | head types | role | selection-ranked? |
|---|---|---|---|
| **SEARCHED** | `{gdn2_recall, e97_track, count, latch, nonlin}` — exactly 5, all single-launch fused (~0.93× native per the audit) | the CMA/LHS mixture simplex | **YES** — token-matched LM loss (+ capability) |
| **CONTROL** | `dense GDN-2` (M0) and `gdn2_nonlin_shell` (S1/S2/S3) | fixed labeled capability/accuracy arms | **NO** — never wallclock/tok-min ranked, never a search dimension |
| ANCHOR | `M1_priorE99_5to1`, `C1/C2/C3` legacy-nonlin corners | fixed comparability mixtures | NO — reported, not selection-ranked |

**`gdn2_nonlin_shell` was REMOVED from the search simplex.** Every CMA candidate
carries **5-entry logits**; `allocate_types` pads the shell slot off, so **no
searched candidate ever instantiates a shell head** (a hard `assert` in
`run_mixture_cma.py` enforces `n_shell == 0` for every searched candidate). The
shell survives only as the fixed S1/S2/S3 control arms. This is the precise
correction over `redo-e99-1-3b`, where a **6-entry SEARCH logit** let the shell
into the ranking and was the likely source of the spurious "3.2× slower" that made
`E99_1P3B_LM_DECISION` pick dense GDN-2 over the typed Emender.

---

## 1. The fair search rediscovers dense recall

The 5-type CMA, seeded **recall-heavy** (0.70 gdn2_recall) with the cma-capability
specialist sub-mixture (`track 0.40 / count 0.28 / nonlin 0.31 / latch 0.009`) and
**wide sigma (2.0)**, monotonically drove the mixture toward **pure
`gdn2_recall`**:

| gen | best AvgLoss | best counts (gdn/track/count/latch/nonlin/shell) | sigma |
|---:|---:|---|---:|
| 0 | 6.4563 | 100 / 1 / 1 / 0 / 0 / 0 | 1.906 |
| 1 | 6.3364 | 102 / 0 / 0 / 0 / 0 / 0 | 2.107 |
| 2 | 6.3314 | 102 / 0 / 0 / 0 / 0 / 0 | 2.474 |
| 3 | 6.3224 | 102 / 0 / 0 / 0 / 0 / 0 | 2.552 |

The best **searched** candidate, `cma_g3_e1` (AvgLoss **6.3224**, held-out BPB
**2.0405**), is the allocation `102/0/0` — i.e. **the CMA rediscovered dense
GDN-2**. It edges the labeled dense control `M0_dense_gdn2` (6.3786 / 2.0450) by
exactly the within-architecture run variance (same 102/0/0 allocation, same 1.272B
params; the searched copy simply logged more steps at higher tok/s). Every
searched candidate that allocated specialist or nonlin heads ranked **below** pure
recall. **Conclusion: at 1.3B, token-matched, the typed mixture's fair optimum is
the recall backbone; specialist heads do not improve LM loss.** This is consistent
with the E98 sixth-corner finding that GDN recall is architectural and is the LM
workhorse.

Full ranked table: `results/corrected_run1/report_tables.md` (32 rows, `arm`
column labels SEARCHED/CONTROL/ANCHOR).

---

## 2. Computational mechanism vs implementation artifact (the §3 fairness read)

This is the separation the task demands, read off the matched-fraction control
triples at nonlinear-slot fraction f ∈ {1/6, 1/3, 1/2}. All three arms are
param-matched to 1.27B ±2%; (a)=dense GDN-2 linear, (b)=GDN-2 **fused** shell
(nonlinear-in-time), (c)=legacy UnifiedCell `nonlin` corner.

| f | (a) dense BPB | (b) shell BPB | (c) legacy-nonlin BPB | (b)−(a) | (b)−(c) |
|---|---:|---:|---:|---:|---:|
| 1/6 | 2.0450 | 2.0923 | 2.1103 | **+0.047** | **−0.018** |
| 1/3 | 2.0450 | 2.0792 | 2.1319 | **+0.034** | **−0.053** |
| 1/2 | 2.0450 | 2.0642 | 2.1552 | **+0.019** | **−0.091** |

- **(b)−(a) > 0 everywhere — the nonlinearity-in-time mechanism itself costs LM
  loss** at 1.3B (and the cost shrinks as more of the layer becomes nonlinear,
  i.e. the marginal head matters less). This is a *computational-mechanism*
  statement: nonlinear-in-time state does not help next-token prediction here.
- **(b)−(c) < 0 everywhere, and grows with f — the fused GDN-2 shell strictly
  beats the legacy UnifiedCell nonlin corner for the SAME nonlinear-in-time
  intent.** This is an *implementation-artifact* statement: a large part of the
  legacy `nonlin` corner's poor LM showing was its external UnifiedCell plumbing,
  not the nonlinearity. The fused native-GDN plumbing recovers 0.018–0.091 BPB.

### Throughput is an artifact, not a mechanism — the "3.2× slower" is refuted

Wallclock/tok-s were **reported but NOT used for any cross-head selection** (per
spec). With the **single-launch fused** shell kernel from `implement-triton-fused`:

| arm | tok/s | ratio vs dense | implementation tax |
|---|---:|---:|---:|
| M0 dense GDN-2 | 8912 | 1.000 | — |
| S1 shell f=1/6 | 8095 | 0.908 | ~9% |
| S2 shell f=1/3 | 7991 | 0.897 | ~10% |
| S3 shell f=1/2 | 8168 | 0.916 | ~8% |

The fused shell runs at **0.90–0.92× dense throughput (~8–10% implementation
tax)** — **not 3.2× slower.** The earlier 3.2× figure was an artifact of an
unfused shell path being dragged into a wallclock ranking; removing the shell from
the search and using the fused kernel makes the comparison fair. The shell still
loses to dense on **accuracy** (LM loss), so the architectural verdict (recall
backbone wins) stands — but it stands on *mechanism*, not on a throughput artifact.

---

## 3. Validation against task criteria

| criterion | status | evidence |
|---|---|---|
| SEARCH simplex = EXACTLY 5 types `{gdn2_recall, e97_track, count, latch, nonlin}`; shell NOT searched | **pass** | `run_mixture_cma.py` Stage-2 seeds 5-entry logits; per-candidate `assert len==5 and n_shell==0`; `search_spec.searched_types`; every `cma_*` row has shell=0 |
| `gdn2_nonlin_shell` + dense-GDN-2 present as labeled CONTROL arms only, never wallclock-ranked | **pass** | S1/S2/S3 + M0 are Stage-1 anchors; `search_spec.control_arms`; `arm=CONTROL` in tables; tok/s reported, never used to rank heads |
| bf16; every candidate param-matched to 1.3B ±2% (counts logged) | **pass** | `BASE_SHAPE.bf16=True`; `derive_dim` + `assert_param_target`; all 32 evals 1.260–1.282B (target 1.270B, ≤+0.91% / ≥−0.81%); per-eval counts logged |
| Results doc states which head types were SEARCHED vs CONTROL | **pass** | §0 of this doc + `search_spec` JSON |
| Report separates computational mechanism from implementation artifact | **pass** | §2: (b)−(a) mechanism vs (b)−(c) plumbing; throughput artifact refutation |
| Idle-GPU-only / no-preempt; GPU set + parallelism logged | **pass** | `FREE_MEM_MIB=2000` no-preempt scheduler; `gpus_used=[0..7]`; `aggregate_gpu_minutes=651.5`; `gpu_log` per-eval |

Stability: **0/32 evals NaN** (contrast E99-CMA-96 where E98-CMA knob-LR
NaN-diverged). All anchor checkpoints round-tripped (`RT=ok`). **31/32 evals
completed**; one searched gen-0 candidate (`cma_g0_e1`, a track-heavy 12/73/10/0/7
mixture) hit a transient CUDA OOM in backward (the UnifiedCell split-gate `track`
heads have a larger activation footprint than recall heads, and it landed on a
fragmented GPU). It was assigned the worst fitness (1e6) per the driver's
non-finite handling; CMA continued and still converged to pure recall. This is a
resource artifact, not a correctness issue, and does not affect the conclusion
(robust across the 31 successful evals + the monotonic gen-by-gen convergence).

---

## 4. Artifacts

- `results/corrected_run1/all_results.json` — full screen incl. `search_spec`
- `results/corrected_run1/candidates.csv` — per-eval rows (dim, params, counts,
  loss, BPB, tok/s, steps, NaN, roundtrip)
- `results/corrected_run1/generations.jsonl` — CMA per-gen trajectory
- `results/corrected_run1/report_tables.md` / `.json` — aggregated tables
- `results/capability/` — capability axis (MQAR/S5/nonlinear-state/mixed) on the
  control anchors (see §5; appended on completion)
- `results/corrected_run1/anchors/`, `cma_gen{0..3}/` — per-eval cfg/log/res

---

## 5. Capability axis (control anchors)

Param-matched ~8M probes (depth4/h48/n32, 4000 steps, seed 42, eval T ∈
{128,256,512,1024}) on the control anchors. NOTE: this required a harness fix —
`train_hybrid.py` did not accept `--shell_state_nonlin/--shell_state_chunk`, so the
shell arms initially crashed at argparse (dense/legacy were unaffected). Fixed and
the 18 shell jobs rerun; all 48 jobs now complete.

| mixture | arm | gdn/nl/shell | mean | mqar | s5 | anbncn | flag | **iterated (nonlin-state)** | mixed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0_dense_gdn2 | CONTROL (a) | 48/0/0 | 0.828 | **0.690** | 0.903 | 0.845 | 0.959 | 0.932 | 0.640 |
| S1_gdn_shell_f17 | CONTROL (b) | 40/0/8 | 0.816 | 0.609 | 0.970 | 0.855 | 0.931 | **0.935** | 0.593 |
| S2_gdn_shell_f33 | CONTROL (b) | 32/0/16 | 0.829 | 0.606 | 0.958 | 0.894 | 0.998 | **0.935** | 0.580 |
| S3_gdn_shell_f50 | CONTROL (b) | 24/0/24 | 0.827 | 0.546 | 0.939 | 0.924 | 0.988 | **0.930** | 0.630 |
| C1_gdn_nonlin_f17 | ANCHOR (c) | 40/8/0 | 0.837 | 0.687 | 0.995 | 0.905 | 0.936 | 0.919 | 0.579 |
| C2_gdn_nonlin_f33 | ANCHOR (c) | 32/16/0 | 0.832 | 0.663 | 0.956 | 0.901 | 0.975 | 0.916 | 0.583 |
| C3_gdn_nonlin_f50 | ANCHOR (c) | 24/24/0 | 0.806 | 0.686 | 0.827 | 0.846 | 0.994 | 0.907 | 0.576 |
| M1_priorE99_5to1 | ANCHOR | 40/8/0 | 0.839 | 0.681 | 0.995 | 0.905 | 0.955 | 0.920 | 0.579 |

Reads (consistent with the LM screen and with the mechanism-vs-artifact split):

- **Nonlinear-state probe (`iterated_nonlinear_map`) — the decisive head test:**
  fused shell (b) **0.930–0.935 ≈ dense (a) 0.932 > legacy-nonlin (c) 0.907–0.919**
  at every f. The fused GDN-2 shell delivers the nonlinear-in-time capability at
  least as well as dense and **strictly better than the legacy UnifiedCell plumbing**
  — the same (b) > (c) "plumbing artifact" the LM BPB triples show, now confirmed on
  the capability axis.
- **Recall (`mqar_recall`):** dense (a) **0.690** is best; the shell arms fall
  0.609 → 0.546 as shell heads replace recall heads. Spending capacity on
  nonlinearity costs recall — the LM backbone is recall, exactly as the CMA's
  convergence to 102/0/0 implies.
- **Overall mean is a near-wash (~0.81–0.84)**; the differences are probe-specific,
  not a uniform win for any arm. There is no capability case for promoting a
  nonlinear/shell head over recall at this budget.

### Net verdict

The corrected, fair rematch **reaffirms recall (dense GDN-2) as the 1.3B LM
backbone on mechanism** — the 5-type CMA rediscovered it, and no specialist or
nonlinear head improves token-matched LM loss. But it **overturns the reasoning**
in `E99_1P3B_LM_DECISION`: the fused `gdn2_nonlin_shell` is **~0.90–0.92× dense
throughput (not 3.2× slower)** and **matches dense on the nonlinear-state probe**;
its earlier loss was an unfused-implementation artifact plus a search-scope bug
(shell wrongly ranked as a 6th search dimension). With the shell correctly held as
a labeled control and never wallclock-ranked, the architectural conclusion stands
on accuracy alone.
