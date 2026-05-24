# CMA-tuned FLOPs-per-Bit Convergence Finding (internal)

> **Internal-coordination document.** The paper-facing version of this
> finding is at `paper/results/cma_flop_rate/README.md`. That version is
> written for an external reader who has never heard of CMA-ES, the
> E-series codenames, or the `~/elman/` working tree. This document
> records the same finding **with** the lab vocabulary, repo paths, and
> caveats that we need internally to keep the evidence reproducible and
> to avoid overclaiming. **N = 4 model families; this is a small-N
> finding** — that constraint sits in §6 below, not buried at the bottom.

## 1. Finding in one paragraph

In a coordinated CMA-ES sweep run on 2026-02-01/02 at the ~480M-parameter
target on the local 8×GPU box (`~/elman/benchmark_results/cmaes_converge/`),
four recurrent baselines were independently hyperparameter-optimized on
identical data, with identical fitness, with the same six-knob CMA-ES
search space (width, depth, head count, state width, output gating,
learning rate). When each family's best CMA-tuned config is replayed and
the cumulative training FLOPs are computed from the run logs, the four
families collapse onto a single FLOPs-vs-bits-per-token curve over more
than two decades of FLOPs. At the loose threshold (2.5 bits/tok) the
four FLOPs-per-bit-of-compression values agree to within ~10%. At the
tight end (1.8 bits/tok) the two linear families (FLA-GDN, Mamba2) agree
to within 1% with each other, NDM is at ~1.55× their rate, and vanilla
Elman (E1) never crosses the tight threshold in the budget. The plot is
`paper/results/cma_flop_rate/convergence.{png,pdf}`.

## 2. The four families in the comparison

| Family       | E-series tag | Update rule                                    | Run dir                                                                                  |
|--------------|--------------|------------------------------------------------|------------------------------------------------------------------------------------------|
| NDM          | E88          | `tanh(d·S + k(v − Sᵀk)ᵀ)` — nonlinear delta    | `~/elman/benchmark_results/cmaes_converge/e88_480M_converge0.01_20260201_190214/`        |
| FLA-GDN      | (external)   | linear gated delta-net                         | `~/elman/benchmark_results/cmaes_converge/fla-gdn_480M_converge0.01_20260201_170104/`    |
| Mamba2       | (external)   | linear selective state-space                   | `~/elman/benchmark_results/cmaes_converge/mamba2_480M_converge0.01_20260201_160019/`     |
| vanilla Elman| E1           | dense nonlinear `tanh(W_h h + W_x x)`          | `~/elman/benchmark_results/cmaes_converge/e1_480M_converge0.01_20260201_200250/`         |

The `cmaes_converge` sweep also ran E75 (a hybrid Elman variant) and E23
(slot-based) at the same budget. E23 diverged in every CMA-ES candidate
(`best_loss = 10.0` in `results.json`); E75 reached `best_loss = 1.394`
but is an in-house hybrid that is not part of the paper's baseline
family, so we exclude it from the N = 4 comparison. E42 was attempted in
the same sweep but its run is marked `"success": false` in
`~/elman/benchmark_results/cmaes_converge/summary.json` and its
`results.json` is absent.

**M2RNN is missing from this sweep.** The CMA-ES driver
(`~/elman/cmaes_search_v2.py`) does register both `m2rnn` and
`m2rnn-paper` in its search-space table, but the `cmaes_converge`
benchmark predates the integration. M2RNN at ~480M with full CMA-tuning
is therefore an *open data gap*. M2RNN does appear at 8M-parameter
expressivity scale under
`~/elman/experiments/expressivity_tasks/results/separation_pilot_20260511/`
and is used in `docs/EXPRESSIVITY_RESULTS_SUMMARY.md`. We should flag
this gap in the paper draft where it currently lists "M2RNN-CMA /
M2RNN-paper" as part of the headline LM comparison
(`paper/OUTLINE.md:162`).

## 3. Reproducing the artifacts in-repo

```
cd paper/results/cma_flop_rate
python3 extract.py    # reads ~/elman/, writes trajectory_<model>.csv, overlay.csv, summary.csv, thresholds.csv
python3 plot.py       # writes convergence.{png,pdf}
```

Neither script touches a GPU; both depend only on Python ≥3.10 and (for
`plot.py`) matplotlib. `extract.py` hard-codes the upstream base path
`~/elman/benchmark_results/cmaes_converge/` — update it there if the
external tree moves.

The numbers in §4 below are produced by these scripts.

## 4. The two tables and the plot

### 4.1 Best CMA-tuned config per family

| Family       | CMA-best config                                                  | Params  | Final nats | Bits/tok |
|--------------|------------------------------------------------------------------|---------|------------|----------|
| NDM (E88)    | dim=3840, depth=22, n_heads=70, n_state=16                       | 480.1 M | 1.2451     | 1.796    |
| FLA-GDN      | dim=1792, depth=19, n_heads=12, expansion=2                      | 488.7 M | 1.1403     | 1.645    |
| Mamba2       | dim=1792, depth=25, d_state=128, expand=2                        | 494.2 M | 1.1555     | 1.667    |
| Vanilla Elman| dim=1408, depth=18, expansion=2                                  | 500.0 M | 1.3394     | 1.932    |

Source: `paper/results/cma_flop_rate/summary.csv`; raw fields read from
each run's `results.json::history[*].actual_params` and the matching
`eval_<id>/<run>/args.json`.

### 4.2 FLOPs per bit of compression at matched thresholds

The denominator is `(log2(50257) − target_bits) ≈ (15.617 − target)`;
the numerator is cumulative training FLOPs at the first step where the
100-step-smoothed nats curve crosses `target × ln 2`.

| Family       | bits ≤ 2.50      | bits ≤ 2.00      | bits ≤ 1.80              |
|--------------|------------------|------------------|--------------------------|
| NDM (E88)    | 1.01·10¹⁵        | 2.73·10¹⁵        | 7.11·10¹⁵                |
| FLA-GDN      | 0.99·10¹⁵        | 1.45·10¹⁵        | 4.60·10¹⁵                |
| Mamba2       | 0.94·10¹⁵        | 1.42·10¹⁵        | 4.65·10¹⁵                |
| Vanilla Elman| 1.03·10¹⁵        | 2.25·10¹⁵        | (not reached in budget)  |

Source: `paper/results/cma_flop_rate/thresholds.csv`.

### 4.3 The convergence plot

`paper/results/cma_flop_rate/convergence.pdf` (also `.png`):

- Panel A — bits-per-token vs cumulative FLOPs (log-x). All four curves
  share a slope; the linear pair (FLA-GDN, Mamba2) sits ~0.15 bits below
  the nonlinear pair through the bulk of training.
- Panel B — running FLOPs / (bits saved vs uniform baseline), log-log.
  The four curves visually collapse onto a single line over more than
  two decades of FLOPs. This panel is the strongest single visual form
  of the finding.

## 5. CMA-ES methodology (matches the elman repo)

- **Driver:** `~/elman/cmaes_search_v2.py` (`run_cmaes_v2.py` orchestrator).
- **Search dimensions (per family):** width, depth, head count, state width,
  use_gate (binary), learning rate (log-scale). Total 6D.
- **Population:** 16 per generation. **Initial seeding:** 64-sample Latin
  hypercube to populate the covariance. **Refinement:** CMA-ES generations
  until either 12 generations have run or three consecutive generations
  fail to improve best-loss by 0.002.
- **Per-candidate training:** 30-minute wall-clock on a single GPU,
  schedule-free AdamW, batch 8, chunk 512, bf16. Tokens-per-step = 4 096.
  Fitness = mean nats over the last 100 logging steps; rejected if NaN
  or > 10.
- **Hardware:** 8× consumer/datacenter GPUs (mix; see
  `~/elman/HANDOFF_CMAES_OPTIMIZATION.md`). The fitness budget is
  symmetric across families because the same scheduler (`run_all_cmaes_v2.py`)
  was used.
- **Param-target tolerance:** ±10% around 480M. Configurations outside
  the tolerance after `find_dim_for_params` resolution were rejected.

## 6. N = 4 caveat — keep this prominent in the paper section

"N = 4" is **four model families**. It is *not* four seeds, four CMA-ES
generations, four parameter budgets, or four data sources. Each curve in
Panel B of `convergence.pdf` carries single-seed noise plus
single-best-CMA-eval noise, neither of which is separately quantified
here. The four families also do not cover the recurrent-design space
densely — there is one nonlinear matrix-state design (NDM/E88), one
linear matrix-state delta design (FLA-GDN), one linear scalar-state
selective SSM (Mamba2), and one nonlinear vector-state baseline
(vanilla Elman). The most obviously missing comparator is a nonlinear
matrix-state design *without* delta correction (M2RNN), which would
isolate the contribution of the delta term itself; that is the M2RNN
data gap noted in §2.

## 7. Connection to the paper narrative

This finding backs the paper's reframed contribution claim. The
original draft phrasing emphasized training speed; the data here does
not support a speed claim — at matched HPO, NDM/E88 is in the same
FLOPs-per-bit band as the leading linear-recurrent baselines but is not
the fastest of them. What the data does support is the *architectural-
option* framing in `paper/ndmpapernotes.md` §"CMA-ES And Geometry
Search":

> Once recurrent geometry and kernels are optimized, pure nonlinear
> recurrence trains in the same wallclock loss band as strong linear-
> recurrent baselines. NDM's value is the architectural option (you
> *can* train pure nonlinear recurrence at scale) plus the
> delta-correction mechanism (expressivity), not raw training speed.

The two open ends:

- **Expressivity carries the asymptotic-capability claim.** The
  expressivity section (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md`) is what
  separates NDM from FLA-GDN/Mamba2 at *any* compute; the FLOPs-per-bit
  finding only says that the cost of acquiring that separation is
  bounded.
- **The M2RNN comparator is the cleanest experiment to add next.** A
  ~480M CMA-tuned M2RNN run, dropped into this same protocol, would let
  us pin down whether the delta term is what brings NDM down to the
  shared FLOPs-per-bit line, or whether nonlinear matrix state alone is
  sufficient.

## 8. Where this finding shows up in the paper outline

- `paper/OUTLINE.md` §2.5 "Language Modeling Results — 1.27B Comparison"
  currently cites "wallclock loss / bits-per-byte" as the headline
  metric. The FLOPs-per-bit framing here is a tighter, parameter-budget-
  controlled version of that metric and should replace or complement the
  wallclock framing in the headline figure caption.
- `paper/OUTLINE.md` §2.5 lists `M2RNN-CMA, M2RNN-paper` as part of the
  comparison; that listing is currently aspirational with respect to
  the CMA-tuned ~480M data — flag in the next reconciliation pass.
- `paper/notes_reconciliation.md` row C2 marks the CMA-ES geometry
  story as *partial* and notes the search code lives in `~/elman/`. The
  `paper/results/cma_flop_rate/` directory created by this task is the
  in-repo artifact that closes the data side of that gap, with
  `SOURCES.md` recording the upstream paths.

## 9. Validation checklist for this task

- [x] CMA search artifacts located in `~/elman/`; paths recorded in
  `paper/results/cma_flop_rate/SOURCES.md`.
- [x] At least 3 model families have CMA-tuned loss-vs-FLOPs curves
  committed (this finding has 4: NDM, FLA-GDN, Mamba2, vanilla Elman).
- [x] N = 4 caveat is in the title-area of both the paper-facing README
  and this internal document; it is *not* buried in a footnote.
- [x] CMA methodology described in §5 (six-knob 6D search, LHS-then-CMA-ES,
  population 16, ~30-min fitness, schedule-free AdamW, bf16, ~480M target).
- [x] Convergence plot exists at `paper/results/cma_flop_rate/convergence.{png,pdf}`
  and visually supports the claim (Panel B is the load-bearing visual).
- [x] No raw checkpoints copied — only CSV, JSON-free summaries, PNG,
  and PDF.
