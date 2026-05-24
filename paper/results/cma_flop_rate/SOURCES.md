# Sources for the CMA-tuned FLOPs-per-bit Finding

All upstream data files live outside this repository, in `~/elman/`. This file
records exactly which run directories were read and which fields were used,
so the extraction can be reproduced if the source artifacts move.

## Primary cross-model run: `cmaes_converge` sweep

A coordinated CMA-ES sweep run on 2026-02-01/02 at a target of 480M parameters,
~1 hour wall time per architecture, on the local 8×GPU box. This is the only
dataset in which the four recurrent baselines below were CMA-tuned under
matched hyperparameter budget, matched data, matched optimizer, and a single
fitness-extraction protocol.

Base path: `~/elman/benchmark_results/cmaes_converge/`

| Model family | Run directory                                   | best `eval_id` | best loss (nats) | actual params |
|--------------|-------------------------------------------------|----------------|------------------|---------------|
| NDM (E88)    | `e88_480M_converge0.01_20260201_190214/`        | 10             | 1.2451           | 480,076,392   |
| FLA-GDN      | `fla-gdn_480M_converge0.01_20260201_170104/`    | 24             | 1.1403           | 488,707,072   |
| Mamba2       | `mamba2_480M_converge0.01_20260201_160019/`     | 13             | 1.1555           | 494,154,752   |
| E1 (Elman)   | `e1_480M_converge0.01_20260201_200250/`         | 16             | 1.3394           | 500,018,816   |

For every model the per-config training log is `eval_<id>.log` (one step-line
emitted every `log_every = 10` optimizer steps). The exact training arguments
used by the best config are at `eval_<id>/<run_subdir>/args.json` — the
extraction reads `batch_size`, `chunk_size`, and the `lr` field from that JSON.

## Fields actually consumed

For each run the extractor reads:

- `results.json`
  - `model_type` — model-family tag
  - `best_loss` — final-window mean loss in nats (sanity check vs trajectory)
  - `history[*].eval_id`, `history[*].loss`, `history[*].actual_params`,
    `history[*].params` — to pick the best non-diverged eval and report its
    architecture knobs
- `eval_<best>.log`
  - regex `step\s+(\d+)\s*\|\s*loss\s+([\d.]+)` — extracts the (step, raw-loss)
    sequence; entries with NaN or loss > 50 (early-warmup spikes) are dropped
- `eval_<best>/<run>/args.json`
  - `batch_size`, `chunk_size` — tokens-per-step = `batch_size × chunk_size`
    (typically 8 × 512 = 4 096)

The fitness function used by the CMA-ES sweep itself is in
`~/elman/cmaes_search_v2.py::extract_loss` — average over the last 100 logged
steps, with NaN/divergence rejection.

## What was *not* used (and why)

- `~/elman/benchmark_results/cmaes_converge/e75_*/` and `e23_*/` —
  hybrid Elman variants with internal codenames; not part of the paper's
  N=4 baseline family.
- `~/elman/benchmark_results/cmaes_converge/e42_*/` — search marked
  `success: false` in `summary.json`; `results.json` is missing.
- M2RNN — `~/elman/elman/models/m2rnn_baseline.py` exists and M2RNN is wired
  into the CMA-ES search spaces (`~/elman/cmaes_search_v2.py` registers
  `'m2rnn'` and `'m2rnn-paper'` under `_E88_SEARCH_SPACE`), but no
  ~480M CMA-tuned M2RNN run is present in `~/elman/benchmark_results/`.
  M2RNN does appear at the smaller expressivity-task scale under
  `~/elman/experiments/expressivity_tasks/results/separation_pilot_20260511/`
  (8M parameters, controlled-task probes, not LM training). Including
  M2RNN in this FLOPs-per-bit finding would require a fresh CMA-tuned
  ~480M run; that work was explicitly listed as out-of-scope for this task.

## Methodology reference (CMA-ES search itself)

- `~/elman/cmaes_search_v2.py` — production CMA-ES driver (two-phase: LHS
  exploration → CMA-ES refinement). 6-dimensional search space per family.
- `~/elman/run_cmaes_v2.py` — invocation harness; specifies population 16,
  sigma 0.35, min-generations 12, converge threshold 0.002 over 3 consecutive
  generations.
- `~/elman/run_all_cmaes.py`, `~/elman/run_all_cmaes_v2.py` — orchestrator
  used to launch the cross-model sweep.
- `~/elman/HANDOFF_CMAES_OPTIMIZATION.md` — narrative handoff describing
  why CMA-ES was adopted and what the original search budget targeted.
- `~/elman/cmaes_logs/` — earlier (200M / 500M) E88-only CMA-ES runs, not
  used here because they are single-family.

## Vocabulary size assumption

Bits-per-token are computed as `nats / ln(2)`. The FLOPs-per-bit-of-
compression ratio additionally needs a baseline; the uniform-vocab
baseline `log2(50257) ≈ 15.617 bits/token` is used, matching the p50k_base BPE
vocabulary used by the underlying trainer (see `~/elman/data/pile_1mb.txt
.p50k_base.tokens.npy` and the `cmaes_search_v2.py` token pipeline).

## FLOPs convention

`FLOPs ≈ 6 · N_params · tokens_processed` (Kaplan/Hoffmann/Chinchilla
approximation for autoregressive LM training: 2 FLOPs per param for the
forward pass, ~4 for the backward+update, totaling 6 per token per param).
For these short ~1-hour runs the linear approximation is well within the
uncertainty introduced by smoothing.

## Reproducing the artifacts in this directory

```
cd paper/results/cma_flop_rate
python3 extract.py    # writes trajectory_<model>.csv, overlay.csv, summary.csv, thresholds.csv
python3 plot.py       # writes convergence.png and convergence.pdf
```

No GPU is needed; the extractors only parse the upstream logs.
