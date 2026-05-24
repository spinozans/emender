# QA / Reasoning Eval Sources

## Eval Logs Location

All eval logs live under `~/racer_eval_runs/` (not under `~/elman/`).
The `~/elman/` directory contains the repo code including the eval scripts,
but the output data was written to the directory configured in
`scripts/run_periodic_racer_evals.py` (`--out_root`).

## Eval Run Directories

| Directory | Description | Snapshots |
|-----------|-------------|-----------|
| `~/racer_eval_runs/ctx2k_panel_20260521_initial/` | Initial 4-model baseline snapshot (2026-05-21) | 1 |
| `~/racer_eval_runs/ctx2k_panel_periodic_20260521/` | Periodic 6-hour interval snapshots of all 4 models | 3 |
| `~/racer_eval_runs/ctx2k_fact_reasoning_20260522/` | Single snapshot with both fact (300-item) and reasoning (BBH+ReCLor+FOLIO) panels | 1 |
| `~/racer_eval_runs/ctx2k_panel_2k_20260522/` | Single snapshot with full-size ~2k fact panel | 1 |

## Earlier Knowledge Probe Logs (Built-in 40-item probe)

| File | Notes |
|------|-------|
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_E88.json` | E88 step 705k |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_FLA-GDN.json` | FLA-GDN step 945k |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_M2RNN.json` | M2RNN step 615k |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_Mamba2.json` | Mamba2 step 1266k |
| `/tmp/racer_generations_20260518/knowledge_probe.json` | Earlier NLL-score run (E88 step 702k, FLA-GDN 942k, M2RNN 612k) |
| `/tmp/racer_generations_20260518/knowledge_probe_avg_{model}.json` | avg_nll score, single checkpoint per model |
| `/tmp/racer_generations_20260518/knowledge_probe_mamba2.json` | Mamba2 step 1263k, NLL score |

## Snapshot Manifest Format

Each snapshot directory contains:
- `manifest.json` — model labels, source checkpoint paths, training step, loss
- `eval.json` — full per-item results + summary (JSON list if multiple models)
- `eval.md` — human-readable Markdown report
- `models/` — hardlinked frozen checkpoint + `args.json` per model

## Models and Architecture

All four models are 1.27B-parameter scale, trained on The Pile with 2k context window.

| Label | Level | dim | depth | Notes |
|-------|-------|-----|-------|-------|
| E88   | E88 (NDM) | 1664 | 12 | Primary NDM model |
| FLA-GDN | fla-gdn | 2688 | 21 | Gated linear attention baseline |
| Mamba2 | mamba2 | 2048 | 32 | SSM baseline |
| M2RNN | m2rnn | 1920 | 21 | Matrix-RNN baseline |

## CSV Files in This Directory

| File | Panel | Items | Scope |
|------|-------|-------|-------|
| `knowledge_probe_40item.csv` | Built-in probe | 40 | Earlier steps, 6 categories |
| `racer_panel_300item_progression.csv` | Racer panel (50 items × 6 tasks) | 300 | 4 snapshots per model, progression |
| `fact_panel_latest.csv` | Larger fact panel | ~1.4k–2k | Single latest snapshot |
| `reasoning_panel_latest.csv` | BBH + ReCLor + FOLIO | ~2k | Single latest snapshot |
