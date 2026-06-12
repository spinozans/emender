# lb-m2rnn2 — CLEAN 1.3B CMA control for the M2RNN baseline (MEASURED data)

Re-run of **m2rnn** (matrix-to-matrix nonlinear RNN, Mishra/Tan/Stoica/Gonzalez/Dao)
through the **standard** driver `scripts/cmaes_search_v2.py` at 1.3B with the
symmetric leaderboard budget, on the SAME data slice as the `lb-gdn2-mlp` sibling
(`pile.txt`, seed 42) so the avg-loss is directly comparable to the OLD leaderboard.

**Supersedes `lb-m2rnn`**, which was marked "done" 25 s after launch (premature) and
then ran CONTENDED with `emender-mlp` (corrupted). This run owned the **WHOLE 8-GPU
box** with no concurrency, and was reported done ONLY after the search exited cleanly
(`done exit=0`) and results were committed.

## Protocol (standard driver, symmetric to the gdn2-mlp / gdn2 / e97 / emender 1.3B arms)

- Driver: `scripts/cmaes_search_v2.py --model m2rnn` → `--level m2rnn`. The released
  M2RNN cell runs through the **XMA fused Triton backend** (`--require_m2rnn_xma`,
  built into the driver's m2rnn worker command; `XMA_PATH=/home/erikg/xma` auto-set).
  **NO eager fallback** — `--require_m2rnn_xma` makes the worker FAIL rather than run
  the per-step scan on CUDA.
- Searched axes (native `_E88_SEARCH_SPACE`): `dim` (1024–4096, ×128), `n_heads`
  (32–2000, int), **`n_state` SWEPT as a free CMA dimension over {16,32}** (`e88_n_state`
  snap — NOT pinned), `depth` (10–50), `lr` (1e-4–3e-3 log), `batch_size` (1–128 log,
  clamped by memory probe). `expansion` pinned 1.0 (square state), `use_gate=1`.
- **Precision/kernel: bf16 uniform (worker `--bf16`) + XMA FUSED Triton kernel, NO eager.**
  Asserted from the live BEST-eval worker (`bf16_xma_fused_assert.txt`, eval_35):
  `level=m2rnn`, `bf16=true`, `require_m2rnn_xma=true`, `expansion=1.0`, and worker
  stdout `M2RNN XMA Triton backend: True`.
- Data slice: SAME `--data /home/erikg/elman/data/pile.txt`, `--seed 42`,
  `--tokenizer p50k_base` (vocab 50281) as the gdn2-mlp sibling → numbers directly
  comparable to the OLD leaderboard.
- CMA: **popsize 8, min_generations 13** (104 evals, matching the old leaderboards'
  96–109), **sigma 0.8**, params **1300M**, **param_tolerance 0.03**, **chunk_size
  2048**, **train_minutes 15**. 8 GPUs (one candidate per GPU), leased via
  `scripts/gpu_lease.sh 8`. Single anchor = the old XMA long-racer geometry
  `dim1920/nh370/n16/dep21` (= 1.307B, the geometry behind the old leaderboard 6.1161);
  `--anchor_only_cmaes`. n_state warm-starts at 16 but remains a free CMA dimension.

## Result (MEASURED) — CONVERGED, 13 gens / 104 evals / 4.66 h

- **Best avg-loss: 6.0636**  (eval_35; found at generation 5, then held FLAT through
  generation 13 — 8 generations without improvement, refinement-sigma 0.20→0.12).
- Best geometry: `dim=3072, n_heads=346, n_state=16, depth=13, expansion=1.0,
  lr=1.040e-3, batch_size=4` → **1274.98M params (−1.92% of 1300M, within ±3%)**.
  final-window loss 6.0636.
- n_state converged/stayed at **16** (the native old-winner regime; the {16,32} sweep
  did not prefer 32). Geometry regime: shallow-wide (depth 13, dim 3072 = the ×128
  ceiling), n_heads 346 ≈ the old 370.

### Reproduction vs the OLD leaderboard (same slice, p50k_base)

| arm        | old leaderboard | this run (standard protocol) |
|------------|----------------:|-----------------------------:|
| gdn2-mlp   |          5.9613 |        5.8949 (lb-gdn2-mlp)  |
| e97-raw    |          5.9511 |                            — |
| fla-gdn    |          6.0854 |                            — |
| **m2rnn**  |      **6.1161** |                  **6.0636**  |

The fresh standard-protocol search **reproduces the m2rnn regime and improves on it**:
6.0636 vs the old 6.1161 (**−0.0525**). The improvement is expected and consistent with
the gdn2-mlp sibling's gain — the old m2rnn run used `sigma 0.305 / 16 gens`; this
re-run uses the symmetric standard budget `sigma 0.8 / min_gens 13` (104 evals) on the
identical clean data slice, with n_state still free. This is now the clean, trustworthy,
uncontended m2rnn control at this protocol. m2rnn remains the weakest of the four arms
(6.0636 vs gdn2-mlp 5.8949 = +0.169), as on the old leaderboard.

### Convergence (best-so-far per generation)

| gen | best_so_far | gen_best | refine-sigma |
|----:|------------:|---------:|-------------:|
|   0 |      6.3258 |   6.3258 |        0.302 |
|   1 |      6.2968 |   6.2968 |        0.279 |
|   2 |      6.2466 |   6.2466 |        0.258 |
|   3 |      6.1565 |   6.1565 |        0.224 |
|   4 |  **6.0636** |   6.0636 |        0.201 |
|   5 |      6.0636 |   6.1640 |        0.203 |
|   6 |      6.0636 |   6.1535 |        0.199 |
|   7 |      6.0636 |   6.1041 |        0.172 |
|   8 |      6.0636 |   6.1449 |        0.174 |
|   9 |      6.0636 |   6.2057 |        0.172 |
|  10 |      6.0636 |   6.2026 |        0.150 |
|  11 |      6.0636 |   6.1135 |        0.140 |
|  12 |      6.0636 |   6.1005 |        0.120 |

(Generation index is 0-based here; `m2rnn_generations.jsonl` holds the 13 raw rows.)
Best fitness was reached at gen 4 (0-based) and never beaten over the remaining 8
generations → a clean fitness plateau at this proxy budget, mirroring the old m2rnn
run (which froze at 6.116 from its gen 10).

## Files

- `m2rnn_generations.jsonl` — per-generation convergence curve (13 rows, raw driver output).
- `m2rnn_results.json` — full driver results (best_loss, best_params, all 104 evals, elapsed_hours).
- `best.json` — best avg-loss + geometry + measured param count + delta vs old.
- `bf16_xma_fused_assert.txt` — bf16 + XMA-fused + level=m2rnn assertion from the live best-eval (eval_35) worker args + stdout.
- `anchors_m2rnn.json` — the single old-XMA-long-racer anchor (dim1920/nh370/n16/dep21).
- `launch_m2rnn.sh` — exact launch command.
