# lb-gdn2-mlp — clean 1.3B CMA control for the PRIMARY rank-2 cell (MEASURED data)

Re-run of **gdn2-mlp** (official-style GDN-2 + SwiGLU MLP — the rank-2 **PRIMARY**
control, NOT the weak mixer-only `gdn2`) through the **standard** driver
`scripts/cmaes_search_v2.py` at 1.3B with the symmetric leaderboard budget, on the
SAME data slice as `~/emender/experiments/local/cmaes_redo_1300m_20260529` so the
avg-loss is directly comparable to the OLD leaderboard.

## Protocol (standard driver, symmetric to the gdn2 / m2rnn / e97 / emender 1.3B arms)

- Driver: `scripts/cmaes_search_v2.py --model gdn2-mlp` → `--level gdn2-mlp`
  (official GatedDeltaNet-2 fused chunked kernel via `GDN2_PATH=/home/erikg/GatedDeltaNet-2`
  + SwiGLU MLP, official short conv `use_conv=1 d_conv=4`).
- Searched axes (`gdn2-mlp` search space): `dim` (1536–3072, ×128), `expansion`
  (1–3), `depth` (10–32), `n_heads` (8–40), `mlp_ratio` (2.0–4.0), `lr`
  (1e-4–3e-3 log), `batch_size` (1–64 log, clamped by memory probe).
- **Precision/kernel: bf16 uniform (`--bf16`) + FUSED official GDN-2 kernel, NO eager.**
  Asserted from the live BEST-eval worker args (`bf16_fused_assert.txt`): `bf16=True`,
  `level=gdn2-mlp`. The GDN-2 family runs FLA's fused `chunk_gated_delta_rule` Triton
  path; the eager T-scan fallback in `train.py` applies ONLY to the E97/E88 raw_write
  families, never to `gdn2`/`gdn2-mlp`.
- Data slice: SAME `--data /home/erikg/elman/data/pile.txt`, `--seed 42`,
  `--tokenizer p50k_base` (vocab 50281) as `cmaes_redo_1300m_20260529` → numbers
  directly comparable to the OLD leaderboard.
- CMA: **popsize 8, min_generations 13** (≥104 evals, matching the old leaderboards'
  96–109), **sigma 0.8**, params **1300M**, **param_tolerance 0.03**, **chunk_size
  2048**, **train_minutes 15**. 8 GPUs (one candidate per GPU), leased via
  `scripts/gpu_lease.sh`. Single anchor = the official primary geometry
  `dim2048/dep18/nh20/exp1/mlp2.69/lr4e-4/bs1` (= 1299.9M, the geometry that
  produced the old leaderboard 5.9613); `--anchor_only_cmaes`.

## Result (MEASURED) — CONVERGED, 13 gens / 104 evals / 4.47 h

- **Best avg-loss: 5.8949**  (eval_86; CONVERGED after 13 generations,
  `generations_without_improvement=2/2`, refinement-sigma 0.29→0.18).
- Best geometry: `dim=2176, expansion=1, depth=12, n_heads=30, mlp_ratio=3.259,
  lr=4.743e-4, batch_size=4` → **1286.7M params (−1.02% of 1300M, within ±3%)**.
  final-window loss 5.8949.

### Reproduction vs the OLD leaderboard (same slice, p50k_base)

| arm        | old leaderboard | this run (standard protocol) |
|------------|----------------:|-----------------------------:|
| gdn2-mlp   |          5.9613 |                   **5.8949** |
| e97-raw    |          5.9511 |                            — |
| fla-gdn    |          6.0854 |                            — |
| m2rnn      |          6.1161 |                            — |

The fresh standard-protocol search **reproduces the gdn2-mlp regime and slightly
improves it**: 5.8949 vs the old 5.9613 (**−0.0664**). The improvement is expected —
the old run used `sigma 0.5 / min_gens 8`; this re-run uses the symmetric standard
budget `sigma 0.8 / min_gens 13` (104 evals), a more thorough search of the same
geometry family on the identical data slice. This is now the clean, trustworthy
gdn2-mlp control at this protocol.

### Convergence (best-so-far)

| gen | best_so_far | gen_best | refine-sigma |
|----:|------------:|---------:|-------------:|
|   0 |      5.9866 |   5.9866 |        0.291 |
|   1 |      5.9866 |   6.0437 |        0.273 |
|   2 |      5.9866 |   6.0229 |        0.254 |
|   3 |      5.9866 |   6.0105 |        0.234 |
|   4 |      5.9780 |   5.9780 |        0.233 |
|   5 |      5.9688 |   5.9688 |        0.222 |
|   6 |      5.9515 |   5.9515 |        0.209 |
|   7 |      5.9390 |   5.9390 |        0.205 |
|   8 |      5.9390 |   6.0014 |        0.208 |
|   9 |      5.9390 |   5.9828 |        0.199 |
|  10 |  **5.8949** |   5.8949 |        0.196 |
|  11 |      5.8949 |   5.9445 |        0.195 |
|  12 |      5.8949 |   5.9622 |        0.178 |

## Files

- `gdn2_mlp_generations.jsonl` — per-generation convergence curve (13 rows, raw driver output).
- `gdn2_mlp_results.json` — full driver results (best_loss, best_params, all 104 evals).
- `best.json` — best avg-loss + geometry + measured param count.
- `bf16_fused_assert.txt` — bf16+fused assertion from the live best-eval worker args.
- `anchors_gdn2_mlp.json` — the single official-primary anchor.
- `launch_gdn2_mlp.sh` — exact launch command.
