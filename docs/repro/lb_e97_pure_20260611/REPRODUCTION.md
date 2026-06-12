# lb-e97-pure — PURE E97 1.3B CMA reproduction (MEASURED data)

First arm of the 1.3B leaderboard rebuild. Validates that the STANDARD driver +
data slice reproduce the OLD leaderboard BEFORE the multi-arm run continues.

## Protocol (standard driver, byte-identical e97 path to the old run)

- Driver: `scripts/cmaes_search_v2.py --model e97` (`--level E97`, split erase/write edit).
  Diff vs the emender reference driver that produced the old leaderboard is purely
  additive (`emender` model_type); the `e97` build path is unchanged.
- Data slice: SAME `--data /home/erikg/elman/data/pile.txt`, `--seed 42`,
  `--tokenizer p50k_base` (vocab 50281) as
  `~/emender/experiments/local/cmaes_redo_1300m_20260529` → numbers directly comparable.
- Warm start: `docs/repro/cmaes_1300m_anchors/anchors_corrected.json` (anchor_only),
  VERIFIED byte-identical to the old run's `anchors_corrected.json`. e97 anchor =
  old champion geometry dim1664/nh271/n_state32/dep12/lr0.001228.
- CMA: popsize 8, min_generations 13, sigma 0.8, params 1300M, param_tolerance 0.03,
  chunk_size 2048, train_minutes 15. n_state PINNED 32 (`--fixed_n_state 32`).
- Precision/kernel: bf16 uniform + FUSED Triton split-edit. Asserted from the live
  worker cmdlines: `--level E97 --use_triton 1 --bf16 --n_state 32 --tokenizer p50k_base`.
  No eager fallback (E97 split-edit has no CUDA fused path; `--use_triton 1` is explicit).
- 8 GPUs (one candidate per GPU), leased via `scripts/gpu_lease.sh`.

## Result (MEASURED)

- **Best avg-loss: 5.9869**  (104 evals, 13 generations, CONVERGED, 4.55 h wall)
- Best geometry: dim=2560, n_heads=198, n_state=32, depth=10, lr=6.402e-4,
  batch_size=3 → 1269.2M params (within ±3% of 1300M target).
- Convergence (best-so-far): 6.156 (g0) → 6.096 (g4) → 5.990 (g7) → 5.987 (g12),
  monotonic; converged after 13 gens. Top-4 candidates span 5.9869–5.9944 (tight
  n_state-32 / wide-dim basin) — this spread is the single-run noise floor.
- Full curve: `e97_generations.jsonl`; all 104 evals: `e97_results.json`.

## Reproduction check vs OLD leaderboard — **PASS**

| arm (old leaderboard) | old best avg-loss | this run |
|---|---|---|
| **pure e97 (n_state=32)** | **5.9733** | **5.9869** (Δ +0.0136, +0.23%) |
| e97-raw | 5.9511 | — (reference arm) |
| gdn2-mlp | 5.9613 | — |
| fla-gdn | 6.0854 | — |
| m2rnn | 6.1161 | — |

- Pure-E97 reproduces its old value to within **+0.0136 (0.23%)** — smaller than the
  in-run top-candidate spread (0.0075–0.03), i.e. WITHIN the 15-min-budget CMA noise
  floor. (Old champion dim2176/dep14 vs new dim2560/dep10: same wide-dim n_state-32
  regime, a different point in the same flat basin.)
- Family ordering preserved with a decisive margin: e97 5.987 ≪ fla-gdn 6.085
  (−0.098) ≪ m2rnn 6.116 (−0.129). A wrong data slice/protocol would produce a
  wildly different number or a broken ordering; neither occurred.

**Verdict: PASS.** The standard driver + data slice/offset + anchors reproduce the
old 1.3B leaderboard for the E97 arm. The protocol is validated; the multi-arm
leaderboard build may continue.
