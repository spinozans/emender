# S5-SYMMETRIC 4-ARM CMA-ES SEARCH — RESULTS (task s5sym-search, agent-1009)

Date: 2026-06-03. Driver: `scripts/cmaes_search_s5.py --objective s5_acc@T128` (REAL `train_hybrid.py` s5_permutation eval).

## Config (per task spec)
- 4 arms; `linear_state` FIXED per arm and REMOVED from the CMA search space (`--s5_linear_state`/`--s5_use_gate`).
- Per-candidate: **300 steps**, seq_len 128, batch 32. Candidate eval **T=128 ONLY** (`--s5_eval_lengths 128`); 256/512/1024 held out for winner-eval.
- CMA over lr/dim/depth/n_heads/n_state, seeded from each arm's MODEL_CONFIG center, ~8M real-param matched ±10% (`rematch_dim_s5`).
- pop 10, 7-gen cap, min 3 gens, converge <0.005 acc over 3 consecutive gens, sigma 0.35, seed 42.
- GPUs: 0-7 (all idle at launch; logged in `driver_logs/gpus_used.txt`), round-robin, ~2 waves/gen.

## Per-generation best S5 acc@T128 (= 1 − gen_best_loss)

### e88-tanh — E88 linear_state=0 (tanh ON), use_gate=1
Run dir: `e88-tanh_20260603_084649` · candidates evaluated: 60 · generations: 6

| gen | best acc@T128 | valid/generated |
|----:|--------------:|:---------------:|
| 0 | 0.0443 | 10/20 |
| 1 | 0.0586 | 10/20 |
| 2 | 0.0674 | 10/20 |
| 3 | 0.0689 | 10/40 |
| 4 | 0.0689 | 10/20 |
| 5 | 0.0649 | 10/20 |

**Winner** (`winners/e88-tanh.args.json`): acc@T128=**0.0689**, dim=256 depth=5 n_heads=39 n_state=32 lr=0.00295 linear_state=0 use_gate=1

### e88-linear — E88 linear_state=1 (linear state), use_gate=1
Run dir: `e88-linear_20260603_094332` · candidates evaluated: 70 · generations: 7

| gen | best acc@T128 | valid/generated |
|----:|--------------:|:---------------:|
| 0 | 0.0465 | 10/20 |
| 1 | 0.0575 | 10/20 |
| 2 | 0.0679 | 10/20 |
| 3 | 0.0699 | 10/20 |
| 4 | 0.0701 | 10/20 |
| 5 | 0.0865 | 10/20 |
| 6 | 0.0758 | 10/20 |

**Winner** (`winners/e88-linear.args.json`): acc@T128=**0.0865**, dim=256 depth=5 n_heads=38 n_state=32 lr=0.002657 linear_state=1 use_gate=1

### m2rnn — M2RNN-CMA
Run dir: `m2rnn_20260603_105144` · candidates evaluated: 40 · generations: 4

| gen | best acc@T128 | valid/generated |
|----:|--------------:|:---------------:|
| 0 | 0.0258 | 10/20 |
| 1 | 0.0264 | 10/20 |
| 2 | 0.0292 | 10/20 |
| 3 | 0.0260 | 10/20 |

**Winner** (`winners/m2rnn.args.json`): acc@T128=**0.0292**, dim=512 depth=5 n_heads=19 n_state=32 lr=0.001606

### gdn — fla-gdn
Run dir: `gdn_20260603_105907` · candidates evaluated: 40 · generations: 4

| gen | best acc@T128 | valid/generated |
|----:|--------------:|:---------------:|
| 0 | 0.0366 | 10/20 |
| 1 | 0.0391 | 10/20 |
| 2 | 0.0385 | 10/20 |
| 3 | 0.0392 | 10/20 |

**Winner** (`winners/gdn.args.json`): acc@T128=**0.0392**, dim=512 depth=6 n_heads=22 n_state=32 lr=0.001206

## Winner ranking by candidate-search S5 acc@T128
(300-step truncated fitness only — NOT the final BL-1/arm decision, which is made at full fidelity in the winner-eval phase with held-out 256/512/1024 extrapolation.)

1. **e88-linear**: 0.0865
2. **e88-tanh**: 0.0689
3. **gdn**: 0.0392
4. **m2rnn**: 0.0292
