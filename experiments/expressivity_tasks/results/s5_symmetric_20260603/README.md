# S5-symmetric CMA-ES results (2026-06-03)

Records directory for the S5-symmetric experiment (protocol
`paper/review/S5_SYMMETRIC_PROTOCOL.md` §D; launch commands in
`paper/review/S5_SYMMETRIC_LAUNCH.md`).

## Committed by the pre-flight (this task, no GPU)
- `seeds_s5_symmetric.json` — CMA-ES seed/anchor configs (the 8M `MODEL_CONFIG`
  centers), loaded by `load_anchor_configs`.
- `README.md` — this file.

## Written by the search task `s5sym-search` (one subdir per arm)
`<model>_<timestamp>/` containing:
- `eval_<id>/params.json` — candidate config + objective.
- `eval_<id>/s5cand_<model>_eval<id>.json` — the **real** `train_hybrid`
  `s5_permutation` output (per-step `eval_acc`, `final_acc`, `length_extrap`).
- `eval_<id>/{cmd.txt,stdout.txt,stderr.txt,.done}`.
- `generations.jsonl` — per-generation CMA snapshots.
- `results.json` — ranked candidates incl. `s5_acc_at_T128`.

Fitness = `1 − mean_S5_acc@T128` over the final 1000 training steps of each
capped (5000-step, seq128, batch32) candidate; NaN/divergence → 1.0. All four
arms (`e88`, `m2rnn`, `fla-gdn`, `m2rnn-paper`) use an identical budget
(pop 16, sigma 0.35, ≤12 generations, search seed 42) on GPUs 2,3,4,5.
