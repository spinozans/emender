# Figure 3 Data Sources

All training logs live on the training host under `/tmp/`. They are large
(1–3 MB) and are NOT copied into the repo. Stable pointers follow.

## Active 1.27B runs (as of 2026-05-27)

### E88 / NDM
Architecture: Level E88 (Nonlinear Delta Memory)
Parameters: 1,273,191,856 (~1.27B)
Batch size: 5, Chunk size: 2048, Tokenizer: p50k_base

Log files (concatenate in order for the full curve):
1. `/tmp/pile_convergence_3arch/ctx2k/e88.log`
   Steps 50 → 247,450 (original run; NaN appeared at ~247,250, repaired)
2. `/tmp/pile_convergence_3arch/ctx2k/e88_repair_from231k.log`
   Steps 231,000 → 247,500 (repair segment)
3. `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log`
   Steps 247,550 → current (active run, resumed 2026-05-11)

### FLA-GDN
Architecture: Level fla-gdn (Flash-Linear Attention with Gated Delta Net)
Parameters: 1,352,352,498 (~1.35B)
Batch size: 4, Chunk size: 2048, Tokenizer: p50k_base

Log files (concatenate in order):
1. `/tmp/pile_convergence_3arch/ctx2k/fla-gdn.log`
   Steps 50 → 352,450 (original run)
2. `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log`
   Steps 351,050 → current (active run, resumed 2026-05-11)

### Mamba2
Architecture: Level mamba2
Parameters: 934,426,624 (~934M)
Batch size: 4, Chunk size: 2048, Tokenizer: p50k_base

Log files (concatenate in order):
1. `/tmp/pile_convergence_3arch/ctx2k/mamba2.log`
   Steps 50 → 433,800 (original run)
2. `/tmp/pile_convergence_3arch/ctx2k/mamba2_resume.log`
   Steps 432,050 → current (active run, resumed 2026-05-11)

### M2RNN-CMA (m2rnn_tied)
Architecture: Level m2rnn (CMA-ES optimized geometry)
Parameters: 1,307,101,140 (~1.31B)
Batch size: 5, Chunk size: 2048, Tokenizer: p50k_base

Log files (concatenate in order):
1. `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied.log`
   Steps 50 → 123,250 (original run)
2. `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log`
   Steps 123,050 → current (active run, resumed 2026-05-11 with XMA backend)

## Inactive / abandoned runs

### M2RNN-paper
Architecture: Level m2rnn (paper-default geometry: dim=3072, depth=10, heads=759)
Parameters: 1,303,628,684 (~1.30B)
Batch size: 4, Chunk size: 2048, Tokenizer: p50k_base
Status: ABANDONED — stopped at step 8,400, loss ~11.5 (never converged)
Log: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_paper.log`

## Run configuration script
`~/elman/run_pile_convergence_3arch.sh` — launches E88, FLA-GDN, Mamba2 at 1.27B
`~/elman/` (various run_*.sh scripts) — M2RNN configuration extracted from manifest:
`/tmp/pile_convergence_m2rnn/ctx2k/manifest.txt`
