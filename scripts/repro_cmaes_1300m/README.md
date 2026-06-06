# 1.3B CMA-ES redo launch scripts (faithful copies)

These are verbatim copies of the launch scripts that drove the ~1.3B / ctx-2k
CMA-ES leaderboard (the sweep in which `e97-raw` ranks #1). The originals lived
under the git-ignored `experiments/local/cmaes_redo_1300m_20260529/` tree; they
are copied here so the exact invocations are reproducible from git.

- `launch_e97_queue.sh` — `e97`, `e97-raw` (rank 1), `e97-linear` (GPUs 4,5; Triton E88)
- `launch_gdn2.sh` — old mixer-only `gdn2` baseline (GPUs 6,7)
- `launch_missing_cma_20260531.sh` — `e88-raw`, `e88-linear`, `fla-gdn`, `m2rnn` (1 GPU each)

They reference the local output root `experiments/local/cmaes_redo_1300m_20260529`
and anchor JSON files; copy the committed anchors from
`docs/repro/cmaes_1300m_anchors/` into that root first. See
[`docs/REPRODUCE_E97_RAW_1P3B.md`](../../docs/REPRODUCE_E97_RAW_1P3B.md) for the
full recipe, and adjust `--gpus` / data path for your environment.
