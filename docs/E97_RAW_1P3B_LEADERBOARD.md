# E97-RAW 1.3B LM CMA-ES Leaderboard (Evidence Summary)

KB-scale evidence extracted from the local CMA-ES redo artifacts. The raw
artifacts (~310 GB of checkpoints, eval logs, CMA-ES state) are intentionally
**git-ignored** under `experiments/local/` and are **not** committed. This file
preserves the leaderboard, the winning configurations, and the local paths so
the result remains traceable from git alone.

- Snapshot source: `experiments/local/cmaes_redo_1300m_20260529/` (local disk, git-ignored)
- Companion handoff: [`docs/HANDOFF_CMAES_GDN2_MLP_20260605.md`](HANDOFF_CMAES_GDN2_MLP_20260605.md)
- Reproduction recipe: [`docs/REPRODUCE_E97_RAW_1P3B.md`](REPRODUCE_E97_RAW_1P3B.md)
- Headline result: **`e97-raw` is rank 1** at the ~1.3B parameter / ctx-2k scale
  on this token-/budget-matched CMA-ES sweep.

## Metric definitions

These come straight from the CMA-ES eval artifacts:

- **`loss` (avg loss)** — the CMA-ES *fitness*: the mean loss over the eval's
  15-minute training trajectory. This is the value the search minimizes and the
  value the leaderboard ranks on.
- **`final_loss`** — the eval's last-window / final reported loss (end of the
  15-minute run). Informative but **not** the ranking key.
- **`params M`** — actual constructed parameter count (millions) of the best
  candidate, after exact accounting (target was 1300M with `--param_tolerance 0.03`).
- **`best eval`** — the `eval_id` of the winning candidate within that model's run.
- **`evals`** — number of de-duplicated completed CMA-ES evals for that model.

All runs used: 15-minute train budget per candidate, ctx/chunk = 2048
(`--chunk_size 2048`), `p50k_base` tokenizer, Pile text data
(`/home/erikg/elman/data/pile.txt`), CMA-ES popsize 8, target 1300M params.

## Leaderboard (ranked by CMA-ES avg loss)

| Rank | Model | Evals | Status | Best avg loss | Final loss | Params (M) | Best eval |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `e97-raw` | 103 | complete | **5.9511** | 5.4738 | 1265.6 | 58 |
| 2 | `gdn2-mlp` | 64 | complete | 5.9613 | 5.5400 | 1285.2 | 57 |
| 3 | `e97` | 109 | complete | 5.9733 | 5.5386 | 1274.7 | 59 |
| 4 | `e88-linear` | 104 | recoverable | 5.9854 | 5.5263 | 1269.1 | 98 |
| 5 | `e88` | 92 | complete | 5.9900 | 5.5338 | 1289.1 | 54 |
| 6 | `e88-raw` | 106 | complete | 6.0390 | 6.0390 | 1288.0 | 55 |
| 7 | `e97-linear` | 109 | complete | 6.0516 | 6.0516 | 1315.8 | 95 |
| 8 | `fla-gdn` | 109 | recoverable | 6.0854 | 5.5951 | 1292.1 | 83 |
| 9 | `m2rnn` | 107 | complete | 6.1161 | 6.1161 | 1296.4 | 72 |
| 10 | `gdn2` | 96 | complete | 6.3850 | 5.8386 | 1262.6 | 13 |
| 11 | `transformer` | 128 | complete | 6.4606 | 6.4606 | 1318.8 | 104 |

Notes on coverage (target was strictly more than 96 completed evals):

- Met target: `e97-raw`, `e97`, `e88-linear`, `e88-raw`, `e97-linear`, `fla-gdn`,
  `m2rnn`, `transformer`.
- Under target: `gdn2-mlp` (64), `e88` (92), mixer-only `gdn2` (96).
- The old mixer-only `gdn2` (rank 10) is **not** the primary GDN-2 comparison;
  `gdn2-mlp` (official-style GDN-2 + SwiGLU MLP) is. `gdn2-mlp` is rank 2 but
  has only 64 evals, so its standing is provisional vs the 100+ eval runs.
- `status = recoverable` means the ranking was rebuilt from per-eval `.done`
  files rather than a finalized `results.json`.

The table is mechanically reproducible from the per-model `results.json` /
`.done` files; see the aggregation snippet in
[`docs/REPRODUCE_E97_RAW_1P3B.md`](REPRODUCE_E97_RAW_1P3B.md).

## Best configuration per model

The winning hyperparameters (the candidate with the lowest avg loss). `e97*`,
`e88*`, `m2rnn` use `{dim, n_heads, n_state, depth, lr, batch_size}`; GDN-2 /
FLA / transformer variants use `{dim, expansion|mlp_ratio, n_heads, depth, lr,
batch_size}`.

| Model | Best config (JSON) | Actual params |
| --- | --- | ---: |
| `e97-raw` | `{"dim":2432,"n_heads":416,"n_state":16,"depth":10,"lr":0.0009851067699366818,"batch_size":3}` | 1,265,553,184 |
| `gdn2-mlp` | `{"dim":2304,"expansion":2,"depth":17,"n_heads":8,"mlp_ratio":2.854220752778522,"lr":0.0003907570359771844,"batch_size":2}` | 1,285,245,320 |
| `e97` | `{"dim":2176,"n_heads":170,"n_state":32,"depth":14,"lr":0.0010403731352768883,"batch_size":2}` | 1,274.7M |
| `e88-linear` | `{"dim":2176,"n_heads":331,"n_state":32,"depth":10,"lr":0.0010059680042677557,"batch_size":2}` | 1,269.1M |
| `e88` | `{"dim":2432,"n_heads":298,"n_state":32,"depth":10,"lr":0.0009203005278730615,"batch_size":2}` | 1,289.1M |
| `e88-raw` | `{"dim":1792,"n_heads":346,"n_state":32,"depth":12,"lr":0.0009560158819812891,"batch_size":4}` | 1,288.0M |
| `e97-linear` | `{"dim":2176,"n_heads":224,"n_state":32,"depth":11,"lr":0.0010531703750126676,"batch_size":4}` | 1,315.8M |
| `fla-gdn` | `{"dim":3712,"expansion":2,"depth":10,"n_heads":35,"lr":0.000731412343100645,"batch_size":3}` | 1,292.1M |
| `m2rnn` | `{"dim":2048,"n_heads":342,"n_state":16,"depth":21,"lr":0.0005502541423305248,"batch_size":5}` | 1,296.4M |
| `gdn2` | `{"dim":1664,"expansion":2,"depth":10,"n_heads":58,"lr":0.0020992079360665113,"batch_size":1}` | 1,262.6M |
| `transformer` | `{"dim":1792,"n_heads":19,"expansion":4,"depth":26,"lr":0.0004301815925391534,"batch_size":8}` | 1,318.8M |

### Winner — `e97-raw`

- Best avg loss **5.951101578947369**, final loss **5.4738**, eval_id **58**.
- Actual params **1,265,553,184** (1265.6M).
- 103 evals, ~6.40 wall-clock hours.
- Architecture: E88/NDM split key-axis erase/read + value-axis write edit cell
  (`--level E97`), with the **raw-write** ablation (`--e88_raw_write 1`) — the
  split edit writes the gated value directly instead of applying the delta
  correction. Gate enabled (`silu`), `expansion=1.0`, `n_state=16`.

### Runner-up — `gdn2-mlp`

- Best avg loss **5.961312021857924**, final loss **5.5400**, eval_id **57**.
- Actual params **1,285,245,320** (1285.2M).
- 64 evals / 8 generations, ~9.30 wall-clock hours.
- Official-style GDN-2 mixer + SwiGLU MLP proxy. Search is flat ~5.96–5.98;
  good configs favor `expansion=2`, dim ~2304–2816, shallow-to-mid depth, small
  GDN-2 head counts. **Provisional** (64 < 96 eval target).

## Local artifact paths (git-ignored, NOT committed)

- CMA-ES redo root: `experiments/local/cmaes_redo_1300m_20260529/`
- Per-model run dirs (each with `results.json`, `eval_*/`, `cmaes_state*.pkl`):
  - `e97-raw/e97-raw_20260530_002342/`  ← **winner**
  - `gdn2-mlp/gdn2-mlp_20260602_201948/`
  - `e97/e97_20260529_152252/`, `e88/e88_20260530_151323/`,
    `e88-raw/e88-raw_20260531_131116/`, `e97-linear/e97-linear_20260530_113253/`,
    `fla-gdn/...`, `m2rnn/m2rnn_20260531_131116/`, `gdn2/gdn2_20260529_152252/`,
    `transformer/transformer_20260530_042721/`
- Warm-start anchor configs (small JSON, git-ignored): `anchors_corrected.json`,
  `anchors_gdn2_primary.json`, `anchors_missing_20260531.json`
- Trajectory analysis: `analysis/cmaes_loss_trajectories.csv`,
  `analysis/cmaes_loss_trajectories_summary.json`,
  `analysis/cmaes_loss_trajectories_bits.png`
- Launch scripts (git-ignored; captured verbatim in the reproduce recipe):
  `launch_e97_queue.sh`, `launch_gdn2.sh`, `launch_missing_cma_20260531.sh`
