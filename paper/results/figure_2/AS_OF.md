# Figure 2 Snapshot - As of 2026-05-29T18:04:51Z

Training is **in progress**. The values below were regenerated from the active
racer logs on 2026-05-29. Because the logs were still appending during the
refresh, the active files were copied once to an uncommitted `/tmp` snapshot and
the smoothing pipeline was run against that consistent cutoff:

- Snapshot root: `/tmp/figure2_refresh_snapshot_20260529T180451Z`
- CSV generation: `paper/results/figure_2/smooth.py`, with log paths patched at
  runtime to the snapshot copies.
- Plot generation: `python3 paper/results/figure_2/plot_normalized.py`
- BPB conversion: `bits/byte = nats/token * log2(e) / 3.918625`, using the
  pinned tokenizer estimate in `scripts/estimate_tokenizer_bytes_per_token.json`.

The paper and Figure 2 labels now use the **100K-step trailing endpoint
average**. This is intentionally more conservative than the previous 10K label
convention because GDN is visibly jumpy near the active tail; the 10K endpoint
can make its current dip look more stable than it is. The plotted curve uses the
same 100K-step trailing average as the labels.

Do not cite these results without re-running `smooth.py` and
`plot_normalized.py` against a fresh active-log snapshot.

---

## Figure 2 tail labels

The normalized Figure 2 renderer plots the three public 1.3 B-class racer
models: Emender/E88, GDN, and M2RNN-CMA. Rounded tail labels at this cutoff are:

- E88 / NDM: `0.977` BPB
- GDN: `0.970` BPB
- M2RNN-CMA: `0.983` BPB

Ordering at the current tail remains:

```text
GDN < E88 / NDM < M2RNN-CMA
```

The 10K, 50K, and 100K endpoint trailing averages are:

| Model | Tail step | 10K trailing loss / BPB | 50K trailing loss / BPB | 100K trailing loss / BPB | Paper label |
| --- | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | 1,405,450 | 2.652421 / 0.976525 | 2.653698 / 0.976995 | 2.653219 / 0.976819 | **0.977** |
| GDN | 1,847,050 | 2.615085 / 0.962779 | 2.621877 / 0.965280 | 2.633709 / 0.969636 | **0.970** |
| M2RNN-CMA | 1,343,050 | 2.667076 / 0.981920 | 2.667397 / 0.982038 | 2.671290 / 0.983472 | **0.983** |

GDN is the unstable row: its 10K endpoint is `0.963` BPB, but the 100K endpoint
is `0.970` BPB. The paper therefore treats GDN's current low endpoint as a
short-window dip inside the shared sub-1-bpb band, not as a stable superiority
claim. GDN remains the lowest BPB endpoint under the 100K convention, but by
about `0.007` BPB rather than the larger 10K-window gap.

## Recomputed endpoints

All runs use: dataset = Pile (`pile.txt`, `p50k_base` tokenizer), context =
2048 tokens, optimizer = schedule-free AdamW, bf16.

| Model | Active log | Log tail time UTC | Log mtime UTC | Tail step | Raw tail loss | 100K trailing loss | 100K trailing BPB | Rounded BPB | Tokens seen | ~FLOPs | Stitched wallclock h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log` | 2026-05-29T18:04:15+00:00 | 2026-05-29T18:04:15.910Z | 1,405,450 | 2.6012 | 2.653219 | 0.976819 | **0.977** | 14.391808B | 1.0994 × 10²⁰ | 512.851 |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log` | 2026-05-29T18:04:37+00:00 | 2026-05-29T18:04:37.349Z | 1,847,050 | 2.6282 | 2.633709 | 0.969636 | **0.970** | 15.131034B | 1.2277 × 10²⁰ | 517.805 |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log` | 2026-05-29T18:04:30+00:00 | 2026-05-29T18:04:30.525Z | 1,343,050 | 2.7481 | 2.671290 | 0.983472 | **0.983** | 13.752832B | 1.0786 × 10²⁰ | 479.021 |

## Comparison with the previous public snapshot

| Model | 2026-05-27 public paper snapshot BPB | 2026-05-29 refreshed 100K BPB | Rounded label changed? |
| --- | ---: | ---: | --- |
| E88 / NDM | 0.979277 -> `0.979` | 0.976819 -> `0.977` | yes |
| GDN | 0.974841 -> `0.975` | 0.969636 -> `0.970` | yes |
| M2RNN-CMA | 0.984356 -> `0.984` | 0.983472 -> `0.983` | yes |

The narrative should say all three public models are sub-1 BPB, GDN is the
lowest BPB endpoint at this cutoff under the stable 100K convention, E88 is
second, and M2RNN-CMA remains third. It should also explicitly avoid treating
the short-window GDN dip as a stable margin.

---

## Per-model caveats

### E88/NDM

- Architecture: Level E88 (Nonlinear Delta Memory / NDM paper name).
  Config: dim=1664, depth=12, n_heads=370, n_state=32, expansion=1.0,
  use_gate=1, gate_activation=silu, lr=8.678e-4.
- Training started 2026-05-07. Original run diverged (NaN gradients) at step
  ~247,250. A repair segment recovered from step 231,000; the postrepair run
  resumed at step 247,500 on 2026-05-11 and has been running continuously since.
- Batch size: 5 (effective tokens/step = 10,240).

### GDN

- Architecture: Level fla-gdn (Flash Linear Attention with Gated Delta Net).
  Config: dim=2688, depth=21, expansion=2, n_heads=44, lr=2.871e-3.
- Training started 2026-05-07. Resumed from checkpoint at step 351,000 on
  2026-05-11; continuous since.
- Parameter count (1.352B) slightly exceeds the 1.27B target due to
  CMA-ES dim/depth selection.
- Batch size: 4 (effective tokens/step = 8,192).
- GDN's active tail is noisier than E88 and M2RNN-CMA under short windows; use
  the side-by-side smoothing table above when interpreting endpoint labels.

### M2RNN-CMA

- Architecture: Level m2rnn (M2RNN with CMA-ES optimized geometry, "tied"
  config). Config: dim=1920, depth=21, heads=370, n_state=16, lr=6.021e-4.
- Training started 2026-05-09. Resumed with XMA (accelerated M2RNN) backend at
  step 123,000 on 2026-05-11; continuous since.
- Uses XMA kernels from the accelerated-model-architectures package.
- Batch size: 5 (effective tokens/step = 10,240).

---

## Absent / excluded runs

### M2RNN-paper

- Config: dim=3072, depth=10, heads=759, n_state=16, lr=4.911e-4.
- Ran 2026-05-09. Stopped at step 8,400, loss ≈ 11.5 (never converged).
- Run was abandoned; no usable loss curve. Not included in Figure 2.
- Log: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_paper.log`

---

## Smoothed loss columns in CSVs

`smooth_5k`, `smooth_10k`, `smooth_50k`, `smooth_100k` — centered moving
averages of raw loss (nats/token). Edges use smaller windows, so centered tail
values are not used for paper endpoint labels.

`trail_5k`, `trail_10k`, `trail_50k`, `trail_100k` — trailing moving averages of
raw loss (nats/token). Figure 2 and the paper labels use `trail_100k`.

The window is computed in log-entry index space. With `log_every=50`, a
100K-step trailing window corresponds to approximately 2,000 log entries.

The figure renderer (`plot_normalized.py`) converts `trail_100k` to bits/byte at
the display step using the canonical bytes/token estimate. The CSV columns
themselves remain in native nats/token units except the legacy `bits_per_base`
column written by `smooth.py`, which is preserved for backward compatibility and
is not used by the normalized figure labels.
