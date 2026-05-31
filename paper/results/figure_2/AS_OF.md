# Figure 2 Snapshot - As of 2026-05-31T13:49:33Z

Training is **in progress**. The values below were regenerated from the active
racer logs on 2026-05-31. Because the logs were still appending during the
refresh, the active files were copied once to an uncommitted snapshot and the
smoothing pipeline was run against that consistent cutoff:

- Snapshot root: `/tmp/figure2_refresh_snapshot_20260531T134933Z`
- CSV generation: `paper/results/figure_2/smooth.py`, with log paths patched at
  runtime to the snapshot copies.
- Plot generation: `python3 paper/results/figure_2/plot_normalized.py`
  (writes the canonical `figure_2.png`).
- BPB conversion: `bits/byte = nats/token * log2(e) / 3.918625`, using the
  pinned tokenizer estimate in `scripts/estimate_tokenizer_bytes_per_token.json`
  (constant `0.3681635882200934`).

The paper and Figure 2 labels use the **100K-step trailing endpoint average**
(`trail_100k`). This is intentionally more conservative than a 10K endpoint
because GDN is visibly jumpy near the active tail; the 10K endpoint can make its
current position look more stable than it is. The plotted curve uses the same
100K-step trailing average as the labels.

Do not cite these results without re-running `smooth.py` and
`plot_normalized.py` against a fresh active-log snapshot.

---

## Figure 2 tail labels

The normalized Figure 2 renderer plots the three public 1.3 B-class racer
models: Emender/E88, GDN, and M2RNN-CMA. Rounded tail labels at this cutoff are:

- E88 / NDM: `0.974` BPB
- GDN: `0.977` BPB
- M2RNN-CMA: `0.980` BPB

Ordering at the current tail (the ordering has **flipped** since the
2026-05-29 snapshot — E88 now leads after further training):

```text
E88 / NDM < GDN < M2RNN-CMA
```

The 10K, 50K, and 100K endpoint trailing averages are:

| Model | Tail step | 10K trailing loss / BPB | 50K trailing loss / BPB | 100K trailing loss / BPB | Paper label |
| --- | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | 1,523,250 | 2.641152 / 0.972376 | 2.645569 / 0.974002 | 2.644925 / 0.973765 | **0.974** |
| GDN | 1,999,300 | 2.652831 / 0.976676 | 2.664149 / 0.980843 | 2.653617 / 0.976965 | **0.977** |
| M2RNN-CMA | 1,466,400 | 2.654668 / 0.977352 | 2.661433 / 0.979843 | 2.661439 / 0.979845 | **0.980** |

GDN remains the noisier row: its 50K endpoint is `0.981` BPB while its 10K and
100K endpoints sit at `0.977`. E88 is now the lowest endpoint across all three
windows. E88 leads GDN by about `0.009` nats / `0.003` BPB at the 100K window;
both remain inside the shared sub-1-bpb band, so the lead is reported as a
narrow current-snapshot ordering, not a stable separation while training is live.

## Recomputed endpoints

All runs use: dataset = Pile (`pile.txt`, `p50k_base` tokenizer), context =
2048 tokens, optimizer = schedule-free AdamW, bf16.

| Model | Active log | Log tail time UTC | Tail step | Raw tail loss | 100K trailing loss | 100K trailing BPB | Rounded BPB | Tokens seen | ~FLOPs | Stitched wallclock h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log` | 2026-05-31T13:49:20+00:00 | 1,523,250 | 2.6368 | 2.644925 | 0.973765 | **0.974** | 15.598080B | 1.1916 × 10²⁰ | 556.603 |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log` | 2026-05-31T13:49:21+00:00 | 1,999,300 | 2.5379 | 2.653617 | 0.976965 | **0.977** | 16.378266B | 1.3290 × 10²⁰ | 561.551 |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log` | 2026-05-31T13:49:19+00:00 | 1,466,400 | 2.6782 | 2.661439 | 0.979845 | **0.980** | 15.015936B | 1.1776 × 10²⁰ | 522.768 |

Stitched wallclock corresponds to roughly 23.2 (E88), 23.4 (GDN), and 21.8
(M2RNN-CMA) GPU-days at this recording.

## Comparison with the previous public snapshot

| Model | 2026-05-29 refreshed 100K BPB | 2026-05-31 refreshed 100K BPB | Rounded label changed? |
| --- | ---: | ---: | --- |
| E88 / NDM | 0.976819 -> `0.977` | 0.973765 -> `0.974` | yes |
| GDN | 0.969636 -> `0.970` | 0.976965 -> `0.977` | yes |
| M2RNN-CMA | 0.983472 -> `0.983` | 0.979845 -> `0.980` | yes |

The 2026-05-29 snapshot had GDN as the lowest endpoint. After two more days of
training the ordering flipped: **E88 is now the lowest-BPB endpoint, GDN second,
M2RNN-CMA third**, all still sub-1-bpb. The narrative should say all three
public models are sub-1 BPB, E88 is the lowest BPB endpoint at this cutoff under
the stable 100K convention, GDN is second, and M2RNN-CMA is third — and should
continue to avoid treating the narrow E88–GDN gap as a stable margin while
training is live.

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

### Mamba2

- Architecture: Level mamba2. Parameters ~934M (below the 1.27B target due to
  CMA-ES dim selection). Retained in the CSVs for reference but not plotted in
  the public normalized Figure 2.
- The active log last advanced 2026-05-25 (stale relative to this snapshot);
  its 100K trailing endpoint at the recorded tail (step 1,982,400) is
  2.693570 nats / 0.991674 BPB.

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
