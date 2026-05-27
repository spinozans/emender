# Figure 2 Snapshot - As of 2026-05-27T20:25 UTC

Training is **in progress**. All numbers below are the 10K-step smoothed
snapshots from CSVs regenerated from the live training logs on
2026-05-27T20:25 UTC. Curves will be updated as training continues. Do not
cite these results without re-running `smooth.py` and `plot_normalized.py` to
regenerate with current logs.

---

## Models and current state

All runs use: dataset = Pile (pile.txt, p50k_base tokenizer), context = 2048 tokens,
optimizer = schedule-free AdamW, bf16.

| Model | Params | Step | 10K-smoothed loss (nats/tok) | Bits/byte† | Tokens seen | ~FLOPs‡ | GPU hours§ |
|-------|--------|------|-----------------|------------|-------------|---------|-----------|
| E88/NDM | 1.273B | 1,281,300 | 2.6599 | 0.9793 | 13.12B | 1.00 × 10²⁰ | ~467 |
| FLA-GDN | 1.352B | 1,687,500 | 2.6478 | 0.9748 | 13.82B | 1.12 × 10²⁰ | ~472 |
| Mamba2 | 0.934B | 1,982,400 | 2.6919 | 0.9911 | 16.24B | 9.10 × 10¹⁹ | ~407 |
| M2RNN-CMA | 1.307B | 1,213,600 | 2.6737 | 0.9844 | 12.43B | 9.75 × 10¹⁹ | ~433 |

†  Bits/byte = nats/token × log2(e) / bytes/token, with
   bytes/token = 3.918625 (canonical 2000-sample sweep of `p50k_base` on
   The Pile at `chunk_tokens=2048`; pinned at
   `scripts/estimate_tokenizer_bytes_per_token.json`). The figure
   applies this conversion at the display step in `plot_normalized.py`;
   CSVs continue to record the raw `loss` and `smooth_*` columns in
   their native nats/token units.

‡  Total FLOPs = 6 × N_params × B × T × step, where B = batch size, T = 2048.
   This is the standard dense-op approximation (2N forward, 4N backward).
   Recurrent state-update FLOPs in gated SSMs are not separately counted;
   the formula may underestimate total compute for E88 and M2RNN variants.

§  GPU hours are taken from the monotonic wallclock reconstructed by
   `smooth.py` (`elapsed_h` field for newer log segments; tok/s
   integration for the older pre-2026-05-11 format), and are accurate to
   within a few percent.

---

## Per-model caveats

### E88/NDM
- Architecture: Level E88 (Nonlinear Delta Memory / NDM paper name).
  Config: dim=1664, depth=12, n_heads=370, n_state=32, expansion=1.0,
  use_gate=1, gate_activation=silu, lr=8.678e-4.
- Training started 2026-05-07. Original run diverged (NaN gradients) at step
  ~247,250. A repair segment recovered from step 231,000; the postrepair run
  resumed at step 247,500 on 2026-05-11 and has been running continuously since.
- NaN event is visible as a spike in the raw loss curve ~77 h into training
  (log scale). The 10K-step smoothed curve does not show it prominently.
- Batch size: 5 (effective tokens/step = 10,240).

### FLA-GDN
- Architecture: Level fla-gdn (Flash Linear Attention with Gated Delta Net).
  Config: dim=2688, depth=21, expansion=2, n_heads=44, lr=2.871e-3.
- Training started 2026-05-07. Resumed from checkpoint at step 351,000 on
  2026-05-11; continuous since.
- Parameter count (1.352B) slightly exceeds the 1.27B target due to
  CMA-ES dim/depth selection.
- Batch size: 4 (effective tokens/step = 8,192).

### Mamba2
- Architecture: Level mamba2 (Mamba2 SSM).
  Config: dim=2048, depth=32, expansion=3, mamba_d_state=160, lr=3.502e-4.
- Training started 2026-05-07. Resumed from checkpoint at step 432,000 on
  2026-05-11; continuous since. The active log last advanced 2026-05-25.
- Parameter count (934M) is below the 1.27B target; this is the CMA-ES
  winner at this architecture family and scale.
- Mamba2 achieves more steps/hour due to parallel scan efficiency (tok/s ≈ 11,250
  vs ≈ 7,750–8,050 for the other models), hence the highest step count.
- Batch size: 4 (effective tokens/step = 8,192).

### M2RNN-CMA
- Architecture: Level m2rnn (M2RNN with CMA-ES optimized geometry, "tied" config).
  Config: dim=1920, depth=21, heads=370, n_state=16, lr=6.021e-4.
- Training started 2026-05-09. Resumed with XMA (accelerated M2RNN) backend
  at step 123,000 on 2026-05-11; continuous since.
- Uses XMA kernels from the accelerated-model-architectures package.
- Batch size: 5 (effective tokens/step = 10,240).
- GPU hours (~433h) reflect the M2RNN-CMA run's independent start date.

---

## Absent / excluded runs

### M2RNN-paper
- Config: dim=3072, depth=10, heads=759, n_state=16, lr=4.911e-4.
- Ran 2026-05-09. Stopped at step 8,400, loss ≈ 11.5 (never converged).
- Run was abandoned; no usable loss curve. Not included in Figure 3.
- Log: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_paper.log`

---

## FLOPs formula detail

    FLOPs_per_step = 6 × N_params × batch_size × chunk_size
    Total_FLOPs    = FLOPs_per_step × step

    Interpretation of 6×:
      Forward pass  ≈ 2N  (matrix multiplies dominate)
      Backward pass ≈ 4N  (gradients w.r.t. weights + activations)
    Total           = 6N

    This is consistent with the Chinchilla / scaling-law literature convention.
    SSM recurrent ops (O(N_state × d_model) per token) are absorbed into the
    denominator of the 6N approximation for sufficiently large models; the error
    is expected to be <10% for these configs.

---

## Smoothed loss columns in CSVs

`smooth_5k`  — 5,000-step centered moving average of raw loss (nats/token)
`smooth_10k` — 10,000-step centered moving average (used in Figure 3)
`smooth_50k` — 50,000-step centered moving average

The window is computed in log-entry index space. With log_every=50, a 10K-step
window corresponds to ≈200 log entries per side.

The figure renderer (`plot_normalized.py`) converts `smooth_10k` to bits/byte
at the display step using the canonical bytes/token; the CSV columns
themselves remain in their native nats/token units (the `bits_per_base`
column written by `smooth.py` uses an older 4.0-bytes/token approximation
and is preserved for backward compatibility; see the `plot_normalized.py`
header for the canonical value the figure uses).
