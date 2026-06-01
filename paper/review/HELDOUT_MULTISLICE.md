# Held-out Multi-Slice Robustness Check

Is the held-out three-way ordering (E88 ≈ GDN, M2RNN-CMA just behind, seen on the single canonical slice) **robust** across independent held-out Pile slices, or is it **slice-specific**? We measure all three of our v0.3 1.27B checkpoints on K=5 independent held-out slices and report every slice — this is a robustness check, **not** a best-of pick.

## Protocol

- **Models / pinned checkpoints** (the same ones that produced the sane 0.966/0.966/0.961 canonical numbers):
  - E88 — `e88_postrepair_ckpt` step 1542000
  - GDN — `fla-gdn_resume_ckpt` step 2031000
  - M2RNN-CMA — `m2rnn_tied_resume_xma_ckpt` step 1491000
  - All three loaded from the **paper-pinned, rotation-immune** copies under `/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/` (sha-verified). Live training rotated E88 step 1542000 out of `/tmp` partway through this run; the pinned copy (sha `64ae1e7e…`) is byte-identical and was used for every E88 slice, so the E88 column is internally consistent and ties back to the canonical 0.9661.
- **Forward**: `scripts/measure_pile_bpb_elman.py` — builds each model exactly as `train.py` does, loads `model_state_dict` strict, applies the schedule-free **y-mode** (training-weights) swap, then runs the sliding-window protocol (context 2048, stride 1024, every token scored once). Run on a single dedicated GPU (GPU 0, then a freed GPU after live training was moved off it mid-run); BPB is invariant to which GPU and to batch size.
- **BPB** = total_NLL_nats / (slice_UTF8_bytes · ln2), per slice. No 3.92 constant.
- **Sanity gate**: per-run block-loss on the first 2048-token block must lie in [1.5, 4.0] nats (model train loss ~2.6) before its BPB is trusted.

## Slices

| slice | requested offset | actual start byte | offset frac | bytes | sha256 (first 16) |
|---|---:|---:|---:|---:|---|
| canonical_1e12 (canonical) | 1,000,000,000,000 | 1,000,000,001,956 | 0.7646 | 9,999,511 | `3e4241a946e76c31` |
| slice_frac0.137 | 179,183,779,947 | 179,183,780,074 | 0.1370 | 9,999,374 | `a5d73ac0eabb13cb` |
| slice_frac0.341 | 445,997,583,666 | 445,997,583,669 | 0.3410 | 9,999,642 | `8794f1b70d9a3a0e` |
| slice_frac0.523 | 684,037,349,728 | 684,037,349,952 | 0.5230 | 9,999,810 | `7efc37aaf089789d` |
| slice_frac0.911 | 1,191,506,741,113 | 1,191,506,741,133 | 0.9110 | 9,999,683 | `9fb1834babd7187c` |

All five slices are independent, non-overlapping, valid-UTF-8, and spread across the 1.31 TB file. The canonical slice (sha `3e4241a9…`, offset 1e12) is included so the numbers tie back to the canonical BPB table.

## Held-out BPB per slice

| slice (offset frac) | E88 | GDN | M2RNN-CMA | lowest |
|---|---:|---:|---:|---|
| canonical_1e12 (0.765) | 0.9661 | 0.9661 | **0.9613** | M2RNN-CMA |
| slice_frac0.137 (0.137) | 0.9492 | **0.9207** | 0.9458 | GDN |
| slice_frac0.341 (0.341) | 0.9441 | 0.9409 | **0.9390** | M2RNN-CMA |
| slice_frac0.523 (0.523) | 0.9771 | 0.9784 | **0.9736** | M2RNN-CMA |
| slice_frac0.911 (0.911) | 0.9772 | 0.9786 | **0.9704** | M2RNN-CMA |

Lowest BPB per slice is **bold**.

## Per-slice ordering (lowest → highest BPB)

- **canonical_1e12**: M2RNN-CMA 0.9613 < GDN 0.9661 < E88 0.9661
- **slice_frac0.137**: GDN 0.9207 < M2RNN-CMA 0.9458 < E88 0.9492
- **slice_frac0.341**: M2RNN-CMA 0.9390 < GDN 0.9409 < E88 0.9441
- **slice_frac0.523**: M2RNN-CMA 0.9736 < E88 0.9771 < GDN 0.9784
- **slice_frac0.911**: M2RNN-CMA 0.9704 < E88 0.9772 < GDN 0.9786

## Aggregate across slices

| model | mean BPB | std (cross-slice) | min | max | # slices lowest |
|---|---:|---:|---:|---:|---:|
| E88 | 0.9627 | 0.0155 | 0.9441 | 0.9772 | 0 |
| GDN | 0.9569 | 0.0254 | 0.9207 | 0.9786 | 1 |
| M2RNN-CMA | 0.9580 | 0.0152 | 0.9390 | 0.9736 | 4 |

## Is any ordering statistically meaningful?

- **GDN** vs **M2RNN-CMA**: mean gap = 0.0011 BPB (0.05× the pooled cross-slice std) → WITHIN noise.
- **M2RNN-CMA** vs **E88**: mean gap = 0.0047 BPB (0.31× the pooled cross-slice std) → WITHIN noise.

- Full spread of means = 0.0058 BPB; largest cross-slice std = 0.0254 BPB.

## Verdict

**TIE / BAND.** Across 5 independent slices the three models sit within a 0.0058 BPB band — smaller than the largest cross-slice std (0.0254). The mean ordering is GDN < M2RNN-CMA < E88, but the gaps do not exceed cross-slice noise, so the three-way ordering is **not statistically meaningful**: E88, GDN, and M2RNN-CMA are effectively tied on held-out Pile. Which model is lowest is slice-dependent (lowest-counts: E88 0/5, GDN 1/5, M2RNN-CMA 4/5).

## Sanity gate (block-loss nats, first 2048-token block)

| slice | E88 | GDN | M2RNN-CMA |
|---|---:|---:|---:|
| canonical_1e12 | 1.809 | 1.603 | 1.711 |
| slice_frac0.137 | 3.202 | 3.265 | 3.250 |
| slice_frac0.341 | 2.323 | 2.363 | 2.301 |
| slice_frac0.523 | 2.703 | 2.769 | 2.720 |
| slice_frac0.911 | 3.022 | 3.120 | 2.967 |

All gated runs lie in [1.5, 4.0] nats (✗ = gate failed, BPB not trusted).

---

*Source data: `paper/review/heldout_multislice_slices.json` (slice manifest), `/tmp/heldout_slices/results/` (per-run JSON). Generated by `scripts/gen_heldout_multislice_report.py` from real single-GPU measurements (no fabricated numbers; every cell sanity-gated).*
