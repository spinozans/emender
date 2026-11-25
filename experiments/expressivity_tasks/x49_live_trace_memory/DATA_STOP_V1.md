# X49 v1 panel reservation refused (metadata only)

No model or corpus payload was opened by the reservation. At local `da3703bd7042432f82b549068dde13e0b51b43d2` (private `output/x49-live/data-gate-v2.log`, SHA-256 `20ce01ee01ecf1ac4fd9303a2c51a01c35599a397538811ed3c958028ea59d30`), the registered selector failed before it could write `reservation.json` or extract a window:

| Unique eligible document SHAs | TRAIN | Calibration | Final |
|---|---:|---:|---:|
| Beyond the offset cutoff, length ≥4097 | 36,155 | 381 | 388 |
| Minus historical manifest memberships | −9,621 | −271 | −123 |
| Minus additional reconstructed early-panel memberships | −19 | 0 | 0 |
| **Remaining** | **26,515** | **110** | **265** |
| Required | 1,024 | **128** | 128 |

An independent set-count over the complete SHA-bound metadata agreed. The 18-document calibration shortfall is a real veto, not an optimizer result. No document was moved between splits, sampled twice, or substituted. No previous evaluation losses were consulted to choose documents. The earlier development run failed a synthetic negative test (its all-zero replacement could equal the original bytes). Correcting it to flip one actual bit made both tests pass before the real reservation was attempted.

**V1 natural work is stopped.** Synthetic and retained-checkpoint qualification can continue. **Proposed v2, not applied:** keep the same exclusions, order, full 4,096-target windows, 1,024 TRAIN, 128 final, three seeds, matched budgets and all nomination thresholds. Use all 110 eligible calibration documents strictly for the registered stage checks, no per-arm selection. This is a feasibility amendment, not a scientific need to shrink the final test. A versioned decision must precede any extraction/learning under it; never overwrite v1 or loosen an exclusion. The alternate decision is to accept the v1 no-run.
