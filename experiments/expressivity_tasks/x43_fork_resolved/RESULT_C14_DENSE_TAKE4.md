# C14 dense control — take 4

Both registered dense heads completed 2700 active training seconds. Emender reused the surviving checkpoint without retraining; transformer trained from seed 3401. Both complete evaluations use the original frozen windows, batch 16, context 128, BF16, and all-zero consultation. GPU broker granted physical card 2.

The read-only pile.txt binding matches 7,831,213 document receipts; two token-truncated documents are excluded. Independent validation rehashed all 7,831,211 admitted document slices (43,494,394,389 bytes), checked offsets and document-disjoint halves, and replayed every recorded target loss and position-stratum aggregate. Both maximum NLL replay errors are 0.0 (required <1e-8). This is aggregation of recorded losses, not independent model inference. No corpus text was materialized in this take.

| Split / position stratum | Targets per arm | Emender NLL | Transformer NLL |
|---|---:|---:|---:|
| calibration / all | 220,107,062 | 2.277418521 | 1.421162234 |
| calibration / window_reset_1_16 | 27,784,009 | 2.289452649 | 1.723704214 |
| calibration / window_interior_17_128 | 192,323,053 | 2.275680007 | 1.377455414 |
| final_test / all | 214,097,022 | 2.276134617 | 1.404434206 |
| final_test / window_reset_1_16 | 27,031,327 | 2.288481328 | 1.711359374 |
| final_test / window_interior_17_128 | 187,065,695 | 2.274350496 | 1.360082971 |

All 3,431,242 evaluation windows stay within a document; there are 434,204,084 real targets per arm. Position strata refer to independent window resets, not support/fork or consultation strata.

| Arm | Steps | Training targets | Active training seconds | Complete evaluation seconds |
|---|---:|---:|---:|---:|
| emender | 111,596 | 225,880,702 | 2700.022528 | 4125.396221 |
| transformer | 480,353 | 972,375,733 | 2700.000951 | 639.874841 |

The inventory found no training.json, one completed Emender training curve/checkpoint, 412,925 complete prediction rows (52,289,140 targets), and no transformer checkpoint. The surviving checkpoint had no earlier receipted hash. Take 4 hash-bound it before reuse; its hash is unchanged in the final receipts. The exact original launch code was recovered and checked against the original code hash.

**Retained failure:** the optional byte-exact hash comparison of the first 412,925 reevaluated rows against the old partial prediction file failed. That file had already been overwritten by the authorized full reevaluation, so numerical differences cannot be quantified. See `receipts/c14_dense/emender_recovery_parity.json`. Full fresh evaluation is the reported result; historical byte parity is not claimed.

Six boundary/padding tests pass. `validation.json` is green and records the validator source hash. Validation checked checkpoint hashes, the immutable window hash, every admitted document digest, and the complete loss/stratum aggregates. `training.json`, `checkpoints_take4.json`, and the consoles retain both arms’ evidence. Checkpoints and per-target predictions remain outside git at /home/erikg/c14_dense_bytes. Historical bytes.bin from take 2 is unused.

One seed and fixed time budget; arms saw different numbers of training targets. Consultation predictions P1/P2/P3, support/fork strata, calibration stacking, and significance remain untested. The earlier failed single-shard result remains historical evidence.
