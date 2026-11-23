# Fixed XSA read on two frozen neural predictors

Both bases pass the primary prediction-quality clause of `PREREGISTRATION_SESSION_NEURAL.md`. Same 114,688 target bytes in 64 document-disjoint final-test windows; both selected lambda=1 on calibration. Values below are nats/byte, lower is better. Neural short blocks are 256 input bytes; full XSA history is the visible prefix in that document window (at most 2047 bytes). Each reloaded checkpoint first reproduced its separate original calibration score exactly.

| Frozen base | Short off | Short + full XSA | Short + last-256 XSA | Short + rotated XSA | 2048-input off | Full-read gain (95% paired document bootstrap) |
|---|---:|---:|---:|---:|---:|---:|
| GDN2 | 1.65346 | **1.49981** | 1.56078 | 1.84041 | 1.62629 | 0.15365 [0.10151, 0.21682] |
| K15 | 1.67351 | **1.51751** | 1.57945 | 1.86020 | 1.65124 | 0.15600 [0.10395, 0.21989] |

The fixed unigram plus full-index reference scored 3.06279 on these targets. Both neural models benefited beyond short lookup and beyond their own longer neural context. Indexed GDN2 still has the lower point estimate by 0.01770; these gains do not show a special advantage of K15's writable memory. This is offline scoring from causal native-session replies, not a measurement of combined append/query/decode latency. No model parameters or addresses were learned in this comparison. See the separate `RESULT_SESSION_CONTROLLER.md` for the small model with a jointly trained read gate. The open large-model experiment is to train that decision against **each actual base's** logits, with the fixed mixtures here as controls.

Receipts: `receipts/xsa_frozen_{gdn2,k15}.json` (per-document NLL, original endpoints, checkpoint hashes, lambda grids and common target-manifest hash). No board or serving-speed verdict.
