# X38 measured result

Selected lambda: 0.01. Final tokens: 100007936.
Read cosine: 0.000750 -> -0.313177; monotonic decrease: False.
Slot JS bits: 0.028949 -> 0.019245.
Perplexity ratio to historical a: 1.114026; within2%: False; over5% worse: True.
Identical occupancy stories: 16/16 (expected).

See receipts/summary.json for all monotonicity tests and failures. Control is B8 and4096 tokens shorter: not an exact matched-token comparison. Slot identity compares physical allocation indices. No causal semantic specialization claim.

The registered quality band fails: final perplexity is 6.734027, **11.4026% worse than arm(a)** (6.044765). The secondary B64/3052-step arm(c) comparison is +0.6133%; its perplexity is6.692981. Thus the full historical gap cannot be attributed to the diversity penalty. X37(c) used a different pinned X36 version; source fingerprints are retained.

**Prediction outcomes:** mean read cosine meets<0.3, but was already<0.3 at initialization. Its decrease is not monotonic: -0.300552 at512 becomes -0.300049 at611. All six W_in pair distances and all four init distances grow monotonically at registered milestones. Slot JS fails the predicted growth: it peaks0.078867 at512 then falls to0.019245, below its initial0.028949. Expected final occupancy equality holds16/16. These results show forced read anticorrelation, not useful slot specialization. The registered adverse-quality/null outcome holds against(a); incremental cost versus(c) is much smaller.

Final nonzero-read pair coverage: mean0.994439, minimum0.983051. Zero-vector exclusion does not explain the negative correlations.

Final depth probes (two corruptions per depth):

| Bin | Depths | Accuracy |
|---|---|---:|
| ID | [16, 32, 64, 128] | 0.5000 |
| 2x | [256] | 0.5000 |
| 4x | [512] | 0.5000 |
| 9x | [1152] | 0.5000 |

Final context probes:

| Context | Windows | Perplexity |
|---:|---:|---:|
| 512 | 2 | 7.268887 |
| 1024 | 2 | 8.200735 |
| 2048 | 2 | 11.253822 |
| 4096 | 2 | 17.524958 |

Final story probes cover the same16 stories. Full scores and per-head telemetry are in receipts/winner_probe.json; final slot distributions are in eval_0.01_3052.json. The simple signed-cosine objective is the only intervention; no balancing term was added.

Residual risks: single seed and8192 validation tokens; historical(a) B8 versus B64 and4096 fewer endpoint tokens; physical-slot JS is not semantic identity; cosine can reward rotations/sign changes rather than useful roles. Total parameter matching includes frozen geometry tables: arm30,269,712 total /21,846,796 trainable versus widened base30,269,863 total/trainable (relative total discrepancy0.000498846%). This receipts total counts, not equal trainable capacity or a new trained baseline.

The interrupted telemetry launch at step0 remains a failure under receipts/aborted_initial_telemetry/. No optimizer updates preceded the telemetry fix. Pretraining gates:179 oracle tests,10 mechanism/objective tests,119 pinned-snapshot tests passed. Final protocol/completeness audit is receipts/artifact_validation.json. No .pt files are included in the commit. Resilient DiLoCo requirement IDs: not applicable; no resilient training behavior changed.
