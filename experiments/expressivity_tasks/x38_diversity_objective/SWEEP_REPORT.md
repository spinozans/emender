# X38 sweep receipt

All four candidates used the same stream prefix and611 B64 updates:20,021,248 tokens each. Selection was registered before training.

| Lambda | Validation perplexity | Relative to lambda0 | Mean read cosine | Within2% |
|---|---:|---:|---:|---|
| 0 | 14.630912 | +0.0000% | 0.001765 | True |
| 0.001 | 14.654112 | +0.1586% | -0.182384 | True |
| 0.01 | 14.820247 | +1.2941% | -0.300049 | True |
| 0.1 | 14.997094 | +2.5028% | -0.327056 | False |

Selected lambda: **0.01**. Lambda0.1 decorrelated more but exceeded the registered2% quality allowance. Continue the selected optimizer state from611 to3052 steps; no reset.

The CE-only baseline was already nearly uncorrelated. The <0.3 threshold alone is therefore uninformative. Cosine anticorrelation is not proof of semantic specialization. See receipts/OBSERVATIONS.md for initial and failed-launch measurements.
