# C14 P3 emender stacking, gain decomposition, and array-alone baseline

**Soft stacking meets both P3 comparisons: 1.505908651 versus v2 alone 1.509260010 and dense+soft calibration 1.695958394. C13 stacking with exact walks: P3 FAIL.** A nonempty single-branch Wikipedia continuation is not a guaranteed Pile continuation. Exact forced walks assign zero probability to real targets; both dense+C13 and v2+C13 have infinite total NLL. No smoothing, target-conditioned abstention, or final-test tuning repairs this literal C13 row. The no-walk variant was declared in analysis_plan.json before final scoring; its PASS does not erase the exact-walk FAIL.

All NLLs are natural-log nats/byte on the frozen document-disjoint final-test half. All weights are frozen. Calibration uses all registered calibration nonempty fork positions, C13’s intercept/max-neural-probability/log1p-context/log1p-support logistic gate, normalization and bounded L-BFGS-B. Both gates and their hash freeze are written before scoring final-test. No extraction, index changes, training, union-service commands, or .pt commits.

## Final-test comparison

| Stratum | Targets | Dense | v2_on | Dense+C13 walk | v2+C13 walk | Dense+soft gate | v2+soft gate | Array alone | Dense transformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 214097022 | 2.276133551 | 1.509260010 | ∞ | ∞ | 1.695958394 | 1.505908651 | ∞ | 1.404434206 |
| window_reset_1_16 | 27031327 | 2.288503431 | 1.748787269 | ∞ | ∞ | 1.841368583 | 1.723841133 | ∞ | 1.711359374 |
| window_interior_17_128 | 187065695 | 2.274346081 | 1.474647893 | ∞ | ∞ | 1.674946362 | 1.474417018 | ∞ | 1.360082971 |
| wikipedia_support | 213177555 | 2.273044696 | 1.503290773 | ∞ | ∞ | 1.690367156 | 1.499924958 | ∞ | 1.401263582 |
| empty_support | 919467 | 2.992281678 | 2.893222008 | 2.992281678 | 2.893222008 | 2.992281678 | 2.893222008 | ∞ | 2.139540329 |
| supported_target | 158891199 | 1.928009239 | 0.811639725 | 0.720811102 | 0.676649869 | 0.955341644 | 0.768105627 | 0.663434002 | 0.982139057 |
| nonempty_zero_target | 54286356 | 3.282931920 | 3.527690131 | ∞ | ∞ | 3.841719605 | 3.641893208 | ∞ | 2.628002735 |
| zero_target | 55205823 | 3.278091066 | 3.517122903 | ∞ | ∞ | 3.827572000 | 3.629423900 | ∞ | 2.619867269 |
| fork | 98425906 | 2.526644206 | 2.010901269 | 2.167601364 | 1.997437883 | 2.167601364 | 1.997437883 | ∞ | 1.810901228 |
| nonfork | 115671116 | 2.062971104 | 1.082407593 | ∞ | ∞ | 1.294631872 | 1.087660676 | ∞ | 1.058566662 |
| forced | 114751649 | 2.055524845 | 1.067898136 | ∞ | ∞ | 1.281029166 | 1.073193310 | ∞ | 1.049905177 |
| forced_zero_target | 30751940 | 3.004971603 | 3.409054235 | ∞ | ∞ | 3.568404799 | 3.532108443 | ∞ | 2.379047773 |

Of the small soft-fusion improvement, **93.981%** comes from window-reset positions 1–16. Interior 17–128 changes by **-0.000230876**. Zero-target and nonfork NLLs worsen; all signs remain in the table.

No-walk comparison signs: beats consultation alone **True**, beats dense+soft calibration **True**, beats dense transformer **False**. Conjunctive P3 without walk: **PASS**.

Forced-walk zero-probability final targets: **30,751,940**. These targets remain in every total. Empty-support positions abstain to the neural model. Supported-target C13 loss is finite but does not rescue total P3.

The v2 soft diagnostic has total NLL **1.505908651**, change **-0.003351359** versus v2 alone and **-0.190049743** versus the actual dense+soft gate. Against the frozen historical dense transformer control **1.404434206**, its change is **+0.101474445**. The calibrated-fusion comparisons pass without walks; the predeclared primary literal C13 recipe remains a failure. The gain over v2 alone is small (about 0.22%) and has no uncertainty estimate.

The requested dense+calibration anchor “2.277” is not a committed fused final-test result: 2.277418521 is the historical raw dense calibration-half NLL; the paired raw dense final-test NLL is 2.276133549. The actual calibrated dense rows are measured above. The transformer control is the matched frozen final-window historical reference (batch 16). Its existing position-wise loss JSONL is hash-checked against the committed training receipt and identity-joined with every frozen window; the support/fork columns above are newly aggregated from those historical losses, with no transformer inference or service access.

## Supported-target gain by support length

Gain below is v2_on minus dense: negative means lower loss. Contribution is the group’s loss-sum change divided by all supported final targets; contributions add to the measured replay supported-target change (within 3e-9 of the committed −1.116369511). Gain share divides the group loss-sum change by the total supported loss-sum change, so harmful groups have negative shares. No bins were used for fitting.

| Length bytes | Supported targets | Dense NLL | v2 NLL | Array NLL | Change | Contribution | Gain share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-3 | 8,789,208 | 2.602628272 | 2.396054632 | 2.836889643 | -0.206573641 | -0.011426805 | 1.024% |
| 4-12 | 57,853,400 | 2.085441157 | 1.114137495 | 0.939296102 | -0.971303661 | -0.353658476 | 31.679% |
| 13-64 | 90,320,622 | 1.767844631 | 0.480000799 | 0.288714455 | -1.287843832 | -0.732066072 | 65.576% |
| 65+ | 1,927,969 | 1.631753646 | 0.047912427 | 0.031882591 | -1.583841220 | -0.019218162 | 1.721% |

## Calibration-half rows (fit split, not an acceptance set)

| Stratum | Targets | Dense | v2_on | Dense+C13 walk | v2+C13 walk | Dense+soft gate | v2+soft gate | Array alone | Dense transformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 220107062 | 2.277426910 | 1.505534350 | ∞ | ∞ | 1.690486868 | 1.501691995 | ∞ | 1.421162234 |
| supported_target | 163617494 | 1.932087515 | 0.809082126 | 0.715823383 | 0.672633959 | 0.952204140 | 0.765034257 | 0.656721263 | 0.999125666 |
| nonempty_zero_target | 55530339 | 3.283700310 | 3.534881398 | ∞ | ∞ | 3.844410265 | 3.649436297 | ∞ | 2.652931044 |
| empty_support | 959229 | 2.928849421 | 2.820588512 | 2.928849421 | 2.820588512 | 2.928849421 | 2.820588512 | ∞ | 2.100901931 |
| fork | 101006646 | 2.519274141 | 2.002502447 | 2.155464985 | 1.988365813 | 2.155464985 | 1.988365813 | ∞ | 1.821073394 |
| nonfork | 119100416 | 2.072321181 | 1.084065791 | ∞ | ∞ | 1.296148362 | 1.088953806 | ∞ | 1.082005695 |

## Frozen all-input-zero reference

Unchanged committed v2_off rows (all 260 inputs zero, no fusion); no replay or tuning of this arm:

| Stratum | Targets | v2_off NLL |
| --- | ---: | ---: |
| total | 214,097,022 | 2.309925375 |
| window_reset_1_16 | 27,031,327 | 2.329848827 |
| window_interior_17_128 | 187,065,695 | 2.307046401 |
| wikipedia_support | 213,177,555 | 2.307176294 |
| empty_support | 919,467 | 2.947297269 |
| supported_target | 158,891,199 | 1.977603692 |
| nonempty_zero_target | 54,286,356 | 3.271805159 |
| zero_target | 55,205,823 | 3.266400397 |
| fork | 98,425,906 | 2.545380678 |

## Fork decomposition

| Length / supported stratum | Targets | Dense NLL | v2 NLL | Change | Contribution to supported gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| supported_fork | 74,891,490 | 2.174847110 | 1.485540858 | -0.689306252 | -0.324896361 |
| supported_nonfork | 83,999,709 | 1.707936382 | 0.210810740 | -1.497125643 | -0.791473153 |
| length_1-3/supported/fork | 8,766,013 | 2.605090963 | 2.400566806 | -0.204524157 | -0.011283579 |
| length_1-3/supported/nonfork | 23,195 | 1.671911343 | 0.690782656 | -0.981128687 | -0.000143226 |
| length_4-12/supported/fork | 38,664,510 | 2.222102011 | 1.508643829 | -0.713458182 | -0.173612580 |
| length_4-12/supported/nonfork | 19,188,890 | 1.810077393 | 0.319229917 | -1.490847475 | -0.180045896 |
| length_13-64/supported/fork | 27,057,051 | 1.997285070 | 1.176153888 | -0.821131181 | -0.139827683 |
| length_13-64/supported/nonfork | 63,263,571 | 1.669715778 | 0.182264687 | -1.487451091 | -0.592238389 |
| length_65+/supported/fork | 403,916 | 0.208335234 | 0.140470583 | -0.067864651 | -0.000172518 |
| length_65+/supported/nonfork | 1,524,053 | 2.008998710 | 0.023381967 | -1.985616743 | -0.019045644 |

All-position fork: **98,425,906** targets, v2 minus dense **-0.515742936**; contribution to total change **-0.237100289** nats/byte.
All-position nonfork: **115,671,116** targets, v2 minus dense **-0.980563511**; contribution to total change **-0.529773252** nats/byte.


Medium suffixes (13–64 bytes) supply **65.576%** of the gain; supported singleton/nonfork positions supply **70.897%** across lengths. Thus neither 1–3-byte transfer nor rare 65+ matches dominate. The concentration in array-singleton medium suffixes is consistent with transferable continuation predictability; it does not identify semantic structure carried by the spine.

Lengths 1–12 account for **32.703%** of the supported gain; 65+ account for **1.721%**. This quantifies short versus long suffix transfer. Length alone cannot distinguish grammatical generalization, repeated boilerplate/whitespace, near-verbatim overlap, or learned structural computation. These are post-hoc descriptive strata, not an intervention or proof that the spine carries semantic structure. Contexts reset at each window and are at most 128 bytes here.

## Array as a language model

On supported targets, the unsmoothed array’s own NLL is **0.663434002**, versus v2_on **0.811639725** (v2 minus array **+0.148205723**). On nonempty Wikipedia support, **54,286,356** zero-target positions make array NLL infinite. Empty support abstains in fusion, but the array alone has no distribution there and is scored with q(target)=0; its full-window total is also infinite.

This is a maximal-supported-suffix empirical branch baseline, analogous in purpose to an infini-gram baseline, not a backoff/smoothed implementation of that system. Its supported-target score is conditional on the answer being in support, not a fair standalone total-LM score or a mathematical ceiling for a trained model. No Pile extraction or held-out index is constructed.

## Validation and limits

The inference pass replays dense/v2 on all 214,097,022 final targets plus the full calibration half. All 40 retained paired denominators match exactly, and means agree within 1e-8 nats/byte. V2 aggregate replay is exact; dense final total changes by +0.471769333 nats, +2.204e-9 nats/byte. Its largest stratum mean drift is retained in replay_difference.json. No root cause or byte-exact dense replay is claimed. Cache/source/window/checkpoint identities and first-shard eval parity follow the committed paired runner. Prediction and gate SHA256s, exact finite-part loss sums, infinite-loss counts, denominators, fitted gates, calibration freeze, inference resume, tests and independent validation are retained in `receipts/c14_stacking/`. The 17 GB position cache (`/home/erikg/c14_stacking_predictions.npy`) and 1.7 GB historical transformer loss join (`/home/erikg/c14_stacking_transformer_losses.npy`) stay outside git with hash receipts; all source inputs remain read-only.

Residual risks: one seed and fixed time budget, unequal training exposures, no uncertainty estimate, float32 decoded branch probabilities and metadata, BF16 neural replay, historical transformer batching difference, and support-length selection cannot establish a causal semantic mechanism. Hard-walk failure is a real foreign-array mismatch, not discarded data. No resilient DiLoCo behavior was changed; its requirement IDs are inapplicable.

Manual notes: `contact_supervisor` was searched first and is absent from the active tool catalog. No supervisor approval or reply is implied. Work stayed in this attended session with the standalone cards-2/4 GPU broker; no WorksGood service, workers, subagents, Frontier or Slurm. Unrelated workspace changes were preserved.
