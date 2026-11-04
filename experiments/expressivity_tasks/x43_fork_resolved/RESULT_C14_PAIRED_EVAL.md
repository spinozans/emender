# C14 Emender paired evaluation at Pile scale

V2 is primary; v1 is retained. NLL averages are natural-log nats/byte; total NLL sums are nats. Weights are frozen: no retraining. The primary verdict uses the preregistered document-disjoint final-test half. Calibration and combined rows are reported separately.

v1: P1 **PASS**, P2 **FAIL**; v2: P1 **PASS**, P2 **FAIL**.

Frozen cache: 3,431,242 windows / 434,204,084 targets, 44,953,751,774 bytes; windows SHA256 `2ab51ce59ad94010d78f6817f29fa06fc5ee64feea766376ff0e35347f196d4d`, journal SHA256 `00e5c17a75180a367162cef6be5046a721b3d9d58ec273ef2e4a90f68522ea3b`.

All shard hashes, contiguity and recorded extraction parity attestations pass. Every eval position passes the c14_verify_sum compensated/sequential float32 entropy oracle. This reuses extraction attestations; no fresh native extractor query rerun is claimed. Feature assembly uses the shared v2 decoder.

## final_test

| Arm/mode | Total NLL sum | Total | Reset 1–16 | Interior 17–128 | Wikipedia support | Empty support | Supported target | Nonempty zero-target | All zero-target | Branch agreement | Argmax agreement (supplemental) | Fork |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 487313414.450466 | 2.276133549 | 2.288503431 | 2.274346079 | 2.273044694 | 2.992281678 | 1.928009236 | 3.282931920 | 3.278091066 | 1.928009236 | 1.676535164 | 2.526644206 |
| v1_on | 342860761.414850 | 1.601427046 | 1.826875245 | 1.568849378 | 1.593819500 | 3.365229293 | 0.845246898 | 3.784823043 | 3.777834602 | 0.845246898 | 0.328326285 | 2.128684564 |
| v1_off | 528272048.893199 | 2.467442302 | 2.479356717 | 2.465720648 | 2.463548679 | 3.370175188 | 2.143647950 | 3.399868854 | 3.399374299 | 2.143647950 | 1.911526631 | 2.695613949 |
| v2_on | 323128073.564732 | 1.509260010 | 1.748787269 | 1.474647893 | 1.503290773 | 2.893222008 | 0.811639725 | 3.527690131 | 3.517122903 | 0.811639725 | 0.329834963 | 2.010901269 |
| v2_off | 494548143.877989 | 2.309925375 | 2.329848827 | 2.307046401 | 2.307176294 | 2.947297269 | 1.977603692 | 3.271805159 | 3.266400397 | 1.977603692 | 1.737352686 | 2.545380678 |

| Stratum | Targets (identical in all five modes) |
| --- | ---: |
| Total | 214,097,022 |
| Reset 1–16 | 27,031,327 |
| Interior 17–128 | 187,065,695 |
| Wikipedia support | 213,177,555 |
| Empty support | 919,467 |
| Supported target | 158,891,199 |
| Nonempty zero-target | 54,286,356 |
| All zero-target | 55,205,823 |
| Branch agreement | 158,891,199 |
| Argmax agreement (supplemental) | 124,737,570 |
| Fork | 98,425,906 |

## calibration

| Arm/mode | Total NLL sum | Total | Reset 1–16 | Interior 17–128 | Wikipedia support | Empty support | Supported target | Nonempty zero-target | All zero-target | Branch agreement | Argmax agreement (supplemental) | Fork |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 501277746.102520 | 2.277426910 | 2.289484654 | 2.275684984 | 2.274575577 | 2.928849421 | 1.932087515 | 3.283700310 | 3.277674715 | 1.932087515 | 1.682506544 | 2.519274141 |
| v1_on | 351348348.166153 | 1.596261133 | 1.823124965 | 1.563487180 | 1.588940740 | 3.268696302 | 0.842827810 | 3.787326883 | 3.778520204 | 0.842827810 | 0.329964261 | 2.116452673 |
| v1_off | 542614232.487434 | 2.465228637 | 2.477460407 | 2.463461571 | 2.461690436 | 3.273574945 | 2.143944927 | 3.397912414 | 3.395801085 | 2.143944927 | 1.913571998 | 2.683947537 |
| v2_on | 331378742.497237 | 1.505534350 | 1.745774202 | 1.470828026 | 1.499778244 | 2.820588512 | 0.809082126 | 3.534881398 | 3.522752248 | 0.809082126 | 0.329803919 | 2.002502447 |
| v2_off | 508456300.989084 | 2.310040834 | 2.330702128 | 2.307055993 | 2.307556491 | 2.877619816 | 1.979427627 | 3.274372533 | 3.267635418 | 1.979427627 | 1.740387426 | 2.537897986 |

| Stratum | Targets (identical in all five modes) |
| --- | ---: |
| Total | 220,107,062 |
| Reset 1–16 | 27,784,009 |
| Interior 17–128 | 192,323,053 |
| Wikipedia support | 219,147,833 |
| Empty support | 959,229 |
| Supported target | 163,617,494 |
| Nonempty zero-target | 55,530,339 |
| All zero-target | 56,489,568 |
| Branch agreement | 163,617,494 |
| Argmax agreement (supplemental) | 128,734,602 |
| Fork | 101,006,646 |

## all

| Arm/mode | Total NLL sum | Total | Reset 1–16 | Interior 17–128 | Wikipedia support | Empty support | Supported target | Nonempty zero-target | All zero-target | Branch agreement | Argmax agreement (supplemental) | Fork |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 988591160.552990 | 2.276789180 | 2.289000779 | 2.275024808 | 2.273820706 | 2.959894288 | 1.930078259 | 3.283320467 | 3.277880498 | 1.930078259 | 1.679567936 | 2.522911487 |
| v1_on | 694209109.581006 | 1.598808337 | 1.824974357 | 1.566131126 | 1.591346433 | 3.315941253 | 0.844019628 | 3.786089144 | 3.778181343 | 0.844019628 | 0.329158187 | 2.122489476 |
| v1_off | 1070886281.380633 | 2.466320150 | 2.478395543 | 2.464575457 | 2.462606726 | 3.320852810 | 2.143798615 | 3.398879553 | 3.397567158 | 2.143798615 | 1.912565441 | 2.689705259 |
| v2_on | 654506816.061968 | 1.507371396 | 1.747260049 | 1.472711493 | 1.501510255 | 2.856136628 | 0.810342185 | 3.531326495 | 3.519969925 | 0.810342185 | 0.329819196 | 2.006647516 |
| v2_off | 1003004444.867070 | 2.309983903 | 2.330281336 | 2.307051264 | 2.307369018 | 2.911721192 | 1.978529024 | 3.273103387 | 3.267025005 | 1.978529024 | 1.738893984 | 2.541590917 |

| Stratum | Targets (identical in all five modes) |
| --- | ---: |
| Total | 434,204,084 |
| Reset 1–16 | 54,815,336 |
| Interior 17–128 | 379,388,748 |
| Wikipedia support | 432,325,388 |
| Empty support | 1,878,696 |
| Supported target | 322,508,693 |
| Nonempty zero-target | 109,816,695 |
| All zero-target | 111,695,391 |
| Branch agreement | 322,508,693 |
| Argmax agreement (supplemental) | 253,472,172 |
| Fork | 199,432,552 |

## Preregistered predictions: all signs retained

Positive gain means dense minus consultation device-on NLL. Wikipedia support means nonempty nonterminal outgoing counts; supported target means q(target)>0. All zero-target includes empty and nonempty zero-target. Registered branch agreement means positive target count (c14_coverage.py branch_agreement_supported), identical to supported-target here; argmax agreement is supplemental. These overlapping diagnostics never filter the total loss.

| Arm | Total gain | Supported-target gain | Zero-target gain | Agreement gain | Coverage gain (support − empty) | Supported-target − nonempty-zero-target gain | P1 | Off − dense | P2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| v1 | +0.674706503 | +1.082762339 | -0.499743536 | +1.082762339 | +1.052172809 | +1.584653461 | PASS | +0.191308754 | FAIL |
| v2 | +0.766873539 | +1.116369511 | -0.239031837 | +1.116369511 | +0.670694251 | +1.361127722 | PASS | +0.033791827 | FAIL |

| Final-test stratum | v1 gain | v2 gain |
| --- | ---: | ---: |
| total | +0.674706503 | +0.766873539 |
| wikipedia_support | +0.679225194 | +0.769753921 |
| empty_support | -0.372947615 | +0.099059669 |
| supported_target | +1.082762339 | +1.116369511 |
| nonempty_zero_target | -0.501891123 | -0.244758211 |
| zero_target | -0.499743536 | -0.239031837 |
| agreement | +1.082762339 | +1.116369511 |
| fork | +0.397959642 | +0.515742936 |

P1 requires lower total final-test NLL and larger nonempty-support, supported-target and branch-agreement gains than all zero-target gains; the separate support/empty and nonempty-zero-target comparisons above retain the coverage interpretation. P2 requires absolute device-off minus dense NLL at most .02. Device-off zeros every one of the 260 features at every position, with evaluation dropout disabled.

## Matched active training budget and unequal exposure

| Arm | Steps | Training targets | Active seconds |
| --- | ---: | ---: | ---: |
| dense | 111,596 | 225,880,702 | 2700.022528 |
| v1 | 27,401 | 55,447,993 | 2700.021958 |
| v2 | 88,520 | 179,162,804 | 2700.019260 |

All three checkpoints use the same Emender architecture with 2,133,700 trainable parameters, seed 3401, training batch 16/context 128, BF16 and AdamW 3e-4/decay .01/clip 1. Consultation training uses independent keep=.5 jointly over all 260 scalars without rescaling. Evaluation uses BF16 autocast, float32 per-target losses and float64 sums. Each source document is digest-checked against frozen offsets; every cached window identity and length is matched to the frozen window file. Windows reset independently and tail padding is masked. Inference batches contain at most 128 windows versus historical dense evaluation batch 16. The independent validation receipt retains the small dense replay difference; no byte-exact historical loss claim.

Evaluation process inference-loop time: 1775.854s; includes cache reads, decode, source checks, all five modes and aggregate receipts; excludes imports, model loading, offsets scan, preflight full cache audit, compiler setup and broker lease wait. No extraction or training cost is hidden in this measurement. GPU broker card 2.

## Receipts and limits

`receipts/c14_paired_eval/` retains exact sums/denominators (`table.json`), running cumulative NLL curves and exact resume state (`resume.json`, `curves.svg`), cache identity/oracle audit, checkpoint and training-exposure hashes, verdicts and independent validation. `frozen_input_hashes.json` binds 19 original committed protocols, training curves/configs, the v2 primary amendment and decoder proofs. `transformer_progress.json` is a read-only live-service snapshot with conditional projections; the service was not modified.

One seed and one fixed time budget; different training exposure is measured, not controlled away. No confidence intervals, matched-step causal claim, P3, coverage swap, calibration fit or transformer quality verdict. Early exploratory inference had a nonfinite loss and nondeterministic loss bits; failed logs and repeat diagnostics remain in MANUAL_NOTES.md. The adopted run requires deterministic algorithms and a fixed cuBLAS workspace, checks first-shard loss-bit parity against normal eval, and fails closed on every nonfinite loss. These checks do not prove exhaustive native-kernel determinism. No resilient DiLoCo work was performed; its gap-matrix IDs do not apply. No .pt files are committed. The enabled tool catalog has no contact_supervisor; no supervisor reply is implied.
