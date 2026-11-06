# C14 transformer consultation: the head-asymmetry verdict

Primary results use the frozen document-disjoint final-test half. NLL is nats/byte. Positive gain means dense minus device-on; every real target remains in the total.

Transformer P1 **PASS**, P2 **FAIL**. Total gain +0.138364999; device-off minus dense +0.058468789 (P2 limit ±.02).

The Emender gain is +0.766873539; transformer minus Emender gain is -0.628508539. Consulted transformer minus consulted Emender total NLL is -0.243196181.

Emender supported-target advantage against the consulted transformer: **DOES NOT HOLD**. Transformer minus Emender supported-target NLL: -0.094495863.

Historical dense transformer: 1.404434206; paired batch-128 replay minus historical batch-16 NLL: -0.000005377. Gain against the historical dense reference: +0.138370377. No prediction byte parity with the historical eval is claimed.

| Final-test stratum | Targets | Transformer dense | Transformer on | Transformer off | Gain | Emender v2 on | Emender gain | Transformer − Emender gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 214,097,022 | 1.404428829 | 1.266063829 | 1.462897618 | +0.138364999 | 1.509260010 | +0.766873539 | -0.628508539 |
| window_reset_1_16 | 27,031,327 | 1.711343480 | 1.622003794 | 1.758783136 | +0.089339687 | 1.748787269 | +0.539716162 | -0.450376475 |
| window_interior_17_128 | 187,065,695 | 1.360079113 | 1.214629869 | 1.420141632 | +0.145449245 | 1.474647893 | +0.799698185 | -0.654248941 |
| wikipedia_support | 213,177,555 | 1.401258172 | 1.261806339 | 1.459559212 | +0.139451832 | 1.503290773 | +0.769753921 | -0.630302089 |
| empty_support | 919,467 | 2.139542621 | 2.253158814 | 2.236903852 | -0.113616194 | 2.893222008 | +0.099059669 | -0.212675863 |
| supported_target | 158,891,199 | 0.982135756 | 0.717143862 | 1.048394787 | +0.264991893 | 0.811639725 | +1.116369511 | -0.851377618 |
| nonempty_zero_target | 54,286,356 | 2.627991150 | 2.855983595 | 2.662999878 | -0.227992445 | 3.527690131 | -0.244758211 | +0.016765766 |
| zero_target | 55,205,823 | 2.619855916 | 2.845943395 | 2.655903140 | -0.226087479 | 3.517122903 | -0.239031837 | +0.012944357 |
| agreement | 158,891,199 | 0.982135756 | 0.717143862 | 1.048394787 | +0.264991893 | 0.811639725 | +1.116369511 | -0.851377618 |
| argmax_agreement | 124,737,570 | 0.641944475 | 0.299683456 | 0.714918664 | +0.342261019 | 0.329834963 | +1.346700201 | -1.004439182 |
| fork | 98,425,906 | 1.810892120 | 1.736909749 | 1.859653219 | +0.073982371 | 2.010901269 | +0.515742936 | -0.441760565 |

At the registered budget, the transformer gains less from the array. Its consulted total and supported-target NLL are lower than the Emender’s. Consultation hurts zero-target positions in both heads.

## Training exposure and cost

| Arm | Steps | Targets | Active seconds |
| --- | ---: | ---: | ---: |
| Transformer dense | 480,353 | 972,375,733 | 2700.000951 |
| Transformer consultation | 281,741 | 570,344,976 | 2700.010381 |

This is a single-seed fixed-time comparison with unequal exposure; it does not isolate architecture at matched training targets. The registered rule is 2700 active seconds, with identical source-order stream from window zero, seed 3401, batch 16/context 128, BF16, AdamW 3e-4/decay .01/clip 1 and independent window resets. Consultation uses keep=.5 joint dropout across all 260 input features per position, without rescaling. Exposure differences are measured consequences of the fixed active budget.

Training wall time: 2707.114s; excluded checkpoint save/hash stalls: 1.996s. Full extraction is prepaid and separately receipted in union_service.jsonl; this is not an end-to-end extraction-inclusive speed claim.

Paired inference loop elapsed: 612.331s across all three modes, separately from training active time.

The transformer uses its registered attention head and the shared byte-exact v2 feature decoder. Phase 1–3 Emender store-kernel evidence is hash-bound, unchanged; those kernels do not execute in the transformer head. No resilient DiLoCo behavior was changed or run; its requirement IDs are not applicable.

## Calibration and combined totals

| Half | Dense | On | Off | Targets |
| --- | ---: | ---: | ---: | ---: |
| calibration | 1.421159443 | 1.271840983 | 1.480030496 | 220,107,062 |
| all | 1.412909924 | 1.268992388 | 1.471582630 | 434,204,084 |

All strata, sums and denominators for calibration/final-test/all are retained in receipts/c14_transformer_eval/table.json. P3 calibration stacking is outside this head-asymmetry arm; no P3 result is claimed. Extraction parity uses retained oracle attestations rather than a fresh native-query rerun.
