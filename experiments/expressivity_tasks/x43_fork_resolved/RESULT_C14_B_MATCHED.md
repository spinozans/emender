# C14-B matched-exposure causality verdict

Terminal exit 0; all 26,807 shards, 3,431,242 document-bounded windows and 434,204,084 masked real targets verified. Each final-test mode has 214,097,022 targets; calibration remains document-disjoint. All 66 row identities and denominators match every frozen triangle arm, with NLL sum/count and calibration plus final-test checks passing. Strata overlap and must not be summed. Full precision and config/source/index/window/checkpoint bindings: receipts/c14_b_matched/HARVEST_COMPLETE.json and eval_resume.json.gz. Training journal: training.json.gz; preregistered boundary gates and prefix receipts remain retained. EVAL_LAUNCH.json binds the phase1/2 exact state/replay/gradient equivalence and phase3 checks for the substituted evaluation kernel; this is the basis for arithmetic comparability, not a new training implementation.

PT1: final-test twin **1.565270853** minus budgeted **1.584700014** = **-0.019429161 nats/byte**. The registered negative sign holds: unbudgeted learns better at matched exposure in this single-seed recipe; budget pressure's observed fixed-time value was compute shaping. The margin is small (about .0194 nats, 1.23% of budgeted NLL), with no uncertainty estimate. This does not isolate pressure from capacity, implementation or training trajectory and is not a general law. Twin training completed 35,419 updates / 71,677,042 targets in 16,094.703 active seconds, versus budgeted 2700.032 seconds, about 5.96x active time. Evaluation took 6241.789 seconds.

PT2: final-test supported / zero-target / fork NLL is twin .853348213 / 3.614298214 / 2.071906972, budgeted .856875769 / 3.679494742 / 2.085738874, old unbudgeted .922328061 / 4.002240227 / 2.253977537. The triangle's actual budgeted fork is 2.085738874; 2.011 is not this table's fork baseline. Reset/interior is twin 1.774828371 / 1.534989417 versus budgeted 1.772668719 / 1.557538200: reset is slightly worse, interior better. These are descriptive single-seed stratum differences, not significance tests.

PT3: dense-anchor improvement per million targets is 0.009917579, 2.3170x passive's endpoint normalization, versus old unbudgeted .038715 (9.045x). The early efficiency advantage translates to a slightly better matched-exposure endpoint, but the 9x magnitude saturates to about 2.32x in this normalization; it does not extrapolate proportionally. These independent endpoints do not measure a learning slope, matched-loss sample complexity or an interpolated curve. Passive remains better in absolute NLL (1.509260010), at 179,162,804 targets. Single seed limits all efficiency and sign claims.

Triangle re-closure: lifecycle operations' fixed-time advantage over append-only is 0.661320101 nats, despite slightly fewer targets. This supports aggregate lifecycle value, without isolating individual operations. The matched twin supplies the missing budget-pressure comparison and reverses the fixed-time quality ordering at equal exposure, under PT1's single-seed limitation. Append-only remains at 75,394,452 targets; the three original arms are not exactly exposure matched. Capacity/implementation, trajectory and active compute remain confounded. This evidence informs the operator's unbounded-family launch call; no new launch decision or authorization is made here.

| Mode / split / stratum | Targets | Budgeted | Append-only | Unbudgeted 14.5M | Twin 71.7M |
|---|---:|---:|---:|---:|---:|
| B1_on/all/total | 434,204,084 | 1.581844925 | 2.243243320 | 1.713985184 | 1.562824636 |
| B1_on/all/window_reset_1_16 | 54,815,336 | 1.770575453 | 1.859236677 | 1.887262972 | 1.772707715 |
| B1_on/all/window_interior_17_128 | 379,388,748 | 1.554576517 | 2.298725864 | 1.688949438 | 1.532500037 |
| B1_on/all/wikipedia_support | 432,325,388 | 1.575678004 | 2.235782863 | 1.705765587 | 1.556737290 |
| B1_on/all/empty_support | 1,878,696 | 3.000976320 | 3.960043003 | 3.605478187 | 2.963644195 |
| B1_on/all/supported_target | 322,508,693 | 0.855293015 | 1.745273253 | 0.921938262 | 0.852074580 |
| B1_on/all/nonempty_zero_target | 109,816,695 | 3.691298232 | 3.676307125 | 4.007702701 | 3.626184470 |
| B1_on/all/zero_target | 111,695,391 | 3.679687144 | 3.681079510 | 4.000937359 | 3.615040664 |
| B1_on/all/fork | 199,432,552 | 2.079896312 | 2.542908792 | 2.247827672 | 2.066673263 |
| B1_on/all/agreement | 322,508,693 | 0.855293015 | 1.745273253 | 0.921938262 | 0.852074580 |
| B1_on/all/argmax_agreement | 253,472,172 | 0.365983238 | 1.401428603 | 0.386095425 | 0.365584945 |
| B1_on/calibration/total | 220,107,062 | 1.579067795 | 2.240542345 | 1.711542493 | 1.560445213 |
| B1_on/calibration/window_reset_1_16 | 27,784,009 | 1.768538894 | 1.857033869 | 1.885555556 | 1.770644508 |
| B1_on/calibration/window_interior_17_128 | 192,323,053 | 1.551695794 | 2.295946015 | 1.686403642 | 1.530078707 |
| B1_on/calibration/wikipedia_support | 219,147,833 | 1.572979014 | 2.233146791 | 1.703330545 | 1.554447188 |
| B1_on/calibration/empty_support | 959,229 | 2.970125902 | 3.930148898 | 3.587664431 | 2.930768894 |
| B1_on/calibration/supported_target | 163,617,494 | 0.853755982 | 1.743719434 | 0.921559724 | 0.850837737 |
| B1_on/calibration/nonempty_zero_target | 55,530,339 | 3.692135355 | 3.675221144 | 4.006780966 | 3.627598859 |
| B1_on/calibration/zero_target | 56,489,568 | 3.679875173 | 3.679549981 | 3.999664099 | 3.615766240 |
| B1_on/calibration/fork | 101,006,646 | 2.074203028 | 2.536420499 | 2.241834938 | 2.061573276 |
| B1_on/calibration/agreement | 163,617,494 | 0.853755982 | 1.743719434 | 0.921559724 | 0.850837737 |
| B1_on/calibration/argmax_agreement | 128,734,602 | 0.366057223 | 1.401021885 | 0.388164731 | 0.365844698 |
| B1_on/final_test/total | 214,097,022 | 1.584700014 | 2.246020115 | 1.716496444 | 1.565270853 |
| B1_on/final_test/window_reset_1_16 | 27,031,327 | 1.772668719 | 1.861500822 | 1.889017930 | 1.774828371 |
| B1_on/final_test/window_interior_17_128 | 187,065,695 | 1.557538200 | 2.301583840 | 1.691566782 | 1.534989417 |
| B1_on/final_test/wikipedia_support | 213,177,555 | 1.578452583 | 2.238492762 | 1.708268825 | 1.559091529 |
| B1_on/final_test/empty_support | 919,467 | 3.033160852 | 3.991229868 | 3.624062292 | 2.997941176 |
| B1_on/final_test/supported_target | 158,891,199 | 0.856875769 | 1.746873291 | 0.922328061 | 0.853348213 |
| B1_on/final_test/nonempty_zero_target | 54,286,356 | 3.690441925 | 3.677417991 | 4.008645559 | 3.624737669 |
| B1_on/final_test/zero_target | 55,205,823 | 3.679494742 | 3.682644607 | 4.002240227 | 3.614298214 |
| B1_on/final_test/fork | 98,425,906 | 2.085738874 | 2.549567209 | 2.253977537 | 2.071906972 |
| B1_on/final_test/agreement | 158,891,199 | 0.856875769 | 1.746873291 | 0.922328061 | 0.853348213 |
| B1_on/final_test/argmax_agreement | 124,737,570 | 0.365906883 | 1.401848354 | 0.383959811 | 0.365316868 |
| B1_off/all/total | 434,204,084 | 2.429446425 | 2.709220798 | 2.766010077 | 2.383652692 |
| B1_off/all/window_reset_1_16 | 54,815,336 | 2.461731925 | 2.699154373 | 2.750134675 | 2.449299230 |
| B1_off/all/window_interior_17_128 | 379,388,748 | 2.424781710 | 2.710675228 | 2.768303807 | 2.374167864 |
| B1_off/all/wikipedia_support | 432,325,388 | 2.427011221 | 2.703904966 | 2.762247804 | 2.381151586 |
| B1_off/all/empty_support | 1,878,696 | 2.989835561 | 3.932499671 | 3.631784165 | 2.959206931 |
| B1_off/all/supported_target | 322,508,693 | 2.121114041 | 2.421045704 | 2.471469275 | 2.067004639 |
| B1_off/all/nonempty_zero_target | 109,816,695 | 3.325367338 | 3.534603530 | 3.616203601 | 3.303735548 |
| B1_off/all/zero_target | 111,695,391 | 3.319723755 | 3.541296070 | 3.616465663 | 3.297940639 |
| B1_off/all/fork | 199,432,552 | 2.642014405 | 2.908659091 | 2.973919900 | 2.611728062 |
| B1_off/all/agreement | 322,508,693 | 2.121114041 | 2.421045704 | 2.471469275 | 2.067004639 |
| B1_off/all/argmax_agreement | 253,472,172 | 1.898971875 | 2.222014744 | 2.274509056 | 1.837059246 |
| B1_off/calibration/total | 220,107,062 | 2.428767396 | 2.708369987 | 2.765869402 | 2.383718699 |
| B1_off/calibration/window_reset_1_16 | 27,784,009 | 2.460773731 | 2.698312654 | 2.750347567 | 2.448463184 |
| B1_off/calibration/window_interior_17_128 | 192,323,053 | 2.424143591 | 2.709822922 | 2.768111769 | 2.374365368 |
| B1_off/calibration/wikipedia_support | 219,147,833 | 2.426444941 | 2.703103399 | 2.762169758 | 2.381344038 |
| B1_off/calibration/empty_support | 959,229 | 2.959361317 | 3.911587703 | 3.611099408 | 2.926239765 |
| B1_off/calibration/supported_target | 163,617,494 | 2.121445208 | 2.421145840 | 2.472916656 | 2.068051484 |
| B1_off/calibration/nonempty_zero_target | 55,530,339 | 3.325112098 | 3.533877891 | 3.614440218 | 3.304445597 |
| B1_off/calibration/zero_target | 56,489,568 | 3.318901416 | 3.540291645 | 3.614383489 | 3.298023420 |
| B1_off/calibration/fork | 101,006,646 | 2.637202914 | 2.903657031 | 2.970039607 | 2.607607185 |
| B1_off/calibration/agreement | 163,617,494 | 2.121445208 | 2.421145840 | 2.472916656 | 2.068051484 |
| B1_off/calibration/argmax_agreement | 128,734,602 | 1.899889444 | 2.222728900 | 2.277145847 | 1.838739338 |
| B1_off/final_test/total | 214,097,022 | 2.430144516 | 2.710095493 | 2.766154701 | 2.383584831 |
| B1_off/final_test/window_reset_1_16 | 27,031,327 | 2.462716799 | 2.700019530 | 2.749915856 | 2.450158555 |
| B1_off/final_test/window_interior_17_128 | 187,065,695 | 2.425437763 | 2.711551487 | 2.768501243 | 2.373964809 |
| B1_off/final_test/wikipedia_support | 213,177,555 | 2.427593360 | 2.704728982 | 2.762328036 | 2.380953745 |
| B1_off/final_test/empty_support | 919,467 | 3.021627652 | 3.954315969 | 3.653363427 | 2.993599749 |
| B1_off/final_test/supported_target | 158,891,199 | 2.120773022 | 2.420942589 | 2.469978841 | 2.065926654 |
| B1_off/final_test/nonempty_zero_target | 54,286,356 | 3.325628427 | 3.535345798 | 3.618007391 | 3.303009228 |
| B1_off/final_test/zero_target | 55,205,823 | 3.320565217 | 3.542323852 | 3.618596255 | 3.297855934 |
| B1_off/final_test/fork | 98,425,906 | 2.646952054 | 2.913792305 | 2.977901934 | 2.615956990 |
| B1_off/final_test/agreement | 158,891,199 | 2.120773022 | 2.420942589 | 2.469978841 | 2.065926654 |
| B1_off/final_test/argmax_agreement | 124,737,570 | 1.898024904 | 2.221277703 | 2.271787774 | 1.835325318 |

Manual notes: CPU harvest only; no GPU acquired, frozen arm changed, or checkpoint committed. Supervisor transport unavailable. Fresh process/broker receipt shows evaluator launcher absent and no GPU2 lease. DiLoCo gap IDs inapplicable. Historical PB verdicts.json is scaffolding; PT1 above is authoritative.
