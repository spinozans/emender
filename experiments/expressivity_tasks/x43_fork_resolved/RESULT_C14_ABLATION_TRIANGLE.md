# C14-B1 measured ablation triangle

| Arm | Final-test on NLL | Training steps | Training byte targets |
|---|---:|---:|---:|
| c14_b_v2 | 1.584700014 | 35,419 | 71,677,042 |
| c14_b_v2_append_only | 2.246020115 | 37,254 | 75,394,452 |
| c14_b_v2_unbudgeted | 1.716496444 | 7,142 | 14,455,380 |

Unbudgeted final-test NLL is **1.716496444**, +0.131796431 versus budgeted and -0.529523671 versus append-only. All arms use the same 214,097,022 final-test targets; calibration remains separate. Full evaluation covers 3,431,242 windows / 434,204,084 targets, 26,807 shards, terminal exit 0. The last progress print (~1.7158) is not the terminal result.

Registered conditional-ablation verdict (PREREGISTRATION_C14_B.md, conditional ablations / PB3): the observed ordering is budgeted full operations < unbudgeted full operations < append-only. Budgeted beats append-only by 0.661320101 despite slightly less exposure, supporting lifecycle value in this fixed-time recipe; it does not isolate individual operations. Unbudgeted retains lifecycle operations with a verified nonbinding cap 257, and beats append-only by 0.529523671 despite 5.2157x fewer training targets. Its 4.9585x exposure deficit versus budgeted cuts **in its favor** as evidence of data efficiency: it achieved 1.716496444 with far less data. It is not evidence that removing pressure inherently hurts quality.

**Budget pressure itself adds value is causally unresolved at these exposures.** The 0.131796431 budgeted advantage cannot cleanly separate eviction pressure from lifecycle value, different realized exposure, allocated capacity / compute cost, or training trajectory. The measured advantage is for the fixed 2700-second recipe, not an equal-exposure architecture result. One seed supplies no uncertainty estimate.

Additional resolving comparison, specified but NOT RUN: continue an independently retained unbudgeted arm to exactly 71,677,042 real byte targets (35,419 steps under the identical stream/batch/window protocol), retaining its nonbinding cap, source/hash gates, seed, optimizer, all lifecycle operations and frozen document-disjoint evaluation. Compare to frozen budgeted B1-v2 at that exposure and report the additional active compute. A complementary budgeted truncation at 14,455,380 targets would test the low-exposure point if its exact checkpoint were retained; do not substitute an interpolated score. Multiple seeds and capacity/implementation controls would further test generality and isolate pressure from implementation effects. No new training/evaluation launched.

Training occupancy evidence remains RESULT_C14_OCCUPANCY_SCALING.md: peaks 81/110/81/96, zero capacity merges, roughly 4.96x step cost, early sampled means nearly identical to budgeted. These do not establish occupancy scaling or a stationary 110-slot mean. Independent document-bounded windows reset state.

All 66 strata/mode/split rows follow, with identical denominators across arms. Strata overlap; do not sum them. Full sums and counts are in receipts/c14_b_v2_unbudgeted/triangle_tables.json.

| Mode / split / stratum | Targets | Budgeted NLL | Append-only NLL | Unbudgeted NLL |
|---|---:|---:|---:|---:|
| B1_on/all/total | 434,204,084 | 1.581844925 | 2.243243320 | 1.713985184 |
| B1_on/all/window_reset_1_16 | 54,815,336 | 1.770575453 | 1.859236677 | 1.887262972 |
| B1_on/all/window_interior_17_128 | 379,388,748 | 1.554576517 | 2.298725864 | 1.688949438 |
| B1_on/all/wikipedia_support | 432,325,388 | 1.575678004 | 2.235782863 | 1.705765587 |
| B1_on/all/empty_support | 1,878,696 | 3.000976320 | 3.960043003 | 3.605478187 |
| B1_on/all/supported_target | 322,508,693 | 0.855293015 | 1.745273253 | 0.921938262 |
| B1_on/all/nonempty_zero_target | 109,816,695 | 3.691298232 | 3.676307125 | 4.007702701 |
| B1_on/all/zero_target | 111,695,391 | 3.679687144 | 3.681079510 | 4.000937359 |
| B1_on/all/fork | 199,432,552 | 2.079896312 | 2.542908792 | 2.247827672 |
| B1_on/all/agreement | 322,508,693 | 0.855293015 | 1.745273253 | 0.921938262 |
| B1_on/all/argmax_agreement | 253,472,172 | 0.365983238 | 1.401428603 | 0.386095425 |
| B1_on/calibration/total | 220,107,062 | 1.579067795 | 2.240542345 | 1.711542493 |
| B1_on/calibration/window_reset_1_16 | 27,784,009 | 1.768538894 | 1.857033869 | 1.885555556 |
| B1_on/calibration/window_interior_17_128 | 192,323,053 | 1.551695794 | 2.295946015 | 1.686403642 |
| B1_on/calibration/wikipedia_support | 219,147,833 | 1.572979014 | 2.233146791 | 1.703330545 |
| B1_on/calibration/empty_support | 959,229 | 2.970125902 | 3.930148898 | 3.587664431 |
| B1_on/calibration/supported_target | 163,617,494 | 0.853755982 | 1.743719434 | 0.921559724 |
| B1_on/calibration/nonempty_zero_target | 55,530,339 | 3.692135355 | 3.675221144 | 4.006780966 |
| B1_on/calibration/zero_target | 56,489,568 | 3.679875173 | 3.679549981 | 3.999664099 |
| B1_on/calibration/fork | 101,006,646 | 2.074203028 | 2.536420499 | 2.241834938 |
| B1_on/calibration/agreement | 163,617,494 | 0.853755982 | 1.743719434 | 0.921559724 |
| B1_on/calibration/argmax_agreement | 128,734,602 | 0.366057223 | 1.401021885 | 0.388164731 |
| B1_on/final_test/total | 214,097,022 | 1.584700014 | 2.246020115 | 1.716496444 |
| B1_on/final_test/window_reset_1_16 | 27,031,327 | 1.772668719 | 1.861500822 | 1.889017930 |
| B1_on/final_test/window_interior_17_128 | 187,065,695 | 1.557538200 | 2.301583840 | 1.691566782 |
| B1_on/final_test/wikipedia_support | 213,177,555 | 1.578452583 | 2.238492762 | 1.708268825 |
| B1_on/final_test/empty_support | 919,467 | 3.033160852 | 3.991229868 | 3.624062292 |
| B1_on/final_test/supported_target | 158,891,199 | 0.856875769 | 1.746873291 | 0.922328061 |
| B1_on/final_test/nonempty_zero_target | 54,286,356 | 3.690441925 | 3.677417991 | 4.008645559 |
| B1_on/final_test/zero_target | 55,205,823 | 3.679494742 | 3.682644607 | 4.002240227 |
| B1_on/final_test/fork | 98,425,906 | 2.085738874 | 2.549567209 | 2.253977537 |
| B1_on/final_test/agreement | 158,891,199 | 0.856875769 | 1.746873291 | 0.922328061 |
| B1_on/final_test/argmax_agreement | 124,737,570 | 0.365906883 | 1.401848354 | 0.383959811 |
| B1_off/all/total | 434,204,084 | 2.429446425 | 2.709220798 | 2.766010077 |
| B1_off/all/window_reset_1_16 | 54,815,336 | 2.461731925 | 2.699154373 | 2.750134675 |
| B1_off/all/window_interior_17_128 | 379,388,748 | 2.424781710 | 2.710675228 | 2.768303807 |
| B1_off/all/wikipedia_support | 432,325,388 | 2.427011221 | 2.703904966 | 2.762247804 |
| B1_off/all/empty_support | 1,878,696 | 2.989835561 | 3.932499671 | 3.631784165 |
| B1_off/all/supported_target | 322,508,693 | 2.121114041 | 2.421045704 | 2.471469275 |
| B1_off/all/nonempty_zero_target | 109,816,695 | 3.325367338 | 3.534603530 | 3.616203601 |
| B1_off/all/zero_target | 111,695,391 | 3.319723755 | 3.541296070 | 3.616465663 |
| B1_off/all/fork | 199,432,552 | 2.642014405 | 2.908659091 | 2.973919900 |
| B1_off/all/agreement | 322,508,693 | 2.121114041 | 2.421045704 | 2.471469275 |
| B1_off/all/argmax_agreement | 253,472,172 | 1.898971875 | 2.222014744 | 2.274509056 |
| B1_off/calibration/total | 220,107,062 | 2.428767396 | 2.708369987 | 2.765869402 |
| B1_off/calibration/window_reset_1_16 | 27,784,009 | 2.460773731 | 2.698312654 | 2.750347567 |
| B1_off/calibration/window_interior_17_128 | 192,323,053 | 2.424143591 | 2.709822922 | 2.768111769 |
| B1_off/calibration/wikipedia_support | 219,147,833 | 2.426444941 | 2.703103399 | 2.762169758 |
| B1_off/calibration/empty_support | 959,229 | 2.959361317 | 3.911587703 | 3.611099408 |
| B1_off/calibration/supported_target | 163,617,494 | 2.121445208 | 2.421145840 | 2.472916656 |
| B1_off/calibration/nonempty_zero_target | 55,530,339 | 3.325112098 | 3.533877891 | 3.614440218 |
| B1_off/calibration/zero_target | 56,489,568 | 3.318901416 | 3.540291645 | 3.614383489 |
| B1_off/calibration/fork | 101,006,646 | 2.637202914 | 2.903657031 | 2.970039607 |
| B1_off/calibration/agreement | 163,617,494 | 2.121445208 | 2.421145840 | 2.472916656 |
| B1_off/calibration/argmax_agreement | 128,734,602 | 1.899889444 | 2.222728900 | 2.277145847 |
| B1_off/final_test/total | 214,097,022 | 2.430144516 | 2.710095493 | 2.766154701 |
| B1_off/final_test/window_reset_1_16 | 27,031,327 | 2.462716799 | 2.700019530 | 2.749915856 |
| B1_off/final_test/window_interior_17_128 | 187,065,695 | 2.425437763 | 2.711551487 | 2.768501243 |
| B1_off/final_test/wikipedia_support | 213,177,555 | 2.427593360 | 2.704728982 | 2.762328036 |
| B1_off/final_test/empty_support | 919,467 | 3.021627652 | 3.954315969 | 3.653363427 |
| B1_off/final_test/supported_target | 158,891,199 | 2.120773022 | 2.420942589 | 2.469978841 |
| B1_off/final_test/nonempty_zero_target | 54,286,356 | 3.325628427 | 3.535345798 | 3.618007391 |
| B1_off/final_test/zero_target | 55,205,823 | 3.320565217 | 3.542323852 | 3.618596255 |
| B1_off/final_test/fork | 98,425,906 | 2.646952054 | 2.913792305 | 2.977901934 |
| B1_off/final_test/agreement | 158,891,199 | 2.120773022 | 2.420942589 | 2.469978841 |
| B1_off/final_test/argmax_agreement | 124,737,570 | 1.898024904 | 2.221277703 | 2.271787774 |

Manual notes: contact_supervisor is absent from the available tool catalog, so no supervisor transport was available. Terminal and process exit, source/checkpoint hashes, all 66 sum/count rows and card-2 lease absence were checked. No card acquired; no frozen arm or training code changed. No WorksGood service, union service, card-1 search, sxgc or /mnt/nvme2n1 touched. No DiLoCo work; gap IDs inapplicable. Historical paused receipts remain historical; HARVEST.json supersedes their run status.
