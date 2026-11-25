# X49-R: better credit assignment recovered most of the deficit, but fixed keys still won

**Keep B as the local reader. Do not switch it to R.** Subtracting the same predictor's literal-only loss removed the useless key gradient at birth and recovered **.17460 nats/byte** against the original learned-key C on fresh text. But R remained **.01660 worse than fixed keys F** and **.01589 worse than B**. All three seeds agreed at equal dose and matched training time. No new effect-size cutoff is needed to make that distinction.

[Prospective protocol](RELATIVE.md) · [implementation and retained fixture failure](RELATIVE_IMPLEMENTATION.md) · [CPU gate](receipts/relative_cpu.json) · [aggregate with all seeds, intervals, clocks, costs, work and strata](receipts/relative_result.json) · [closed-receipt audit](relative_report.py).

## Fresh held-out predictions

R differs from C only in the detached likelihood-ratio baseline for the same hard, sampled filing and query actions. It trains from the same X48 direct/H origin, not from C or B. Three × 1,024 new R updates, identical 1,024 TRAIN documents and shifted targets per trajectory; retained full-dose and timestamped B/F/C weights as controls. **No control retraining or reuse of prior test text.** All remaining 137 eligible native final documents were reserved before R training. Freeze/hashes preceded raw source opening. Each cell scores 561,152 distinct shifted external bytes, including misses (not three times that many distinct observations). Frozen test visit 0, 19 distinct checkpoint endpoints, 2,603 endpoint-windows. One same-weight host prefill per window/endpoint, no archive.

| Nats per external byte (lower is better) | H128 | H1024 | H4096 |
|---|---:|---:|---:|
| B: bound earlier neural value | **1.348240** | 1.207499 | 1.167635 |
| F: B + fixed private keys | 1.348745 | **1.206853** | **1.166923** |
| C: learned private keys, constant score baseline | 1.595857 | 1.410372 | 1.358125 |
| R: same learned learner, literal score baseline | 1.369109 | 1.224258 | 1.183528 |

Gains = control loss minus R loss. Percentile 95% intervals jointly resample 137 documents across weights/seeds, 10,000 draws with RNG 4904. No bytewise independence assumption.

| H4096 comparison | Equal-exposure gain [CI] | Each paired seed (4901, 4902, 4903) | Common-wall gain [CI] |
|---|---|---|---|
| R:C — repair | **+.174597 [.168962,.179912]** | +.178032 / +.172787 / +.172972 | +.173068 [.167493,.178304] |
| R:F — learned vs fixed | **−.016605 [−.018019,−.015297]** | −.018623 / −.015217 / −.015975 | −.014751 [−.016188,−.013410] |
| R:B — learned vs working reader | **−.015893 [−.017260,−.014614]** | −.017861 / −.014765 / −.015053 | −.018437 [−.019818,−.017132] |

R recovered 91.3% of the original C-to-F full-dose deficit on **the same fresh panel**, not by comparing two different test populations. The ordering vs C, F and B held separately at all three horizons. All common-wall pairs were conclusive (max endpoint slack 2.08%, values in the receipt; controls measured earlier, not run simultaneously). At matched wall the R/F steps were 1,024/960; 1,024/992; 992/1,024. There is no evidence here that learning where to file beat using the fixed literal prior.

**Bindings and coverage.** At H4096 shuffling just R's *literal* neural payload increased loss by .000702 [.000606,.000797]; shuffling just its *private* neural payload increased loss by .000016 [.000004,.000027]. The latter is detectable but far smaller than R's deficit. B's literal sham again worsened loss, by .000847 [.000738,.000955]. Across the three R seed passes, 1,398,672 literal and 1,068,693 private payloads changed in the respective shams. There were 1,608,764 R private hits vs 1,585,922 for F out of the same 1,683,456 queries; R had only 20,803 private reads older than 1,024 vs F's 38,047. More hits did not mean better predictions. These are diagnostics, not conditional substitutes for all-byte loss.

**Optimization.** On the first natural update of every seed, R reproduced the fixed control's work, pre-update NLL and even its measured preclip norm (1.099 / 2.523 / 1.139), versus C's 180.239 / 186.819 / 155.530. Across all updates: median norms F .640, C 213.381, R 1.111; clips 547 / 3,072 / 1,775 respectively out of 3,072. Mean pre-update TRAIN NLL F 1.22643, C 1.35251, R 1.23211. This isolates the benefit of changing the baseline, **not** a proof that all remaining loss is due to clipping or that all private-symbol learning must fail.

**Cost.** New training ≤2,928 inclusive card-s; new final ≤1,608; **≤4,536 total = 1.26 GPU-h**. All four detached leases renewed, reached their exact registered work and released rc=0. Sum of inclusive job wall clocks 4,552.57 s, owner/waited children 8,074.39 core-s, driver/short parents 37.89 core-s. Fresh metadata preparation 4.50 wall / 4.50 CPU s; shared original TRAIN preparation and historical control training were not paid twice. Peak GPU allocations/reservations and per-process RSS, actual per-seed milestone times/targets and separate one-byte **full recompute**, not cached-serving, timings are in the aggregate. Original panel cost ≤14,022 card-s; not part of R's incremental charge. No remote work.

Source `c0f19aef96baf62697a67b034c5890385739bef2`, private summary SHA-256 `4f1d709fd4716fb36543a4e9d9bbb76567b5b70ec7a33a664697592406a112a6`. Post-release audit reproduced the decisions from bound, closed score receipts, not another evaluation. The registered 4K population beyond offset 100.6 GB now has **zero unspent eligible final documents** (128 + 137 = 265). A future natural study needs a demonstrably fresh document population; a different window in a spent document does not qualify.
