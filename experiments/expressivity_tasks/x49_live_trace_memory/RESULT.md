# X49 final: use the bound earlier value; the registered learned key policy lost

**Yes, this model left a useful computation for its future self.** With identical literal addresses, follower bytes, counts, capacity and training exposure, B's earlier neural value beat A's current computation on unseen documents. The advantage survived matching training time, grew with accessible history, and shrank when only the bound value was shuffled. **Use B as the local baseline.** Fixed private keys (F) gave a further small full-dose benefit at long horizons; learning the keys with the registered estimator (C) made predictions much worse. This is about one model, dose, data regime and optimizer, not whether all learned keys fail.

[Preregistered plan](PLAN.md) · [operational freeze](PANEL.md) · [calibration and policy diagnosis](STAGE_CHECK.md) · [aggregate receipt, including every registered contrast by seed](receipts/panel_result.json) · [read-only verifier](panel_report.py).

## Final held-out prediction

Three matched seeds. Per model: 1,024 optimizer steps on 4,194,304 shifted TRAIN bytes; all 12 trajectories start at the same X48 direct/H origin with fresh optimizer state. One frozen final: **128 full-document-verified windows, 524,288 distinct external targets per model/cell**. Every position counts, including misses. One causal host prefill per document/model; only eligibility in the memory changes across horizons. Lower is better (nats/byte).

| Arm | H=128 | H=1,024 | H=4,096 |
|---|---:|---:|---:|
| A — observed byte + current computation | 1.357368 | 1.216815 | 1.187433 |
| **B — same byte + bound earlier value** | **1.356753** | 1.215766 | 1.186309 |
| F — B + fixed private keys | 1.357624 | **1.215470** | **1.185901** |
| C — B + learned private keys | 1.605163 | 1.417570 | 1.375434 |

Positive differences below favor the candidate. Percentile 95% intervals resample the same 128 documents jointly across arms and seeds (10,000 draws); they do not treat repeated bytes/seeds as independent new documents.

| Candidate : control | Full-dose H4096 gain [95% CI] | Each seed's gain: 4901 / 4902 / 4903 | Common-wall H4096 gain [95% CI] |
|---|---:|---|---:|
| **B : A** | **+.001124 [.000964,.001289]** | +.001053 / +.001253 / +.001065 | **+.003125 [.002547,.003992]** |
| C : A | −.188001 [−.194514,−.181214] | −.192787 / −.185342 / −.185874 | −.187476 [−.193898,−.180699] |
| C : B | −.189124 [−.195633,−.182351] | −.193840 / −.186595 / −.186938 | −.191622 [−.198308,−.184625] |
| C : F | −.189532 [−.196076,−.182669] | −.193962 / −.186936 / −.187699 | −.188024 [−.194592,−.181079] |

**Binding and history.** B's H4096 advantage over A exceeded its H128 advantage by **.000509 [.000354,.000699]**. At H4096 a sham changed 1,315,314 of 1,538,700 literal read payloads across seeds, preserving addresses, counts, follower and age bucket; loss worsened by **.000974 [.000851,.001096]**. At a single seed H4096 had 512,900 literal hits and 11,388 misses; 299,163 reads were >128 positions old and 68,935 >1,024. B−A *improvements* by literal source age: 16–128 +.000886, 129–1,024 +.001362, >1,024 +.001256, miss −.000043. These are descriptive strata, not new gates or primary denominators.

**The private table.** At full dose/H4096 the secondary F:B gain was .000408 [.000054,.000660], positive in all three seeds. F was worse than B at H128; there is no registered F:B common-wall comparison. Correct vs shuffled *private* payload: F +.000004 [−.000011,.000019] (956,237 reassignments), C +.000048 [.000031,.000065] (885,717). Correct literal binding benefited both (F +.000856; C +.001010). F's predictor benefit is **not** evidence that its extra neural payload explains that benefit. The table also supplies a retrieved observed follower and more readout capacity. C's absolute losses are poor despite binding sensitivity. Its key-gradient variance at birth is a plausible issue, not a proven explanation of the whole loss gap.

**Two decisions, not one.** B is the adopted local reader. Separately, the frozen Lux rule returns **no nominee**: B passed the common-wall, binding and horizon checks and all three full-dose gains were positive, but the gain did not reach the experiment designer's .005 scale-pilot threshold. C failed exposure and common wall. Reporting that rule unchanged does not negate B's measured gain or constrain its local use. No Lux job or allocation request.

## Work and price

| Across three training seeds | A | B | F | C |
|---|---:|---:|---:|---:|
| Inclusive charged card-s, upper bound | 2,883 | 2,835 | 3,000 | 2,964 |
| Pre-update TRAIN NLL (not final) | 1.227420 | 1.226914 | 1.226429 | 1.352506 |
| H4096 frozen host prefill + correct head/score s per 4K final window | .1945 | .2012 | .2172 | .2222 |

Actual B:A common-wall steps (B/A): 992/1,024; 1,024/960; 1,024/992. Largest B:A slack 2.68%; no interpolation. All registered pairs met the 5% slack criterion. These are measured elapsed budgets, not a claim B trains intrinsically faster. Synthetic one-byte advances required full-prefix recomputation (~.175–.199 s across full-dose probes), **not** cached serving throughput.

Stage checks used ≤320 + ≤456 card-s; the final used ≤1,564 for **21 deduplicated weight endpoints × 128 documents**. All 15 leases renewed, witnessed their terminal progress and released with rc=0, **≤14,022 card-s total (3.895 GPU-h)**, against the 14,558 s forecast. Sum of inclusive job wall clocks 14,056.8 s; owner + waited children 21,628.07 core-s; driver and short parents 81.04 core-s. Successful common data preparation 4.45 wall s / 4.34 core-s separate; earlier development/refusals and the ≤76 s synthetic meter are not natural training. Peak Torch 6.3165 GiB allocated / 6.3809 reserved; process max RSS 2.503 GiB (not summed over processes). No native archive calls. Cost, clocks/targets/slack for every pair, per-seed endpoint hashes, age/hit/keys/collisions and all interventions are in the aggregate.

The completed training source was `b63e3e17786d91456daf7281d6f0699b8a560c79`, not the later report source. The post-release audit re-bound every charge, training journal, clock, stage map and closed score log, reproduced the summary and bootstraps, and read **no corpus text**. Original private summary SHA-256 `6e30a78d88480d670851a4da505e18bed154c261ab7103cf87bc0e602a515bb6`. Do not relaunch the panel or reuse this final to tune the next learner.
