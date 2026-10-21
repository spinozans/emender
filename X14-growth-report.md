# X14 — no-freebie composition: completed growth report

**Uniform 4/4 noise-level verdict:** E97 seeds0/1 and stock GDN2 seeds0/1 all completed **500,000 updates** with P1 masked from training loss. Neither architecture acquired any position: the maximum over every position, observed validation and arm is **4/512** (diagnostic P1); the composition-only maximum is **3/512**, against the registered materiality bar of **50/512**. Fixed-final composition perfect-prefix length is **0 in all four arms**, as it is throughout each trajectory. Noise-level does not mean literally all-zero.

**Universal-ignition finding, scoped to these regimes:** the visible-position learning signal is load-bearing for **both** architectures; neither learns composition cold here. Scaffold-dependence belongs to this **task family under the tested training regimes**, not uniquely to E97. The architecture separation is in the observed **ignition threshold and GDN2's deeper extension once ignited**, not raw composition learnability or a demonstrated runtime-capacity difference. **Mechanism of the threshold difference: unknown, unclaimed.**

## 1. Authority, design and successful terminal receipts

Evidence root **C/** is `experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m/`. The [prelaunch registration](experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m/REGISTRATION.md), [machine registration](experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m/registration.json) and [launch receipt](experiments/expressivity_tasks/results_invertible_stage14_nofreebie_8m/LAUNCH_REPORT.md) govern this experiment. Registration/source commit **7587a57579eb52eeead2b86e0ed66268b00f5178** preceded launch; registration SHA256 **efce847a2d0b95cc65bb7596bbbd9afd45fa5a80a5bf46b0badcbead0d630335**.

Scratch deep-first T16, p61/vocab3721, dim256/four layers (nominal8M; E97 9,900,800 / GDN2 9,887,056 parameters), batch32, fp32 parameters/bf16 autocast. E97 uses legacy recurrent precision and sequential Triton evaluation; GDN2 uses stock bf16 chunk recurrence, no fp32 adapter. The actual X10 AdamW update is reused: betas(.9,.999), eps1e-8, all-parameter decay.01, clip1; 5K linear warmup to1e-3, cosine to1e-5 at500K. Inputs and targets are unchanged; **P1 loss coefficient is zero**, and the objective is the uniform mean CE on **P2..P16**. P1 remains diagnostic-only **NON-COMPOSITION**, never contributing to the primary aggregate or best-checkpoint selection. No curriculum, carry supervision, resumed weights, tuning, score-selected restart or metric-based early stop.

Two matched seed pairs,0 and1, each have a frozen512-sequence validation pool matching X10; fresh procedural training draws are separate. These are validation observations, not independent test-set evidence. Each arm has **500,000 consecutive finite training rows and1,001 validations**, baseline plus every500 updates. Read-only native `status(C)` returns **0**, `DONE e97_seed0; DONE gdn2_seed0; DONE e97_seed1; DONE gdn2_seed1`: atomic DONE markers match successful terminal receipts and SHA256-bound complete `result.json` files, all `stop_reason=safety_ceiling`. No model was trained or re-evaluated for this report.

## 2. Full final, selected-best and maximum-anywhere columns

All counts are **joint full-state correct/512**, not coordinate accuracy. Fixed-final500K is primary. Best maximizes the **P2..P16 aggregate count**, then minimizes its CE on ties; it is not minimum-CE selection. Independent per-position maxima need not coexist. P1 is diagnostic only; P2 is the first composition position.

| Arm | Selected-best step | Best composition correct/7680 | Best CE (nats) | Final composition correct/7680 | Final CE (nats) |
|---|---:|---:|---:|---:|---:|
| e97_seed0 | 82,000 | 7 | 8.222045157 | 2 | 8.223234276 |
| gdn2_seed0 | 482,000 | 9 | 8.222922657 | 1 | 8.223849648 |
| e97_seed1 | 361,500 | 8 | 8.227215892 | 0 | 8.224306048 |
| gdn2_seed1 | 200,000 | 8 | 8.221585810 | 2 | 8.223179206 |

### e97_seed0

| Position | Role | Final /512 | Best /512 | Max anywhere /512 | Final CE | Best CE |
|---|---|---:|---:|---:|---:|---:|
| P1 | diagnostic non-composition | 0 | 1 | 4 | 8.222674757 | 8.225721262 |
| P2 | composition | 0 | 2 | 2 | 8.226299418 | 8.222048096 |
| P3 | composition | 0 | 0 | 2 | 8.226488519 | 8.221853673 |
| P4 | composition | 2 | 0 | 2 | 8.220706649 | 8.221586267 |
| P5 | composition | 0 | 0 | 3 | 8.223965608 | 8.221275231 |
| P6 | composition | 0 | 0 | 2 | 8.223392494 | 8.222324366 |
| P7 | composition | 0 | 2 | 3 | 8.223838376 | 8.222625442 |
| P8 | composition | 0 | 0 | 2 | 8.220799958 | 8.221654739 |
| P9 | composition | 0 | 0 | 3 | 8.224410750 | 8.221115001 |
| P10 | composition | 0 | 0 | 2 | 8.219515637 | 8.222671514 |
| P11 | composition | 0 | 0 | 2 | 8.223889096 | 8.222467361 |
| P12 | composition | 0 | 0 | 2 | 8.224687368 | 8.222514344 |
| P13 | composition | 0 | 0 | 2 | 8.226640003 | 8.221836371 |
| P14 | composition | 0 | 1 | 2 | 8.221035851 | 8.220933929 |
| P15 | composition | 0 | 0 | 3 | 8.220888324 | 8.223767700 |
| P16 | composition | 0 | 2 | 2 | 8.221956091 | 8.222003328 |

### gdn2_seed0

| Position | Role | Final /512 | Best /512 | Max anywhere /512 | Final CE | Best CE |
|---|---|---:|---:|---:|---:|---:|
| P1 | diagnostic non-composition | 0 | 0 | 2 | 8.223423760 | 8.219771946 |
| P2 | composition | 0 | 1 | 2 | 8.225758465 | 8.218835711 |
| P3 | composition | 0 | 0 | 2 | 8.224098884 | 8.220701599 |
| P4 | composition | 0 | 0 | 3 | 8.220943958 | 8.222675598 |
| P5 | composition | 0 | 1 | 3 | 8.225060727 | 8.224117070 |
| P6 | composition | 0 | 0 | 2 | 8.229971612 | 8.227243713 |
| P7 | composition | 1 | 1 | 2 | 8.225942351 | 8.221980236 |
| P8 | composition | 0 | 1 | 2 | 8.220788701 | 8.226635433 |
| P9 | composition | 0 | 1 | 3 | 8.224035148 | 8.216822913 |
| P10 | composition | 0 | 1 | 2 | 8.224959442 | 8.226633628 |
| P11 | composition | 0 | 0 | 2 | 8.221466128 | 8.223152244 |
| P12 | composition | 0 | 1 | 2 | 8.221671985 | 8.225923456 |
| P13 | composition | 0 | 0 | 2 | 8.222698400 | 8.222084142 |
| P14 | composition | 0 | 0 | 2 | 8.225994758 | 8.223547071 |
| P15 | composition | 0 | 1 | 2 | 8.223329816 | 8.219734967 |
| P16 | composition | 0 | 1 | 2 | 8.221024351 | 8.223752076 |

### e97_seed1

| Position | Role | Final /512 | Best /512 | Max anywhere /512 | Final CE | Best CE |
|---|---|---:|---:|---:|---:|---:|
| P1 | diagnostic non-composition | 1 | 0 | 3 | 8.228575397 | 8.226640387 |
| P2 | composition | 0 | 2 | 3 | 8.222673690 | 8.228927134 |
| P3 | composition | 0 | 0 | 3 | 8.225957606 | 8.218704387 |
| P4 | composition | 0 | 0 | 2 | 8.222883357 | 8.224259764 |
| P5 | composition | 0 | 0 | 3 | 8.223129004 | 8.228151902 |
| P6 | composition | 0 | 0 | 2 | 8.223585552 | 8.224034288 |
| P7 | composition | 0 | 2 | 3 | 8.224277487 | 8.219264358 |
| P8 | composition | 0 | 1 | 2 | 8.225911422 | 8.229045205 |
| P9 | composition | 0 | 0 | 2 | 8.223031482 | 8.237525395 |
| P10 | composition | 0 | 1 | 3 | 8.223597413 | 8.220472127 |
| P11 | composition | 0 | 1 | 2 | 8.226510126 | 8.227079777 |
| P12 | composition | 0 | 1 | 2 | 8.226881927 | 8.229683843 |
| P13 | composition | 0 | 0 | 2 | 8.224624339 | 8.222566654 |
| P14 | composition | 0 | 0 | 3 | 8.225183764 | 8.224116865 |
| P15 | composition | 0 | 0 | 2 | 8.222538045 | 8.232344545 |
| P16 | composition | 0 | 0 | 2 | 8.223805506 | 8.242062139 |

### gdn2_seed1

| Position | Role | Final /512 | Best /512 | Max anywhere /512 | Final CE | Best CE |
|---|---|---:|---:|---:|---:|---:|
| P1 | diagnostic non-composition | 0 | 0 | 3 | 8.223672390 | 8.221444869 |
| P2 | composition | 0 | 2 | 2 | 8.223072076 | 8.221925626 |
| P3 | composition | 0 | 0 | 2 | 8.222329680 | 8.222243296 |
| P4 | composition | 0 | 0 | 2 | 8.225291705 | 8.223135322 |
| P5 | composition | 1 | 0 | 3 | 8.224226527 | 8.221223978 |
| P6 | composition | 0 | 0 | 2 | 8.220953012 | 8.221281486 |
| P7 | composition | 0 | 2 | 2 | 8.223364504 | 8.221734434 |
| P8 | composition | 0 | 1 | 2 | 8.222941088 | 8.222604122 |
| P9 | composition | 0 | 0 | 3 | 8.223505795 | 8.222038291 |
| P10 | composition | 0 | 1 | 2 | 8.225623108 | 8.222223971 |
| P11 | composition | 0 | 1 | 2 | 8.221550515 | 8.220026074 |
| P12 | composition | 0 | 1 | 3 | 8.221739270 | 8.220108708 |
| P13 | composition | 1 | 0 | 2 | 8.224021197 | 8.219367366 |
| P14 | composition | 0 | 0 | 2 | 8.225113258 | 8.222092949 |
| P15 | composition | 0 | 0 | 2 | 8.221382819 | 8.224403307 |
| P16 | composition | 0 | 0 | 2 | 8.222573537 | 8.219378214 |

**First>=50 and first512:** not reached for **every P1..P16 in every arm**, right-censored at500K (not invented zero-step crossings or proof of impossibility). No transient material solver was learned and then lost. P16 final is0/512 in all four arms. The small aggregate-best counts do not establish composition.

## 3. Loss trajectories and the exact-marginal signature check

The reference is **ln(3721)=8.221747728 nats**, joint uniform accuracy **1/3721=0.0268744961%**, about **0.1376 expected matches/512 per position**. A full composition aggregate contains7680 targets. Repeated maximization over1,001 validations can yield isolated2s/3s/4s; this is a descriptive floor reference, not a significance test or proof of the model's internal guessing strategy.

**Did all four sit exactly at ln(3721)? No.** All four descend from very high initial CE into a near-marginal basin, with small nonzero drift, including a late rise and recovery. None stays identically at the marginal; the final excess is about0.00143–0.00256 nats. Near-marginal CE alone cannot diagnose absence of learning: the all-position acquisition counts above supply that verdict. Neither exact-uniform logits nor a particular internal marginal solution is established.

Below, CE is frozen-validation **P2..P16** mean, not P1-inclusive CE or fresh-batch training loss. Parentheses give CE minus ln3721.

| Step | E97 seed0 | GDN2 seed0 | E97 seed1 | GDN2 seed1 |
|---:|---:|---:|---:|---:|
| 0 | 250.989728332 (+242.767980603) | 243.687484397 (+235.465736669) | 251.415184383 (+243.193436654) | 245.326049129 (+237.104301401) |
| 500 | 30.400700977 (+22.178953249) | 21.066287022 (+12.844539293) | 30.128133351 (+21.906385622) | 21.084372956 (+12.862625228) |
| 1,000 | 26.031651105 (+17.809903376) | 11.683091306 (+3.461343577) | 25.863725291 (+17.641977563) | 11.959359484 (+3.737611756) |
| 5,000 | 8.334388560 (+0.112640832) | 8.414710716 (+0.192962988) | 8.352244164 (+0.130496436) | 8.454898217 (+0.233150489) |
| 50,000 | 8.221724899 (-0.000022830) | 8.221835246 (+0.000087517) | 8.222189686 (+0.000441957) | 8.222205254 (+0.000457526) |
| 100,000 | 8.222067687 (+0.000319959) | 8.222092354 (+0.000344626) | 8.222098712 (+0.000350984) | 8.222018027 (+0.000270298) |
| 250,000 | 8.224375843 (+0.002628114) | 8.223939625 (+0.002191896) | 8.224852486 (+0.003104758) | 8.224511238 (+0.002763510) |
| 400,000 | 8.229155758 (+0.007408030) | 8.229829158 (+0.008081430) | 8.228402739 (+0.006655010) | 8.228712779 (+0.006965051) |
| 500,000 | 8.223234276 (+0.001486548) | 8.223849648 (+0.002101920) | 8.224306048 (+0.002558320) | 8.223179206 (+0.001431478) |

| Arm | Validation CE range,50K–500K (901 observations) | Maximum absolute deviation from ln3721 | Training loss update1 → update500K |
|---|---:|---:|---:|
| e97_seed0 | 8.221248519–8.233357131 | 0.011609403 | 251.532241821 → 8.224679947 |
| gdn2_seed0 | 8.221316778–8.232514001 | 0.010766272 | 244.326110840 → 8.223201752 |
| e97_seed1 | 8.221327609–8.233221284 | 0.011473556 | 250.068313599 → 8.222210884 |
| gdn2_seed1 | 8.221286563–8.232320367 | 0.010572639 | 244.153747559 → 8.218767166 |

Training losses are fresh-batch masked CE; final differences from validation are not a different validation outcome. Full validation and per-update trajectories remain in `C/NAME/validation.json` and `training.jsonl`.

## 4. Universal ignition and explicit supervisor corrections

Removing P1 credit kills material acquisition for **both architectures at both tested seeds**. The visible inputs remain present; what was removed was the direct P1 training signal. Thus “scaffold” here means a rewarded, directly computable visible-position foothold, not removal of the information needed to solve P2. Under this T16 AdamW+cosine/500K recipe, neither architecture bootstraps composition from the P2..P16 loss alone. This is finite **learning dependence**, not representational necessity across all optimizers, seeds, scales, budgets or tasks.

On receipt review, the supervisor explicitly approved the following corrections to the task text; these are recorded rather than silently repaired:

1. **“GDN2 has no such dependency” is RETRACTED.** X14's two GDN2 arms remain noise-level alongside both E97 arms. The scaffold-dependence is not E97-specific.
2. **“Nothing above2/512 anywhere in any run” is RETRACTED.** Raw maxima include E97 seed0 diagnostic P1=4 and composition3s across arms. Correct verdict: **no position in any arm reached anywhere near materiality (global max4/512 versus50/512)**.
3. **“GDN2 fails at~1/136 in X8” is RETRACTED.** X8 GDN2 **did ignite**: final P1=505/P2=511. Its third position was foregone, not its ignition. The corrected low-share evidence is below.

Registered prediction(a), E97 no-freebie carry acquisition, is not observed; prediction(b)'s **E97-only dependence with GDN2 acquisition** is also not observed. The joint all-noise outcome is reported as such, not forced into either branch. Prediction(c)'s GDN2 carry-only bootstrap does **not** survive in these two runs. X14 stands on stage10-pattern paths independently of X12; X13's CLEAN input-path audit supports use of X12 in the descriptive series assembly but supplies no mechanism.

## 5. Honest cross-experiment scaffold-dose assembly

“Ignition” denotes material acquisition (>=50/512); a perfect prefix is a stricter, different readout. P1 is **not composition** even when included in older reports' perfect prefixes. Nominal objective coefficients below are not measured gradient shares or fractions of realized CE.

| P1 nominal objective coefficient | Receipt / recipe | E97 outcome | GDN2 outcome |
|---|---|---|---|
| **0** | **X14**, T16 uniform P2..P16, AdamW+cosine,500K, seeds0/1 | No position>=50; composition max3 | No position>=50; composition max3 |
| **1/136 (~0.7353%)** | **X8**, T16 t-weighted objective t/136, ScheduleFree,500K, seed0 | No ignition; P1 final2/max3, P2 final1 | **Ignition**: P1 final505, P2 final511; P3 final9/max10; no strict perfect prefix |
| **1/64 (1.5625%)** | **X12**, T64 uniform objective, AdamW+cosine,1M, seed0 | No ignition; P1 max2, no position>=50; replication pending X13 | **Five-perfect** P1..P5 (four composition positions), P6=491 final/max496 |
| **1/16 (6.25%)** | **X10 E97 / X11 GDN2**, T16 uniform objective, AdamW+cosine,500K, seed0 | P1..P4 perfect, P5=7 final | P1..P4 perfect, P5=318 final |

Sources: [X8](X8-growth-report.md), [X10](X10-growth-report.md), [X11](X11-growth-report.md), [X12](X12-growth-report.md), [X13 audit/replication](X13-report.md). Raw X8/X12 final vectors were independently checked for this report. **X13 seeds1/2 replication remains pending; partial zeros are not replications.**

Descriptively, E97 ignites **at1/16** and fails **at1/64** in seed0, suggesting a threshold bracket **(1/64,1/16]** *if* a monotone coefficient-only threshold model were valid. GDN2 already ignites **at1/136**, so its descriptive ignition threshold is **<=1/136**, with zero insufficient in X14. This is the corrected series separation, not the erroneous claim of GDN2 failure at1/136.

**Caveat: this is NOT a measured dose-response curve.** Natural uniform loss share changes with horizon (T16 versusT64); X8 instead explicitly reweights every position and uses ScheduleFree rather than AdamW+cosine. Horizon, frozen pool, schedule denominator, budget and backend history also differ across comparisons. The X8 GDN2 comparison to X7 additionally changed fp32-adapter versus stock-bf16 backend. These receipts cannot isolate P1 coefficient as the sole causal variable, prove monotonicity or estimate a clean architecture-specific numerical threshold. GDN2's smaller successful coefficient coexists with weaker X8 downstream learning; ignition and extension are separate readouts.

**REGISTERED FOLLOW-UP — explicit P1-weight sweep at fixed T.** Record the clean next hypothesis test here: vary only the normalized P1 objective coefficient at a fixed horizon, with uniform relative weighting on the remaining positions, matched architecture/seed pairs and a fixed optimizer, schedule, budget and evaluation protocol. Score P1 ignition and P2+ composition separately, retain full-position final/best/maxima and censored first50/first512, and test the architecture threshold separation. This is a **registered follow-up commitment, not existing data, an executable launch registration or authorization to run**. No sweep points or outcomes are claimed; exact grid and operational protocol require prelaunch registration before any confirmatory execution.

## 6. Corrected series synthesis and standing limits

**Composition learning is scaffold-initiated for both architectures in this task family under the tested regimes.** X10/X11 show both can learn composition with adequate visible-position reward; X14 kills the reading that only GDN2 learns it cold. The observed separation lives in the ignition threshold and **GDN2's deeper learned extension once ignited** (X12: P1..P5 perfect, P6 near-perfect), not a blanket difference in raw composition learnability.

Nor is a runtime-capacity explanation established. The **X13/X15 zero-shot probes are symmetric in the scoped noncollapse sense**, not literally identical scores: the unchanged E97 T16 champion retains P1..P4=512 each atT64 (P5=8), while the unchanged GDN2 T16 champion gives P1..P4=512,512,512,511 (P5=295 versus its T16 best333). Neither loses its trained-prefix competence simply because the input is longer. GDN2's strict perfect prefix does fall4→3 by one example; do not call that exact parity or hide the mixed counts. These probes neither demonstrate a64-step solver nor establish unlimited carry capacity, and do not identify why scratch learning differs. X15's corrected aggregate-best checkpoint was used; final P5=318 was not substituted for best333. Source: `results_invertible_stage15_probes_8m/gdn2_amendment/gdn2_zeroshot/result.json` and its growth report.

**Mechanism unknown and unclaimed:** no causal attribution to linear state, tanh, overwrite, gradient dominance, layer count, optimizer or a discovered circuit. The threshold account is a descriptive learning interpretation, not a mechanism measurement. Four layers do not impose a four-step temporal nonlinear-composition budget; X12 already refutes that learned-depth reading.

Standing [interpretation constraints](experiments/expressivity_tasks/INVERTIBLE_QUADRATIC.md#interpretation-constraints) remain binding. Every finite-state task admits a sufficiently large one-hot linear-state permutation representation. No impossibility, capacity bound, universal expressivity separation, production capability or architecture ranking follows. Invertibility removes task-state merging, not all shortcuts or optimization confounds. Task performance does not prove the intended internal recurrence. The suffix-only uniformity theorem concerns observers missing at least two iid drivers, **not prefix-retaining recurrent models**; chance-sized suffix counts or aggregate prefix credit establish no deep tracking.

**Limits:** X14 has only **two matched seed pairs**, one task family and one finite500K recipe; the dose assembly's X8/X10/X11/X12 arms are **single seed0**, not cross-seed threshold estimates. No score-selected seeds/restarts. Fixed-final is primary; all transient crossings/best checkpoints are descriptive. Absent crossings are right-censored at the observed ceiling; infrastructure/time/memory failures would be censored, not architecture failures. This report launches nothing, changes no training code and authorizes no checkpoint deletion or scale-up.

## 7. Reproduction, validation and local evidence identities

Read-only audit on **lambda01**: `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x14_growth.py` **passed**. It checked native hash-bound completion, all four result/final/best identities, complete500-step grids, every positional denominator and finite CE, P2..P16 aggregate count/CE reconciliation, all-position maxima and absence of materiality, zero composition prefixes, and **2,000,000 finite consecutive training loss/LR/gradient rows**. It also checked raw X8/X12 final vectors. This is a temporary local receipt audit, not a committed dependency; no checkpoint re-evaluation or new training was performed. Training LR values were checked for finiteness here, not re-derived against the schedule.

CPU regression: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_invertible_stage14_nofreebie_8m.py -k 'not gpu'` — **16 passed,2 GPU cases deselected (11.09s)**. No tests added/changed for this documentation-only task. An initial ad-hoc cross-probe extractor assumed X15's final had a training `step` field and raised `KeyError`; the corrected extractor checked `training_updates=0`, `weights_unchanged` and literal T16/T64 positional counts successfully. This was a reporting-script assumption, not a failed model probe.

The embedded reconstruction command below also passed. A separate read-only report check matched **all64 printed positional rows** (final/best/max counts and final/best CE) against raw receipts and resolved all local Markdown links; the registration SHA256 matches the launch receipt. Whitespace/diff checks are part of the report-only commit validation.

Raw streams/configs/checkpoints remain **local/untracked**. This commit contains only `X14-growth-report.md`; it does not publish those raw artifacts. Under `C/NAME/`, SHA256 identities are:

| Arm | result.json | validation.json |
|---|---|---|
| e97_seed0 | `74d94e50bbc7e7d52c84e23bc52478834d24370172add559694552a97cdfbdae` | `0dd2f249af30941ab28d5c9f15ccfa0a804373db5929fb3cfba3b6e35110767a` |
| gdn2_seed0 | `bf2f6309f1dd6d6696e42602cb6a9dfd3f0e76e7be8f3a302d302adf8a2dc983` | `00d59182e8778edf0539281aeda58ce5762d40542b717fda2fb3945383f46c15` |
| e97_seed1 | `2ddd3e6a96f4911aae819e13704d0f8f6ecafb53aa8f65738695c74437e72d68` | `3824f930852421d504bda4ba679c3fde7f028ba0cc74cebc0c312450341f194b` |
| gdn2_seed1 | `7ccbcd9bab725b1a690c84af475910975d0b231addab05bc0bea4759715ca2bb` | `a8cc851a5405100bb056fcb754a2d50357d4e9f3083cf17c384392be78cfcb4d` |

| Arm | training.jsonl | config.json |
|---|---|---|
| e97_seed0 | `188eeefa70d9fe1b84139ab7228b4d213c9281d9b05869fa717d5243e5d3833b` | `0913cd91c110e6e2d4ca2aa561f8e90e38e9222abd37f16492ff97e044efd00f` |
| gdn2_seed0 | `49add475c84a3a4da3efaea1d8cacde448cf50545b29ed93b2c10ef02a81cb1c` | `ff3961bd6dca852ea1a4cdafc72da7987f89cb1c2c3fa38b01e0032fa6cb7975` |
| e97_seed1 | `0775b9ac46ac63e13114f2399692ab7b7bb756af2e77acc44535470fda03028f` | `b0cdda88e087211f5a80ebdb291a95a4ebb666d87de63ca0e5585eabba4606a8` |
| gdn2_seed1 | `bf328c7b626bd39c6ab2cc45bd112e0f0284e2299e723ca94cf0f22e70d78d30` | `6d94f34f59dccd9f35ff307259fe2d13b2cfbba8ea1869902ceff01832cee0c7` |

Minimal read-only reconstruction (lambda01; on Frontier first source the canonical environment and use `$EMENDER_PYTHON`):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json, math
from scripts.watch_invertible_stage14_nofreebie import DEFAULT_ROOT as c, NAMES, status
assert status(c)[0] == 0
for name in NAMES:
    rows = json.loads((c / name / 'validation.json').read_text())
    assert [r['step'] for r in rows] == list(range(0, 500001, 500))
    best = max(rows, key=lambda r: (r['aggregate']['correct'], -r['aggregate']['loss']))
    for label, row in [('final', rows[-1]), ('best', best)]:
        print(name, label, row['step'], [p['correct'] for p in row['positions']], row['aggregate'])
    print('maxima', [max(r['positions'][i]['correct'] for r in rows) for i in range(16)])
    print('first50/512', [[next((r['step'] for r in rows if r['positions'][i]['correct'] >= t), None)
                             for t in (50, 512)] for i in range(16)])
    for step in (0, 5000, 50000, 250000, 400000, 500000):
        ce = rows[step // 500]['aggregate']['loss']
        print('CE', step, ce, 'minus marginal', ce - math.log(3721))
PY
```

Standalone local expressivity, **not resilient DiLoCo or Frontier acceptance**: no distributed behavior changed or tested; resilient requirement IDs and Slurm Partition/QOS evidence are **not applicable**. Independent acceptance review remains reviewer-owned. Unrelated existing worktree changes are left untouched; no files are left staged after the report commit.
