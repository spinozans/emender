# X6 — completed curriculum matrix: the SF-era instrument record

**11 full T1→T128 ladders + 1 adjudicated infrastructure-partial; not 12 successful ladders.** Arm A E97 crosses T1 in all three seeds and never crosses T2. Arm B's training-only carry supervision crosses T2 in **3/3 seeds**, including the partial run before its disk failure. GDN2 learns a substantially longer prefix under the same ScheduleFree (SF) regime, but neither arm crosses T8. These are replicated optimization outcomes, **not intrinsic composition limits**. X9/X10 subsequently overturned the architectural interpretation of E97's shallow SF failures; X6 remains load-bearing as replication breadth across curriculum designs.

Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/`. Protocol: [registration](experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/registration.json), [launch report](X6-launch-report.md), and [X6D positional diagnosis](X6D-report.md). The CPU-only `C/receipts/growth/audit.py` reconciles transitions, stage records, frozen-pool validations, DONE/terminal/result identities, and the infra adjudication. Its committed `audit.json` preserves exact stage-best/exit metrics and source/file hashes, rather than committing multi-gigabyte training streams or checkpoint payloads.

## Protocol and denominators

Scratch E97-MLP and real FLA GDN2, seeds0/1/2, each in A and B: **12 registered cells**. AdamWScheduleFree constant lr1e-3, betas(.9,.95), wd.01, clip1, batch32, no warmup, bf16 autocast/fp32 parameters; GDN2 `allow_neg_eigval=True`. Parameter counts A: **9,900,800 / 9,887,056** (E97/GDN2); B: **9,901,056 / 9,887,312**. Exact X3 frozen pools are shared across all four cells of each seed; fresh procedural training excludes those pools.

The ladder is **1→2→4→8→16→32→64→128**. Advancement uses **≥90% aggregate joint-state accuracy**, observed every500 updates, not every position individually reaching90%. A T2 threshold therefore need not mean P2 itself reaches90%; a T4 threshold need not mean four solved positions. No stagnation stop:500,000 updates per stage,4,000,000 global safety ceiling. Threshold advances carry weights/full optimizer; a non-threshold ceiling carries the stage's best-validation weights/full optimizer into the next depth without rewinding consumed compute/data RNG. Traversing T128 is not solving T128. “Best” maximizes aggregate correct count, then minimizes full-state CE; it is not the minimum-loss checkpoint. Exit is the actual end-of-stage checkpoint, before best restoration.

B inserts a true **previous** state into a fixed carry slot with training probability `p=.5*(1-stage_step/500000)`, reset per stage; otherwise the slot is neutral. **Every evaluation uses neutral slots, p=0.** No current/future targets enter inputs. B also changes input structure:3722 input ids,3721 output classes,256 extra neutral-embedding parameters; sequence length2T+1 versus A's T+2. This is a training-format intervention, not an architecture-only contrast.

All counts below are **joint-state** correct, not coordinate accuracy. Each pool has512 sequences and512T scored drivers; loss is **full-state CE in nats**. Uniform marginal **ln(3721)=8.221747728 nats**; suffix-only reference **1/3721=0.0268744961%**. The suffix theorem concerns an observer missing at least two iid drivers, not a recurrent model retaining prefix information. P1 credit diluted across T positions is not deep tracking; isolated chance-sized matches are not composition. Pools are reused for selection, not independent held-out/test generalization evidence.

## Completed matrix and exact threshold times

Numbers in threshold columns are **stage updates**, not cumulative updates. “—” means no threshold in a completed500K stage. All T8+ completed stages are non-threshold ceilings. The partial's last observed update is not a T64 exit or a T128 result.

| Cell | T1 threshold | T2 threshold | T4 threshold | Last validated global update | Validation rows |
|---|---:|---:|---:|---:|---:|
| A_e97_seed0 | 50,000 | — | — | 3,550,000 | 7,108 |
| A_e97_seed1 | 35,500 | — | — | 3,535,500 | 7,079 |
| A_e97_seed2 | 52,500 | — | — | 3,552,500 | 7,113 |
| A_gdn2_seed0 | 13,000 | 15,500 | 381,000 | 2,909,500 | 5,827 |
| A_gdn2_seed1 | 12,500 | 16,500 | 326,500 | 2,855,500 | 5,719 |
| A_gdn2_seed2 | 13,500 | 16,500 | — | 3,030,000 | 6,068 |
| B_e97_seed0 | 49,000 | 448,000 | — | 3,497,000 | 7,002 |
| B_e97_seed1 | 35,500 | 448,000 | — | 3,483,500 | 6,975 |
| B_e97_seed2 **infra-partial** | 53,500 | 379,000 | — | 2,705,000 (mid-T64) | 5,417 |
| B_gdn2_seed0 | 13,000 | 19,000 | 440,500 | 2,972,500 | 5,953 |
| B_gdn2_seed1 | 12,500 | 20,000 | — | 3,032,500 | 6,073 |
| B_gdn2_seed2 | 13,500 | 23,000 | — | 3,036,500 | 6,081 |

Total: **76,415 validation rows**, **94 completed stage exits =24 thresholds +70 non-threshold ceilings**. Eleven cells have8 exits each; B_e97_seed2 has6 (through T32), plus mid-T64 observations. No run exhausted the global4M safety cap. Validation rows include stage entry/exit observations, sometimes repeating a grid endpoint; they are not76,415 independent pools.

## Arm A: E97's early P1-dominant plateau, and the real seed2 exception

T1 is acquired in every seed, at exactly50,000/35,500/52,500 updates. T2 spends500K in every seed without crossing90%: bests **527/1024,526/1024,522/1024** (51.464844%,51.367188%,50.976562%). X6D identifies the misleading coordinate success: seed0 T2-best P2 has **511/512 X-coordinate** matches but only **15/512 joint** matches. There is no reliable full-state carry unlock here.

| E97 cell | T2 stage updates | T2 global exit | Best /1024 | Best CE | Exit /1024 | Exit CE | Best P2 /512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_seed0 | 500,000 | 550,000 | 527 | 2.176699340 | 521 | 2.175208896 | 15 |
| A_seed1 | 500,000 | 535,500 | 526 | 2.151103482 | 520 | 2.117430180 | 14 |
| A_seed2 | 500,000 | 552,500 | 522 | 2.912055865 | 519 | 2.154637054 | 10 |
| B_seed0 | **448,000** | **497,000** | **926** | **0.328656082** | 926 | 0.328656082 | **414** |
| B_seed1 | **448,000** | **483,500** | **923** | **0.371144488** | 923 | 0.371144488 | **411** |
| B_seed2 | **379,000** | **432,500** | **923** | **0.369795577** | 923 | 0.369795577 | **411** |

At T4 all A E97 bests remain near one-position credit. Seeds0/1 largely carry that solution up the ladder, with within-stage deterioration; their T128 bests still contain only538/644 joint matches among65,536 targets. But **“P1 only everywhere” is false**: X6D directly measures seed2 T8-best P2=**511/512**, then T16-best P3=**370/512**,P4=**349/512**, alongside P1=412/512 and P2=474/512. These are real deeper approximate joint-state gains, with retention damage, not an aggregate-denominator illusion. They do not constitute a contiguous ≥90% three-position prefix or a crossed T8/T16 threshold.

| Depth | A E97 seed0 best→exit correct | Seed1 best→exit | Seed2 best→exit | Denominator |
|---:|---:|---:|---:|---:|
| T4 | 528→520 | 529→519 | 532→522 | 2,048 |
| T8 | 532→515 | 529→519 | 1,061→1,055 | 4,096 |
| T16 | 526→276 | 591→580 | 1,606→1,602 | 8,192 |
| T32 | 528→192 | 603→10 | 1,899→1,625 | 16,384 |
| T64 | 535→401 | 599→8 | 1,908→1,776 | 32,768 |
| T128 | 538→363 | 644→18 | 1,911→1,817 | 65,536 |

Thus the early ceilings are mostly the diluted-P1 solution, not a universal one-position invariant. **A E97 T128 exits are exactly0.553894%,0.027466%,2.772522%; bests0.820923%,0.982666%,2.915955%.** The task shorthand “1.79–3.6% finals” does not describe these receipts. Low aggregate accuracy alone neither diagnoses a particular position nor proves a capacity limit.

## Arm A: GDN2's longer prefix, then cliff

T1 thresholds take13,000/12,500/13,500 stage updates; T2 another15,500/16,500/16,500. T4 crosses in **2/3 seeds**, at **381,000 and326,500** stage updates, with1850/2048 (CE0.366927141) and1845/2048 (CE0.261791346). Seed2 reaches1538/2048 best (75.097656%,CE2.802727520), exits1536/2048, and continues after its ceiling.

X6D makes the structure explicit: the T4 threshold is not four individually solved positions (seed0 P4 is314/512). By T8-best, seed0's first five joint columns are **512,512,512,504,1**; seed1's first six are **504,510,510,499,417,0**; seed2's first five are **511,512,498,219,0**. Three high-accuracy positions replicate across A seeds, with a fourth acquired under long training in seeds0/1, and a partial fifth shoulder in seed1. Best T8 totals are2041/4096,2440/4096,1740/4096, with CE5.724623531,3.182132885,7.139796525. No T8 threshold is reached.

This is a **prefix-then-cliff**, sometimes with a one-position shoulder, not smooth monotone carry decay: P2/P3 can outperform P1. T16-best seed1 extends the shoulder to448/512 at P5, then collapses at P6; it is not an additional solved position. Later aggregate dilution and the T128 table below are not evidence of tracking128 transformations.

## Arm B: reliable supervised single-carry acquisition, deeper instability

The T2 result is **3/3 neutral-evaluation threshold crossings**, at **448,000/448,000/379,000 stage updates**, or **497,000/483,500/432,500 global updates**. This was the **only reliable E97 single-carry unlock in the pre-X9 series**, not a claim that no A checkpoint ever acquired P2 at a later stage. All three P2 terminal counts at T2 threshold are materially learned (414/411/411 out of512), but below90% individually. The registered90% criterion is aggregate. The infra-partial legitimately contributes to this T2 denominator because acquisition predates the failure.

No B E97 crosses T4 or any later stage. T4 bests are1031/2048,1028/2048,1050/2048 (CE3.444996178,3.668652013,4.062654495). Later stages show acquisition/regression/recovery rather than stable depth extension. The sharpest **X6D B_seed0** receipt is T8: best P1/P2/P3=**505/511/459** out of512, declining to**73/56/18** at exit. Aggregate falls1475/4096→149/4096 (**36.010742%→3.637695%**), CE5.218444288→15.079596102; separate T1 retention is70/512. This is actual forgetting, not denominator dilution.

“Cycling” is descriptive instability during per-stage weaning/reset, **not a proved deterministic or causal read-the-carry mechanism**. B_seed0 T16-best occurs at499,500 stage updates, training p=.0005, and recovers P1–P4=**400,498,343,251** with no true-state feedback in evaluation. Near-zero training feedback can recover approximate deeper composition; aggregate-best can hide offsetting position changes. X6D did not evaluate B_e97_seed2; do not transfer these positional columns to that seed.

B GDN2 has the same qualitative shallow-prefix/T8+ ceiling shape, **not equal seedwise success**: T2 thresholds19,000/20,000/23,000; T4 crosses only **1/3**, seed0 at440,500 (1849/2048,CE0.284024019). Seeds1/2 top out at1289/2048 and1559/2048 at T4 (CE5.524982125/1.028617460). T8 bests2045/4096,1300/4096,2023/4096 (CE4.543555230/9.628274679/4.142706990). X6D B GDN2 seed0 confirms the cliff: T8-best P1–P5=512,512,512,507,1; T16-best=512,512,512,503,0. Carry supervision is not a general deep-ladder cure for either architecture under this recipe.

## T128 endpoint ledger — best is not exit

Every best/exit correct-count denominator is**65,536**; terminal denominator**512**; post-suffix7 denominator**61,440** (P9–P128). B_e97_seed2 has **no T128 observation**. Losses above the marginal are retained as observed, not clipped or omitted.

| Cell | Best correct (accuracy) | Best CE | Exit correct (accuracy) | Exit CE | Exit terminal /512 | Exit post-suffix7 /61440 |
|---|---:|---:|---:|---:|---:|---:|
| A_e97_seed0 | 538 (0.820923%) | 9.355451047 | 363 (0.553894%) | 17.745432496 | 0 | 15 |
| A_e97_seed1 | 644 (0.982666%) | 8.813167036 | 18 (0.027466%) | 12.895137966 | 0 | 16 |
| A_e97_seed2 | 1911 (2.915955%) | 7.975714684 | 1817 (2.772522%) | 11.440355837 | 1 | 15 |
| A_gdn2_seed0 | 2350 (3.585815%) | 11.109283209 | 1905 (2.906799%) | 11.547087014 | 0 | 23 |
| A_gdn2_seed1 | 2535 (3.868103%) | 8.411898553 | 2485 (3.791809%) | 9.024596989 | 1 | 17 |
| A_gdn2_seed2 | 1784 (2.722168%) | 8.266729653 | 1504 (2.294922%) | 9.036285043 | 1 | 20 |
| B_e97_seed0 | 1785 (2.723694%) | 7.996803701 | 1331 (2.030945%) | 8.147938669 | 0 | 16 |
| B_e97_seed1 | 1428 (2.178955%) | 8.237927198 | 1174 (1.791382%) | 8.164771020 | 1 | 10 |
| B_gdn2_seed0 | 2065 (3.150940%) | 8.615064025 | 1409 (2.149963%) | 8.016304493 | 0 | 15 |
| B_gdn2_seed1 | 1547 (2.360535%) | 8.088528514 | 1428 (2.178955%) | 8.299008846 | 0 | 18 |
| B_gdn2_seed2 | 2022 (3.085327%) | 7.991661251 | 1898 (2.896118%) | 10.780068219 | 0 | 15 |

The suffix-only reference corresponds to approximately16.51 expected matches among61,440 targets; these observed tail counts do not establish deep tracking. No significance test or universal chance model for the recurrent predictor is claimed.

## Historical placement: interpretation superseded, replication preserved

X6's Arm-A ceilings were initially read as intrinsic composition limits. **That interpretation is superseded by X9/X10/X7b; the measurements are not invalidated.** [X9](X9-growth-report.md) trained scratch E97 at T16 without a ladder or auxiliary carry supervision and reached512/512 at P1–P3 under AdamW+warmup/cosine. X10's fresh seed0 AdamW control subsequently exits500K at **512,512,512,512,7,0,1,0,…**, CE**5.949373424**, versus the fresh SF-constant control's **510,0,1,0,…**, CE**7.847842544**. X7b's stock GDN2 SF-constant T16 control exits at **512,512,512,512,0,…**, CE**6.166569799**, without X6's ladder. Terminal/result hashes and exact columns for these completed historical controls are frozen in this report's audit; their launch reports alone are not treated as completed evidence.

Thus **E97's repeated T2–T4 failures under SF-constant were optimizer-regime artifacts, not architectural impossibility**: curriculum could not compensate reliably for that optimizer regime. This is a reachability counterexample, not a claim that all SF variants fail or an isolated causal diagnosis of averaging. X9/X10 change optimizer class, schedule and betas as a bundle; nominal equal decay does not ensure equal shrink dynamics. X10/X7b comparisons here are seed0 only, not3-seed robustness estimates or a complete optimizer ranking. X6's late seed2 gains also rule out the stronger claim that SF absolutely forbids any deeper composition.

What the receipts actually establish:

1. **GDN2 composes three positions robustly in the A positional seeds, and four with long training in named SF runs**, with a cliff/shoulder beyond the prefix; aggregate T4 threshold success is not four-position perfection. X7b later shows a perfect four-position prefix without the ladder.
2. **Training-only carry supervision reliably teaches E97 a single carry** across all three seeds, measured without carry feedback; it does not stabilize the whole ladder.
3. **E97's failure to cross the shallow aggregate thresholds under SF replicates across the ladder and the broader series' horizon/ceiling designs**, rather than being rescued by curriculum alone. The horizon/ceiling evidence belongs to X5b/X5c; X6 supplies the twelve-cell ladder replication, not those separate interventions. Failure means the registered outcomes under this recipe, not zero deeper positional learning in every checkpoint.

X6 is therefore the **SF-era instrument record**, load-bearing as breadth and controls, not the final interpretation of model expressivity or an architecture-superiority result.

## Adjudicated infrastructure-partial and evidence limitations

`C/markers/B_e97_seed2.INFRA-FAILURE-DISK.json` adjudicates **disk exhaustion, not a training verdict**. Original terminal exit_code1 and the archived FAIL identity are preserved; console traceback is `OSError: [Errno 28] No space left on device`. It failed around global2,705,484 (adjudicated approximate location), during T64. Last persisted validation is exactly **global2,705,000 / T64 stage272,500**,948/32768 joint matches (2.893066%), CE**8.979399741**, terminal0/512 and post-suffix7 7/28672. It had completed T1,T2,T4,T8,T16,T32; T64 has no completed exit, and T128 never ran. The approved disposition is **partial ladder, banked science preserved, no rerun** (no resume path; freed compute assigned elsewhere). Do not count the unfinished T64 as a learning failure or the run as a successful full ladder.

The original adjudication's `banked_science` numerical shorthand was stale: “T1 threshold (506,945 steps)” and “T2…448K-class” are not seed2's primary receipts. The dated append-only companion **`C/markers/B_e97_seed2.INFRA-FAILURE-DISK.correction.json`** preserves the original text verbatim and its SHA, corrects T1 to**53,500**, T2 to**379,000 stage /432,500 global**, and distinguishes X6D's A_seed2/B_seed0 positional findings. The original adjudication JSON is **unchanged**, and so is its infra disposition. This report uses primary receipts, not remembered ranges.

Checkpoint evidence has a separate, already disclosed limitation: X6D found that the original pruning logic deleted stage-entry payloads after their best role expired. All40 same-pool boundary aggregate comparisons through T16→T32 matched, but only one actual next-entry payload survived for direct tensor/position equality;39 were aggregate-only comparisons with explicitly labeled surrogate forwards. Fix **b11a7d7e** applied to future imports, not already-running trainers; queued seeds1/2 subsequently bind the corrected runner hash in their configs. This report records each cell's actual source commit/hash, rather than pretending every process used identical retention code. No boundary-wide direct tensor proof, recovered checkpoint or GPU re-evaluation is claimed.

## Validation and scope

Reproduce on lambda01, CPU only:

```bash
.venv/bin/python experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/receipts/growth/audit.py > /tmp/x6-growth-audit.json
cmp /tmp/x6-growth-audit.json experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/receipts/growth/audit.json
git diff --check
```

The audit checks **12 cells,94 stage exits,76,415 validation rows**, same-seed archived pool identities, metric denominators/finite CE/zero out-of-range counts, stage-grid coverage, selected-best and exit reconciliation,11 DONE/terminal/result hash matches, the infra console cause/correction, and the named historical terminal/result identities. Full stage metrics/source hashes and lower-depth endpoint retention remain in `audit.json`. It does **not** rerun forwards, deserialize checkpoints, audit every training update, recover pruned payloads, or prove causal mechanisms. No new training or GPU evaluation was performed. Existing X6 prelaunch159-test and X6D131-test receipts remain historical evidence, not newly rerun test counts.

Local single-card expressivity only. Resilient design authority/gap matrix **R16 evidence discipline by analogy only**; no resilient DiLoCo, scheduler queue binding, systems conformance, production promotion or scale authorization is claimed. No operational code or live processes changed. Independent reviewer acceptance remains supervisor-owned.
