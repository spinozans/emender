# X8 — pay-for-depth T16: completed growth report

**Both arms DONE at the 500,000-update safety ceiling. Incentive suspect: CLOSED as REFUTED in this tested regime, not merely weakened.** E97 never consolidated even P1; GDN2 acquired P2 late and forewent P3 relative to the still-live X7 uniform-loss control. Increasing the reward for depth did not buy depth. This is a learning/optimization finding on this task, seed and budget, **not a capacity bound**.

Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage8_paydepth_8m/`. Registration/source commit **`3ccc6252`** preceded launch; [X8 launch report](X8-launch-report.md) records the intervention. E97 finished **2026-10-19T18:03:13Z**, GDN2 **21:07:18Z**. Both atomic DONE markers match their successful terminal receipts and result hashes. The original watcher recorded `DONE both pay-for-depth runs` at **21:09:01Z**; an independent `--once` returned **0** at **21:12:17Z** (separate `/tmp` log, original receipts untouched).

## 1. What changed, what the counts mean

Both scratch seed0 runs used T16 directly, p61/vocabulary3721, batch32, dim256/four layers/MLP ratio2, **9,900,800 E97 / 9,887,056 GDN2 parameters**, AdamWScheduleFree lr0.001, betas(0.9,0.95), wd0.01, clip1, no warmup. No ladder, carry supervision, threshold stop, stagnation stop or restart. Training examples were fresh procedural draws. The only stop was the safety ceiling, not successful learning.

Training loss was `mean_batch(sum((t/136)*CE_t, t=1..16))`, instead of X7's uniform `1/16` coefficients. P1's coefficient fell from **1/16 to 1/136**, an **8.5-fold reduction**; P16 received16/136. Coefficients sum to1; they are not measured fractions of gradient norm or realized CE. Validation stayed **unweighted full-state CE in nats**. Each of **1,001 validations per arm**, step0 through500K every500 updates, evaluated the same frozen512 T16 sequences as X7. Every position count below is **joint-state correct out of512**, not a coordinate count; aggregate denominator8192. P1 is computable from the visible initialization and first driver, with no deep carry prerequisite.

E97 retains X7 bf16's legacy state policy and sequential Triton `fused_inference=True` evaluation; its live scan accumulator and parameters were already fp32. GDN2 uses **stock FLA bf16-autocast kernels**, whereas X7 GDN2 uses its **sequential fp32 recurrence adapter**. Thus E97's corresponding comparison isolates weighting, but **GDN2's X7 comparison also changes precision/backend**. Its one-position deficit is descriptive, not an isolated causal estimate of weighting. Neither arm is an entirely bf16 accumulator.

Standing references: uniform marginal **ln(3721)=8.221747728 nats**; suffix-only full-state reference **1/3721** (0.0268744961%). The suffix theorem requires at least two omitted iid drivers and applies to an observer restricted to that suffix, **not** a recurrent model retaining prefix information. Here post-suffix7 scoring covers P9–P16,4096 targets. Chance-sized isolated counts do not establish a carry circuit; aggregate prefix credit does not establish deep tracking. These are validation observations, not independent held-out/test-set selection evidence.

## 2. E97: complete learning failure — P1 never gained

**Final P1 is2/512, aggregate4/8192, CE8.376918614 (above8.221747728).** This is worse than retaining X7's visible-prefix foothold, not merely another `512,2,0,...` shallow plateau. Across all1,001 observations P1's maximum is **3/512**: first at357,500, last at495,500. P1 counts are0 in735 observations,1 in189,2 in63,3 in14; never50, never512. There is **no learned P1 to collapse from**. The late deterioration is in CE, not the destruction of a previously accurate P1 solver.

| Update | P1 | P2 | P3 | P1 CE | Unweighted aggregate CE |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 250.771703959 | 250.976102829 |
| 500 | 1 | 0 | 0 | 20.853789577 | 20.757297397 |
| 1,000 | 0 | 0 | 0 | 10.869473550 | 10.914302111 |
| 5,000 | 0 | 0 | 0 | 8.221824942 | 8.225294948 |
| 50,000 | 0 | 0 | 0 | 8.222206870 | 8.221930921 |
| 150,000 | 0 | 0 | 0 | 8.221676910 | 8.221714616 |
| 250,000 | 0 | 0 | 0 | 8.221899712 | 8.221654594 |
| 300,000 | 0 | 1 | 0 | 8.220871637 | 8.221648395 |
| 350,000 | 2 | 0 | 0 | 8.019110894 | 8.209159672 |
| 357,500 (selected best) | 3 | 1 | 1 | 7.863669456 | 8.199530780 |
| 400,000 | 0 | 0 | 0 | 7.625972068 | 8.186608911 |
| 431,000 (minimum CE) | 1 | 0 | 0 | 7.142167341 | 8.158401191 |
| 450,000 | 0 | 0 | 0 | 7.698074222 | 8.191112578 |
| 475,000 | 1 | 0 | 1 | 9.903856896 | 8.329443693 |
| 490,000 | 1 | 1 | 0 | 10.725016078 | 8.387029648 |
| 500,000 | 2 | 1 | 0 | 10.249937594 | 8.376918614 |

P1 remains around the marginal through300K, exhibits limited CE improvement thereafter without joint-state consolidation, then deteriorates sharply between450K and475K. Even the **minimum-CE** row has only2/8192 joint matches. The retained **best checkpoint is selected by aggregate correct count, not minimum CE**:357,500 has7/8192, no learned foothold. Best-checkpoint columns and losses are below. No position P2+ ever reaches the preregistered **50/512** materiality criterion.

## 3. GDN2: delayed foothold acquisition and a foregone third position

**Final P1–P4:505,511,9,0; aggregate1027/8192; CE8.673311293, also above the marginal.** Its selected aggregate-best499,500 has505,511,10,0 and1028/8192, CE8.647817910. It eventually acquires a near-perfect P2, but P3 never exceeds10/512 in the entire run. P4 never exceeds2/512 and no P3+ position reaches50/512. This is not an extension beyondP4.

**Correction to the proposed collapse narrative:** P2 is **delayed acquisition**, not collapse/recovery: at281,000 it is8/512, but its maximum through that point is13/512, not a previously consolidated512. Likewise P3 is **foregone relative to X7**, not learned and then forgotten within X8. The actual mid-run regression is **P1**:511 at275,000 falls to210 at347,500, then recovers to505 at500,000.

| Update | P1 | P2 | P3 | P4 | Unweighted aggregate CE | Elapsed seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0 | 243.962320328 | 1.002 |
| 50,000 | 6 | 0 | 1 | 0 | 7.978224784 | 1,911.586 |
| 139,500 | 50 | 0 | 0 | 0 | 7.905971318 | 5,321.254 |
| 200,000 | 467 | 0 | 0 | 1 | 7.771081746 | 7,652.992 |
| 232,000 | 511 | 10 | 0 | 0 | 7.790532470 | 9,035.005 |
| 275,000 | 511 | 5 | 0 | 0 | 7.801663220 | 11,149.881 |
| 281,000 | 510 | 8 | 0 | 0 | 7.830403805 | 11,455.943 |
| 330,000 | 264 | 11 | 0 | 0 | 8.042632937 | 13,840.035 |
| 347,500 (P1 trough) | 210 | 29 | 0 | 0 | 8.130752206 | 14,622.888 |
| 351,500 (first P2 >=50) | 226 | 52 | 0 | 0 | 8.088728935 | 14,801.904 |
| 371,000 (first P2 >=256) | 326 | 256 | 0 | 1 | 7.641084611 | 15,734.136 |
| 401,500 (first P2 >=461) | 408 | 463 | 5 | 0 | 7.910088897 | 17,182.374 |
| 449,000 (first P2 >=500) | 472 | 501 | 6 | 0 | 9.090678275 | 19,458.107 |
| 494,000 (first P2 >=511) | 504 | 511 | 4 | 0 | 8.766505241 | 21,541.986 |
| 499,500 (selected best) | 505 | 511 | 10 | 0 | 8.647817910 | 21,804.203 |
| 500,000 | 505 | 511 | 9 | 0 | 8.673311293 | 21,826.282 |

Elapsed times are runner observations including intervening evaluation/checkpoint costs, not exclusive GPU compute. Crossings are first observed every500 updates, not exact continuous-time learning events or claims of monotonic retention. Neither P1 nor P2 ever attains512/512, so X8 GDN2's **strict perfect-prefix depth remains0**, even at its near-perfect two-position finish.

### X7 control: explicitly mid-run, not a completed control verdict

X7 GDN2's **uniform-loss, fp32-adapter** control has **no `result.json` yet** and remains in flight. At the requested **330,000** receipt its P1–P4 are **512,512,512,175**, aggregate1713/8192, CE6.269330531. At the latest validation captured **2026-10-19T21:11:34Z**, update**393,000**, they are **512,511,512,223**, aggregate1758/8192, CE7.286917061. These are two named observations of a live control, not an invented final500K result; X7's own growth report must finalize it. The registration's earlier **P4=116/512** comparison point is not the control's terminal score either.

In that control P2 first reaches50 at44,000 and512 at55,500; P3 first reaches50 at105,000 and512 at140,500. X8 spends its first351,000 updates below50 atP2, reaching52 at351,500: a **307,500-update lag** versus the control's first50 crossing, rather than an exact350K between-arm delay. At500K X8 still has only9 atP3 versus512 in both cited control rows. The observed **one-position cost** is two near-perfect positions instead of three perfect positions at the330K comparison (and no P4 shoulder instead of175). Backend/precision and unequal observation budgets prevent assigning this entire difference solely to loss weighting; they do not turn it into a depth gain.

## 4. Exact position columns, selected best versus exit

All entries are joint counts out of512; “best” is the runner's aggregate-count selection, not a per-position oracle. Maxima rows independently maximize each column across1,001 validations and **are not one checkpoint**.

| Arm / observation | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | Aggregate /8192 | CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E97 best357,500 | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 7 | 8.199530780 |
| E97 exit500,000 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 8.376918614 |
| E97 per-column maxima | 3 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | — | — |
| GDN2 best499,500 | 505 | 511 | 10 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1028 | 8.647817910 |
| GDN2 exit500,000 | 505 | 511 | 9 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1027 | 8.673311293 |
| GDN2 per-column maxima | 511 | 511 | 10 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 3 | 2 | 1 | 3 | — | — |

| Position | E97 exit CE | GDN2 exit CE |
|---:|---:|---:|
| P1 | 10.249937594 | 0.057940803 |
| P2 | 8.239100248 | 0.017106243 |
| P3 | 8.258748502 | 4.247877514 |
| P4 | 8.237292334 | 10.305707700 |
| P5 | 8.254709044 | 10.466099848 |
| P6 | 8.265720290 | 10.553275144 |
| P7 | 8.254412275 | 10.276129999 |
| P8 | 8.255654085 | 10.384213245 |
| P9 | 8.258750758 | 10.326287725 |
| P10 | 8.242318157 | 10.288873133 |
| P11 | 8.254078520 | 10.272658735 |
| P12 | 8.262351345 | 10.228873491 |
| P13 | 8.243024749 | 10.283777289 |
| P14 | 8.256087997 | 10.435482852 |
| P15 | 8.255429354 | 10.299932263 |
| P16 | 8.243083605 | 10.328745795 |

Exit terminal counts are **0/512 in both**; post-suffix7 counts are **E97 0/4096 / GDN2 1/4096** (selected best:1/4096 each). GDN2's prefix learning coexists with very poor deep CE; neither its aggregate counts nor its lower P3 CE establish a correct P3 solver. Final fresh-batch **weighted training** losses are E97**8.252180099** / GDN2**8.013441086**; these are not the unweighted frozen-validation losses above, nor the same parameter presentation as ScheduleFree evaluation. No loss series is silently substituted for another.

## 5. Two-arm reading and registered verdicts

- **Prediction(a), incentive rescue: REFUTED.** E97 gains no P2+ materiality (maxP2=2), and removes the supposed P1 attractor only by failing to learn P1 itself (max3). X7 E97 bf16's uniform control exits at`512,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0`, CE7.805131197. Its count-selected best has`512,6,0,0,0,1,0,0,0,0,0,0,1,0,0,0`, CE7.619156957. X8 does not trade easy credit for deep structure.
- **Prediction(b), failure to unlock entry by changing incentive: CONFIRMED for E97 and supported by both-arm joint reading.** Registration(b) literally concerns Run1/E97; the GDN2 extension is the joint interpretation, not a retroactively registered GDN2(b). GDN2 eventually learnsP2, but gains no third position or deeper frontier relative to control. “Initialization-structural” names the remaining basin-entry suspect, not proof that initialization alone is the cause or that optimizers/scale cannot help.
- **Prediction(c), GDN2 frontier extension beyondP4: REFUTED at X8's ceiling.** P4 max2; no P3+ reaches50. Relative to both the registered P4=116 snapshot and later175/223 observations, there is no extension. The precision/backend caveat remains mandatory.

**Joint reading (scoped per X8R): for E97 — the bf16-matched comparison — the receipts support the interpretation that learning needs a learnable foothold sufficiently prominent in the learning signal:** starving the objective with deep weighting did not force a carry circuit; it stopped E97 from consolidating even the foothold position itself. This is an interpretation of E97's matched evidence, not an established cross-architecture mechanism — per-position gradient dominance was not measured, and no intervention isolates the mechanism. **The GDN2 arm is descriptive only**: relative to the fp32-adapter X7 control it shows delayed P2 acquisition (first ≥50 at 351,500; ≥511 at 494,000) and a foregone third position, but the stock-bf16-vs-fp32-adapter backend change confounds any weighting-only causal reading; a matched stock-bf16 uniform-loss deep-first GDN2 control is commissioned to close this. The specific suspect "easy-position reward traps the model, so paying for depth unlocks it" is **CLOSED as REFUTED for E97 on matched evidence**, and closed for GDN2 descriptively pending the matched control — not left open as merely "weak."

## 6. Series ledger and next experiment boundary

| Probe | Ledger position | What remains unresolved |
|---|---|---|
| **X6: ladder / carry supervision** | **PARTIAL**: scaffolding unlocks learning; retention/weaning at greater depth remains problematic (see [X6D](X6D-report.md)). | Not a completed all-seed growth verdict; redesign the supervision schedule, not relabel the current wean as successful. |
| **X7: horizon + precision** | **REFUTED as the proposed unlock under tested interventions**; direct T16 and numerical relief do not establish the hoped-for deep carry rescue. | GDN2 control still live; finalize its exact trajectory in X7's growth report. No universal precision-irrelevance claim. |
| **X8: incentive / pay-for-depth** | **REFUTED; suspect CLOSED**. E97 never consolidatesP1; GDN2 acquiresP2 late and foregoesP3 relative to control. | Single seed/500K; GDN2 backend confound, not a capacity result. |
| **X9: optimizer regime** | **IN FLIGHT**, no growth verdict here. | AdamW+standard betas+warmup/cosine is a regime bundle versus ScheduleFree, not a clean schedule-only test ([launch](X9-launch-report.md)). |

Remaining interrogations after this ledger are **scale** and **supervision-schedule redesign**, alongside the already-running X9 optimizer-regime probe. This report authorizes no scale launch, restart, new arm, checkpoint pruning or production change. A negative small-model learning result is not evidence that either architecture cannot represent the recurrence.

## 7. Receipt verification and reproducibility anchors

Read-only audit on lambda01, **2026-10-19T21:12:17Z**: `.venv/bin/python /tmp/audit_x8_growth.py` passed. It checked all registered operational-source hashes and config-bound source/installed-FLA hashes; DONE/terminal/result identity; exactly500,000 finite, consecutive T16 training rows per arm; exactly1,001 validation steps per arm; all joint/x/y/OOR totals, terminal and suffix totals; position-mean versus aggregate CE (fp32 reduction tolerance); and perfect-prefix depth0 throughout. All OOR counts are0. Frozen archive/array hashes match receipts and X7 seed0. All **six retained entry/best/exit checkpoint payloads** match index SHA256s, step/depth metadata and empty-entry/nonempty-trained optimizer state. No checkpoint re-evaluation or training was performed. The temporary audit is a local validation aid, not a new production dependency or committed test; source receipts are the authoritative inputs.

Independent completion command: `.venv/bin/python scripts/watch_invertible_stage8_paydepth.py --once --log /tmp/x8-growth-watcher.log` returned0. No live receipt was rewritten. To reconstruct curves, read `C/NAME/validation.json`, select exact `step`, then extract `[p['correct'] for p in row['positions']]` and `row['aggregate']['loss']`; maxima and first crossings are over all1,001 rows, not the selected table alone. Training is `C/NAME/training.jsonl`; checkpoint identities are `checkpoints.json`; completion is `C/markers/NAME.DONE` plus `C/receipts/NAME/terminal.json`.

| Artifact | E97 SHA256 | GDN2 SHA256 |
|---|---|---|
| `result.json` | `e2748a2ad119f6cddd878217df379171abf7838c75bc1c3adafcfeacfc3a8040` | `dc5b45411eb8247bdcf50096e3b2dd052e421d0d9b323aa90ecaf53fc84a79eb` |
| `validation.json` | `7a4f79f21205978cdb76bbf5aea5ae45b4bc0b24da4f9f9aa23cc274e62b7385` | `ac07f58194cfcf21cd368b25dbb5ce96f0b1b2d8ad66423e7c4a426748815086` |
| `training.jsonl` | `9bfdfeda28855b61eb248d69cae8ed2d695fa354442e5696785d15d52217e9c1` | `118d9cdfdf4e9980c18472a9f65438c27a733d97e8a94b9eb278c2de34f41323` |
| entry step0 checkpoint | `cc68965848ef7128495596de5ed95da75bcbef3c810c2466a1607073daaec324` | `9bf137df5990b6c56dc7d77b9b0f7717f0a8e4694a2b384e3c32c60f5f08008b` |
| selected-best checkpoint | `42cee5d29911bcd45ec12b9e9b8c1c8d08d3ccea809fd9bfee260a1137bf053e` | `ec569bf8d65612c035e0dc693df1ab5021129e421220199e3cbc0f74d24dc576` |
| exit step500K checkpoint | `322327c2613292994d14298c3f37373b3592e3781096d06eaf586af90ccfb2c0` | `ad0ebb8926fd288a2750ac03162d70c7bc3a91548b3f11ef77078caa47cf9919` |

The X7 live validation file captured for the393K comparison had SHA256`cc31d0946cc6a6d32e94ba76a49fad93dd76f0803a17aaf612c17b4cef331179`; later live writes will change it. Its330K checkpoint receipt records SHA256`cea1a8cfc6bac9a79018c3c2f52389de101ca0e8b3911420a182591afed0fb84`; that historical payload was not claimed retained or re-evaluated here. Existing raw completion streams/checkpoints remain local; this commit adds only the report and does not sweep unrelated untracked experiment evidence into Git.

Scope follows `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` / `docs/RESILIENT_DILOCO_GAP_MATRIX.md`: **R16 exact-source/evidence discipline only by analogy**, not satisfaction of a production/resilient DiLoCo gate. No Slurm Partition/QOS binding, systems-scale acceptance or capacity claim. No tests added or changed for this documentation-only task; receipt auditing and diff checks are its validation. Independent acceptance review remains reviewer-owned.
