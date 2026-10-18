# X6D — positional diagnosis of the moving X6 ceilings

## Verdicts

**(a) E97 seed2 late-ladder composition is REAL, but not a binary solved-depth ladder.** At T4-best driver2 is mostly coordinate-only: J=20/512, X=509, Y=21. At T8-best driver2 becomes **511/512** (driver3 still only41 joint, X425/Y42). At T16-best drivers3 **and4** reach **370/512 and349/512**, with X418/388 and Y371/349: not isolated joint hits or an aggregate-fraction artifact. This is approximate deeper full-state composition, accompanied by damage to earlier positions: driver1=412/512 and driver2=474/512. Thus T16 does not solve a contiguous three-driver prefix at >=90%; the first position is worse than the second. The 25% -> 25.8% -> 19.6% aggregates hide both genuine position gains and retention loss.

A E97 seed0's T16 **exit**, not best, is below one-position fraction: driver1 falls from512/512 at best to266/512 at exit (51.95%; X494/Y266), explaining aggregate3.369%. Seed1 exit driver1 is471/512, with driver2=104/512; aggregate7.080% is not proof of two solved positions. Best accuracies are6.421% and7.214%, respectively. These are explicitly different checkpoints.

**(b) B-arm weaning-associated positional regression is CONFIRMED at T8; a universal/causal read-the-carry cycling mechanism is NOT established.** T4-best tracks drivers1/2 (512/511), not driver3 (8 joint, X277/Y14). T8-best improves driver3 to459/512 (89.65%, just below registered90%) while drivers1/2 remain505/511. By T8-exit those three collapse to73/56/18 (the separately recorded T1 retention pool is70/512,13.6719%). This is real loss of already learned positions during the anneal, not denominator dilution. However, T16-best occurs at stage499500, training p=0.0005, with J400/498/343/251: near-zero-feedback training can recover deep approximate composition. At T16, aggregate barely changes from entry18.213% to best18.237%, while drivers1/3 decline and driver4 grows from0 to251. Therefore “best aggregate” can conceal interference, and repeated deterministic cycling or inability to internalize carry is not demonstrated. All diagnostic forwards have **zero** true-state feedback; no teacher-forced comparison or causal mechanism intervention was authorized.

**(c) GDN2 T8+ is a PREFIX PLATEAU WITH A CLIFF / ONE-POSITION SHOULDER, not smooth monotone decaying carry.** A seed0 and B seed0 retain approximately four drivers and crash at driver5. A seed1 retains approximately four plus a partially learned fifth (417/512 at T8;448/512 at T16), then crashes at driver6. Its59.5703% T8 is exactly2440/4096 = (504+510+510+499+417)/4096, not a position fraction. Driver2/3 can outperform driver1: strict monotonicity is false. Small later isolated counts are shown, not called carried states. This is empirical positional structure, not a model-capacity bound or proof of a specific carry circuit.

**(d) Boundary verdict: no observed aggregate boundary loss; direct positional evidence is limited by a real retention defect.** All40 same-previous-pool boundary comparisons through T16->T32 reproduce the carried checkpoint's recorded aggregate metrics exactly, including full-state loss and coordinate counts. The one extant next-entry payload, B GDN2 T16->T32, has byte-for-byte equal model tensors and identical full positional rows to the carried best. The other39 next-entry payloads are missing and have different full-payload SHAs from the prior carried files, so direct positional equality cannot honestly be proved there. Surrogate entry forwards from retained carried weights reproduce every captured entry aggregate exactly but remain explicitly labeled surrogates. Equal aggregates alone cannot exclude offsetting position changes. There is strong **within-stage** interference (especially B T8 and A E97 seed0 T16); no evidence that changing stage labels alone causes it.

## Registration, runtime and provenance

- Registered before forwards: root `X6D-plan.md` and evidence `registration.md`; registration SHA is frozen in `inventory.json`. The post-discovery amendment, also before forwards, explicitly authorized missing-entry surrogates, pruning audit, and a future-launch-only fix. Snapshot starts2026-10-18T13:27:03Z. All eight runs were live; no files inside any live per-run directory were written, loaded directly, restored, pruned or interrupted.
- Evidence directory `E = experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/receipts/positional_X6D`. Frozen per-run metadata contains original file hashes, selected rows, stages, transitions, checkpoint identities, pruning events and data receipts. Full metadata copies, original frozen NPZs and **hash-verified checkpoint copies** reside in `/home/erikg/emender-scratch/X6D-positiondiagnosis`. Only scratch payload paths were deserialized. Full live indexes were hashed but only relevant excerpts are committed, avoiding mutable entire histories.
- Selection frozen to completed T1/T2/T4/T8/T16 best **and** exit for every run, plus own-stage entry diagnostics and next-entry boundary comparisons. No current-stage best search, data regeneration, training or test-pool use. Stage best remains maximum aggregate validation accuracy, then minimum loss. T32 ongoing/completed composition and later stages are outside this registered ceiling study; T32-entry on T16 is included.
- Python3.12.3, Torch2.9.1+cu128, CUDA12.8, lambda01 RTX6000 Ada. Sequential, **co-resident GPU3** (UUID GPU-e316c33c-fbf1-5e77-041e-36161c04652a), not an exclusive lease, explicitly supervisor-approved. GPU before/after snapshots retained. Evaluation2026-10-18T13:27:41.747Z–13:29:04.213Z, **82.280s**. Temporary slowdown was estimated pre-pass as up to roughly50% instantaneous throughput while overlapping; actual slowdown cannot be separated from live-run variation without a counterfactual and is **not measured**.
- Executed source commit `f806efa427d0c3d7877b188f7b92936cf35ba5fb`; exact executed evaluator/runner dependencies are content-hashed in every receipt (new files/future-fix contents were not already in that HEAD). The current runner retention fix was on disk before these evals but changes no forward/model behavior. Live trainer config hashes bind the original pre-fix source; `audit.json` records that distinction. No live process was restarted.
- Strict saved ScheduleFree eval-x load, model.eval/no_grad, fixed seed, batch32, deterministic algorithms, TF32 matmul off, CUBLAS `:4096:8`, bf16 autocast/fp32 parameters. X6's own forward including B's neutral carry slots and original causal padding. Exactly512 sequences per pool; joint argmax decoded X=token//61,Y=token%61; invalid tokens excluded. Loss is full-state CE in nats, **not coordinate CE**. Every OOR count is0.
- **96 distinct checkpoint/pool forwards**,16 batches each (=1536 batches,49152 sequence evaluations); **17 explicitly recorded aliases** avoid identical checkpoint/pool re-evaluation (e.g. best=exit).113 logical jobs. JSON per distinct checkpoint/pool plus aliases binds all requested checkpoint roles. All integer aggregate/terminal/suffix counts reconcile exactly; canonical aggregate loss delta is **0.0 for all96 and17 aliases**. Position-average versus canonical loss summation differs by at most1.094e-6 nats, from fp64 column means versus fp32 batch reductions, not a reproduced-metric mismatch.
- “High-accuracy solved position” means >=461/512 (registered >=90%); empirical nonzero positions are never silently discarded. Different depth pools contain different sequences, so cross-depth position differences are descriptive rather than paired-example causal tests. Same-depth best/exit and same-previous-pool boundaries are paired.

## Sharpest per-position joint table

All entries are **joint correct/512**; vectors enumerate drivers1 throughT, including tails. `best` and `exit` are distinct selections unless identical.

| Run | Pool | Checkpoint | J by driver |
|---|---:|---|---|
| A_e97_seed2 | T4 | best | 512, 20, 0, 0 |
| A_e97_seed2 | T4 | exit | 511, 10, 0, 1 |
| A_e97_seed2 | T8 | best | 508, 511, 41, 1, 0, 0, 0, 0 |
| A_e97_seed2 | T8 | exit | 508, 511, 35, 1, 0, 0, 0, 0 |
| A_e97_seed2 | T16 | best | 412, 474, 370, 349, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 |
| A_e97_seed2 | T16 | exit | 409, 453, 385, 354, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 |
| A_e97_seed0 | T16 | best | 512, 12, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0 |
| A_e97_seed0 | T16 | exit | 266, 8, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| A_e97_seed1 | T16 | best | 476, 109, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 |
| A_e97_seed1 | T16 | exit | 471, 104, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| B_e97_seed0 | T4 | best | 512, 511, 8, 0 |
| B_e97_seed0 | T4 | exit | 493, 508, 1, 0 |
| B_e97_seed0 | T8 | best | 505, 511, 459, 0, 0, 0, 0, 0 |
| B_e97_seed0 | T8 | exit | 73, 56, 18, 0, 0, 0, 2, 0 |
| B_e97_seed0 | T16 | best | 400, 498, 343, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 |
| B_e97_seed0 | T16 | exit | 398, 497, 339, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 |

## GDN2 profile shape at retained bests

| Run | Pool | J by driver | First imperfect | First <90% |
|---|---:|---|---:|---:|
| A_gdn2_seed0 | T8 | 512, 512, 512, 504, 1, 0, 0, 0 | 4 | 5 |
| A_gdn2_seed0 | T16 | 512, 512, 509, 502, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 | 3 | 5 |
| A_gdn2_seed1 | T8 | 504, 510, 510, 499, 417, 0, 0, 0 | 1 | 5 |
| A_gdn2_seed1 | T16 | 510, 509, 512, 497, 448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 | 1 | 5 |
| B_gdn2_seed0 | T8 | 512, 512, 512, 507, 1, 0, 0, 1 | 4 | 5 |
| B_gdn2_seed0 | T16 | 512, 512, 512, 503, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 | 4 | 5 |
| A_gdn2_seed2 | T8 | 511, 512, 498, 219, 0, 0, 0, 0 | 1 | 4 |
| A_gdn2_seed2 | T16 | 463, 501, 487, 311, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0 | 1 | 4 |

## Entry -> best -> exit interference on each stage’s own frozen pool

Entry rows marked **S** are surrogate forwards from the recorded carried checkpoint, not recovered entry payloads. Initial entries unavailable. Driver sets below meet >=90%; the full J/X/Y/loss table follows. Entry metrics and all surrogate reproductions reconcile exactly. The stage_step here is the **selected best’s own training step**, not the inherited checkpoint’s former-stage step.

| Run | Stage | Entry J total (S) | Best J total | Exit J total | >=90% drivers entry -> best -> exit | Best stage step | Best training carry p |
|---|---:|---:|---:|---:|---|---:|---:|
| A_e97_seed0 | T1 | unavailable | 462 | 462 | unavailable -> 1 -> 1 | 50000 | 0.000000 |
| A_e97_seed0 | T2 | 473 S | 527 | 521 | 1 -> 1 -> 1 | 499500 | 0.000000 |
| A_e97_seed0 | T4 | 519 S | 528 | 520 | 1 -> 1 -> 1 | 97500 | 0.000000 |
| A_e97_seed0 | T8 | 525 S | 532 | 515 | 1 -> 1 -> 1 | 10500 | 0.000000 |
| A_e97_seed0 | T16 | 521 S | 526 | 276 | 1 -> 1 -> none | 147500 | 0.000000 |
| A_e97_seed1 | T1 | unavailable | 466 | 466 | unavailable -> 1 -> 1 | 35500 | 0.000000 |
| A_e97_seed1 | T2 | 454 S | 526 | 520 | none -> 1 -> 1 | 395000 | 0.000000 |
| A_e97_seed1 | T4 | 518 S | 529 | 519 | 1 -> 1 -> 1 | 217000 | 0.000000 |
| A_e97_seed1 | T8 | 525 S | 529 | 519 | 1 -> 1 -> 1 | 24000 | 0.000000 |
| A_e97_seed1 | T16 | 525 S | 591 | 580 | 1 -> 1 -> 1 | 496500 | 0.000000 |
| A_e97_seed2 | T1 | unavailable | 466 | 466 | unavailable -> 1 -> 1 | 52500 | 0.000000 |
| A_e97_seed2 | T2 | 478 S | 522 | 519 | 1 -> 1 -> 1 | 55000 | 0.000000 |
| A_e97_seed2 | T4 | 519 S | 532 | 522 | 1 -> 1 -> 1 | 379500 | 0.000000 |
| A_e97_seed2 | T8 | 517 S | 1061 | 1055 | 1 -> 1,2 -> 1,2 | 496500 | 0.000000 |
| A_e97_seed2 | T16 | 1057 S | 1606 | 1602 | 1,2 -> 2 -> none | 455500 | 0.000000 |
| A_gdn2_seed0 | T1 | unavailable | 478 | 478 | unavailable -> 1 -> 1 | 13000 | 0.000000 |
| A_gdn2_seed0 | T2 | 484 S | 969 | 969 | 1 -> 1 -> 1 | 15500 | 0.000000 |
| A_gdn2_seed0 | T4 | 972 S | 1850 | 1850 | 1 -> 1,2,3 -> 1,2,3 | 381000 | 0.000000 |
| A_gdn2_seed0 | T8 | 1829 S | 2041 | 1830 | 1,2,3 -> 1,2,3,4 -> 2,3 | 90000 | 0.000000 |
| A_gdn2_seed0 | T16 | 2035 S | 2038 | 1876 | 1,2,3,4 -> 1,2,3,4 -> 1,2,3 | 25000 | 0.000000 |
| A_gdn2_seed1 | T1 | unavailable | 474 | 474 | unavailable -> 1 -> 1 | 12500 | 0.000000 |
| A_gdn2_seed1 | T2 | 478 S | 980 | 980 | 1 -> 1,2 -> 1,2 | 16500 | 0.000000 |
| A_gdn2_seed1 | T4 | 983 S | 1845 | 1845 | 1,2 -> 1,2,3 -> 1,2,3 | 326500 | 0.000000 |
| A_gdn2_seed1 | T8 | 1838 S | 2440 | 2440 | 1,2,3 -> 1,2,3,4 -> 1,2,3,4 | 500000 | 0.000000 |
| A_gdn2_seed1 | T16 | 2438 S | 2477 | 2294 | 1,2,3,4 -> 1,2,3,4 -> 1,2,3 | 47000 | 0.000000 |
| A_gdn2_seed2 | T1 | unavailable | 474 | 474 | unavailable -> 1 -> 1 | 13500 | 0.000000 |
| A_gdn2_seed2 | T2 | 489 S | 926 | 926 | 1 -> 1 -> 1 | 16500 | 0.000000 |
| A_gdn2_seed2 | T4 | 925 S | 1538 | 1536 | 1 -> 1,2,3 -> 1,2,3 | 496000 | 0.000000 |
| A_gdn2_seed2 | T8 | 1537 S | 1740 | 1731 | 1,2,3 -> 1,2,3 -> 1,2,3 | 482500 | 0.000000 |
| A_gdn2_seed2 | T16 | 1728 S | 1766 | 1642 | 1,2,3 -> 1,2,3 -> 2 | 360500 | 0.000000 |
| B_e97_seed0 | T1 | unavailable | 461 | 461 | unavailable -> 1 -> 1 | 49000 | 0.451000 |
| B_e97_seed0 | T2 | 463 S | 926 | 926 | 1 -> 1 -> 1 | 448000 | 0.052000 |
| B_e97_seed0 | T4 | 938 S | 1031 | 1002 | 1 -> 1,2 -> 1,2 | 326500 | 0.173500 |
| B_e97_seed0 | T8 | 1029 S | 1475 | 149 | 1,2 -> 1,2 -> none | 295500 | 0.204500 |
| B_e97_seed0 | T16 | 1492 S | 1494 | 1487 | 1,2,3 -> 2 -> 2 | 499500 | 0.000500 |
| B_gdn2_seed0 | T1 | unavailable | 465 | 465 | unavailable -> 1 -> 1 | 13000 | 0.487000 |
| B_gdn2_seed0 | T2 | 469 S | 982 | 982 | 1 -> 1,2 -> 1,2 | 19000 | 0.481000 |
| B_gdn2_seed0 | T4 | 980 S | 1849 | 1849 | 1,2 -> 1,2,3 -> 1,2,3 | 440500 | 0.059500 |
| B_gdn2_seed0 | T8 | 1841 S | 2045 | 1676 | 1,2,3 -> 1,2,3,4 -> 1,2,3 | 117000 | 0.383000 |
| B_gdn2_seed0 | T16 | 2038 S | 2041 | 2005 | 1,2,3,4 -> 1,2,3,4 -> 1,2,3,4 | 10000 | 0.490000 |

## FINDING: boundary retention contract violated; fixed for future launches only

The original runner used one `required` set for overlapping roles. At entry, the checkpoint was both `stage-entry` and `stage-best`. When best changed, `required.discard(previous_best_path)` removed its boundary protection too. Under the below400GiB free-space policy, prune then deleted that entry. This is not merely an inference from absent filenames: captured `pruning.jsonl` events explicitly name every missing entry and give deletion epoch timestamps. All40 entries at T1–T16 are missing; another7 T32 entries are missing; B GDN2's T32 entry remains. No required selected stage-best or exit was missing.

Fix commit **b11a7d7e** adds a persistent `boundary_checkpoints` set and prevents removal of boundary members when their best role expires. The regression invokes the real tiny mocked training loop and real pruning under simulated low disk, changes best after entry, and asserts every stage-entry, exit and best payload survives while superseded nonboundary best is pruned. The old logic removed entry protection on a best change; the new regression passes with explicit boundary-role protection. This future-launch change does not affect already-imported live trainers or recover missing files. No claim is made that current processes now honor corrected retention.

Serialized entry files and carried files include different depth/stage-step/config/RNG metadata, so their **full payload SHAs differ**, even when model weights ought to match. We did not mislabel unequal SHA chains as proof of equality. The sole retained boundary's tensor comparison supplies direct evidence independent of serialized metadata.

### Same-previous-pool boundary evidence

Every row has exactly equal recorded aggregate metric dicts (including loss); direct means the actual next-entry payload was copied and evaluated. Missing rows are aggregate-only, not direct positional proof. Full entry and carried checkpoint hashes plus exact deletion paths/timestamps are in metadata/audit JSONs.

| Run | Boundary | Direct positional evidence | Entry payload deletion UTC |
|---|---|---|---|
| A_e97_seed0 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T20:04:04.720714+00:00 |
| A_e97_seed0 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T22:38:17.455977+00:00 |
| A_e97_seed0 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T01:48:21.214741+00:00 |
| A_e97_seed0 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T05:05:33.850785+00:00 |
| A_e97_seed0 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T08:58:12.451443+00:00 |
| A_e97_seed1 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T19:53:33.658271+00:00 |
| A_e97_seed1 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T22:31:55.035413+00:00 |
| A_e97_seed1 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T01:39:43.348873+00:00 |
| A_e97_seed1 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T04:53:44.646666+00:00 |
| A_e97_seed1 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T08:42:27.966527+00:00 |
| A_e97_seed2 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T19:50:45.024556+00:00 |
| A_e97_seed2 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T22:42:14.247992+00:00 |
| A_e97_seed2 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T01:42:28.234953+00:00 |
| A_e97_seed2 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T04:57:05.439399+00:00 |
| A_e97_seed2 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T08:47:51.416624+00:00 |
| A_gdn2_seed0 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T20:37:05.646035+00:00 |
| A_gdn2_seed0 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T20:37:05.646035+00:00 |
| A_gdn2_seed0 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-17T23:48:04.454865+00:00 |
| A_gdn2_seed0 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T04:57:21.006072+00:00 |
| A_gdn2_seed0 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T10:16:11.502267+00:00 |
| A_gdn2_seed1 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T19:57:12.739769+00:00 |
| A_gdn2_seed1 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T19:57:12.739769+00:00 |
| A_gdn2_seed1 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-17T23:10:50.869219+00:00 |
| A_gdn2_seed1 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T04:29:49.998194+00:00 |
| A_gdn2_seed1 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T10:10:23.328179+00:00 |
| A_gdn2_seed2 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T20:16:12.855641+00:00 |
| A_gdn2_seed2 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T20:16:12.855641+00:00 |
| A_gdn2_seed2 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T01:06:13.984491+00:00 |
| A_gdn2_seed2 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T06:09:07.270344+00:00 |
| A_gdn2_seed2 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T11:25:38.378328+00:00 |
| B_e97_seed0 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T19:50:46.797180+00:00 |
| B_e97_seed0 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T22:27:20.010345+00:00 |
| B_e97_seed0 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T01:43:30.632984+00:00 |
| B_e97_seed0 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T09:23:35.905758+00:00 |
| B_e97_seed0 | T16 -> T32 | unavailable; aggregate equal only | 2026-10-18T09:24:49.811860+00:00 |
| B_gdn2_seed0 | T1 -> T2 | unavailable; aggregate equal only | 2026-10-17T19:57:12.655736+00:00 |
| B_gdn2_seed0 | T2 -> T4 | unavailable; aggregate equal only | 2026-10-17T19:57:12.655736+00:00 |
| B_gdn2_seed0 | T4 -> T8 | unavailable; aggregate equal only | 2026-10-18T00:36:42.868117+00:00 |
| B_gdn2_seed0 | T8 -> T16 | unavailable; aggregate equal only | 2026-10-18T05:58:35.058765+00:00 |
| B_gdn2_seed0 | T16 -> T32 | equal tensors + positions | retained |

### Exact discarded entry inventory (T1–T16)

Each path basename is under that named live run; SHA and full path are in its immutable metadata. This table supplies exact pruning times, not filesystem-mtime guesses.

| Run | Entry depth | Discarded checkpoint | Pruned UTC |
|---|---:|---|---|
| A_e97_seed0 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T20:04:04.720714+00:00 |
| A_e97_seed0 | T2 | `step_050000_T2_s00000.pt` | 2026-10-17T20:04:04.720714+00:00 |
| A_e97_seed0 | T4 | `step_550000_T4_s00000.pt` | 2026-10-17T22:38:17.455977+00:00 |
| A_e97_seed0 | T8 | `step_1050000_T8_s00000.pt` | 2026-10-18T01:48:21.214741+00:00 |
| A_e97_seed0 | T16 | `step_1550000_T16_s00000.pt` | 2026-10-18T05:05:33.850785+00:00 |
| A_e97_seed1 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T19:53:33.658271+00:00 |
| A_e97_seed1 | T2 | `step_035500_T2_s00000.pt` | 2026-10-17T19:53:33.658271+00:00 |
| A_e97_seed1 | T4 | `step_535500_T4_s00000.pt` | 2026-10-17T22:31:55.035413+00:00 |
| A_e97_seed1 | T8 | `step_1035500_T8_s00000.pt` | 2026-10-18T01:39:43.348873+00:00 |
| A_e97_seed1 | T16 | `step_1535500_T16_s00000.pt` | 2026-10-18T04:53:44.646666+00:00 |
| A_e97_seed2 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T19:48:42.391837+00:00 |
| A_e97_seed2 | T2 | `step_052500_T2_s00000.pt` | 2026-10-17T19:50:45.024556+00:00 |
| A_e97_seed2 | T4 | `step_552500_T4_s00000.pt` | 2026-10-17T22:42:14.247992+00:00 |
| A_e97_seed2 | T8 | `step_1052500_T8_s00000.pt` | 2026-10-18T01:42:28.234953+00:00 |
| A_e97_seed2 | T16 | `step_1552500_T16_s00000.pt` | 2026-10-18T04:57:05.439399+00:00 |
| A_gdn2_seed0 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T20:37:05.646035+00:00 |
| A_gdn2_seed0 | T2 | `step_013000_T2_s00000.pt` | 2026-10-17T20:37:05.646035+00:00 |
| A_gdn2_seed0 | T4 | `step_028500_T4_s00000.pt` | 2026-10-17T20:37:05.646035+00:00 |
| A_gdn2_seed0 | T8 | `step_409500_T8_s00000.pt` | 2026-10-17T23:48:04.454865+00:00 |
| A_gdn2_seed0 | T16 | `step_909500_T16_s00000.pt` | 2026-10-18T04:57:21.006072+00:00 |
| A_gdn2_seed1 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T19:57:12.739769+00:00 |
| A_gdn2_seed1 | T2 | `step_012500_T2_s00000.pt` | 2026-10-17T19:57:12.739769+00:00 |
| A_gdn2_seed1 | T4 | `step_029000_T4_s00000.pt` | 2026-10-17T19:57:12.739769+00:00 |
| A_gdn2_seed1 | T8 | `step_355500_T8_s00000.pt` | 2026-10-17T23:10:50.869219+00:00 |
| A_gdn2_seed1 | T16 | `step_855500_T16_s00000.pt` | 2026-10-18T04:29:49.998194+00:00 |
| A_gdn2_seed2 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T20:16:12.855641+00:00 |
| A_gdn2_seed2 | T2 | `step_013500_T2_s00000.pt` | 2026-10-17T20:16:12.855641+00:00 |
| A_gdn2_seed2 | T4 | `step_030000_T4_s00000.pt` | 2026-10-17T20:16:12.855641+00:00 |
| A_gdn2_seed2 | T8 | `step_530000_T8_s00000.pt` | 2026-10-18T01:06:13.984491+00:00 |
| A_gdn2_seed2 | T16 | `step_1030000_T16_s00000.pt` | 2026-10-18T06:09:07.270344+00:00 |
| B_e97_seed0 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T19:50:46.797180+00:00 |
| B_e97_seed0 | T2 | `step_049000_T2_s00000.pt` | 2026-10-17T19:50:46.797180+00:00 |
| B_e97_seed0 | T4 | `step_497000_T4_s00000.pt` | 2026-10-17T22:27:20.010345+00:00 |
| B_e97_seed0 | T8 | `step_997000_T8_s00000.pt` | 2026-10-18T01:43:30.632984+00:00 |
| B_e97_seed0 | T16 | `step_1497000_T16_s00000.pt` | 2026-10-18T09:23:35.905758+00:00 |
| B_gdn2_seed0 | T1 | `step_000000_T1_s00000.pt` | 2026-10-17T19:57:12.655736+00:00 |
| B_gdn2_seed0 | T2 | `step_013000_T2_s00000.pt` | 2026-10-17T19:57:12.655736+00:00 |
| B_gdn2_seed0 | T4 | `step_032000_T4_s00000.pt` | 2026-10-17T19:57:12.655736+00:00 |
| B_gdn2_seed0 | T8 | `step_472500_T8_s00000.pt` | 2026-10-18T00:36:42.868117+00:00 |
| B_gdn2_seed0 | T16 | `step_972500_T16_s00000.pt` | 2026-10-18T05:58:35.058765+00:00 |

## Validation / reproducibility

Completed commands (do **not** rerun GPU evaluation for acceptance):

```bash
.venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage6_positions snapshot
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage6_positions evaluate
.venv/bin/python -m experiments.expressivity_tasks.audit_invertible_stage6_positions
.venv/bin/python -m pytest -q tests/test_eval_invertible_stage6_positions.py tests/test_invertible_stage6_truecur_8m.py tests/test_eval_invertible_stage5c_positions.py tests/test_invertible_stage3_8m.py tests/test_invertible_stage5_curriculum_8m.py tests/test_invertible_stage5b_growth_8m.py tests/test_replay_stagnation_rule.py tests/test_invertible_quadratic.py
git diff --check
```

CPU suite: **131 passed**. Initial suite:130 passed/1 failed because the new test demanded a fixed1e-6 tolerance on position-mean versus canonical reduction at entry loss33.8987; observed delta1.09323e-6. The test was corrected to an absolute-plus-relative fp32 summation tolerance (1e-6 +1e-7*loss). **Canonical reproduced aggregate loss remains exactly equal**, not tolerance-relaxed. Initial output retained as `tests_initial.txt`; no receipt edits or GPU reruns. The earlier targeted gate passed25 tests before receipt audit.

`audit.json` and `audit_console.txt` verify all96 hashes/count reconciliations,17 aliases, all40 boundary aggregates,47 missing entries via captured pruning records (40 within T1–T16 plus7 T32), and the actual B GDN2 boundary model-tensor equality. `SHA256SUMS` binds diagnostic artifacts and report; scratch checkpoint payloads and NPZs are intentionally not committed. Snapshot metadata and receipts already include their SHA256s. New accounting tests cover every registered depth, coordinate semantics/invalid tokens, suffix/terminal sums, hash-safe copies, partial JSONL appends, and immutable receipt reconciliation. This task is not resilient/DiLoCo training and uses no Frontier/Slurm queue; requirement IDs in the resilient gap matrix are not applicable. Independent reviewer acceptance remains supervisor-owned.

Residual limitations: same selected validation pools, not independent generalization evidence; entry tensor/position proof missing at39 boundaries; no causal teacher-feedback contrast or proof of oscillatory mechanism; exact slowdown unmeasured; future retention correction requires new launches and cannot retroactively repair live runs. Existing unrelated artifacts remain untouched/uncommitted.

## Complete checkpoint-role positional tables

Every row denominator is512; X/Y are coordinates of the **same joint argmax**, and all OOR=0. CE loss shown to9 significant digits; JSON retains full precision. Best=exit roles are displayed separately even when their forward is an explicit alias. `entry-S` rows are carried-checkpoint surrogates and identify the inherited checkpoint's original stage/step. Missing initial entries have no fabricated table.


### A_e97_seed0


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 50000 | 1 | 462 | 512 | 462 | 0.39784117 |
| exit | T1 / 50000 | 1 | 462 | 512 | 462 | 0.39784117 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 50000 | 1 | 472 | 512 | 472 | 0.372732208 |
| entry-S | T1 / 50000 | 2 | 1 | 8 | 10 | 22.8726528 |
| best | T2 / 499500 | 1 | 512 | 512 | 512 | 0.000847830019 |
| best | T2 / 499500 | 2 | 15 | 511 | 15 | 4.35255078 |
| exit | T2 / 500000 | 1 | 512 | 512 | 512 | 0.000846722537 |
| exit | T2 / 500000 | 2 | 9 | 511 | 9 | 4.34957109 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 499500 | 1 | 512 | 512 | 512 | 0.000674661001 |
| entry-S | T2 / 499500 | 2 | 7 | 512 | 7 | 4.36704628 |
| entry-S | T2 / 499500 | 3 | 0 | 12 | 6 | 12.432192 |
| entry-S | T2 / 499500 | 4 | 0 | 7 | 11 | 12.4907553 |
| best | T4 / 97500 | 1 | 512 | 512 | 512 | 0.00259337233 |
| best | T4 / 97500 | 2 | 16 | 512 | 16 | 4.28953526 |
| best | T4 / 97500 | 3 | 0 | 10 | 5 | 8.25090448 |
| best | T4 / 97500 | 4 | 0 | 5 | 8 | 8.24367277 |
| exit | T4 / 500000 | 1 | 510 | 510 | 510 | 0.0158910942 |
| exit | T4 / 500000 | 2 | 10 | 512 | 10 | 4.1241683 |
| exit | T4 / 500000 | 3 | 0 | 8 | 14 | 8.22333857 |
| exit | T4 / 500000 | 4 | 0 | 7 | 8 | 8.22175923 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 97500 | 1 | 512 | 512 | 512 | 0.00253683221 |
| entry-S | T4 / 97500 | 2 | 12 | 512 | 12 | 4.2818033 |
| entry-S | T4 / 97500 | 3 | 1 | 10 | 10 | 8.25300927 |
| entry-S | T4 / 97500 | 4 | 0 | 5 | 9 | 8.25492812 |
| entry-S | T4 / 97500 | 5 | 0 | 12 | 5 | 8.23542477 |
| entry-S | T4 / 97500 | 6 | 0 | 10 | 4 | 8.24584483 |
| entry-S | T4 / 97500 | 7 | 0 | 11 | 7 | 8.2359914 |
| entry-S | T4 / 97500 | 8 | 0 | 7 | 11 | 8.25049184 |
| best | T8 / 10500 | 1 | 512 | 512 | 512 | 0.00306226741 |
| best | T8 / 10500 | 2 | 18 | 512 | 18 | 4.27003179 |
| best | T8 / 10500 | 3 | 0 | 10 | 9 | 8.24792434 |
| best | T8 / 10500 | 4 | 0 | 5 | 7 | 8.24779619 |
| best | T8 / 10500 | 5 | 1 | 12 | 8 | 8.23090577 |
| best | T8 / 10500 | 6 | 1 | 10 | 12 | 8.23809441 |
| best | T8 / 10500 | 7 | 0 | 11 | 9 | 8.23092866 |
| best | T8 / 10500 | 8 | 0 | 7 | 8 | 8.2430055 |
| exit | T8 / 500000 | 1 | 498 | 503 | 501 | 0.260584629 |
| exit | T8 / 500000 | 2 | 17 | 508 | 17 | 5.07481016 |
| exit | T8 / 500000 | 3 | 0 | 19 | 7 | 8.22261862 |
| exit | T8 / 500000 | 4 | 0 | 6 | 11 | 8.22642369 |
| exit | T8 / 500000 | 5 | 0 | 8 | 5 | 8.24699552 |
| exit | T8 / 500000 | 6 | 0 | 6 | 9 | 8.2412392 |
| exit | T8 / 500000 | 7 | 0 | 6 | 6 | 8.22845112 |
| exit | T8 / 500000 | 8 | 0 | 13 | 9 | 8.22543029 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 10500 | 1 | 512 | 512 | 512 | 0.00171394547 |
| entry-S | T8 / 10500 | 2 | 7 | 512 | 7 | 4.27537808 |
| entry-S | T8 / 10500 | 3 | 0 | 5 | 7 | 8.25747923 |
| entry-S | T8 / 10500 | 4 | 0 | 6 | 10 | 8.23700793 |
| entry-S | T8 / 10500 | 5 | 0 | 8 | 8 | 8.24033521 |
| entry-S | T8 / 10500 | 6 | 0 | 6 | 10 | 8.25511903 |
| entry-S | T8 / 10500 | 7 | 0 | 9 | 5 | 8.24089466 |
| entry-S | T8 / 10500 | 8 | 0 | 8 | 8 | 8.23591723 |
| entry-S | T8 / 10500 | 9 | 0 | 10 | 10 | 8.23931526 |
| entry-S | T8 / 10500 | 10 | 0 | 11 | 8 | 8.23631499 |
| entry-S | T8 / 10500 | 11 | 0 | 9 | 7 | 8.23856093 |
| entry-S | T8 / 10500 | 12 | 1 | 6 | 11 | 8.24180996 |
| entry-S | T8 / 10500 | 13 | 0 | 7 | 6 | 8.23333284 |
| entry-S | T8 / 10500 | 14 | 1 | 8 | 8 | 8.2445202 |
| entry-S | T8 / 10500 | 15 | 0 | 8 | 11 | 8.23455518 |
| entry-S | T8 / 10500 | 16 | 0 | 9 | 3 | 8.23735498 |
| best | T16 / 147500 | 1 | 512 | 512 | 512 | 0.0111051297 |
| best | T16 / 147500 | 2 | 12 | 512 | 12 | 4.14913147 |
| best | T16 / 147500 | 3 | 0 | 9 | 8 | 8.2282246 |
| best | T16 / 147500 | 4 | 0 | 3 | 4 | 8.22511014 |
| best | T16 / 147500 | 5 | 0 | 5 | 12 | 8.22572628 |
| best | T16 / 147500 | 6 | 0 | 7 | 7 | 8.22355856 |
| best | T16 / 147500 | 7 | 0 | 9 | 10 | 8.22777525 |
| best | T16 / 147500 | 8 | 0 | 8 | 7 | 8.22653303 |
| best | T16 / 147500 | 9 | 0 | 10 | 6 | 8.21927609 |
| best | T16 / 147500 | 10 | 0 | 8 | 9 | 8.2258238 |
| best | T16 / 147500 | 11 | 1 | 8 | 8 | 8.22406982 |
| best | T16 / 147500 | 12 | 0 | 9 | 10 | 8.22593093 |
| best | T16 / 147500 | 13 | 1 | 11 | 9 | 8.22363449 |
| best | T16 / 147500 | 14 | 0 | 6 | 12 | 8.22502837 |
| best | T16 / 147500 | 15 | 0 | 9 | 15 | 8.2225852 |
| best | T16 / 147500 | 16 | 0 | 4 | 7 | 8.22417635 |
| exit | T16 / 500000 | 1 | 266 | 494 | 266 | 2.83359956 |
| exit | T16 / 500000 | 2 | 8 | 294 | 10 | 15.5058411 |
| exit | T16 / 500000 | 3 | 0 | 21 | 9 | 9.6920427 |
| exit | T16 / 500000 | 4 | 2 | 8 | 5 | 9.21626868 |
| exit | T16 / 500000 | 5 | 0 | 5 | 7 | 9.20439947 |
| exit | T16 / 500000 | 6 | 0 | 5 | 6 | 9.24482399 |
| exit | T16 / 500000 | 7 | 0 | 5 | 7 | 9.23247691 |
| exit | T16 / 500000 | 8 | 0 | 9 | 3 | 9.24596147 |
| exit | T16 / 500000 | 9 | 0 | 5 | 10 | 9.27523176 |
| exit | T16 / 500000 | 10 | 0 | 6 | 5 | 9.13595733 |
| exit | T16 / 500000 | 11 | 0 | 15 | 6 | 8.99932726 |
| exit | T16 / 500000 | 12 | 0 | 5 | 13 | 9.29758685 |
| exit | T16 / 500000 | 13 | 0 | 2 | 5 | 9.23842978 |
| exit | T16 / 500000 | 14 | 0 | 6 | 5 | 9.24132008 |
| exit | T16 / 500000 | 15 | 0 | 9 | 10 | 9.09760367 |
| exit | T16 / 500000 | 16 | 0 | 12 | 8 | 9.22150773 |

### A_e97_seed1


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 35500 | 1 | 466 | 512 | 466 | 0.340934969 |
| exit | T1 / 35500 | 1 | 466 | 512 | 466 | 0.340934969 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 35500 | 1 | 454 | 512 | 454 | 0.378260055 |
| entry-S | T1 / 35500 | 2 | 0 | 8 | 11 | 19.9487662 |
| best | T2 / 395000 | 1 | 512 | 512 | 512 | 0.000683534639 |
| best | T2 / 395000 | 2 | 14 | 511 | 14 | 4.30152346 |
| exit | T2 / 500000 | 1 | 512 | 512 | 512 | 0.000716370267 |
| exit | T2 / 500000 | 2 | 8 | 512 | 8 | 4.23414396 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 395000 | 1 | 512 | 512 | 512 | 0.000546588638 |
| entry-S | T2 / 395000 | 2 | 6 | 512 | 6 | 4.32009534 |
| entry-S | T2 / 395000 | 3 | 0 | 10 | 12 | 12.4245966 |
| entry-S | T2 / 395000 | 4 | 0 | 8 | 2 | 12.6791457 |
| best | T4 / 217000 | 1 | 512 | 512 | 512 | 0.00226518625 |
| best | T4 / 217000 | 2 | 17 | 512 | 17 | 4.15074243 |
| best | T4 / 217000 | 3 | 0 | 6 | 14 | 8.22937701 |
| best | T4 / 217000 | 4 | 0 | 3 | 10 | 8.22798417 |
| exit | T4 / 500000 | 1 | 510 | 510 | 512 | 0.0124998341 |
| exit | T4 / 500000 | 2 | 9 | 511 | 9 | 4.21928156 |
| exit | T4 / 500000 | 3 | 0 | 9 | 8 | 8.22337962 |
| exit | T4 / 500000 | 4 | 0 | 9 | 8 | 8.22565055 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 217000 | 1 | 512 | 512 | 512 | 0.00275338934 |
| entry-S | T4 / 217000 | 2 | 13 | 512 | 13 | 4.15002454 |
| entry-S | T4 / 217000 | 3 | 0 | 8 | 9 | 8.22977871 |
| entry-S | T4 / 217000 | 4 | 0 | 7 | 7 | 8.21800395 |
| entry-S | T4 / 217000 | 5 | 0 | 14 | 5 | 8.21946218 |
| entry-S | T4 / 217000 | 6 | 0 | 7 | 8 | 8.22407793 |
| entry-S | T4 / 217000 | 7 | 0 | 9 | 10 | 8.22534892 |
| entry-S | T4 / 217000 | 8 | 0 | 9 | 3 | 8.21995832 |
| best | T8 / 24000 | 1 | 512 | 512 | 512 | 0.00196295545 |
| best | T8 / 24000 | 2 | 16 | 512 | 16 | 4.15240708 |
| best | T8 / 24000 | 3 | 0 | 8 | 9 | 8.22731288 |
| best | T8 / 24000 | 4 | 0 | 7 | 6 | 8.21882939 |
| best | T8 / 24000 | 5 | 1 | 14 | 5 | 8.22016025 |
| best | T8 / 24000 | 6 | 0 | 7 | 8 | 8.223046 |
| best | T8 / 24000 | 7 | 0 | 9 | 6 | 8.22415542 |
| best | T8 / 24000 | 8 | 0 | 9 | 9 | 8.2215938 |
| exit | T8 / 500000 | 1 | 490 | 510 | 490 | 0.244396255 |
| exit | T8 / 500000 | 2 | 28 | 472 | 29 | 5.83045547 |
| exit | T8 / 500000 | 3 | 0 | 37 | 9 | 8.29101101 |
| exit | T8 / 500000 | 4 | 1 | 5 | 10 | 8.27143928 |
| exit | T8 / 500000 | 5 | 0 | 11 | 7 | 8.24246355 |
| exit | T8 / 500000 | 6 | 0 | 7 | 8 | 8.25451053 |
| exit | T8 / 500000 | 7 | 0 | 7 | 11 | 8.2495428 |
| exit | T8 / 500000 | 8 | 0 | 12 | 6 | 8.25833227 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 24000 | 1 | 512 | 512 | 512 | 0.00797654925 |
| entry-S | T8 / 24000 | 2 | 13 | 511 | 13 | 4.1607857 |
| entry-S | T8 / 24000 | 3 | 0 | 7 | 6 | 8.2183818 |
| entry-S | T8 / 24000 | 4 | 0 | 10 | 5 | 8.22408866 |
| entry-S | T8 / 24000 | 5 | 0 | 5 | 6 | 8.22266675 |
| entry-S | T8 / 24000 | 6 | 0 | 14 | 5 | 8.22043599 |
| entry-S | T8 / 24000 | 7 | 0 | 5 | 7 | 8.22577523 |
| entry-S | T8 / 24000 | 8 | 0 | 8 | 8 | 8.22099334 |
| entry-S | T8 / 24000 | 9 | 0 | 10 | 7 | 8.22836398 |
| entry-S | T8 / 24000 | 10 | 0 | 5 | 12 | 8.22567514 |
| entry-S | T8 / 24000 | 11 | 0 | 8 | 9 | 8.22366349 |
| entry-S | T8 / 24000 | 12 | 0 | 9 | 4 | 8.21848541 |
| entry-S | T8 / 24000 | 13 | 0 | 6 | 12 | 8.22454107 |
| entry-S | T8 / 24000 | 14 | 0 | 9 | 10 | 8.22581235 |
| entry-S | T8 / 24000 | 15 | 0 | 9 | 5 | 8.22621125 |
| entry-S | T8 / 24000 | 16 | 0 | 10 | 6 | 8.2180639 |
| best | T16 / 496500 | 1 | 476 | 497 | 478 | 0.504363456 |
| best | T16 / 496500 | 2 | 109 | 472 | 110 | 3.93095731 |
| best | T16 / 496500 | 3 | 5 | 101 | 11 | 11.2908826 |
| best | T16 / 496500 | 4 | 0 | 12 | 7 | 9.01979629 |
| best | T16 / 496500 | 5 | 0 | 7 | 4 | 9.01135492 |
| best | T16 / 496500 | 6 | 0 | 8 | 11 | 9.07713898 |
| best | T16 / 496500 | 7 | 0 | 10 | 10 | 8.86632388 |
| best | T16 / 496500 | 8 | 0 | 7 | 9 | 8.95977086 |
| best | T16 / 496500 | 9 | 0 | 4 | 11 | 8.96592708 |
| best | T16 / 496500 | 10 | 0 | 9 | 8 | 9.10001015 |
| best | T16 / 496500 | 11 | 0 | 9 | 5 | 8.91266604 |
| best | T16 / 496500 | 12 | 0 | 10 | 11 | 9.04113049 |
| best | T16 / 496500 | 13 | 0 | 7 | 10 | 8.95810754 |
| best | T16 / 496500 | 14 | 0 | 13 | 7 | 8.97391116 |
| best | T16 / 496500 | 15 | 1 | 15 | 5 | 8.85274576 |
| best | T16 / 496500 | 16 | 0 | 10 | 5 | 9.08805947 |
| exit | T16 / 500000 | 1 | 471 | 495 | 473 | 0.514340434 |
| exit | T16 / 500000 | 2 | 104 | 473 | 105 | 3.95303609 |
| exit | T16 / 500000 | 3 | 4 | 106 | 8 | 11.3461646 |
| exit | T16 / 500000 | 4 | 0 | 12 | 9 | 9.00188917 |
| exit | T16 / 500000 | 5 | 0 | 7 | 8 | 9.00485862 |
| exit | T16 / 500000 | 6 | 0 | 8 | 9 | 9.04735295 |
| exit | T16 / 500000 | 7 | 1 | 10 | 9 | 8.8449348 |
| exit | T16 / 500000 | 8 | 0 | 7 | 12 | 8.94334562 |
| exit | T16 / 500000 | 9 | 0 | 4 | 7 | 8.96289487 |
| exit | T16 / 500000 | 10 | 0 | 9 | 11 | 9.08886282 |
| exit | T16 / 500000 | 11 | 0 | 9 | 8 | 8.92581168 |
| exit | T16 / 500000 | 12 | 0 | 10 | 7 | 9.01678893 |
| exit | T16 / 500000 | 13 | 0 | 7 | 5 | 8.94240096 |
| exit | T16 / 500000 | 14 | 0 | 13 | 5 | 8.94636992 |
| exit | T16 / 500000 | 15 | 0 | 15 | 7 | 8.82855617 |
| exit | T16 / 500000 | 16 | 0 | 10 | 11 | 9.07459213 |

### A_e97_seed2


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 52500 | 1 | 466 | 512 | 466 | 0.382823742 |
| exit | T1 / 52500 | 1 | 466 | 512 | 466 | 0.382823742 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 52500 | 1 | 477 | 512 | 477 | 0.350996138 |
| entry-S | T1 / 52500 | 2 | 1 | 7 | 10 | 23.1806224 |
| best | T2 / 55000 | 1 | 512 | 512 | 512 | 0.00149163743 |
| best | T2 / 55000 | 2 | 10 | 467 | 11 | 5.82262018 |
| exit | T2 / 500000 | 1 | 512 | 512 | 512 | 0.000608818533 |
| exit | T2 / 500000 | 2 | 7 | 508 | 7 | 4.30866526 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 55000 | 1 | 512 | 512 | 512 | 0.00148693709 |
| entry-S | T2 / 55000 | 2 | 6 | 458 | 7 | 5.86739127 |
| entry-S | T2 / 55000 | 3 | 0 | 10 | 7 | 10.9747648 |
| entry-S | T2 / 55000 | 4 | 1 | 11 | 12 | 10.8835324 |
| best | T4 / 379500 | 1 | 512 | 512 | 512 | 0.00100920315 |
| best | T4 / 379500 | 2 | 20 | 509 | 21 | 4.35691451 |
| best | T4 / 379500 | 3 | 0 | 7 | 12 | 8.22321179 |
| best | T4 / 379500 | 4 | 0 | 12 | 7 | 8.22206536 |
| exit | T4 / 500000 | 1 | 511 | 511 | 511 | 0.00764586135 |
| exit | T4 / 500000 | 2 | 10 | 463 | 11 | 4.71815033 |
| exit | T4 / 500000 | 3 | 0 | 7 | 7 | 8.22184831 |
| exit | T4 / 500000 | 4 | 1 | 9 | 11 | 8.22258558 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 379500 | 1 | 512 | 512 | 512 | 0.00187577805 |
| entry-S | T4 / 379500 | 2 | 5 | 510 | 5 | 4.35260747 |
| entry-S | T4 / 379500 | 3 | 0 | 6 | 4 | 8.22289457 |
| entry-S | T4 / 379500 | 4 | 0 | 4 | 9 | 8.22396534 |
| entry-S | T4 / 379500 | 5 | 0 | 9 | 7 | 8.22180387 |
| entry-S | T4 / 379500 | 6 | 0 | 12 | 3 | 8.22402016 |
| entry-S | T4 / 379500 | 7 | 0 | 8 | 8 | 8.22650551 |
| entry-S | T4 / 379500 | 8 | 0 | 2 | 13 | 8.22375674 |
| best | T8 / 496500 | 1 | 508 | 511 | 509 | 0.0166401875 |
| best | T8 / 496500 | 2 | 511 | 511 | 511 | 0.0632028729 |
| best | T8 / 496500 | 3 | 41 | 425 | 42 | 8.88389804 |
| best | T8 / 496500 | 4 | 1 | 8 | 6 | 8.21687524 |
| best | T8 / 496500 | 5 | 0 | 4 | 10 | 8.22013291 |
| best | T8 / 496500 | 6 | 0 | 12 | 7 | 8.22589492 |
| best | T8 / 496500 | 7 | 0 | 9 | 6 | 8.21889271 |
| best | T8 / 496500 | 8 | 0 | 12 | 8 | 8.22242848 |
| exit | T8 / 500000 | 1 | 508 | 511 | 509 | 0.0183374138 |
| exit | T8 / 500000 | 2 | 511 | 511 | 511 | 0.0636645331 |
| exit | T8 / 500000 | 3 | 35 | 429 | 35 | 8.59976601 |
| exit | T8 / 500000 | 4 | 1 | 9 | 8 | 8.21543906 |
| exit | T8 / 500000 | 5 | 0 | 2 | 11 | 8.22030287 |
| exit | T8 / 500000 | 6 | 0 | 8 | 7 | 8.22614378 |
| exit | T8 / 500000 | 7 | 0 | 12 | 6 | 8.21906686 |
| exit | T8 / 500000 | 8 | 0 | 10 | 7 | 8.22238099 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 496500 | 1 | 510 | 512 | 510 | 0.0167317231 |
| entry-S | T8 / 496500 | 2 | 510 | 511 | 511 | 0.0285414696 |
| entry-S | T8 / 496500 | 3 | 36 | 413 | 36 | 8.75839042 |
| entry-S | T8 / 496500 | 4 | 0 | 7 | 12 | 8.2071466 |
| entry-S | T8 / 496500 | 5 | 0 | 9 | 7 | 8.22869048 |
| entry-S | T8 / 496500 | 6 | 0 | 4 | 3 | 8.22151858 |
| entry-S | T8 / 496500 | 7 | 0 | 10 | 3 | 8.222221 |
| entry-S | T8 / 496500 | 8 | 0 | 8 | 11 | 8.22367414 |
| entry-S | T8 / 496500 | 9 | 0 | 10 | 8 | 8.22368849 |
| entry-S | T8 / 496500 | 10 | 0 | 9 | 9 | 8.22079725 |
| entry-S | T8 / 496500 | 11 | 0 | 4 | 11 | 8.22669624 |
| entry-S | T8 / 496500 | 12 | 0 | 4 | 8 | 8.2245927 |
| entry-S | T8 / 496500 | 13 | 0 | 9 | 9 | 8.22333248 |
| entry-S | T8 / 496500 | 14 | 0 | 6 | 10 | 8.21943159 |
| entry-S | T8 / 496500 | 15 | 1 | 9 | 13 | 8.22202263 |
| entry-S | T8 / 496500 | 16 | 0 | 7 | 6 | 8.22404486 |
| best | T16 / 455500 | 1 | 412 | 494 | 413 | 1.40922944 |
| best | T16 / 455500 | 2 | 474 | 474 | 474 | 1.73679002 |
| best | T16 / 455500 | 3 | 370 | 418 | 371 | 3.56445391 |
| best | T16 / 455500 | 4 | 349 | 388 | 349 | 3.7984052 |
| best | T16 / 455500 | 5 | 0 | 8 | 4 | 8.21997465 |
| best | T16 / 455500 | 6 | 0 | 7 | 2 | 8.22153236 |
| best | T16 / 455500 | 7 | 0 | 11 | 9 | 8.22258935 |
| best | T16 / 455500 | 8 | 0 | 10 | 8 | 8.22209418 |
| best | T16 / 455500 | 9 | 0 | 5 | 7 | 8.22117826 |
| best | T16 / 455500 | 10 | 0 | 8 | 8 | 8.22310219 |
| best | T16 / 455500 | 11 | 0 | 11 | 5 | 8.22042424 |
| best | T16 / 455500 | 12 | 1 | 5 | 9 | 8.22125492 |
| best | T16 / 455500 | 13 | 0 | 7 | 6 | 8.22689107 |
| best | T16 / 455500 | 14 | 0 | 7 | 10 | 8.21983537 |
| best | T16 / 455500 | 15 | 0 | 7 | 11 | 8.22165569 |
| best | T16 / 455500 | 16 | 0 | 7 | 13 | 8.22165467 |
| exit | T16 / 500000 | 1 | 409 | 495 | 411 | 1.67601271 |
| exit | T16 / 500000 | 2 | 453 | 454 | 454 | 2.61378579 |
| exit | T16 / 500000 | 3 | 385 | 422 | 387 | 4.60174928 |
| exit | T16 / 500000 | 4 | 354 | 374 | 354 | 5.67078591 |
| exit | T16 / 500000 | 5 | 0 | 10 | 4 | 8.2198662 |
| exit | T16 / 500000 | 6 | 0 | 8 | 11 | 8.22081701 |
| exit | T16 / 500000 | 7 | 0 | 7 | 9 | 8.22216039 |
| exit | T16 / 500000 | 8 | 0 | 10 | 7 | 8.22152969 |
| exit | T16 / 500000 | 9 | 0 | 9 | 3 | 8.22122393 |
| exit | T16 / 500000 | 10 | 0 | 13 | 5 | 8.22174846 |
| exit | T16 / 500000 | 11 | 0 | 12 | 11 | 8.21975832 |
| exit | T16 / 500000 | 12 | 0 | 9 | 5 | 8.22064023 |
| exit | T16 / 500000 | 13 | 1 | 7 | 8 | 8.22619587 |
| exit | T16 / 500000 | 14 | 0 | 10 | 15 | 8.22028408 |
| exit | T16 / 500000 | 15 | 0 | 6 | 4 | 8.22164332 |
| exit | T16 / 500000 | 16 | 0 | 9 | 14 | 8.22149999 |

### A_gdn2_seed0


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 13000 | 1 | 478 | 508 | 478 | 0.305319822 |
| exit | T1 / 13000 | 1 | 478 | 508 | 478 | 0.305319822 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 13000 | 1 | 484 | 511 | 484 | 0.270862251 |
| entry-S | T1 / 13000 | 2 | 0 | 10 | 4 | 21.3376628 |
| best | T2 / 15500 | 1 | 512 | 512 | 512 | 0.000711386585 |
| best | T2 / 15500 | 2 | 457 | 477 | 457 | 0.527836782 |
| exit | T2 / 15500 | 1 | 512 | 512 | 512 | 0.000711386585 |
| exit | T2 / 15500 | 2 | 457 | 477 | 457 | 0.527836782 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 15500 | 1 | 512 | 512 | 512 | 0.000606535335 |
| entry-S | T2 / 15500 | 2 | 460 | 474 | 461 | 0.53017064 |
| entry-S | T2 / 15500 | 3 | 0 | 14 | 8 | 16.3548392 |
| entry-S | T2 / 15500 | 4 | 0 | 8 | 17 | 17.0867548 |
| best | T4 / 381000 | 1 | 512 | 512 | 512 | 1.67225097e-05 |
| best | T4 / 381000 | 2 | 512 | 512 | 512 | 3.13035172e-05 |
| best | T4 / 381000 | 3 | 512 | 512 | 512 | 0.000171910798 |
| best | T4 / 381000 | 4 | 314 | 500 | 314 | 1.46748863 |
| exit | T4 / 381000 | 1 | 512 | 512 | 512 | 1.67225097e-05 |
| exit | T4 / 381000 | 2 | 512 | 512 | 512 | 3.13035172e-05 |
| exit | T4 / 381000 | 3 | 512 | 512 | 512 | 0.000171910798 |
| exit | T4 / 381000 | 4 | 314 | 500 | 314 | 1.46748863 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 381000 | 1 | 512 | 512 | 512 | 2.08438624e-05 |
| entry-S | T4 / 381000 | 2 | 512 | 512 | 512 | 4.09658798e-05 |
| entry-S | T4 / 381000 | 3 | 512 | 512 | 512 | 0.000243688856 |
| entry-S | T4 / 381000 | 4 | 292 | 503 | 292 | 1.58754471 |
| entry-S | T4 / 381000 | 5 | 0 | 12 | 11 | 100.421501 |
| entry-S | T4 / 381000 | 6 | 1 | 10 | 8 | 101.887514 |
| entry-S | T4 / 381000 | 7 | 0 | 4 | 2 | 108.863399 |
| entry-S | T4 / 381000 | 8 | 0 | 6 | 12 | 106.933933 |
| best | T8 / 90000 | 1 | 512 | 512 | 512 | 1.1615781e-05 |
| best | T8 / 90000 | 2 | 512 | 512 | 512 | 3.21512868e-05 |
| best | T8 / 90000 | 3 | 512 | 512 | 512 | 0.00268970861 |
| best | T8 / 90000 | 4 | 504 | 512 | 504 | 0.0680284687 |
| best | T8 / 90000 | 5 | 1 | 11 | 8 | 11.2319432 |
| best | T8 / 90000 | 6 | 0 | 8 | 9 | 11.3204435 |
| best | T8 / 90000 | 7 | 0 | 10 | 6 | 11.7506807 |
| best | T8 / 90000 | 8 | 0 | 10 | 7 | 11.4231586 |
| exit | T8 / 500000 | 1 | 425 | 453 | 431 | 0.678806868 |
| exit | T8 / 500000 | 2 | 490 | 504 | 493 | 0.141756421 |
| exit | T8 / 500000 | 3 | 505 | 510 | 507 | 0.0560220042 |
| exit | T8 / 500000 | 4 | 408 | 439 | 411 | 0.684973852 |
| exit | T8 / 500000 | 5 | 1 | 28 | 10 | 24.4490076 |
| exit | T8 / 500000 | 6 | 0 | 7 | 8 | 8.21290863 |
| exit | T8 / 500000 | 7 | 0 | 9 | 6 | 8.22253908 |
| exit | T8 / 500000 | 8 | 1 | 14 | 4 | 8.22363694 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 90000 | 1 | 512 | 512 | 512 | 1.33864171e-05 |
| entry-S | T8 / 90000 | 2 | 512 | 512 | 512 | 3.08284476e-05 |
| entry-S | T8 / 90000 | 3 | 511 | 511 | 511 | 0.0116871188 |
| entry-S | T8 / 90000 | 4 | 497 | 510 | 497 | 0.0812415169 |
| entry-S | T8 / 90000 | 5 | 1 | 11 | 11 | 11.3258136 |
| entry-S | T8 / 90000 | 6 | 0 | 6 | 9 | 11.3796938 |
| entry-S | T8 / 90000 | 7 | 0 | 6 | 11 | 11.6610712 |
| entry-S | T8 / 90000 | 8 | 0 | 8 | 8 | 11.4977311 |
| entry-S | T8 / 90000 | 9 | 0 | 16 | 5 | 11.2633764 |
| entry-S | T8 / 90000 | 10 | 1 | 6 | 5 | 11.561002 |
| entry-S | T8 / 90000 | 11 | 0 | 11 | 10 | 11.5321491 |
| entry-S | T8 / 90000 | 12 | 0 | 2 | 6 | 11.4777794 |
| entry-S | T8 / 90000 | 13 | 1 | 7 | 11 | 11.4711975 |
| entry-S | T8 / 90000 | 14 | 0 | 7 | 7 | 11.3480464 |
| entry-S | T8 / 90000 | 15 | 0 | 9 | 7 | 11.5419781 |
| entry-S | T8 / 90000 | 16 | 0 | 6 | 7 | 11.7652614 |
| best | T16 / 25000 | 1 | 512 | 512 | 512 | 1.80482823e-05 |
| best | T16 / 25000 | 2 | 512 | 512 | 512 | 4.61435126e-05 |
| best | T16 / 25000 | 3 | 509 | 509 | 509 | 0.0408402634 |
| best | T16 / 25000 | 4 | 502 | 510 | 502 | 0.064040836 |
| best | T16 / 25000 | 5 | 0 | 11 | 8 | 10.1149818 |
| best | T16 / 25000 | 6 | 0 | 6 | 11 | 10.1569427 |
| best | T16 / 25000 | 7 | 0 | 6 | 20 | 10.3001293 |
| best | T16 / 25000 | 8 | 1 | 8 | 10 | 10.2742143 |
| best | T16 / 25000 | 9 | 0 | 16 | 9 | 10.1053613 |
| best | T16 / 25000 | 10 | 0 | 6 | 7 | 10.3149111 |
| best | T16 / 25000 | 11 | 1 | 11 | 9 | 10.2545716 |
| best | T16 / 25000 | 12 | 0 | 2 | 6 | 10.1659906 |
| best | T16 / 25000 | 13 | 0 | 7 | 11 | 10.2228688 |
| best | T16 / 25000 | 14 | 0 | 7 | 5 | 10.0692993 |
| best | T16 / 25000 | 15 | 0 | 10 | 7 | 10.2155818 |
| best | T16 / 25000 | 16 | 1 | 11 | 11 | 10.3236676 |
| exit | T16 / 500000 | 1 | 466 | 490 | 469 | 0.325467626 |
| exit | T16 / 500000 | 2 | 479 | 496 | 489 | 0.266716863 |
| exit | T16 / 500000 | 3 | 510 | 512 | 510 | 0.0115752119 |
| exit | T16 / 500000 | 4 | 415 | 451 | 417 | 0.578898714 |
| exit | T16 / 500000 | 5 | 2 | 25 | 7 | 18.3485069 |
| exit | T16 / 500000 | 6 | 0 | 15 | 12 | 8.22369212 |
| exit | T16 / 500000 | 7 | 1 | 10 | 11 | 8.22308228 |
| exit | T16 / 500000 | 8 | 0 | 12 | 7 | 8.22214901 |
| exit | T16 / 500000 | 9 | 0 | 8 | 8 | 8.22328003 |
| exit | T16 / 500000 | 10 | 1 | 9 | 5 | 8.22213275 |
| exit | T16 / 500000 | 11 | 0 | 8 | 9 | 8.22407272 |
| exit | T16 / 500000 | 12 | 0 | 9 | 7 | 8.22614132 |
| exit | T16 / 500000 | 13 | 0 | 4 | 13 | 8.22296184 |
| exit | T16 / 500000 | 14 | 1 | 14 | 10 | 8.22436653 |
| exit | T16 / 500000 | 15 | 1 | 15 | 9 | 8.22259481 |
| exit | T16 / 500000 | 16 | 0 | 12 | 9 | 8.22163225 |

### A_gdn2_seed1


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 12500 | 1 | 474 | 510 | 474 | 0.334320525 |
| exit | T1 / 12500 | 1 | 474 | 510 | 474 | 0.334320525 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 12500 | 1 | 477 | 510 | 477 | 0.299520481 |
| entry-S | T1 / 12500 | 2 | 1 | 8 | 7 | 17.799637 |
| best | T2 / 16500 | 1 | 512 | 512 | 512 | 0.000740792661 |
| best | T2 / 16500 | 2 | 468 | 491 | 470 | 0.585754959 |
| exit | T2 / 16500 | 1 | 512 | 512 | 512 | 0.000740792661 |
| exit | T2 / 16500 | 2 | 468 | 491 | 470 | 0.585754959 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 16500 | 1 | 512 | 512 | 512 | 0.00069839012 |
| entry-S | T2 / 16500 | 2 | 471 | 499 | 471 | 0.582120918 |
| entry-S | T2 / 16500 | 3 | 0 | 12 | 12 | 14.803139 |
| entry-S | T2 / 16500 | 4 | 0 | 12 | 6 | 14.162923 |
| best | T4 / 326500 | 1 | 512 | 512 | 512 | 0.000480891595 |
| best | T4 / 326500 | 2 | 512 | 512 | 512 | 0.00035268063 |
| best | T4 / 326500 | 3 | 512 | 512 | 512 | 0.00256518851 |
| best | T4 / 326500 | 4 | 309 | 505 | 309 | 1.04376662 |
| exit | T4 / 326500 | 1 | 512 | 512 | 512 | 0.000480891595 |
| exit | T4 / 326500 | 2 | 512 | 512 | 512 | 0.00035268063 |
| exit | T4 / 326500 | 3 | 512 | 512 | 512 | 0.00256518851 |
| exit | T4 / 326500 | 4 | 309 | 505 | 309 | 1.04376662 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 326500 | 1 | 512 | 512 | 512 | 0.000490426588 |
| entry-S | T4 / 326500 | 2 | 512 | 512 | 512 | 0.000394196119 |
| entry-S | T4 / 326500 | 3 | 512 | 512 | 512 | 0.00146381649 |
| entry-S | T4 / 326500 | 4 | 302 | 502 | 302 | 1.03886035 |
| entry-S | T4 / 326500 | 5 | 0 | 7 | 10 | 71.9906768 |
| entry-S | T4 / 326500 | 6 | 0 | 6 | 8 | 77.8426309 |
| entry-S | T4 / 326500 | 7 | 0 | 4 | 8 | 73.6119892 |
| entry-S | T4 / 326500 | 8 | 0 | 6 | 11 | 75.7198708 |
| best | T8 / 500000 | 1 | 504 | 512 | 504 | 0.0339728049 |
| best | T8 / 500000 | 2 | 510 | 512 | 510 | 0.0114814004 |
| best | T8 / 500000 | 3 | 510 | 511 | 510 | 0.014799009 |
| best | T8 / 500000 | 4 | 499 | 501 | 499 | 0.0590102207 |
| best | T8 / 500000 | 5 | 417 | 455 | 419 | 0.585765323 |
| best | T8 / 500000 | 6 | 0 | 10 | 9 | 8.23393676 |
| best | T8 / 500000 | 7 | 0 | 8 | 8 | 8.26925943 |
| best | T8 / 500000 | 8 | 0 | 11 | 9 | 8.24883855 |
| exit | T8 / 500000 | 1 | 504 | 512 | 504 | 0.0339728049 |
| exit | T8 / 500000 | 2 | 510 | 512 | 510 | 0.0114814004 |
| exit | T8 / 500000 | 3 | 510 | 511 | 510 | 0.014799009 |
| exit | T8 / 500000 | 4 | 499 | 501 | 499 | 0.0590102207 |
| exit | T8 / 500000 | 5 | 417 | 455 | 419 | 0.585765323 |
| exit | T8 / 500000 | 6 | 0 | 10 | 9 | 8.23393676 |
| exit | T8 / 500000 | 7 | 0 | 8 | 8 | 8.26925943 |
| exit | T8 / 500000 | 8 | 0 | 11 | 9 | 8.24883855 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 500000 | 1 | 509 | 512 | 509 | 0.0264375917 |
| entry-S | T8 / 500000 | 2 | 509 | 512 | 509 | 0.0190100885 |
| entry-S | T8 / 500000 | 3 | 511 | 512 | 511 | 0.00356821916 |
| entry-S | T8 / 500000 | 4 | 502 | 505 | 502 | 0.066734744 |
| entry-S | T8 / 500000 | 5 | 406 | 450 | 407 | 0.631186314 |
| entry-S | T8 / 500000 | 6 | 0 | 15 | 11 | 8.19208055 |
| entry-S | T8 / 500000 | 7 | 0 | 3 | 9 | 8.29643276 |
| entry-S | T8 / 500000 | 8 | 0 | 7 | 6 | 8.25469975 |
| entry-S | T8 / 500000 | 9 | 0 | 11 | 13 | 8.26514107 |
| entry-S | T8 / 500000 | 10 | 1 | 8 | 4 | 8.2391108 |
| entry-S | T8 / 500000 | 11 | 0 | 2 | 9 | 8.27396265 |
| entry-S | T8 / 500000 | 12 | 0 | 10 | 12 | 8.26725286 |
| entry-S | T8 / 500000 | 13 | 0 | 12 | 11 | 8.26938989 |
| entry-S | T8 / 500000 | 14 | 0 | 8 | 9 | 8.27735229 |
| entry-S | T8 / 500000 | 15 | 0 | 8 | 8 | 8.26030083 |
| entry-S | T8 / 500000 | 16 | 0 | 7 | 11 | 8.27312293 |
| best | T16 / 47000 | 1 | 510 | 512 | 510 | 0.0219238655 |
| best | T16 / 47000 | 2 | 509 | 512 | 509 | 0.0156421091 |
| best | T16 / 47000 | 3 | 512 | 512 | 512 | 0.00205324032 |
| best | T16 / 47000 | 4 | 497 | 502 | 497 | 0.116851505 |
| best | T16 / 47000 | 5 | 448 | 461 | 450 | 0.400779153 |
| best | T16 / 47000 | 6 | 0 | 12 | 12 | 8.78073894 |
| best | T16 / 47000 | 7 | 0 | 4 | 8 | 8.26782799 |
| best | T16 / 47000 | 8 | 0 | 8 | 10 | 8.22518507 |
| best | T16 / 47000 | 9 | 0 | 8 | 5 | 8.24427389 |
| best | T16 / 47000 | 10 | 0 | 8 | 11 | 8.23274599 |
| best | T16 / 47000 | 11 | 0 | 6 | 6 | 8.25079335 |
| best | T16 / 47000 | 12 | 0 | 3 | 8 | 8.23663988 |
| best | T16 / 47000 | 13 | 0 | 5 | 8 | 8.2513988 |
| best | T16 / 47000 | 14 | 0 | 9 | 7 | 8.25501674 |
| best | T16 / 47000 | 15 | 1 | 8 | 8 | 8.24161564 |
| best | T16 / 47000 | 16 | 0 | 6 | 8 | 8.25448763 |
| exit | T16 / 500000 | 1 | 462 | 496 | 464 | 0.444482794 |
| exit | T16 / 500000 | 2 | 495 | 504 | 495 | 0.118981629 |
| exit | T16 / 500000 | 3 | 484 | 511 | 484 | 0.203509133 |
| exit | T16 / 500000 | 4 | 443 | 504 | 443 | 0.567328454 |
| exit | T16 / 500000 | 5 | 409 | 475 | 410 | 0.887407506 |
| exit | T16 / 500000 | 6 | 0 | 10 | 8 | 10.3288563 |
| exit | T16 / 500000 | 7 | 0 | 8 | 6 | 8.22413962 |
| exit | T16 / 500000 | 8 | 0 | 7 | 9 | 8.22527584 |
| exit | T16 / 500000 | 9 | 0 | 8 | 10 | 8.22286703 |
| exit | T16 / 500000 | 10 | 1 | 10 | 6 | 8.2249953 |
| exit | T16 / 500000 | 11 | 0 | 8 | 7 | 8.22148682 |
| exit | T16 / 500000 | 12 | 0 | 6 | 7 | 8.22559476 |
| exit | T16 / 500000 | 13 | 0 | 9 | 6 | 8.22424841 |
| exit | T16 / 500000 | 14 | 0 | 8 | 4 | 8.22254777 |
| exit | T16 / 500000 | 15 | 0 | 9 | 10 | 8.22361735 |
| exit | T16 / 500000 | 16 | 0 | 7 | 7 | 8.22437979 |

### A_gdn2_seed2


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 13500 | 1 | 474 | 511 | 474 | 0.297243677 |
| exit | T1 / 13500 | 1 | 474 | 511 | 474 | 0.297243677 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 13500 | 1 | 488 | 511 | 488 | 0.226770477 |
| entry-S | T1 / 13500 | 2 | 1 | 12 | 6 | 19.0009638 |
| best | T2 / 16500 | 1 | 512 | 512 | 512 | 0.00101577381 |
| best | T2 / 16500 | 2 | 414 | 430 | 415 | 0.852972452 |
| exit | T2 / 16500 | 1 | 512 | 512 | 512 | 0.00101577381 |
| exit | T2 / 16500 | 2 | 414 | 430 | 415 | 0.852972452 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 16500 | 1 | 512 | 512 | 512 | 0.000821791595 |
| entry-S | T2 / 16500 | 2 | 413 | 430 | 416 | 0.920639943 |
| entry-S | T2 / 16500 | 3 | 0 | 13 | 6 | 16.7652778 |
| entry-S | T2 / 16500 | 4 | 0 | 6 | 8 | 17.8222813 |
| best | T4 / 496000 | 1 | 512 | 512 | 512 | 0.000467809825 |
| best | T4 / 496000 | 2 | 512 | 512 | 512 | 0.000178433651 |
| best | T4 / 496000 | 3 | 512 | 512 | 512 | 0.0031081411 |
| best | T4 / 496000 | 4 | 2 | 22 | 17 | 11.2071558 |
| exit | T4 / 500000 | 1 | 512 | 512 | 512 | 0.000436558797 |
| exit | T4 / 500000 | 2 | 512 | 512 | 512 | 0.00016071234 |
| exit | T4 / 500000 | 3 | 512 | 512 | 512 | 0.00285346824 |
| exit | T4 / 500000 | 4 | 0 | 27 | 15 | 11.2064728 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 496000 | 1 | 512 | 512 | 512 | 0.000288265431 |
| entry-S | T4 / 496000 | 2 | 512 | 512 | 512 | 0.000436020596 |
| entry-S | T4 / 496000 | 3 | 511 | 511 | 511 | 0.00719831497 |
| entry-S | T4 / 496000 | 4 | 0 | 18 | 5 | 11.2037895 |
| entry-S | T4 / 496000 | 5 | 0 | 7 | 8 | 29.6119592 |
| entry-S | T4 / 496000 | 6 | 2 | 12 | 11 | 30.9743877 |
| entry-S | T4 / 496000 | 7 | 0 | 5 | 3 | 36.6646659 |
| entry-S | T4 / 496000 | 8 | 0 | 6 | 5 | 30.3105304 |
| best | T8 / 482500 | 1 | 511 | 511 | 512 | 0.00558996564 |
| best | T8 / 482500 | 2 | 512 | 512 | 512 | 0.00319488191 |
| best | T8 / 482500 | 3 | 498 | 512 | 498 | 0.109963548 |
| best | T8 / 482500 | 4 | 219 | 429 | 220 | 3.83504456 |
| best | T8 / 482500 | 5 | 0 | 7 | 9 | 28.4767032 |
| best | T8 / 482500 | 6 | 0 | 9 | 7 | 8.22693596 |
| best | T8 / 482500 | 7 | 0 | 9 | 17 | 8.22283737 |
| best | T8 / 482500 | 8 | 0 | 7 | 2 | 8.23810158 |
| exit | T8 / 500000 | 1 | 511 | 512 | 511 | 0.00438196062 |
| exit | T8 / 500000 | 2 | 511 | 512 | 511 | 0.0085263475 |
| exit | T8 / 500000 | 3 | 495 | 512 | 495 | 0.117447909 |
| exit | T8 / 500000 | 4 | 214 | 421 | 214 | 3.90879349 |
| exit | T8 / 500000 | 5 | 0 | 11 | 9 | 31.3439925 |
| exit | T8 / 500000 | 6 | 0 | 12 | 8 | 8.22675668 |
| exit | T8 / 500000 | 7 | 0 | 7 | 9 | 8.22292979 |
| exit | T8 / 500000 | 8 | 0 | 9 | 10 | 8.23647938 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 482500 | 1 | 509 | 511 | 510 | 0.023802829 |
| entry-S | T8 / 482500 | 2 | 511 | 511 | 511 | 0.00389156386 |
| entry-S | T8 / 482500 | 3 | 502 | 512 | 502 | 0.0694188328 |
| entry-S | T8 / 482500 | 4 | 206 | 408 | 208 | 3.98294611 |
| entry-S | T8 / 482500 | 5 | 0 | 17 | 9 | 29.9234674 |
| entry-S | T8 / 482500 | 6 | 0 | 11 | 5 | 8.23505957 |
| entry-S | T8 / 482500 | 7 | 0 | 14 | 6 | 8.23362052 |
| entry-S | T8 / 482500 | 8 | 0 | 14 | 6 | 8.23275099 |
| entry-S | T8 / 482500 | 9 | 0 | 8 | 12 | 8.23603089 |
| entry-S | T8 / 482500 | 10 | 0 | 9 | 8 | 8.23123715 |
| entry-S | T8 / 482500 | 11 | 0 | 9 | 5 | 8.22931409 |
| entry-S | T8 / 482500 | 12 | 0 | 11 | 5 | 8.22602816 |
| entry-S | T8 / 482500 | 13 | 0 | 10 | 8 | 8.22316103 |
| entry-S | T8 / 482500 | 14 | 0 | 5 | 7 | 8.22392655 |
| entry-S | T8 / 482500 | 15 | 0 | 6 | 12 | 8.2287209 |
| entry-S | T8 / 482500 | 16 | 0 | 6 | 12 | 8.22407032 |
| best | T16 / 360500 | 1 | 463 | 492 | 473 | 0.351679088 |
| best | T16 / 360500 | 2 | 501 | 507 | 505 | 0.0590062091 |
| best | T16 / 360500 | 3 | 487 | 509 | 488 | 0.184388284 |
| best | T16 / 360500 | 4 | 311 | 386 | 312 | 1.79651279 |
| best | T16 / 360500 | 5 | 0 | 9 | 11 | 38.3390083 |
| best | T16 / 360500 | 6 | 0 | 5 | 6 | 8.34507068 |
| best | T16 / 360500 | 7 | 2 | 12 | 8 | 8.29421366 |
| best | T16 / 360500 | 8 | 0 | 9 | 14 | 8.30843518 |
| best | T16 / 360500 | 9 | 0 | 9 | 7 | 8.2900296 |
| best | T16 / 360500 | 10 | 1 | 8 | 11 | 8.29010478 |
| best | T16 / 360500 | 11 | 0 | 7 | 7 | 8.31714185 |
| best | T16 / 360500 | 12 | 1 | 4 | 8 | 8.2869406 |
| best | T16 / 360500 | 13 | 0 | 11 | 12 | 8.30283905 |
| best | T16 / 360500 | 14 | 0 | 5 | 9 | 8.31701569 |
| best | T16 / 360500 | 15 | 0 | 5 | 9 | 8.30810382 |
| best | T16 / 360500 | 16 | 0 | 9 | 7 | 8.30072908 |
| exit | T16 / 500000 | 1 | 405 | 447 | 433 | 1.03339381 |
| exit | T16 / 500000 | 2 | 478 | 491 | 494 | 0.353607487 |
| exit | T16 / 500000 | 3 | 441 | 491 | 442 | 0.474754365 |
| exit | T16 / 500000 | 4 | 317 | 383 | 317 | 1.60714768 |
| exit | T16 / 500000 | 5 | 0 | 8 | 5 | 32.9020539 |
| exit | T16 / 500000 | 6 | 0 | 5 | 8 | 8.32222402 |
| exit | T16 / 500000 | 7 | 0 | 12 | 7 | 8.28478419 |
| exit | T16 / 500000 | 8 | 0 | 9 | 8 | 8.30331219 |
| exit | T16 / 500000 | 9 | 1 | 9 | 10 | 8.29190089 |
| exit | T16 / 500000 | 10 | 0 | 8 | 9 | 8.30655791 |
| exit | T16 / 500000 | 11 | 0 | 7 | 5 | 8.34395026 |
| exit | T16 / 500000 | 12 | 0 | 4 | 7 | 8.32486022 |
| exit | T16 / 500000 | 13 | 0 | 11 | 6 | 8.3002742 |
| exit | T16 / 500000 | 14 | 0 | 5 | 8 | 8.33214927 |
| exit | T16 / 500000 | 15 | 0 | 5 | 7 | 8.29778789 |
| exit | T16 / 500000 | 16 | 0 | 9 | 5 | 8.31932124 |

### B_e97_seed0


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 49000 | 1 | 461 | 512 | 461 | 0.420747646 |
| exit | T1 / 49000 | 1 | 461 | 512 | 461 | 0.420747646 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 49000 | 1 | 463 | 512 | 463 | 0.40342167 |
| entry-S | T1 / 49000 | 2 | 0 | 8 | 9 | 22.7899949 |
| best | T2 / 448000 | 1 | 512 | 512 | 512 | 7.4813509e-05 |
| best | T2 / 448000 | 2 | 414 | 511 | 414 | 0.657237348 |
| exit | T2 / 448000 | 1 | 512 | 512 | 512 | 7.4813509e-05 |
| exit | T2 / 448000 | 2 | 414 | 511 | 414 | 0.657237348 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 448000 | 1 | 512 | 512 | 512 | 0.000204717308 |
| entry-S | T2 / 448000 | 2 | 425 | 508 | 425 | 0.652532505 |
| entry-S | T2 / 448000 | 3 | 1 | 14 | 7 | 69.7625527 |
| entry-S | T2 / 448000 | 4 | 0 | 7 | 6 | 65.1793847 |
| best | T4 / 326500 | 1 | 512 | 512 | 512 | 0.00020587154 |
| best | T4 / 326500 | 2 | 511 | 511 | 511 | 0.00744830767 |
| best | T4 / 326500 | 3 | 8 | 277 | 14 | 5.52500634 |
| best | T4 / 326500 | 4 | 0 | 11 | 9 | 8.24732428 |
| exit | T4 / 500000 | 1 | 493 | 493 | 496 | 0.176519555 |
| exit | T4 / 500000 | 2 | 508 | 510 | 508 | 0.0564175313 |
| exit | T4 / 500000 | 3 | 1 | 44 | 10 | 8.70435571 |
| exit | T4 / 500000 | 4 | 0 | 4 | 7 | 8.22613703 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 326500 | 1 | 512 | 512 | 512 | 0.000138554997 |
| entry-S | T4 / 326500 | 2 | 512 | 512 | 512 | 0.00110382563 |
| entry-S | T4 / 326500 | 3 | 4 | 231 | 9 | 5.62089284 |
| entry-S | T4 / 326500 | 4 | 0 | 7 | 5 | 8.28019445 |
| entry-S | T4 / 326500 | 5 | 0 | 7 | 5 | 8.32891227 |
| entry-S | T4 / 326500 | 6 | 0 | 8 | 12 | 8.24828517 |
| entry-S | T4 / 326500 | 7 | 0 | 4 | 11 | 8.25984751 |
| entry-S | T4 / 326500 | 8 | 1 | 6 | 14 | 8.2656198 |
| best | T8 / 295500 | 1 | 505 | 505 | 507 | 0.041082466 |
| best | T8 / 295500 | 2 | 511 | 511 | 511 | 0.00697604713 |
| best | T8 / 295500 | 3 | 459 | 504 | 459 | 0.607857439 |
| best | T8 / 295500 | 4 | 0 | 9 | 6 | 8.17662839 |
| best | T8 / 295500 | 5 | 0 | 7 | 8 | 8.24406344 |
| best | T8 / 295500 | 6 | 0 | 13 | 10 | 8.21986603 |
| best | T8 / 295500 | 7 | 0 | 10 | 8 | 8.23062019 |
| best | T8 / 295500 | 8 | 0 | 5 | 3 | 8.22046099 |
| exit | T8 / 500000 | 1 | 73 | 80 | 80 | 25.3599217 |
| exit | T8 / 500000 | 2 | 56 | 89 | 76 | 20.5009873 |
| exit | T8 / 500000 | 3 | 18 | 44 | 25 | 28.181158 |
| exit | T8 / 500000 | 4 | 0 | 7 | 10 | 13.6902557 |
| exit | T8 / 500000 | 5 | 0 | 10 | 7 | 8.22019969 |
| exit | T8 / 500000 | 6 | 0 | 9 | 4 | 8.2335547 |
| exit | T8 / 500000 | 7 | 2 | 17 | 8 | 8.22638794 |
| exit | T8 / 500000 | 8 | 0 | 8 | 7 | 8.22430355 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 295500 | 1 | 508 | 508 | 509 | 0.0189013035 |
| entry-S | T8 / 295500 | 2 | 511 | 511 | 511 | 0.00546855253 |
| entry-S | T8 / 295500 | 3 | 472 | 498 | 472 | 0.614072013 |
| entry-S | T8 / 295500 | 4 | 0 | 17 | 10 | 8.11978203 |
| entry-S | T8 / 295500 | 5 | 1 | 7 | 13 | 8.23892485 |
| entry-S | T8 / 295500 | 6 | 0 | 7 | 5 | 8.23309212 |
| entry-S | T8 / 295500 | 7 | 0 | 10 | 4 | 8.23457509 |
| entry-S | T8 / 295500 | 8 | 0 | 8 | 7 | 8.23783855 |
| entry-S | T8 / 295500 | 9 | 0 | 7 | 4 | 8.23601896 |
| entry-S | T8 / 295500 | 10 | 0 | 13 | 10 | 8.23619936 |
| entry-S | T8 / 295500 | 11 | 0 | 14 | 5 | 8.23209054 |
| entry-S | T8 / 295500 | 12 | 0 | 4 | 8 | 8.23741249 |
| entry-S | T8 / 295500 | 13 | 0 | 6 | 8 | 8.23041582 |
| entry-S | T8 / 295500 | 14 | 0 | 5 | 6 | 8.23552298 |
| entry-S | T8 / 295500 | 15 | 0 | 10 | 9 | 8.23446329 |
| entry-S | T8 / 295500 | 16 | 0 | 11 | 4 | 8.23314641 |
| best | T16 / 499500 | 1 | 400 | 401 | 405 | 4.98928981 |
| best | T16 / 499500 | 2 | 498 | 500 | 498 | 0.26266148 |
| best | T16 / 499500 | 3 | 343 | 412 | 346 | 3.41051445 |
| best | T16 / 499500 | 4 | 251 | 333 | 251 | 6.15118844 |
| best | T16 / 499500 | 5 | 0 | 8 | 6 | 8.20537321 |
| best | T16 / 499500 | 6 | 0 | 4 | 10 | 8.23013385 |
| best | T16 / 499500 | 7 | 0 | 15 | 7 | 8.21768804 |
| best | T16 / 499500 | 8 | 0 | 3 | 10 | 8.22919304 |
| best | T16 / 499500 | 9 | 0 | 7 | 12 | 8.23001241 |
| best | T16 / 499500 | 10 | 0 | 9 | 6 | 8.22967478 |
| best | T16 / 499500 | 11 | 0 | 13 | 12 | 8.22523765 |
| best | T16 / 499500 | 12 | 0 | 6 | 6 | 8.22715982 |
| best | T16 / 499500 | 13 | 0 | 4 | 13 | 8.22578096 |
| best | T16 / 499500 | 14 | 1 | 13 | 10 | 8.22498801 |
| best | T16 / 499500 | 15 | 1 | 7 | 11 | 8.22008582 |
| best | T16 / 499500 | 16 | 0 | 8 | 3 | 8.22870118 |
| exit | T16 / 500000 | 1 | 398 | 399 | 403 | 4.96478137 |
| exit | T16 / 500000 | 2 | 497 | 499 | 497 | 0.268851539 |
| exit | T16 / 500000 | 3 | 339 | 412 | 342 | 3.41023524 |
| exit | T16 / 500000 | 4 | 251 | 334 | 251 | 6.10876308 |
| exit | T16 / 500000 | 5 | 0 | 6 | 6 | 8.20536039 |
| exit | T16 / 500000 | 6 | 0 | 3 | 12 | 8.23034754 |
| exit | T16 / 500000 | 7 | 0 | 11 | 9 | 8.21767537 |
| exit | T16 / 500000 | 8 | 0 | 5 | 9 | 8.22890378 |
| exit | T16 / 500000 | 9 | 0 | 8 | 11 | 8.22997812 |
| exit | T16 / 500000 | 10 | 0 | 6 | 11 | 8.22988739 |
| exit | T16 / 500000 | 11 | 0 | 8 | 8 | 8.22538945 |
| exit | T16 / 500000 | 12 | 0 | 6 | 7 | 8.22730056 |
| exit | T16 / 500000 | 13 | 0 | 4 | 11 | 8.22595133 |
| exit | T16 / 500000 | 14 | 1 | 17 | 9 | 8.22501495 |
| exit | T16 / 500000 | 15 | 1 | 7 | 13 | 8.22015743 |
| exit | T16 / 500000 | 16 | 0 | 10 | 6 | 8.22882986 |

### B_gdn2_seed0


#### T1

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| best | T1 / 13000 | 1 | 465 | 505 | 465 | 0.391479025 |
| exit | T1 / 13000 | 1 | 465 | 505 | 465 | 0.391479025 |

#### T2

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T1 / 13000 | 1 | 469 | 509 | 469 | 0.388904663 |
| entry-S | T1 / 13000 | 2 | 0 | 7 | 5 | 19.6563696 |
| best | T2 / 19000 | 1 | 512 | 512 | 512 | 0.000942767694 |
| best | T2 / 19000 | 2 | 470 | 494 | 470 | 0.477249118 |
| exit | T2 / 19000 | 1 | 512 | 512 | 512 | 0.000942767694 |
| exit | T2 / 19000 | 2 | 470 | 494 | 470 | 0.477249118 |

#### T4

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T2 / 19000 | 1 | 512 | 512 | 512 | 0.000789902699 |
| entry-S | T2 / 19000 | 2 | 468 | 488 | 468 | 0.474288807 |
| entry-S | T2 / 19000 | 3 | 0 | 8 | 7 | 15.7710566 |
| entry-S | T2 / 19000 | 4 | 0 | 8 | 5 | 16.1015647 |
| best | T4 / 440500 | 1 | 512 | 512 | 512 | 2.93809722e-05 |
| best | T4 / 440500 | 2 | 512 | 512 | 512 | 0.000105461482 |
| best | T4 / 440500 | 3 | 512 | 512 | 512 | 0.000416662714 |
| best | T4 / 440500 | 4 | 313 | 509 | 313 | 1.13554458 |
| exit | T4 / 440500 | 1 | 512 | 512 | 512 | 2.93809722e-05 |
| exit | T4 / 440500 | 2 | 512 | 512 | 512 | 0.000105461482 |
| exit | T4 / 440500 | 3 | 512 | 512 | 512 | 0.000416662714 |
| exit | T4 / 440500 | 4 | 313 | 509 | 313 | 1.13554458 |

#### T8

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T4 / 440500 | 1 | 512 | 512 | 512 | 5.0843149e-05 |
| entry-S | T4 / 440500 | 2 | 512 | 512 | 512 | 9.17750674e-05 |
| entry-S | T4 / 440500 | 3 | 512 | 512 | 512 | 0.000272156335 |
| entry-S | T4 / 440500 | 4 | 303 | 508 | 303 | 1.13561089 |
| entry-S | T4 / 440500 | 5 | 0 | 5 | 4 | 83.992103 |
| entry-S | T4 / 440500 | 6 | 0 | 7 | 9 | 85.4592792 |
| entry-S | T4 / 440500 | 7 | 2 | 12 | 9 | 88.6436432 |
| entry-S | T4 / 440500 | 8 | 0 | 14 | 12 | 89.1036867 |
| best | T8 / 117000 | 1 | 512 | 512 | 512 | 1.52751773e-05 |
| best | T8 / 117000 | 2 | 512 | 512 | 512 | 8.66319623e-05 |
| best | T8 / 117000 | 3 | 512 | 512 | 512 | 7.28796083e-05 |
| best | T8 / 117000 | 4 | 507 | 512 | 507 | 0.0732304177 |
| best | T8 / 117000 | 5 | 1 | 12 | 7 | 9.04775919 |
| best | T8 / 117000 | 6 | 0 | 10 | 4 | 9.07831933 |
| best | T8 / 117000 | 7 | 0 | 11 | 9 | 9.04697485 |
| best | T8 / 117000 | 8 | 1 | 7 | 6 | 9.10198288 |
| exit | T8 / 500000 | 1 | 509 | 510 | 511 | 0.0138484653 |
| exit | T8 / 500000 | 2 | 511 | 511 | 511 | 0.00403853123 |
| exit | T8 / 500000 | 3 | 506 | 512 | 506 | 0.0345681095 |
| exit | T8 / 500000 | 4 | 150 | 405 | 153 | 5.2997812 |
| exit | T8 / 500000 | 5 | 0 | 12 | 5 | 8.22051243 |
| exit | T8 / 500000 | 6 | 0 | 10 | 2 | 8.22466758 |
| exit | T8 / 500000 | 7 | 0 | 12 | 9 | 8.22330661 |
| exit | T8 / 500000 | 8 | 0 | 10 | 4 | 8.22759429 |

#### T16

| Role | Checkpoint original depth / stage step | Driver | J /512 | X /512 | Y /512 | CE nats |
|---|---|---:|---:|---:|---:|---:|
| entry-S | T8 / 117000 | 1 | 512 | 512 | 512 | 2.1785557e-05 |
| entry-S | T8 / 117000 | 2 | 512 | 512 | 512 | 5.86673326e-05 |
| entry-S | T8 / 117000 | 3 | 512 | 512 | 512 | 9.69508192e-05 |
| entry-S | T8 / 117000 | 4 | 500 | 511 | 500 | 0.0874587243 |
| entry-S | T8 / 117000 | 5 | 0 | 8 | 8 | 9.10667585 |
| entry-S | T8 / 117000 | 6 | 0 | 6 | 6 | 9.15749662 |
| entry-S | T8 / 117000 | 7 | 0 | 9 | 7 | 9.09388188 |
| entry-S | T8 / 117000 | 8 | 0 | 8 | 3 | 9.06370109 |
| entry-S | T8 / 117000 | 9 | 0 | 10 | 10 | 9.11839715 |
| entry-S | T8 / 117000 | 10 | 0 | 11 | 5 | 9.07217321 |
| entry-S | T8 / 117000 | 11 | 0 | 9 | 6 | 9.12790838 |
| entry-S | T8 / 117000 | 12 | 1 | 6 | 12 | 9.18862651 |
| entry-S | T8 / 117000 | 13 | 0 | 7 | 5 | 9.11069296 |
| entry-S | T8 / 117000 | 14 | 1 | 8 | 5 | 9.07832594 |
| entry-S | T8 / 117000 | 15 | 0 | 8 | 10 | 9.10832826 |
| entry-S | T8 / 117000 | 16 | 0 | 9 | 7 | 9.0660363 |
| best | T16 / 10000 | 1 | 512 | 512 | 512 | 2.45479384e-05 |
| best | T16 / 10000 | 2 | 512 | 512 | 512 | 6.7958546e-05 |
| best | T16 / 10000 | 3 | 512 | 512 | 512 | 0.000120427766 |
| best | T16 / 10000 | 4 | 503 | 512 | 503 | 0.0879812312 |
| best | T16 / 10000 | 5 | 0 | 8 | 7 | 8.87572667 |
| best | T16 / 10000 | 6 | 0 | 6 | 6 | 8.95213518 |
| best | T16 / 10000 | 7 | 0 | 9 | 7 | 8.87491865 |
| best | T16 / 10000 | 8 | 0 | 8 | 3 | 8.85774576 |
| best | T16 / 10000 | 9 | 0 | 10 | 10 | 8.90322961 |
| best | T16 / 10000 | 10 | 0 | 11 | 5 | 8.84919068 |
| best | T16 / 10000 | 11 | 0 | 9 | 6 | 8.89882552 |
| best | T16 / 10000 | 12 | 1 | 6 | 12 | 8.95856263 |
| best | T16 / 10000 | 13 | 0 | 7 | 5 | 8.88971888 |
| best | T16 / 10000 | 14 | 1 | 8 | 5 | 8.86655547 |
| best | T16 / 10000 | 15 | 0 | 8 | 10 | 8.89556793 |
| best | T16 / 10000 | 16 | 0 | 9 | 7 | 8.85869157 |
| exit | T16 / 500000 | 1 | 512 | 512 | 512 | 0.00233577133 |
| exit | T16 / 500000 | 2 | 512 | 512 | 512 | 0.000232793062 |
| exit | T16 / 500000 | 3 | 512 | 512 | 512 | 0.000270993414 |
| exit | T16 / 500000 | 4 | 467 | 501 | 467 | 0.234157481 |
| exit | T16 / 500000 | 5 | 1 | 4 | 11 | 8.21452599 |
| exit | T16 / 500000 | 6 | 0 | 15 | 9 | 8.22551164 |
| exit | T16 / 500000 | 7 | 0 | 6 | 12 | 8.22445296 |
| exit | T16 / 500000 | 8 | 0 | 6 | 6 | 8.22595442 |
| exit | T16 / 500000 | 9 | 0 | 6 | 12 | 8.22227534 |
| exit | T16 / 500000 | 10 | 0 | 8 | 8 | 8.22230442 |
| exit | T16 / 500000 | 11 | 0 | 7 | 10 | 8.2214143 |
| exit | T16 / 500000 | 12 | 1 | 10 | 9 | 8.21994325 |
| exit | T16 / 500000 | 13 | 0 | 11 | 12 | 8.22295596 |
| exit | T16 / 500000 | 14 | 0 | 6 | 7 | 8.22400681 |
| exit | T16 / 500000 | 15 | 0 | 2 | 11 | 8.22045645 |
| exit | T16 / 500000 | 16 | 0 | 6 | 12 | 8.22145928 |
