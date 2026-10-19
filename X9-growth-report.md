# X9 — E97 regime-swap breakthrough (completed)

**Three perfect positions, without ladder or carry supervision.** Scratch E97 T16 seed0 reached joint-state **512/512 at P1, P2 and P3**, the first **unsupervised, ladder-free E97 composition result in this series**. Here “unsupervised” means no auxiliary carry/intermediate-state supervision: ordinary task-target cross entropy remains supervised. This establishes that the capability is **reachable**, not guaranteed: **one seed, one depth, one regime**, not a capacity theorem, robustness estimate or deep-T16 solution.

Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage9_anneal_8m/`. Registration/source commit `8e6d1393` preceded launch; [launch report](X9-launch-report.md). The run completed all **500,000 updates**, with **1,001 frozen-pool validations** every500 updates, stopping only at its safety ceiling. Terminal receipt and atomic DONE marker agree and bind the result hash. The original watcher recorded **DONE at2026-10-19T21:50:25Z**. Runtime was12,379.112 seconds. The reproducible read-only audit `C/receipts/growth/audit.py` and derived `audit.json` bind the exact counts, sources, LR stream, frozen pool and retained checkpoint hashes. Raw streams/checkpoints remain local, not silently swept into Git.

## Acquisition and the late fourth-position shoulder

All counts are **joint-state correct out of512**, not coordinate counts. Crossings are first observed on the500-update validation grid, not exact learning events or claims of monotonic retention. Fresh procedural training never sampled the frozen seed0 T16 pool, whose array hashes match X7.

| Position | First >=50/512 | First512/512 | LR at first512 | Maximum |
|---|---:|---:|---:|---:|
| P1 | 47,000 | **62,500** | 9.674031908e-4 | 512 |
| P2 | 198,000 | **245,500** | 5.269839152e-4 | 512 |
| P3 | 274,000 | **347,500** | 2.243053674e-4 | 512 |
| P4 | never | never | — | **18 at465,000** |

P4's18/512 shoulder coincides with cosine approaching its floor (**LR2.216225549e-5 at465K**), but remains **below the registered50/512 materiality bar**. P4 exits at10/512; do not call it a consolidated fourth position. The timing is descriptive, not an isolated causal effect of low LR. No P5+ reaches50; independent column maxima P5–P16 are`2,3,4,3,2,2,2,3,2,3,3,2`, not a single checkpoint.

| Observation | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | Correct/8192 | CE nats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X9 best465,000 | 512 | 512 | 512 | 18 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1557 | 6.428425401 |
| X9 exit500,000 | 512 | 512 | 512 | 10 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1549 | 6.427503824 |
| X7 SF-constant exit500,000 | 512 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 514 | 7.805131197 |

“Best” selects aggregate correct count, breaking ties by lower CE, **not minimum CE**. Strict perfect-prefix depth is3 at best and exit. Final terminal1/512 and suffix7-restricted P9–P16 total2/4096 do not establish deep tracking. The uniform marginal is ln3721=8.221747728 nats; chance-sized isolated counts are not composition. The suffix-only theorem concerns observers missing at least two iid drivers, not recurrent models allowed prefix information.

## Registered verdict and unresolved attribution

**Prediction(a): CONFIRMED.** P2 and P3 both reach512, far above the registered >=50 materiality threshold against X7's`512,2,0,...`. This is more than small-P2 consolidation-only evidence: the tested regime reaches a three-position composition circuit without the X6 ladder. Prediction(b)'s flat-run antecedent did not occur; its negative extrapolation is not invoked. The result reverses a universalized reading of earlier E97 failures, not their actual observations.

**Mandatory regime-bundle caveat:** optimizer **class AND schedule changed together**, also betas. X7 used AdamWScheduleFree constant1e-3, betas(.9,.95); X9 used ordinary AdamW, betas(.9,.999), explicit warmup/cosine. Nominal wd.01 is not effective decay equivalence (SF decay-at-y versus AdamW decoupled shrink). X9 alone cannot attribute the unlock to schedule, SF removal, beta change, or their interaction. No averaging/anchor mechanism was isolated or measured.

The actual LR was logged and audited for **every update**: step1=2e-7; linear step/5000 through5000=1e-3; cosine thereafter through500000=**1e-5**. Otherwise X7 bf16 recipe remained:9,900,800 parameters, batch32, uniform target CE, clip1, T16 scratch, fp32 parameters/bf16 autocast, legacy recurrent policy and sequential Triton evaluation. No ladder, carry supervision, stagnation stop, restart or threshold stop. Entry, aggregate-best and exit checkpoint hashes were verified under the existing bounded-retention policy; no historical files were pruned by this report.

**Next approved decomposition:** X9b retains X7 ScheduleFree averaging/betas and applies X9's explicit schedule with internal warmup disabled. Its prelaunch predictions and unresolved-beta caveat belong in its own registration/launch report; this report does not claim that cell's result or launch AdamW+constant.

## Reproducibility anchors and scope

`C/receipts/growth/audit.json` contains full best/exit position metrics, LR samples, first-crossing arrays, X7 final comparison and retained checkpoint identities. Re-run from repository root: `.venv/bin/python -m experiments.expressivity_tasks.results_invertible_stage9_anneal_8m.receipts.growth.audit`. It checks committed operational/config/FLA source hashes, marker/terminal/result identity, all500K finite consecutive scheduled training rows, all1,001 validations and position totals, same frozen arrays as X7, and retained checkpoint file hashes. It does **not** re-evaluate checkpoint predictions, test other seeds, or claim independent held-out/test selection.

| Artifact | SHA256 |
|---|---|
| result.json | `a1a97da6e76bc8648c4e1a49cb74fb7a3bc85918dee67b360631fbf6cc5e7b6b` |
| validation.json | `e00d80229215cb12b9a9f1f4919eacba5eb569512389c1e5c8d1e003c82047df` |
| training.jsonl | `7811bf65a36a8a45a6a578103645cadbbfde72e3f9de54d0f01b8688910b6c95` |

Local single-card expressivity only. Architecture authority `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` / gap matrix **R16 exact-source/evidence discipline by analogy only**: no production/resilient DiLoCo, Slurm Partition/QOS or systems-scale acceptance claim. Independent reviewer gate remains supervisor-owned.
