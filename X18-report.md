# X18 — fresh-data generalization check

**Registered outcome (a), EXECUTING: supported for all four receipted champions.** Every previously perfect position scores at least **510/512** on fresh data (registered retention threshold **480/512**); none collapses below300. X10 and X11 retain four exact-perfect positions each; X13-seed2 retains two. X12 retains four exact-perfect positions, while its formerly perfect fifth is **510/512**, not literally perfect on fresh data. Thus its frozen five-perfect result generalizes at **99.609375%** in the fifth position, not a claim of zero error on the distribution.

The frozen-pool/full-sequence-memorization worry is not supported: these are new full input sequences, with zero overlap against each champion's actual training stream or original frozen pool. This supports executing the learned task on the **same procedural distribution**, not an identified circuit or mathematical proof of execution. Short prefixes/local state-driver combinations may have been seen in training; finite-domain lookup mechanisms are not excluded. This is **not out-of-distribution extrapolation**, a universal depth bound, or training-seed robustness evidence.

## Frozen versus fresh: decisive columns

All counts are **joint full-state correct/512**, not coordinate accuracy. P1–P6 are shown without concealing partial columns; the complete position table below includes every tail. The same fresh T16 pool is shared by X10/X11; the same fresh T64 pool by X12/X13. Comparisons use each champion's **own original frozen pool**, not a substituted final checkpoint or a different seed's pool.

| Receipted champion | Depth / update | Frozen P1–P6 | Fresh P1–P6 | Frozen exact-perfect prefix → fresh | Verdict |
|---|---|---|---|---|---|
| X10-adamw | T16 / 468,000 | `512,512,512,512,14,0` | `512,512,512,512,4,0` | 4 → 4 | (a), all formerly perfect ≥480 |
| X11-GDN2 | T16 / 482,500 | `512,512,512,512,333,3` | `512,512,512,512,324,4` | 4 → 4 | (a), all formerly perfect ≥480 |
| X12-GDN2 | T64 / 1,000,000 | `512,512,512,512,512,491` | `512,512,512,512,510,487` | 5 → 4 | (a), all formerly perfect ≥480 |
| X13-seed2 | T64 / 1,000,000 | `512,512,309,1,0,0` | `512,512,320,0,0,0` | 2 → 2 | (a), all formerly perfect ≥480 |

**Exact partial readings, not forced binary labels:** X10 P5 **14→4**; X11 P5 **333→324** (65.0390625%→63.28125%), P6 **3→4**; X12 P6 **491→487** (95.8984375%→95.1171875%); X13-seed2 P3 **309→320** (60.3515625%→62.5%). X10's partial fifth remains shallow; neither the small partial-column movements nor X12's two P5 errors constitute the registered memorization collapse. No champion suddenly solves its long tail.

| Champion | Frozen → fresh joint total | Aggregate CE nats, frozen → fresh | Terminal joint/512, frozen → fresh | Suffix7-restricted joint, frozen → fresh |
|---|---|---|---|---|
| X10-adamw | 2066 → 2052 / 8192 | 5.952114731 → 5.951430708 | 1 → 0 | 4 → 0 / 4096 |
| X11-GDN2 | 2384 → 2377 / 8192 | 5.642261356 → 5.637930274 | 0 → 0 | 0 → 1 / 4096 |
| X12-GDN2 | 3057 → 3055 / 32768 | 7.457879722 → 7.458621562 | 0 → 0 | 6 → 10 / 28672 |
| X13-seed2 | 1338 → 1352 / 32768 | 7.858765036 → 7.858317822 | 0 → 0 | 3 → 8 / 28672 |

The full-state uniform marginal is **1/3721 = 0.0268744961%**, CE **ln(3721)=8.221747728**. Isolated tail hits do not establish composition. The suffix-only bound applies to observers missing drivers, not a bound on models allowed the full prefix.

## Registration and selection authority

`R = experiments/expressivity_tasks/results_invertible_stage18_freshdata_8m/`.

- Checkpoint selection was resolved with the live supervisor **before any fresh generation**: X10's X13-named aggregate-best468000; X11's X15-probed aggregate-best482500 (**P5=333, not final318**); X12's receipted fixed-final1M; X13-seed2 fixed-final1M (**P3=309, not an independently maximized column**). No checkpoint search using fresh outcomes.
- Registration was written at **2026-10-21T20:11:03.261435Z**, SHA256 `ea0cf2b219ca8c86e5e9f62cb3b3598266e0490b73423e752e48add52bb414d3`, then **committed as9445d165 before generation**. Generation started **20:11:19.291759Z**, ended20:11:19.359050Z. CPU preflight used unrelated test seeds, not either registered experimental seed.
- Registered predictions: **(a)** all formerly perfect positions ≥480/512 supports execution/generalization; **(b)** any formerly perfect position <300/512 means fresh-data collapse and requires re-scoping composition claims; **(c)** all intermediate/partial readings reported exactly. The registration explicitly says **same p=61 uniform-driver distribution, NOT OOD extrapolation**. No post-result threshold, seed or checkpoint changes.
- Every selected original payload and frozen NPZ was SHA-verified before/after copying to `/home/erikg/emender-scratch/X18-freshdata/`. Only copied payloads were deserialized. Original model/data/kernel source bytes and registered dependency hashes were checked against historical training receipts. `registration.json` embeds the exact historical selected validation rows, pool identities, source/dependency hashes and original metadata hashes.

| Champion | Update | Selected checkpoint SHA256 |
|---|---:|---|
| X10-adamw | 468000 | `0501a81ff50e34017a973f400eec5975faaf5bd9d84df64d6fa84b51ed6c9fa7` |
| X11-GDN2 | 482500 | `49a0f96fe7890ca9bb8a860698641ad8407dca412a3b38a01d3ede996c011cd0` |
| X12-GDN2 | 1000000 | `2af398b68b31f9902cb0271492426cf731d4ff960114376180b9cf3a61184ac0` |
| X13-seed2 | 1000000 | `a18239b989a714c24eb1f0f51a79dcbf8e1cc3ca9a1889af76a8f81eb1ae2428` |

## Fresh generation and never-seen full-sequence evidence

The **actual X2b instrument** `run_invertible_stage2.fresh_training_batch` invokes the unchanged `InvertibleQuadraticTask(61).generate_batch(512,T+2,rng)`. Uniform initial coordinates are encoded in the first two input tokens; iid uniform field drivers follow. Targets never enter inputs: independently checked scalar `(x,y),c → (y,(x+y²+c) mod61)` and joint class `61*x+y` agree on all **40,960** fresh targets. T16/T64 denote drivers, hence18/66 real tokens and original right-padding to32/80.

Each pool has512 unique full sequences. Fresh/frozen full-sequence intersections are zero for every same-depth original pool. `generation.json` binds seeds, initial/final PCG64 states, dtype/shape, independent scalar checks, archive and all three array hashes; both fresh NPZs are included with this report's receipts. No re-generation/reseed after looking at outcomes.

| Fresh pool | New seed | Archive SHA256 |
|---|---:|---|
| T16 | 202610211816 | `fd1048f68b3da2bb43d3c6e83ff14a75867ce853780accefed59b4fdc329b90b` |
| T64 | 202610211864 | `9cd220bd22b876932a8b3614a6f03522c1e86398df88f3cf784f26d0d9512e77` |

**Not just a probability-of-collision argument:** CPU replay reconstructs the exact original batch32 initial-then-driver RNG input stream through each champion's selected update. Vectorized RNG replay is tested byte-for-byte against actual `generate_batch` calls at both depths/seeds, including terminal RNG state. In each real replay the final RNG state also equals the saved checkpoint's `numpy_rng`. A three-token exact prefilter is followed by **full input-byte equality**, not a hash-only or prefix-only novelty test. Targets need not be regenerated for input membership. These are no-gradient CPU input replays, not model training.

| Champion | Original training updates replayed | Full training sequences checked | Fresh full-sequence matches | Terminal RNG matches checkpoint |
|---|---:|---:|---:|---|
| X10-adamw | 468,000 | 14,976,000 | 0 | yes |
| X11-GDN2 | 482,500 | 15,440,000 | 0 | yes |
| X12-GDN2 | 1,000,000 | 32,000,000 | 0 | yes |
| X13-seed2 | 1,000,000 | 32,000,000 | 0 | yes |

Total **94,416,000** full-sequence checks (X10's seed0 stream is nested within X11's same-depth stream, so this is not a claim of94M independent unique training examples). Novelty is against these four champions' own training streams/frozen pools, not an exhaustive search of every historical experiment. Prefix/state-driver recurrence is expected and not claimed novel.

## Read-only execution and controls

- **lambda01**, Python3.12.3, Torch2.9.1+cu128/CUDA12.8; repository `.venv`. Sequential **co-resident GPU3**, RTX6000 Ada, UUID `GPU-e316c33c-fbf1-5e77-041e-36161c04652a`, explicitly approved X6D pattern. **No exclusive lease claimed**, no scheduler/Slurm/WorksGood service used. No resilient/DiLoCo behavior changed or tested; gap-matrix requirement IDs **not applicable**.
- Execution **2026-10-21T20:11:21.392379Z–20:11:57.483966Z**, **36.092s** including CPU replay/loading; summed frozen+fresh timed evaluation blocks **3.716s**. Before/after GPU snapshots, versions and co-residency are retained in `runtime.json`. Possible temporary training slowdown is disclosed, **not measured** against any counterfactual. X16/X17 and all original run directories were never written, stopped, resumed, pruned or trained by this task.
- Strict saved **ordinary live AdamW** state-dict load, four layers, exact parameter counts E97=9,900,800 / GDN2=9,887,056; **no ScheduleFree transform**, optimizer construction or update. model.eval/no_grad, batch32, fixed seed0, deterministic algorithms, TF32 off, CUBLAS`:4096:8`, fp32 parameters/bf16 autocast; original E97 legacy-sequential and GDN2 stock chunk forward, causal padding and slicing.
- **All four frozen control dicts reproduce exactly**, including every joint/coordinate/OOR count, position loss, aggregate canonical CE and prefix—not just trained-perfect positions. Frozen control is evaluated before fresh within each model. Eight total pool passes, **128 batches /4,096 sequence evaluations /163,840 scored targets**. Zero nonfinite logits; every OOR count zero.
- Model tensors were byte-equal before/after the passes; copied and original checkpoint payload hashes remain unchanged. Every per-champion JSON binds registration, generation and checkpoint hashes, frozen and fresh full metrics, zero updates, unchanged weights and timing. Position/aggregate, terminal/suffix counts and CE reductions reconcile.

## Validation and reproducibility

Completed commands, in order (local lambda01; do not rerun GPU evaluation merely for review):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_eval_invertible_stage18_freshdata.py tests/test_eval_invertible_stage6_positions.py tests/test_invertible_quadratic.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage18_freshdata register
# Registration/source/tests committed as9445d165 here, before the first experimental draw.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage18_freshdata generate
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage18_freshdata evaluate
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage18_freshdata audit
```

CPU preflight: **66 passed**, including14 new X18 cases: scalar targets, byte-exact original RNG replay/order across chunks/depths/seeds, collision detection versus shared prefixes, no unregistered draws, source mutation/reselection/regeneration/overlap guards and exception-safe depth scoping. Final broader CPU validation: **137 passed, 8 skipped** (GPU-only checks deliberately skipped with CUDA hidden); output is retained in `R/receipts/`. `manual_verification.json` checks the preregistration commit/time ordering, unchanged original metadata/payloads and all160 report rows against receipts; `SHA256SUMS` supplies the integrity check. No missing outcome is imputed and no failed GPU pass or reseed was discarded. One unrelated working-tree-wide `git diff --check` found pre-existing blank-at-EOF warnings in live X14/X16 reports; task-only/staged diff checks are the acceptance scope and those files were not edited/staged.

`R/audit.json` records **a_executing** and every retention count. `R/SHA256SUMS` binds report/source/tests and all task receipts/pools (excluding itself). Frozen original NPZs/checkpoint binaries remain local scratch/original artifacts, not pushed; their hashes and historical rows are preserved in registration. Fresh pools and all X18 result JSONs are pushed. Independent acceptance review remains parent-owned.

**Residual scope:** one fresh512 pool per depth; selected successful champions, not random training-seed sampling; same finite distribution and native lengths only; no longer-depth/OOD tests or mechanism intervention. Generalization of the high-accuracy learned prefix is demonstrated here, not unlimited composition, a unique arithmetic algorithm, absence of every lookup mechanism or perfect distribution-wide accuracy. No paper claim requires narrowing under registered(b), which did not occur.

## Complete frozen → fresh positional comparison

J/X/Y are counts out of512. X/Y are coordinates of the **same joint argmax**, not separately optimized coordinate predictions. CE is full-state loss in nats (not coordinate CE); JSON retains full precision. Every row, including chance-sized tails, is included; every OOR=0.

### X10-adamw / T16

| Driver | J frozen→fresh | X frozen→fresh | Y frozen→fresh | CE frozen→fresh |
|---|---:|---:|---:|---:|
| P1 | 512→512 | 512→512 | 512→512 | 7.25198529e-05→6.79321611e-05 |
| P2 | 512→512 | 512→512 | 512→512 | 0.00034969022→0.000320873639 |
| P3 | 512→512 | 512→512 | 512→512 | 0.00203622636→0.00238333514 |
| P4 | 512→512 | 512→512 | 512→512 | 0.0465751312→0.0508141305 |
| P5 | 14→4 | 14→6 | 23→12 | 4.74359508→4.73831389 |
| P6 | 0→0 | 6→7 | 9→8 | 8.22316125→8.21517738 |
| P7 | 0→0 | 8→10 | 9→9 | 8.22282815→8.21995896 |
| P8 | 0→0 | 12→3 | 8→12 | 8.22132887→8.22168961 |
| P9 | 1→0 | 12→4 | 7→8 | 8.22195911→8.22179862 |
| P10 | 0→0 | 9→10 | 7→9 | 8.22142692→8.22153922 |
| P11 | 0→0 | 14→6 | 10→4 | 8.22175654→8.22235051 |
| P12 | 1→0 | 7→10 | 10→7 | 8.22167408→8.22199134 |
| P13 | 1→0 | 7→9 | 8→8 | 8.22198901→8.2214374 |
| P14 | 0→0 | 6→13 | 13→8 | 8.22159401→8.22181199 |
| P15 | 0→0 | 8→7 | 5→8 | 8.22206224→8.22167492 |
| P16 | 1→0 | 11→8 | 15→9 | 8.22142687→8.22156142 |

### X11-GDN2 / T16

| Driver | J frozen→fresh | X frozen→fresh | Y frozen→fresh | CE frozen→fresh |
|---|---:|---:|---:|---:|
| P1 | 512→512 | 512→512 | 512→512 | 0.000234710471→0.000233093143 |
| P2 | 512→512 | 512→512 | 512→512 | 0.000294372912→0.000297840239 |
| P3 | 512→512 | 512→512 | 512→512 | 0.001164251→0.00127530575 |
| P4 | 512→512 | 512→512 | 512→512 | 0.0252289433→0.0246442539 |
| P5 | 333→324 | 333→324 | 338→331 | 1.22044894→1.22567505 |
| P6 | 3→4 | 6→13 | 16→13 | 6.8118422→6.73855394 |
| P7 | 0→0 | 8→9 | 6→10 | 8.22134707→8.22049134 |
| P8 | 0→0 | 11→8 | 9→11 | 8.22169866→8.22158237 |
| P9 | 0→0 | 6→11 | 4→6 | 8.22153803→8.22087809 |
| P10 | 0→0 | 4→7 | 7→14 | 8.22165594→8.22153166 |
| P11 | 0→0 | 5→8 | 9→11 | 8.22168854→8.22244144 |
| P12 | 0→0 | 10→9 | 7→3 | 8.22189267→8.22238417 |
| P13 | 0→0 | 7→11 | 8→5 | 8.22161631→8.22172842 |
| P14 | 0→1 | 13→19 | 14→6 | 8.22173229→8.2215525 |
| P15 | 0→0 | 5→4 | 16→3 | 8.22172763→8.22162761 |
| P16 | 0→0 | 7→5 | 12→6 | 8.2220721→8.22198647 |

### X12-GDN2 / T64

| Driver | J frozen→fresh | X frozen→fresh | Y frozen→fresh | CE frozen→fresh |
|---|---:|---:|---:|---:|
| P1 | 512→512 | 512→512 | 512→512 | 8.72372544e-05→9.23714754e-05 |
| P2 | 512→512 | 512→512 | 512→512 | 6.55400426e-05→6.15123768e-05 |
| P3 | 512→512 | 512→512 | 512→512 | 5.54005183e-05→5.77905689e-05 |
| P4 | 512→512 | 512→512 | 512→512 | 0.000119930538→0.000144721854 |
| P5 | 512→510 | 512→510 | 512→510 | 0.00358234367→0.00962542681 |
| P6 | 491→487 | 491→488 | 491→487 | 0.350182296→0.393819981 |
| P7 | 0→0 | 10→9 | 13→9 | 8.22152188→8.21994125 |
| P8 | 0→0 | 6→8 | 6→10 | 8.22334654→8.22219348 |
| P9 | 0→0 | 2→8 | 7→8 | 8.22627475→8.22559717 |
| P10 | 0→0 | 12→10 | 12→9 | 8.22041067→8.22359505 |
| P11 | 0→0 | 13→4 | 9→12 | 8.22377605→8.22411332 |
| P12 | 0→0 | 10→6 | 6→10 | 8.22556819→8.22023188 |
| P13 | 0→0 | 8→11 | 6→14 | 8.21918539→8.22266187 |
| P14 | 0→0 | 11→13 | 9→8 | 8.22405992→8.22250795 |
| P15 | 0→0 | 11→10 | 12→12 | 8.22730728→8.22096543 |
| P16 | 0→0 | 4→9 | 6→7 | 8.22050651→8.22589347 |
| P17 | 0→0 | 14→11 | 10→9 | 8.22373691→8.2239379 |
| P18 | 0→0 | 5→12 | 7→7 | 8.21882831→8.22491863 |
| P19 | 0→0 | 10→16 | 8→10 | 8.22636336→8.22187163 |
| P20 | 0→0 | 7→3 | 13→8 | 8.22678039→8.22501697 |
| P21 | 0→2 | 8→6 | 12→9 | 8.22138377→8.22286479 |
| P22 | 0→0 | 6→8 | 4→10 | 8.22488873→8.21944508 |
| P23 | 0→0 | 4→10 | 11→9 | 8.21977057→8.21983762 |
| P24 | 0→0 | 10→11 | 10→8 | 8.22565758→8.22391884 |
| P25 | 0→0 | 9→7 | 2→8 | 8.2276853→8.2258014 |
| P26 | 0→0 | 5→14 | 9→13 | 8.22214262→8.22240643 |
| P27 | 0→0 | 5→11 | 8→9 | 8.22408677→8.22090085 |
| P28 | 0→0 | 12→14 | 9→9 | 8.22284834→8.2216868 |
| P29 | 0→0 | 10→11 | 8→5 | 8.22437525→8.21763632 |
| P30 | 0→0 | 11→12 | 6→13 | 8.22383473→8.22496142 |
| P31 | 0→1 | 8→7 | 7→11 | 8.22277161→8.22140014 |
| P32 | 0→0 | 5→9 | 9→7 | 8.22661322→8.22348284 |
| P33 | 1→0 | 10→11 | 6→9 | 8.22810238→8.22276606 |
| P34 | 0→1 | 17→5 | 4→7 | 8.22266671→8.224159 |
| P35 | 0→0 | 4→2 | 4→9 | 8.2231789→8.22354649 |
| P36 | 0→0 | 10→7 | 10→10 | 8.22115834→8.22493012 |
| P37 | 0→1 | 9→8 | 5→8 | 8.22489804→8.22190875 |
| P38 | 0→0 | 12→8 | 18→6 | 8.22138276→8.22642672 |
| P39 | 0→1 | 10→8 | 6→9 | 8.22834338→8.22103782 |
| P40 | 0→0 | 11→14 | 8→9 | 8.22321268→8.2186091 |
| P41 | 0→0 | 6→8 | 11→10 | 8.22058337→8.22377166 |
| P42 | 1→1 | 7→8 | 10→11 | 8.22274282→8.22257794 |
| P43 | 0→0 | 13→10 | 8→12 | 8.22128346→8.22506809 |
| P44 | 0→0 | 6→8 | 3→4 | 8.22421718→8.22397697 |
| P45 | 0→0 | 7→10 | 16→10 | 8.22305963→8.22449723 |
| P46 | 0→0 | 8→7 | 10→9 | 8.22122062→8.2300343 |
| P47 | 0→0 | 6→8 | 8→5 | 8.22087572→8.22184267 |
| P48 | 1→0 | 5→3 | 18→7 | 8.22088458→8.22364052 |
| P49 | 0→0 | 8→10 | 7→11 | 8.22426659→8.21815918 |
| P50 | 0→0 | 4→7 | 9→11 | 8.22323489→8.22527376 |
| P51 | 0→0 | 12→8 | 8→9 | 8.22025537→8.22592749 |
| P52 | 0→0 | 9→12 | 11→7 | 8.22247048→8.22196773 |
| P53 | 0→0 | 6→11 | 4→7 | 8.22778492→8.22342968 |
| P54 | 0→1 | 10→13 | 6→11 | 8.2189094→8.2225448 |
| P55 | 0→0 | 11→6 | 8→12 | 8.22021952→8.22382732 |
| P56 | 0→0 | 4→9 | 5→8 | 8.22538919→8.22638838 |
| P57 | 0→0 | 9→2 | 9→7 | 8.22503751→8.22513488 |
| P58 | 2→1 | 12→7 | 9→8 | 8.22023122→8.2254474 |
| P59 | 0→0 | 8→14 | 10→4 | 8.22525367→8.22421884 |
| P60 | 0→0 | 8→8 | 5→11 | 8.22167199→8.22343175 |
| P61 | 1→0 | 13→7 | 11→12 | 8.22447518→8.22209293 |
| P62 | 0→0 | 10→10 | 11→11 | 8.22394788→8.22116243 |
| P63 | 0→1 | 7→12 | 8→5 | 8.22175455→8.22688079 |
| P64 | 0→0 | 2→11 | 13→6 | 8.22377532→8.22550824 |

### X13-seed2 / T64

| Driver | J frozen→fresh | X frozen→fresh | Y frozen→fresh | CE frozen→fresh |
|---|---:|---:|---:|---:|
| P1 | 512→512 | 512→512 | 512→512 | 0.00180224196→0.00191724896 |
| P2 | 512→512 | 512→512 | 512→512 | 0.00415155495→0.00412516284 |
| P3 | 309→320 | 310→321 | 314→324 | 1.42687054→1.40098799 |
| P4 | 1→0 | 9→12 | 11→4 | 8.22168002→8.22140064 |
| P5 | 0→0 | 15→8 | 9→11 | 8.22044843→8.2228306 |
| P6 | 0→0 | 12→7 | 10→7 | 8.22163092→8.22141599 |
| P7 | 1→0 | 9→8 | 7→8 | 8.22078983→8.22327182 |
| P8 | 0→0 | 6→7 | 12→7 | 8.22212922→8.22196509 |
| P9 | 0→0 | 8→5 | 10→9 | 8.22114019→8.22156353 |
| P10 | 0→0 | 4→11 | 14→9 | 8.22136873→8.22119484 |
| P11 | 0→0 | 7→11 | 9→9 | 8.22208522→8.22157402 |
| P12 | 0→1 | 3→10 | 10→7 | 8.22216399→8.22239168 |
| P13 | 0→0 | 6→10 | 9→5 | 8.22202255→8.22177023 |
| P14 | 1→0 | 12→4 | 6→7 | 8.22256361→8.22146576 |
| P15 | 0→0 | 8→5 | 8→7 | 8.22240118→8.22107114 |
| P16 | 0→0 | 11→6 | 7→10 | 8.22282011→8.22126457 |
| P17 | 0→1 | 8→12 | 8→10 | 8.22161828→8.22105408 |
| P18 | 0→0 | 10→11 | 6→2 | 8.22144689→8.22259749 |
| P19 | 0→0 | 14→4 | 8→7 | 8.22141492→8.22166038 |
| P20 | 0→0 | 8→10 | 9→10 | 8.22192875→8.2210904 |
| P21 | 0→0 | 6→8 | 8→9 | 8.22161129→8.22142667 |
| P22 | 0→1 | 9→6 | 7→14 | 8.22171804→8.22203106 |
| P23 | 0→0 | 12→7 | 8→8 | 8.22210676→8.22135098 |
| P24 | 0→0 | 5→10 | 8→6 | 8.22160112→8.22107334 |
| P25 | 0→0 | 13→10 | 8→6 | 8.22103092→8.22190618 |
| P26 | 0→0 | 12→9 | 5→9 | 8.22217189→8.22187681 |
| P27 | 0→0 | 5→7 | 5→9 | 8.22247641→8.22242181 |
| P28 | 1→0 | 8→10 | 12→12 | 8.22159953→8.22063346 |
| P29 | 0→0 | 5→10 | 5→7 | 8.22196664→8.22225483 |
| P30 | 0→0 | 7→8 | 6→7 | 8.22089785→8.22115971 |
| P31 | 0→0 | 5→10 | 10→10 | 8.22147865→8.22100521 |
| P32 | 0→1 | 5→9 | 13→5 | 8.2218046→8.22084956 |
| P33 | 0→0 | 5→16 | 7→6 | 8.22190916→8.22108854 |
| P34 | 0→0 | 6→12 | 10→8 | 8.22111238→8.22136094 |
| P35 | 0→0 | 8→11 | 8→9 | 8.22188496→8.22155388 |
| P36 | 0→0 | 4→14 | 8→3 | 8.22136122→8.22186054 |
| P37 | 0→0 | 13→6 | 8→8 | 8.22178372→8.22158565 |
| P38 | 0→0 | 9→9 | 10→8 | 8.22119094→8.22256717 |
| P39 | 0→0 | 11→6 | 9→8 | 8.22192561→8.22195932 |
| P40 | 1→0 | 8→6 | 7→7 | 8.22159504→8.22145479 |
| P41 | 0→1 | 8→14 | 10→15 | 8.22206514→8.2223883 |
| P42 | 0→0 | 7→6 | 8→12 | 8.22135146→8.22178105 |
| P43 | 0→1 | 5→6 | 6→13 | 8.22190772→8.22121422 |
| P44 | 0→1 | 13→14 | 8→8 | 8.22078744→8.22169097 |
| P45 | 0→0 | 11→10 | 7→12 | 8.22198358→8.22188704 |
| P46 | 0→0 | 8→14 | 4→6 | 8.22180616→8.22144748 |
| P47 | 0→0 | 8→9 | 8→6 | 8.22304326→8.22187735 |
| P48 | 0→0 | 11→9 | 9→10 | 8.22145772→8.22230205 |
| P49 | 0→0 | 16→4 | 10→6 | 8.22230465→8.2216176 |
| P50 | 0→0 | 8→6 | 5→9 | 8.2219399→8.22185923 |
| P51 | 0→0 | 9→8 | 9→6 | 8.22232244→8.22058783 |
| P52 | 0→0 | 5→8 | 7→6 | 8.22177632→8.22234345 |
| P53 | 0→0 | 6→8 | 5→11 | 8.2224723→8.22192134 |
| P54 | 0→1 | 2→4 | 7→7 | 8.2216618→8.22145929 |
| P55 | 0→0 | 5→14 | 6→9 | 8.22121994→8.22245042 |
| P56 | 0→0 | 9→7 | 6→7 | 8.22199056→8.22273044 |
| P57 | 0→0 | 9→7 | 4→13 | 8.22139284→8.2218095 |
| P58 | 0→0 | 7→4 | 11→11 | 8.22228565→8.22158687 |
| P59 | 0→0 | 9→5 | 11→10 | 8.22127532→8.22260863 |
| P60 | 0→0 | 9→7 | 10→9 | 8.2223324→8.22177266 |
| P61 | 0→0 | 6→4 | 3→12 | 8.22186196→8.22142449 |
| P62 | 0→0 | 3→4 | 9→6 | 8.22217704→8.2214976 |
| P63 | 0→0 | 3→8 | 14→8 | 8.22212023→8.22216784 |
| P64 | 0→0 | 7→7 | 5→4 | 8.22172211→8.22188552 |
