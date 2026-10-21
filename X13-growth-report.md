# X13 — completed cold-T64 seed replication: split verdict

**Seed1 stayed at the marginal; seed2 escaped.** Both completed the registered **1,000,000 updates**. Seed1's global maximum is **4/512**, with no position reaching the registered50/512 bar. Seed2 finishes **P1=P2=512/512, P3=309/512 (60.3515625%)**, with P3 maximum **321/512 (62.6953125%)**. Fixed-final and maximum simultaneous perfect prefixes are **0 versus2**. P1 is visible-position learning, **not composition**; seed2's P2/P3 establish material composition under the registered behavioral criterion, not an identified internal circuit.

**Lottery finding, scoped:** across historical X12 seed0 and X13 seeds1/2, cold-T64 E97 has **1/3 observed escapes**, not a reliable ignition recipe. “Can learn” is receipted by seed2; “reliably learns” is refuted by seeds0/1 for this finite recipe/ceiling. No architectural impossibility claim survives. The empirical fraction is not an estimated universal success probability or a claim that full T64 is solved.

## 1. Authority, clean audit and complete terminals

Evidence root **R/**: `experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/`. The [investigation record](X13-report.md) retains the ordered **STEP0 CLEAN** audit and launch provenance; the unchanged T16 E97 champion retained P1–P4=512 each at T64. That audit exonerated the tested data path, not every conceivable implementation detail, and supplied no mechanism diagnosis. The [prelaunch registration](experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/REGISTRATION.md) and [machine manifest](experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/registration.json) govern this replication.

Scratch E97 seeds1/2 reuse the exact X12 four-layer/dim256 model (9,900,800 parameters), p61/vocab3721, legacy bf16 recurrence with fp32 parameters/bf16 autocast, batch32,64 drivers/66 real tokens padded to80. Uniform masked CE, ordinary live-parameter AdamW, all-parameter decay.01, clip1,5K warmup to1e-3 then cosine to1e-5 at1M. No curriculum, carry supervision, champion initialization, score-selected seed/restart or early stop. Each seed uses its own frozen512-example historical validation pool, never training examples. Validation evidence is not held-out/test evidence.

Read-only native `status(R)` returns **`(0, 'DONE X13 seeds1/2 matched1M')`**. Both DONE markers equal successful terminal receipts and SHA256-bind complete results, `stop_reason=safety_ceiling`, final step1M. Each history has **2,001 validations** on the complete0..1M/500 grid and **1M finite consecutive training rows**, including exact registered LR. The archived completion watcher records **2026-10-21T17:15:44Z DONE X13 seeds1/2 matched1M**. No new training or checkpoint evaluation was performed for this report.

## 2. Split verdict, full-position counts and acquisition times

All counts are **joint full-state correct/512**, not coordinate accuracy. Fixed-final1M is primary; aggregate-best maximizes aggregate correct, then minimizes CE on ties. Independent positional maxima need not coexist at one checkpoint. Seed2's P3 maximum321 occurs at996,500, also its aggregate-best step; this is not a perfect third position.

| Seed / readout | Step | Perfect prefix | P1 / P2 / P3 | Aggregate correct/32768 | CE (nats) |
|---|---:|---:|---|---:|---:|
| seed1 fixed-final | 1,000,000 | 0 | 0 / 0 / 0 | 10 | 8.221747398 |
| seed1 aggregate-best | 214,000 | 0 | 0 / 0 / 1 | 20 | 8.221737027 |
| seed2 fixed-final | 1,000,000 | 2 | 512 / 512 / 309 | 1338 | 7.858765036 |
| seed2 aggregate-best | 996,500 | 2 | 512 / 512 / 321 | 1353 | 7.858886302 |

Seed1's P1 maximum is3; its global4 occurs at P7. No seed1 position reaches50. Seed2 passes **both** distinct registered criteria: marginal escape (ANY P1..P64>=50) and composition (ANY P2..P64>=50). P4+ never reaches50; their global maximum is4. “Dead” means no material acquisition by the ceiling, **not literally zero hits**. Seed1 final P64=1 and seed2 final P64=0; neither solves the64-step task.

### Sequential acquisition signature, not a mechanism claim

| Seed2 position | First >=50/512 | First512/512 | Final /512 | Maximum /512 |
|---|---:|---:|---:|---:|
| P1 (non-composition) | 523,000 | 540,000 | 512 | 512 |
| P2 (composition) | 661,500 | 726,500 | 512 | 512 |
| P3 (composition) | 840,500 | not reached by1M | 309 | 321 |

P1 consolidates **late, after half the budget**: first50 at523K and first perfect at540K. First perfection is not permanent consolidation: **110 later validations** have P1<512, last at733K; P1 stays perfect on the sampled grid from733,500 through1M. P2 first perfect726,500 likewise has **22 later subperfect validations**, last at783,500, and stays perfect from784K. The first simultaneous two-perfect prefix is726,500. The ordered P1→P2→P3 crossings are a **sequential acquisition signature**, not proof of a particular recurrent computation or causal training dependency.

First crossings are observed at500-update resolution, not exact learning-event timestamps. All absent seed1 crossings, seed2 P4+ crossings and seed2 P3 perfection are **right-censored at1M**; absence is not impossibility. Infrastructure failures would be censored, not scientific zeros; neither replication was infrastructure-censored.

### Full64-position fixed-final/maxima and first-crossing receipt

The complete table below is reproduced from the native [growth receipt](experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/GROWTH_REPORT.md). “—” means not reached by1M. Every first50/first512 pair is in updates; maxima are independent over all2,001 validations.

| Position | seed1 final | seed2 final | seed1 max | seed2 max | seed1 first50 / first512 | seed2 first50 / first512 |
|---|---:|---:|---:|---:|---|---|
| P1 | 0 | 512 | 3 | 512 | — / — | 523000 / 540000 |
| P2 | 0 | 512 | 2 | 512 | — / — | 661500 / 726500 |
| P3 | 0 | 309 | 2 | 321 | — / — | 840500 / — |
| P4 | 0 | 1 | 2 | 2 | — / — | — / — |
| P5 | 0 | 0 | 2 | 2 | — / — | — / — |
| P6 | 0 | 0 | 2 | 3 | — / — | — / — |
| P7 | 0 | 1 | 4 | 3 | — / — | — / — |
| P8 | 0 | 0 | 2 | 2 | — / — | — / — |
| P9 | 0 | 0 | 2 | 2 | — / — | — / — |
| P10 | 1 | 0 | 2 | 2 | — / — | — / — |
| P11 | 1 | 0 | 3 | 2 | — / — | — / — |
| P12 | 1 | 0 | 2 | 2 | — / — | — / — |
| P13 | 0 | 0 | 3 | 2 | — / — | — / — |
| P14 | 0 | 1 | 2 | 2 | — / — | — / — |
| P15 | 1 | 0 | 2 | 2 | — / — | — / — |
| P16 | 0 | 0 | 2 | 2 | — / — | — / — |
| P17 | 0 | 0 | 3 | 3 | — / — | — / — |
| P18 | 0 | 0 | 3 | 2 | — / — | — / — |
| P19 | 0 | 0 | 2 | 2 | — / — | — / — |
| P20 | 0 | 0 | 2 | 3 | — / — | — / — |
| P21 | 0 | 0 | 3 | 2 | — / — | — / — |
| P22 | 0 | 0 | 2 | 3 | — / — | — / — |
| P23 | 0 | 0 | 2 | 3 | — / — | — / — |
| P24 | 0 | 0 | 2 | 2 | — / — | — / — |
| P25 | 0 | 0 | 2 | 3 | — / — | — / — |
| P26 | 0 | 0 | 2 | 2 | — / — | — / — |
| P27 | 1 | 0 | 3 | 2 | — / — | — / — |
| P28 | 1 | 1 | 2 | 3 | — / — | — / — |
| P29 | 0 | 0 | 3 | 3 | — / — | — / — |
| P30 | 0 | 0 | 3 | 3 | — / — | — / — |
| P31 | 0 | 0 | 3 | 2 | — / — | — / — |
| P32 | 0 | 0 | 3 | 2 | — / — | — / — |
| P33 | 0 | 0 | 2 | 2 | — / — | — / — |
| P34 | 0 | 0 | 2 | 3 | — / — | — / — |
| P35 | 0 | 0 | 2 | 2 | — / — | — / — |
| P36 | 0 | 0 | 2 | 2 | — / — | — / — |
| P37 | 0 | 0 | 2 | 2 | — / — | — / — |
| P38 | 0 | 0 | 2 | 2 | — / — | — / — |
| P39 | 1 | 0 | 2 | 2 | — / — | — / — |
| P40 | 0 | 1 | 3 | 2 | — / — | — / — |
| P41 | 0 | 0 | 2 | 2 | — / — | — / — |
| P42 | 0 | 0 | 2 | 3 | — / — | — / — |
| P43 | 0 | 0 | 2 | 2 | — / — | — / — |
| P44 | 0 | 0 | 3 | 2 | — / — | — / — |
| P45 | 0 | 0 | 2 | 3 | — / — | — / — |
| P46 | 0 | 0 | 2 | 3 | — / — | — / — |
| P47 | 0 | 0 | 3 | 3 | — / — | — / — |
| P48 | 0 | 0 | 2 | 2 | — / — | — / — |
| P49 | 0 | 0 | 2 | 4 | — / — | — / — |
| P50 | 1 | 0 | 2 | 2 | — / — | — / — |
| P51 | 0 | 0 | 2 | 2 | — / — | — / — |
| P52 | 0 | 0 | 3 | 2 | — / — | — / — |
| P53 | 0 | 0 | 3 | 3 | — / — | — / — |
| P54 | 0 | 0 | 3 | 3 | — / — | — / — |
| P55 | 0 | 0 | 3 | 2 | — / — | — / — |
| P56 | 0 | 0 | 2 | 2 | — / — | — / — |
| P57 | 0 | 0 | 3 | 2 | — / — | — / — |
| P58 | 0 | 0 | 2 | 2 | — / — | — / — |
| P59 | 0 | 0 | 2 | 2 | — / — | — / — |
| P60 | 0 | 0 | 3 | 2 | — / — | — / — |
| P61 | 0 | 0 | 3 | 2 | — / — | — / — |
| P62 | 1 | 0 | 2 | 2 | — / — | — / — |
| P63 | 0 | 0 | 3 | 2 | — / — | — / — |
| P64 | 1 | 0 | 3 | 2 | — / — | — / — |

## 3. Loss trajectories: marginal basin versus late escape

The full-state uniform reference is **ln(3721)=8.221747728 nats**, accuracy **1/3721=0.0268744961%**, about0.1376 expected correct per512-example position. Seed1 exhibits the **exact-marginal signature numerically**, not literally exact uniform logits or identically constant loss: after50K its aggregate validation CE ranges **8.221666038–8.222116947**; final8.221747398 differs from ln3721 by about−0.000000330. Chance-sized repeated maxima3/4 are not evidence of acquisition, nor a formal significance test or proof of the model's internal guessing strategy.

These are frozen-validation **P1..P64 mean CE**, not training minibatch losses. Seed2 initially shares the marginal basin, then separates late; aggregate averaging dilutes its learned-prefix gain across64 positions.

| Update | Seed1 aggregate CE | Seed2 aggregate CE | Seed2 P1 / P2 / P3 correct/512 |
|---:|---:|---:|---|
| 0 | 251.298354149 | 248.919535637 | 0 / 0 / 0 |
| 500 | 27.829275131 | 27.863780379 | 0 / 0 / 0 |
| 5,000 | 8.298197746 | 8.286286771 | 0 / 0 / 1 |
| 50,000 | 8.221728683 | 8.221828401 | 0 / 0 / 0 |
| 250,000 | 8.221752644 | 8.221792459 | 0 / 0 / 0 |
| 500,000 | 8.221789956 | 8.160626531 | 11 / 0 / 0 |
| 523,000 | 8.221781731 | 8.143976688 | 52 / 0 / 0 |
| 540,000 | 8.221742094 | 8.093921363 | 512 / 0 / 1 |
| 661,500 | 8.221769154 | 8.013364494 | 509 / 58 / 0 |
| 726,500 | 8.221713841 | 7.964988023 | 512 / 512 / 0 |
| 840,500 | 8.221743107 | 7.893392205 | 512 / 512 / 58 |
| 900,000 | 8.221743107 | 7.871473968 | 512 / 512 / 165 |
| 950,000 | 8.221747220 | 7.862580299 | 512 / 512 / 259 |
| 975,000 | 8.221751332 | 7.860568762 | 512 / 512 / 290 |
| 1,000,000 | 8.221747398 | 7.858765036 | 512 / 512 / 309 |

Seed2's P1 CE falls to4.303248464 at500K,3.236315036 at first50, then0.035446468 at first perfection; final P1/P2/P3 CE is **0.001802242 / 0.004151555 / 1.426870537**. Its P3 count improves290→309 over the last25K but fluctuates (maximum321 at996.5K): incomplete acquisition, not a projected time-to512 or a proven plateau. Fresh-batch training loss goes **251.049468994→8.221766472** for seed1 and **248.791015625→7.858142376** for seed2. These trajectories do not identify why one seed escapes.

## 4. Revise the threshold story to a probability landscape

[X12's seed0 observation](X12-growth-report.md) remains valid, but “E97 cannot ignite at T64” and any architectural impossibility reading are **retracted by seed2**. Seeds0/1 remain learned-nothing outcomes under the same finite recipe; seed2 demonstrates reachable composition, not reliable or full-horizon learning. The registered ANY-replication-escape ruling is met; the alternative “3/3 zero” is refuted. Historical seed0 was not a third newly run X13 arm.

The [X14 growth report](X14-growth-report.md)'s descriptive threshold bracket **(1/64,1/16]** is **REVISED**:1/64 is not a deterministic failure point. Think **ignition probability landscape**, not a hard coefficient threshold:

| Nominal P1 objective share | Observed E97 evidence | Licensed conclusion |
|---|---|---|
| 0 | X14 T16 seeds0/1, no position>=50 by500K | 0/2 observed ignition; not proof of zero probability |
| 1/64 | X12/X13 cold T64 seeds0/1/2,1M | 1/3 observed escape; two-perfect + partial P3 on seed2; ignition lottery |
| 1/16 | X9/X10 T16 working AdamW recipes, seed0 | Repeated recipe-level ignition/composition, **one independent seed**, not deterministic probability |

**Supervisor correction/concession #8:** the task's “T16 two-seed receipts” claim was incorrect. X9 and X10 AdamW are independent trainings/code paths but **both seed0**; recipe replication is not seed replication. X16's teach phase is also registered seed0 and supplies no second independent seed. Two-independent-seed T16 AdamW evidence is unavailable; **no deterministic-probability claim near1/16 is licensed**. A T16 independent-seed replication is recorded as a **registered follow-up commitment**, not an executable preregistration or authorization to launch; its exact arm must be preregistered before execution.

These are nominal objective coefficients, **not measured gradient shares**. Changing T16→T64 also changes horizon, pool, budget and cosine denominator; X9 additionally used its own AdamW recipe. X14's zero-share result is T16, not a matched zero-share T64 experiment. Thus the landscape is a hypothesis-organizing view, **not a measured causal success-probability curve**, a proof of monotonicity, or an estimate from enough seeds to distinguish architectures reliably.

## 5. Registered fixes: motivation, not claimed success

The split explains why a fix must make ignition **reliable**, not merely demonstrate a possible escape. These existing registrations target that goal; all are seed0-only, so none alone can establish deterministic success across seeds:

- **[X16 explicit-share sweep](experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/REGISTRATION.md):** fixed T16, E97 w=1/64 and1/32, GDN2 w=1/512 and1/2048,500K. Varying explicit P1 weight while holding within-architecture recipe/seed fixed isolates dose better than the historical horizon comparison. It maps ignition versus share at one seed; **a share-probability curve requires subsequent independent-seed replication**. No assumed monotonicity; cold T64 is context, not a T16 sweep point.
- **X16 teach-short-then-extend curriculum:** teach T16 to the preregistered >=4 positions>=508 criterion, then continue the same model/AdamW moments/RNG streams through500K T64 updates. This tests bypassing cold ignition; inherited zero-shot prefix retention must be separated from **new** T64 material acquisition. Criterion-not-reached censors extension, not a T64 scientific failure.
- **[X17 carry-wean](experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/REGISTRATION.md):** scratch T64 with true previous-state slots before P2..P64, one global p=0.5→0 wean across1M; evaluation always has neutral slots. Test whether temporary carry scaffolding transfers to a no-crutch solution. Early material acquisition while training feedback is present is not final p=0 survival.

No terminal fix outcome is asserted here; partial zeros in those live experiments remain right-censored. This report motivates the interventions but launches nothing, changes no recipe and authorizes no restart or checkpoint deletion.

## 6. GDN2 symmetric replication remains OPEN; standing limits

GDN2's cold-T64 success in X12 (**P1..P5 perfect, P6=491/512 final**) is **one seed0**, not demonstrated reliable ignition. E97's split makes **matched GDN2 seeds1/2 under the exact X12 recipe/1M ceiling** an explicit symmetric-replication open item, requiring its own prelaunch registration. Do not compare E97's three-seed empirical fraction to an imagined deterministic GDN2 rate. The observed GDN2 deeper extension remains receipted; its cross-seed reliability, numerical ignition-probability advantage and mechanism remain unresolved.

Standing [interpretation constraints](experiments/expressivity_tasks/INVERTIBLE_QUADRATIC.md#interpretation-constraints) bind this report. Every finite-state task admits a sufficiently large one-hot linear-state permutation representation. No impossibility, architectural capacity bound, universal expressivity separation, architecture ranking or production capability follows. Invertibility removes task-state merging, not every shortcut or optimization confound. Task success does not prove the intended internal recurrence. The suffix-only uniformity theorem concerns observers missing at least two iid drivers, **not prefix-retaining recurrent models**; chance suffix counts or prefix aggregate credit do not establish deep tracking. No causal claim about tanh, linear state, overwrite, gradient dominance, layers, optimizer or a discovered circuit is made. Three E97 seeds, one GDN2 seed, one task family and finite ceilings are the scope.

## 7. Reproduction and validation

Read-only audit on **lambda01**: `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x13_growth.py` — **passed**. Checked native hash-bound DONE/terminal/results, complete validation grids, positional denominators/finite CE, aggregate reconciliation, fixed-final/aggregate-best/prefix identities, both1M consecutive finite training loss/gradient/LR streams against the exact schedule, and regenerated native table equality without writing it. Temporary audit only, not a committed test dependency. Initial extraction incorrectly expected `result.best.aggregate`; the receipt stores best aggregate fields directly; corrected extraction passed. No model evaluation or training was run.

CPU regression: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_invertible_stage13_zerocheck.py -k 'not gpu'` — **11 passed (30.45s)**. No tests added/changed for this documentation-only task. A separate read-only check reconciled the printed64 positional rows, summary/acquisition/loss tables and local Markdown links with the receipts. `git diff --check` is included in report commit validation.

Raw streams/checkpoints remain **local/untracked**, not published by this two-document commit; the generated `R/GROWTH_REPORT.md` is left untouched. Local evidence SHA256 identities:

| Artifact under R/NAME | seed1 | seed2 |
|---|---|---|
| result.json | `00f0411fb6efc064cbd13403ac65f554ba8fd994f1037d8d81aebd514021277a` | `169ff77c2e4e0b4cb61e4fd9ddb3dd55511687b04032694cf12576824c20afdc` |
| validation.json | `dabbb43304ede52674193919cef6d59a74e27a9711d148bc8c64de385d76b633` | `d43c029c914c90070675b8faa3ec1d2785031d1eb304781d7f3570ee5ab2e548` |
| training.jsonl | `2f0cfb88496c69f70f0647b528a61e416eb880b3a7af434f1165a4386bfc1e72` | `9cf671cc8c87cc82c9c55e972a0e66ec5cf6f629193535bbe6368868d9237cfb` |

Minimal read-only reconstruction (lambda01; on Frontier first source the canonical environment and use `$EMENDER_PYTHON`):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from scripts.watch_invertible_stage13_zerocheck import DEFAULT_ROOT as r, NAMES, status
assert status(r) == (0, 'DONE X13 seeds1/2 matched1M')
for name in NAMES:
    rows = json.loads((r / name / 'validation.json').read_text())
    assert [x['step'] for x in rows] == list(range(0, 1000001, 500))
    assert json.loads((r / name / 'result.json').read_text())['final'] == rows[-1]
    print(name, 'final/max-prefix', rows[-1]['perfect_prefix_depth'],
          max(x['perfect_prefix_depth'] for x in rows))
    for i in range(64):
        print('P', i+1, 'final/max', rows[-1]['positions'][i]['correct'],
              max(x['positions'][i]['correct'] for x in rows), 'first50/512',
              [next((x['step'] for x in rows if x['positions'][i]['correct'] >= t), None)
               for t in (50, 512)])
PY
```

Standalone local expressivity only: **not resilient DiLoCo or Frontier acceptance**. No distributed behavior changed/tested; resilient requirement IDs and Slurm Partition/QOS evidence are **not applicable**. Independent acceptance review remains reviewer-owned. Unrelated pre-existing worktree changes are not part of this commit; no files are left staged after committing.
