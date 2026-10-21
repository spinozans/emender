# X12 — completed T64 / 1M separation growth report

**GDN2: five perfect positions, then P6 = 491/512 (95.8984375%). E97: no learned positions; P1–P11 zero, scattered chance suffix hits.** Both scratch seed0 runs completed the registered **1,000,000-update safety ceiling**. Fixed-final perfect prefixes are **5 versus 0**, also their maximum simultaneous prefixes. The four-layer operator layer-budget prediction **(d) is REFUTED**; carry-robustness/separation prediction **(a) is confirmed in direction and meets its registered relative margin**. E97's failure even to acquire P1 is a third, unanticipated finding, not evidence that an acquired T64 carry circuit collapsed.

## Evidence authority and matched recipe

Evidence root **C/** is `experiments/expressivity_tasks/results_invertible_stage12_separation_8m/`. Authority is the [prelaunch registration](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/REGISTRATION.md), [machine registration](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/registration.json), [amendment-1](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/registration-amendment-1.json) and [supersession receipt](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/registration-supersession.json), not the earlier armed/pending launch snapshot. Original registration commit **acae576d** remains in history; the supersession receipt refers to the **83c01065 chain**. Amendment-1 admits W3's committed INVALID/INCONCLUSIVE partial publication as completed gate evidence, not a passing W3 gate. Prelaunch re-registration **f378d715b29e7cf751a7f6756fd53911f1ff5c5d** binds the amended admission-control bytes and is both trainers' recorded source commit. Effective registration SHA256 is **`0aa85346a269daab740a8fbd7b532444e332b46a45f6ed9c238498cda1a0f89d`**. Admissions bind W3 report commit `4968681076da344ffe2034dd0ab60a84b9d810ca` and `receipts/w3-matcheddose/gate-read/partial-verdict.json`.

Both models have **four layers**, dim256, MLP ratio2: E97 **9,900,800** parameters, stock FLA GDN2 **9,887,056** (nominal8M). The runner explicitly asserts `len(model.layers) == 4` and the parameter counts. Both use fp32 parameters and bf16 autocast; E97 retains its established legacy-bf16 recurrence, GDN2 the unmodified stock chunk kernel, not X7's fp32 adapter. Scratch seed0, p61 / vocabulary3721, direct T64 from update1, batch32, uniform masked-target CE, ordinary live-parameter AdamW, betas(.9,.999), eps1e-8, all-parameter decay.01, clip1. Warmup through5K to1e-3, cosine to1e-5 at1M. No ladder, auxiliary carry/intermediate-state supervision, curriculum, metric stop, restart or wall-time stop. Procedural batch RNGs reset identically per architecture; the frozen X3 seed0 T64 validation pool has512 sequences and is never sampled for training. Each arm has **1M finite, consecutive training rows and2,001 validations**, baseline plus every500 updates. These are validation observations, not held-out/test-set or cross-seed evidence.

Atomic DONE markers equal successful terminal receipts and hash-bind `result.json`; each result is `complete`, `stop_reason=safety_ceiling`, final step1M. E97 wall time **23,926.752s**, GDN2 **38,045.675s**, including evaluation/checkpoint overhead, not exclusive GPU compute. Original watcher records **2026-10-21T07:23:09Z DONE X12 T64 matched1M**. Read-only watcher `status(C)` independently returns `(0, 'DONE X12 T64 matched1M')`. No new training or model evaluation was performed for this report.

## Final, best and independent maxima are different readouts

All position entries below are **joint full-state correct/512**, not coordinate accuracy. The primary decision uses fixed-final1M, not the aggregate-best retained checkpoint and not independent column maxima.

| Arm / checkpoint | Perfect prefix | P1–P6 correct/512 | Aggregate correct/32768 | CE (nats) | Terminal P64 | Post-suffix7 P9–P64 |
|---|---:|---|---:|---:|---:|---:|
| E97 fixed-final1M | 0 | 0,0,0,0,0,0 | 11 | 8.221735179 | 0/512 | 11/28672 |
| E97 aggregate-best496K | 0 | 2,0,2,0,1,1 | 20 | 8.221748233 | 0/512 | 14/28672 |
| GDN2 fixed-final1M | **5** | **512,512,512,512,512,491** | 3057 | 7.457879722 | 0/512 | 6/28672 |
| GDN2 aggregate-best990.5K | 5 | 512,512,512,512,512,495 | 3068 | 7.457407027 | 0/512 | 13/28672 |

Aggregate-best selection maximizes aggregate joint accuracy, then minimizes CE on ties; it is not minimum-CE selection. GDN2's independent P6 maximum is **496/512 at992,500**, not491 or512. No P7+ ever reaches the material50/512 threshold (largest suffix maximum4). E97's **P1 best-anywhere is2/512** across all2,001 observations, first at193,500, last at783,000. Its entire per-position maxima row is printed below: some other columns attain3, so2 is not a blanket across-position maximum. No E97 position ever reaches50 or512; no P1 foothold consolidates anywhere in the1M trajectory.

**Correction to “E97 all-zero”:** the literal final vector contains11 isolated ones, at P12,20,21,29,30,31,35,45,51,58,63; all other positions are0. These are chance-level suffix hits, not learned positions. The uniform full-state reference is **1/3721 =0.0268744961%**, about **0.1376 expected matches per512-example position**, CE **ln3721 =8.221747728**. Repeatedly maximizing over2,001 validations can produce isolated2s/3s without acquisition. This is a descriptive guessing-floor comparison, not a significance test or proof of a model's internal strategy. The suffix-only theorem applies only to observers missing at least two iid drivers; it is **not a bound on recurrent models allowed the prefix**. Aggregate prefix credit and chance suffix hits do not establish deep tracking.

### Acquisition and the full64-position receipt

| GDN2 position | First >=50/512 | First512/512 | Final/512 | Maximum/512 |
|---|---:|---:|---:|---:|
| P1 | 20,500 | 38,000 | 512 | 512 |
| P2 | 54,500 | 58,500 | 512 | 512 |
| P3 | 68,500 | 81,500 | 512 | 512 |
| P4 | 77,000 | 87,500 | 512 | 512 |
| **P5** | **101,500** | **829,000** | **512** | **512** |
| **P6** | **532,000** | not reached by1M | **491** | **496** |

First crossings are observed on a500-update grid, not exact learning-event times or permanent retention. The first simultaneous five-perfect prefix is829K; **165 later validations fall below five**. The final five is real but not an assertion of uninterrupted perfection since829K. All E97 acquisition times and all GDN2 P7+ acquisition times are right-censored at1M. Independent maxima below need not coexist at a single checkpoint.

| Position | E97 final | GDN2 final | E97 maximum | GDN2 maximum |
|---|---:|---:|---:|---:|
| P1 | 0 | 512 | 2 | 512 |
| P2 | 0 | 512 | 2 | 512 |
| P3 | 0 | 512 | 3 | 512 |
| P4 | 0 | 512 | 2 | 512 |
| P5 | 0 | 512 | 2 | 512 |
| P6 | 0 | 491 | 2 | 496 |
| P7 | 0 | 0 | 2 | 3 |
| P8 | 0 | 0 | 2 | 3 |
| P9 | 0 | 0 | 2 | 2 |
| P10 | 0 | 0 | 2 | 2 |
| P11 | 0 | 0 | 3 | 3 |
| P12 | 1 | 0 | 2 | 2 |
| P13 | 0 | 0 | 2 | 3 |
| P14 | 0 | 0 | 2 | 2 |
| P15 | 0 | 0 | 3 | 2 |
| P16 | 0 | 0 | 2 | 2 |
| P17 | 0 | 0 | 3 | 3 |
| P18 | 0 | 0 | 2 | 3 |
| P19 | 0 | 0 | 2 | 3 |
| P20 | 1 | 0 | 3 | 2 |
| P21 | 1 | 0 | 2 | 2 |
| P22 | 0 | 0 | 3 | 2 |
| P23 | 0 | 0 | 3 | 2 |
| P24 | 0 | 0 | 2 | 2 |
| P25 | 0 | 0 | 3 | 2 |
| P26 | 0 | 0 | 3 | 2 |
| P27 | 0 | 0 | 2 | 2 |
| P28 | 0 | 0 | 2 | 2 |
| P29 | 1 | 0 | 3 | 2 |
| P30 | 1 | 0 | 2 | 3 |
| P31 | 1 | 0 | 2 | 2 |
| P32 | 0 | 0 | 2 | 3 |
| P33 | 0 | 1 | 3 | 2 |
| P34 | 0 | 0 | 2 | 2 |
| P35 | 1 | 0 | 2 | 2 |
| P36 | 0 | 0 | 2 | 2 |
| P37 | 0 | 0 | 2 | 3 |
| P38 | 0 | 0 | 2 | 3 |
| P39 | 0 | 0 | 2 | 2 |
| P40 | 0 | 0 | 2 | 3 |
| P41 | 0 | 0 | 3 | 2 |
| P42 | 0 | 1 | 2 | 3 |
| P43 | 0 | 0 | 3 | 2 |
| P44 | 0 | 0 | 3 | 3 |
| P45 | 1 | 0 | 2 | 2 |
| P46 | 0 | 0 | 3 | 2 |
| P47 | 0 | 0 | 3 | 2 |
| P48 | 0 | 1 | 3 | 3 |
| P49 | 0 | 0 | 2 | 3 |
| P50 | 0 | 0 | 2 | 2 |
| P51 | 1 | 0 | 2 | 2 |
| P52 | 0 | 0 | 3 | 3 |
| P53 | 0 | 0 | 3 | 3 |
| P54 | 0 | 0 | 2 | 2 |
| P55 | 0 | 0 | 3 | 2 |
| P56 | 0 | 0 | 3 | 2 |
| P57 | 0 | 0 | 3 | 2 |
| P58 | 1 | 2 | 2 | 3 |
| P59 | 0 | 0 | 2 | 2 |
| P60 | 0 | 0 | 3 | 2 |
| P61 | 0 | 1 | 3 | 4 |
| P62 | 0 | 0 | 2 | 2 |
| P63 | 1 | 0 | 3 | 3 |
| P64 | 0 | 0 | 2 | 3 |

## P6 end slope: still consolidating, but not a sustained endpoint-count climb

Apply [X11's censoring discipline](X11-growth-report.md): separate broad-window gain, narrow-window noise, CE improvement and the unresolved time-to-perfection.

| Update | GDN2 P5/512 | P6/512 | P6 CE (nats) |
|---:|---:|---:|---:|
| 500,000 | 501 | 27 | 3.935682818 |
| 600,000 | 503 | 107 | 2.812830004 |
| 700,000 | 504 | 222 | 1.807023074 |
| 800,000 | 509 | 368 | 0.931004440 |
| 829,000 | 512 | 377 | 0.831523298 |
| 850,000 | 509 | 415 | 0.667586369 |
| 900,000 | 511 | 462 | 0.487760833 |
| 950,000 | 511 | 490 | 0.379155032 |
| 975,000 | 512 | 492 | 0.360550693 |
| 990,500 (aggregate-best) | 512 | 495 | 0.345887671 |
| 999,500 | 512 | 487 | 0.352246885 |
| 1,000,000 (fixed-final) | 512 | 491 | 0.350182296 |

Descriptive OLS uses **all validations in each inclusive window**, units correct examples per1,000 updates:

| Window | Observations | P6 OLS slope /1K | Endpoint counts | Count range |
|---|---:|---:|---|---|
| 900K–1M | 201 | **+0.334671** | 462→491 | 453–496 |
| 950K–1M | 101 | **+0.222365** | 490→491 | 474–496 |
| 975K–1M | 51 | **+0.075656** | 492→491 | 483–496 |

**Was P6 still climbing at1M?** It was improving over the broader late trajectory, with positive but slowing OLS slopes and lower CE even across975K–1M. It was **not a sustained endpoint-count climb**: last25K endpoints492→491, versus487→491 over the last500 updates. The short-window slope is small relative to the13-count fluctuation range. Best P6 occurs7.5K before the ceiling. Report **incomplete, late consolidating / right-censored**, not “six perfect,” a proven plateau, or a projected time-to512. More compute might consolidate P6; these receipts cannot guarantee it.

The scoped **compute-bound reading** is that GDN2 encountered the imposed update ceiling while its sixth position was near-perfect and still showing late loss improvement; **no architectural depth wall was observed**. The observed perfect-depth limit was the experiment's ceiling-limited readout, not an established four-layer or five-step capacity limit. This does not prove unlimited depth, that more updates must work, or isolate compute as the sole cause of the X11→X12 improvement: X12 is scratch T64 with a different frozen pool and a1M cosine denominator, not continuation of X11's T16 checkpoint.

## Adjudicating both prelaunch predictions, plus the third finding

**(d) Operator layer-budget: REFUTED in the registered learned-solution sense.** Four asserted layers composed **five perfect nonlinear task steps**, with a sixth at491/512. Thus “no more nonlinear steps than layers” fails plainly; the E97-beyond-four expectation also fails here. X11's four-perfect plus partial P5 was not a demonstrated layer cap: X12 converts P5 to512 and materially composes P6. The registered readout→MLP→write-back account supplies per-timestep nonlinearity; a once-per-layer-per-timestep opportunity is **not a once-per-layer budget for the whole sequence**. Layer count alone does not bound temporal nonlinear composition. This reconciles the observation with the available nonlinear path; it does **not** identify the learned circuit or prove which path the trained model uses, and is not an unbounded-capacity theorem.

**(a) Carry-robustness/separation: confirmed in direction, registered margin met.** Fixed-final GDN2 exceeds E97 by **five perfect positions**, greater than the registered example margin of two. GDN2's working-recipe prefix progresses **T16 four perfect + P5=318/512 (X11)** to **T64 five perfect + P6=491/512**. This is horizon robustness of the learned prefix under the tested recipes, not successful full-T64 tracking or proof that affine carry alone caused the advantage. **(a) and(d) were competing predictions; they are not retroactively merged into one successful story.** Parity alternative(b) does not hold; (c)'s “neither learns P5” antecedent is false because GDN2 acquires it.

**Unanticipated third finding: E97's total T64 learning collapse.** It did not merely stop at four or fail to extend P5: even P1 never consolidated (trajectory maximum2/512). “Collapse” here names failure relative to its working T16 result, **not forgetting a learned T64 P1**. X10 AdamW at T16 composed exactly its layer count (**4/4**, final P5=7/512, maximum15); that numerical echo is an observation, **not a mechanism claim**. T64's failure is not evidence of a four-layer capacity cap.

This third finding is **consistent with [X8's foothold-starvation interpretation](X8-growth-report.md)**: uniform T64 gives P1 an objective coefficient **1/64 =1.5625% (~1.6%)**, fourfold less than uniform T16's6.25%. X8's depth-weighted T16 coefficient was1/136 (~0.7353%), and E97 likewise never acquired P1 there. P1 needs no deep carry prerequisite. These are nominal loss coefficients, **not measured gradient shares or fractions of realized CE**. X8 used a different optimizer regime, and X12 changes horizon, pool, ceiling and schedule together versus T16. No matched T64 foothold-weight intervention or gradient/circuit measurement isolates the cause. Therefore **register consistency, not causal attribution to starvation**, tanh, state overwrite, optimizer or layer count.

## Scoped series synthesis

At nominal8M **under working recipes**, the linear-in-state GDN2 architecture learns to chain nonlinear composition **deeper than its layer count**, with a learned prefix robust to the longer training horizon. The delta-correcting E97 architecture composed exactly its four layers' worth of steps at T16 and failed entirely to acquire even P1 at T64 under uniform loss. This is the learned-solution outcome of these runs, not a universal architecture ranking.

The series does not support a simple “linear state cannot compose nonlinearly” story. X8 showed that paying for deep positions did not rescue acquisition and could starve the easy foothold; X10/X11 showed optimizer-recipe sensitivity and four-perfect T16 parity, with very different partial P5 acquisition; X12 breaks that parity in GDN2's favor while refuting the proposed layer-count cap. Earlier scaffold/precision/recipe probes remain finite learning evidence, not proofs about representability. No component-level causal attribution follows from recipe bundles.

**Single seed (seed0), one task family, nominal8M, finite ceiling, frozen validation only.** No capacity bound, impossibility, universal expressivity separation, proof of unlimited composition, production capability or cross-seed robustness claim. Standing [task framing](experiments/expressivity_tasks/INVERTIBLE_QUADRATIC.md#interpretation-constraints) remains binding: the finite-state recurrence admits a sufficiently large one-hot linear-state permutation representation; invertibility removes task-state merging, not every shortcut or optimization confound. Accurate task columns establish learned task performance, not discovery of the intended exact internal recurrence. Neither architecture solves full T64; both finish P64 at0/512 and chance-level post-suffix7 readout. No new arm, continuation, scale launch or checkpoint deletion is authorized by this report.

## Reproduction, hashes and validation

Raw streams/configs/checkpoints remain **local/untracked**, as with prior growth reports; this commit adds only `X12-growth-report.md`, not a claim that raw receipts were pushed. Exact local evidence identities:

| Artifact under C/NAME | E97 SHA256 | GDN2 SHA256 |
|---|---|---|
| result.json | `91a180de7c5710e786f7ea76db06b454de8f8ac5bb840a2c0c8fd35c2e944ee7` | `5f0e820fe28103d8df69f8449463d100e7f6dae78716046ef905d84d5fb43208` |
| validation.json | `193d9021689744e695d1bb08ac63679ca3fbcc3a3ca3044910071355d6235fa2` | `96225b5979597c282296e5fb62638e0ee8a9edb85f3bfe23e06656690032d489` |
| training.jsonl | `195c7fb5e43a476dbfe9d0c4808ea560bca1cb7d0e06f912469f86061d54f15d` | `9e40a3693e690bd4d892aca924fb1e9b811157380fac606a4a7fdfd2fe2a1bfd` |
| config.json | `b2359775816da8b4b02d7443ee26ec882dc5c560578efd3f342deaebb3b83587` | `a002365a66334dd291e73920923d9db757240a395f7f7335d865c10f20e491f0` |

Read-only local audit on lambda01: `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/audit_x12_growth.py` **passed**. It verified all registered source/dependency hashes, config/registration identity, DONE/terminal/result bindings, complete500-update grids, all positional counts/prefixes/aggregate joint totals, finite losses, both1M consecutive finite loss/gradient/LR streams against the exact extended schedule, and extracted maxima/acquisition/slope rows. Initial invocation lacked `PYTHONPATH=.` and failed to import `scripts`; corrected invocation passed. This temporary audit is not a committed test dependency and does not re-evaluate checkpoints or assert hidden-circuit mechanisms.

Reconstruct the key counts and slopes directly (CPU-only on lambda01; on Frontier first activate the canonical environment and use `$EMENDER_PYTHON`):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.watch_invertible_stage12_separation import status
c = Path('experiments/expressivity_tasks/results_invertible_stage12_separation_8m')
assert status(c) == (0, 'DONE X12 T64 matched1M')
for name in ('e97_bf16_seed0', 'gdn2_bf16_seed0'):
    rows = json.loads((c / name / 'validation.json').read_text())
    assert [r['step'] for r in rows] == list(range(0, 1000001, 500))
    assert json.loads((c / name / 'result.json').read_text())['final'] == rows[-1]
    print(name, 'final:', [p['correct'] for p in rows[-1]['positions']])
    print('maxima:', [max(r['positions'][i]['correct'] for r in rows) for i in range(64)])
    print('prefix final/max:', rows[-1]['perfect_prefix_depth'], max(r['perfect_prefix_depth'] for r in rows))
    for i in range(6):
        print('P', i+1, 'first50/512:', [next((r['step'] for r in rows if r['positions'][i]['correct'] >= t), None) for t in (50, 512)])
    if name.startswith('gdn2'):
        for start in (900000, 950000, 975000):
            selected = [r for r in rows if r['step'] >= start]
            xs = [r['step']/1000 for r in selected]
            ys = [r['positions'][5]['correct'] for r in selected]
            xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
            slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
            print('P6 OLS:', start, len(xs), slope, 'range:', min(ys), max(ys))
PY
```

Existing CPU suite: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_invertible_stage12_separation_8m.py` — **15 passed (7.89s)**. No tests added/changed for this documentation-only task. No resilient DiLoCo behavior, distributed acceptance gate or Frontier Slurm binding is changed or claimed; distributed requirement IDs are not applicable. Independent acceptance review remains required.
