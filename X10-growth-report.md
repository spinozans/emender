# X10 — completed three-arm optimizer growth report

**Muon composes: CONFIRMED for the tested seed0 rule.** At the fixed **500,000-update** endpoint, hybrid Muon has three perfect positions and **P4=448/512 (87.5%)**; AdamW+cosine has four perfect positions; SF-constant remains shallow. Muon enters P2/P3 earlier in updates than paired AdamW, but reaches material P4 later and is slower in measured wall time. P4 perfection for Muon is **censored, not resolved** by the ceiling. These are learned-solution observations, not architecture/capacity bounds or a general optimizer ranking.

Evidence root **C/**: `experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m/`. Executable [registration](experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m/registration.json) at **d9785a65530fd8c720c26063541c89522d7124eb**, the [registered Stage-1 predictions](docs/OPTIMIZERS-2026-report.md#stage-1--the-repeated-map-abc-instrument), and [launch report](X10-launch-report.md) at **3f30c87215154443649132e2e45ddcca311c62db** precede the outcomes. Each arm's atomic DONE equals its successful terminal receipt and binds `result.json`. All three results are complete, stop reason **safety_ceiling**, with **500,000 consecutive training rows and 1,001 validations** (baseline plus every500 updates). This report only reads CPU-side receipts; no training, model loading or GPU evaluation was performed. W3 and its GPU ownership are untouched.

## 1. Receipted fixed-final comparison

All counts are **joint-state correct out of512**, not coordinate accuracy. Vectors list **all P1–P16**, aggregate denominator8192. Perfect-prefix depth requires every preceding position simultaneously512/512; a partial P4 is not depth4.

| Arm | Fixed-final P1–P16 counts | Correct/8192 | Final CE (nats) | Final perfect prefix | Maximum simultaneous perfect prefix |
|---|---|---:|---:|---:|---:|
| SF-constant | `510,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0` | 511 | **7.847842544** | 0 | 1 |
| AdamW+cosine | `512,512,512,512,7,0,1,0,0,0,0,0,0,0,0,0` | 2056 | **5.949373424** | **4** | **4** |
| Hybrid Muon | `512,512,512,448,0,0,0,0,0,0,0,0,0,0,0,0` | 1984 | **6.203891933** | **3** | **3** |

**One seed per arm, seed0 only.** AdamW freshly repeats X9's ladder-free composition unlock and extends it to perfect P4 (X9 final P4=10, maximum18). This is a **fresh-run replication, not a new independent seed**: the launch receipt explicitly verifies bit-identical X9 initialization and the same seed0 frozen pool. Seeds1/2 were registered but not run here. No cross-seed robustness or promotion gate is declared passed.

The shared instrument is scratch E97 T16, p61/vocab3721, dim256/four layers, **9,900,800 parameters**, batch32, uniform target CE, clip1, fp32 parameters/bf16 autocast, legacy recurrence and sequential Triton evaluation. There is no ladder or auxiliary carry/intermediate-state supervision; ordinary task-target CE is supervised. Initialization, procedural batch order and frozen512-example pools are matched across arms. Muon/AdamW evaluate **ordinary live parameters**; SF evaluates **eval-x**, restoring train-y before subsequent updates.

### Best versus final versus independent maxima

“Best” selects aggregate correct count, breaking ties by lower CE; it does **not** select minimum CE and is not the primary endpoint.

| Arm / aggregate-best update | P1–P16 counts at that one checkpoint | Correct/8192 | CE | Terminal P16 | Suffix7-restricted P9–P16 |
|---|---|---:|---:|---:|---:|
| SF /221,500 | `512,11,0,2,0,0,0,0,0,0,0,0,0,0,2,0` | 527 | 7.600624114 | 0/512 | 2/4096 |
| AdamW /468,000 | `512,512,512,512,14,0,0,0,1,0,0,1,1,0,0,1` | 2066 | 5.952114731 | 1/512 | 4/4096 |
| Muon /497,500 | `512,512,512,450,0,0,0,0,0,0,0,0,0,0,0,0` | 1986 | 6.204250395 | 0/512 | 0/4096 |

Independent per-column maxima across all1,001 validations, **not one checkpoint**:

- SF: `512,11,2,2,2,2,3,2,2,2,1,1,2,2,2,2`.
- AdamW: `512,512,512,512,15,2,2,3,2,2,2,3,2,3,3,2`.
- Muon: `512,512,512,450,2,2,4,2,2,3,2,2,3,2,3,2`.

**Every arm's fixed-final terminal score is0/512 and suffix7-restricted score is0/4096.** The uniform joint marginal is **1/3721=0.0268744961%**, with CE **ln3721=8.221747728 nats**. The suffix-only bound concerns observers missing at least two iid drivers, not recurrent models allowed prefix information. Isolated chance-sized suffix counts and aggregate prefix credit do not establish deep tracking. No P5+ position reaches the registered50/512 materiality bar in any arm.

## 2. Acquisition pace: paired thresholds and measured seconds

First observed crossings on the **500-update grid**, not exact learning-event times or monotonic retention. Each populated cell is **update / elapsed seconds**, using that validation's runner `elapsed_s`, rounded to0.001s. The complete training streams independently check the update clock, LR and timing rows; validation elapsed includes evaluation through the crossing rather than only the preceding training update.

| Position / threshold | AdamW+cosine: update / seconds | Hybrid Muon: update / seconds | SF-constant: update / seconds |
|---|---:|---:|---:|
| P1 >=50 | 24,000 /689.050 | 19,000 /1,646.557 | 29,500 /756.496 |
| P1 =512 | 50,000 /1,335.961 | 100,000 /6,707.446 | 79,000 /1,921.302 |
| P2 >=50 | 187,500 /4,557.050 | 128,000 /8,257.525 | not reached |
| P2 =512 | 223,000 /5,186.064 | 186,000 /11,430.645 | not reached |
| P3 >=50 | 231,500 /5,357.632 | 189,500 /11,613.969 | not reached |
| P3 =512 | 306,500 /6,890.168 | 281,500 /16,661.497 | not reached |
| P4 >=50 | 309,000 /6,933.623 | 348,500 /20,410.467 | not reached |
| P4 =512 | 391,500 /8,605.185 | not reached by500K | not reached |

**Ahead on P3, behind on P4—not on-pace throughout.** Muon first reaches perfect P2 **37,000 updates earlier** and perfect P3 **25,000 earlier** than paired AdamW; material P3 is42,000 earlier. Material P4 is **39,500 updates later**, and Muon never reaches perfect P4. P1 perfection is50,000 updates later despite its earlier material crossing. Against the historical descriptive X9 P2-first512 benchmark245,500, Muon's186,000 is59,500 earlier, but the **paired fresh AdamW223,000** is the valid same-threshold comparator.

| Arm | End-to-end wall seconds | Mean wall seconds/update | Summed optimizer seconds | Summed training-update seconds |
|---|---:|---:|---:|---:|
| SF | 10,321.531 | 0.020643 | 385.450 | 9,793.949 |
| AdamW | 10,540.194 | 0.021080 | 514.178 | 9,999.365 |
| Muon | 28,071.817 | 0.056144 | 17,569.074 | 27,582.441 |

Muon uses about **2.66x AdamW's end-to-end seconds** for the same500K updates; perfect P3 takes16,661.497s versus6,890.168s. It is **not a wall-time speed win**. Optimizer time includes the hybrid's auxiliary AdamW; end-to-end time also includes compilation, evaluation and checkpoint costs. Timings are observed on the launched cards, not an equal-tuned or hardware-normalized ranking.

## 3. Stage-1 verdict against the registrations

1. **Requested(a), Muon composes: CONFIRMED at seed0.** P2/P3 reach512 and P4 reaches448 at final (maximum450), all exceeding the **>=50/512** materiality threshold. The **faster-basin-dynamics sub-prediction is supported for P2/P3 in update counts**, but not for P4 or end-to-end time; a blanket “Muon is faster” prediction is not supported. The actual timings above resolve the pace question without converting a partial P4 into perfect composition.
2. **Requested(b), P4: the registered ANY-arm >=50 question is answered YES**, for both AdamW and Muon. AdamW also reaches **P4-perfect**, retained at final; Muon exits at **87.5%, not perfect**. Whether Muon would consolidate to512 beyond500K is **censored, not resolved**, following [X11's censoring discipline](X11-growth-report.md). It is neither a proven plateau nor a promised later success. AdamW's P4 result prevents a Muon-specific deeper-composition claim.
3. **Requested(c)/R4, SF flat: CONFIRMED for this recipe.** P2's maximum is11, P3/P4 maxima2; no P2+ position ever reaches50. Final P1=510 means strict final prefix0, not1, despite transient perfect P1. This repeats the series' shallow-SF pattern: five distinct completed E97 runs are X7 fp32, X7 bf16, X8 pay-depth, X9b annealed-SF and X10 constant-SF; their maximum-anywhere P2+ counts are respectively **6,6,2,11,11**. These are **not five independent seeds or five identical interventions** (precision, loss and schedule differ), nor evidence that every SF repair fails. X8 never even acquires P1 materially.

**Formal R1/R2/R3 remain unadjudicated across seeds.** R1's B>=3/C<=1 and R2's A>=3 conditions hold for this one seed, but each registered replication criterion requires **at least2/3 seeds**. Muon has no seed0 perfect-depth advantage (3 versus AdamW4), and cannot satisfy R3's positive depth>=4 condition from this endpoint; the robust/no-robust-improvement prediction still cannot be resolved from one seed. No automatic seed1/2 launch, post-result tuning, production promotion or Stage2 authorization follows this report.

### Is Muon still gaining at the ceiling?

| Update | Muon P4 correct/512 | P4 CE (nats) |
|---:|---:|---:|
| 300,000 | 8 | 4.338019314 |
| 350,000 | 58 | 3.324844270 |
| 400,000 | 216 | 1.443670629 |
| 450,000 | 375 | 0.845723374 |
| 475,000 | 419 | 0.703900697 |
| 490,000 | 435 | 0.654940858 |
| 497,500 (maximum) | 450 | 0.632376497 |
| 499,500 | 441 | 0.630647139 |
| 500,000 | **448** | **0.629351722** |

Unlike X11's nearly flat last25K P5 counts, Muon P4 has positive descriptive OLS slopes over **all101 validations450K–500K: +1.355900 correct/1K updates**, and **all51 validations475K–500K: +1.204706/1K**. The final500-update count rises441→448, but final is below the450 maximum. This supports **still gaining in late windows**, not monotonic growth, a projected perfection time, or an asymptotic conclusion. The ceiling censors consolidation even with a positive late slope.

## 4. Decay, RMS scaling and routing receipts

The executable historical recipe and approved [launch clarification](X10-launch-report.md#approved-decay-policy-clarification) apply **nominal decay0.01 to all parameters in all three arms**, including the **eight `A_log`/`dt_bias` tensors whose `_no_weight_decay` tags are explicitly ignored**. LR multipliers are1 throughout. The earlier conceptual spec's shared-exclusion wording was superseded by this registered historical-recipe clarification, not silently repaired after outcomes. Nominal0.01 does **not** imply equivalent SF/AdamW/Muon shrink dynamics. Decay-policy sensitivity remains a registered follow-up, not a variable changed here.

Muon is pinned **x10-hybrid-muon-fp32-v1**: momentum0.95, Nesterov enabled, five FP32 quintic Newton–Schulz iterations with coefficients **(3.4445,-4.7750,2.0315)**, Frobenius normalization `norm+1e-7`, tall transpose and zero→zero handling. Its **`0.2*sqrt(max(rows,cols))` shape scaling** is approximate AdamW update-RMS matching **as part of Muon proper**, not an unregistered recipe mismatch, exact polar orthogonality, or forced exact0.2RMS. Auxiliary AdamW and the AdamW arm use betas(.9,.999), eps1e-8; both hybrid groups and AdamW share5K warmup to1e-3 and cosine to1e-5 at500K. SF uses installed AdamWScheduleFree, betas(.9,.95), constant1e-3 with zero warmup. Class, beta2, schedule and averaging/decay semantics are bundled; no unique mechanism attribution is licensed.

Exhaustive names, aliases, shapes, numel, owner, decay, LR multiplier and ignored-tag flags are receipted in **`C/registration.json:parameter_routes`**, with source/probe identities in the same registration and actual launch/config bindings in **`C/receipts/launch_verification.json`**, **`launch_verification_{muon,adamw,sf}.json`**, and **`launch_snapshot/<arm>_seed0/config.json`**. The map assigns exactly **36 hidden linear matrices** to Muon (qkv, a/g/erase/value-write/o projections plus MLP w1/w2/w3 per layer). Embedding/tied vocabulary head, norms/scalars/vectors stay auxiliary AdamW. The tied aliases have one owner; routing is semantic, **not ndim-only**. Existing source-named routing/NS/RMS/serialization tests remain the implementation evidence, not a post-result reinterpretation of the optimizer.

## 5. Reproduction and audit bounds

Read-only audit on **lambda01**, using the approved project `.venv` and `CUDA_VISIBLE_DEVICES=''`, passed terminal/DONE/result binding, complete validation grids/position totals/prefixes, all **1.5 million** finite consecutive training loss/gradient/LR/timing rows, exact per-update schedules, timing sums, routing totals and first crossings. The appendix reproduces the extraction; hashes identify the local raw streams, **not a claim they were pushed**. Checkpoint predictions were not re-evaluated and the complete source/dependency/checkpoint-hash audit was not rerun; launch evidence remains separately identified above.

| Arm | Artifact | SHA256 |
|---|---|---|
| AdamW | result.json | `6f8d1b9e601900eb6e197ab2daee904391e070d2c3219f9e086a4ae422d122b4` |
| AdamW | validation.json | `704f6212052d6f1924033e93f02840747f23b2f2b36d1e0fa227daa122428da6` |
| AdamW | training.jsonl | `c67556d0ce9d1f85ab7a70b1e214303c7764f717265e2cf1456bdc76a70b7549` |
| Muon | result.json | `1255fa1ec49120dd31c9f0dcb29fc831bc764737529c8f6d2fcf4c989b372937` |
| Muon | validation.json | `21b79cb6bf1a8bff7cd31e5e778f8a4bb7bd38ffb166f185d4c3d212877ca7f9` |
| Muon | training.jsonl | `e94e85a002b76b074a69b6faefde8cf7131a06ffaa9db6589e833d2e1463963d` |
| SF | result.json | `f9a7f1fbb9243959e87dfa792ff58d3046a010c8c76ff6a22d8463cbd6cf46dc` |
| SF | validation.json | `41ebe5d8bb0529c59182f428b15f0c7466cc6b779bc54107826a97210733fe63` |
| SF | training.jsonl | `777a542bd692990dfc86ad8ababeac1fc0db1bda33506941fd73315533257cd6` |

Existing CPU regression command: `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_invertible_stage10_optimizers_8m.py -k 'not gpu'` — **19 passed,4 GPU tests deselected**. No tests/code were added or changed. Local single-card learned-solution scope only; no distributed/resilient training behavior or requirements are tested or claimed. Independent reviewer acceptance remains required.

### CPU-only receipt audit (run from repository root)

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import hashlib, json, math
from pathlib import Path
root = Path('experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = read(root / 'registration.json')
routes = reg['parameter_routes']
assert sum(r['numel'] for r in routes) == 9900800
assert sum(r['owner'] == 'muon' for r in routes) == 36
assert sum(r['no_weight_decay_tag_ignored'] for r in routes) == 8
assert all(r['weight_decay'] == .01 and r['lr_multiplier'] == 1 for r in routes)
expected = {
    'adamw': ([512]*4+[7,0,1]+[0]*9, [24000,187500,231500,309000], [50000,223000,306500,391500]),
    'muon': ([512]*3+[448]+[0]*12, [19000,128000,189500,348500], [100000,186000,281500,None]),
    'sf': ([510,0,1]+[0]*13, [29500,None,None,None], [79000,None,None,None]),
}
summary = {}
for arm, (counts, material, perfect) in expected.items():
    p = root / f'{arm}_seed0'
    v, res = read(p/'validation.json'), read(p/'result.json')
    term = read(root/f'receipts/{arm}_seed0/terminal.json')
    assert term == read(root/f'markers/{arm}_seed0.DONE')
    assert term['exit_code'] == 0 and term['result_sha256'] == sha(p/'result.json')
    assert res['status'] == 'complete' and res['total_steps'] == 500000
    assert res['stop_reason'] == 'safety_ceiling' and res['final'] == v[-1]
    assert [a['step'] for a in v] == list(range(0,500001,500))
    for a in v:
        assert len(a['positions']) == 16
        assert all(b['count'] == 512 and 0 <= b['correct'] <= 512 for b in a['positions'])
        assert sum(b['correct'] for b in a['positions']) == a['aggregate']['correct']
        assert math.isfinite(a['aggregate']['loss'])
        prefix = next((i for i,b in enumerate(a['positions']) if b['correct'] != 512),16)
        assert prefix == a['perfect_prefix_depth']
    assert [b['correct'] for b in v[-1]['positions']] == counts
    assert v[-1]['aggregate']['terminal_correct'] == v[-1]['aggregate']['post_suffix7_correct'] == 0
    first = lambda i,t: next((a['step'] for a in v if a['positions'][i]['correct'] >= t),None)
    assert [first(i,50) for i in range(4)] == material
    assert [first(i,512) for i in range(4)] == perfect
    assert all(first(i,50) is None for i in range(4,16))
    selected = set(x for x in material+perfect if x is not None)
    times, opt_total, update_total = {}, 0., 0.
    with (p/'training.jsonl').open() as stream:
        for n,line in enumerate(stream,1):
            a = json.loads(line)
            lr = (.001 if arm == 'sf' else .001*n/5000 if n <= 5000 else
                  .00001+.5*(.001-.00001)*(1+math.cos(math.pi*(n-5000)/495000)))
            assert a['step'] == n and math.isclose(a['lr'],lr,rel_tol=1e-14)
            assert all(math.isfinite(a[k]) for k in ('loss','grad_norm','lr','optimizer_seconds','update_seconds','elapsed_s'))
            opt_total += a['optimizer_seconds']; update_total += a['update_seconds']
            if n in selected: times[n] = a['elapsed_s']
    assert n == 500000
    assert math.isclose(opt_total,v[-1]['optimizer_seconds_total'],abs_tol=1e-6)
    best = next(a for a in v if a['step'] == res['best']['step'])
    assert best['aggregate']['correct'] == max(a['aggregate']['correct'] for a in v)
    def slope(start):
        rows = [a for a in v if a['step'] >= start]
        xs,ys = [a['step']/1000 for a in rows],[a['positions'][3]['correct'] for a in rows]
        xm,ym = sum(xs)/len(xs),sum(ys)/len(ys)
        return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    summary[arm] = dict(hashes={f:sha(p/f) for f in ('result.json','validation.json','training.jsonl')},
        final=counts, ce=v[-1]['aggregate']['loss'], final_prefix=v[-1]['perfect_prefix_depth'],
        max_prefix=max(a['perfect_prefix_depth'] for a in v),
        maxima=[max(a['positions'][i]['correct'] for a in v) for i in range(16)],
        first50=material, first512=perfect, training_elapsed=times,
        validation_elapsed={a['step']:a['elapsed_s'] for a in v if a['step'] in selected},
        best_step=best['step'], best_counts=[b['correct'] for b in best['positions']], best_ce=best['aggregate']['loss'],
        best_terminal=best['aggregate']['terminal_correct'], best_suffix=best['aggregate']['post_suffix7_correct'],
        wall_seconds=res['wall_seconds'], optimizer_seconds=opt_total, update_seconds=update_total,
        p4_slope_450k=slope(450000), p4_slope_475k=slope(475000))
print(json.dumps(summary,indent=2))
print('PASS: X10 three-arm terminal/500K-stream/grid/route/acquisition audit')
PY
```
