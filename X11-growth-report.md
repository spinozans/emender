# X11 — completed stock GDN2 + AdamW growth report

**Four perfect positions plus a material, incomplete fifth:** at the fixed **500,000-update** endpoint, seed0 has joint-state counts **`512,512,512,512,318,4,0,0,0,0,0,0,0,0,0,0`**, each out of512. P5 is **318/512 (62.109375%)**, not a fifth perfect position. This is the deepest material composition observed in this series under the tested recipes, not an architectural depth theorem or a claim about other seeds. No ladder or auxiliary carry/intermediate-state supervision was used; ordinary supervised target CE was used throughout.

Evidence root **C/**: `experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m/`. [Registration](experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m/REGISTRATION.md), source commit **9255ebb67ac3b6082afbd1dac5d4524fc944dbca**, and [launch receipt](experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m/LAUNCH_REPORT.md) precede the run. `markers/gdn2_adamw_seed0.DONE` equals the successful terminal receipt and binds `result.json`; the original watcher logged **DONE at2026-10-20T17:48:51Z**. Result status is complete, stop reason **safety_ceiling**, wall time **21,415.662s**. There are **500,000** training rows and **1,001** validations (baseline plus every500 updates).

## Acquisition: first observed crossings, not monotonic retention

| Position | First >=50/512 (material) | First512/512 (perfect) | Final correct/512 | Maximum correct/512 |
|---|---:|---:|---:|---:|
| P1 | 32,500 | **71,000** | 512 | 512 |
| P2 | 70,000 | **80,500** | 512 | 512 |
| P3 | 85,500 | **213,500** | 512 | 512 |
| P4 | 112,000 | **348,500** | 512 | 512 |
| P5 | **285,000** | not reached by500K | **318** | **333 at482,500** |
| P6 | not reached | not reached | **4** | 8 at464,000 |

The first simultaneous perfect prefix of four is348,500; maximum simultaneous and fixed-final perfect-prefix depths are both **4**. There are98 later validations below four, so a first crossing is not permanent acquisition. Independent P1–P16 maxima are **`512,512,512,512,333,8,2,2,3,2,2,2,2,3,3,3`**; they are not one checkpoint. P6's4/512 does not establish material sixth-position composition.

P5's first nonzero count is1 at7,500, which is not evidence of acquisition by itself. It first reaches10 at136,500, the registered material threshold50 at285,000,100 at315,500,200 at395,000,256 at418,000, and300 at450,000. These are noisy threshold crossings on a500-update grid, not exact learning-event times.

| Update | P5 correct/512 | P5 CE (nats) |
|---|---:|---:|
| 200,000 | 13 | 4.732780187 |
| 250,000 | 34 | 3.990705285 |
| 300,000 | 49 | 3.268124801 |
| 350,000 | 96 | 2.490126102 |
| 400,000 | 185 | 1.768002014 |
| 450,000 | 300 | 1.323446240 |
| 475,000 | 319 | 1.254307295 |
| 482,500 (P5/aggregate best) | 333 | 1.220448936 |
| 490,000 | 316 | 1.220118944 |
| 499,500 | 325 | 1.214546908 |
| 500,000 (fixed final) | 318 | 1.219246435 |

**Was it still gaining at500K? Not in a sustained endpoint-count sense.** Descriptive OLS over all101 validations from450K through500K gives **+0.592918 correct examples per1,000 updates** (endpoint300→318). Over the last51 validations,475K–500K, the slope is only **+0.001991/1K**, effectively flat/noisy (319→318; range308–333). The last500-update window is **325→318**, not an increase. P5 CE nevertheless improves from1.254307295 at475K to1.219246435 at500K. Window-dependent slopes are not a projected time-to-perfection. The500K ceiling **censors whether P5 would consolidate to perfection**; these receipts do **not** demonstrate sustained endpoint growth or prove that more updates must succeed.

## Losses, marginal and suffix readout

| Checkpoint | Perfect prefix | Joint correct/8192 | Aggregate CE | P5 correct/512 | P6 correct/512 | Terminal P16 | Suffix7-restricted P9–P16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggregate-best482,500 | 4 | 2384 | 5.642261356 | 333 | 3 | 0/512 | 0/4096 |
| Fixed-final500,000 | 4 | 2370 | **5.640439212** | 318 | 4 | **0/512** | **0/4096** |

Aggregate-best selects highest joint correct count, then lower CE; it is not minimum-loss selection and is not the primary fixed-final endpoint. Final P1–P6 CE values are **0.000234096,0.000299627,0.001155894,0.024254570,1.219246435,6.784369491** nats; P7–P16 CE ranges8.221097944–8.222008238. The uniform joint marginal is **1/3721 =0.0268744961%**, with CE **ln3721 =8.221747728** nats. The suffix-only bound applies to observers missing at least two iid drivers, not to full recurrent models allowed the prefix. Neither isolated near-chance counts nor zero suffix successes establish deep tracking.

## Matched comparisons and the working-recipe depth-parity table

X11 and **X7b** match GDN2 architecture, initialization discipline, seed0, frozen T16 pool, procedural batch stream, batch32, uniform CE, clip1,500K endpoint and **stock bf16 chunk kernel (no adapter)**. Both have **four layers**, dim256, **9,887,056 parameters**, fp32 parameters/bf16 autocast. X7b uses **AdamWScheduleFree constant1e-3, betas(.9,.95)**; X11 uses ordinary **AdamW, betas(.9,.999)**,5K warmup to1e-3 then cosine to1e-5 at500K. Nominal all-parameter decay.01 and eps1e-8 match, but effective decay under SF is not equivalent to ordinary AdamW. Optimizer class, schedule and betas change as a bundle: no isolated attribution to any one component or to averaging is justified.

**Exact X7b correction:** “four perfect plus P5=3/512” mixes endpoint and maximum. X7b's **final** vector is four512s followed by twelve zeros; P5's **maximum** is3/512, first at455,500 and last at481,500. Its aggregate-best486,500 has `512,512,512,511,2,0,0,0,1,0,0,1,0,0,0,0` (P5=2, not3). Thus the like-for-like P5 comparison is **final0→318**, or **maximum3→333**. The optimizer-regime change moves GDN2's learned composition deeper in the same direction as E97's AdamW improvement, without increasing GDN2's perfect prefix beyond four yet.

All rows below are scratch seed0, T16,500K, four-layer nominal-8M models, uniform target loss, no ladder/carry loss. E97 has **9,900,800 parameters** and its established legacy-bf16 recurrence; GDN2 uses its stock kernel. The table is recipe-matched where applicable, not a kernel/parameter-identical architecture intervention. X10's SF row is the contemporaneous rerun, not silently substituted for historical X7 E97.

| Run / recipe | Fixed-final perfect prefix | Maximum simultaneous perfect prefix | Final P4/512 | Final P5/512 | Maximum P5/512 | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| X10 E97 / SF-constant | 0 | 1 | 0 | 0 | 2 | 7.847842544 |
| **X10 E97 / AdamW** | **4** | **4** | **512** | **7** | **15** | **5.949373424** |
| X10 E97 / hybrid Muon | 3 | 3 | 448 | 0 | 2 | 6.203891933 |
| X7b GDN2 / SF-constant | 4 | 4 | 512 | 0 | 3 | 6.166569799 |
| **X11 GDN2 / AdamW** | **4** | **4** | **512** | **318 (62.109375%)** | **333** | **5.640439212** |

Working-recipe **perfect-depth parity is4 vs4**, with markedly different partial P5 acquisition (E97 final7/512 versus GDN2 final318/512). It is not equivalence of depth robustness. Every table row has final terminal0/512 and suffix7-restricted0/4096.

The **X7 fp32-recurrence-adapter GDN2 arm is superseded as the GDN2 reference by both stock-kernel runs**, not erased: its P4 maximum is **294/512 at476,000**, never512; final columns begin **`512,512,511,278,0,0,...`**, final prefix2, maximum simultaneous prefix3, final CE **7.448415577**, terminal0/512 and suffix0/4096. Its partial P4 must not be compared as if it were the best stock-GDN2 depth result, nor can precision alone be causally blamed across these recipe/kernel differences.

X11 registered prediction(a)'s **5+ perfect** condition is **not met**, but its explicitly requested material P5 readout **is met**. Prediction(b)'s **four perfect with no P5 gain** condition is **not met**. This is the registered “other outcome”: four perfect plus substantial incomplete P5. Prediction(c)'s five-cell optimizer/architecture table is now filled above; one seed does not establish robustness.

## Exact forward link: X12 T64 /1M, both competing predictions retained

The authority is [X12's prelaunch registration, predictions(a)–(d)](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/REGISTRATION.md#predictions-registered-verbatim-before-launch), with [machine registration](experiments/expressivity_tasks/results_invertible_stage12_separation_8m/registration.json). X12 keeps **E97 four layers / GDN2 four layers**, the same parameter counts and working AdamW recipe, but trains **scratch T64** to a fixed **1,000,000** updates with cosine extended to1M. It is **not** continuation of X11's checkpoint or an isolated extra-update intervention: sequence length, frozen pool and schedule denominator also change. No X12 result is claimed here; admission remains behind its registered post-W3 gate.

- **Carry-robustness/separation prediction(a):** GDN2's affine carry should yield a materially longer perfect prefix than E97 past the four-position parity point (registered example: **GDN2 >=2 positions beyond E97** at T64's ceiling). X11's partial P5 is compatible with eventual separation, but does not confirm it; P5 perfection alone would not establish the registered relative margin.
- **Operator layer-budget prediction(d):** GDN2 should not perform more nonlinear steps than its layers: its prefix should cap at or near **four**, while depth-robust E97 should extend beyond its own **four**. Registered rationale contrasts linear-in-state GDN2 carry/readout→MLP→write-back with E97's state nonlinearity. These are mechanism predictions, not established conclusions. X11's material P5 puts pressure on a literal no-beyond-four reading but does not refute a perfect-prefix cap; a consolidated GDN2 fifth/deeper perfect position would test that reading directly.

**(a) and(d) remain contradictory, not a merged explanation.** X12 can determine, under its T64/1M recipe, whether GDN2 P5 **consolidates to512, plateaus short of perfection, or regresses**, and whether either architecture then advances its contiguous perfect prefix and by how much. X11's censored, near-flat/noisy endpoint cannot settle those alternatives. X12's primary decision uses **fixed-final1M** perfect prefixes; maximum simultaneous prefixes, all64 per-position maxima, and first>=50/first512 acquisition times must remain separate. Earlier perfect transients alone do not decide it. Equal prefixes invoke registered parity alternative(b), with its one-task/one-seed/recipe limitation rather than general architectural equivalence. If neither learns P5, registered(c) calls for longer ceilings or working-recipe staging; a finite failure still cannot prove an asymptotic layer cap. Other outcomes must be reported plainly.

## Reproduction and audit scope

Read-only receipt extraction/audit below runs on lambda01 using the approved project `.venv` (no GPU use or new training). It checks the terminal hash binding, complete validation grid/counts, all500K finite consecutive loss/gradient/LR rows against the exact schedule, report crossings/slopes, and final comparison columns. It does not re-evaluate model predictions or validate other seeds. Raw streams/checkpoints remain local/untracked as in the launch receipt; hashes below identify the exact evidence, not a claim that raw data was pushed.

| C/gdn2_adamw_seed0 artifact | SHA256 |
|---|---|
| result.json | `e739643041f5ac60856d93817b08d5e95230c3051f6bc95e157d243b011a84c5` |
| validation.json | `fd358d4a3a4197dda82ea22a780f383a907be4358c20d202fa7725c880690557` |
| training.jsonl | `f568a23b2887388c216aeb04f90e35094eaffb5d60082e0a353651de0c73aa75` |

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import hashlib, json, math
from pathlib import Path
root = Path('experiments/expressivity_tasks')
c = root / 'results_invertible_stage11_gdn2_adamw_8m'
p = c / 'gdn2_adamw_seed0'
read = lambda path: json.loads(path.read_text())
for name, expected in {
    'result.json': 'e739643041f5ac60856d93817b08d5e95230c3051f6bc95e157d243b011a84c5',
    'validation.json': 'fd358d4a3a4197dda82ea22a780f383a907be4358c20d202fa7725c880690557',
    'training.jsonl': 'f568a23b2887388c216aeb04f90e35094eaffb5d60082e0a353651de0c73aa75',
}.items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected
v, result = read(p / 'validation.json'), read(p / 'result.json')
terminal = read(c / 'receipts/gdn2_adamw_seed0/terminal.json')
assert terminal == read(c / 'markers/gdn2_adamw_seed0.DONE')
assert terminal['exit_code'] == 0
assert terminal['result_sha256'] == hashlib.sha256((p / 'result.json').read_bytes()).hexdigest()
assert result['status'] == 'complete' and result['total_steps'] == 500000
assert result['stop_reason'] == 'safety_ceiling' and result['final'] == v[-1]
assert [a['step'] for a in v] == list(range(0, 500001, 500))
for a in v:
    assert len(a['positions']) == 16
    assert all(b['count'] == 512 and 0 <= b['correct'] <= 512 for b in a['positions'])
    assert sum(b['correct'] for b in a['positions']) == a['aggregate']['correct']
    assert math.isfinite(a['aggregate']['loss'])
with (p / 'training.jsonl').open() as stream:
    for n, line in enumerate(stream, 1):
        a = json.loads(line)
        lr = (1e-3 * n / 5000 if n <= 5000 else
              1e-5 + .5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (n - 5000) / 495000)))
        assert a['step'] == n and math.isclose(a['lr'], lr, rel_tol=1e-14)
        assert all(math.isfinite(a[k]) for k in ('loss', 'grad_norm', 'lr'))
assert n == 500000
first = lambda i, threshold: next((a['step'] for a in v if a['positions'][i]['correct'] >= threshold), None)
assert [first(i, 512) for i in range(6)] == [71000, 80500, 213500, 348500, None, None]
assert [first(i, 50) for i in range(6)] == [32500, 70000, 85500, 112000, 285000, None]
assert max(a['positions'][4]['correct'] for a in v) == 333
for start, expected in [(450000, .5929178800232965), (475000, .001990950226244344)]:
    rows = [a for a in v if a['step'] >= start]
    xs, ys = [a['step']/1000 for a in rows], [a['positions'][4]['correct'] for a in rows]
    xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
    slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    assert math.isclose(slope, expected, abs_tol=1e-12)
    print('P5 OLS start/count/slope:', start, len(rows), slope)
for stage, arm, expected, prefix, max_prefix, max_p5 in [
    ('11_gdn2_adamw', 'gdn2_adamw', [512]*4+[318,4]+[0]*10, 4, 4, 333),
    ('7b_stock', 'gdn2_uniform', [512]*4+[0]*12, 4, 4, 3),
    ('7_deepfirst', 'gdn2_fp32', [512,512,511,278]+[0]*12, 2, 3, 2),
    ('10_optimizers', 'adamw', [512]*4+[7,0,1]+[0]*9, 4, 4, 15),
    ('10_optimizers', 'sf', [510,0,1]+[0]*13, 0, 1, 2),
    ('10_optimizers', 'muon', [512]*3+[448]+[0]*12, 3, 3, 2)]:
    path = root / f'results_invertible_stage{stage}_8m' / f'{arm}_seed0'
    d, history = read(path / 'result.json')['final'], read(path / 'validation.json')
    assert [a['correct'] for a in d['positions']] == expected
    assert d['perfect_prefix_depth'] == prefix
    assert max(a['perfect_prefix_depth'] for a in history) == max_prefix
    assert max(a['positions'][4]['correct'] for a in history) == max_p5
    assert d['aggregate']['terminal_correct'] == d['aggregate']['post_suffix7_correct'] == 0
    print(stage, arm, expected, 'CE:', d['aggregate']['loss'])
print('PASS: X11 stream/endpoint/acquisition/slope and comparison audit')
PY
```

Validation on lambda01: the embedded read-only audit passed; `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_invertible_stage11_gdn2_adamw_8m.py -k 'not gpu'` passed **8 tests,1 GPU test deselected**. No new training, GPU evaluation or test-suite changes were needed for this report.

Scope is local single-card learned-solution expressivity research only: no resilient DiLoCo behavior, Frontier scheduler binding, production capability, capacity theorem or cross-seed claim. No distributed requirement IDs apply. Independent review remains required.
