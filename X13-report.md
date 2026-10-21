# X13 v2 — E97 T64 zero investigation (audit first)

**STEP 0 verdict: CLEAN.** The hash-pinned four-perfect T16 champion reads exactly at its receipted levels through the suspect stage12 path; CPU T64 construction agrees token-by-token. Trusted-stage10 T64 zero-shot retains **P1–P4=512/512 each**, P5=8/512. The audited E97 data path is exonerated. **No mechanism claim.**

**STEP 1 status: preregistered, not yet launched.** Seeds1/2 use the exact X12 recipe at the fixed1M ceiling; final replication verdict pending. This section will be updated after audited admission.

Evidence root **R/**: `experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m`. Task authority: `/home/erikg/emender-scratch/fix-fanout/taskX13-zerocheck-v2.md` (supersedes v1); [registration](experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/REGISTRATION.md), executable [registration.json](experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m/registration.json).

## 1. STEP 0: ordered runner-path audit

Execution order was (a), then (b), then (c); no seed-replication training preceded the audit. The original X12 E97 interpretation was SUSPENDED pending this audit, without rewriting its receipted observations. The audit excludes this input-construction explanation; it does not identify a cause for the learning outcome or establish an architectural capacity bound. Earlier state/memory/mechanism explanations are not endorsed here.

Champion: X10 AdamW seed0 aggregate-best update **468000**, ordinary live parameters; checkpoint `step_468000.pt`, SHA256 **`0501a81ff50e34017a973f400eec5975faaf5bd9d84df64d6fa84b51ed6c9fa7`**. The file hash, original result pointer, checkpoint step, exact historical validation arrays, strict state-dict loading and historical X10/X12 model/task/kernel source hashes were checked before evaluation. Checkpoint weights were not trained or changed.

### (a) Suspect stage12 path on identical frozen T16 validation

Used `stage12.build_model("e97")`, `stage12.evaluate` and its actual `x7.forward_batch` loader/padder, bf16 autocast and legacy Triton recurrence. Only the evaluator's expected-depth constant was scoped to16 for its loss reshape/assertion; no inputs, masks, targets, model or forward function were modified. The registered broken example was P1<400/512; implementation also fails closed if another trained position is<400 and requires four exactly-perfect columns to proceed.

- P1–P16 through stage12: `512,512,512,512,14,0,0,0,1,0,0,1,1,0,0,1`.
- Original same-checkpoint P1–P16: `512,512,512,512,14,0,0,0,1,0,0,1,1,0,0,1`.
- **Exact all-position count match**, not just P1; aggregate CE 5.952114731073. Not instrument-broken under the registered test.

### (b) CPU T64 input/target equivalence

Compared the actual stage12/stage10 loader and padding paths on frozen rows **0,1,2,7,63,127,255,511**. Only `.cuda()` became identity and CUDA autocast became a null context; a capturing model recorded every padded token. Both runners resolve to the **same task class, frozen-pool function, forward function and padder**. Every input token, target and mask agrees, with 66 real tokens, initial x/y at0/1, target/driver indices2..65, and zero right-padding to80. The logits are truncated to66. Independent scalar integer arithmetic checks every target: `(x,y),c -> (y,(x+y*y+c)%61)`, joint index61*x+y; no target feedback into input.

**No divergent example.** The receipt contains both paths' padded inputs, targets and masks and all66 decoded token roles/states for each of eight examples, plus full512-array hashes. This checks construction/padding, not learned-state dynamics or all possible length-dependent kernel behavior.

### (c) Unchanged T16 champion, zero-shot T64 via trusted stage10

Used trusted `stage10.x7.evaluate` / `forward_batch`, only scoping its positional loss-reshape depth to64. All512 examples are the historical X12 seed0 frozen T64 validation pool. No gradient updates, checkpoint selection, new test-set score or curriculum. The table lists correct/512; P17+ did not exist in the T16 read.

| Position | stage12 T16 | stage10 zero-shot T64 | T64 CE (nats) |
|---|---:|---:|---:|
| P1 | 512 | 512 | 7.088235000185517e-05 |
| P2 | 512 | 512 | 0.0003612684299350022 |
| P3 | 512 | 512 | 0.0025243244801664844 |
| P4 | 512 | 512 | 0.046687882058904506 |
| P5 | 14 | 8 | 4.77377064479515 |
| P6 | 0 | 0 | 8.213694922626019 |
| P7 | 0 | 0 | 8.219661095179617 |
| P8 | 0 | 0 | 8.221732942387462 |
| P9 | 1 | 0 | 8.220581140369177 |
| P10 | 0 | 0 | 8.22191508114338 |
| P11 | 0 | 0 | 8.221998024731874 |
| P12 | 1 | 0 | 8.221199167892337 |
| P13 | 1 | 1 | 8.221330625936389 |
| P14 | 0 | 1 | 8.221408745273948 |
| P15 | 0 | 0 | 8.22139072790742 |
| P16 | 1 | 0 | 8.222418563440442 |
| P17 | — | 0 | 8.221367627382278 |
| P18 | — | 0 | 8.221713481470942 |
| P19 | — | 0 | 8.22221114858985 |
| P20 | — | 0 | 8.22255945764482 |
| P21 | — | 0 | 8.222210416570306 |
| P22 | — | 1 | 8.221797248348594 |
| P23 | — | 1 | 8.22114591114223 |
| P24 | — | 1 | 8.221211321651936 |
| P25 | — | 0 | 8.221605459228158 |
| P26 | — | 0 | 8.221545230597258 |
| P27 | — | 0 | 8.222002141177654 |
| P28 | — | 0 | 8.221783950924873 |
| P29 | — | 0 | 8.22224228642881 |
| P30 | — | 0 | 8.222206613048911 |
| P31 | — | 0 | 8.221477951854467 |
| P32 | — | 0 | 8.221885494887829 |
| P33 | — | 0 | 8.220746310427785 |
| P34 | — | 0 | 8.221245352178812 |
| P35 | — | 0 | 8.221398824825883 |
| P36 | — | 1 | 8.221825502812862 |
| P37 | — | 0 | 8.221958734095097 |
| P38 | — | 0 | 8.221986342221498 |
| P39 | — | 1 | 8.221893357113004 |
| P40 | — | 0 | 8.221678517758846 |
| P41 | — | 0 | 8.22119328379631 |
| P42 | — | 0 | 8.221347300335765 |
| P43 | — | 0 | 8.222334336489439 |
| P44 | — | 0 | 8.22180919162929 |
| P45 | — | 0 | 8.221923816949129 |
| P46 | — | 0 | 8.221656566485763 |
| P47 | — | 0 | 8.221554439514875 |
| P48 | — | 0 | 8.222220035269856 |
| P49 | — | 1 | 8.22170382924378 |
| P50 | — | 0 | 8.221123486757278 |
| P51 | — | 0 | 8.221560088917613 |
| P52 | — | 1 | 8.221441425383091 |
| P53 | — | 1 | 8.22158526442945 |
| P54 | — | 0 | 8.22149303369224 |
| P55 | — | 0 | 8.222401758655906 |
| P56 | — | 0 | 8.221892155706882 |
| P57 | — | 0 | 8.221752941608429 |
| P58 | — | 0 | 8.22198573127389 |
| P59 | — | 0 | 8.221613882109523 |
| P60 | — | 0 | 8.221635522320867 |
| P61 | — | 0 | 8.222645005211234 |
| P62 | — | 1 | 8.221568185836077 |
| P63 | — | 0 | 8.221464389935136 |
| P64 | — | 0 | 8.221765579655766 |

**Observed retention:** all four trained positions remain exactly512/512 on the longer sequences; no P5+ reaches50. Thus the learned trained-position computation executes at T64 under this probe. This is not evidence that the champion learned a64-step chain, nor a state-capacity diagnosis or an explanation of seed0 training. P5=8 versus T16's14 compares different frozen examples/lengths, not a paired degradation estimate.

## 2. STEP 1: exact-X12 scratch replication, seeds1/2

Four-layer E97/dim256/9,900,800 parameters, fp32 parameters/bf16 legacy recurrence, p61/vocab3721, 64 drivers/66 tokens right-padded to80, batch32; uniform masked CE, clip1, ordinary live-parameter AdamW, betas(.9,.999), eps1e-8, all-parameter decay.01; warmup1..5000 to1e-3, cosine to1e-5 at1M. The unchanged X12 `build_model`, `make_optimizer`, `train_update`, `learning_rate` and `evaluate` callables are reused. No champion weights loaded by either trainer.

Independently seeded torch initialization/procedural training order and historical seed-specific frozen512-example validation pools; arrays hashed/archived, never used as training examples. Validate all64 positions at0 and every500. Ceiling1M only: no stagnation, threshold or wall-time stop, no automatic retry. Entry/best/latest retention stays bounded to3 checkpoints; infrastructure failures are censored.

### Prelaunch supervisor ruling: two distinct thresholds

The task's "material composition anywhere" needed a positional definition. Before any seed1/2 training, the supervisor explicitly ruled:

1. **Marginal escape:** ANY P1..P64 >=50/512 at ANY validation. **ANY replication seed** meeting it means seed0's learned-nothing zero was lottery; re-scope. P1-only success re-scopes to visible-position learning without material carry at T64.
2. **Composition:** separately, ANY P2..P64 >=50/512 at ANY validation. P1-only learning is NOT composition.
3. BOTH seeds with NO position>=50 => the learned-nothing zero replicates as3/3 (with historical seed0), bounded to this finite recipe/ceiling. Partial zeros are right-censored. Report P1 and P2+ separately, all64 counts/maxima/first50/first512, and fixed-final perfect prefixes.

No seed outcome or replication verdict is claimed yet. See `R/GROWTH_REPORT.md` for live per-position columns after launch.

## 3. Validation, reproducibility and supervision

- CPU regression: `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_invertible_stage13_zerocheck.py tests/test_invertible_stage12_separation_8m.py tests/test_invertible_stage10_optimizers_8m.py -k 'not gpu'` — **45 passed,4 GPU cases deselected** (`R/receipts/tests_prelaunch.txt`). Eleven X13 tests cover historical paths and independent target decoding, deliberate token/prefix/target/mask corruption, exact-X12 callable reuse, audit gating/tampering, scratch seed/pool/order/ceiling/retention, worker gating/owner lease/heartbeat/release, hash-bound terminal watcher and P1-versus-composition classification.
- Actual leased GPU audit command: `.venv/bin/python -m scripts.audit_invertible_stage13_zerocheck` — completed (a)/(b)/(c) in order, console in `R/receipts/audit_console.txt`; source and environment recorded in `R/audit/registration.json`.
- Audit receipts: `a_stage12_champion_t16.json`, `b_cpu_t64_equivalence.json`, `c_stage10_champion_zeroshot_t64.json`, `registration.json`; hashes bound by `R/audit/verdict.json`.
- Sources, dependencies, frozen pools and audit verdict are bound by immutable preregistration. Registration and its bound sources must be committed before a worker can lease or train.
- Native supervisor command (exit0 both successful1M terminals; exit1 failure; `--once` exit2 pending):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage13_zerocheck --interval 900 --report
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage13_zerocheck --once --report
```

On wake, regenerate growth table and update this report with final or explicitly censored results. No WorksGood service. Local single-card expressivity only: no resilient DiLoCo/Frontier/distributed training behavior or requirement IDs are tested or claimed. Independent review is required.
