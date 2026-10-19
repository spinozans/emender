# X7 — deep-first T16 + state-precision pilot: launch report

**LIVE, three scratch runs on three exclusive cards; native supervisor watcher armed.** Registered and committed **before launch** in `2d5a651a`; launched **2026-10-19 04:42:09 UTC** on lambda01. Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage7_deepfirst_8m/`. This is a launch receipt, not a learning result. The five existing X6 B trainers remain alive with their original PID/start identities; no X6 data, source or leases were changed.

## Pre-launch predictions (verbatim)

(a) Operator's horizon theory (ladder/starvation is structural): Run 1 learns carries at T16 (per-position structure shows P2+ substantially solved, no T1/T2 pre-training needed).

(b) Precision-wall theory (bf16 cannot pay y^2's per-step error-doubling beyond a few steps): Run 1 stalls shallow BUT Run 2 extends the perfect-prefix depth materially beyond bf16's ~3.

(c) Optimization theory (fixed-lr oscillation): later follow-up arm (lr anneal) - not in this pilot; note as registered follow-up.

`C/registration.json` contains these predictions, finalization timestamp, operational source hashes, exact scopes and preflight hashes. No annealing follow-up was launched. “Substantially” and “materially” are the operator's qualitative predictions, not post hoc numerical significance thresholds. Record the full position trajectories, not only a binary verdict. Perfect-prefix depth in the runner means consecutive positions from P1 with **512/512 joint-state predictions correct** on the frozen pool; report the counts too, because one error can truncate that statistic.

## Matched recipe and actual intervention

All runs start directly at **T16 from initialization**, seed0, no checkpoint loaded, empty ScheduleFree optimizer. **No ladder, no carry slots/supervision, no stagnation detector, no threshold stop**: **500,000 updates is the safety ceiling only**. Same X6/X3 task-sized tied input/output vocabulary3721 (p61), dim256, four layers, E97 expansion1 / GDN2 expansion8.625, MLP ratio2, batch32, lr1e-3, AdamWScheduleFree betas(.9,.95), wd.01, clip1, no warmup. Trainable counts **9,900,800 / 9,887,056**. GDN2 is real FLA GatedDeltaNet with `allow_neg_eigval=True`. Fresh procedural training batches never use the frozen validation pool. Actual T16 input length18 is causally right-padded to32 by the unchanged X3 helper, then sliced before loss/metrics. All16 driver targets contribute to training loss.

Every **500 updates**, including update0 and the terminal boundary, run a **complete 512-sequence frozen T16 validation pass** and record **P1…P16 joint/x/y counts, per-position CE loss**, aggregate/suffix/terminal metrics and perfect-prefix depth. The X6D `position_counts` instrument is reused; position totals reconcile exactly. The same seed0 T16 arrays (exact X3/X6 frozen-pool discipline) are shared across all three runs; pool bytes/hashes are archived. No test-set selection.

| Run | What remains bf16 | What is fp32 / implementation scope |
|---|---|---|
| 1 E97 bf16 | Projected q/k/v, decay/erase/write/output gates and recurrent output; legacy replay checkpoints/scratch and state-gradient storage | Parameters; **live within-scan state accumulator and arithmetic were already fp32**. Historical legacy launch tuning. |
| 2 E97 fp32 state | Same projected/gate tensors and recurrent output as Run1 | Existing `recurrent_state_precision='fp32'` knob: initial/final state, backward replay checkpoints/scratch and state-gradient buffers fp32; live register state/gradient arithmetic already fp32. Also selects fixed BH1/NW4 kernel geometry. Discarded eval/no-grad replay workspace remains legacy; live eval state stays fp32. |
| 3 GDN2 fp32 state | Same upstream projected/convolved q/k/v and raw alpha/beta logits; recurrent output cast back to bf16 before unchanged output norm/gate/projection | Pilot-local differentiable **sequential fp32** recurrence promotes those inputs before q/k normalization, alpha/beta transforms, decay, delta retrieval/write, state and readout arithmetic; recurrence autograd fp32, then gradients return through input casts. Parameters fp32. Upstream convolution/cache/projection/output forward bytecode is unchanged; only the two recurrence lookups are replaced in a private per-instance globals copy. |

**Interpretation constraints:** Run1 is not a wholly bf16 forward accumulator. Consequently Run1-vs2 does **not** isolate pure forward per-step bf16 error doubling: it tests the existing state-policy knob, including backward replay precision and launch geometry. GDN2 stock also accumulates state in fp32, but its chunk intermediates are bf16 and its fp32-input matrix dots use **TF32**, not full fp32 arithmetic. Thus Run3 provides real numerical relief, not a renamed stock fp32 accumulator; it also changes arithmetic reassociation and normalized/gate precision. Projections remain quantized in both precision interventions. A depth difference alone is not proof of the proposed mechanism.

### Matched E97 evaluation backend and parity caveat

Supervisor approved **`fused_inference=True` for BOTH X7 E97 arms**, so eval uses the same sequential Triton recurrence as training. The fp32 knob rejects eager eval; using different backends between arms would confound this comparison. **X6 E97 eval ran eager**, so cross-experiment E97 comparisons must name this backend difference; the training recipe is unchanged.

`C/receipts/e97_eval_backend_parity.json` compares full-model eager-bf16 and fused-bf16 outputs on identical scratch weights and32×18 input tokens: relative L2 **0.0025615**, max absolute logit difference **2.0** versus max logit274, **100% argmax agreement**, CE **251.16718 eager / 251.15837 fused**. This passes the registered preflight gate (relative L2<1%, argmax agreement≥99%) but is not bitwise equality or trained-checkpoint parity. The initial GPU smoke failure that exposed the eager-eval incompatibility is preserved in `gpu_smoke.txt`; corrected finite forward/backward/eval checks are in the final suite and `gpu_smoke_retry.txt`.

### GDN2 equivalence guard (before launch)

Both **T32 and T96**, batch2, actual2-head key128/value1104 geometry, identical **fp32 input tensors**, negative-eigenvalue beta, fp32 initial-zero state, full outputs/final state, and random VJPs for **q/k/v/alpha/beta/A_log/dt_bias** were compared against the stock FLA chunk kernel. All pass the supervisor-approved **relative L2≤1% AND max-absolute-error/stock-maximum≤1% for every compared tensor**. Exact values: `C/receipts/precision_equivalence.json`. Output relative L2 is approximately0.0015; state approximately0.0008; worst VJP relative L2 approximately0.0066. The stock kernel's explicit TF32 dots prevent a tight IEEE-fp32 comparison even with fp32 tensor inputs. The adapter separately matches FLA's independent naive fp32 sequential reference, including state layout/nonzero initial state and all gradients, at **rtol1e-5/atol1e-6 outputs/state and rtol2e-4/atol2e-5 gradients**.

The initially attempted strict elementwise stock-kernel tolerance failed (max output error0.000167 at T32); its unmodified stdout is retained in `tests_initial.txt`. The supervisor explicitly approved the normwise guard after inspecting this TF32-scale residual; no installed FLA kernels were edited or IEEE overrides installed. This supports mathematical delta-rule equivalence, not bitwise identity or a claim that trained trajectories are identical. The failed receipt contains pytest-generated trailing whitespace; it is retained as evidence rather than rewritten.

## Verified launch and positional baseline

`C/receipts/launch_verification.json` and reproducible `verify_launch.py` validate all worker/trainer start identities, **worker PPID1/SID=worker PID**, trainer parent/session identity, unique active leases, committed-source hashes, source/dependency hashes, identical frozen arrays, initial empty optimizers and identical E97 initial weights. Baseline checkpoint hashes were checked against payloads. Each stream was finite/consecutive and T16-only. Snapshot time **04:43 UTC**:

| Run | GPU | Worker / trainer PID | Observed update | Operational updates/s | Remaining to500K at this rate |
|---|---:|---|---:|---:|---:|
| E97 bf16 | 2 | 947974 / 948287 | 2,508 | 58.42 | 2.37h |
| E97 fp32 state | 7 | 947976 / 948321 | 2,832 | 50.50 | 2.73h |
| GDN2 fp32 state | 4 | 947978 / 948306 | 297 | 5.76 | 24.10h |

Rates run from update100 to the listed observation and include intervening eval/checkpoint cost. They are early measured operational rates, **not ETAs or learning predictions**; GDN2 is an eager differentiable sequential reference and deliberately slower than the stock fused chunk kernel. Contention and later checkpoint/eval costs may change rates. All three overlap; do not add their wall times. No discretionary walltime stop or restart is scheduled.

**Update0 baseline: joint-state correct counts, each position out of512.** All three have **2/8192 =0.024414% aggregate**, perfect-prefix depth0; chance full-state reference is1/3721≈0.026874%. High initial CE reflects the scratch tied embedding, not a learning finding.

| Run | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E97 bf16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 250.976103 |
| E97 fp32 state | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 250.976553 |
| GDN2 fp32 state | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 243.960819 |

Immutable early configs/validation/data/checkpoint indexes and finite training prefixes are in `C/receipts/launch_snapshot/`. Live streams are not committed mid-write. `full_launch/` archives executed operational sources, installed FLA dependency sources, packages, launch base commit/diff, GPU and disk snapshots. Entry/best/latest checkpoints are retained under the existing X6 bounded-pruning policy below400GiB free; terminal becomes latest. About285GiB was free before launch; no historical experiment files were deleted.

## Watcher armed → growth report on wake

**Supervisor confirmed native supervised watcher armed: PID952860, process `proc_b85f`**, command `.venv/bin/python scripts/watch_invertible_stage7_deepfirst.py --interval 300`; first healthy pending poll **2026-10-19T04:44:08Z**. Its exit wakes the supervisor session natively. Confirmation is receipted in `C/receipts/watcher_armed.json`; an independent `--once` returned2 (healthy/pending). No WorksGood machinery was started or used.

Each detached worker publishes atomic `receipts/NAME/terminal.json` plus `markers/NAME.DONE` or `.FAIL`. Watcher exits0 only after all three successful terminal/result receipts; failure or dead/reused worker/trainer identity returns nonzero. Re-arm after host reboot; this is not host-failure recovery. On wake, inspect markers/checkpoint/source identities and all per-position validation trajectories, then write the **growth report** against predictions(a)/(b); retain(c) as unlaunched follow-up. Do not infer prediction support from launch or replace position evidence with aggregate accuracy.

## Validation / scope

**107 tests passed**, including **9 X7 tests** covering paired initialization/knob/backend setup, instance-local recurrence replacement and unchanged parameter identities, strict independent fp32 reference outputs/state/gradients, full-fp32-input stock equivalence at two sequence lengths (including cross-chunk), real-model finite GPU forward/backward/eval and backend parity, safety-only continuation even at100% accuracy/no ladder, positional reconciliation, and pending/DONE/FAIL watcher states. Existing X3/X5/X6/task/position tests also pass. Commands and exact output are in `C/receipts/tests_prelaunch.txt`; exit0 is separately receipted. `bash -n` and Python compile checks pass; current working-source `git diff --check` passes. The failed strict preflight and corrected preflight are both retained.

Architecture-authority applicability: **R16 evidence discipline only** (`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` / `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, ADR-003 crosswalk). These are local single-GPU expressivity probes, **not production or resilient DiLoCo training**. No systems acceptance, Slurm Partition/QOS binding or scaling authorization is claimed. Independent review remains supervisor-owned; next scientific deliverable is the growth report on native watcher wake.
