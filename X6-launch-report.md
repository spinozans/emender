# X6 — true bottom-up curriculum: launch report

**LIVE: twelve detached workers, eight training and four awaiting exclusive GPU leases on lambda01.** Source and protocol were committed before calibration/launch in **cd519578**. Launch: **2026-10-17 19:34:26 UTC**. This is a launch receipt, not a completed growth comparison. Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/`.

## Registered experiment and supervisor-approved details

Both curricula independently initialize scratch **E97-MLP and real FLA GDN2**, seeds **0/1/2**, the X3 approximately-8M recipe (actual A counts **9,900,800 / 9,887,056**), lr **1e-3**, batch32, ScheduleFree AdamW betas(.9,.95), wd.01, clip1, no warmup, bf16 autocast/fp32 parameters; GDN2 `allow_neg_eigval=True`. No historical weights were resumed. Exact X3 frozen validation pools per seed/depth are shared between models/curricula; training remains fresh procedural data, never those pools. Both use the **1→2→4→8→16→32→64→128** ladder, **≥90% full-state validation** threshold, observed every500 updates. Current plus **every lower depth** is evaluated at entry, every500, and exit. Growth receipts record each stage's steps-to-threshold, accuracy/retention, and ceilings encountered. All receipts print the uniform marginal **ln(3721)=8.221747728 nats** and suffix-only reference **1/3721=0.0268744961%**; the suffix theorem is not a recurrent-model capacity statement.

**Arm A — true curriculum:** threshold advancement carries the current weights **and optimizer**. There is **no stagnation detector or stagnation kill**, no discretionary walltime kill, **500,000 updates/stage** and **4,000,000 total** safety ceilings. A stage ceiling without threshold records **`ceiling_non_threshold`**, restores its **best-validation checkpoint's weights AND full optimizer**, and proceeds to the next depth anyway. Already consumed data/RNG and global compute are **not rewound**; this avoids repeating procedural batches or undercounting compute. Threshold takes priority at a simultaneous safety boundary. Global safety exhaustion terminates honestly as `global_safety_cap`; completion otherwise means the stage ladder was traversed, not that every threshold was reached.

**Arm B — training-only carry supervision:** before every driver after the first, insert one fixed-position carry slot. Independently per example/slot, insert the **true previous** joint-state id `61*x+y` with probability **p=.5*(1-stage_step/500000)**; otherwise insert the neutral token. The schedule resets at each stage and reaches zero at its ceiling. **Evaluation always supplies neutral slots (p=0)**; neither current nor future targets enter inputs. Slot positions/sequence lengths are identical for supervised and unsupervised B examples. At depthT, A length is T+2; B length is 2T+1. Driver logits are gathered back to original positions before loss/metrics, so masks, suffix-driver indices and terminal scoring retain their meanings. Stage advancement/ceilings and retention evaluation use **neutral-only evaluation**, not teacher-forced accuracy. At T1 there is no carry slot.

Supervisor explicitly approved an **input-only reserved neutral id3721**, beyond the 3721 state ids. Thus B has **3722 input ids but still exactly3721 output classes**, preserves the original tied state embedding/output, and adds only a separate256-parameter neutral embedding: B counts **9,901,056 / 9,887,312**. This approved exception to the task-sized input vocabulary is registered, not silently hidden. The fixed-slot format, per-stage linear anneal, scheduling below, and bounded checkpoint retention were approved through the live coordination channel. A/B also differ in slot structure and inference token count; any later comparison must acknowledge that training-format intervention. There is no mechanistic or architecture superiority finding at launch. Interpretive claims await actual Arm A **and** Arm B growth evidence.

## Verified detached launch

`C/receipts/launch_verification.json` records the snapshot below; all twelve worker `/proc` start identities, **PPID=1 and SID=worker PID** were checked. Each active trainer is a child in its worker's isolated session. Eight distinct live heartbeat leases bind their owning worker identities. B seeds1/2 have live isolated workers but deliberately no trainer yet. The lease wait timeout is **90days**, not a training walltime stop; no automatic restart is scheduled.

| Curriculum/model/seed | GPU | Worker / trainer PID | Observed T1 updates | Wall updates/s |
|---|---:|---|---:|---:|
| A/E97/0 | 0 | 2445505 / 2446051 | 3,216 | 51.27 |
| A/GDN2/0 | 2 | 2445507 / 2446085 | 1,765 | 28.73 |
| A/E97/1 | 1 | 2445510 / 2446069 | 3,330 | 52.90 |
| A/GDN2/1 | 3 | 2445514 / 2446101 | 1,809 | 29.47 |
| A/E97/2 | 5 | 2445517 / 2446135 | 3,436 | 54.55 |
| A/GDN2/2 | 4 | 2445522 / 2446117 | 1,775 | 28.60 |
| B/E97/0 | 6 | 2445525 / 2446152 | 3,158 | 49.67 |
| B/GDN2/0 | 7 | 2445531 / 2446170 | 1,754 | 28.41 |
| B/E97/1 | queued | 2446203 / not started | — | — |
| B/GDN2/1 | queued | 2446205 / not started | — | — |
| B/E97/2 | queued | 2446208 / not started | — | — |
| B/GDN2/2 | queued | 2446212 / not started | — | — |

Rates use update100 through each listed update, excluding startup/JIT before update100 but including intervening evaluation/checkpoint costs. They are operational throughput, **not steps-to-learning**. All eight streams were checked finite and consecutive. Config/registration/source hashes, all eight stage-entry checkpoint hashes and payloads (model/full optimizer/NumPy/Torch/CUDA RNG), all archived validation array hashes, and same-seed base-pool equality were checked. Snapshot copies in `C/receipts/launch_snapshot/NAME/` freeze configs, validation, transitions, data receipts/NPZ, checkpoint metadata and rate endpoints. Live training directories/consoles are intentionally not committed mid-write. `C/receipts/verify_launch.py` contains the reproducible checks.

`C/receipts/full_launch/` archives executed runner, shared task/model/checkpoint helpers, launcher, lease broker, watcher, source commit, tracked diff at launch, packages, GPU and disk snapshots. `registration.json` binds all operational sources including the separate calibration source. Full launch source was clean relative to its committed registration.

## Wall-clock planning — measured costs, not a one-day promise

`C/receipts/throughput/throughput.json` measures all **32 curriculum/model/depth combinations**, each with10 warmup+30 timed updates on a separately leased GPU released before launch. No calibration scores become learning evidence. Costs include all-lower-depth neutral-only evaluation and checkpoint save/hash every500 (conservative versus rolling every1000). For each combination below, charge the **slowest calibrated depth (T128)** for all4M updates, and inflate the whole cost by isolated-T1 training rate / slowest live-T1 operational rate when above1. This conservatively overcharges shallower stages **conditional on measured costs**, not on guaranteed future throughput.

| Curriculum/model | Measured slowdown | 500K stage planning cost | 4M run planning cost |
|---|---:|---:|---:|
| A/E97 | 1.225× | 7.87h | **62.98h** |
| A/GDN2 | 1.075× | 5.65h | **45.22h** |
| B/E97 | 1.182× | 11.98h | **95.81h** |
| B/GDN2 | 1.092× | 5.64h | **45.11h** |

At T1 rates alone4M implies about21.7/38.9h (A E97/GDN2) and22.4/39.1h (B), which omits deeper costs. The eight initial runs overlap; do **not** add their individual wall times. Four queued workers can all acquire by the fourth initial completion if these planning costs hold (about45.22h), then the longest queued B run adds95.81h: **approximately141.03h / 5.88days for the full12-run matrix**, around **2026-10-23 16:36 UTC** if every safety budget is consumed at these modeled costs. Earlier threshold crossings may shorten this; future GPU/CPU/disk contention, checkpoint frequency, JIT, host availability and throughput drift may lengthen it. This is **not** an ETA guarantee or a learning prediction. The task's approximately-one-day intuition is not supported by these conservative measured deep-stage/queue projections. Exact inputs/formula: `C/receipts/projections.json`.

## Disk and checkpoint policy

Only about457GiB was free before calibration (about453GiB at early launch). Supervisor therefore approved **X6-specific bounded retention**: below **400GiB free**, preserve stage-entry, stage-exit, each stage's best, and current latest checkpoints only; no permanent every10K set. Atomic checkpoints are written on stage entry/exit, best-validation improvement and rolling every1000. Every deletion produces `pruning.jsonl` with removed paths, retained hashes and free space. All per-update metrics remain in compact JSONL; checkpoint payloads remain local/Git-ignored. Pruning and all protected classes are tested, including removal of unprotected10K checkpoints. No historical experiment files were edited or deleted.

## Markers → supervisor-side watcher → native wake

```bash
cd /home/erikg/emender
.venv/bin/python scripts/watch_invertible_stage6_truecur.py
```

Default root is **C/**, interval900s, log `C/receipts/completion_watcher.log`. Workers atomically publish `receipts/NAME/terminal.json` and **`markers/{A,B}_{e97,gdn2}_seed{0,1,2}.DONE`** or **`.FAIL`**. The watcher requires **all12** DONE+result receipts for exit0; any FAIL or dead/reused worker/trainer identity without a terminal marker returns nonzero after publication grace. Queued workers legitimately lack trainers. `--once` returned **2, healthy/pending**, at19:36:17 UTC (`C/receipts/watcher_once.txt`). A detached watcher cannot wake the native supervisor session: the supervisor must run this watcher in its supervised session context. This worker sent the explicit arming command and pending/rate updates through live coordination; **native wake arming is not claimed without supervisor confirmation**. Re-arm manually after host reboot.

On wake: verify terminal markers and payload/source identities, inspect all12 growth/retention curves and ceiling events, then write the growth report. Do not interpret an unstarted queued run as Arm B evidence.

## Validation and scope

**159 tests passed** (`C/receipts/tests_prelaunch.txt`), including **16 new X6 tests**: eight-depth/12-run registration, no stagnation behavior, threshold priority, invalid observations, carry previous-state/no-current-target mapping, p=0 target independence, anneal endpoints/reset, input-only neutral/tied3721-class output, neutral-only evaluation and original driver-index scoring, actual weights/full-ScheduleFree optimizer restoration without RNG rewind, threshold/stage/global paths, continuation after ceilings with lower-depth retention, bounded pruning, and12-run watcher pending/DONE/FAIL behavior. One initial test used a nonexistent generic optimizer `step` field; it was corrected to assert ScheduleFree's actual group `k` counter and **all optimizer tensors**, then the full suite passed. The original failed test receipt is retained transparently. One earlier pytest invocation named a nonexistent test path and collected no tests; the final receipt uses the correct instrument-test filename. `bash -n`, Python compile checks and `git diff --check` passed before launch.

Architecture-authority scope: the relevant ADR-003/R16 traceability sections of `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and `docs/RESILIENT_DILOCO_GAP_MATRIX.md` were consulted. **R16 evidence discipline only**: these local single-GPU task probes are not production E97 or resilient DiLoCo; no systems gate, Slurm Partition/QOS binding, elastic/async conformance or scale authorization is claimed. Independent reviewer acceptance remains **supervisor-owned and pending**. The next scientific deliverable is the growth report on supervised watcher wake, not a launch-time capability interpretation.
