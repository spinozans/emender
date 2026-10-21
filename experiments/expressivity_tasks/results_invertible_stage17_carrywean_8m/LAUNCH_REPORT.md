# X17 carry-wean launch receipt

## Preregistered, committed, pushed, THEN launched

- Immutable registration: **9af69e1317b422c87d15951ed16cbc367daab5b7**, pushed to
  origin/main before arming; registered 2026-10-21T19:53:32Z.
- Registration SHA256: `a1d6cabd337f66f61b2d3020f55880978b4fdb3c28422ba0b84b1ce427649909`.
  All 166 bound source files and 388 dependency files verified against live bytes;
  repository sources and registration also verified against committed HEAD.
- Exact supervisor-approved X6 Arm B format: P2..P64 carry slots ONLY, none
  before P1; 129 tokens padded to144, original66-position target mask. Eval
  ALWAYS neutral. One global linear p=.5→0 across1M, never stage-reset.
- Scratch E97 seed0, 9,901,056 parameters (including256 neutral parameters),
  four layers. X12 uniform AdamW+1M cosine recipe unchanged; predictions(a)/(b)/
  (c) registered verbatim before any GPU launch. No alternate arm authorized.

## First-free-card scheduling and native supervision

At 19:54:00Z the first broker poll returned **[4,5]**: X13 cards had freed while
CPU preparation ran. Scheduler chose the FIRST card, **4**, without taking any
waiting lease. Worker atomically acquired4 `--no-wait`, then spawned the trainer;
no GPU transfer/probe preceded acquisition. Card5 remains untouched. Broker
errors and malformed lists cannot admit a card; an acquisition race fails closed
without automatic retry. Polling/no-lease and lost-race paths were CPU-tested.

| Role | PID | /proc starttime | Host |
|---|---:|---:|---|
| Scheduler (launched, exited normally) | 3207772 | 267842598 | lambda01 |
| Leased worker | 3208316 | 267842870 | lambda01 |
| Trainer | 3208689 | 267843092 | lambda01 |
| Native completion/growth watcher | 3207964 | 267842775 | lambda01 |

Worker heartbeat every60s, releases only its own lease, terminates the trainer
process group on failure. Native watcher detached with `--interval 900 --report`;
writes full positional growth on every wake and terminal exit. Live supervisor
channel notified of admission, identities, current progress and wake command;
no WorksGood service/dispatcher was started or used.

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage17_carrywean --once --report
```

Exit2=pending, 0=hash-bound successful1M terminal/p0, 1=failure or broken process/
receipt. No retry, restart or outcome-dependent early stop is armed.

## Validation and first growth wake

- CPU tests: **76 passed, 7 skipped** across X17/X10/X12/X6/X7. The19 X17 tests
  all passed; skipped tests require GPUs. Receipt: `receipts/tests_prelaunch.txt`.
- Actual GPU launch passed initial neutral evaluation and backprop/update path.
  At 19:55:21Z, training log verified contiguous updates1..**2571**, EVERY logged
  p and LR matching registration. At2500, training p=.49875, eval p=0, all64
  positions have512 examples. Aggregate neutral CE8.294949; perfect prefix0;
  peak allocated memory774,778,368 bytes. Entry+latest checkpoint count2.
- Archived validation array hashes exactly match historical X12 seed0 T64.
  `receipts/launch_validation.json` binds initial source commit, pool archive
  hash, first/latest update rows, validation cadence and process liveness.
- `GROWTH_REPORT.md` is the wake snapshot, with P1..P64 latest/max/first50/first512,
  peak step/p, and explicitly contextual cold P1 timing (Muon horizons differ).

**RUNNING / RIGHT-CENSORED.** At this early wake neither material P1 nor P2+
composition has appeared; this is NOT a completed zero or a wean verdict.
Predictions(a)/(b)/(c) remain unresolved. In particular early material acquisition
would not by itself establish survival at zero crutch; report final1M separately.
Infrastructure failures are censored, not architecture evidence. The full1M
outcome is intentionally still pending.

Scope: standalone single-host/single-card expressivity, not resilient DiLoCo or
Frontier; resilient gap-matrix requirement IDs and Slurm partition/QoS evidence
are not applicable. Existing unrelated worktree changes were not modified or
staged. Runtime checkpoints/training logs remain local; bounded retention applies.
