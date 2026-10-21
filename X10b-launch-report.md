# X10b — higher-LR hybrid-Muon sweep (launched)

**LIVE; seed0 only, lambda01. No optimizer-quality verdict.** Registration,
source, tests and frozen baseline snapshots were committed and pushed as
**`502329abbf96018616529807bbd78f771091bb56` before either launch**.
Registered UTC **2026-10-21T01:06:25Z**, launches at approximately01:07Z.
Canonical root (**C/**):
`experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/`.

| Arm | LR peak -> cosine floor | Card | Worker PID / starttime | Trainer PID / starttime | Audited finite updates |
|---|---|---:|---|---|---:|
| muonlr3x_seed0 | 3e-3 -> 3e-5 | 3 | 1510500 /261080710 | 1529535 /261080964 | 1,526 |
| muonlr10x_seed0 | 1e-2 -> 1e-4 | 4 | 1528524 /261080952 | 1546242 /261081199 | 1,495 |

Immutable audit snapshot: **2026-10-21T01:08:32Z**. Workers have PPID1;
trainers are children of their respective detached workers. Exclusive live
leases and PID/starttime identities checked. Cards0/1 were never used; X12's
pair and W4's busy card2 were not disturbed. Only physically idle, unleased
cards3/4 admitted. The standalone lease broker was reused, not WorksGood
service/dispatcher/completion machinery. No scheduler submission was made.

## Registered design and predictions

See [`C/REGISTRATION.md`](experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/REGISTRATION.md)
and executable [`C/registration.json`](experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/registration.json).
Only LR scale differs from the X10 hybrid-Muon implementation: both hybrid
parameter groups receive3x or10x the historical schedule, once per update.
Warmup updates1..5000; cosine floor at500,000. Actual LR logged every update.
The adapter delegates to the unchanged X10 loop and `hybrid_muon.py`; no
historical registered source was edited.

Scratch T16 deep-first E97,9,900,800 parameters (historical8M-class), dim256,
four layers, p61/vocab3721, batch32, uniform target CE, clip1, fp32 parameters
and bf16 autocast, legacy state/sequential Triton eval. Same seed0 torch
initialization, procedural numpy batches and X10 frozen512-example validation
pool. All P1-P16 evaluated every500 updates; ordinary live weights. Same
semantic36-matrix routing, auxiliary AdamW, approximate RMS matching,
Nesterov/momentum0.95, five FP32 NS steps and all-parameter decay0.01,
including ignored no-weight-decay tags. No ladder/carry/continuation or tuning.
One seed per arm; **500K safety-only ceiling**, not early-stop selection.

Predictions registered before launch:

- **(a) LR-UNDERSCCALED:** materially better P4 consolidation than historical
  Muon448/512 (e.g. >=508/512 at500K), or earlier same-threshold P4 acquisition,
  supports an LR-scale artifact for this learned solution; record which
  multiplier wins. Separate transient crossings from fixed-final retention.
- **(b) SCALE-RIGHT:** P1-P3 end degradation (any below512/512) or numerical
  instability at3x/10x supports1e-3-class for this tested hybrid/scale, with the
  corresponding method-intrinsic lag hypothesis. This conditional one-seed
  diagnostic is not a universal causal verdict; infrastructure censoring is
  not optimizer instability. Mixed/neither outcomes remain inconclusive.
- **(c) Both baselines:** each higher-LR arm gets a comparison with exactly
  **three outcome columns: higher-LR arm | X10-muon@1e-3 | X10-adamw@1e-3**,
  plus position/readout labels. Print all P1-P16 first>=50 and first512 crossings
  in updates/elapsed seconds, independent maxima and fixed-final counts.
  X10 Muon P3-perfect281,500 is25K earlier than AdamW306,500; P4-material348,500
  is39.5K later than AdamW309,000. Muon P4-perfect is censored at500K;
  AdamW first512 is391,500. Baseline endpoint P1-P4 counts are
  `[512,512,512,448]` and `[512,512,512,512]`, respectively.

Counts are joint-state correct out of512, not coordinate accuracy. Perfect
prefix is contiguous. The readout prints marginal **1/3721=0.0268744961%**,
CE **ln3721=8.221747728 nats**, terminal P16/512 and suffix7-restricted
P9-P16/4096. The suffix-only floor concerns observers missing >=2 iid drivers,
not prefix-aware recurrent models. Learned-solution scope only; no capacity
bound, broad optimizer ranking, cross-seed robustness or production claim.

## Prelaunch validation and immutable launch evidence

- CPU: **26 passed,6 GPU cases deselected** across X10b and X10 regressions.
  X10b tests exhaustively check every scheduled update0..500K for both groups,
  auxiliary propagation at endpoints, exact historical initialization/routing/
  frozen pools, delegated safety-only stopping and per-update LR logging,
  global restoration, retention and strict watcher hash/terminal bindings.
- Leased-card disposable seed99 GPU tests: **2 passed,7 deselected**. Both
  higher-LR arms perform finite full-model updates, including their peak LR,
  and complete all-position validation; no confirmatory-pool score selection.
- Card-specific25-update no-score probes: **bit-exact next-update reload** for
  both arms (max absolute difference0). Each two-model peak allocation is
  **464,979,456 bytes**. Registered wall ceiling **177,738 seconds per arm**
  (at least historical Muon bound), memory **8GiB per arm**. Timeout/memory
  infrastructure failure is censored; no automatic retry/restart. External
  deadline adds120s cleanup; TERM then30s/KILL. Entry + aggregate-best +
  latest/final retained, with bounded atomic staging and deletion manifests.
- Executable registration verifies **184 source/artifact bindings and389
  installed dependencies**, both current bytes and committed project blobs.
  Parent registration, pool arrays, parameter routes, probes/tests, exact
  historical baseline trajectories and new operational code are bound.
- `C/receipts/verify_launch.py` and `launch_verification.json` check the
  prelaunch commit ancestry/config, source identity, leases/processes, same
  frozen arrays, **bit-identical initial weights to X10-muon seed0**, empty
  initial optimizer, per-position denominators and consecutive finite update
  rows with exact actual higher LR. Both entry count vectors are
  `1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0`. Immutable config/data/validation/training
  prefixes with hashes are in `C/receipts/launch_snapshot/`; mutable streams
  and binary checkpoints were not staged.

Commands used (cwd repository root; lambda01 project `.venv`, not Frontier):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_invertible_stage10b_muonlr_8m.py tests/test_invertible_stage10_optimizers_8m.py -k 'not gpu'
# Exclusive standalone lease restricted to card3; disposable GPU checks:
.venv/bin/python -m pytest -q tests/test_invertible_stage10b_muonlr_8m.py -k gpu
# Separate exclusive leases restricted to cards3 and4 respectively:
.venv/bin/python -m scripts.invertible_stage10b_control probe --arm muonlr3x
.venv/bin/python -m scripts.invertible_stage10b_control probe --arm muonlr10x
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.invertible_stage10b_control register
# Commit+push502329ab, verify committed registration, then:
.venv/bin/python -m scripts.invertible_stage10b_control launch --arm muonlr3x
.venv/bin/python -m scripts.invertible_stage10b_control launch --arm muonlr10x
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m/receipts/verify_launch.py
```

The initial direct audit invocation without `PYTHONPATH=.` failed import before
reading any artifacts; the corrected command above passed. Watcher `--once`
returned2 (healthy pending); this is its documented nonterminal result, not a
failed training run. A CPU live-readout smoke check passed: two three-outcome-
column P1-P16 tables, floors printed, both explicitly LIVE/not fixed-final.

## Native arming and wake gate

Parent-side native arming command:

```bash
cd /home/erikg/emender
.venv/bin/python scripts/watch_invertible_stage10b_muonlr.py --interval 300
```

Watcher returns0 only after both successful terminal/DONE receipts bind complete
500K results;1 for explicit failure/dead/reused identities,2 for `--once` while
pending. Supplying this script does **not** claim native wake ownership was
armed here. Parent was given the command and identities. Reboot needs human
re-arming. No unattended re-launch, new seeds or post-score extensions.

On wake, audit complete500K consecutive finite LR streams, full trajectories,
source/dependency and terminal/final-checkpoint identities, then run:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.invertible_stage10b_control analyze --output X10b-growth-report.md
```

The script prints the preregistered three-outcome-column tables against BOTH
baselines for EACH arm. Independently review, interpret(a)/(b)/(c), name any
winning multiplier, and commit/push the growth report. **Growth outcomes are
pending, not fabricated at launch.** No resilient/distributed training behavior
was changed or tested; requirements-matrix/DiLoCo and scheduler queue evidence
are not applicable to this standalone local experiment. Independent acceptance
review remains required before declaring the implementation accepted.
