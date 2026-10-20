# X11 — stock GDN2 + exact X10 ordinary AdamW: launch receipt

## Registration and launch order

- Registered **2026-10-20 11:40:07 UTC**, before confirmatory evaluation.
  Predictions (a)/(b)/(c), material P5≥50/512 vs perfect512/512, and X7b
  P5=3/512 / “what500K buys” caveat are fixed in `REGISTRATION.md` and JSON.
- Preregistration/source commit **9255ebb67ac3b6082afbd1dac5d4524fc944dbca**
  was pushed to `origin/main` **before launch**. Registration SHA256:
  `26459bba052c8052f00c4f0dee1d7af2d51464d927c4f305066bd83d14485675`.
- Detached launch **11:40:35 UTC**, trainer started **11:40:37 UTC** on
  `lambda01`, physical **GPU0**, RTX6000 Ada, UUID
  `GPU-5265d71b-3b75-00e3-d129-2ed12b9be786`. Free-card preflight and
  exclusive standalone GPU lease succeeded. Worker **201241**, trainer
  **201472**; host/starttime identities recorded. No WorksGood service used.

Launch command (from `/home/erikg/emender`, Python3.12 project `.venv`):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m scripts.invertible_stage11_control launch
```

## Recipe and independent launch checks

Stock real-FLA GDN2, scratch seed0, T16, uniform loss; ordinary AdamW
betas(.9,.999)/eps1e-8/all-parameter decay.01, including A_log/dt_bias.
The **actual X10 factory/update** is reused, not a similar optimizer recipe.
Warmup1..5000 to1e-3 then cosine to1e-5 at500K; every update logs actual LR.
500K only, no stagnation/threshold stop or automatic retry. Per-position
512-example validation every500 uses the exact X7b/X10 frozen array hashes.
Retention is entry + best + latest/final only.

`receipts/launch_verification.json` records verification at **11:41:52 UTC**:

- All **169 project/source** and **388 dependency** bindings verified, with
  committed project bytes verified against HEAD; config records preregistration
  source commit and identical registration hash.
- Both detached identities live; GPU process query maps trainer201472 to
  registered GPU0 UUID; lease PID/starttime matches worker201241 and heartbeat
  is fresh. Other cards/jobs were not touched.
- **2192 sequential training updates** audited: every logged LR exactly equals
  the registered X9/X10 schedule; losses finite. Full P1–P16 validation through
  **step2000** retained in immutable launch snapshot (baseline/500/1000/1500/2000).
- Frozen archive array hashes match X7b and X10. Entry checkpoint has empty
  optimizer state and exact AdamW defaults, ordinary live weights. Two checkpoints
  retained at audit (within3-checkpoint bound); peak allocated359,366,144 bytes
  at validation2000. Entry SHA256 is recorded; large checkpoints are not committed.
- Native watcher `--once` returned **2**, correctly indicating RUNNING.

The audit is implemented in `receipts/verify_launch.py`; it refuses to overwrite
its first-launch snapshot. `receipts/launch_manifest.json` binds immutable
metadata/snapshot receipts. Live run data/checkpoints/console remain untracked,
not confused with frozen acceptance evidence. This report is a **launch** report,
not a terminal500K result or a prediction outcome claim.

## Validation before launch

- CPU-focused source tests: **8 passed**, target-GPU test deselected.
- Full suite on exclusively leased GPU0: **9 passed in13.32s**, including real
  stock-bf16 train/backprop and all16 per-position evaluation counters. Tests
  also cover exact X7b initialization, exact X10 defaults/all-parameter decay/LR,
  uniform CE, no all-perfect early stop, bounded retention, watcher success/fail/
  tampering/dead-process detection, and failed-registration terminal publication.
- Disposable seed99 procedural25-update probe (no confirmatory frozen pool):
  optimizer/model reload next-update max difference **0.0**; full16-position
  evaluation finite. Probe-derived infrastructure wall guard **211152s**,
  memory guard **8GiB**; infrastructure failure is censored, not optimizer loss.
- `git diff --cached --check` passed before preregistration commit.
- Post-launch adjacent X10 + X11 CPU regression: **27 passed,5 GPU cases
  deselected in26.83s**; `receipts/tests_adjacent_regression.txt`.

Commands/evidence: `receipts/tests_cpu_prelaunch.txt`, `tests_prelaunch.txt`,
`probe.json`, `gpu_before.txt`, and `launch_verification.json`.

## Native supervisor arming (not auto-armed)

```bash
cd /home/erikg/emender
.venv/bin/python -m scripts.watch_invertible_stage11_gdn2_adamw --interval 900
```

Watcher: exit0 only for matching terminal/DONE receipt plus hashed complete500K
result; exit1 failure or vanished identity; `--once` exits2 while running.
Default log: `receipts/completion_watcher.log`. No orchestration service required.

## Limits and handoff

Only seed0 is authorized. 500K remains in progress; no P5 conclusion yet.
Final analysis must distinguish final vs transient material/perfect acquisition.
Single-seed and fixed-ceiling observations do not establish asymptotic depth.
Standalone single-card experiment: resilient DiLoCo/Frontier requirements are
not applicable; no distributed resilience claim is made. Unrelated worktree
changes were left untouched. Independent reviewer gate remains with supervisor.
