# X12 launch report — armed, not yet GPU-admitted

Observed 2026-10-20 12:25Z on lambda01. Registration/source/dependency bytes were
committed and pushed **before arming** at
`acae576d28f00790faa55a4c5aa92a1804bc30e6`.
Registration SHA256: `ad978a427758938cb615ea82280b2389d0b159066340d7de52c97765d0c6c90d`.

**Hard scheduling rule preserved: ZERO X12 GPU leases; neither trainer started.**
W3 gate read has not produced the authoritative committed completion evidence.
The latest remote W3-report change is `c6000251`; no final Gate verdict heading
and accepted committed gate receipt are present. A free board alone cannot admit X12.

The supervisor explicitly approved CPU-only preparation plus a **detached
post-marker launcher** when the W3 read cannot finish in this session. That
launcher is alive as **PID241637**, host lambda01, `/proc` starttime256511879,
PPID1. It polls origin/main every300s with no GPU broker calls while W3 is
incomplete. `receipts/scheduler/polls.jsonl` initially records `WAIT_GATE`.
`receipts/launch_snapshot.json` receipts this observation; worker/detached
identity receipts are retained. Fetch/unknown-marker failures keep admission
closed. Unknown final receipt formats require attended clarification, not bypass.

After the registered gate evidence appears, the launcher chooses two
broker-grantable cards and starts one leased worker per architecture. Each
worker re-verifies the committed gate and all registration/source/dependency
hashes before acquiring a single card, heartbeats throughout, and releases
only its own lease. Actual card/PID/command/terminal receipts will appear under
`receipts/e97_bf16_seed0` and `receipts/gdn2_bf16_seed0`. There is no automatic
retry/restart if launch or training fails. No other GPU jobs were changed.

## Recipe and predictions

Both scratch seed0, four-layer nominal8M models use T64, stockbf16, uniform CE,
all-parameter AdamW decay.01, warmup1..5000 to1e-3 then cosine to1e-5 at1M,
and frozen512-example P1-P64 validation every500 updates. No stagnation rule.
Predictions(a)-(c) are verbatim in REGISTRATION.md/registration.json. The
supervisor-added operator prediction(d), **“GDN2 should not be able to do more
nonlinear steps than it has layers”**, is also registered; (a)/(d) explicitly
compete. Both layer counts and exact parameter counts are pinned, and the
previous four-perfect/budding-P5 observation must be reconciled, not omitted.

## Validation

- CPU-only `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
  tests/test_invertible_stage12_separation_8m.py`: **15 passed** (6.94s).
- Tests cover stock four-layer models, all-parameter AdamW, actual extended1M
  LR schedule and uniform CE, all64 position counters/reconciliation, fixed
  ceiling despite perfect scores, frozen pool checks/bounded checkpoints,
  remote report+receipt admission, zero broker calls while waiting, two distinct
  post-gate cards, lease/heartbeat/release lifecycle, watcher terminal tamper
  detection, and complete side-by-side growth-table rendering.
- Postcommit verification: **166 source files and388 dependency files match**
  registration and committed blobs. No production/resilient behavior changed;
  resilient requirement IDs are not applicable to this standalone experiment.
- Native watcher `--once --report`: **exit2 (pending)**, not success or failure.
  Initial GROWTH_REPORT.md is a pending64-position table, not trained results.
- First CPU test attempt failed only because the mock checkpoint attempted CUDA
  RNG collection with devices hidden; the CPU test now stubs that operation.
  No GPU lease was acquired. The final test receipt is registered above.

## Native wake / next step

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage12_separation --interval 900
# On wake, write the two complete positional columns (partial reads labelled):
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage12_separation --once --report
```

Exit0 requires both hash-bound1M successful terminal results; exit1 means a
failure; exit2 is pending in once mode. On wake, review `GROWTH_REPORT.md`, both
fixed-final perfect prefixes, independent per-position maxima, maximum single-
validation prefixes, and P5 first50/first512 timing. Commit/push the actual
post-gate launch receipts and eventual growth report when available. Target-GPU
T64 kernels have deliberately not been exercised before W3; runtime failure is
censoring, not a prediction verdict. Independent reviewer gate remains required.
