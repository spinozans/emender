# X14 — no-freebie composition: launch report

## Registration before launch

Predictions (a)/(b)/(c) and P1-zero-loss protocol registered at
**2026-10-21 11:55:19 UTC**. Preregistration/source commit
**7587a57579eb52eeead2b86e0ed66268b00f5178** was pushed to origin/main before
confirmatory launch. Registration SHA256:
`efce847a2d0b95cc65bb7596bbbd9afd45fa5a80a5bf46b0badcbead0d630335`.
Committed-byte verification passed for171 project/source files and388 dependencies.
**No stage12 source is present in the bound source closure.** X13's audit question
remains open; no X12 E97-T64 conclusion is imported.

## Four running arms, lambda01

All four started from scratch, deep-first T16, seeds0/1, ceiling500K. Physical
cards0–3 were free before launch and acquired with atomic exclusive leases;
worker PID/starttime identity, trainer identity, lease ownership/heartbeat and
actual trainer-to-GPU-UUID mapping were verified. No live job was disturbed;
cards4–7 remain unused. No WorksGood service or dispatcher was used.

|Arm|Physical GPU|Worker PID|Trainer PID|Audited updates|Frozen eval through|
|---|---|---|---|---|---|
|E97 seed0|0|1070546|1071020|5775|5500|
|GDN2 seed0|1|1071135|1071814|2996|2500|
|E97 seed1|2|1071912|1072543|5590|5500|
|GDN2 seed1|3|1072699|1073197|2839|2500|

Audit timestamp **11:57:46 UTC**; running counts above are immutable snapshots,
not terminal results. Start identities and UUIDs are in
`receipts/launch_verification.json`, plus per-arm detached/command/trainer/worker
receipts. Launch commands from `/home/erikg/emender`:

```bash
for name in e97_seed0 gdn2_seed0 e97_seed1 gdn2_seed1; do
  .venv/bin/python -m scripts.invertible_stage14_control launch --name "$name"
done
```

## Protocol / launch checks

Only loss mask changes: copy the ordinary task mask and clear token index2,
P1, before calling the **actual X10 AdamW training update**. No input or target
is removed, and the frozen pool mask is untouched. Uniform CE mean on P2..P16;
P1 has zero loss gradient at every update. P2 is the first learnable composition.
P1 remains diagnostic-only NON-COMPOSITION in evaluation. Primary aggregate,
best selection and perfect-prefix length all exclude P1.

Exact X10 AdamW+cosine, betas(.9,.999)/eps1e-8/all-parameter decay.01 including
A_log/dt_bias, clip1, batch32; warmup5K to1e-3, cosine to1e-5 at500K. fp32
parameters/bf16 autocast; E97 legacy recurrent precision/sequential Triton eval;
GDN2 stock bf16 chunk path without fp32 adapter. Per-seed procedural RNG and
frozen pool array hashes match X10. No metric-based stop, no retry or tuning.

Launch audit checked all logged LRs exactly, finite losses, contiguous updates,
all16 positional counters and P1 labels, P2..P16-only aggregate reconciliation,
matched frozen arrays/archive hashes, entry checkpoint SHA256, empty initial
optimizer and exact ordinary-AdamW defaults. Each arm retained2–3 checkpoints.
Bounded entry + composition-best + latest/final retention is unchanged from
stage10 patterns. Resource guards:216967s wall /8GiB allocated memory; an
infrastructure/resource failure is censored, not architecture failure.

`receipts/launch_snapshot/` holds immutable config/data/index/validation/lease
snapshots; `receipts/launch_manifest.json` hashes them. Live logs/checkpoints
remain untracked and must not be confused with acceptance snapshots.
Audit invocation (one-shot; refuses overwrite):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.expressivity_tasks.results_invertible_stage14_nofreebie_8m.receipts.verify_launch
```

## Validation

- Prelaunch CPU:16 passed,2 GPU cases skipped.
- Leased GPU0 full suite:**18 passed in14.21s**. Tests cover both architectures
  and seeds' exact stage10-pattern initialization, stock GDN2 kernel, exact
  optimizer/schedule/all-parameter decay, zero P1 logit gradient and uniform
  remaining gradients, unchanged inputs/targets/pool, P1-excluded primary
  evaluation/best metric,500K-only stop/retention, and native watcher
  complete/fail/tamper/dead identities/all-four aggregation/growth report.
- Disposable seed99 probes,25 updates each architecture, no confirmatory pool:
  finite full16 evaluation; reload-next-update max difference **0.0** for both.
- Postlaunch adjacent X10/X11/X14 CPU regression:**43 passed,7 skipped in36.06s**.
- Native watcher `--once --report` returns **2**, expected RUNNING for all four.
- `git diff --check` clean; unrelated existing worktree changes left untouched.

## Native arming / growth on wake

Watcher script is supplied for the attended supervisor to arm (not auto-armed):

```bash
cd /home/erikg/emender
.venv/bin/python -m scripts.watch_invertible_stage14_nofreebie --interval 900
```

Exit0 requires all four hashed500K completion receipts, exit1 indicates failure
or vanished identity. It writes `GROWTH_REPORT.md` on terminal wake. For a current
censored snapshot (exit2 pending):

```bash
.venv/bin/python -m scripts.watch_invertible_stage14_nofreebie --once --report
```

Initial `GROWTH_REPORT.md` is a partial snapshot through E97 step5500 and GDN2
step3000 (seed0)/2500 (seed1), not a prediction verdict. It labels P1 diagnostic,
reports first50/first512/max/latest separately and censors absent crossings.
The two seeds must be analyzed separately and jointly at fixed-final500K;
transients stay descriptive. No conclusion (a)/(b)/(c) is claimed at launch.
Standalone expressivity only: resilient DiLoCo requirement IDs and Frontier
queue evidence are not applicable. Independent reviewer gate remains with the
supervisor; final growth assessment follows native wake.
