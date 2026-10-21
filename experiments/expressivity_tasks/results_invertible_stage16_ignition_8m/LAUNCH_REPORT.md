# X16 launch report

Launch snapshot UTC: 2026-10-21T19:43:07.729220+00:00.
Preregistration commit **8f84dcee** was committed AND pushed before all five launches.
All source/dependency hashes passed worktree AND committed-blob verification before launch.
No WorksGood service/dispatcher/review machinery, Frontier, Slurm or distributed training used.

## Live bindings and initial progress

|Arm|Physical GPU|Worker PID|Trainer PID|Observed update|Last evaluation|Last loss|
|---|---:|---:|---:|---:|---:|---:|
|e97_w64_seed0|0|300299|308813|2087|2000|9.057577|
|e97_w32_seed0|1|309537|319819|1996|1500|9.488090|
|gdn2_w512_seed0|2|321710|331500|889|500|13.031175|
|gdn2_w2048_seed0|3|332119|341786|798|500|14.330572|
|e97_curriculum_seed0|6|342123|351071|1437|1000|21.880985|

Every worker/trainer passed PID+starttime liveness checks. All configs bind source commit8f84dcee.
`receipts/compute_apps_after.txt` maps trainer PIDs to GPU UUIDs; `gpu_after.txt` maps UUIDs to indices.
Cards4/5 remain idle and untouched; existing GPU7 Muon trainer1216577/worker1215981 remain untouched.
No extra arms authorized or automatically queued. All five arms are still RUNNING, not completed results.

## Registered curriculum choice

Teach E97 T16 to first evaluation with >=4 positions >=508/512, P1 included, at most500K teaching updates.
Then preserve model/AdamW moments/RNG and use warmup-free cosine from attained LR to1e-5 over500K T64 updates.
No criterion relaxation, optimizer reset, T16 replay or automatic retries. If teaching misses criterion,
report criterion_not_reached, not T64 failure. T64 update0 transfer baseline plus T16 retention every500.
All sweep/curriculum predictions (a)/(b)/(c) were registered before launch; see REGISTRATION.md and registration.json.

## Watchers and supervisor arming

Native detached watcher PID**381106**, interval900s, identity/argv at `receipts/watcher/worker.json`.
Parent CONFIRMED authoritative supervisor-side native monitoring: process **proc_27d6**, PID**728529**,
`scripts.watch_invertible_stage16_ignition --interval 900`; first poll2026-10-21T19:44:01Z found all five RUNNING.
Its exit wakes the supervisor session natively. Detached watcher alone was NOT treated as sufficient.
Both monitor all five arms; failed arms are censored and do not disarm monitoring of remaining live arms.
Growth is regenerated every wake at `GROWTH_REPORT.md`; consolidated terminal receipt:
`receipts/watcher/terminal.json` (only after all arms terminal). No architecture failure inferred from infrastructure failure.
One-shot command from repository root (exit2=pending,0=verified terminal,1=terminal error):

```bash
.venv/bin/python -m scripts.watch_invertible_stage16_ignition --once
```

## Validation

- CPU:16 passed,2 GPU tests deselected (`receipts/tests_prelaunch.txt`).
- Leased physical GPU0:2 GPU tests passed (`receipts/tests_gpu_prelaunch.txt`), no4/5/7 usage.
- Disposable seed99 preflight, no confirmatory evaluation: E97 and GDN2 next-update reload max difference0.0.
- E97 T16/T64/T16 depth-safe evaluation and T64 update shape exercised; stock GDN2 T16 update/eval exercised.
- Probe peak allocations (two models plus eval): E97 764779520B, GDN2 562250752B; all finite.
- Explicit-share loss/gradient tests cover all four registered weights plus0/1/16/1; unchanged mask, exact AdamW defaults/schedule.
- Mocked full lifecycle checks criterion-gated phase switch, preserved model/optimizer/moments, distinct criterion/T64-entry checkpoints, retention every evaluation and full extension ceiling.
- Watcher tests cover tampered terminal hashes, all-arm monitoring after one failure, retention report and right-censoring.
- Initial CPU test run had3 test-harness failures due to an unmocked CUDA RNG call; corrected mock and reran successfully before registration.
- Registration constructed with CUDA hidden may emit FLA CPU-platform warnings; actual leased GPU probes/tests and live runs confirm CUDA paths.
- `git diff --cached --check` passed for the preregistration commit. Whole-worktree diff check flags a pre-existing X14 report blank line; unrelated changes were not modified/staged.

## Interpretation / follow-up

No ignition/curriculum verdict yet. Final500K primary; crossings right-censored at latest observed update.
Historical X8 GDN21/136 differs in optimizer and non-P1 weights: contextual, not a matched causal endpoint.
Single seed0, no multi-seed generality. Bracket-closing arms require NEW registration and permitted free-card authorization.
Reviewer gate remains parent-owned; this report supplies launch/preflight evidence, not a completed measurement claim.
On native wake inspect terminal receipts, GROWTH_REPORT, first-crossing timing, new T64 positions beyond transfer baseline, and taught-position retention.
Only task-owned files were staged; all pre-existing dirty files and unrelated result directories were left alone.
