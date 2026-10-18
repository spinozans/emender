# X6D registered mid-run positional diagnosis

Registered UTC: 2026-10-18T13:23:54Z (before any X6D GPU forward).
Task authority: `/home/erikg/emender-scratch/fix-fanout/taskX6D-positiondiagnosis.md`.
This is a **mid-run, post-selection diagnostic**, not new training or an independent test set. Eight X6 runs remain live. Never write inside their run directories, prune, restore, interrupt, or replace their files.

## Frozen selection and instrument

1. Snapshot config, stages, validation, transitions, checkpoint index and frozen validation receipts/NPZ into `/home/erikg/emender-scratch/X6D-positiondiagnosis`; retain metadata snapshots in the diagnostic evidence directory. Select only existing completed stages T1/T2/T4/T8/T16; stage best is the already-recorded aggregate-accuracy/loss winner, never selected by position. Evaluate **both best and exit** for all eight runs, covering the requested E97 seed2 T4/T8/T16, A E97 seeds0/1 T16, B E97 T4/T8/T16 peaks and A GDN2 seeds0/1 and B GDN2 T8/T16. A GDN2 seed2 is included as the eighth boundary/control run.
2. Copy each selected retained checkpoint and stage-entry checkpoint to scratch **before loading**; match SHA-256 to the captured original metadata, fail closed on missing, unstable or mismatching payloads. Do not load a live path. Only frozen metadata is used after snapshot. Hash every executed source and validation array. No data generation, checkpoint search, optimizer transforms or training.
3. Adapt X5cP accounting to variable depth and X6's exact forward (including B neutral carry slots, probability zero). Strict-load saved ScheduleFree eval-x weights; fixed seed, model.eval/no_grad, batch32, bf16 autocast, deterministic algorithms, no TF32, CUBLAS workspace configured. Per driver: joint correct/512, x/y of the same joint argmax, out-of-range, full-state CE loss in nats. Preserve every partial-coordinate result.
4. Reconcile position sums, terminal and suffix counts and aggregate CE with the captured validation row for that exact checkpoint/depth. Stop on a mismatch rather than patching evidence. Coincident best/exit or boundary best/entry may reuse a single checkpoint+pool evaluation only if model states are exactly identical; retain explicit alias provenance.
5. Stage-boundary check: for each completed stage through T16 and its retained next-stage entry (including T16->T32), compare previous best vs next entry on the **same previous-stage pool**, not different depth pools; compare tensor equality and positional changes. Separately compare stage entry to stage best on its own pool to distinguish within-stage learning/damage from instantaneous boundary loss. Record best-vs-exit losses explicitly. Higher-depth ongoing stages beyond these T16 boundaries are outside this diagnostic's ceiling hypotheses.

## Registered interpretation rules

- A high-accuracy solved driver means >=90% joint (>=461/512); also show raw counts and coordinates. Isolated successes are not composition. E97 seed2: real late composition requires substantial non-chance joint accuracy beyond driver1, especially drivers2/3, not merely aggregate fractions. Contrast seeds0/1 driver1 decay at T16.
- B cycling: compare best/exit/entry positional profiles and stored carry probability at the checkpoint. Position gains followed by regression support cycling; **neutral-only evaluation cannot causally identify read-the-carry versus internal-carry mechanisms**. Label that mechanistic hypothesis unidentifiable if unsupported, not confirmed from aggregate shape alone.
- GDN2: show the full T8/T16 profile, first imperfect driver and first <90% driver; characterize monotone decay, nonmonotonicity and long tails from actual counts. No architecture capacity bound.
- Boundary loss means changed same-pool positional counts between carried previous best (or threshold exit if selected by the runner) and next entry; distinguish the intended best restoration from carrying the exit. Training interference is entry-to-later-stage degradation, not boundary restoration failure.

## Runtime and approvals

Host lambda01, not Frontier; use repository `.venv/bin/python` (Python3.12). No Slurm submission or resilient-training changes, so Frontier queue/DiLoCo requirements are not applicable.
All GPUs currently run training (~1.1–1.5GiB/card). Supervisor explicitly approved **sequential GPU evaluation co-resident on one live GPU**; no exclusive lease is claimed. Choose a low-utilization card once, record physical ID/UUID and before/after GPU snapshots. Only temporary slowdown is expected: roughly one eval workload's GPU demand during the short pass (instantaneous training throughput could drop up to ~50% while overlapping; no reliable counterfactual measurement is available). Record measured elapsed evaluation time, observed training throughput before/after if available, and do not present an inferred exact slowdown as measured.

## Deliverables

Dedicated `experiments/expressivity_tasks/results_invertible_stage6_truecur_8m/receipts/positional_X6D/` (outside all live per-run directories): timestamped registration, immutable metadata snapshots, checkpoint inventory, per-checkpoint/depth positional JSONs, SHA256SUMS, GPU console/runtime and CPU tests/audit. Root `X6D-report.md` with complete positional tables, three main verdicts and boundary verdict/limitations. Add focused CPU accounting/snapshot/receipt tests. Commit and push only task-owned source/tests/docs/diagnostic receipts; never stage checkpoints, NPZs, or unrelated live artifacts. Verify no staged files after commit.

## Post-discovery amendment (2026-10-18T13:25:17Z, before forwards)

Stage-entry payloads have been pruned despite declared boundary retention. Supervisor approved (a) audit exact entry hashes and pruning events, (b) use prior carried checkpoint as a proven entry only if payload SHA matches; otherwise explicitly label reconstructed-entry forwards as surrogates and direct boundary position evidence unavailable, (c) compare recorded same-pool boundary aggregates, and (d) fix required-set boundary protection plus a regression test **for future launches only**. No current process is restarted or modified. Serialized payload hashes may differ solely from depth/step/RNG/optimizer metadata, so no model-state equality is inferred from unequal hashes.
