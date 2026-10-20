# X10 / OPTIMIZERS Stage1 — hybrid Muon matched triplet (launched)

**LIVE, seed0 only, lambda01; no optimizer-quality verdict.** Source/tests/executable registrations were committed and pushed as **`d9785a65530fd8c720c26063541c89522d7124eb` before any launch**. Canonical root (**C/**): `experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m/`. This implements the reviewed `docs/OPTIMIZERS-2026-report.md` Stage1 only; W3, production `train.py`, outer merging and SF async semantics are untouched.

| Arm | Card | Detached worker PID/starttime | Trainer PID/starttime | Finite updates at immutable snapshot |
|---|---:|---|---|---:|
| A hybrid Muon | 2 | 3915906 /253250593 | 3916445 /253250910 | 1,121 |
| B fresh AdamW+cosine | 3 | 3916541 /253250944 | 3917180 /253251288 | 2,220 |
| C fresh SF-constant | 7 | 3917208 /253251303 | 3917717 /253251630 | 2,500 |

Snapshot **2026-10-20T03:23:35Z**. Workers detached under PID1; trainers retain their worker as parent. PID/start identities and exclusive leases checked. The standalone lease broker was reused, **not WorksGood service/dispatcher/worker machinery**. Historical X9/X7 scores are context, never reused as contemporaneous arms.

## Implemented rule and per-tensor receipt

Experiment-local `hybrid_muon.py`, version **x10-hybrid-muon-fp32-v1**, supplies one optimizer facade: explicit `param_groups`/`base_lr`, `zero_grad`, one `step` per update, complete nested state serialization and strict wrong-family/routing rejection. Auxiliary AdamW is stepped/scheduled exactly once. No SF x/z points are invented for ordinary Muon/AdamW weights.

Semantic routing allowlists **36 hidden linear matrix weights**: each layer's qkv, decay projection a, output gate g, erase gate, value-write gate, output o; each MLP's w1/w2/w3. Embedding/tied vocabulary head and all norms/scalars/vectors remain AdamW. Non-map2D `value_residual`/biases are explicitly tested as auxiliary; actual E97 configuration has neither. One assignment per unique parameter; tied head/embedding aliases share one ownership row. Exhaustive **per-tensor names, aliases, shapes, counts, family, decay and LR multiplier** are in `C/registration.json:parameter_routes`; total **9,900,800**. Unknown parameters default auxiliary, never ndim-only Muon expansion.

Muon: momentum.95 via `buffer.lerp_(gradient,.05)`, **Nesterov enabled** via `gradient.lerp(buffer,.95)`. Five quintic Newton–Schulz iterations, coefficients **(3.4445,-4.7750,2.0315)**; **FP32 full matrices/momentum/NS**, tall transpose, Frobenius normalization **norm+1e-7**, zero maps to zero, explicit finite rejection. This is a locally pinned KellerJordan-convention reference, not an unspecified library default. Moonshot scaling **0.2*sqrt(max(rows,cols))** is **approximate** AdamW update-RMS matching, not exact polar orthogonality or forced exact0.2RMS. CPU/SVD reference tests verify that precise convention across square/tall/wide/zero matrices.

A's auxiliary AdamW and B: betas(.9,.999), eps1e-8. Both A groups and B use exact X9 schedule: update1=2e-7; linear5K warmup to1e-3; cosine to1e-5 at500K. C: exact X7 installed AdamWScheduleFree, betas(.9,.95), eps1e-8, constant1e-3, zero warmup; installed implementation/package defaults hashed. **Muon/AdamW evaluate ordinary live parameters; SF evaluates eval-x, restoring train-y before each update.**

### Approved decay-policy clarification

The historical controls actually passed **all parameters at decay.01**, including `A_log`/`dt_bias` despite `_no_weight_decay` tags. The supervisor explicitly approved **identical all-parameter.01 in all three arms**, tags ignored and recorded per tensor; honoring tags only in A would contaminate the comparison. No knob LR multipliers exist in this stack. Muon decay-policy sensitivity is a **registered FOLLOW-UP**, not an added variable here. A-specific Moonshot RMS shape scaling is part of the Muon method, not an unregistered matched-recipe deviation. Nominal.01 does not imply equivalent SF/AdamW/Muon shrink dynamics.

## Shared instrument and hypotheses registered before launch

Exact X9 scratch E97/T16, p61/vocab3721, dim256/depth4, batch32, uniform target CE, clip1; fp32 parameters/bf16 autocast, legacy recurrent policy and sequential Triton evaluation. Same per-seed torch initialization/procedural numpy batch order and frozen512-example pools, validations every500, safety-only **500K updates**. No ladder/carry/threshold/stagnation rule, post-result tuning, best-seed rescue or historical-result substitution. Pool hashes for **seeds0/1/2** are preregistered. Cheap-first policy launches **seed0's triplet only**; seeds1/2 await clean implementation/receipt review, **not selection by seed0 scores**, and are not automatically authorized.

Registered predictions retained from reviewed spec:

- **R1 controls:** B fixed-final contiguous perfect-prefix depth>=3 in>=2/3 seeds; C<=1 in>=2/3. Failure makes pathology-transfer comparison inconclusive before headlines.
- **R2 primary:** A depth>=3 in>=2/3 seeds; fewer falsifies. Report paired P3 acquisition times/depths; matching is not a speed win.
- **R3:** conservative prediction no robust P4+ perfect-depth gain. Higher bar: A final depth>=4 in>=2/3, relative depth advantage only if A also exceeds paired B in>=2/3. Transient P4>=50 or512 in one seed is acquisition, not robust final composition.
- **R4:** C flat only tests this SF recipe; beta2/decay/schedule/averaging are not isolated mechanisms.
- **Requested (a):** Muon composes; any P2+>=50/512 is material,512 perfect. Faster than historical B P2 first512~245,500 is a descriptive basin-entry benchmark, **not a like-threshold speed claim**; paired fresh B same-threshold timings and actual wall seconds govern speed evidence.
- **Requested (b), P4 question:** does **ANY** arm reach **P4>=50/512**, the position not materially cracked in this series? Report all arms, maxima versus fixed-final separately; do not mislabel50 as perfect composition.
- **Requested (c):** if A remains flat like C, its own implementation/source/routing receipts exonerate-or-indict the tested rule; report the result without post-result tuning or generalizing to all Muon/SF variants.

Readout command: `.venv/bin/python -m scripts.invertible_stage10_control analyze --arm muon` (likewise adamw/sf). Reports first50/512 crossings, max-anywhere, final columns and strict contiguous prefix; fixed-final is flagged invalid until500K. Full trajectory/CE, gradient/weight norms, per-matrix Muon direction RMS, full applied displacement RMS at validation updates, optimizer seconds, per-update and end-to-end seconds, peak allocation are retained. Compilation/evaluation/checkpoint costs remain in total elapsed. This fixed-recipe screen is not an equal-tuned ranking. R1/R2/R3 cannot be evaluated from seed0 alone.

## Prelaunch validation and resource bounds

**30 CPU tests passed/eight GPU skips**, including source-named routing/tied aliases, independent SVD singular-polynomial NS reference, tall/wide/zero handling, shape-scaled update RMS, decay isolation, auxiliary AdamW parity, Nesterov/momentum convention, shared schedule endpoints, finite FP32/BF16 updates, wrong-family/map rejection, full-E97 fresh-process next-update parity, matched stock GDN2, safety-only runner despite mocked-perfect metrics, SF eval-x serialization and bounded retention. Existing X7/X9 regression tests included. **Four leased-card GPU tests passed**: each optimizer performs full-model finite backward/update and complete per-position evaluation on a disposable synthetic **seed99 pool**, not confirmatory-pool selection.

All four source-bound25-update target-card probes passed **bit-exact full-model serialization/next-update parity**, with **no frozen-task evaluation**. Probe/test/package/GPU receipts themselves are hashed by executable registration; registration rejects changed worktree **or uncommitted HEAD blob** bytes. Source bindings cover171 project files plus389 installed FLA/optimizer dependency files; package inventory retained. Source-bound numerical probes were completed before launch; Muon's measured optimizer overhead (~0.05s/update) is real, not hidden behind token-speed claims.

| Arm | Probe peak allocated bytes (two models) | Registered wall ceiling seconds | GPU allocated-memory ceiling |
|---|---:|---:|---:|
| A | 464,979,456 | 177,738 | 8GiB |
| B | 542,799,360 | 118,689 | 8GiB |
| C | 542,799,360 | 125,845 | 8GiB |

Ceilings are three times compile-inclusive probe projection, at least24h; no run is extended based on scores. Infrastructure timeout/memory failure is **censored, not an optimizer loss**; external deadline adds120s cleanup allowance, TERM then30s/KILL, no automatic retry/restart. Always retain **entry + aggregate-best + latest/final**, plus one atomic staging file; deletion manifests preserve hashes. No historical files are deleted.

## Launch proof, watcher and next gate

`C/receipts/verify_launch.py` verifies registered worktree/committed source/dependency identities, preregistration ancestry, source commit, GPU/worker/trainer/lease identities, historical frozen arrays and **bit-identical X9 initialization**, empty entry optimizer, finite consecutive rows and exact actual per-update LR. All three initial model hashes agree. Entry P1–P16 counts for every arm: **`1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0`**. Checkpoint/file hashes and immutable config/validation/training-prefix snapshots are in `launch_verification.json` / `launch_snapshot/`. Live checkpoints and streams were not staged.

Supervisor arming command: **`.venv/bin/python scripts/watch_invertible_stage10_optimizers.py --interval 300`** (cwd `/home/erikg/emender`). `--once` returned**2 healthy/pending**. Parent received the command/identities for native arming; native wake ownership is parent-side, not inferred from polling. Watcher fails on explicit FAIL/dead/reused identities and returns0 only for successful DONE/result. Workers atomically publish terminal/DONE or FAIL; host reboot requires human re-arming. On wake, audit terminal/source/final checkpoint and full trajectories; independently review clean receipts before authorizing the remaining matched triplets. No production promotion follows a launch.

Local single-GPU instrument only. Architecture/gap-matrix **R16 evidence discipline by analogy**; atomic checkpoint/finite termination follow **R07/R12, R14/NDP13, NDP15 safety intent**, not production conformance. **DiLoCo K-boundary/reference-merge tests are not applicable: this path has no merges or outer state.** No Frontier Partition/QOS, elastic/native/async resilience, production SFT, 8B or scale claim. Independent reviewer acceptance remains supervisor-owned.
