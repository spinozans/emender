# W3 matched-dose wash-2 regime experiment

Status: **W3G INVALID/INCONCLUSIVE; rescue question OPEN.** Both arms completed exactly 512 updates / 3,571,182 inputs / 154,386 targets. After the eight-free-GPU window opened, parent and A completed all three Stage-B repetitions and four execution replicas; B repetition1 fail-stopped at the frozen boundary guard during case11. No retry, replay, resume, checkpoint promotion or negative optimizer finding. The supervisor authorized publication of preserved partial data and read-only diagnosis, not further reads. Both arms trained serially on GPU6 and passed their own root-capacity gates. X9b final DONE/full-trajectory attestation verified; B used the operator-preselected AdamW+cosine regime bundle. W3R's sole P1 condition is addressed and regression-tested. Preregistration SHA256 `4976d3761102c6e28cf556a8b897db397edc3e0e813758644f83d2053420c52a` (supersedes preserved pre-W3R/pre-capacity snapshots).
Operator go and frozen contract: `/home/erikg/emender-scratch/fix-fanout/taskW3-matcheddose.md`, `REGIME-audit-report.md` sections 1–8, REGIME-R, and its disclosed W3 prelaunch amendments. No production growth verdict exists.

## Preflight verification table

| Binding | Evidence | State |
|---|---|---|
| Production model | 4,045,972,080 unique parameters, 236 parameter tensors plus tied head alias | verified |
| Parent file | full SHA256 `cb50c6586c91cb8f01e3c614b504ecc53dbad087a38af225b9e62ef762a24737` | verified |
| Fresh-state recipe | Original wash-1 x `2f2b209d…` → FP32 SF y interpolation → materialized `4132b84a…` → identical fresh seed; empty moments, k=0, warmup0 | independently verified across all parameters; `parent-conversion.json` |
| Authority / packs | `55aca12a4952338e576909d2e52541b5b5807ad14a970410309f740eff510184` / `f9b3a5e103c5d94f692fcd736dda95cfea75c78bc9614d247c37019f1aacc17e` | manifests and every payload hash verified |
| Actual exposure | **154,386 supervised targets / 3,571,182 inputs**, 512 updates | independent binary-index/mask recount passes every historical step |
| Batch identities | Every sampled pack/record ID, historical sample ID, prediction-aligned loss mask, recurrent reset/valid mask and padded token digests | frozen in `preflight.json` |
| Stage-B | exact 14 cases, Pi inventory, CLI image, tools, args, tokenizer receipt | bound; four evaluator sources byte-match wash-2 |
| Execution | exact 96 cases, case/grading module, runtime image manifest, backend bundle | bound; four replicas, fixed lowest-rank primary [0,1] |
| docNLL | 8-doc / 198,295-target panel `5725d820…`, saved-x | scorer, phase scripts, panel, substrate and original ruling hash-bound; diagnostic trajectory only |
| AdamW serialization | explicit typed class/version, live-nonaveraged point; frozen args unchanged | 47 identity/facade/precision tests pass |
| AdamW viability | full production BF16 update and both-mode reload/hash equality | **passed** GPU6; 33,362,854,400 bytes peak allocated; train/saved point hashes both `fcdfcedcb16210af11c48f588b08a5b31ba19471b831553eb44ca9f723f8e5df` |
| Regime branch | X9b verified DONE at500K + completed1001-row trajectory SHA `42c859d9…` | **fixed AdamW+cosine**: max-anywhere P2+ **11/512 <50**, fixed-final counts `[95,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0]`; no material composition anywhere |
| Root-capacity launch gate | user-available free space on `/` ≥**214,748,364,800 bytes (200GiB)** | enforced/logged immediately before A and rechecked before B; failed gate hard-stops; no readiness reading authorizes launch |
| Topology / training | **serial A then B, one leased free card from 6/7**, world1 per arm | selected before results; actual **GPU6** / UUID `GPU-6d8d6161-e8e8-0b81-9abb-1a8f7d15e65d`; both DONE |
| Source / runtime | `runtime-freeze-v2.json`; all 512 current Dataset/sample/token/loss/valid/reset hashes match independent recount | exact Python source/package/optimizer versions frozen; tokenizer agrees; **14,070 Pi payload files verified** |
| Preregistration / final reads | no threshold stops, retries or best checkpoint; pin before any gate read | committed prerequisite in runner; completed-trajectory selector, bounded serial launcher, watcher, frozen read/tally harness prepared; W3R review completed with sole P1 condition addressed |

All local receipt paths above are under `receipts/w3-matcheddose/` unless absolute. The raw historical receipts are unmodified. The independent recount decodes little-endian indexes and drops the first target of every independent record; it does not trust manifest target totals or call the training Dataset to count targets.

## Frozen predictions (verbatim from section 6)

Strong regimen-rescue hypothesis predicts B reaches **valid≥12, correct≥10, primary execution≥64/96**, clears every existing retention bound, and gains **≥4 correct actions against the contemporaneous parent6**, not merely5→6 or one lucky execution replica. To establish relative regimen benefit, B must also exceed A by **≥3 correct actions** (outside the historical ±1–2 band) without sacrificing the execution floor. A is predicted to remain in the historical correct-action4–8 range and may reproduce execution59–60; this is a falsifiable prediction, not a required outcome. B≤8 correct, or <3 advantage over A, fails the **large matched-dose regimen-rescue** prediction; 9 correct is partial acquisition but gate failure. If both arms improve/pass, old failure was not uniquely explained by the new regimen. If B improves Stage-B but execution/retention fails, the dual gate still fails. If either arm's identity, dose or instrument differs, classify **invalid/inconclusive**, not a negative optimizer finding.

The retention-bound phrases in that verbatim prediction are subject only to the preregistered authority correction: there are no numeric docNLL pass/fail limits; docNLL is diagnostic trajectory, not a stop signal. The 2026-10-02 operator ruling supersedes invented comparative bounds/noise bands. Behavioral thresholds and all other predictions remain verbatim.

## Read protocol and hold

A = contemporaneous SF constant 1e-5, warmup0. B = SF external cosine if X9b final P2+ materiality ≥50/512; otherwise operator-preselected AdamW+cosine bundle (.9,.95), eps1e-8, wd.01. Both use exactly 512 identical batches and target-normalized loss, with entry and fixed final stored. Schedule is `1e-7 + .5*(1e-5-1e-7)*(1+cos(pi*(t-1)/511))`, t=1..512; no adaptive dose, retries or best-checkpoint selection. Log scheduled/actual LR, per-step targets and cumulative objective-weighted exposure. Parent/A/B behavioral primary is train-y; saved-x is diagnostic. Three unchanged greedy Stage-B repetitions and four execution replicas per checkpoint; report all spreads, never average/best-rescue. No gate read before pinning the read checkpoint and receipt.

## Validation

- `.venv/bin/python scripts/w3_matcheddose_preflight.py --output receipts/w3-matcheddose/preflight.json`: passed; exact 512-step exposure and frozen instrument hashes.
- `PYTHONPATH=$PWD OMP_NUM_THREADS=2 .venv/bin/python scripts/w3_verify_parent.py --output receipts/w3-matcheddose/parent-conversion.json`: passed. First attempt exposed tied embedding/head alias (237 state-dict entries versus 236 unique parameters); verification now derives named-parameter layout from the exact production model on meta and independently verifies the alias. No data/seed changes or training retries occurred.
- Initial identity/facade/precision suite: **47 passed in 22.83s**, committed before GPU viability. Initial harness suite including `tests/test_w3_regime_and_gate.py`: **55 passed in 20.30s** (`tests-launch.txt`); W3R P1 regression suite: **60 passed in 20.65s**, exit0 (`tests-attestor.txt` / `tests-attestor-exit.txt`); amended capacity suite **63 passed in20.85s**, exit0 (`tests-capacity.txt` / `tests-capacity-exit.txt`). Covers independent boundary masking, hard mismatch stop, typed optimizer conflicts in both modes, AdamW point-hash equality, SF typed-versus-legacy byte equality, final-receipt-only branching, max-anywhere materiality, exact LR endpoints/fixed stop, conservative repetition comparisons, prediction falsifiers and serving nondeterminism.
- Disposable one-update `scripts/w3_run_adamw_viability.sh`: **passed**, detached PID3610471, GPU6 lease released, terminal exit0, no retry. Full 4B BF16 model/moments finite; first pack loss **1.2679857436057442**, exactly reproducing historical wash-2's first-pack loss. Peak reserved35,418,800,128 bytes. Both read-mode hashes agree. Receipt/checkpoint identities in `adamw-viability.json`; this is not arm B and had no gate reads. Per the later supervisor capacity directive, its24,276,152,095-byte disposable checkpoint was hash-verified then deleted with its disposable latest symlink; hashes/results remain receipted in Git and `viability-checkpoint-retirement.json`. No additional4B preflight checkpoints were written.
- `scripts/w3_freeze_runtime.py`: all512 Dataset samples/masks/tokens independently cross-checked; Pi inventory all14,070 files and tokenizer verified; docNLL authority chain/substrate bound. Initial environment inventory attempt found no pip module, so inventory uses `importlib.metadata.distributions` without installing or altering the environment. `runtime-freeze-v2.json` supersedes earlier source snapshots by adding completed harness and W3R attestor sources; initial, pre-W3R and pre-capacity snapshots are retained.
- `bash -n scripts/w3_run_adamw_viability.sh scripts/w3_run_arms.sh scripts/w3_run_gate_reads.sh` and `git diff --check`: passed. At this prelaunch-validation point no production arm or gate had been invoked; the subsequent detached launch is separately receipted below.

Applicable architecture intent: **R07/NDP15 atomic publication, R12 exact parent/state binding, R14/NDP13 bounded fail-stop execution, R16 immutable evidence discipline**, per `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` ADR-003 and its gap matrix. This is local world1 full-model SFT, not resilient systems acceptance, Frontier submission, or a new scale rung. Elastic/native/async requirements remain explicitly unclaimed. No WG service, scheduler, dispatcher or workers started. GPU lease utility alone is used for resource exclusion as explicitly required by W3.

## Final X9b branch receipt

X9b completed its fixed500K ceiling with successful original DONE/terminal/result. The committed/tested completion-only attestor bound all1001 validation rows/order, original config/data/frozen pool and terminal evidence; `x9b-completed-trajectory.json` SHA `5eeee50585408ced35bc105945eb2c06b02a3dc2f6ed2bf9980b24c267121130`. Full trajectory SHA `42c859d94706e8c42dbfbd57a8f3eb1adc9a064c3f97c4280f533c6428ede22f`.

Maximum P2+ was **11/512**, below the registered50/512 materiality bar. Fixed-final vector was `[95,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0]`. Arm B is therefore **ordinary AdamW+the exact external cosine**, betas(.9,.95), eps1e-8, wd.01, internal warmup0. A remains contemporaneous fresh SF constant1e-5. This is explicitly the operator-preselected **regime-bundle** contrast, not unique attribution to SF averaging. No partial X9b score selected the branch. `fixed-branch.json` records the complete evidence chain and both maximum/fixed-final statistics.

## Prelaunch rulings and runbook

Before any production results, supervisor approved: (1) docNLL original trajectory-only ruling supersedes nonexistent retention limits; (2) typed AdamW live-point serialization with train/saved equality, without SF shim state or changing frozen args; (3) X9b **max-anywhere P2+≥50/512 over its completed trajectory**, with fixed-final counts alongside and escape-without-consolidation disclosed; (4) EACH Stage-B repetition must pass, and advantages use min(B)−max(parent)≥4 and min(B)−max(A)≥3, also reporting historical parent6. Any divergent case grades/reasons or tallies are a serving-nondeterminism finding, not independent samples or an averaged rescue.

**New infrastructure precondition:** after the root volume filled and an unrelated X6 run failed with ENOSPC, the supervisor required ≥200G of root-volume free space at each W3 arm start. `w3_run_arms.sh` now invokes `w3_disk_preflight.py` immediately before EACH training command (after lease acquisition), conservatively enforcing200GiB and writing `arms/{A,B}-disk-gate.json` with free bytes/threshold/time. A blocked gate writes a terminal reason, FAIL marker and exits; there is no automatic retry. Concurrent capacity pruning does not waive this check. This amendment is preregistered; the three new CPU tests cover both arms, inclusive threshold, logged nonzero failure. Neither the incident nor the capacity gate is an optimizer finding.

Historical prelaunch action order (retained as the original runbook; current completion status is in Results and the final W3G publication):

1. Await existing X9b watcher/verified 500K DONE. First run `.venv/bin/python -m scripts.w3_attest_x9b_completion` (enforces its source/tests/green test evidence are committed). It creates `x9b-completed-trajectory.json` once, binding original DONE/terminal/result/config/data/frozen-pool and the complete1001-row validation digest/order without modifying any original receipt or live trainer. Then run `.venv/bin/python -m scripts.w3_fix_regime_branch --output receipts/w3-matcheddose/fixed-branch.json`. Missing/failed/partial/malformed/tampered terminal or supplemental evidence hard-stops before reading intermediate counts. Inspect/report completed maximum and final values; commit the branch receipt **before either arm**. The training runner enforces both preregistration and branch bytes are committed.
2. W3R review is persisted in `W3R-review.md`; its sole P1 condition (full trajectory digest in consumed terminal evidence, including intermediate49→50 and missing-receipt rejection) is implemented and covered by the60-test suite. Supervisor approved the supplemental completion-only attestation seam because the live Bash publisher was already parsed at launch. After the final attestation/branch receipt is committed, launch `nohup setsid bash scripts/w3_run_arms.sh > /home/erikg/emender-scratch/w3-arms-console.txt 2>&1 < /dev/null &`. One serial pair, same leased card, **fresh ≥200GiB root-volume check immediately before each arm**, bounded3h per arm, no retries. Arm directories are create-once. Monitor `.venv/bin/python -m scripts.watch_w3_arms`; it checks live PID/starttime and terminal receipts, never launches, retries or grades.
3. After both fixed-final completions, obtain a **free eight-GPU window** and run the detached `scripts/w3_run_gate_reads.sh`. This is required by the unmodified frozen execution CLI; current cards0–5 are occupied, so no gate launch is claimed. It fails before any read if eight free GPUs cannot be leased. Parent/A/B are re-pinned before reads; same three Stage-B repetitions, exact four execution replicas with lowest rank primary; original saved-x docNLL scorer; pinned verdict receipt.
4. Fill the full results table from pinned case-level receipts, apply predictions verbatim with only the disclosed prelaunch amendments, commit and push. No quality verdict or checkpoint promotion before then.

The ordinary-AdamW viability establishes one finite full-size update and exact point serialization, not a 512-update quality outcome. All original historical receipts remain unchanged. The report's readiness is **conditional on X9b final plus completion-only attestation, committed branch, and a green root-capacity gate at each actual arm start**, never a claim that production training/readout finished. Original W3R review was BLOCKED pending P1; that historical review is preserved unedited, with remediation/test evidence disclosed here.

## Launch-readiness receipt (2026-10-20 01:27:55Z)

- Detached launcher **PID963676**, starttime252566296, lambda01; trainer A PID963769. Serial A→B, one physicalGPU6, world1 per arm,3h fail-stop timeout per arm; no restart/retry.
- Mandatory root-capacity gate immediately before A: **1,023,977,017,344 free bytes** on `/`, versus214,748,364,800 required; passed at01:27:55Z. The wrapper must repeat the actual check immediately before B; today's A reading cannot authorize B.
- A's immutable entry pin/start receipt exists. Update1 reproduced historical first-pack loss **1.2679857436057442**,482 targets/6,677 inputs, scheduled=actual LR1e-5, objective-weighted cumulative exposure1.0. This is training telemetry, not a capability read.
- Supervisor-side completion watcher **PID964036**, command `.venv/bin/python -m scripts.watch_w3_arms --interval 300`; log `/home/erikg/emender-scratch/w3-arms-watcher.txt`. Launcher console `/home/erikg/emender-scratch/w3-arms-console.txt`. Completion/failure markers and exact process/source bindings are in `receipts/w3-matcheddose/launch-readiness.json` (SHA `4cb556af134aa3bedb60fae0f6d6ae183d246537478137d5dd355c9fc014b9bb`).
- Actual training roots: `e97-rl-loop-v1/w3-matcheddose/arms/{A,B}`. The frozen gate follows BOTH fixed-u512 completions and an eight-free-GPU window. No partial-score read, best selection or changed dose is authorized.

## Results

| checkpoint | Stage-B repetitions (valid/correct/invalid) | Execution replicas /96 | Fixed primary /96 | saved-x docNLL |
|---|---|---|---|---|
| parent contemporaneous | 12/6/1; 13/7/0; 12/6/1 | 64, 64, 64, 64 | 64 | not run |
| A fixed u512 | 12/5/0; 12/5/0; 12/5/0 | 59, 60, 59, 60 | 59 | not run |
| B fixed u512 | rep1 prefix **7/4/1 over10 completed cases only**; case11 guard stop; reps2/3 not run | not run | not available | not run |
| historical wash-1 parent | **12/6/1** | historical primary64 | 64 | historical, not substituted |
| historical wash-2 u512 | **12/5/0** in all three repetitions | 59, 59, 60, 60 | **59** | historical, not substituted |

Verdict: **INVALID/INCONCLUSIVE; B's gate is not evaluable and the decisive rescue question remains OPEN.** Parent/A complete read data are preserved and fully tabulated below, including serving nondeterminism. B's10-case prefix is descriptive only: it is not a completed14-case score, a gate failure, or a negative optimizer finding. All docNLL reads were downstream of the stop and therefore were not run; no values or retention bounds are invented. See the final W3G section for the supervisor ruling, observation, evidence and operator-decision register.

## W3G gate-read scheduling (2026-10-20 04:02:22Z)

- Detached external supervisor PID **3965152**, `/proc` starttime **253493066**, polls the eight-free-GPU window every300s. Initial free set was GPU6 only; no co-resident reads were launched. Console: `/home/erikg/emender-scratch/w3-gate-window-console.txt`.
- The approved external guard `scripts/watch_w3_gate_window.py` leaves the frozen runner and all bound instrument sources untouched. It re-verifies both entry and fixed-final training pins before the frozen runner re-pins parent/A/B, checks the window again, and monitors all eight GPUs every5s for non-descendant compute processes. A collision terminates the entire read process group, preserves partial evidence and waits; no automatic replay/retry or negative optimizer finding is permitted.
- Durable append-only occupancy evidence, read-time pin verification, process identities and terminal receipt: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w3-matcheddose/gate-supervision/`. Frozen runner SHA256 `79afa82dd50ddc8ea6b1aac3f167e8986958b097a3b80e73c07410fe00144d6b`.
- A final checkpoint SHA256 `d61b9475b10059635095aa4de11222e68c57b91576780665fc3daed40b630b0a`; B `aef26f5f9c78210e071dfd9c5f8c040675c8c489511ffd6288bfa33b6bac1e23`. Training entries were pinned before training and will be freshly SHA-verified at read time.
- Publication applies every frozen scalar rule without averaging/best-rescue. Instrument divergence, including serving-nondeterminism, overrides any raw classifier label to **invalid/inconclusive**; the raw classifier flags and all case-level findings remain reported alongside, never rewritten.
- Guard/regime validation: `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_watch_w3_gate_window.py tests/test_w3_regime_and_gate.py`: **18 passed in10.98s**; frozen Bash syntax and `git diff --check` passed. Applicable safety intent remains R12 exact binding, R14/NDP13 bounded termination, R16 immutable evidence; no resilient systems/Frontier acceptance is claimed.

### Attended continuation (2026-10-20 11:40Z)

The prior supervisor workflow timed out while waiting; the detached guard **remained alive with the same PID/starttime and source/runner hashes**. No duplicate launcher, retry, checkpoint read or gate invocation was made. Its append-only occupancy log is continuous; the read directory does not yet exist. At continuation only X6 B_e97_seed1 occupies GPU5 (final T128 stage409K/500K); all eight cards must still be free before reads.

Personally re-verified the committed frozen receipts and source bindings, all preregistration bindings, X9b completed-trajectory/branch selection, runtime versions, tokenizer, docNLL chain, all14,070 Pi payloads, and every512-step independent exposure/mask recount plus current sampler/token/loss/valid/reset replay. `gate-resume-preflight.json` binds this read-only verification and its script; `gate-resume-verify.py` preserves the exact script bytes. Fresh entry/final checkpoint SHA verification remains scheduled at the window itself, not substituted by this earlier preflight. The frozen runner will pin all parent/A/B checkpoints before serving.

Continuation validation: `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_watch_w3_gate_window.py tests/test_w3_regime_and_gate.py tests/test_w3_matcheddose_preflight.py tests/test_w3_disk_preflight.py`: **31 passed in9.31s** (`tests-gate-resume.txt`). Frozen Bash syntax and `git diff --check` passed. `A/B-complete-receipt.json`, `A/B-terminal.json`, both launch disk gates and `gate-launcher.json` preserve original evidence bytes; originals are unchanged.

## W3G final publication: INVALID/INCONCLUSIVE (2026-10-20)

### Execution and fail-stop receipt

X6 and X11 finished before any W3G model serving. The guard first observed all eight cards free at17:38:31Z, then freshly re-hashed both training entry pins and both fixed-final pins. `supervision/read-time-training-pins.json` receipts all four SHA256 identities; both entries equal parent `cb50c658…`, A final `d61b9475…`, B final `aef26f5f…`. At **17:40:08Z**, the second window check showed all eight cards at4MiB/0% utilization, no compute processes, and all eight grantable. Frozen runner **PID3246750**, starttime258399648, leased GPUs0–7; parent/A/B were all re-pinned and their read identities published before the first Stage-B serving.

The runner stopped nonzero at **19:52:44Z**: B repetition1 case11 `choice-write-file` raised `ValueError('private reasoning crossed boundary')` at frozen `eval_e97_pi_native_stage_b.py:63`, before publishing that case's row. Parent/A each completed all42 Stage-B episodes and all384 execution episodes; B completed10 published rows plus the tripped case's saved transcript. B cases12–14, repetitions2/3, execution and **all checkpoints' docNLL** were not run. The frozen runner never reached its verdict command; no fabricated `reads/verdict-receipt.json` exists. The separate publication is `gate-read/partial-verdict.json`.

Across **1,528 reading-phase occupancy snapshots**, the external guard observed **zero foreign compute processes**. GPU leases were released, all model processes exited, and no collision or automatic retry occurred. All source/instrument/checkpoint bindings, complete summary/plan bindings, eight rank reports per completed checkpoint,96-case oracle smokes and model immutability/precision receipts were independently rechecked during publication. Raw evidence remains in the original immutable run directory; its11,265 non-checkpoint/non-Triton evidence files are individually SHA-bound in `gate-read/raw-evidence-manifest.json`. Private transcripts are retained there, not copied into Git.

**Supervisor ruling after the stop:** publish INVALID/INCONCLUSIVE now with every preserved parent/A result and B's10-case prefix; inspect the tripped case and neighbors read-only; document behavior as an observation, not a B gate verdict; keep the rescue question OPEN. No evaluator patch, model generation, regrade, replay, retry or resume was authorized or performed.

### All completed scalar reads and spreads

`invalid` means terminal reason beginning `invalid_`, **not**14−valid. End-to-end success is distinct from correct first action. A Stage-B gate requires **each repetition** valid≥12 AND correct≥10; execution uses literal primary≥64/96, never the best/mean replica.

| Checkpoint | Rep | Valid /14 | Correct /14 | Invalid | Missing first frame | End-to-end success /14 | Stage-B scalar gate |
|---|---:|---:|---:|---:|---:|---:|---|
| parent | 1 | 12 | 6 | 1 | 2 | 1 | not met |
| parent | 2 | 13 | 7 | 0 | 1 | 1 | not met |
| parent | 3 | 12 | 6 | 1 | 2 | 1 | not met |
| A | 1 | 12 | 5 | 0 | 2 | 0 | not met |
| A | 2 | 12 | 5 | 0 | 2 | 0 | not met |
| A | 3 | 12 | 5 | 0 | 2 | 0 | not met |
| B | 1 | **7/10 prefix** | **4/10 prefix** | **1 in10** | **3 in10** | **0/10 prefix** | **not evaluable; aborted at case11** |
| B | 2 | not run | not run | not run | not run | not run | not evaluable |
| B | 3 | not run | not run | not run | not run | not run | not evaluable |

Parent ranges: valid12–13 (spread1), correct6–7 (spread1), invalid0–1 (spread1), end-to-end1–1 (spread0). A ranges: valid12–12, correct5–5, invalid0–0, end-to-end0–0 (all scalar spreads0). **Both nevertheless exhibit case-grade/reason serving nondeterminism**. Parent `path-32768` changes invalid_frame→finished→invalid_frame, altering valid/correct counts; `choice-source-check` and `choice-fetch-url` finish in reps1/2 but hit turn_budget in rep3. A `path-01024`, `path-32768` and `choice-bash-command` have differing termination reasons despite identical aggregate tallies. Tool-result counts and some non-primary grade checks also differ; every complete check dictionary is archived. These are unchanged greedy reproducibility repetitions, not independent samples; no averaging or best-repetition rescue is used. Under the preregistered rule, divergence is an instrument/reproducibility finding, never a negative optimizer finding.

| Checkpoint | Replica | Ranks | Sum /24 | Edit /24 | Lookup /24 | Recovery /24 | Total /96 | Floor≥64 |
|---|---|---|---:|---:|---:|---:|---:|---|
| parent | **r0 primary** | **0,1** | 20 | 16 | 15 | 13 | **64** | yes |
| parent | r1 | 2,3 | 20 | 16 | 15 | 13 | 64 | yes |
| parent | r2 | 4,5 | 20 | 16 | 15 | 13 | 64 | yes |
| parent | r3 | 6,7 | 20 | 16 | 15 | 13 | 64 | yes |
| A | **r0 primary** | **0,1** | 18 | 16 | 13 | 12 | **59** | no |
| A | r1 | 2,3 | 18 | 16 | 14 | 12 | 60 | no |
| A | r2 | 4,5 | 18 | 16 | 13 | 12 | 59 | no |
| A | r3 | 6,7 | 18 | 16 | 14 | 12 | 60 | no |
| B | all four | preregistered0–7 | not run | not run | not run | not run | not run | not evaluable |

Parent execution range64–64, total spread0, maximum pairwise success-label disagreements0, all grades identical. A range59–60, total spread1, maximum pairwise success-label disagreements1; sole disagreeing case `regression-fresh-lookup-0` fails in r0/r2 and succeeds in r1/r3 (finished-correctly check differs; reason finished in all four). Family spreads are0 except A lookup13–14. **The primary remains59**, not60. All96 outcomes for each of the eight completed replicas are preserved in the two archived execution summaries plus all16 rank receipts.

Historical comparison remains explicitly alongside: wash-1 parent **12 valid /6 correct /1 invalid**, execution64/96; wash-2 **12/5/0** in each of three repetitions, fixed-primary **59/96**, other replicas59,60,60. Today's A gives12/5/0 in each repetition and59/96 primary, descriptively reproducing those historical scalar values, not proving a dose or optimizer effect.

| Diagnostic | substrate | parent | A | B | Bound |
|---|---|---|---|---|---|
| saved-x docNLL, original8-doc/198,295-target panel5725d820 | not run | not run | not run | not run | **none; trajectory-only** |

No historical docNLL value is substituted for a current read. The diagnostic remains downstream of the fail-stop and supplies no gate or retention conclusion.

### parent: all completed Stage-B case rows

Each cell is **V/C/I/S; terminal reason; T**: valid first frame, correct first action, invalid-frame termination, end-to-end success; T is observed tool-result count. Boolean flags are 0/1. All complete grade/check dictionaries are preserved in the archived summaries.

| Case | Repetition 1 | Repetition 2 | Repetition 3 |
|---|---|---|---|
| copy-01024 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 |
| bind-01024 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 |
| path-01024 | 1/1/0/0; finished; 2 | 1/1/0/0; finished; 2 | 1/1/0/0; finished; 2 |
| copy-32768 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 |
| bind-32768 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 |
| path-32768 | 0/0/1/0; invalid_frame; 0 | 1/1/0/0; finished; 6 | 0/0/1/0; invalid_frame; 0 |
| choice-read-explicit | 1/1/0/1; finished; 1 | 1/1/0/1; finished; 2 | 1/1/0/1; finished; 2 |
| choice-grep-symbol | 1/0/0/0; finished; 2 | 1/0/0/0; finished; 2 | 1/0/0/0; finished; 2 |
| choice-bash-command | 1/0/0/0; finished; 6 | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 11 |
| choice-edit-file | 1/1/0/0; finished; 2 | 1/1/0/0; finished; 2 | 1/1/0/0; finished; 2 |
| choice-write-file | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 |
| choice-process-start | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 |
| choice-source-check | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; turn_budget; 12 |
| choice-fetch-url | 1/0/0/0; finished; 10 | 1/0/0/0; finished; 11 | 1/0/0/0; turn_budget; 12 |

### A: all completed Stage-B case rows

Each cell is **V/C/I/S; terminal reason; T**: valid first frame, correct first action, invalid-frame termination, end-to-end success; T is observed tool-result count. Boolean flags are 0/1. All complete grade/check dictionaries are preserved in the archived summaries.

| Case | Repetition 1 | Repetition 2 | Repetition 3 |
|---|---|---|---|
| copy-01024 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 |
| bind-01024 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 |
| path-01024 | 1/1/0/0; generation_budget; 2 | 1/1/0/0; generation_budget; 1 | 1/1/0/0; turn_budget; 12 |
| copy-32768 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 | 1/1/0/0; finished; 0 |
| bind-32768 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 | 0/0/0/0; generation_budget; 0 |
| path-32768 | 1/1/0/0; turn_budget; 12 | 1/1/0/0; finished; 10 | 1/1/0/0; turn_budget; 12 |
| choice-read-explicit | 1/1/0/0; finished; 4 | 1/1/0/0; finished; 3 | 1/1/0/0; finished; 2 |
| choice-grep-symbol | 1/0/0/0; turn_budget; 12 | 1/0/0/0; turn_budget; 12 | 1/0/0/0; turn_budget; 12 |
| choice-bash-command | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; turn_budget; 12 |
| choice-edit-file | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 2 |
| choice-write-file | 1/0/0/0; finished; 2 | 1/0/0/0; finished; 2 | 1/0/0/0; finished; 2 |
| choice-process-start | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 | 1/0/0/0; finished; 1 |
| choice-source-check | 1/0/0/0; finished; 2 | 1/0/0/0; finished; 3 | 1/0/0/0; finished; 2 |
| choice-fetch-url | 1/0/0/0; finished; 5 | 1/0/0/0; finished; 4 | 1/0/0/0; finished; 4 |


### B repetition1: preserved ten-case prefix (not a gate score)

| Order | Case | Valid first frame | Correct first action | Invalid-frame termination | End-to-end success | Terminal reason | Tool results |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | copy-01024 | 0 | 0 | 0 | 0 | generation_budget | 0 |
| 2 | bind-01024 | 1 | 1 | 0 | 0 | finished | 0 |
| 3 | path-01024 | 1 | 1 | 0 | 0 | finished | 3 |
| 4 | copy-32768 | 0 | 0 | 0 | 0 | separator_before_valid_turn | 0 |
| 5 | bind-32768 | 0 | 0 | 1 | 0 | invalid_frame | 0 |
| 6 | path-32768 | 1 | 1 | 0 | 0 | turn_budget | 12 |
| 7 | choice-read-explicit | 1 | 1 | 0 | 0 | finished | 1 |
| 8 | choice-grep-symbol | 1 | 0 | 0 | 0 | turn_budget | 12 |
| 9 | choice-bash-command | 1 | 0 | 0 | 0 | turn_budget | 12 |
| 10 | choice-edit-file | 1 | 0 | 0 | 0 | finished | 2 |
| 11 | choice-write-file | **row not published** | not graded for publication | — | — | evaluator boundary assertion after normally closed Pi transcript | 2 saved read calls |
| 12–14 | choice-process-start / choice-source-check / choice-fetch-url | not run | not run | not run | not run | not run | not run |

The prefix denominator is **10**, not14. Case11 is an observation only; it has no published episode grade and is not silently counted as success/failure. No B14-case aggregate or conservative B-vs-parent/A advantage is computed; no falsifier is applied to the partial prefix.

### Read-only observation: exactly what tripped case11

Inspection used only saved token IDs, public Pi events, transport terminal and neighboring already-published episodes. It did not load a model, generate a token, replay an episode, alter an evaluator or regrade the tripped case. `gate-read/boundary-observation.json` binds the source hashes and matching event locations; the original transcript stays private in the run directory.

| Case11 turn (zero-based) | Saved frame | Action | Observed public summary |
|---:|---|---|---|
| 0 | structurally valid,83 tokens | read `result.txt` | “Reading the result file first.” |
| 1 | structurally valid,62 tokens | read `result.txt` again | “The file content is 'result.txt'. The newline is 'done'.” |
| 2 | structurally valid,48 tokens | finish `done` | “The file content is 'done'.” |

The **third turn's private Analysis is byte-for-byte identical to the second turn's already-public Summary** (56 characters). Frozen line63 collects every nonempty private Analysis/think string from every turn and tests substring membership against the **entire** Pi-events file, without turn ordering or field provenance. This matches the earlier public Summary in six event records (lines20,22,23,27,32,43: message_start, message_update twice, message_end, turn_end, agent_end).

Thus the observed trigger is **cross-turn text reuse / a lexical boundary-guard collision**, not malformed reasoning-close/boundary tokens or demonstrated transport serialization of a private field. The three frames passed the frozen structural validator. Pi exited0, reason finished, close_verified=true; model fingerprints, no-gradient and BF16 checks passed. The workspace remained empty: the model read the absent output file twice and finished without writing it, despite the requested write action. Its public claims about file contents were not backed by a created file. In neighboring cases, `choice-grep-symbol` and `choice-bash-command` each repeat read actions until the12-turn limit; `choice-edit-file` reads `state.txt` twice then finishes without the requested edit. These are **observations of B under the registered AdamW+cosine recipe at matched dose**, not a gate verdict or a causal attribution to annealing/optimizer identity.

The guard firing is preserved as data. The lexical-collision diagnosis does **not** authorize waiving the guard, rewriting its error into an ordinary episode failure, fabricating a summary, or resuming the interrupted instrument.

### Frozen verdict criteria applied without substitution

The verbatim predictions and sole prelaunch docNLL correction remain in the earlier **Frozen predictions** section and in `preregistration.json`; no post-observation threshold amendment has been made.

| Preregistered criterion / prediction | Preserved evidence | W3G disposition |
|---|---|---|
| EACH B repetition valid≥12 AND correct≥10 | no complete B repetition | **not evaluable** |
| B fixed-primary execution≥64/96 | B execution not run | **not evaluable** |
| min(B correct)−max(parent correct)≥4 | parent complete correct counts6,7,6; B incomplete | **not evaluable**; no imputed B score |
| min(B correct)−max(A correct)≥3 | A complete correct counts5,5,5; B incomplete | **not evaluable** |
| B≤8 or advantage<3 falsifies large-rescue prediction | B14-case counts/advantages unavailable | **not applied to partial data** |
| 9 correct = partial acquisition, gate failure | no complete B14-case score | **not evaluable** |
| A predicted historical correct4–8, execution59–60 | A correct5 in all reps; execution59,60,59,60 | descriptively within predicted ranges; not a decisive B comparison |
| If both arms improve/pass, old failure not uniquely explained by new regimen | B not evaluable; A scalar gates unmet | no such finding |
| Instrument divergence / nondeterminism → invalid/inconclusive, never negative optimizer finding | B guard fail-stop; parent/A greedy grade/reason divergence documented | **INVALID/INCONCLUSIVE** |
| Original saved-x docNLL trajectory only; no numeric bound | no current reads | not a pass/fail criterion; no retention conclusion |

**Final verdict: INVALID/INCONCLUSIVE.** Parent/A's completed measurements remain valid observed data, with all nondeterminism disclosed; their unmet raw scalar thresholds are not a negative optimizer-mechanism conclusion. B's interrupted gate read is not evaluable. The **large matched-dose regimen-rescue question remains OPEN**; neither rescue nor its preregistered falsification is established by this interrupted comparison. No checkpoint is promoted; no scale, robustness or unique-mechanism permission follows.

### Open operator-decision register

| Registered after observations | Authorized now | Required for any next read |
|---|---|---|
| 2026-10-20, after19:52:44Z stop and read-only case11 inspection; recorded in this publication | publish preserved partial evidence and accept INVALID/INCONCLUSIVE for this frozen attempt | either operator acceptance of inconclusive, **or** explicit operator authorization through the supervisor for an amended-instrument B re-read, preregistered with timestamps and the already-observed state before new sampling |

This register is **not** a proposed evaluator patch or read authorization. No retry/replay/resume is scheduled. Any amended-instrument re-read requires a separate operator decision; this report does not choose the amendment, waive the observed parent/A nondeterminism, or promise that a B-only re-read would establish the original identical-instrument comparison.

### Publication evidence and validation

All paths in this paragraph are under `receipts/w3-matcheddose/gate-read/`: `partial-verdict.json` contains complete parent/A tallies, per-case grades/reasons, all replica spreads and B's exact ten-row prefix; `parent/` and `A/` retain original summary/plan/rank/smoke bytes; `B/` retains the frozen plan, failure console, pin and model-after receipt; `supervision/` retains the continuous occupancy log, both window/pin receipts, process identities and terminal exit1. `raw-evidence-manifest.json` SHA-binds11,265 original raw files without publishing private transcripts. `publish-partial.py` is the exact offline archival/verification script, not an evaluator replacement; it confirmed every manifested original remained unchanged after inspection/publication. All three read-checkpoint SHA256 identities and all frozen source/instrument/docNLL-chain bindings were rechecked during publication.

Applicable safety intent: **R12** exact parent/state/source binding, **R14/NDP13** fail-stop and bounded teardown, **R16** immutable evidence discipline, with **R07/NDP15** checkpoint atomicity intent retained. This is local quality evidence, not resilient DiLoCo or Frontier/scheduler acceptance. The runtime failure is explicitly retained, not represented as a green end-to-end read. The receipt-only test suite checks completed summary arithmetic/bindings, all execution replicas and fixed primary, incomplete-B nonclassification, lexical-collision metadata, and fail-stop/window receipts. Final command: `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_w3_gate_partial_publication.py tests/test_watch_w3_gate_window.py tests/test_w3_regime_and_gate.py tests/test_w3_matcheddose_preflight.py tests/test_w3_disk_preflight.py`: **38 passed in11.13s** (`gate-read/tests-final.txt`). Frozen Bash syntax and `git diff --check` also passed. The actual frozen read remains explicitly **exit1**, not a passing test result.
