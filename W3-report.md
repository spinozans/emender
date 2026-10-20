# W3 matched-dose wash-2 regime experiment

Status: **launch-ready; X9b final DONE and full-trajectory terminal attestation verified. B is fixed to the operator-preselected AdamW+cosine regime bundle. The ≥200GiB root-volume gate remains mandatory immediately before each arm.** W3R's sole P1 condition is addressed and regression-tested. Preregistration SHA256 `4976d3761102c6e28cf556a8b897db397edc3e0e813758644f83d2053420c52a` (supersedes preserved pre-W3R/pre-capacity snapshots).
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
| Topology / training | **serial A then B, one leased free card from 6/7**, world1 per arm | selected before results; actual card/UUID retained at launch; no production launch |
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
- `bash -n scripts/w3_run_adamw_viability.sh scripts/w3_run_arms.sh scripts/w3_run_gate_reads.sh` and `git diff --check`: passed. No production-arm or gate invocation occurred.

Applicable architecture intent: **R07/NDP15 atomic publication, R12 exact parent/state binding, R14/NDP13 bounded fail-stop execution, R16 immutable evidence discipline**, per `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` ADR-003 and its gap matrix. This is local world1 full-model SFT, not resilient systems acceptance, Frontier submission, or a new scale rung. Elastic/native/async requirements remain explicitly unclaimed. No WG service, scheduler, dispatcher or workers started. GPU lease utility alone is used for resource exclusion as explicitly required by W3.

## Final X9b branch receipt

X9b completed its fixed500K ceiling with successful original DONE/terminal/result. The committed/tested completion-only attestor bound all1001 validation rows/order, original config/data/frozen pool and terminal evidence; `x9b-completed-trajectory.json` SHA `5eeee50585408ced35bc105945eb2c06b02a3dc2f6ed2bf9980b24c267121130`. Full trajectory SHA `42c859d94706e8c42dbfbd57a8f3eb1adc9a064c3f97c4280f533c6428ede22f`.

Maximum P2+ was **11/512**, below the registered50/512 materiality bar. Fixed-final vector was `[95,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0]`. Arm B is therefore **ordinary AdamW+the exact external cosine**, betas(.9,.95), eps1e-8, wd.01, internal warmup0. A remains contemporaneous fresh SF constant1e-5. This is explicitly the operator-preselected **regime-bundle** contrast, not unique attribution to SF averaging. No partial X9b score selected the branch. `fixed-branch.json` records the complete evidence chain and both maximum/fixed-final statistics.

## Prelaunch rulings and runbook

Before any production results, supervisor approved: (1) docNLL original trajectory-only ruling supersedes nonexistent retention limits; (2) typed AdamW live-point serialization with train/saved equality, without SF shim state or changing frozen args; (3) X9b **max-anywhere P2+≥50/512 over its completed trajectory**, with fixed-final counts alongside and escape-without-consolidation disclosed; (4) EACH Stage-B repetition must pass, and advantages use min(B)−max(parent)≥4 and min(B)−max(A)≥3, also reporting historical parent6. Any divergent case grades/reasons or tallies are a serving-nondeterminism finding, not independent samples or an averaged rescue.

**New infrastructure precondition:** after the root volume filled and an unrelated X6 run failed with ENOSPC, the supervisor required ≥200G of root-volume free space at each W3 arm start. `w3_run_arms.sh` now invokes `w3_disk_preflight.py` immediately before EACH training command (after lease acquisition), conservatively enforcing200GiB and writing `arms/{A,B}-disk-gate.json` with free bytes/threshold/time. A blocked gate writes a terminal reason, FAIL marker and exits; there is no automatic retry. Concurrent capacity pruning does not waive this check. This amendment is preregistered; the three new CPU tests cover both arms, inclusive threshold, logged nonzero failure. Neither the incident nor the capacity gate is an optimizer finding.

Next actions, strictly in order:

1. Await existing X9b watcher/verified 500K DONE. First run `.venv/bin/python -m scripts.w3_attest_x9b_completion` (enforces its source/tests/green test evidence are committed). It creates `x9b-completed-trajectory.json` once, binding original DONE/terminal/result/config/data/frozen-pool and the complete1001-row validation digest/order without modifying any original receipt or live trainer. Then run `.venv/bin/python -m scripts.w3_fix_regime_branch --output receipts/w3-matcheddose/fixed-branch.json`. Missing/failed/partial/malformed/tampered terminal or supplemental evidence hard-stops before reading intermediate counts. Inspect/report completed maximum and final values; commit the branch receipt **before either arm**. The training runner enforces both preregistration and branch bytes are committed.
2. W3R review is persisted in `W3R-review.md`; its sole P1 condition (full trajectory digest in consumed terminal evidence, including intermediate49→50 and missing-receipt rejection) is implemented and covered by the60-test suite. Supervisor approved the supplemental completion-only attestation seam because the live Bash publisher was already parsed at launch. After the final attestation/branch receipt is committed, launch `nohup setsid bash scripts/w3_run_arms.sh > /home/erikg/emender-scratch/w3-arms-console.txt 2>&1 < /dev/null &`. One serial pair, same leased card, **fresh ≥200GiB root-volume check immediately before each arm**, bounded3h per arm, no retries. Arm directories are create-once. Monitor `.venv/bin/python -m scripts.watch_w3_arms`; it checks live PID/starttime and terminal receipts, never launches, retries or grades.
3. After both fixed-final completions, obtain a **free eight-GPU window** and run the detached `scripts/w3_run_gate_reads.sh`. This is required by the unmodified frozen execution CLI; current cards0–5 are occupied, so no gate launch is claimed. It fails before any read if eight free GPUs cannot be leased. Parent/A/B are re-pinned before reads; same three Stage-B repetitions, exact four execution replicas with lowest rank primary; original saved-x docNLL scorer; pinned verdict receipt.
4. Fill the full results table from pinned case-level receipts, apply predictions verbatim with only the disclosed prelaunch amendments, commit and push. No quality verdict or checkpoint promotion before then.

The ordinary-AdamW viability establishes one finite full-size update and exact point serialization, not a 512-update quality outcome. All original historical receipts remain unchanged. The report's readiness is **conditional on X9b final plus completion-only attestation, committed branch, and a green root-capacity gate at each actual arm start**, never a claim that production training/readout finished. Original W3R review was BLOCKED pending P1; that historical review is preserved unedited, with remediation/test evidence disclosed here.

## Results

| checkpoint | Stage-B repetitions (valid/correct) | Execution replicas /96 | Primary | saved-x docNLL |
|---|---|---|---|---|
| parent contemporaneous | pending | pending | rank-pair [0,1] fixed | pending |
| A fixed u512 | not trained | not trained | rank-pair [0,1] fixed | pending |
| B fixed u512 | not trained | not trained | rank-pair [0,1] fixed | pending |

Verdict: **not read; no optimizer finding**. Final verdict receipt/commit/push follow completed fixed training and frozen reads, not a partial-score decision.
