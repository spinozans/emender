# W3 matched-dose wash-2 regime experiment

Status: **preflight in progress; neither production arm launched; branch WAIT_X9b_FINAL**.
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
| docNLL | 8-doc / 198,295-target panel `5725d820…`, saved-x | authority found; trajectory-only under original operator ruling; see amendment |
| AdamW serialization | explicit typed class/version, live-nonaveraged point; frozen args unchanged | 47 identity/facade/precision tests pass |
| AdamW viability | full production BF16 update and both-mode reload/hash equality | pending |
| Regime branch | X9b final receipt only; no partial-score branching | waiting; final receipt absent |
| Topology / training | single GPU per arm; free cards 6/7 observed, not reserved | no production launch; topology not yet selected |
| Preregistration / final reads | no threshold stops, retries or best checkpoint; pin before any gate read | pending runner binding/validation and preregistration commit |

All local receipt paths above are under `receipts/w3-matcheddose/` unless absolute. The raw historical receipts are unmodified. The independent recount decodes little-endian indexes and drops the first target of every independent record; it does not trust manifest target totals or call the training Dataset to count targets.

## Frozen predictions (verbatim from section 6)

Strong regimen-rescue hypothesis predicts B reaches **valid≥12, correct≥10, primary execution≥64/96**, clears every existing retention bound, and gains **≥4 correct actions against the contemporaneous parent6**, not merely5→6 or one lucky execution replica. To establish relative regimen benefit, B must also exceed A by **≥3 correct actions** (outside the historical ±1–2 band) without sacrificing the execution floor. A is predicted to remain in the historical correct-action4–8 range and may reproduce execution59–60; this is a falsifiable prediction, not a required outcome. B≤8 correct, or <3 advantage over A, fails the **large matched-dose regimen-rescue** prediction; 9 correct is partial acquisition but gate failure. If both arms improve/pass, old failure was not uniquely explained by the new regimen. If B improves Stage-B but execution/retention fails, the dual gate still fails. If either arm's identity, dose or instrument differs, classify **invalid/inconclusive**, not a negative optimizer finding.

The retention-bound phrases in that verbatim prediction are subject only to the preregistered authority correction: there are no numeric docNLL pass/fail limits; docNLL is diagnostic trajectory, not a stop signal. The 2026-10-02 operator ruling supersedes invented comparative bounds/noise bands. Behavioral thresholds and all other predictions remain verbatim.

## Read protocol and hold

A = contemporaneous SF constant 1e-5, warmup0. B = SF external cosine if X9b final P2+ materiality ≥50/512; otherwise operator-preselected AdamW+cosine bundle (.9,.95), eps1e-8, wd.01. Both use exactly 512 identical batches and target-normalized loss, with entry and fixed final stored. Schedule is `1e-7 + .5*(1e-5-1e-7)*(1+cos(pi*(t-1)/511))`, t=1..512; no adaptive dose, retries or best-checkpoint selection. Log scheduled/actual LR, per-step targets and cumulative objective-weighted exposure. Parent/A/B behavioral primary is train-y; saved-x is diagnostic. Three unchanged greedy Stage-B repetitions and four execution replicas per checkpoint; report all spreads, never average/best-rescue. No gate read before pinning the read checkpoint and receipt.

## Validation

- `.venv/bin/python scripts/w3_matcheddose_preflight.py --output receipts/w3-matcheddose/preflight.json`: passed; exact 512-step exposure and frozen instrument hashes.
- `PYTHONPATH=$PWD OMP_NUM_THREADS=2 .venv/bin/python scripts/w3_verify_parent.py --output receipts/w3-matcheddose/parent-conversion.json`: passed. First attempt exposed tied embedding/head alias (237 state-dict entries versus 236 unique parameters); verification now derives named-parameter layout from the exact production model on meta and independently verifies the alias. No data/seed changes or training retries occurred.
- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_w3_matcheddose_preflight.py tests/test_e97_facade.py tests/test_e97_sft_precision_integration.py`: **47 passed in 22.83s**. Covers independent boundary masking, hard mismatch stop, typed optimizer conflicts in both modes, AdamW point-hash equality, SF typed-versus-legacy byte equality, existing facade and precision regression tests.

Applicable architecture intent: **R07/NDP15 atomic publication, R12 exact parent/state binding, R14/NDP13 bounded fail-stop execution, R16 immutable evidence discipline**, per `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` ADR-003 and its gap matrix. This is local world1 full-model SFT, not resilient systems acceptance, Frontier submission, or a new scale rung. Elastic/native/async requirements remain explicitly unclaimed. No WG service, scheduler, dispatcher or workers started. GPU lease utility alone is used for resource exclusion as explicitly required by W3.

## Results

| checkpoint | Stage-B repetitions (valid/correct) | Execution replicas /96 | Primary | saved-x docNLL |
|---|---|---|---|---|
| parent contemporaneous | pending | pending | rank-pair [0,1] fixed | pending |
| A fixed u512 | not trained | not trained | rank-pair [0,1] fixed | pending |
| B fixed u512 | not trained | not trained | rank-pair [0,1] fixed | pending |

Verdict: **not read; no optimizer finding**. Final verdict receipt/commit/push follow completed fixed training and frozen reads, not a partial-score decision.
