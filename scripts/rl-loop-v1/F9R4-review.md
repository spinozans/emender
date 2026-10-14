# F9R4 independent round-4 re-review — PASS-WITH-CONDITIONS

**Deploy verdict: PASS-WITH-CONDITIONS — C1 separate attended operator approval; C2 existing live release/ownership/import/stop-drain/watchdog/rollback/seed GO gates.** CPU/internal command-grammar QA **PASS**; F9R3 B1-R3 is closed, with no implementation blocker found. This clears the independent CPU review gate for **F9G**, not permission to deploy or claim fresh Pi qualification (`S/F9-deployment.md:87-110,282-295,299-318`; `E/F9R4-closure.log:1-6,60`; `E/F9R4-byte.log:10-13,42-45`). A nonblocking stale runbook label is recorded below, not silently repaired.

## Scope, authority and citation roots

Reviewed commit **9fd8b47fba174e15f108881f1db69c876d621611**, lambda01, project Python **3.12.3** (`E/F9R4-scope.log:1-4`; `E/F9R4-regression.log:2`). Read the supplied task, F9R3 report, scratch F9C repair report and the **repo report's appended F9G section** (the supplied scratch F9C report stops before F9G; current contract is `S/F9C-repair.md:226-303`). Review-only for implementation/authority/runtime bytes: new QA scripts, logs, JSON and this report are review artifacts. No prepare/regenerate write entrypoint, live Pi/teacher API, GPU training, Slurm, deployment, bank admission/injection, daemon/watchdog stop/restart or live rollback was invoked. Candidate negatives and new boundary controls were admission/synthetic-grading only; only the previously authorized 13 equivalence controls executed in isolated CPU workspaces (`E/F9R4-closure-qa.py:105-158`; `E/F9R4-closure.log:60`; `E/F9R4-grammar-qa.py:151-180`). Bank processes/signals are CPU fixtures, not live owners (`S/loop/rl_bank_unit_test.py:2109-2290`; `E/F9R4-bank.log:310-324`). Bytecode and pytest cache writes were disabled (commands below).

Citation roots:

- `S` = `/home/erikg/emender/scripts/rl-loop-v1/F9-staged`.
- `Q` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9G-final`.
- `H` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final` (historical receipts only).
- `E` = `/home/erikg/emender-scratch/fix-fanout`.
- Unprefixed paths are relative to `/home/erikg/emender`.

Q/H binding is explicit (`S/F9-evidence.json:172-173,211`). Every staged runtime/provider byte was independently compared to Q; S source citations also apply to Q (`E/F9R4-byte.log:11`). Probe imports explicitly assert Q driver and Q shell paths, not deployed fallback (`E/F9R4-closure-qa.py:21-28`). Scratch and repo reports are byte-identical (`E/F9R4-worktree.log:1-4`).

Applicable IDs: **R14/NDP13 bounded local safety intent and R16 exact-source/execution evidence discipline**; no distributed/resilient-training, dynamic membership, overlap, convergence, scale or scheduler conformance is claimed (`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:113-122`; `docs/RESILIENT_DILOCO_GAP_MATRIX.md:33-39`).

## 1. F9R3 blocker closure — PASS

### One authored bound; admission/grading closure

`MAX_SHELL_LENGTH = 4096` is authored once at `S/scripts/e97_offline_shell.py:13-14`. Raw parsing uses that symbol at `:137-139`; `shell_allowed` serializes parsed structure, then checks **canonical** length before returning at `:182-186`. `action_allowed` supplies only this admitted canonical command at `:195-204`. This is a **character** bound (`len(str)`), not an asserted UTF-8-byte maximum; byte-level source/hash equality is separately verified. All new length probes are ASCII, so their logged character/UTF-8-byte lengths coincide (`E/F9R4-closure-qa.py:65-101,153-160`; `E/F9R4-closure.log:1-54`).

The existing generator reads that exact shell source, strips only its external mechanics import and embeds the whole remaining source, including the constant (`S/sync_validator.py:7-17`). The sealed definition and both uses are at `S/scripts/e97_first_party_validator_diversity.py:846-847,970-972,1015-1019`. This is generated duplication, **not an independent retyped bound**. In-memory generation equals both S and Q byte-for-byte; all nine shell function ASTs and the binding agree in both trees (`E/F9R4-byte-qa.py:53-77`; `E/F9R4-byte.log:13,42`). Sealed grading routes every diversity action through that same `action_allowed` before either grading-mode branch (`S/scripts/e97_first_party_validator_diversity.py:1083-1090`). No candidate command is dispatched by this grading call.

For the same input representation, direct/reference/sealed admission and real sealed grading now reject for the same reason. Policy rejects before dispatch, clears pending and drops the public call; its public stop reason intentionally abstracts the internal ValueError (`S/scripts/e97_pi_scripted_v8.py:39-59`; `E/F9R4-closure-qa.py:105-156`).

| Exact F9R3 source-named case | Raw -> canonical length | Direct/sealed/reference/policy | Real focused + regression grades |
|---|---:|---|---|
| ordinary-raw-4091 | 4091 -> 4097 | REJECT | Raw REJECT: canonical length; canonical REJECT: raw input length |
| ordinary-raw-4096 | 4096 -> 4102 | REJECT | Same |
| apostrophes-817 | 829 -> 4101 | REJECT | Same |
| apostrophes-1000 | 1012 -> 5016 | REJECT | Same |

Exact commands are independently reconstructed, not copied from the repair's test rows (`E/F9R4-closure-qa.py:56-64`). Exact gate/grade receipts are `E/F9R4-closure.log:1-4`. Both neighboring controls canonicalize to **4096**, accept at all four gates and pass both modes in both representations (`:5-6`). On raw blocker inputs the message is `canonical offline shell exceeds length bound`; on the already-over-bound canonical strings it is `bounded offline shell required`. That representation-specific distinction is correct, **not a remaining admission/grading disagreement** (`S/scripts/e97_offline_shell.py:137-139,182-186`; `E/F9R4-closure.log:1-6`).

The permanent tests retain the four exact blockers, raw exact/one-over and canonical ordinary/apostrophe exact/one-over boundaries, gate idempotence and both real grading modes (`S/test_f9g_canonical_bound.py:22-32,72-90,106-131`). Grading inputs borrow historical end-state fields only to isolate the action gate: **they are not proof that any new probe executed or solved the historical task** (`:1-5,93-131`; `E/F9R4-closure-qa.py:36-40,140-156`).

## 2. No independent bound / drift guard / embedding integrity — PASS

Repo and integrated-tree grep receipts enumerate `4096` and `MAX_SHELL_LENGTH` occurrences (`E/F9R4-bound-grep.log:1-35`; `E/F9R4-runtime-bound-grep.log:1-33`). Shell-length comparisons use only `MAX_SHELL_LENGTH`; other actual numeric limits cover archive-member count, model output tokens, durable backlog, author prompts or natural workspace content, not shell length (`S/ndm/e97_first_party_read_observe.py:192`; `S/configs/pi/e97-pi-native-scripted-v8.ts:24`; `S/loop/rl_correction_config.py:26-27`; `Q/loop/teacher_author.py:554,601,633`). Workspace 150–4096 references are instruction strings, not independent executable shell bounds (`S/scripts/e97_diversity.py:23,39`). Independent AST inspection in **both S and Q** finds exactly one numerical 4096 AST constant per shell/embedded implementation and no second such literal (`E/F9R4-byte-qa.py:59-77`; `E/F9R4-byte.log:14-42`).

The permanent drift guard compares all direct function ASTs with the sealed copy, requires exactly one assignment per implementation, compares the assignment ASTs, checks runtime constant equality, and prohibits another bound-value AST literal (`S/test_f9g_canonical_bound.py:50-69`). It is supplemented by in-memory generation equality at the fixture (`S/test_f9r2_grammar.py:85-95`), so stale embedded source is not silently accepted.

**Sensitivity was exercised without touching staged/authority bytes.** The real permanent test first passes on temporary source copies, then fails as expected for (1) sealed constant 4097 drift, (2) `>` to `>=` canonical-check drift, and (3) a second independent 4096 assignment. Separately, changing only the temp direct constant propagates 4097 into generated source with no retained 4096 binding; removing the temp shell source raises FileNotFoundError rather than using a cached fallback (`E/F9R4-closure-qa.py:165-205`; `E/F9R4-closure.log:55-59`). Final pin checks still pass after these experiments (`E/F9R4-postreview-byte.log:10-13,42-45`).

Construction guarantee is scoped to the **reviewed exact-source closure**: one authored constant, generated sealed copy, equality guards and sealed pins, not a claim that unrelated future unreviewed edits cannot break anything (`S/sync_validator.py:15-17`; `S/prepare.py:20-25`; `E/F9R4-byte.log:13,42-45`).

## 3. New independent closure negatives — PASS

Added **48 distinct new strings**, disjoint from the retained 117 matrix, 237 F9R3 adopted cases and permanent F9G boundary rows. They comprise **24 new rejecting attempts + 24 accepting boundary controls**, plus the six historical probes, for **54 total / 28 rejects / 26 accepts**. All four gates agree; **216 real sealed mode/form verdicts** agree with the independent expected decisions; **zero candidate executions and zero unexpected cases** (`E/F9R4-closure-qa.py:65-107,108-158,208-212`; `E/F9R4-closure.log:7-54,60`).

The independent expected serializer is constructed from literal words and checked with POSIX shlex punctuation decoding, not the implementation's parser/serializer (`E/F9R4-closure-qa.py:42-55`). New interactions:

- Canonical targets **4095/4096/4097/4098**, with tabs, empty/concatenated quote fragments, double quotes, split quoting and whitespace padding (`:66-74`; `E/F9R4-closure.log:7-11,17-21,27-31,37-41`).
- Apostrophe-escaping expansion combined with a literal tail, plus empty argv words (`:75-80`; `E/F9R4-closure.log:12-13,22-23,32-33,42-43`).
- Operator-adjacent redirects, sequential commands and approved pipeline forms whose separator/word quoting pushes canonical output over the bound (`:81-89`; `E/F9R4-closure.log:14-16,24-26,34-36,44-46`).
- Raw **4095/4096/4097/4098** via leading tabs/trailing spaces, with a short **24-character canonical** form (`:90-101`; `E/F9R4-closure.log:47-54`). Over-bound raw forms reject at admission and raw grading; the independently supplied short canonical form legitimately accepts at direct/sealed grammar and canonical grading. It is a different valid input, never dispatched from a rejected raw frame (`E/F9R4-closure-qa.py:121-156`; `E/F9R4-closure.log:51-54`).

No admitted canonical action outside the enumerated grammar and no same-representation admission/grading disagreement was observed. New accepting controls also verify pending/returned/public-history canonical bytes and raw failure/PG retention; the separate retained matrix rerun exercises canonical ingest and honest projection (`E/F9R4-closure-qa.py:120-140`; `E/F9R4-grammar-qa.py:61-92`; `E/F9R4-grammar.log:355`). This is not an OS sandbox or arbitrary-shell-equivalence claim (`S/scripts/e97_offline_shell.py:1-6`).

## 4. Independent full batteries and regression sweep — PASS

All counts match the repair's exact claims. They overlap and are **not summed as unique cases** (`S/F9C-repair.md:281-285`).

| Battery independently rerun | Result | Receipt |
|---|---|---|
| Permanent combined regression | **417 passed**, 12.14s | `E/F9R4-regression.log:6,18` |
| F9 CPU | **74 passed / 3 deselected**, 3.30s | `E/F9R4-cpu.log:6,12` |
| Same Q tree bank CPU fixtures | **215 passed** | `E/F9R4-bank.log:324` |
| Original legacy CPU selection | **44 passed / 5 real-Pi deselected**, 9.47s | `E/F9R4-legacy44.log:6,13` |
| Full legacy validator set | **59 passed**, 11.21s | `E/F9R4-validators.log:6,13` |
| Independent original 117 matrix | **104 reject / 13 accept**, four gates agree | `E/F9R4-grammar-qa.py:99-105`; `E/F9R4-grammar.log:1-117,355` |
| Retained reviewer 233 negatives + 4 literal controls | **233 reject / 4 accept**, zero misses | `E/F9R4-grammar.log:118-355` |
| Authorized equivalence | **13** argument/exit/stdout/stderr/all-file-byte equivalences | `E/F9R4-grammar-qa.py:151-180`; `E/F9R4-grammar.log:355`; `E/F9R4-grammar-results.json` (`equivalences`) |
| New boundary/embedding QA | **54 probes / 216 grades / 5 temp integrity checks**, zero misses | `E/F9R4-closure.log:1-60` |

The 117 matrix is folded into 417 and additionally rerun from the **original reviewer JSON**, not merely the repair's matrix constructor; retained baseline SHA256 is `fd938f83067a37985e8a1f99ed989db87fab41aee0d60e00df7d63d1daa5e5d4` (`E/F9R4-grammar-qa.py:99-105`; `E/F9R4-grammar.log:355`). New round-4 boundary probes are independently authored scratch-only tests, not edits to the permanent battery (`E/F9R4-closure-qa.py:42-101`).

Prior PASS findings remain exercised:

- **Versioning/conversation/ordinary finish:** deterministic authorized continuation and complete exchange, legacy irreversible finish, old validator verdict/proof compatibility (`S/unit_tests.py:119-148,178-207,293-307`; `E/F9R4-cpu.log:12`; `E/F9R4-validators.log:13`).
- **Rotation/knobs:** exact 60-call schedule, tranche **96**, lanes **24/24/48/48** (`S/loop/standing_supply.py:126-129,151-175`; `S/unit_tests.py:170-175,434-441`; `E/F9R4-cpu.log:12`).
- **Morph/private/seed gates:** revalidation/dedupe/split isolation, private prescreen/harvest, verified supply intake and explicit disable before seeded construction (`S/unit_tests.py:151-168,310-317,353-362,487-550`; `E/F9R4-cpu.log:12`).
- **Fresh durable conversation correction:** actual fixture manager/worker path, workspace/prefix reset, fresh zero-prefix publication and exactly-once claim retirement (`S/loop/rl_bank_unit_test.py:2109-2200`; `E/F9R4-bank.log:310,312-313,324`). Current helper byte identities still match (`E/F9R4-byte.log:12`).
- **End-state record honesty:** canonical executed actions/messages, raw native/source PG evidence retained, ordinary projection unchanged and correctly labeled (`S/loop/rl_loop_driver.py:369-370,469-475,530-533,567-572`; `S/test_f9r2_grammar.py:192-214`; `E/F9R4-grammar-qa.py:64-92`; `E/F9R4-regression.log:18`; `E/F9R4-grammar.log:355`).
- **Canonical-only execution trace:** driver selects safe/scripted bridges; bridge validates before returning and rewrites pending/returned/history; common socket sends only the returned result, error responses expose no call; provider emits only response block arguments (`S/loop/rl_loop_driver.py:350-376`; `S/scripts/e97_pi_scripted_v8.py:39-59,62-75`; `scripts/e97_pi_native_tool_transport.py:48-59`; `S/configs/pi/e97-pi-native-scripted-v8.ts:23,26-28`; `configs/pi/e97-pi-native-frozen-tools-v2.ts:33,36-38`). The independent Q harness checks these execution-facing representations, raw PG retention, next-call ingest and canonical projection (`E/F9R4-grammar-qa.py:61-92`). This remains static wire tracing + synthetic bridge evidence, not a new live Pi run.

F9G does not edit driver, providers, rotation, conversation or correction runtime; commit path inspection corroborates the narrow shell/validator/test/pin/documentation change (`E/F9R4-scope.log:5-15`). Historical terminals still regrade and missing-output mutants reject, but historical Pi/protected-overlap evidence is not relabeled fresh against F9G (`S/unit_tests.py:571-586`; `S/F9-evidence.json:172-173`; `E/F9R4-cpu.log:12`). Three F9 Pi-dependent and five legacy real-Pi tests remain excluded, not relabeled passing (`E/F9R4-cpu.log:6,12`; `E/F9R4-legacy44.log:6,13`).

## 5. Byte census and rebinding — PASS

Independently enumerated **136 pin occurrences**: **22 staged + 60 integrated runtime + 17 historical artifacts + 12 historical qualification + 2 F9E regression + 7 F9E qualification + 4 F9G regression + 8 F9G qualification + 4 retained reviewer artifacts**. All match; counts are occurrences, not distinct files (`E/F9R4-byte-qa.py:21-47`; `E/F9R4-byte.log:1-10`; full per-occurrence paths/digests in `E/F9R4-pin-occurrences.json:1-682`). The repair verifier also independently reruns successfully, but is not the sole census evidence (`E/F9R4-adopted-byte.log:1-15`).

Generation equality, prepare's bound validator pin, all staged/runtime bytes and current correction helpers match. Old **17** and additive **18** closures parse; unauthorized extra component rejects; all 18 archive members equal Q source and manifest pins; era2/3/4 bytes equal retained repo validators; both admitted first-party registry pins equal the current archive (`E/F9R4-byte-qa.py:48-102`; `E/F9R4-byte.log:11-13,42-45`; `S/prepare.py:20-25,73-79`).

- Validator SHA256: **975d389c085bdf97f670d88e66629a7547c64074801c7f8df009ddd488f2bf90**.
- Source archive SHA256: **73ea83a69f2ae79d2fc8748b26d1d61e3da1cb0fce23db4e2ef322dd91c1f954**.
- Independent receipts: `E/F9R4-byte.log:44-45`.

Original 111-pin census + nine extra F9E pins also rerun; documented restore code aborts before destination changes on corrupt backup and restores only temporary fixtures on valid backup; exact F9_TOOLS assignment resolves the retained rollback script with `--help`, no bank touched (`E/F9R4-old-byte.log:1-17`). Final post-review census again matches all 136 occurrences (`E/F9R4-postreview-byte.log:1-13,42-45`). Historical live-source annotations are intentionally not tested as current-source hashes (`S/F9-evidence.json:150-151,172-173`; `E/F9R4-byte.log:10`).

## 6. Conditions and nonblocking finding

**C1 — Operator authority:** separate attended operator approval remains required; this read-only QA grants none (`S/F9-deployment.md:87-90`).

**C2 — Existing live gates:** bind release/import/archive/registry/seed manifests to this reviewed **F9G** commit and digests; execute the existing ownership/stop-drain/watchdog/restart/rollback boundary and, if seeded rollout is intended, verified seed-intake GO checks in the attended deployment window. CPU fixtures do not certify actual live owner/PID/import/environment state (`S/F9-deployment.md:97-111,125-160,206-295,299-318`). Fresh Pi transport/protected-overlap qualification remains unclaimed, as the repair itself states (`S/F9C-repair.md:299-303`).

**N1 — Nonblocking runbook label:** `S/F9-deployment.md:88` still calls the closure “F9E”, while `:10` correctly names `qualified-F9G-final`. This is a stale predecessor label, not a byte/admission failure. Use **F9G/9fd8b47f and the explicit current digests above** for any attended release; optional documentation cleanup must not regenerate or alter reviewed authority bytes. No source repair is requested by this QA.

An unrelated concurrent unstaged modification of `configs/pi/e97-firstparty-collection-authorizations-v1.json` appeared during review; it was neither made nor reverted here. Existing untracked F9R2/F9R3 repo review reports were left untouched (`E/F9R4-scope.log:16-23`; `E/F9R4-worktree.log:5-14`). No files staged; no reviewed implementation change (`E/F9R4-worktree.log:5-14`; final pin census above).

## 7. Reproduction

Run on lambda01 (not Frontier); no prepare write entrypoint. Commands were run sequentially, with separate new logs. The earlier combined command's final log-summary `tail -2` expansion failed **after all suites completed successfully**; the corrected per-file tail merely displayed existing results. No test failed (`E/F9R4-regression.log:18`; `E/F9R4-cpu.log:12`; `E/F9R4-bank.log:324`; `E/F9R4-legacy44.log:13`; `E/F9R4-validators.log:13`).

```bash
source /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9G-final/activate.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/erikg/emender/.venv/bin/python
S=/home/erikg/emender/scripts/rl-loop-v1/F9-staged
E=/home/erikg/emender-scratch/fix-fanout
cd "$F9_STAGE_ROOT"
"$PY" -m pytest -q -p no:cacheprovider "$S/test_f9r2_grammar.py" "$S/test_f9g_canonical_bound.py"
"$PY" -m pytest -q -p no:cacheprovider "$S/unit_tests.py" -k 'not real_pi and not failed_reference_solution'
"$PY" loop/rl_bank_unit_test.py --scratch "$E/F9R4-bank-fixtures"
"$PY" "$S/verify_f9g_evidence.py"
"$PY" "$E/F9R2-byte-qa.py"
"$PY" "$E/F9R3-extra-byte-qa.py"
"$PY" "$E/F9R4-byte-qa.py"
"$PY" "$E/F9R4-closure-qa.py"
"$PY" "$E/F9R4-grammar-qa.py"
cd /home/erikg/emender
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator*.py
grep -RInE '4096|MAX_SHELL_LENGTH' "$S"
grep -RInE '4096|MAX_SHELL_LENGTH' "$F9_STAGE_ROOT" --exclude-dir=author-work --exclude-dir=__pycache__
git diff --check
git diff --cached --name-only
cmp "$E/F9R4-review.md" scripts/rl-loop-v1/F9R4-review.md
```

## Disposition

**PASS-WITH-CONDITIONS (C1/C2); CPU grammar deployment gate PASS.** The four F9R3 blockers reject before dispatch; source-bound admission and sealed grading agree; permanent and independent new negatives, exact-count batteries, drift sensitivity and 136 pins pass. No implementation blocker remains in this review. Preserve immutable historical evidence and canonical execution/receipt honesty. Next step is parent/operator review of this report, then only separately approved attended runbook gates—not deployment by this reviewer (`E/F9R4-closure.log:1-6,55-60`; `E/F9R4-byte.log:10-13,42-45`; `S/F9-deployment.md:87-110,282-295`).
