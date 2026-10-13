# F9R independent staged review — FAIL

**Deployment gate: FAIL. Do not deploy these staged bytes as written.**

Reviewed on lambda01 against repository HEAD `1628b8873546b836516220b9f6d122f106332b83`; F9 staging commit is `ed3ed7451cbf50051d2d3d440dfc2415e94edf6d`. Only this report is changed. No Pi episode, teacher API, GPU operation, process inspection/signal, daemon operation, admission, preparation/harvest wrapper, or live pin replacement was run. CPU tests use temporary fixtures; Python bytecode and pytest cache writes were disabled. Historical daemon/PID assertions were not re-certified.

## Citation roots

- `S` = `/home/erikg/emender/scripts/rl-loop-v1/F9-staged`.
- `L` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged`.
- `Q` = `L/qualified-final` (the designated evidence checkout: `S/F9-deployment.md:8-9`).
- `P` = `Q/author-work/teacher-authored/tranche-test-diversity-test_real_pi_new_families_and_0`.
- `E` = `/home/erikg/emender-scratch/fix-fanout`.
- Unprefixed source paths are relative to `/home/erikg/emender`.

All shorthand citations below denote these exact files and line numbers. The reviewed scratch report adds a scoped-commit receipt at `E/F9-report.md:210-217`; the tracked report ends at `S/F9-report.md:208`.

## Blockers

### B1 — Allowlist admits shell expansion before dispatch (safety)

`S/scripts/e97_offline_shell.py:21-24` excludes `$`, backticks, backslashes and newlines from literals, but not brace expansion, tilde expansion or globbing. `printf` permits these literals without path validation (`:55-59`). `shell_allowed` strips quoting with POSIX shlex (`:78-86`), then validates normalized tokens rather than constructing a safe command. The original command is returned unchanged for tool execution: `S/scripts/e97_pi_scripted_v8.py:39-53`, `scripts/e97_pi_native_tool_bridge.py:75-76`.

CPU-only validation (no shell/transport execution) confirmed all three:

```text
printf %s {1..999999999}  reference guards=ACCEPT; policy=toolUse; pending command unchanged; failed=False
printf %s ~root          reference guards=ACCEPT; policy=toolUse; pending command unchanged; failed=False
printf %s *              reference guards=ACCEPT; policy=toolUse; pending command unchanged; failed=False
```

Reproduce after the staged environment/cwd activation shown below (project Python, validation only):

```python
import runpy, tiktoken
from copy import deepcopy
import teacher_author as author
import rl_loop_driver as driver
from scripts.e97_pi_native_codec import native_turn
from scripts.e97_pi_native_tool_bridge import MODEL, PROVIDER
from scripts.e97_pi_scripted_v8 import SafeWorkspaceBridgeV8
from scripts.e97_diversity import canonical
fixture = runpy.run_path('/home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py')['terminal_fixture']
enc = tiktoken.get_encoding('p50k_base')
c = driver.load_curriculum()
tools = driver.tool_manifest(c)['model_visible_tools']
for command in ['printf %s {1..999999999}', 'printf %s ~root', 'printf %s *']:
    spec = fixture()
    spec['solution']['actions'][1]['arguments']['command'] = command
    author.guards(spec, set(), set())
    frame = native_turn({'role': 'assistant', 'content': None,
        'reasoning_content': 'Check admission only; do not execute.',
        'tool_calls': [{'id': 'probe', 'type': 'function', 'function': {
            'name': 'bash', 'arguments': canonical({'command': command})}}]}, tools)
    b = SafeWorkspaceBridgeV8(driver.make_panel(c.SYSTEM, tools, driver.POLICY_PANEL),
        'opening', enc, lambda *a: (frame, enc.encode_ordinary(frame), 'valid'))
    result = b.next({'systemPrompt': c.SYSTEM, 'model': MODEL, 'provider': PROVIDER,
        'tools': tools, 'messages': deepcopy(b.history)})
    print(command, result['stop_reason'], b.pending['arguments'], b.failed)
```

This contradicts the mandatory no-expansion/no-glob grammar (`S/scripts/e97_diversity.py:337-344`) and the bounded offline safety claim (`S/F9-report.md:66-77`). In particular, the huge brace range can consume host resources *during shell expansion*, outside the parser's 128-token limit. This is an admission/boundedness failure, not a claim that an outside-workspace write was demonstrated. A timer is not a substitute for rejecting the unauthorized form.

**Required before acceptance:** reject unsupported expansions at the raw-shell boundary or safely execute a parsed/serialized grammar; preserve any explicitly authorized quoted AWK/printf forms; add direct/reference/policy rejection tests for brace ranges, tilde and globs, and requalify the regenerated sealed validator. The same faulty code is embedded at `S/scripts/e97_first_party_validator_diversity.py:854-857,888-892,911-917`.

### B2 — Full-file deployment would regress current F8 and supply settings

The procedure directs materializing entire staged loop sources into repo and deployed loop copies (`S/F9-deployment.md:41-47`). Those snapshots predate the current collector:

- Staged bank imports `CorrectionPool` (`S/loop/rl_bank_lane.py:45`) and waits for pending corrections at the cycle-end barrier (`:540-554`). Current bank imports `CorrectionService` (`scripts/rl-loop-v1/rl_bank_lane.py:45`), durably submits jobs (`:651-664`), and explicitly has **no correction fence** (`:666-667`). Overwriting it would remove the accepted cross-cycle correction path and its workspace-integrity checks (`scripts/rl-loop-v1/rl_bank_lane.py:217-224` versus `S/loop/rl_bank_lane.py:209-238`). This is not authorized by task diversity.
- Staged supply is `TRANCHE_SIZE=48`, lanes `16/16/24/24` (`S/loop/standing_supply.py:126-129`); current repo supply is `96`, `24/24/48/48` (`scripts/rl-loop-v1/standing_supply.py:126-129`). Blind deployment violates the document's instruction not to change standing lane/quota knobs (`S/F9-deployment.md:62`).
- The historical retained supply hash (`S/F9-evidence.json:10`) no longer matches: current SHA256 is `5b6203639e3d7892a2adf08475560444b5aeaac2ed20aa99067446ec1c91502c`. `git diff ed3ed745..HEAD -- scripts/rl-loop-v1/standing_supply.py` attributes the knob drift to later changes, not silent F9 staging mutation.

**Required before acceptance:** rebase the narrowly scoped F9 changes onto current source, including the durable correction service's fresh-conversation path; preserve current knobs; qualify that integrated closure and update exact-byte evidence. This review does not authorize reverting F8.

### B3 — Deployment document lacks an actionable rollback boundary

Stopping supply/watchdog/collectors, preserving old pins/archives, persistent environment, collector-before-new-task compatibility, and committed-byte rebuild are good requirements (`S/F9-deployment.md:37-66`). But the entire operational procedure has no rollback sequence or distinction between rollback **before** and **after** admitting/injecting new-family work (`:35-75`), despite the report promising detailed rollback boundaries (`S/F9-report.md:179-183`).

**Required before deployment:** identify backup/release manifests and restore commands; explicit stop/drain/ownership handling for durable correction jobs and bank claims; restart ordering and import/hash verification; go/no-go and abort criteria. Before first new-family injection, restoration can return to old code/authority. After injection, restoring old validators/graders alone is unsafe: pending/claimed new-family tasks and durable correction jobs must be safely isolated or completed under retained compatible code, without deleting immutable receipts/pins. Specify and test that boundary; watchdog must not race restoration. These are missing instructions, not operations performed by this review.

## Scope-by-scope findings

### 1. Reference and policy/teacher safety — FAIL (B1)

The gate is an actual finite command-name/form allowlist, not a command blocklist (`S/scripts/e97_offline_shell.py:27-75`). Reference actions and assertion commands pass it in schema guards (`S/scripts/e97_diversity.py:200-211`). Real reference proof also selects the safe bridge (`S/scripts/e97_diversity_author.py:188-192`). Policy and teacher run_episode select `SafeWorkspaceBridgeV8` for end-state tasks and its scripted subclass for conversations (`S/loop/rl_loop_driver.py:350-359`); policy and correction callers supply these flags (`:749`, `:821-827`; `S/loop/rl_bank_lane.py:187-194,441-446`). The safety check runs before `next` returns its executable call (`S/scripts/e97_pi_scripted_v8.py:39-53`). Thus wiring is shared, but the grammar admits B1.

For forms it actually rejects, the bridge clears pending, removes the executable public-history frame and stops with `invalid_native_turn`; the native generation is retained (`S/scripts/e97_pi_scripted_v8.py:46-53`). The CPU test verifies no dispatch, retained frame and positive PG mask (`S/unit_tests.py:246-264`); failure records retain source_messages/generations (`S/loop/rl_loop_driver.py:375-379,399-410`). This test passed independently. The sealed validator rejects recorded invalid actions too (`S/scripts/e97_first_party_validator_diversity.py:1009-1014`). These are failing native attempts, not success/close receipts.

Existing dangerous-form tests cover traversal, absolute paths, `$HOME`, substitutions, backticks, backgrounding, network, newline interpreter injection, arbitrary AWK execution, git alias configuration and symlink creation (`S/unit_tests.py:238-243`). They do **not** explicitly cover standalone package installers/loaders, brace/tilde/glob expansion. Validation-only probes additionally rejected `pip install x`, `python evil.py`, `sh evil.sh`, `env cat file`, `bash -c cat`; the missing expansion cases were accepted. Assertion commands are restricted to read-only forms (`S/scripts/e97_offline_shell.py:118-121`; mutation test `S/unit_tests.py:267-271`).

Actual file capture is owner-side, not a model echo, and uses descriptor-relative `O_NOFOLLOW`, regular-file and size checks (`S/scripts/e97_diversity.py:230-251,254-276`); receipt coverage/identity is exact (`:279-303`). Symlink leaf/parent tests passed (`S/unit_tests.py:104-119`). Git system/global configuration is disabled for execution and command assertions (`S/loop/rl_loop_driver.py:374`; `S/scripts/e97_diversity.py:268-271`). This does not establish an OS sandbox or broader environment sanitization.

### 2. Validator versioning/source closure — PASS for the reviewed snapshot

The additive spec delegates old kinds to unchanged `_spec`; rejects mixing end-state assertions with era-4 outcome fields (`S/scripts/e97_first_party_validator_diversity.py:969-987`); old focused/regression branches remain distinct (`:1015-1047`). Compatibility tests compare the same constructed era-4 receipt-shaped terminal and negative final under both versions/modes (`S/unit_tests.py:178-200`), and passed. This test is a constructed authentic-shape receipt, not an independently retained historical Pi episode.

All three era-2/3/4 validator archive members were independently compared byte-for-byte with retained repo validators and matched; F9's commit changes none of them. The v8 self-contained file reproduced **exactly in memory** from old source + mechanics + shell + additive tail, using `S/sync_validator.py:5-13` without running its write. Old program selection remains SHA-based per task pin (`S/loop/inject_pool.py:134-157`).

Closure accepts exactly the old 17 paths or those 17 plus the new validator (`S/ndm/e97_first_party_source_archive.py:30-51,84-112`); extra component was independently rejected. Archive names/digests/canonical encoding remain closed (`:181-212`). Quarantine accepts only archived validator digests (`S/ndm/e97_first_party_read_observe.py:552-560`). Both first-party registry archive pins match the retained `9c333d4c...` archive; preparation updates those pins (`S/prepare.py:65-71`). No admission schema relaxation was found in this seam. B1 still requires regenerating/rebinding the new validator after repair.

### 3. Conversation mechanics — PASS in staged synchronous/pool paths; current-service integration unqualified

User branches are sealed mappings with bounded scripts, first-turn question/no-tools, fullmatch constraints and missing-branch rejection (`S/scripts/e97_diversity.py:139-176`). Same accepted behavior class at a given index selects the same text, independent of incidental wording (`:167-176`); deterministic/acting-early tests passed (`S/unit_tests.py:121-125`). The full exchange verifies authorized user text, no skipped turns, intermediate constraints and final finish (`S/scripts/e97_diversity.py:306-336`); skip/altered-user/final tests passed (`S/unit_tests.py:293-307`).

Only `ScriptedEpisodeV8` changes the protocol profile and reopens a turn-ending finish after sealed authorization (`S/scripts/e97_pi_scripted_v8.py:11-35`). Ordinary codec/provider bytes match retained hashes (`S/F9-evidence.json:6,8-9`), and ordinary irrevocable finish is tested (`S/unit_tests.py:132-148`). The v8 provider queues follow-up before the streamed done event (`S/configs/pi/e97-pi-native-scripted-v8.ts:28`); legacy families keep frozen-v2 (`S/loop/rl_loop_driver.py:70,372`).

Fresh corrections reset workspace/prefix in direct synchronous driver (`S/loop/rl_loop_driver.py:807-810,821-827`). Bank sync and old pooled paths share `_correction_requires_fresh_solve`, then choose clean workspace/no prefix (`S/loop/rl_bank_lane.py:109-113,173-194,522-533`); child rechecks routing (`:237-238`). The shared-rule test passed (`S/unit_tests.py:335-341`), but is a helper test, not an end-to-end correction execution. None of this qualifies F9 with today's durable F8 service (B2).

### 4. Seeds/morphs/revalidation/dedupe — Core chain PASS; supply intake gap named C1

Harvest requires admitted TRAIN bundles and matching passing local proofs, binds exact fixtures and author prompt, and records provenance/digests (`S/scripts/e97_diversity_author.py:59-93`). `verify_seed` rechecks spec/proof/receipt/bundle/prompt/fixture authority (`:96-116`). Corpus and manifest hashes independently match; 12-seed manifest contains 8 solved seeds plus 4 invented entries (`S/F9-seed-manifest.json:3-4,5-147`; creation `S/harvest_seeds.py:12-23`). No external download path is added here.

Morph CLI verifies admitted TRAIN base, keeps family/split, rejects base token/tree/prompt collisions and builds a fresh candidate (`S/teacher_author.patch:99-119`; `S/scripts/e97_diversity_author.py:130-142`). Build then redoes guards, vocabulary checks, sealed/public protected pre-screen, lake/tranche prompt/token/tree/identity dedupe (`S/teacher_author.patch:35-43,59-72`). Every new-family variant receives a new real-Pi proof, sealed grades, degeneracy/canonical/negative-control checks before quarantine publication (`L/scripts/teacher_author.py:1562-1573,1602-1607`; `S/scripts/e97_diversity_author.py:183-228`). Operational supply still runs protected overlap, operator allowlist commit, admission and injection (`S/loop/standing_supply.py:932-947`). Proof/admission are not inherited from seed metadata. Dedupe includes locked intra-tranche acceptance (`:603-626`); morph collision/split tests passed (`S/unit_tests.py:151-167`).

**C1:** standing supply reads `seed-corpus.json` without verifying its manifest hash or invoking `verify_seed`, and calls `author_instruction` directly (`S/loop/standing_supply.py:547-548,568`); that prompt path only calls `seed_instruction` (`S/teacher_author.patch:15-17`), whose solved-seed provenance check is just claimed TRAIN/internal-base metadata (`S/scripts/e97_diversity_author.py:24-37`). The stronger verification occurs in the separate author CLI (`S/teacher_author.patch:25-28`) and morph CLI, not this production call. Supply's absent-file fallback silently disables seeding. Require a verified corpus intake/pin or reverified seeds before enabling seeded supply, plus explicit corpus deployment ownership/location. Candidate full gates still stand; C1 is not cached admission reuse.

### 5. Rotation — PASS

Exact shares are first_action=16, edit/lookup/recovery/sum=6 each, chat=6, writing=6, core=1, terminal=4, conversation=3; cycle=60 and generated Counter equals shares (`S/loop/standing_supply.py:151-175`). The test checks every share and cycle length, not just sum (`S/unit_tests.py:170-175`), and passed independently. The current-to-staged operational knob drift is separate B2, not a rotation arithmetic failure.

### 6. Claimed 45/45 checks and three Pi proofs — Historical evidence confirmed; 42/42 independently rerun

Retained log records **45 passed** (`E/F9-staged-tests.log:6-11`); postcommit wrapper records its outer test passing (`E/F9-postcommit-tests.log:6-10`). Independently collected all 45, then ran **42 passed / 3 deselected**. The three exclusions are the Pi proof test (`S/unit_tests.py:210-235`), unsafe-policy Pi test (`:274-290`), and wrong-reference build test (`:320-332`, which also invokes real Pi indirectly). Running all 45 or the wrapper would violate this review's no-Pi constraint (`tests/test_f9_task_diversity.py:12-16`). **This review does not claim an independent 45/45 execution.**

All 17 staged source hashes and all 17 retained private-artifact hashes match `S/F9-evidence.json:14-53`; staged scripts/ndm/loop/configs match Q, and L's teacher copy matches Q's teacher. Five of six historical live-source hashes match; supply drift is B2 (`S/F9-evidence.json:5-11`).

Three proof summaries have distinct identities, native-close verification, canonical targets 180/267/180, both passing validators and degeneracy (`P/proofs.json:2-67`). Retained transports show `pi_exit=0`, distinct owned peer PIDs, next requests and verified close (`P/quarantine/era8-proofs/18c99597ae7d714c927c6362f59339e241abe6310299e971b00a22e6d9fc0cb9/transport-terminal.json:1`, corresponding `b839a3659911beee5feeadf86726fbc63d6a044ae9aef098cb68dff92745e805/transport-terminal.json:1`, `aae2fc331246ad5b7f332493206bc0ee8e9ceb988a1eea90c9c900819d486aa2/transport-terminal.json:1`). All three episode records have finished/close_verified and grades passed (each corresponding `episode-private.json:1`, `grade.json:1`), matching manifest hashes (`S/F9-evidence.json:18-29`).

CPU-only regrading of each retained terminal passed focused and regression; missing-output mutants passed through anonymous memfds were all rejected with `sealed end state failed`. No reference action was replayed. Protected-overlap/quarantine assertions exist in the retained successful test (`S/unit_tests.py:228-235`); no separate full protected-overlap receipt is pinned in `S/F9-evidence.json:14-32`. Reference success is not measured policy success (`S/F9-report.md:192-204`).

### 7. Five legacy v1 Pi failures — Not introduced/hidden by F9; precise runtime cause remains historical

All five real-Pi tests remain unmodified and explicitly use original v1 `configs/pi/e97-pi-native.ts` (`tests/test_e97_pi_native_tool_bridge.py:39-95`). F9 commit diff contains no modification to those tests, that provider, the ordinary codec/bridge, or old validators. The retained full invocation reports 44 passes and those five failures (`E/F9-regression-tests.log:14-60`); filtered run reports 44 passes / 5 deselected (`E/F9-cpu-regression-tests.log:6-13`). This review independently reproduced **44 passed / 5 deselected** from repo cwd.

The claimed schema-contract explanation is documented (`S/F9-report.md:155-159`) and consistent with v1 forwarding `context.tools` (`configs/pi/e97-pi-native.ts:22`) into exact owner contract equality (`scripts/e97_pi_native_tool_bridge.py:52-57`), versus v8 forwarding frozen `config.tools` (`S/configs/pi/e97-pi-native-scripted-v8.ts:26`). However the retained regression log exposes only `pi_native_tool_transport_not_qualified`, not the inner `pi_contract_mismatch`; no failure transport is pinned by the manifest. Therefore: **F9 did not cause or conceal these v1 failures; their exact runtime inner reason is not independently re-certified here.** Preserve this limitation, rather than claiming live v1 qualification or rerunning Pi.

### 8. Deployment readiness — FAIL (B2, B3; C1 also unresolved)

The coordinated stop/rebuild/persistent-env plan is necessary but not complete. In addition to B2/B3/C1, require exact repo-to-deployed path/import roster and runtime-source hashes, collector/teacher/correction-worker version probes, old/new task probes, watchdog relaunch verification, and named restart/abort ordering. Stop and account for the **current durable correction service**, not just collectors, before changing its imported source. Retain old authority; do not restore old-only grading while new-family work remains runnable. General preservation and monitoring language (`S/F9-deployment.md:37-66`) does not supply these executable gates.

## Independent CPU commands/results

On lambda01, project Python 3.12.3; Frontier activation/Slurm is not applicable. Common staged prefix:

```bash
source /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final/activate.sh
cd "$F9_STAGE_ROOT"
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m pytest -q -p no:cacheprovider /home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py -k 'not real_pi and not failed_reference_solution'
```

Result: **42 passed, 3 deselected in 2.14s**. Inline Python validation used `shell_allowed`, `author.guards` on `terminal_fixture()`, then a synthetic native-turn `SafeWorkspaceBridgeV8.next` with frozen request tools; no transport invocation. B1 forms were accepted by BOTH gates. Additional dangerous commands named above were rejected.

From repo cwd:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
```

Result: **44 passed, 5 deselected in 10.66s**. An initial legacy invocation from Q failed 4 tests solely because they open repo-relative `configs/pi/e97-active-tool-surface-v1.json` (`tests/test_e97_pi_native_codec.py:41`; `tests/test_e97_pi_native_tool_bridge.py:11`); rerun from required repo cwd resolved it. This was a review harness cwd error, not an F9 regression.

Other read-only checks: evidence SHA256 census; L/Q/S byte comparisons; archive member equality and old/additive closure validation; in-memory self-contained regeneration equality; in-memory compilation of 15 staged Python files; 6 retained-terminal sealed grades passed and 3 missing-output controls rejected; both registry archive pins matched. One combined inspection script then failed reporting a nonexistent registry `source_id` key; corrected read-only inspection used the actual `id` field and verified both pins. `git diff --check` passed. None of these checks ran the real-Pi proofs.

Applicable evidence/safety intent is **R14/NDP13 bounded local termination and R16 exact-source evidence discipline**, not resilient-training conformance (`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:113-122`; `docs/RESILIENT_DILOCO_GAP_MATRIX.md:33-34`). No checkpoint, distributed-training, scale, Slurm or convergence claim is made.

## Acceptance disposition / next step

**FAIL:** repair B1, integrate/requalify against current source (B2), and provide tested coordinated deployment/rollback boundaries (B3), with verified seed intake (C1) and honest v1 evidence limitations. Keep live supply and collection untouched. Seek a new independent review of the repaired exact-byte closure before any operator restart/admission. No implementation fix is included in this review.
