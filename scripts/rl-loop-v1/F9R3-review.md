# F9R3 independent round-3 re-review — FAIL

**Deployment verdict: FAIL / NO DEPLOY this closure.** Canonical-only dispatch is structurally present, and the F9R2 operator-run blocker is fixed. However, a new canonical-byte-bound closure defect admits a raw command and dispatches its canonical representation even when that representation violates the same sealed grammar's 4096-character limit. Honest execution receipts then fail both sealed grading modes. This is a bounded-grammar/receipt-consistency blocker, **not an interpreter-execution or shell-injection finding** (`S/scripts/e97_offline_shell.py:134-136,165-180`; `E/F9R3-boundary.log:2-3,5-7`). Passing the original battery and byte pins does not override it.

## Scope and citation roots

Reviewed HEAD `c8686c13426fb38f53e5d1dc17baaaddf069efef` (F9E grammar commit `825f90db40c8458e969c8e47a6f0041c522e56f7` plus separate runbook commit), on lambda01, project `.venv` Python 3.12.3. Review-only: implementation, authority archives/pins and historical evidence were not modified. New scripts/results/logs are scratch QA artifacts; only this report is copied into the repo. No live Pi, teacher API, GPU training, Slurm, daemon/watchdog, live bank admission/injection, source deployment, stop/restart or rollback occurred. All command-candidate negatives and the six byte-bound probes were admission-only; only the 13 already-authorized equivalence controls ran in isolated scratch CPU directories (`E/F9R3-grammar-qa.py:151-180`; `E/F9R3-boundary-qa.py:1,73-98`; `E/F9R3-boundary.log:7`). Bank-suite processes/signals are CPU fixtures, including real fixture manager/worker teardown, not live owners (`S/loop/rl_bank_unit_test.py:2109-2200,2208-2290`; `E/F9R3-bank.log:310-324`). Bytecode and pytest cache writes were disabled.

Citation prefixes are absolute roots:

- `S` = `/home/erikg/emender/scripts/rl-loop-v1/F9-staged`.
- `Q` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9E-final` (bound integrated source closure, `S/F9-evidence.json:166`).
- `H` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final` (historical evidence only, `S/F9-evidence.json:127-128`).
- `E` = `/home/erikg/emender-scratch/fix-fanout`.
- Unprefixed paths are relative to `/home/erikg/emender`.

The scratch/repo reports are identical. Source citations in S also apply to Q runtime bytes, independently compared (`E/F9R3-byte.log:7`). The QA asserts that the imported driver is Q's driver, not a live/deployed fallback (`E/F9R3-grammar-qa.py:23-33`).

## 1. Canonical-only execution — PASS structurally; canonical bound closure FAIL

### Complete execution-facing path

1. Reference guards call `action_allowed` before proof construction (`S/scripts/e97_diversity.py:200-211`). The guard does not mutate the raw reference plan, but new-family proof routing selects `diversity_author.prove_bundle`, not the old dense proof transport (`Q/loop/teacher_author.py:1563-1573`). The reference generator puts raw action arguments into **generated native evidence**, then calls the common driver with `end_state_task=True` (`S/scripts/e97_diversity_author.py:199-218,228-232`). This is not a raw-command dispatch bypass.
2. Hash-checked private specs select end-state/user-script behavior (`S/loop/rl_loop_driver.py:313-330`). Both normal policy and teacher/correction calls pass those flags, including bank and durable correction paths (`S/loop/rl_loop_driver.py:756-760,832-838`; `S/loop/rl_bank_lane.py:187-195,546-551`). Driver chooses `ScriptedPiBridgeV8` for conversations, `SafeWorkspaceBridgeV8` for other end-state tasks, and ordinary `NativePiToolBridge` only for old/ordinary tasks (`S/loop/rl_loop_driver.py:350-359`). Prefix replay appends native evidence only; it does not transport/re-execute prefix commands (`S/loop/rl_loop_driver.py:273-284`).
3. Base bridge initially constructs a raw pending/public tool call, but returns only to the overriding safe bridge, not directly to the socket (`scripts/e97_pi_native_tool_bridge.py:64-76`). Safe bridge calls `action_allowed`; on rejection it removes the public call, clears pending and raises before returning. On success it replaces **pending, returned toolCall arguments and public-history arguments** with canonical arguments before return (`S/scripts/e97_pi_scripted_v8.py:39-59`). Conversation subclass inherits this gate (`:62,72-75`).
4. `action_allowed` returns a new arguments dict containing `shell_allowed(raw)`; `shell_allowed` is exactly `serialize_shell(parse_shell(raw))` (`S/scripts/e97_offline_shell.py:179-180,189-202`). Serializer emits only parsed argv/target words using single-quote escaping and enumerated separators; it never splices the raw command (`:165-176`).
5. Driver supplies that bridge to the common owner transport (`S/loop/rl_loop_driver.py:372-376`). Socket egress invokes `bridge.next`, builds its result response and sends that response; exceptions send `ok=False`, not an unvalidated call (`scripts/e97_pi_native_tool_transport.py:48-59`). Both provider variants emit **only `r.message.content` / `block.arguments` from that response** into Pi tool-call events (`configs/pi/e97-pi-native-frozen-tools-v2.ts:36-38`; `S/configs/pi/e97-pi-native-scripted-v8.ts:26-28`). No separate raw native-frame dispatch is present there. Host `user_bash` is disabled in both (`configs/pi/e97-pi-native-frozen-tools-v2.ts:33`; `S/configs/pi/e97-pi-native-scripted-v8.ts:23`). This is static first-party wire/dispatch tracing plus synthetic bridge checks, not a fresh Pi runtime execution proof.
6. Other relevant subprocess paths do not execute native command strings: command assertions first validate read-only **argv**, then call `subprocess.run(argv)` without a shell (`S/scripts/e97_diversity.py:254-271`; sealed equivalent `S/scripts/e97_first_party_validator_diversity.py:752-765`). New-family injected bodies have `setup: []`, so driver's argv seed-setup loop does not provide a reference shell bypass (`Q/loop/inject_pool.py:179-185`; `S/loop/rl_loop_driver.py:303-308`). Sealed validator validates received terminal actions, rather than dispatching them (`S/scripts/e97_first_party_validator_diversity.py:1077-1082`). Its return value is intentionally unused at that validation-only callsite; this is not an execution egress.

Independently exercised pending/returned/public-history equality with canonical bytes, subsequent canonical-history ingest, raw native/source preservation and positive PG masks at all accepted controls (`E/F9R3-grammar-qa.py:61-92`; `E/F9R3-grammar.log:104-116,351-355`). Original bytes remain evidence, not execution-facing command bytes.

### NEW BLOCKER B1-R3 — canonical serialization escapes the grammar byte bound

`parse_shell` limits **input** to 4096 characters (`S/scripts/e97_offline_shell.py:134-136`), but `serialize_shell` can expand quoting and `shell_allowed` returns it without checking the canonical length (`:165-180`). `action_allowed` therefore returns out-of-bound canonical arguments, and safe bridge dispatches them (`:195-198`; `S/scripts/e97_pi_scripted_v8.py:54-59`). The sealed copy has the identical behavior (`S/scripts/e97_first_party_validator_diversity.py:967-969,998-1013,1028-1031`). Canonical-only dispatch prevents injection but does **not** close the allowed language under serialization.

Minimal, non-executed probes are defined exactly in `E/F9R3-boundary-qa.py:34-39`:

```python
'printf %s ' + 'x' * 4081           # raw 4091 -> canonical 4097
'printf %s ' + 'x' * 4086           # raw 4096 -> canonical 4102
'printf %s "' + "'" * 817 + '"'     # raw  829 -> canonical 4101
'printf %s "' + "'" * 1000 + '"'    # raw 1012 -> canonical 5016
```

All four raw forms **ACCEPT** at direct, regenerated sealed, reference and policy gates; bridge returns `toolUse` with the over-bound canonical bytes, and honest driver projection records those same bytes (`E/F9R3-boundary-qa.py:43-68`; `E/F9R3-boundary.log:2-3,5-6`). Direct and sealed gates then **REJECT the canonical command** with `bounded offline shell required`. Two neighboring controls serialize to exactly 4096 and accept (`E/F9R3-boundary.log:1,4`). Hence this is a demonstrated boundary discontinuity, not a malformed-input guess.

To test the actual sealed executable without candidate execution, the QA clones a successful historical terminal only as a **synthetic validator input**, replaces its bash argument with raw versus canonical probe bytes and retains its end-state fields. Both modes pass for the raw-form synthetic inputs; both modes fail for the four over-bound canonical counterparts with `sealed invalid-action outcome: bounded offline shell required` (`E/F9R3-boundary-qa.py:73-102`; `E/F9R3-boundary.log:2-3,5-6`). These borrowed end-state fields are **not proof that any probe ran or solved anything**. The experiment isolates the grammar/receipt mismatch; synthetic bridge ingest separately proves what the honest projection would record. No negative command reaches subprocess; only the sealed validator executable runs (`E/F9R3-boundary-qa.py:84-85`).

Impact: an admitted reference/policy action can dispatch an executed form that the sealed action gate rejects, even with correct output; canonical grammar roundtrip acceptance also ceases to hold. Quoted apostrophes demonstrate that the mismatch occurs well below the raw 4096 limit, not only at an extreme-length prompt. Repair must restore the invariant that every admitted canonical action satisfies the same sealed grammar **before dispatch**, add the six boundary controls as regression evidence, and regenerate/rebind/requalify. The implementation choice for the bound belongs to the repair authority; this review does not broaden it or apply a fix. Do not falsify records by reverting to the raw arguments.

## 2. Full consumption / independent representation checks

Unsupported operator runs are lexed as operators and rejected unless exactly `;`, `|` or `>`; they cannot become printf argv (`S/scripts/e97_offline_shell.py:87-101`). Quoting retains word identity, concatenates fragments, rejects unterminated quotes/expansion/unsupported whitespace, and consumes the entire input (`:103-125`). Parser consumes finite approved argv, one optional safe redirect, and only allowed separators; extra redirect words, empty/trailing commands and redirect-before-pipe reject (`:140-162`). Numeric AWK DSL and printf format/argc checks remain finite (`:43-59`). Apart from B1-R3's emitted-length bound, no unsupported-form admission was observed in this review.

The independent rerun reads **the original reviewer results JSON** as input, not the adopted implementation's matrix constructor (`E/F9R3-grammar-qa.py:99-105`). SHA256 `fd938f83067a37985e8a1f99ed989db87fab41aee0d60e00df7d63d1daa5e5d4` matches the retained baseline. All **117** original cases now agree across all four gates: **104 reject / 13 accept**, zero unexpected matrix decisions; all five former `||`/`;;` misses reject (`E/F9R3-grammar.log:96-99,104-117,355`). The three original expansion repros also reject (`:1-3`).

New distinct cases are deduplicated against the old matrix, never executed, and tested at all four gates (`E/F9R3-grammar-qa.py:107-145`). **233 new negatives all reject**, plus four authorized literal controls accept (`E/F9R3-grammar.log:118-355`):

| New category | Distinct count | Result / exact evidence |
|---|---:|---|
| `||`, `;;`, `&&`, `|&`, `;&`, other mixed/operator runs; naked/quoted/concatenated left word; adjacent, space, tab forms | 153 | All reject; `E/F9R3-grammar-qa.py:109-112`; `E/F9R3-grammar.log:118-270` |
| Mixed separated operators and unconsumed redirect suffixes | 8 | All reject; `E/F9R3-grammar-qa.py:113-114`; `E/F9R3-grammar.log:271-278` |
| Operator-adjacent NBSP, em-space, Unicode line/paragraph separators, NEL, VT/FF/CR/LF/DEL | 30 | All reject; `E/F9R3-grammar-qa.py:115-117`; `E/F9R3-grammar.log:279-308` |
| Partial quotes, ANSI-C quoting, empty redirect target, trailing/extra suffixes | 13 | All reject; `E/F9R3-grammar-qa.py:118-125`; `E/F9R3-grammar.log:309-321` |
| printf formats, escapes, positional directives, argc/options drift | 15 | All reject; `E/F9R3-grammar-qa.py:126-129`; `E/F9R3-grammar.log:322-336` |
| AWK width/operator/number/regex/program/field-delimiter/extra-path drift | 14 | All reject; `E/F9R3-grammar-qa.py:130-136`; `E/F9R3-grammar.log:337-350` |
| Quoted operators/Unicode, literal apostrophe composition, concatenated safe redirect path | 4 controls | Accept as literal grammar data, not un-rejected syntax; `E/F9R3-grammar-qa.py:147-149`; `E/F9R3-grammar.log:351-354` |

All **13 authorized controls** independently preserve argument-by-argument POSIX decoding (command name, format/program, every argument and redirect target). In separate scratch directories original/canonical forms have identical exit status, stdout, stderr and every file's bytes; all exit 0. Unlike prior data with multi-letter names that the AWK regex ignores, the fixture additionally contains `a,2`, exercising each actual arithmetic transformation (`E/F9R3-grammar-qa.py:151-180`). Exact argv and hex output/file receipts are retained in `E/F9R3-grammar-results.json` under `equivalences`; summary confirms 13 (`E/F9R3-grammar.log:355`). This is equivalence evidence for these controls, not a claim of arbitrary-shell equivalence.

## 3. End-state execution-record honesty — PASS; bound mismatch remains visible

Driver marks both safe/scripted records as end-state records (`S/loop/rl_loop_driver.py:369-370`). Owner publication retains `native_record`, generation token IDs, public history and raw `source_messages` separately (`:401-407`). Projection uses `action_allowed` to regenerate the executed canonical arguments for both dense action and message, not raw source arguments (`:504-533`). Docstring and label explicitly distinguish **executed canonical actions** from generated raw PG evidence; the ordinary format-only label is confined to ordinary tasks (`:471-475,567-572`). This is an explicit semantic label update, not a softened byte-faithful claim.

The independent bridge-to-projection QA asserts that source messages are unmodified, raw frames remain in native_record, projected actions/messages equal pending execution arguments, ordinary projection remains raw, and PG masks remain positive (`E/F9R3-grammar-qa.py:64-92`). For example `printf\t%s\tliteral` becomes `'printf' '%s' 'literal'`, so the record is **not equal to the generated command string** (`E/F9R3-grammar.log:112`). Permanent identical/differing-input projection controls also pass (`S/test_f9r2_grammar.py:192-213`; `E/F9R3-regression.log:13`). B1-R3 does not falsify projection: it exposes that the honest canonical projection is rejected by the bound gate (`E/F9R3-boundary.log:2-3,5-6`).

## 4. Validator rebinding / byte closure — PASS, not an override

Independent census re-verifies **all 111 pin occurrences**: 22 staged sources, 60 integrated runtime files, 17 historical artifacts, 12 historical qualification logs. Counts are occurrences, not distinct files (`E/F9R2-byte-qa.py:20-27`; `E/F9R3-byte.log:1-5`). Current correction service/pool/config/bank and all staged runtime/provider bytes match Q (`E/F9R3-byte.log:6-7`). Supplementary checks verify **two F9E permanent regression source pins and seven new F9E log pins**, making nine additional occurrences (`E/F9R3-extra-byte-qa.py:18-21`; `E/F9R3-extra-byte.log:1-2`). No historical live source hashes are mislabeled as current (`E/F9R3-byte.log:5`; `S/F9-evidence.json:33-40,106,127-128`).

In-memory `validator_source()` equality matches S and Q; generation's write entrypoint is not called (`S/sync_validator.py:7-21`; `E/F9R3-grammar-qa.py:25-29`). Nine shell function ASTs, including lexer/parser/serializer/action gate, are identical between direct and embedded sealed source: not a stale/raw-only implementation (`E/F9R3-extra-byte-qa.py:23-32`; `E/F9R3-extra-byte.log:3`). This equality also reproduces B1-R3, not just its fixes.

Old 17-component and additive 18-component closures parse, extra component rejects, canonical archive verification and both admitted first-party registry pins match, era-2/3/4 archive members remain byte-exact, and all 18 archive components equal Q source bytes (`E/F9R2-byte-qa.py:37-60`; `E/F9R3-byte.log:8`; `E/F9R3-extra-byte-qa.py:33-40`; `E/F9R3-extra-byte.log:4`). Prepare enforces the new validator hash before overlay and rebinds both admitted first-party archive pins (`S/prepare.py:20-25,73-79`).

- Sealed validator: `7abebc445b1cf48feba7a24bb585bf8644c9c144d972b651fa4f9d599d522cd0`.
- Source archive: `061133d6f564f19e580bc864876e65dd032c5832f9c7ce2aaee50f1470270f69`.
- Independent receipts: `E/F9R3-extra-byte.log:6-7`.

## 5. Runbook / earlier integration findings — PASS within CPU scope

C2 is fixed: `F9_TOOLS="$REPO/scripts/rl-loop-v1/F9-staged"` is explicitly defined before either boundary sequence, retained across rollback and recorded in the release manifest (`S/F9-deployment.md:91-96,212-214`). Independently extracted that **exact documented assignment** and exercised the resulting `rollback_gate.py --help` with the project interpreter: exit 0, correct `--bank` option, no bank touched (`E/F9R3-extra-byte-qa.py:42-50`; `E/F9R3-extra-byte.log:5`). Exact documented restore code is separately re-rehearsed only against temporary fixtures: corrupt backup aborts before either destination changes; valid backup restores old bytes and removes only the added fixture (`E/F9R2-byte-qa.py:61-83`; `E/F9R3-byte.log:9`). Live owner/import/environment/STOP/watchdog/restart checks remain attended deployment-time gates, not CPU-certified runtime state (`S/F9-deployment.md:87-110,285-295`).

Earlier integration PASS findings remain exercised:

- Durable fresh-conversation manager/worker route resets workspace/prefix, publishes fresh zero-prefix supervision, retires claim exactly once (`S/loop/rl_bank_unit_test.py:2109-2200`; `E/F9R3-bank.log:310,312-313`). Current helper identity preserved (`E/F9R3-byte.log:6`).
- Tranche **96**, lane knobs **24/24/48/48**, exact 60-call rotation shares preserved (`S/loop/standing_supply.py:126-129,151-175`; `S/unit_tests.py:170-175,434-441`; `E/F9R3-cpu.log:12`).
- Versioned conversation clarification/continuation/full-exchange and irrevocable ordinary finish, morph split isolation/revalidation/dedupe/private prescreen/harvest, and old validator compatibility remain passing (`S/unit_tests.py:119-207,293-317,353-362`; `E/F9R3-cpu.log:12`).
- Verified seed intake still precedes seeded instruction construction, logs explicit disable/verified intake, and retains authority/pin checks (`S/loop/standing_supply.py:547-553,573`; `S/unit_tests.py:487-550`; `E/F9R3-cpu.log:12`). No teacher call is needed for this CPU validation.
- Historical private artifacts retain their pins; retained terminals regrade in six modes and missing-output mutants reject (`E/F9R3-byte.log:3`; `S/unit_tests.py:571-586`; `E/F9R3-cpu.log:12`). These remain historical Pi evidence, **not fresh corrected-source transport qualification**; five legacy real-Pi tests and three F9 Pi-dependent tests are excluded, not relabeled passing (`E/F9R3-legacy44.log:13`; `E/F9R3-cpu.log:12`). No new protected-overlap qualification is claimed.

## 6. Independent commands and results

Common setup on lambda01 (not Frontier):

```bash
source /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9E-final/activate.sh
cd "$F9_STAGE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/erikg/emender/.venv/bin/python
S=/home/erikg/emender/scripts/rl-loop-v1/F9-staged
E=/home/erikg/emender-scratch/fix-fanout
"$PY" -m pytest -q -p no:cacheprovider "$S/unit_tests.py" -k 'not real_pi and not failed_reference_solution'
"$PY" -m pytest -q -p no:cacheprovider "$S/test_f9r2_grammar.py"
"$PY" loop/rl_bank_unit_test.py --scratch "$E/F9R3-bank-fixtures"
"$PY" "$E/F9R2-byte-qa.py"
"$PY" "$E/F9R3-grammar-qa.py"
"$PY" "$E/F9R3-extra-byte-qa.py"
"$PY" "$E/F9R3-boundary-qa.py"
cd /home/erikg/emender
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
git diff --check
git diff --cached --name-only
```

Independent receipts:

| Check | Result | Evidence |
|---|---|---|
| F9 CPU | **74 passed / 3 deselected**, 3.64s | `E/F9R3-cpu.log:12` |
| Permanent regression battery | **148 passed**, 4.89s | `E/F9R3-regression.log:13` |
| SAME-tree bank CPU fixtures | **215 passed** | `E/F9R3-bank.log:324` |
| Original legacy CPU selection | **44 passed / 5 deselected**, 10.55s | `E/F9R3-legacy44.log:13` |
| Independent original matrix + new negatives + authorized equivalence | **117 matrix + 233 new rejects + 4 literal controls; 13 equivalences; zero misses**, exit 0 | `E/F9R3-grammar.log:355` |
| Byte census/closures/restore | **111 pins match**, exit 0 | `E/F9R3-byte.log:1-10` |
| F9E additional pins/AST/archive/runbook | **9 extra pins; AST/18 members/assignment pass**, exit 0 | `E/F9R3-extra-byte.log:1-7` |
| NEW canonical bound QA | **4 closure violations / 6 cases**, exit **1** (intentional failing deployment QA) | `E/F9R3-boundary.log:1-7` |

These are independently run suites, not copied F9E logs. The 148 already include the adopted 117 matrix, so counts are not summed into a misleading unique-case total. Whitespace/index/report-copy checks are retained in `E/F9R3-worktree.log`; existing untracked `scripts/rl-loop-v1/F9R2-review.md` is left untouched. No files were staged.

Applicable requirements are **R14/NDP13 bounded local safety intent and R16 exact-source/execution evidence discipline**, not resilient-training conformance (`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:113-122`; `docs/RESILIENT_DILOCO_GAP_MATRIX.md:33-39`). No elastic/native/v2.1/ISP, distributed membership, overlap, convergence, scale or scheduler pass is claimed.

## Disposition

**FAIL / NO DEPLOY.** The raw-operator injection blocker is repaired and canonical-only execution/record honesty are real. Nevertheless B1-R3 permits dispatch outside the sealed grammar's bound and causes honest canonical receipts to be rejected. Fix canonical-bound closure without returning to raw execution/receipts, add the boundary probes to permanent direct/reference/policy/sealed and grading regressions, regenerate/rebind/requalify, and obtain independent review again. Preserve all earlier integration fixes and immutable historical artifacts. Separate operator approval and the runbook's live source/ownership/import/rollback/seed GO checks remain required afterward; this CPU-only review performs or authorizes none of them.
