# F9R2 independent re-review — FAIL

**Deployment verdict: FAIL. Do not deploy the reviewed closure.** The original three expansion repros are repaired, but a second command-grammar admission defect remains: `||` and `;;` are swallowed as printf operands. Unsupported conditional branches, including script loaders and a forbidden printf format, reach policy `toolUse` unchanged. B2 and C1 repairs pass; B3 is substantially repaired with one named setup omission below. Passing existing tests and matching hashes do not override the grammar blocker.

## Scope / citation roots

Reviewed repository HEAD `3353739817da0e0c2711c808878f0f10c51c5c2c` on lambda01, project `.venv` Python 3.12.3. This is first-party internal QA, not an implementation repair. Only review artifacts are written; candidate shell strings are **never executed**. No live Pi episode, teacher API, GPU operation, daemon operation, Slurm, live admission/injection, runtime source replacement, watchdog operation or live bank mutation was performed. Bank-suite OS processes and signals are exclusively CPU fixtures (`S/loop/rl_bank_unit_test.py:2109-2200,2208-2290`; `E/F9R2-bank.log:300-314`). Pytest/bytecode cache writes are disabled.

Citation prefixes denote exact absolute roots:

- `S` = `/home/erikg/emender/scripts/rl-loop-v1/F9-staged`.
- `Q` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9C-reviewed` (source closure: `S/F9-evidence.json:135-136`).
- `H` = `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final` (historical evidence only: `S/F9-evidence.json:97-98`).
- `E` = `/home/erikg/emender-scratch/fix-fanout`.
- Unprefixed source paths are relative to `/home/erikg/emender`.

The scratch and repository copies of this report are identical. Added standalone QA scripts/results remain in E, not in the staged implementation or its authority archive.

## B1 — original repair verified, but grammar gate still FAILS

### Original F9R snippet rerun independently

The original `terminal_fixture` mutation, `author.guards`, synthetic `native_turn`, frozen tools/panel and `SafeWorkspaceBridgeV8.next` reproduction was rerun, adding only rejection-catching so all three cases complete (`E/F9R2-grammar-qa.py:16-60`). Direct and regenerated sealed grammar gates were also checked. Results (`E/F9R2-grammar.log:1-3`):

| Original bytes | Reference guards | Policy bridge | Direct / sealed |
|---|---|---|---|
| `printf %s {1..999999999}` | REJECT | REJECT | REJECT / REJECT |
| `printf %s ~root` | REJECT | REJECT | REJECT / REJECT |
| `printf %s *` | REJECT | REJECT | REJECT / REJECT |

For every rejected case, the bridge raises `BridgeStopped`, has `failed=True`, `pending=None`, keeps only opening public history, retains the generated native frame and retains a positive encoded PG target mask (`E/F9R2-grammar-qa.py:49-55`). This matches the before-return gate and sealed-failure retention code (`S/scripts/e97_pi_scripted_v8.py:39-52`). Reference guards use the same action gate (`S/scripts/e97_diversity.py:200-211`). The repaired raw boundary precedes shlex and refuses unquoted `{}`, `~`, `*`, `?`, `[]`, groups/comments and non-single-quoted substitution/escapes (`S/scripts/e97_offline_shell.py:78-108`). No candidate reached an actual transport in this review.

### NEW BLOCKER B1-R2 — operator runs normalized into printf arguments

POSIX shlex groups adjacent punctuation into tokens such as `||` and `;;`. The parser handles only exact `;`, `|` and `>`, rejects only tokens containing `&`, `<` or `>`, then appends everything else to argv (`S/scripts/e97_offline_shell.py:108-138`). `_literal` does not reject pipe/semicolon characters (`:21-24`), and printf validates these apparent operands under the first `%s` format (`:55-59`). Consequently it never validates a second conditional command as a command. The original shell bytes, not a safe serialized argv, remain pending and are returned for tool execution (`S/scripts/e97_pi_scripted_v8.py:39-52`; `scripts/e97_pi_native_tool_bridge.py:75-76`). This contradicts the published **semicolon/pipeline-only** grammar and no-script/no-interpreter constraint (`S/scripts/e97_diversity.py:338-345`).

All four gates independently **ACCEPT** each of the following negative controls (`E/F9R2-grammar.log:96-99,117`):

```text
printf %s x || printf %b literal
printf %s x || python evil.py
printf %s 'x' || 'sh' 'evil.sh'
printf %s x ;; printf %s y
printf %s x || printf %s y
```

For these cases policy returns `stop_reason=toolUse`, `failed=False`, with the exact original command unchanged in pending (`E/F9R2-grammar-qa.py:56-59`). The regenerated sealed validator repeats the same defect (`S/scripts/e97_first_party_validator_diversity.py:937-971`). Therefore exact byte rebinding is correct but binds unsafe admission behavior.

This is a grammar/admission finding, **not a claim that Python/sh was executed or that an outside-workspace write occurred**. `||` evaluates its right side only if the left command fails; printf normally succeeds but may fail on output errors. The validator must reject the unauthorized branch regardless. The `;;` control is invalid top-level shell syntax, not a demonstrated executable bypass; its acceptance additionally proves unsupported operator runs are not fail-closed. Quoting the branch's command name does not cure that admission defect.

Required next repair: reject every unsupported operator token/run before treating tokens as argv, preserving intentionally authorized literal quoting; add direct/reference/policy/sealed negatives for these exact cases and representation variants. Regenerate the sealed source and rebind/requalify the archive/evidence again. No fix is applied in this read-only review.

### Additional independent representation matrix

The standalone negative tests and authorized controls are source-named in `E/F9R2-grammar-qa.py:62-114`; complete exact strings and outcomes are retained in `E/F9R2-grammar-results.json` and line-oriented `E/F9R2-grammar.log:1-118`. All four gates agree across **117 cases**: 104 expected-reject cases (3 original + 101 additional), of which **99 reject and 5 unexpectedly accept**, plus 13 correctly accepted authorized-literal controls (`E/F9R2-summary.log:1-3`; `E/F9R2-grammar.log:118`). The QA command intentionally exits 1 on the five misses.

| Category | Tested behavior / observed result | Evidence |
|---|---|---|
| Raw expansion, 30 | Brace range/list, tilde, glob, `?`, `[ab]` under unquoted, quoted-format, fragmented-command-name and tab-separated forms: all REJECT | `E/F9R2-grammar-qa.py:64-66`; `E/F9R2-grammar.log:4-58` (interleaved categories) |
| Fragmented/unicode expansion, 30 | Empty quote concatenation, literal quoted prefix/suffix and nested-quote prefix around an exposed ASCII expansion; NBSP/em-space neighbors: all REJECT | `E/F9R2-grammar-qa.py:67-68`; `E/F9R2-grammar.log:9-63` (interleaved categories) |
| Quoting/format/whitespace, 20 | Double-quoted `$()`/backticks/variables, escapes, ANSI-C quoting, nested quote fragments exposing substitution, unterminated quotes, newline/CR/NUL, `%b`, `%n`, `%s%s`, unsupported escape format, leading `--`, unicode separators: all REJECT | `E/F9R2-grammar-qa.py:69-77`; `E/F9R2-grammar.log:64-83` |
| AWK boundary, 12 | Double-quoted program, seven-digit arithmetic, division, wrong field, arbitrary system form, extra statement, wrong delimiter, traversal/tilde/glob input, non-ASCII digit and changed internal whitespace: all REJECT | `E/F9R2-grammar-qa.py:78-84`; `E/F9R2-grammar.log:84-95`; finite program regex `S/scripts/e97_offline_shell.py:43-50` |
| Operator boundary, 9 | `&&`, `&`, separated empty pipeline and traversal redirect REJECT; four `||` variants and one `;;` variant ACCEPT — B1-R2 above | `E/F9R2-grammar.log:96-103,117` |
| Authorized controls, 13 | Approved single-quoted AWK +/*/- forms; safely quoted expansion-looking literals, nested quotes and quote concatenation; tab-delimited ordinary literal; fullwidth Unicode lookalikes; fragmented command name: all ACCEPT as intended | `E/F9R2-grammar-qa.py:90-95`; `E/F9R2-grammar.log:104-116` |

Quoted brace/tilde/glob text and fullwidth Unicode lookalikes are **literal controls, not un-rejected expansion variants**: the approved repair explicitly preserves such quoted literals (`S/scripts/e97_offline_shell.py:81-84`; `S/unit_tests.py:400-408`). No Unicode normalization is performed. This distinction prevents falsely labeling authorized quoted data as a bypass; every actual unsupported form not rejected is listed above.

## Exact-byte validator/archive/evidence qualification — PASS, not a safety override

Independently checked all **111 claimed pins**: 22 staged source files, 60 integrated runtime files, 17 historical private artifacts and 12 retained qualification logs (`E/F9R2-byte-qa.py:20-27`; `E/F9R2-byte.log:1-5`). These are pin occurrences, not 111 distinct files. Historical live-source hashes are separately labeled and not counted; the old supply hash is not misrepresented as current (`S/F9-evidence.json:3-10,76,97`). The other five shared/teacher historical hashes still match (`E/F9R2-summary.log:4`). Every staged scripts/ndm/loop/provider byte also matches Q (`E/F9R2-byte-qa.py:32-36`; `E/F9R2-byte.log:7`).

In-memory regeneration using `validator_source()` equals both staged and Q validator bytes; it does not invoke the generator's write entry point (`S/sync_validator.py:7-21`; `E/F9R2-byte-qa.py:37-40`). Independently verified old 17-path and additive 18-path closures, rejection of an extra component, canonical archive verification, both admitted first-party registry pins and byte-exact era-2/3/4 archive members (`E/F9R2-byte-qa.py:42-60`; `E/F9R2-byte.log:8`; preserved CPU checks `S/unit_tests.py:411-431,553-568`).

- Reviewed sealed validator SHA256: `a06740f36101bf571791510fde93f4c06739e27f94bec91245cf0b160c5eb634` (`S/F9-evidence.json:127`).
- Reviewed additive archive SHA256: `ba0a76b26fb8ae0d5df1883d34601413f4fdb974380306838509ed94b3c489a8` (`S/F9-evidence.json:135`).
- Prepare enforces that sealed pin before overlay and updates both first-party registry archive pins (`S/prepare.py:20-25,73-79`).

## B2 — durable-service/current-source integration PASS

The staged bank retains the current `CorrectionService` import (`S/loop/rl_bank_lane.py:45`), durable `submit` (`:629-642`) and explicit **no correction fence** (`:644-645`). Owned workspace, immutable attempt/grade and fresh-routing identity checks remain (`:210-242`). Its diff against current repo is confined to F9 sealed-script/end-state plumbing and removing F8's temporary conversation rejection, not reverting service ownership. Service/pool/config and bank helper bytes match current repo byte-for-byte in S/Q (`E/F9R2-byte-qa.py:28-31`; `E/F9R2-byte.log:6`; `S/unit_tests.py:434-441`).

Fresh conversation corrections are explicitly routed to fresh workspace with no retained prefix, carry the sealed user script/end-state setting, and are rechecked inside `execute_correction_spec` (`S/loop/rl_bank_lane.py:109-113,175-195,241-250`). Independently compared the two bank suites: only the one F8 conversation block changes; its three obsolete rejection checks become three positive durable-path checks. The new harness launches a real OS manager and worker through the actual service/pool and correction executor; only model/Pi/grade transport is stubbed (`S/loop/rl_bank_unit_test.py:2109-2192`). The three passing checks verify mutated-workspace reset, absent prefix, fresh receipt supervision from zero and exactly-once publication/claim retirement (`E/F9R2-bank.log:300-303`). This is a real durable-path CPU test, **not a fresh live-Pi proof**.

Supply preserves tranche **96** and lanes **24/24/48/48**, while layering F9's 60-call rotation, terminal/conversation families and verified seeding (`S/loop/standing_supply.py:126-129,151-175,547-573`). Independently reran in the SAME Q checkout: **215 bank checks** and **74 F9 CPU tests / 3 Pi-dependent deselected** (`E/F9R2-bank.log:314`; `E/F9R2-cpu.log:6,12`). No count inflation or hidden suite flip found. Deployment still must update both wrapper preflight literals to 215 as documented (`S/F9-deployment.md:48-52,200-204`).

## B3 — rollback repair substantially PASS; named setup omission C2

Operator-readable before/after-injection boundaries, named manifests and ownership are now present:

- Backup/release/launch/environment/cron/import/boundary/injection manifests and exact destination roster are named (`S/F9-deployment.md:96-123`).
- Disable relaunchers and hold watchdog lock through stop/restore/initial restart; verified PID-birth/PGID ownership, bounded stop, STOP semantics, sole publisher and separately stopped durable service are explicit (`:127-165`).
- Before injection, require boundary gate GO, restore only manifest code/mutable authority and verify exact hashes/imports; restart service/publisher/collector before learner and supply, re-arm watchdog last (`:208-254`).
- After injection, compatible completion/publication is required before old-only restoration; otherwise isolate the **whole bank in place**, preserve sealed absolute paths and run any approved old-only code only on a new empty old-only bank (`:258-284`). No immutable job/receipt/archive/pin deletion is authorized.

The read-only boundary gate rejects pending/claimed new-family specs and any unpublished durable job, plus spec/publication identity drift (`S/rollback_gate.py:8-38`). Its offline boundary/no-write tests independently pass (`S/unit_tests.py:444-473`; `E/F9R2-cpu.log:12`). I extracted and executed the **exact documented restore Python block** only against new scratch fixtures: corrupt backup aborted before modifying either destination; valid backup restored old bytes and removed the added fixture source (`S/F9-deployment.md:218-239`; `E/F9R2-byte-qa.py:61-83`; `E/F9R2-byte.log:9`).

**C2 — setup omission, not a new architecture decision:** the gate command uses `$F9_TOOLS/rollback_gate.py` without defining F9_TOOLS in the runbook setup (`S/F9-deployment.md:89-94,209-210`). In an otherwise freshly initialized shell it resolves to `/rollback_gate.py`, so that command is not executable *as written*. Explicitly set `F9_TOOLS="$REPO/scripts/rl-loop-v1/F9-staged"` (or the retained compatible tool directory recorded by the release manifest) before either boundary sequence. `RECORDED_CWD`, `PROBE_MODULES`, PGIDs, BANK, deadlines and launch commands are attended inputs to be populated from the named manifest; do not infer them (`:93-106,134-138,177-196`). No other intrinsically unexecutable restore command was found given those operator inputs. This is not certification of actual owners, locks, cron or restart: those remain deployment-time GO/NO-GO gates (`:286-291`). The overall FAIL already blocks deployment regardless of this documentation correction.

## C1 — verified supply intake PASS

Production supply invokes `load_verified_seed_corpus` **before** seeded `author_instruction`, prints the intake decision and writes tranche `seed-intake.log`; verified seeds are held in memory (`S/loop/standing_supply.py:547-573`). Intake verifies the explicit manifest SHA pin, exact schema, corpus digest, count/item hashes and provenance, then re-verifies every solved seed through the real authority path; any caught absent/corrupt/admission `SystemExit` failure returns no seeds with an explicit disable reason (`S/scripts/e97_diversity_author.py:119-156`; actual proof/receipt/TRAIN/bundle/prompt/fixture verification `:96-116`). There is no silent fallback.

Independently executed CPU tests cover absent pin/manifest/corpus, corrupted bytes, count/item drift, authority ValueError/SystemExit, external fragment provenance, all **12 retained entries / 8 solved**, and the real supply callsite's logged disable without a teacher call (`S/unit_tests.py:487-550`; `E/F9R2-cpu.log:12`). Corpus/manifest hashes match the pinned historical bytes (`S/F9-evidence.json:79-80,128-133`; `E/F9R2-byte.log:3`). Ownership, read-only install and durable supply/watchdog environment pins are now explicit; intended seeded rollout is NO-GO on disable (`S/F9-deployment.md:293-316`).

## Regression sweep — existing PASS findings preserved within CPU scope

- Validator old-kind compatibility and archive retention remain passing; no era-2/3/4 validator bytes change (`S/unit_tests.py:178-207,553-568`; `E/F9R2-byte.log:8`).
- Exact rotation shares still sum to and populate 60 (`S/unit_tests.py:170-175`; `S/loop/standing_supply.py:151-175`).
- Deterministic sealed conversations, no acting before clarification, complete exchange/authorized user text and irrevocable legacy finish still pass (`S/unit_tests.py:119-148,293-307`). Driver selects the scripted safe bridge for conversations, safe bridge for end-state tasks and original bridge otherwise; ordinary tasks retain frozen-tools-v2 (`S/loop/rl_loop_driver.py:350-374`). Ordinary codec/bridge/provider shared hashes still match (`E/F9R2-summary.log:4`). This preserves wiring, not a claim that B1-R2 is safe.
- Morph split isolation/revalidation/dedupe, private-script prescreen, harvest and authority drift controls still pass (`S/unit_tests.py:151-167,310-317,353-362`). Fresh reference Pi/grade/degeneracy/canonical/missing-output gates remain in source (`S/scripts/e97_diversity_author.py:228-257`); no fresh runtime proof is claimed.
- Independently reran exactly the original legacy CPU selection: **44 passed / 5 real-Pi deselected** (`E/F9R2-legacy44.log:6,13`). Also ran the broader validator+codec+bridge selection: **66 passed / 5 deselected** (`E/F9R2-legacy66.log:6,15`). The five historical v1 Pi failures are not rerun or relabeled as qualified; their exact inner cause remains un-recertified (`tests/test_e97_pi_native_tool_bridge.py:39-95`; `E/F9R-review.md`, scope 7).
- All 17 historical private artifacts retain their hashes, including three distinct native-close proofs with canonical targets **180/267/180** and passing focused/regression summaries (`S/F9-evidence.json:78-98`; `E/F9R2-byte.log:3`; `E/F9R2-summary.log:5-8`). The three retained terminals independently regrade under repaired bytes in both modes; all three missing-output mutants reject (`S/unit_tests.py:571-586`; `E/F9R2-cpu.log:12`). These remain historical transports, **not fresh corrected-closure Pi qualification**. No new independently pinned protected-overlap receipt is claimed.

Applicable requirement IDs are **R14/NDP13 bounded local termination/safety intent and R16 exact-source evidence discipline**, not resilient-training conformance (`docs/RESILIENT_DILOCO_COMPUTE_POOL.md:113-122`; `docs/RESILIENT_DILOCO_GAP_MATRIX.md:33-34`). No scale, convergence, distributed membership or scheduler claim is made.

## Reproducible independent commands / receipts

On lambda01; common staged prefix (not Frontier):

```bash
source /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-F9C-reviewed/activate.sh
cd "$F9_STAGE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/erikg/emender/.venv/bin/python
"$PY" -m pytest -q -p no:cacheprovider /home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py -k 'not real_pi and not failed_reference_solution'
"$PY" loop/rl_bank_unit_test.py --scratch /home/erikg/emender-scratch/fix-fanout/F9R2-bank-fixtures
"$PY" /home/erikg/emender-scratch/fix-fanout/F9R2-grammar-qa.py
"$PY" /home/erikg/emender-scratch/fix-fanout/F9R2-byte-qa.py
```

Results: 74/3 deselected in 3.86s; 215 bank passed; grammar QA **exit 1, five real negative-control misses** across 117 cases; byte/restore QA passed (`E/F9R2-cpu.log:12`; `E/F9R2-bank.log:314`; `E/F9R2-grammar.log:118`; `E/F9R2-byte.log:1-10`). JSON/log output preserves ASCII escapes for Unicode bytes and every gate decision.

From repo cwd, same cache/CPU settings:

```bash
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator*.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
git diff --check
git diff --cached --name-only
```

Results: 44/5 deselected in 9.95s; 66/5 deselected in 12.70s; whitespace check passes; index empty. An ancillary line-number inspection initially used repo-relative paths after changing to Q and failed to open two documents; corrected inspection used absolute paths. It did not affect test outcomes or mutate source.

## Disposition

**FAIL / NO DEPLOY:** B1-R2 is a blocker at reference, policy and sealed gates. Repair unsupported operator normalization, add the independent negatives to implementation tests, regenerate/rebind/requalify exact bytes, correct C2's F9_TOOLS setup and obtain another independent review before operator deployment. Preserve the passing B2/C1 integration and all immutable historical evidence. Live ownership, restored imports, watchdog relaunch and any fresh Pi qualification require separate attended approval; this CPU-only review grants none.
