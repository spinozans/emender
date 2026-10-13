# F9C blocker repairs (staged only; deployment remains NO-GO pending review)

## B1 — F9C raw grammar, regenerated sealed validator (superseded by F9E)

F9C used raw checking before POSIX shlex stripped quoting, preserving
approved single-quoted AWK and printf forms while rejecting expansion. It
returned original command bytes after checks. F9R2 proved that insufficient;
the F9E structural parser/canonical-execution strategy below supersedes it.

- F9R B1 and scope 1: `scripts/e97_offline_shell.py:78` raw boundary;
  `unit_tests.py:373` exact reviewer repro at direct/reference/policy gates;
  `unit_tests.py:400` authorized quoting and unsafe double quotes.
- F9R scope 2 regeneration: `sync_validator.py:7` returns generated source in
  memory; `unit_tests.py:411` equality, embedded rejection, 17/18-path closure
  and extra-component negative control. Embedded shell is regenerated.
- `prepare.py:20` binds repaired validator digest; its existing archive build
  and registry rebind continue to update BOTH first-party pins consistently.
  `F9-evidence.json` distinguishes repaired closure from historical Pi proofs.

Validation: F9C-B1-tests.log: **50 passed / 3 deselected** (the original 42 CPU
checks plus eight repair cases); F9C-validator-tests.log: **59 passed** (full
first-action + protocol-breadth validators). Initial closure test had a wrong
helper name (49 passed/1 failed), corrected to the real manifest parser and
rerun successfully. No Pi episode/reference replay/teacher API was invoked.
Old immutable proofs/pins/archives are not rewritten or claimed as fresh.
Applicable safety intent: R14/NDP13 bounded local termination, R16 exact-source
evidence discipline; no distributed-training or resilient-conformance claim.

## B2 — current-source integration

- F9R B2/scope 3/service integration and scope 8: `loop/rl_bank_lane.py:45`
  retains CorrectionService; `:636` durable submission and no cycle fence.
  F9 layers only sealed-script/end-state arguments and removes F8's temporary
  conversation deferral. Owned workspace/attempt/grade/routing checks remain.
- `loop/rl_correction_{service,pool,config}.py` are byte-identical to current
  repo sources; `prepare.py:42` copies current repo helpers over deployed
  fallback helpers before overlaying F9. No live repo runtime is modified.
- `loop/standing_supply.py:126` retains tranche 96 and lanes 24/24/48/48;
  F9 rotation, family gates and dedupe remain. `unit_tests.py:434` asserts
  current source identity and knobs in the integrated checkout.
- `loop/rl_bank_unit_test.py:2109` replaces ONLY three superseded F8 rejection
  checks with real durable manager/OS worker/execute_correction_spec/fresh
  routing/publication/claim-retirement checks. The 212 others remain intact;
  count remains **215**. Model/Pi transport and teacher grade are CPU fixtures,
  not a live scripted correction proof. The worker checks mutated workspace
  reset, no prefix, authorized script; publication verifies real receipt chain.
- `F9-deployment.md:48` now requires deploying the suite flip and checking the
  preflight literal 215 in both wrappers. No additional bank count is hidden.

Validation: F9C-B2-bank.log **215 passed**; F9C-B2-tests.log **50 passed / 3
Pi-dependent deselected** in the SAME integrated tree. An initial bank assertion
used the wrong receipt field path; fixed to episode.supervise_from, rerun 215.
The additional source/knob identity test will be included in final qualification.
Historical supply digest is labeled historical, not a retained-current claim.

## B3 — executable rollback boundaries and ownership

- F9R B3/scope 8: `F9-deployment.md:85` names backup/release/launch manifests,
  repo-to-deployed roster, exact import/hash probes and preflight literal;
  `:125` stop/drain and watchdog-race guard includes durable service ownership;
  `:206` before-injection restore commands and restart order;
  `:256` after-injection completion or whole-bank isolation, preserving sealed
  absolute paths and all immutable evidence. Every step has GO/NO-GO/ABORT gates.
- `rollback_gate.py:8` is read-only and fails closed on layout/spec/publication
  drift; it refuses old-only restore with new pending/claimed work or ANY
  unpublished durable jobs. `unit_tests.py:444` tests both boundaries, no writes
  by the gate, and retained published evidence.

Validation: F9C-B3-tests.log **52 passed / 3 deselected**. No live stop, drain,
process inspection, restart or rollback was executed. Watchdog/PID ownership
and actual restored runtime probes remain explicit deploy-time gates, not
historical evidence recertified by documentation.

## C1 — verified seed intake at supply callsite

- F9R C1/scope 4: `scripts/e97_diversity_author.py:119` validates an explicitly
  pinned manifest hash, corpus hash, bounded schema, exact count/item hashes
  and provenance. Each solved seed goes through real verify_seed; invented
  fragments receive the bounded first-party provenance check. Any failure
  disables all seeding with a logged reason, never silent fallback.
- `loop/standing_supply.py:548` invokes this intake before author_instruction,
  prints the decision and records tranche seed-intake.log. Verified entries are
  held in memory; candidate guards/proofs/admission are never inherited.
- `F9-deployment.md:293` names corpus owner/location, exact corpus+manifest pins,
  read-only install and persistent watchdog/supply env. Unpinned seeding is
  explicitly disabled; intended seeded rollout is NO-GO on a disable log.
- `unit_tests.py:487,500,530,538` cover pin/hash/schema/count/entry/authority
  failures, absent corpus/manifest, external fragments, all 12 actual retained
  corpus entries (eight solved), and the REAL supply callsite's logged disable.

## Every F9R scope finding disposition

All paths in this report are relative to scripts/rl-loop-v1/F9-staged unless
an absolute root is stated. Scope findings not identified as defects are retained
rather than silently widened into new feature work.

| F9R finding | Repair / preserved evidence anchor |
|---|---|
| 1 Reference/policy expansion safety | B1; scripts/e97_offline_shell.py:78; unit_tests.py:373 uses exact repro at BOTH gates, also pip/python/sh/env/bash loaders. |
| 1 Failing native PG evidence, read-only assertions, descriptor capture | unit_tests.py:246,267,103 remain passing; unsafe public calls never dispatched. No OS sandbox claim. |
| 2 Self-contained equality, old/additive closure and registry pins | unit_tests.py:411,553; prepare.py:20,78; regenerated validator SHA a06740f36101bf571791510fde93f4c06739e27f94bec91245cf0b160c5eb634; repaired archive SHA ba0a76b26fb8ae0d5df1883d34601413f4fdb974380306838509ed94b3c489a8. |
| 2 Era2/3/4 immutability and old verdict compatibility | unit_tests.py:178,203,553; all three archive members compare byte-exact; old validators not edited. |
| 3 Sealed conversation/full exchange and irrevocable legacy finish | unit_tests.py:119,132,293 remain passing. |
| 3 Current F8-service fresh-conversation path formerly unqualified | loop/rl_bank_unit_test.py:2109 integrates durable admission, actual manager/worker, identity checks, workspace/prefix reset, passed fixture result and exactly-once fresh publication. CPU transport/grade fixtures explicitly not real-Pi proof. |
| 4 Harvest/morph/revalidation/dedupe gates | unit_tests.py:151,310,353 remain passing; C1 intake fixes production authority gap, no cached admission reuse. |
| 5 Exact 60-cycle rotation / operational knob drift | unit_tests.py:170,434; loop/standing_supply.py:126,129,151 retain CURRENT knobs + F9 rotation. |
| 6 Historical 45/45 and three Pi proofs | F9-evidence.json preserves private_evidence_root and private_artifacts unchanged with explicit historical labels; unit_tests.py:571 fresh CPU regrades six modes and rejects three missing-output controls. No replay/transport/API. |
| 6 Independent 42 CPU cases / source closure | Final 74 includes original 42 + 32 repair cases; exact-source evidence now binds the integrated closure and all unchanged helper bytes. Historical supply hash not represented as current. |
| 6 Protected-overlap receipt limitation | No new separately pinned protected-overlap proof is claimed; retained historical Pi build assertions remain historical. No admission/injection performed. |
| 7 Five legacy v1 Pi failures | tests/test_e97_pi_native_tool_bridge.py unchanged; five real-Pi cases excluded. Exact inner failure reason is NOT recertified; no live v1 qualification claimed. |
| 8 Deployment ownership/import/restart/watchdog gates, rollback omission | F9-deployment.md:85–290 supplies exact roster/manifests/probes, named stop/drain/ownership, restore commands, first-injection boundary, isolation/completion and GO/NO-GO/ABORT; rollback_gate.py:8 + unit_tests.py:444 test the boundary offline. Actual process/env verification remains deploy-time. |

## Integrated exact-byte qualification and scope receipts

Final private source closure: `$LAKE/F9-staged/qualified-F9C-reviewed`, rebuilt
from committed staged files for independent re-review. Historical Pi artifacts
remain at qualified-final; neither their pins nor archives were modified.
F9-evidence.json separates historical artifact roots from repaired source root,
sealed pin, all staged hashes and the complete integrated helper/runtime roster.

Commands (lambda01 project Python 3.12.3; bytecode/cache disabled):
- prepare.py --root qualified-F9C-final (private overlay only), then stage cwd;
- loop/rl_bank_unit_test.py --scratch E/F9C-C1-final-bank-fixtures: **215 passed**;
- pytest unit_tests.py -k 'not real_pi and not failed_reference_solution':
  **74 passed / 3 deselected** (E/F9C-C1-final-tests.log);
- repo cwd pytest tests/test_e97_first_party_validator*.py
  tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py
  -k 'not real_pi': **66 passed / 5 deselected** (E/F9C-final-legacy.log).
  The full validator-only set independently passed 59 in F9C-validator-tests.log.
- git diff --check; source/private artifact digest census; no staged files after
  commits. Scoped commits pushed: B1 a7c08222, B2 b9fc1d14, B3 6c68391f;
  C1 initial 6eb365b5; the follow-up commit containing this final report/evidence
  also covers SystemExit-based admission rejection, explicitly logged disable.

`E` is /home/erikg/emender-scratch/fix-fanout; `$LAKE` is the review's L parent.
Only F9-staged repo/lake sources and private CPU fixture artifacts were changed.
The live supply/async lane, bank config/tasks/claims/streams, old immutable pins,
GPU, teacher API and Pi episode transport were untouched. The current repo
runtime and bank suite are untouched; source copies in the staged tree are
integrated. A concurrent live authorization-file working-tree modification is
outside F9C and was left unstaged/unmodified by this repair.

Applicable IDs: R14/NDP13 bounded local safety and R16 exact-source evidence;
no training, scaling, checkpoint or resilient-conformance claim. Ready for new
independent review, NOT operator deployment. Live rollback, watchdog relaunch,
and new fresh-Pi closure are deliberately unclaimed deploy-time gates.

Final C1 inspection found production verify_admitted intentionally raises
SystemExit on schema/hash drift. Intake now catches that authority failure as
an explicit logged disable (not KeyboardInterrupt), with regression coverage.
Final integrated count is 215 bank / 74 F9 CPU; 3 Pi tests excluded.
The committed rebuild below is byte-bound to this same qualified closure.

## F9E — F9R2 structural grammar repair (supersedes F9C B1 strategy)

The supervisor's strategy ruling replaces validate-then-return-original with
**full-consumption parse then canonical execution**. F9C's raw-only strategy
above is historical and superseded. The original source bytes never dispatch:
`scripts/e97_offline_shell.py:78` lexes words/operators with quote provenance;
`:128` consumes the complete finite argv/separator/redirect grammar; `:165`
serializes only parsed structure with single-quoted words and supported syntax.
Unsupported operator runs, expansions, escapes, control bytes, empty/trailing
commands and unconsumed redirect suffixes reject. No argv fallthrough for syntax.

| F9R2 finding | F9E fix and permanent evidence |
|---|---|
| B1-R2 five accepted `||`/`;;` negatives | `test_f9r2_grammar.py:62` names forbidden printf format, Python loader, quoted sh loader, double-semicolon and plain-double-pipe cases; `:100` checks all four direct/sealed/reference/policy gates. All five now reject. |
| Full 117-case representation matrix | Reviewer construction is retained in `test_f9r2_grammar.py`; exact original results JSON is retained as HISTORICAL baseline in `F9R2-grammar-results.json`. `:75` asserts matrix identity/counts (104 reject, 13 accept); `:100` fails on any gate regression, never executing negative strings. |
| Authorized forms preserved and original bytes never execute | `:147` executes ONLY the 13 approved controls in isolated scratch CPU workspaces, comparing exit/stdout/stderr/all file bytes for original and canonical forms. `:100` proves pending, returned calls and public history contain canonical bytes, raw frames/positive PG masks remain, and subsequent canonical-history ingest succeeds. |
| Exact execution receipts after canonicalization | Supervisor-approved R16 fix: `loop/rl_loop_driver.py:369,530` marks only end-state/safe scripted records and projects canonical executed action arguments, retaining generated raw actions in native_record/source_messages for PG. `test_f9r2_grammar.py:192` tests differing and identical forms plus unchanged ordinary projections. Labels now distinguish executed versus generated actions. |
| Sealed rebinding and 111 pins | Regenerated validator equality, old 17/additive 18 closures, extra-component rejection, both first-party registry pins, exact era2/3/4 archive members, 22 staged + 60 runtime + 17 historical artifact + 12 retained log pins all pass; `prepare.py:20` binds the new validator. Historical receipts/archives/qualification logs were not overwritten. |
| C2 undefined F9_TOOLS | Separate runbook commit sets `F9_TOOLS="$REPO/scripts/rl-loop-v1/F9-staged"` at `F9-deployment.md:93` before both boundary sequences, records/retains tool path and hashes in release.manifest.json; boundary command now resolves to the retained `rollback_gate.py`, not `/rollback_gate.py`. |
| B2/C1/B3 previously passing findings | Same integrated tree retains current durable service, 96/24/24/48/48 knobs, verified seed intake, read-only boundary gate, legacy compatibility and historical proof hashes. No live qualification is claimed. |

New sealed validator SHA256:
`7abebc445b1cf48feba7a24bb585bf8644c9c144d972b651fa4f9d599d522cd0`.
New additive source archive SHA256:
`061133d6f564f19e580bc864876e65dd032c5832f9c7ce2aaee50f1470270f69`.
Private integrated source closure: `$LAKE/F9-staged/qualified-F9E-final`.
`F9-evidence.json` rebinds all applicable source pins, retains historical pins,
and separately pins the permanent regression sources and seven new CPU logs.

### Reproducible validation (lambda01, project Python 3.12.3)

`E=/home/erikg/emender-scratch/fix-fanout`; `S` is this staged directory.
No Frontier, scheduler, training, daemon, GPU, teacher, admission/injection or
live Pi transport was used. Only approved literal equivalence controls execute;
bank-suite processes/signals are CPU fixtures, as in F9R2.

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/erikg/emender/.venv/bin/python
"$PY" "$S/prepare.py" --root "$LAKE/F9-staged/qualified-F9E-final"
source "$LAKE/F9-staged/qualified-F9E-final/activate.sh"
cd "$F9_STAGE_ROOT"
"$PY" -m pytest -q -p no:cacheprovider "$S/unit_tests.py" "$S/test_f9r2_grammar.py" -k 'not real_pi and not failed_reference_solution'
"$PY" loop/rl_bank_unit_test.py --scratch "$E/F9E-final-bank-fixtures"
"$PY" "$E/F9R2-byte-qa.py"
cd /home/erikg/emender
"$PY" -m pytest -q -p no:cacheprovider tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
git diff --check
git diff --cached --name-only
```

Receipts: `F9E-final-grammar.log` **148 passed**, including the entire **117**
four-gate matrix, **13** actual authorized equivalence tests, matrix identity,
**12** suffix negatives, **3** canonical composition checks and **2** execution
projection checks. `F9E-final-cpu.log` **74 passed / 3 Pi-dependent deselected**;
combined documented suite `F9E-final-integrated-cpu.log` **222 passed / 3
deselected**. `F9E-final-bank.log` **215 passed** in the SAME integrated tree.
`F9E-final-legacy44.log` **44 passed / 5 real-Pi deselected**.
`F9E-final-byte.log`: **all 111 pin occurrences MATCH**, regeneration/closures/
archive pins/legacy members pass; exact documented restore block passes both
corrupt-backup abort and valid restore using new scratch-only fixtures.
Initial battery run: 145 passed/1 failed because an added pipeline test used
`wc -c` without the required path; corrected to supported `cat | sort` path
forms, then all 148 pass. No implementation or approved grammar was broadened.

Applicable IDs: **R14/NDP13 bounded local safety intent, R16 exact-source
execution evidence discipline**, not resilient-training conformance. Residual
scope: independent review still REQUIRED; deployment remains **NO-GO** pending
review and separate attended approval. Historical Pi receipts remain historical;
no fresh corrected-closure Pi proof or new protected-overlap receipt is claimed.
Old immutable sources, archives, pins, receipts and all live systems untouched.
