# F9C blocker repairs (staged only; deployment remains NO-GO pending review)

## B1 — raw grammar, regenerated sealed validator

Raw checking is safer here than changing execution serialization: it preserves
approved single-quoted AWK and printf forms while rejecting expansion before
POSIX shlex strips quoting. Unquoted brace/tilde/glob/group/comment syntax and
unquoted or double-quoted substitution/escapes are refused. Original command
bytes are returned only after this raw check and the finite parsed grammar.

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
- `unit_tests.py:487,500,528,536` cover pin/hash/schema/count/entry/authority
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
| 2 Self-contained equality, old/additive closure and registry pins | unit_tests.py:411,551; prepare.py:20,74; regenerated validator SHA a06740f36101bf571791510fde93f4c06739e27f94bec91245cf0b160c5eb634; repaired archive SHA ba0a76b26fb8ae0d5df1883d34601413f4fdb974380306838509ed94b3c489a8. |
| 2 Era2/3/4 immutability and old verdict compatibility | unit_tests.py:178,203,551; all three archive members compare byte-exact; old validators not edited. |
| 3 Sealed conversation/full exchange and irrevocable legacy finish | unit_tests.py:119,132,293 remain passing. |
| 3 Current F8-service fresh-conversation path formerly unqualified | loop/rl_bank_unit_test.py:2109 integrates durable admission, actual manager/worker, identity checks, workspace/prefix reset, passed fixture result and exactly-once fresh publication. CPU transport/grade fixtures explicitly not real-Pi proof. |
| 4 Harvest/morph/revalidation/dedupe gates | unit_tests.py:151,310,353 remain passing; C1 intake fixes production authority gap, no cached admission reuse. |
| 5 Exact 60-cycle rotation / operational knob drift | unit_tests.py:170,434; loop/standing_supply.py:126,129,151 retain CURRENT knobs + F9 rotation. |
| 6 Historical 45/45 and three Pi proofs | F9-evidence.json preserves private_evidence_root and private_artifacts unchanged with explicit historical labels; unit_tests.py:569 fresh CPU regrades six modes and rejects three missing-output controls. No replay/transport/API. |
| 6 Independent 42 CPU cases / source closure | Final 73 includes original 42 + 31 repair cases; exact-source evidence now binds the integrated closure and all unchanged helper bytes. Historical supply hash not represented as current. |
| 6 Protected-overlap receipt limitation | No new separately pinned protected-overlap proof is claimed; retained historical Pi build assertions remain historical. No admission/injection performed. |
| 7 Five legacy v1 Pi failures | tests/test_e97_pi_native_tool_bridge.py unchanged; five real-Pi cases excluded. Exact inner failure reason is NOT recertified; no live v1 qualification claimed. |
| 8 Deployment ownership/import/restart/watchdog gates, rollback omission | F9-deployment.md:85–290 supplies exact roster/manifests/probes, named stop/drain/ownership, restore commands, first-injection boundary, isolation/completion and GO/NO-GO/ABORT; rollback_gate.py:8 + unit_tests.py:444 test the boundary offline. Actual process/env verification remains deploy-time. |

## Integrated exact-byte qualification and scope receipts

Final private source closure: `$LAKE/F9-staged/qualified-F9C-committed`, rebuilt
from committed staged files for independent re-review. Historical Pi artifacts
remain at qualified-final; neither their pins nor archives were modified.
F9-evidence.json separates historical artifact roots from repaired source root,
sealed pin, all staged hashes and the complete integrated helper/runtime roster.

Commands (lambda01 project Python 3.12.3; bytecode/cache disabled):
- prepare.py --root qualified-F9C-final (private overlay only), then stage cwd;
- loop/rl_bank_unit_test.py --scratch E/F9C-final-bank-fixtures: **215 passed**;
- pytest unit_tests.py -k 'not real_pi and not failed_reference_solution':
  **73 passed / 3 deselected** (E/F9C-final-tests.log);
- repo cwd pytest tests/test_e97_first_party_validator*.py
  tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py
  -k 'not real_pi': **66 passed / 5 deselected** (E/F9C-final-legacy.log).
  The full validator-only set independently passed 59 in F9C-validator-tests.log.
- git diff --check; source/private artifact digest census; no staged files after
  commits. Scoped commits pushed: B1 a7c08222, B2 b9fc1d14, B3 6c68391f;
  C1 commit is the commit containing this final report/evidence.

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
