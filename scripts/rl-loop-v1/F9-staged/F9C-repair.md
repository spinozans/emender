# F9C blocker repairs (staged only; deployment remains NO-GO pending review)

## B1 — raw grammar, regenerated sealed validator

Raw checking is safer here than changing execution serialization: it preserves
approved single-quoted AWK and printf forms while rejecting expansion before
POSIX shlex strips quoting. Unquoted brace/tilde/glob/group/comment syntax and
unquoted or double-quoted substitution/escapes are refused. Original command
bytes are returned only after this raw check and the finite parsed grammar.

- F9R B1 and scope 1: `scripts/e97_offline_shell.py:78` raw boundary;
  `unit_tests.py:367` exact reviewer repro at direct/reference/policy gates;
  `unit_tests.py:397` authorized quoting and unsafe double quotes.
- F9R scope 2 regeneration: `sync_validator.py:7` returns generated source in
  memory; `unit_tests.py:410` equality, embedded rejection, 17/18-path closure
  and extra-component negative control. Embedded shell is regenerated.
- `prepare.py:21` binds repaired validator digest; its existing archive build
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
  retains CorrectionService; `:645` durable submission and no cycle fence.
  F9 layers only sealed-script/end-state arguments and removes F8's temporary
  conversation deferral. Owned workspace/attempt/grade/routing checks remain.
- `loop/rl_correction_{service,pool,config}.py` are byte-identical to current
  repo sources; `prepare.py:43` copies current repo helpers over deployed
  fallback helpers before overlaying F9. No live repo runtime is modified.
- `loop/standing_supply.py:126` retains tranche 96 and lanes 24/24/48/48;
  F9 rotation, family gates and dedupe remain. `unit_tests.py:438` asserts
  current source identity and knobs in the integrated checkout.
- `loop/rl_bank_unit_test.py:2109` replaces ONLY three superseded F8 rejection
  checks with real durable manager/OS worker/execute_correction_spec/fresh
  routing/publication/claim-retirement checks. The 212 others remain intact;
  count remains **215**. Model/Pi transport and teacher grade are CPU fixtures,
  not a live scripted correction proof. The worker checks mutated workspace
  reset, no prefix, authorized script; publication verifies real receipt chain.
- `F9-deployment.md:46` now requires deploying the suite flip and checking the
  preflight literal 215 in both wrappers. No additional bank count is hidden.

Validation: F9C-B2-bank.log **215 passed**; F9C-B2-tests.log **50 passed / 3
Pi-dependent deselected** in the SAME integrated tree. An initial bank assertion
used the wrong receipt field path; fixed to episode.supervise_from, rerun 215.
The additional source/knob identity test will be included in final qualification.
Historical supply digest is labeled historical, not a retained-current claim.

## Pending sections
B3 rollback and C1 verified corpus intake are not yet complete.
