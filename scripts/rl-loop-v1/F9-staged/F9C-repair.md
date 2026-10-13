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

## Pending sections
B2 integration, B3 rollback, C1 verified corpus intake are not yet complete.
