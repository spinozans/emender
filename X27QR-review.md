## Review

**Acceptance verdict: ACCEPT-WITH-CONDITIONS.** No issues found.

Reviewed the supplied committed-code patch and current X27 files, relying on the parent’s attestation that those files equal `bfa665fb`. No edits or test executions performed.

### Correct
- **Parser:** comma/whitespace splitting fixes the producer’s CSV format; seven cases cover empty, mixed and newline output (`scripts/watch_small_architecture_separators.py:53–54`; `tests/test_small_architecture_separators.py:76–92`).
- **Guard scope:** restartability explicitly requires task X27; the default shared-launch guard remains (`scripts/watch_merrill26_separators.py:95–117`). Exhaustive caller grep found only X27 passing `restartable_x27=True`; replication, recipe-controlled, native-control and softplus watchers retain `protocol.launch()`.
- **Independent arms:** exclusive receipt-directory creation prevents duplicate launches without blocking another queued arm (`scripts/watch_small_architecture_separators.py:28–38`). The restart test preserves A/shared receipts, launches B once and rejects duplicate B (`tests/test_small_architecture_separators.py:193–225`).
- **Fallback:** registration is reverified before each spawn; preferred card wins when available, otherwise lowest free index; each selected card is consumed once (`scripts/watch_small_architecture_separators.py:56–65`).
- **Actual launches:** both per-arm `launch.json:2–12` receipts record commit `50f9fe8d`, registration SHA and watcher PID1384720:
  - `e97softplus_graph_seed0`: preferred0 → actual0.
  - `gdn2_graph_seed0`: preferred1 → actual2.
  Their `lease.txt:1–7` receipts agree. `receipts/queuefix/postlaunch_verification.json:17–85` records matching trainer PID/GPU UUID bindings and `nvidia-smi` evidence. Acceptance does **not** depend on potentially stale polled states.
- **Recovery evidence:** original console explicitly records `ValueError` parsing `'4,5'` (`receipts/watcher_console.txt:264–276`); shared-receipt blockage is preserved in `receipts/queuefix/diagnosis.txt:1–39`. Parent audit confirms persistent PID1384720 remains running.
- **Tests:** pre/post-launch receipts both show **62 passed total, including 24 X27 tests**, not 24 plus 62 (`receipts/queuefix/{tests,postlaunch_tests}.txt:6–14`).

### Exact acceptance conditions
1. **Retain the explicit append-only exception:** the parent reports that `source_sha256` was refreshed in place. Therefore strict append-only comparison fails. Accept this only as the supervisor-authorized, ledgered exception—not as strict append-only compliance. The original byte-identical registration and five before/after hash entries are preserved (`registration.json:1061–1099`).
2. **Retain the parent’s preservation/provenance audit with this review:** original registration SHA matches; original `REGISTRATION.md` remains a byte-prefix; shared launch and existing tanh identity receipts are unchanged; prior watcher wiring is preserved; reviewed paths match `bfa665fb`.
3. Report the test count accurately as **24 X27 / 62 total**, and live-process evidence as audit-time evidence rather than a continuing guarantee.

- **Fixed:** None; review-only.
- **Merge verdict:** **OK with notes**, subject to the conditions above.
---

## Parent attestation (2026-10-24)

X27QR conditions satisfied: (1) The append-only exception is retained EXPLICITLY as
supervisor-authorized and ledgered - source_sha256 refreshed in place per the parent's
authorization ('refresh only the changed X27 infra source hashes'), with the byte-identical
original preserved at receipts/queuefix/registration_before.json (sha 83834d65... verified
against the d3cac70b git blob and the amendment's recorded original_registration_sha256) and
the five before/after hash entries in the amendment's source_sha256_changes ledger; this is
NOT strict append-only compliance and is not claimed as such. (2) The parent's
preservation/provenance audit is retained with this review in full: original registration
sha match, REGISTRATION.md byte-prefix preservation, shared launch.json and tanh identity
receipts (trainer.json, command.txt, lease.txt) unchanged since 4cc2d1b0, prior watcher
wiring preserved at queuefix/watcher_wiring_before.json, reviewed paths equal bfa665fb
(parent-verified, nothing staged). (3) Test counts stated accurately: 24 X27-specific
tests within a 62-test total run (not 24+62); live-process evidence (launcher PID 1384720,
GPU/UUID bindings) is audit-time evidence, not a continuing guarantee. The 62-test suite
reruns green under the parent (24+38 combined scope verified at audit time).

Verdict: X27QR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X27q queue-fix closed.
