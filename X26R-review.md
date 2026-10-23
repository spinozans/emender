## Review

**Verdict: ACCEPT-WITH-CONDITIONS** for the four-arm **partial replication under reconciled sources**, not an exact Figure 2 replication.

Paths below abbreviate `experiments/expressivity_tasks/results_stage26_replication/` as **X26/**.

### Correct

- **Source reconciliation:** Table 1 extraction matches the retained paper layout: all models dm256/L2, LR3e-4/60k; RNN dh256; DeltaNet/RWKV-7 four heads; Mamba state128 DGC/state64 IMM (`X26/sources/paper-layout.txt:1898–1911`). Section 6 instead specifies BCE, AdamW lr/wd1e-4, batch64, clip1 and early stopping (`:589–598`). The ledger explicitly reconciles these with appendix batch128 and repository architectures (`X26/DISCREPANCIES.md:12–48`).

- **Task-contract ruling:** The reviewed X23E scalar stream remains unchanged: 9N tokens, exact Python-integer products, final `P[0,0]>0`, no header or product-state inputs. Auxiliary supervision uses the shared output head after each intermediate matrix’s ninth scalar, with exact positivity targets and effective-batch normalization (`experiments/expressivity_tasks/run_merrill26_replication.py:69–109,191–196`). Scalar encoding, curriculum and DGC mean pooling are disclosed adapter choices (`X26/REGISTRATION.md:31–37`). Only four primary arms are authorized; native tokenization is explicitly deferred pending failure and new registration (`:16–24,28–37`).

- **Models:** Constructors instantiate the retained upstream classes: DGC clamped-ReLU1×256/embedding128, IMM GRU2×256, and task-specific DeltaNet1×256/four-head configurations (`experiments/expressivity_tasks/merrill26_replication_models.py:39–74`). Upstream recurrence, pooling and heads support those descriptions. Smoke parameter counts are **232450 / 791297 / 430146 / 267841** for RNN graph/integer and DeltaNet graph/integer respectively.

- **Receipts:** All four smoke JSON receipts report passed forward/backward/evaluation, zero optimizer updates and peak allocated memory. Both graph probes contain **90,904 tokens**; integer probes contain 2,700 (`X26/receipts/smoke_*.json:1–13`). Raw tests report **32 passed**, exit0 (`X26/receipts/tests.txt:6–12`; `tests_exit.txt:1`). Launch verification records cards2–5, trainer identities, physical GPU UUIDs and reconciled optimizer settings (`X26/receipts/launch_verified.json:1–163`).

- **Registration and interpretation:** Supplied history artifacts plus the parent’s ancestry audit establish `af649fd8 → caf82621 → 4ce3af75`; launch/config receipts identify registration commit `af649fd8`. Predictions explicitly target the qualitative pattern and prohibit inferring a generator bug or architectural inability from failure alone (`X26/REGISTRATION.md:7–24`).

### Finding

- **P2 — misleading numerical-threshold wording.** `X26/DISCREPANCIES.md:53–54` says numerical thresholds are registered, while `REGISTRATION.md:14–15` explicitly describes qualitative targets without numeric Figure 2 coordinates or post-hoc thresholds. Smallest correction: clarify that **qualitative predictions**, not numerical success thresholds, were registered. Preserve the original hash-bound registration; receipt any clarification separately.

### Conditions

1. Parent audits must confirm registration/source/dependency hashes, upstream snapshot identity against `dbfa365ad680f425dfb79a9e78b5e0d7915757bb`, and the recorded 32-test suite. I inspected receipts and source; I did not execute commands or independently inspect committed diffs.
2. Carry the qualitative-only clarification into the acceptance record. No exact-Figure-2 or task-bug attribution is licensed; retain the registered failure-triggered native-control procedure.

- **Fixed:** None; review-only.
- **Merge verdict:** **OK with notes**, conditional on the parent audits above. No training outcomes adjudicated.
---

## Parent attestation (2026-10-23)

X26R conditions satisfied: (1) parent audits - registration.json sha256 matches the
watcher-wiring registered hash (e8b236cc...); upstream snapshot identity verified:
all 14 vendored LinearRNN files hash-match the sources/manifest.json at commit
dbfa365ad680f425dfb79a9e78b5e0d7915757bb (0 mismatches); the receipted 32-test suite
reruns green by the parent (32 passed), and test_merrill26_separators separately
(23 passed). (2) The qualitative-only clarification is appended to DISCREPANCIES.md
as a separately-receipted note; the hash-bound REGISTRATION is unchanged. The
partial-replication scoping and failure-triggered native-control procedure stand.

Verdict: X26R ACCEPT-WITH-CONDITIONS -> ACCEPTED. X26 replication machinery closed;
four arms training under accepted machinery.
