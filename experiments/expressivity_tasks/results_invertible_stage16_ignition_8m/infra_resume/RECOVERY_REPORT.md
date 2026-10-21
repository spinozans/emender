# Attended infrastructure recovery validation
2026-10-21: root ENOSPC interruption, not model failure. Original registrations verify
against worktree and HEAD; all retained checkpoint SHA256 values verify and the final
pruning receipts explain the retained set and every missing registered checkpoint.
AdamW state and all three RNG states are present in each chosen checkpoint.
Admission uses per-run dated amendments/supersessions and new output namespaces.

Command (lambda01, existing Python 3.12 venv; this is NOT Frontier):
`.venv/bin/python -m pytest -q tests/test_invertible_stage16_17_resume.py tests/test_invertible_stage16_ignition_8m.py tests/test_invertible_stage17_carrywean_8m.py`
Result: 48 passed. Eleven new parametrized recovery cases cover disk fail-closed writes,
exact diagnostic mismatches, truncated prefix scope, missing/duplicate replay rows,
checkpoint/hash/pruning/optimizer admission, exact restored state, original schedules,
fixed GPU bindings excluding 5/7, and pruning only the new output namespace.

Architecture authority: RESILIENT_DILOCO_COMPUTE_POOL.md ADR-003 (2026-07-31).
This is standalone expressivity, not resilient DiLoCo. R07/R12 checkpoint/recovery
safety intent and R16 evidence discipline are relevant analogies only, NOT conformance.
R02-R06/R08-R11, NDP01-NDP17, V21S01-V21S17, ISP01-ISP07 and Frontier scheduler
requirements are unclaimed/not applicable. No scale or overlap gate is asserted.

Disk floor: 200 GiB before every normal runner write, including per-update JSONL,
checkpoint staging, validation, receipts and deletion logs. Below floor: retain the
last hash-receipted checkpoint, exit nonzero; tiny emergency terminal metadata is the
explicit exception. A pre-write guard cannot promise success during concurrent
external exhaustion. Original retained checkpoints are never deleted; new checkpoints
follow the same best/latest/criterion selection, with hashed deletion receipts.
Original scientific evaluation/schedules/ceilings are unchanged; no automatic retry.

Replay results and live identities will be appended after committed-source launch.
Exact diagnostic equality is evidence, not proof of bitwise full-state trajectory identity.
