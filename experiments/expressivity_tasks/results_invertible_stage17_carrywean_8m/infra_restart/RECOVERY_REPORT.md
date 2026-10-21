# Approved restart admission after exact-replay divergence
Supervisor explicitly approved checkpoint re-execution for e97_w32, curriculum and X17.
The three verified arms (e97_w64, GDN2 w512 and w2048) continue without redo.
New amendments cite each actual mismatch and the independent X15 divergence receipt.
No tolerance was weakened, and no at-death-state restoration is claimed for restarts.
Every new restart training row identifies the superseded historical tail boundary.
Original global schedules and original ceilings apply; X17 remains1M, sweep500K,
criterion-gated curriculum retains500K teaching cap and500K T64 extension.

Validation command on lambda01 (not Frontier):
`.venv/bin/python -m pytest -q tests/test_invertible_stage16_17_resume.py tests/test_invertible_stage16_17_restart.py tests/test_invertible_stage16_ignition_8m.py tests/test_invertible_stage17_carrywean_8m.py`
Result:54 passed. The first runner is deliberately frozen: changing its registered bytes
would invalidate the three already-running replay-verified admissions. The separate
restart runner reuses its guard/admission primitives but implements explicit fresh
post-checkpoint provenance, without replay comparisons or replay-success receipts.

RESILIENT_DILOCO_COMPUTE_POOL.md ADR-003 authority reviewed. Standalone expressivity,
not DiLoCo/Frontier: R07/R12 safety intent and R16 evidence discipline are analogies,
not conformance; elastic/native NDP/V21S/ISP and scheduler/overlap claims remain unclaimed.
Disk guard is pre-write200GiB, tiny emergency metadata excepted; concurrent external
exhaustion still cannot be absolutely prevented. No automatic retry or GPU7 use.
Native supervisors must monitor scripts.watch_invertible_stage16_17_recovery with
--stage16/17 (separate processes); it routes each arm to its actual immutable admission.
