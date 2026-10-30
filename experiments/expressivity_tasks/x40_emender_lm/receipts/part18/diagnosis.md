# X40 part 18: isolated mask proof; repair candidates rejected

**FAIL / incomplete. No production promotion, race repair, recertification,
9x acceptance or depth verdict is claimed.** Production `kernels.py` is byte
identical to commit `231fcd9c`. The requested green-completion commit title
would be false. This receipt preserves bounded, verified partial work and all
failed experiments. There is no production candidate left installed to revert.

## Cache operand reasoning and isolated tests

`mask_proof.md` gives the branch/lifetime induction, including local `t==0`
on every native continuation and imported state. The mask-only candidate is
separate from the part-17 barriers: no reduction expression, warp count,
assertion or production source is changed. Zero-initialization was not chosen;
it cannot repair shared scratch or the failed all-shape certification.

`lifecycle_retry.json` passes **16 fixtures**: all four policies, empty/imported
origins, ordinary/unit decay, and native continuation through chunk lengths
1/2/31/1/35. Each fixture compares original source with finite cache poisoning,
original source with NaN poisoning, and the mask-only candidate with NaN
poisoning. All compared bytes agree, including public values, defined native
state/metadata, traces, pressure, evidence/errors and history. Every fixture
observes POPs, frees, allocations and slot reuse. The first invocation failed
in the comparison harness because a singleton contiguous tensor retained a
nonunit stride for dtype reinterpretation. The retry compares NumPy bytes;
the initial failure log is retained. This is a harness correction, not a
changed kernel or tolerance.

Strict unchanged-X31 gates retain values, state, traces, pressure and gradients:

| Batch/context | Result |
| --- | --- |
| B64/T512, all four policies | PASS |
| B128/T512, all four policies | PASS |
| B64/T2048, direct existing certify() | FAIL in X31 backward journal rebase: CUDA invalid argument |
| B64/T2048, existing validated launch adapter | PASS, all four policies, strict pressure retained |
| B128/T2048, existing validated launch adapter | FAIL in X31 journal allocation: checked int32 capacity exceeded |

The adapter is the existing `probe_rebase.TransposedLaunch`, not a new oracle:
its validation checks integral address mapping and the formerly oversized
grid. Its use and original direct-call failure are separately retained. The
B128/T2048 capacity guard fires in the oracle backward before a candidate
comparison can complete. No batch splitting, reference substitution, pressure
exclusion, numerical tolerance change or bypass of that guard was made.
**The all-shape proof suite is therefore incomplete, and the mask was not
promoted.** Passing the smaller rows does not satisfy the standing contract.

## Fully instrumented sanitizer results

Only the process-local mask candidate is used in these rows:

| Input | Initcheck | Racecheck |
| --- | --- | --- |
| Captured B8/T2286/S2944 scan-10 input | PASS, exit 0, application PASSED, zero errors | FAIL, exit 86, 32 groups, int32-add assertion |
| B64/T512, four production policies | PASS, exit 0, zero errors | FAIL, exit 86, 20 groups, heap inverse-position assertion |

These are complete isolated invocations, with no kernel filtering and unlimited
report printing. A different assertion is not a resolution of the original
assertion. Racecheck stops on failure; it does not cover all later policies.
No group is certified benign and no sanitizer acceptance is claimed.

An initial attempt instrumented **the entire X31 certification harness**,
rather than just the candidate. It reported additional uninitialized reads in
the unchanged X31 `materialize_state`, kernels.py:378, reached via
cell_fused.py:204 and parity.py:45. It was terminated deliberately (exit 143)
after retaining 68,620,056 raw bytes; this is neither a terminal sanitizer pass
nor a complete error inventory. `mask_certified_initcheck.txt.gz` is lossless
(decompressed SHA256
`bdc8b5f49cdc0554abf1579cd6e89de1caa60941ccbde2792ddec530ebc76c9c`).
This reference-side report must not be attributed to the X40 cache site or
silently ignored when making a future whole-harness sanitizer claim.

## Generated scratch repair attempts, all rejected

1. PTX v1 adds 310 CTA barriers bracketing scalar/vector `ld.shared` and
   `st.shared` accesses. It preserves all original instruction text apart from
   barriers. **FAIL:** 11 groups and the int32-add assertion. The initial
   launcher suffix was edited while its shell was running, causing a suffix
   parse error; raw sanitizer output and API exit 86 are retained as manual
   accounting in `summary.json`, not a normal runner status entry.
2. PTX v2 also brackets `ldmatrix` accesses (330 added barriers). **FAIL:**
   10 groups and the int32-add assertion. Generated SASS confirms that barriers
   survived assembly; source-level barrier insertion alone is not sufficient.
3. V3 additionally restricts two duplicate-address vector stores using exact
   register/address maps from the hash-bound failing specialization. This is
   explicitly **not** a proved value-neutral repair. **FAIL:** application
   hits the heap inverse-position assertion although racecheck displays zero
   groups before termination. It is rejected. Zero groups on an aborted,
   changed-value-unproved application cannot establish either successful
   race repair or independence of the original assertion. The first v3
   invocation instead failed a PTX declaration-matching precondition before
   launch; its zero-group summary is also not a pass.
4. Source-only expanded masked vector stores attempt to keep reduction results
   in their input layout without changing reduction expressions. **FAIL:**
   26 groups and the int32-add assertion. This candidate is rejected.

All experiments are process-local. Control-flow legality of the blanket PTX
barriers and value-neutrality of V3 are unproved. They must not be copied into
production. The original and transformed PTX/cubins, hashes, and selected SASS
are retained. The synthetic 128x8 reduction probe passes racecheck and its own
repeat/count checks; it **does not reproduce** the full scan's failure. It
lacks the captured operands and full mutation/control-flow context and is not
an upstream-quality reduction of the defect.

`summary.json` maps all **193 printed access endpoints in the mask-only run's
32 groups** to instructions/context from its exact captured cubin, using IDs
P18MASK-G01–G32. These are new experiment IDs. They do not replace R01–R35.
The original 35 IDs remain individually listed as REAL and unresolved. Exact
original-PC/cubin provenance and cross-specialization scratch ownership are
still missing, so no subtraction of displayed group counts is a repaired-ID
claim. No R-ID is closed by this session.

## Part-4 control and remaining gates

The original pressure probe was rerun with only its lease allowlist and output
directory adapted; X31 source remains unchanged. This is a **pre-promotion
control**, not the requested post-fix connection test. Its six unchanged-X31
runs have 102–109 semantic pressure mismatches against reconstructed history;
repeat pressure differs at 42–75 positions. The six temporary barrier-fork
runs have zero semantic mismatches and zero repeat differences, with other
fields equal; 140 one-ULP differences versus torch division remain recorded.
The harness exits 0 because its diagnostic invariants pass, **not** because
unchanged X31 is pressure-deterministic. Shared origin versus two independent
bugs remains unresolved; an X40 cache mask cannot mechanically repair X31.

The failing sanitizer and all-shape gates prevent promotion. Consequently the
109-oracle suite, graph/eager recertification, 2M-token bit-matched curve, 100
full production 9x repeats, post-fix connection test, production 9x bin,
budget-scaled depth probe and final verdict are **not run / not satisfied**.
No claim of full recertification is supported by the passing local fixtures.

Next work needs a correct scratch ownership/lifetime repair with preserved
values/reduction trees, then all sanitizer and standing recertification gates.
The reference-side journal guards and uninitialized-read report also require
explicit handling in the certification harness; they are not permission to
weaken or bypass acceptance.

## Execution discipline

Host lambda01; only `/home/erikg/emender/.venv/bin/python`. GPU work was
single-threaded and sequential, exclusively broker-leased on physical card 2
with allowlist 2/3/4. No forbidden card selection, search interruption, agents,
WG service/dispatcher/workers, Slurm or resilient training changes. DiLoCo
requirement IDs are inapplicable. No .pt is committed; the original external
capture remains `/tmp/x40_part15_scan10.pt`. Unrelated workspace edits are
untouched. `contact_supervisor` was searched first and again by name and
description but is unavailable; no substitute supervisor channel was used.
