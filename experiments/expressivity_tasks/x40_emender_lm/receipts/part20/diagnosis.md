# X40 part 20: promotion gate rejected; production restored

FAIL / incomplete. Neither mask is promoted. No all-shape acceptance, race
repair, 100-repeat 9x result, depth probe, pressure connection or final depth
verdict is claimed. The requested "full battery green" commit title would be
false. This is a failure receipt with a reusable combined lifecycle fixture.

## Combined candidate and lifecycle proof

Read part19 first, including 680d0359 and c53c8d42. `candidate.patch` records
one combined candidate: mask ReadStrengthCache by
`r_live & (r_touch<Start+t-1)` in X40, and mask Birth by `valid & live` in
both X31 and X40 materialization. `candidate.json` binds before/after hashes.
No reduction, store, barrier, numerical tolerance, pressure assertion or
journal guard changes were included.

`lifecycle.json` passes 32 fixtures (16 X40, 16 X31). Each target tests all four
policies, fresh/imported origins, ordinary/unit decay, and native continuation
through chunks 1/2/31/1/35. Every fixture observes allocations, POP deaths,
frees and slot reuse. Baseline and candidate snapshots agree byte-for-byte
on public values, defined state/metadata, traces, pressure and history;
X40 evidence/errors agree too. Never-live undefined metadata is excluded
using the existing fixture contract. X40 cache variants retain finite/NaN
poisoning; X31 has no ReadStrengthCache and compares the Birth variants alone.
Both Birth versions are exercised through real lifecycle calls, rather than
substituting the isolated part19 materialization assertions. These fixtures
are differential evidence, not proof of race freedom or full recertification.

`isolated_initcheck.txt` passes the original captured B8/T2286/S2944 scan-10
input with zero errors. `isolated_initcheck.json` binds the captured input by
SHA256. The fixture remains external in /tmp; no .pt file is committed.

## The full-harness initcheck fails on host API clones

The B64/T512 certification harness was instrumented without kernel filters or
suppression (`--tool initcheck --print-limit 0 --error-exitcode 86`). After the
two load masks, it produced cudaMemcpy-source uninitialized-access reports
in the backward clone path. This violates the requested zero-error gate.
The invocation was deliberately stopped once this failure was established;
it is not a terminal sanitizer pass or a complete error inventory.

A receipt-handling mistake lost the original failing raw log: `run.sh` was
edited to insert provenance recording while its shell was active. On
termination the shell resumed at shifted file offsets, encountered a command
and syntax error, and reopened/truncated the log. The surviving compressed
file contains only that shell error. The outer shell returned 2 and did not
append its normal status entry. `certified_B64_T512_initcheck.failure.json`
explicitly records this evidence loss. Do not interpret its surviving-file
zero counts as zero sanitizer errors. The first observed tool excerpt named
`backward.py:245` and a 285,999,104-byte copy; it cannot replace the lost log.
No active shell runner was edited during the independent diagnostic below.

After stopping the failed invocation, `rollback.py` restored both production
files, verifying their exact pre-candidate SHA256 values. `rollback.json`
records that restoration; neither production file has a git diff.

`reduced_clone.py` independently instruments B1/T8 with the same masks applied
only in-process after source rollback. Its printed reports include X31
`backward.py:237` (whole replay-owner clones) and `:245` (whole journal-address
clone). The diagnostic has a print limit of 100, so printed reports are only
a sample. This is a bounded reproduction, not a certified-shape replacement.

Source inspection explains why a new *class of host API report* becomes
visible: X31 allocates journal addresses with `torch.empty`, writes counted
events, then clones the entire capacity before rebase (whose loads themselves
are masked by counts). Replay also clones whole owner tensors whose dead
metadata lanes can remain undefined. The two masked kernel operand sites
are not an exhaustive inventory of these host copies. No waiver, blanket
zero-fill, disabled API checking, or third production repair is introduced.
The exact ownership/indexing fix and full gates remain future work.

## Scope and remaining work

The T2048 journal limits remain certification-harness constraints. B64/T2048
needs the part5 transposed launch adapter ported into the certification path;
no such port was made after the failed prerequisite gate. The direct unpacked
B128/T2048 X31 guard remains intact: its three journal arrays demand
51,732,807,680 bytes before state and replay clones. Production uses X36's
smaller reservation and the receipted unpacked B32 checkpoint microbatches;
this receipt does not overturn that production-envelope evidence.

The rest of the required battery was not run after failure: all-shape parity,
109 oracle tests, graph/eager, bit-matched 2M curve, full sanitizer sweep, and
racecheck at both scales. All 18 R-inventory groups remain unresolved; no
subtraction or benign-race claim is made. Race repairs and payoff evaluations
remain gated on acceptance of the combined masks.

Only /home/erikg/emender/.venv/bin/python was used, with sequential GPU work
leased on physical card 2 from allowlist 2/3/4. Searches on 0/1/5/6 and forbidden
card 7 were not used. No WG service/dispatcher/workers, Slurm, resilient
training, parallel agents or committed .pt files. Existing unrelated workspace
changes were left alone. `contact_supervisor` was searched first and again
by name/description; no callable supervisor/intercom tool exists here.

The reduced diagnostic reached its 180-second bound and returned 124. It retains
100 printed host API errors and no terminal sanitizer summary or completed
parity result. Its compressed log is lossless; `summary.json` gives the raw
hash and sampled Python frames. This independently retained failure evidence
does not recover the lost certified-shape log. `run.sh` now defines its body as
a function before invoking it, avoiding execution from shifting body offsets;
that receipt-runner correction is syntax-checked but was not GPU-retested.
