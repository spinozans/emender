# X40 part 23: fresh race failures; repair incomplete

FAIL / incomplete. No race repair, accepted batch, battery-green result, 9x
acceptance, depth probe or final depth verdict is claimed. Production is
unchanged, including the promoted uninitialized-access fixes. This is a
diagnostic receipt, not the requested completed implementation.

## Baseline reconciliation

The requested “18 groups from the original 35” is not an established set of
18 surviving R IDs. Part17 reports 18 displayed groups in a rejected
process-local mask-plus-11-barriers diagnostic. Its inventory conservatively
leaves R01–R35 unresolved and explicitly forbids subtracting group counts.
Part18 rejects blanket PTX barriers, duplicate-writer predication and expanded
masked stores. Part20 rejects its uninitialized-access promotion attempt;
part22 subsequently promotes the three mask sets after its required battery.
Part22's post-promotion racecheck displays 31 groups before an assertion, not
a completed 18-group sweep. These facts are taken from the retained receipts;
no rejected repair was reinstalled.

## Fresh evidence

Both fresh runs use production source with a capture-only X40 launch wrapper.
It warms up the exact launch, retains PTX/cubin/TTIR/TTGIR, then calls the
original launcher without changing source, assembly, arithmetic, assertions,
warp count or launch options. The isolated replay contains one X40 launch.

| Gate | Result |
| --- | --- |
| Captured B8/T2286/S2944 replay, racecheck | FAIL: exit 86, 30 displayed groups, int32-add assertion |
| Full B64/T512 certify harness, racecheck | FAIL: exit 86, five displayed groups, int32-add assertion inside X31 reference |
| Artifact/endpoint audit | PASS: artifact hashes verified; all 175 isolated endpoints map to LDS/STS instructions in the exact cubin |

Both sanitizer invocations retain unlimited unsuppressed output and terminal
summaries. Neither application completes. Their counts are partial coverage,
not improvements or accepted zero passes. The certified harness fails at
parity.py:45 inside the unchanged X31 oracle before the X40 capture hook runs;
its five groups are not bound to the X40 cubin. No B32/T2048 racecheck pass is
claimed. The previous part22 initcheck battery remains previous evidence,
not a newly run post-repair battery.

Raw consoles are losslessly gzip-compressed after completion. Their original
byte hashes remain in console.sha256 and audit.json; sanitizer whitespace is
preserved verbatim in compressed evidence.

`audit.json` binds raw logs, all displayed endpoints, cubin and SASS hashes.
The fresh IDs are P23 specialization/run IDs. All historical R IDs remain
unresolved in the audit: no mapped subset of 18 can be claimed closed.
Per-endpoint ownership and lifetime are explicitly unproved. Matching a PC
to LDS/STS does not prove which dynamic phase or lane owns its shared bytes.

## Unfinished work and risks

No minimal value-neutral scratch repair has been established. The outstanding
work includes dynamic scratch ownership/lifetime proofs, the actual per-group
repairs and incremental paired-shape sanitizer/parity gates. All subsequent
requested gates remain gated and not run: full battery, 100-repeat 9x,
hash-bound budget-scaled prediction and evaluation, post-fix pressure connection
and final depth verdict. The X31 reference failure also prevents using the
full certify harness as a completed X40 racecheck result. These failures do
not establish one common cause for the depth and pressure symptoms.

Execution was single-threaded, using only the specified venv interpreter and
sequential broker leases on physical card 2 from allowlist 2/3/4. No service,
WG dispatcher/worker, Slurm, resilient training, agent or checkpoint commit was
used. AGENTS.md and CLAUDE.md were unchanged. Unrelated workspace edits remain
untouched. Supervisor/intercom tool names and descriptions were searched at
entry; no callable contact_supervisor exists in this runtime.
