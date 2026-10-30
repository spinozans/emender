# X40 part 21: backward clones diagnosed; promotion remains rejected

Honest partial / no production promotion. The combined process-local candidate
passes the full B64/T512 certification harness under initcheck, including all
four policies, strict bitwise values/state/traces/pressure/gradients against
X31. It also passes the captured 9x scan initcheck. B64/T2048 fails with CUDA
OOM; B128/T2048 still fails its unchanged X31 journal-capacity preflight. Neither
failure is waived. Both shape families have NOT returned complete zero passes.

## Backward population and candidate

Read commit 4051a48f part20 diagnosis and summary first. `clone_inventory.json`
clusters all 100 retained reports into four source allocations: 35 reports in
three owner buffers (due, terminal, births) and 65 in the journal-address tail.
The two equally sized integer-owner names are inferred from source allocation
order, not measured owner labels in that old log. The original raw SHA256 is
verified; the sample is not an exhaustive error population.

`candidate.py` applies the two part20 forward masks and a backward repair only
inside its process. It initializes exactly due/terminal/births in X31, X36 and
X40 forward allocation. Whole-owner copying in backward, retained-state freeze,
and X36 replay_setup.prepare then has defined sources. Final-live masking would
be insufficient because reverse undo restores historically live rows.

The X31 backward address clone becomes a counted masked copy with zero tails.
The explicit AllFieldJournal.snapshot address/before/width copies use the same
helper. Source loads are masked by Counts; rebase and undo already consume only
that prefix. The helper uses transposed grid axes to support >65535 log blocks.
The existing part5 transposed X31 rebase adapter is validated and used for the
T2048 certification attempt. No numerical operation, tolerance, journal guard,
barrier, race suppression or API-check disabling is changed. See
`clone_analysis.md` for site and ownership reasoning.

## Completed validation

- The original combined mask lifecycle test passes 32 fixtures (16 X31, 16 X40).
- A separate 32-fixture extension varies all three metadata initializations
  between finite sentinels, a NaN terminal sentinel, and candidate zeros. Every
  policy, fresh/imported origin, ordinary/unit decay, and real continuation
  through 1/2/31/1/35 chunks passes byte comparisons of defined public state,
  trace, pressure, history and X40 evidence/errors. Every configuration observes
  POP deaths, frees, allocation and reuse. These are forward/state neutrality
  proofs, not completion of the retained-gradient matrix.
- The counted-copy test passes initcheck with zero errors for counts 0/1/full,
  uint64/int32, non-block-aligned capacity, and a 65537-block launch.
- The real journal snapshot/parent/pressure-expiry test passes initcheck with
  zero errors, including the snapshot's three counted fields.
- The B64/T512 full harness passes all four policies under unsuppressed
  initcheck (`--print-limit 0`, exit 0, terminal ERROR SUMMARY: 0 errors).
- The original captured B8/T2286/S2944 scan-10 input passes initcheck, exit 0,
  application PASSED and terminal zero errors. Its external .pt SHA256 is
  recorded. No .pt file is committed.
- Both the first and final candidates pass all 109 oracle tests in independent
  terminal runs.
- Both runner regression tests pass, including reruns of the same failed name
  preserving the original log/hash and timeout 124 preserving partial output.

## Failed and incomplete gates

The B64/T2048 full harness exits 1 with CUDA OOM in parity.py:35, while forming
`one_hot(best, cfg.slots) * fired[..., None]` in the unchanged reference. The
physical leased card is 2; PyTorch calls it logical GPU 0. The device has 47.38
GiB total and the process has 47.37 GiB in use (41.53 GiB PyTorch allocated).
The tool prints zero errors on application exit, but certification never
completes: this is a FAIL, not a zero-error acceptance. No memory-retention
cause is established, no smaller substitute shape is used, and no complete
T2048 uninit inventory is claimed.

The independent B128/T2048/S1472 preflight exits 1 at the unchanged checked
int32 reservation guard. It allocates nothing and is not a GPU sanitizer run.
The original all-shape contract remains incomplete; B128/T512 was not rerun in
part21 after these blockers. Graph/eager and the bit-matched 2M curve were not
run. No forward-only or combined production promotion is claimed.

The retained-gradient lifecycle matrix is also incomplete. The original
64/128-dimensional run was deliberately stopped (143) after nine passing X31
rows to reduce compilation cost. A 4/8-dimensional version retained the same
70-token continuation and coverage assertions, but was deliberately stopped
(143) after the B128 preflight established the all-shape blocker, to prioritize
full-harness diagnostics. Its completed rows are recorded separately. Neither
stop is a timeout pass, and the 32-fixture metadata test does not replace the
missing X36/X40 retained-gradient coverage. The original large-dimension source
is retained separately; no running shell runner was edited.

An initial process-local loader mistakenly redefined FusedState while existing
modules held its prior class identity. That diagnostic exited 1 despite a tool
zero count. It is retained as a harness failure. The loader now recompiles only
tape_scan, preserving class identity; subsequent successful runs use that fix.

## Receipt handling and remaining work

`run.sh` parses its body before invocation, reserves a unique mktemp directory
per attempt, and appends all setup, tool and error output to its console. An
error or rerun cannot truncate another attempt's log. It retains the command,
source hashes, revision, lease, exit status and console hash. Timeout uses TERM
plus a 10-second kill bound. Reduced probes are bounded at 180 seconds; full
harness attempts at 600 seconds. The runner was edited only between invocations.
Older command metadata had trailing spaces normalized; raw console bytes and
console hashes are unchanged. All intentional regression failures are retained.

By explicit operator scoping, the 18 surviving race groups are outside the
uninitialized-access promotion blocker set. They remain unresolved defects;
there is no new racecheck sweep, benign-race finding, subtraction or race-free
claim. Race repair, the 100-repeat 9x bin, preregistered budget-scaled depth
probe, part-4 connection and final depth verdict remain pending acceptance.

Next gates: finish retained backward lifecycle coverage; resolve the T2048
harness memory failure and unchanged B128 reference envelope with explicit
certification evidence; complete the all-shape parity/sanitizer and graph/curve
battery. Production remains byte-identical to 4051a48f across all eleven relevant
source files, verified by summarize.py. The candidate is reviewable scaffolding,
not an installed production repair.

Only /home/erikg/emender/.venv/bin/python was used. GPU work was sequential and
broker-leased on card 2 from allowlist 2/3/4. Searches on 0/1/5/6 and forbidden
card 7 were not used. No WG service, workers, agents, Slurm or resilient training.
Unrelated workspace edits were left alone. contact_supervisor was searched first
and again by name/description; no callable supervisor/intercom tool exists in
this environment. Incremental commit 6285bd43 was pushed; final receipts follow
in an honest-partial commit rather than claiming a full promotion battery.
