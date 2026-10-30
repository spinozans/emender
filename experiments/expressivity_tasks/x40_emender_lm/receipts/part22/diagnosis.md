# X40 part 22: three uninitialized-access mask sets promoted

The required production battery is green at **B64/T512** and the explicitly
accepted **B32/T2048 unpacked production envelope**, for all four policies.
The T2048 certification is complete, not pending. The conditional pending-T2048
promotion exception was not used. All three part21 candidate sets are installed
in production together. **9x remains blocked**: its 100 recorded reproduction
attempts fail stability/liveness checks, and racecheck still reports real races.
Those post-promotion defects remain scoped to subsequent race work by the
operator's part22 work order, steps 3–4.

## Reference memory repair and certification scope

`parity.py` retains complete comparison records on CPU and isolates each policy
in a function scope, releasing temporary references before the next policy.
The seed is still set once; reference operations, policy ordering, input
construction and comparisons are preserved. The entire 2048-token reference
and production scan execute under initcheck in the same process. There is no
sanitizer-free reference pass, temporal tiling, numerical approximation,
changed reduction tree, or weakened journal guard.

The authorized long-context target is B32/T2048, matching the production
unpacked envelope, rather than the former B64/T2048 harness shape. We do not
claim a B64/T2048 pass or an established root cause for every retained allocation
in its part21 OOM. Final T2048 PyTorch peaks are 43,099,527,680 allocated bytes
and 43,201,331,200 reserved bytes. The existing transposed X31 reference rebase
adapter is validated and used, exactly as in part21: small mapping bit-match
and the 65,537-block launch both pass. It is a harness adapter, not another
production arithmetic change.

B128/T2048 remains **FAIL / explicitly excluded** at the unchanged X31 checked
int32 capacity guard. The new host preflight exits 1 before GPU allocation;
it is not a sanitizer pass. `source_audit.json` verifies that `capacity_for`'s
AST is identical to baseline commit 63ba1c3b. No larger shape is silently
substituted for a claimed certification.

## Installed production change

- X40 ReadStrengthCache loads require both a live row and an older touch;
  the existing reference/cache selection and arithmetic are unchanged.
- X31 and X40 materialization load Birth only for valid live rows.
- X31/X36/X40 initialize exactly due, terminal and births, whose whole-owner
  replay/freeze copies must preserve historically live rows. X31 backward's
  address clone and journal snapshot address/before/width copies use counted
  prefix copies with defined zero tails. The transposed copy launch supports
  more than 65,535 log blocks.

The counted-copy implementation matches part21's final candidate verbatim.
`promotion_sources.json`, `source_audit.json` and per-run source hashes bind the
installed files to the tested code. No barrier, suppression, API-check disabling,
integer counter widening, mathematical reduction change or race repair is
included. AGENTS.md and CLAUDE.md are unchanged.

## Completed required battery

| Gate | Result |
| --- | --- |
| Full B64/T512 certification, four policies | PASS, initcheck exit 0, terminal 0 errors |
| Full B32/T2048 unpacked certification, four policies | PASS, initcheck exit 0, terminal 0 errors |
| X31 oracle suite | 109 passed |
| Additional X36 regression suite | 70 passed; final run 304.47 seconds |
| Counted copies | Seven cases, including 65,537 blocks; initcheck exit 0, terminal 0 errors |
| Real snapshot/parent/pressure-expiry fixture | One passed; initcheck exit 0, terminal 0 errors |
| Native-width forward lifecycle | 32 fixtures passed |
| Retained-gradient initialization neutrality | All 48 configurations, three variants, passed |
| Graph versus eager | Two distinct B64/T512 batches; losses and all parameter gradients match |
| Paired real-data curve | 64 updates / 2,097,152 tokens, every row exactly matches |
| Captured B8/T2286/S2944 scan | Application PASSED; initcheck exit 0, terminal 0 errors |

Certification preserves the standing exact (`rtol=0, atol=0`) X31 comparisons
for outputs, final keys/values/strength, gradients, routing traces, live/age and
pressure, including independently computed beta. The pre-existing evidence/error
telemetry tolerances remain rtol 2e-5 / atol 2e-6; gate decisions remain exact.
No blanket bitwise claim is made for that independent floating-point telemetry.

The 48 retained-gradient configurations cover X31, X36 and X40, all policies,
fresh/imported origins and ordinary/unit decay, each with zero, finite and NaN
terminal initialization variants through 1/2/31/1/35 consuming chunks. Defined
state, metadata, traces, history, pressure and gradients are compared as bytes;
X40 evidence/error snapshots are included. Every configuration observes POP,
freeing, allocation and reuse. This is the existing part21 4/8-dimensional
retained fixture, now completed. Native 64/128 dimensions are covered by full
certification, the 32 forward lifecycle fixtures and the oracle suites; this
is not a separate full native-width sentinel-gradient matrix claim.

The curve is the FP32 six-block pilot LM on TinyStories, B64/T512 with two B32
accumulation envelopes, checkpointing and packed journals. `curve_base.json`
records original production sources before installing masks; `curve_promoted.json`
matches its initial parameter digest and every loss, norm, gradient byte SHA256
and post-step parameter byte SHA256. The final parameter digest is
`68d45aa6143950258ad66bf4019e93dbea2cf949861d125e9e2a373a90bee89f`.
This is an exact paired pilot slice, not a new 1.3B or depth-quality claim.

All initcheck invocations use unsuppressed `--print-limit 0` and error exit 86.
Acceptance requires application completion, exit 0 and terminal zero errors;
a tool-only zero or partial application run is never counted as a pass.

## Failed installation attempts and complete rollbacks

All three earlier failed installations were reverted before correcting their
harnesses and installing the identical production change again. Their unique
run directories, original source versions and rollback receipts are retained.

1. `production_x36_oracles.8gR0S9LW`: exit 124 at the 600-second harness bound,
   with 45 passing indicators but no completed suite. The bound was extended
   to 1800 seconds with verbose node reporting; subsequent full 70-test runs
   pass. This first attempt remains a failure, not a 70-test pass.
2. `production_backward_lifecycle.plQFhxTU`: exit 1 after all 32 X31/X36 rows.
   The previously unreached X40 fixture called its forward without mandatory
   wire logits/coefficient/evidence arguments. The fixture now supplies the
   same partial wire contract as model.scan and retains evidence/errors.
3. `production_backward_lifecycle_x40_preflight.x55E1MPe`: exit 1 because the
   repaired fixture shared beta views across chunks while the wire writes beta.
   Saved version 4 versus current version 5 was correctly rejected by autograd.
   Beta is now independently cloned per chunk, exactly as in model.scan.

The corrected X40 target is front-loaded and passes all 16 configurations;
then the final installation completes the entire required battery, including
all 48 retained-gradient configurations. No production API or numerical repair
was added to hide either fixture error. No running runner was edited. The
seven-file rollback plus removal of the new helper is recorded for each failure.

## Post-promotion 9x: FAIL, remains blocked on race work

There are **100 terminal clean-process attempt records**, each with a fresh CUDA
context and the same immutable checkpoint/child source. The initial 1200-second
cohort bound interrupted an additional attempt after 95 terminal records; that
interruption is retained separately and is not counted among the 100. The last
five terminal records were completed with a 60-second bound, preserving all
prior records and failures. `nine_x.json` and `nine_x_analysis.json` bind every
terminal console by SHA256.

| Outcome | Recorded attempts |
| --- | ---: |
| Known int32-add device assertion | 39 |
| Nonfinite logits | 2 |
| Finite application exits | 56 |
| 300-second attempt timeout | 1 |
| Attended termination of stalled own children | 2 |

The 56 finite exits produce **47 distinct logit byte hashes**; only one matches
the first finite result. That reference is a repeat-consistency check, not an
accuracy oracle. All 97 available terminal metadata footers report identical
checkpoint and input hashes; the three stalled attempts have no terminal
metadata footer. One stalled child was terminated after 143.29 seconds and
another after 114.54 seconds, with PID, parent and start-time identity checks
retained in `attended_terminations.json`. No other process was stopped.

The external content-addressed checkpoint is retained and hash-bound in
`external_checkpoint.json`; no weights or .pt files are committed. These
assertions, nonfinite outputs, hangs and byte variations are failures. They
coexist with the unresolved race inventory; no proof that a particular race
explains every observed failure is claimed. The 9x bin, payoff/depth claims and
final depth verdict remain blocked.

## Race work begun; no race-free claim

The fresh isolated racecheck exits **86**, prints **31 race-error groups**, and
then aborts with the known device assertion. This is a partial sweep, not a
completed zero pass or a completed 18-group sweep. `race_inventory.json` retains
all displayed endpoints as P22-G01–G31, the raw hash and terminal summary.

The operator's 18 historical tracked groups remain unresolved. New displayed
group counts are specialization-specific and are not a mapping or subtraction
against those IDs. No group is declared benign, repaired or closed. Next work
must bind each PC to its exact compiled specialization and shared-scratch
owner/lifetime before proving a value-neutral repair with unchanged reductions.

The scoping authority is explicit in the part22 work order, step 3:
“racecheck at the known 18 groups, receipted as the next work item, not a
promotion blocker,” followed by step 4's expected post-promotion 9x race
failures. It is also retained in [part21 diagnosis](../part21/diagnosis.md).
These scoped post-promotion failures are not required-gate waivers. All required
uninitialized-access promotion gates completed before this diagnostic work.

## Execution and evidence

Only `/home/erikg/emender/.venv/bin/python` was used. GPU work was sequential and
broker-leased from allowlist 2/3/4; searches on 0/1/5/6 and forbidden card 7 were
not disturbed. No WG service, dispatcher, workers, agents, Slurm or resilient
training was used. Unrelated workspace changes were left untouched.

contact_supervisor was searched first by tool name and then by name/description;
no callable supervisor/intercom tool exists in this runtime. No substitute
messaging channel was used. `summary.json` audits terminal results fail closed;
`manifest.json` binds the retained artifacts, including failed raw evidence.

The final source whitespace review reports one cosmetic extra EOF blank line in
`counted_copy.py:20`. It is retained with the exact tested source bytes and
reported in `final_review.json`; no runtime/API/sanitizer check is disabled.
No other source whitespace finding is reported.
