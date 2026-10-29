# X28E censored-arm adjudication

The five-corner certification remains **incomplete**. All three censored arms
hit the registered disk guard, not a training gate or demonstrated model fault.
The frozen launcher provides neither the requested evaluation subcommand nor a
resume path. No GPU was acquired, no process was signalled, and no frozen code
or pre-existing receipt was modified by this adjudication.

| Corner | Slots | ID | 2x | 4x | 9x | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 736 | 0.994140625 | 0.98046875 | 0.984375 | 0.5 | Existing final evaluation |
| B0x4 | 2944 | 0.990234375 | 0.98046875 | 0.970703125 | 0.947265625 | Existing final evaluation |
| B0x16 | 11776 | — | — | — | — | Training reached stage3 u30000; selected u9000 validation 0.98; final evaluation unavailable |
| unbounded | Unbounded | — | — | — | — | Stage2 u1500; infrastructure right-censored; no supported resume |
| appendonly | Unbounded | — | — | — | — | Stage2 u1500; infrastructure right-censored; no supported resume |

Numbers are balanced accuracy from existing final evaluations, not new runs.
Each corner JSON carries a canonical payload hash, source file SHA-256s,
validated source receipt payloads and launch/freeze identity checks.
`phase_diagram.json` binds the five corner files and the watcher/log evidence.
Weights are hash-bound only; no `.pt` files are included.

## Exact censor condition

`launcher.py:632` checks
`size > RUN_BYTES or wall > WALL_SECONDS or free < FREE_BYTES`.
All three guard receipts record `free_bytes=428719493120`, below
`FREE_BYTES=429496729600` (400 GiB), a deficit of 777236480 bytes.
Their run sizes are approximately 1.63 MB, below 40 GiB; their elapsed times
are approximately 309467 seconds, below 604800 seconds. Thus only the shared
disk clause fired. Guard decisions began at 1793286647.7235126,
1793286647.8357046 and 1793286647.9470236 respectively. The poll at
1793286708.0610473 classified all three as censored together. This timestamp
is the observation time, not simultaneous independent training failure.
`arm_state` at lines 210–220 accepts a hash-valid, arm- and launch-identity-bound
guard as infrastructure censoring. Source payload hashes validate for all three.
TERM and independent 30-second KILL escalation are recorded in their guards.

## Evaluation and resume failures

`eval_entrypoint_failure.json` records the actual CLI attempt against the selected
u9000 weights and the exact parser response: `unrecognized arguments`, exit 2.
This is **unsupported CLI**, not an X30-style gate signal and not an evaluation
result. `launcher.main` exposes only `--launch`, `--once`, `--terminal`, and
`--exit-code`. No selected-checkpoint evaluation subcommand exists.
The trainer exposes no eval-only or resume CLI either. Its private
`_final_evaluation` loads `best.pt`, but is reached through `run()` after training;
this task did not introduce an alternate wrapper around that private method.
The u30000 state was not evaluated or substituted for the selected u9000 state.

`--launch` only spawns queued arms; validated guard receipts keep these arms
censored. Deleting guards or restarting full training would not constitute the
requested frozen resume path. No such bypass was performed. The slow arms'
stage2 u1500 trajectories are **not proven impossible**: F12h uses occupancy
growth, not ordered-emission accuracy, for stage2 passage. Disk censoring
cannot establish a scientific boundary.

## Validation and limitations

Run `collect.py` with `/home/erikg/emender/.venv/bin/python`. It uses only the
standard library, validates source JSON hashes and identities, checks the exact
guard inequalities, captures the rejected CLI attempt, and writes the new
receipts here. No training tests were added or run. Original source files and
the temporary launcher log remain externally located and are hash-bound with
the relevant payloads or log/poll tails embedded for review.

The requested `contact_supervisor` capability was searched for and is absent
from the available tools. No supervisor decision was obtained. Evaluation and
resumption remain blocked by the frozen interface contract; completion must
not be represented as a certified five-corner result.
