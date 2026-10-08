# F7 staged correction pipeline (OFF until an operator relaunch)

Live B source/config/leases must not be changed by qualification. Matching
NVMe bytes are in `e97-rl-loop-v1/F7-staged/scripts`, NOT the live `scripts`.
The endpoint probe imports an isolated byte-identical pre-F7 pilot and does not
use the deadline adapter. Nothing here authorizes a live relaunch or model flip.

At the approved relaunch, deploy the staged files and create the bank's durable
`correction-config.json` atomically, for example:

```json
{"teacher_model":"deepseek-4.1-flash-background","async_enabled":false,"pool_width":1}
```

This is an example, not a deployment verdict. Consult F7-report.md for the
qualified width/model and quality criterion. Missing file preserves the actual
live default `glm-5.3-flash-background`, async OFF. Model overrides are allowed
by CLI; otherwise the daemon rereads the durable file each cycle. The correction
model does not change the writing judge, which stays at the live flash baseline.
Do not change lane teacher-min-interval (10s); it is still an admission gate.

Pool width is PER LANE, not a global quota. The default is one worker per lane:
eight B lanes offer at most eight corrections, reserving four of the operator-
stated twelve global background slots for other callers. Four-task cycles cannot
occupy more than four slots even if a larger width is configured.
The 12-lane cap is operator-stated, not empirically isolated: B and other
traffic share the endpoint. A proposed width must account for ALL callers;
changing max-tasks or number of lanes requires requalification. At capacity,
the task gets an explicit failed correction outcome and no receipt, not an
unbounded queued job or a silently dropped correction. Round refresh retains
the same existing failed-task retry semantics. Failures still have PG attempts;
no PG/SFT arithmetic or training adoption/exposure rules change.

Each admitted spec is immutable and contains task/attempt/grade/checkpoint
identity, workspace ownership and fresh/spliced routing. Each worker owns its
own generator and Pi process; only the collect owner builds/appends receipts at
the current stream tail. Completed corrections may publish in completion order
(with admission-order ties), while summary outcomes stay in attempt order.
All pending corrections finalize before the cycle summary/training boundary;
there is NO cross-cycle or checkpoint-mutating overlap. A cycle close fences
publication before killing its workers. Linux parent-death fencing and a worker
hard wall-clock alarm contain descendants even if collect crashes or is killed.
A crashed collect emits no final summary; existing claim expiry/retry handles it.
The 750s worker bound includes imports, the 480s teacher episode, grading and
cleanup. The deadline adapter further bounds HTTP socket waits/retry sleeps to
the episode deadline. It is a separate deploy-at-relaunch change, not measured
by the unmodified pilot probe. Sustained mixed-workload targets/h and receipt
ordering/crash behavior require an OFF-to-ON controlled rollout measurement.
