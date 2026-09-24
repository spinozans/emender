# E97 private-analysis 5e-5 LR screen

Status: q8 and u64 numerical/reload qualification passed.
u64 saved/x failed the retention floor; advancement to u128 is stopped.
Train/y retained both legacy suites. Fresh-development acquisition is not yet established.
Date: 2026-09-09. Local/uncommitted; no publication or promotion claim.

## Decision and scope

The operator authorized one bounded private-analysis run at **5e-5**, restarting
from trusted parent train/y `aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`
with a fresh Schedule-Free optimizer. This is 25x the previous 2e-6 rate.
The matched action-only control is deferred. The earlier roughly 515M-target
5e-5 collapse is a risk observation, not a prediction about this 50M screen.

Both representations of the prior private checkpoint failed the diagnostic
analysis HTTP request before dispatch. CUDA tokenwise replay and the bounded
serialization audits passed; see
[e97-analysis-runtime-candidate-qualification.md](e97-analysis-runtime-candidate-qualification.md).
This new experiment screens learning-rate responsiveness. It does not declare
the full roadmap's Phase 0/1 gates satisfied or choose a representation.

## Frozen inputs and policy

Artifact root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/private-analysis-lr5e5-screen-v1`.

- Recipe: `configs/pi/e97-private-analysis-lr5e5-screen-v1.json`, copied to
  `recipe.json`; SHA-256 `b8244b2e8458dfa230f71094d633a4761eee41a6384f12b2acac0f1c1f3183bc`.
- Freeze receipt: `freeze.json`; SHA-256
  `777083cc9b8c886838de94db817f2e77845e0dba67ad55b55c49dab0bfd57da5`.
- Training source: unchanged read-only source archive
  `57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4`.
  The interval wrapper also compares archived files with the executing tree.
- Private authority: `08c16e0fa08b960ada87e197ba94f1a842dd0062691aaf28911f1df6bc552247`.
- Boundary packs: `22a69dd252773cf13c8803e191ae9f88d1d87cadaa51e10850095c47254c47ea`.
- Same epoch-permutation order, key 974121, world 8, context 65,536, K8,
  full-world DDP without redundant outer averaging. Same warmup8, betas,
  weight decay, clipping, checkpoint grouping and allocator settings.
- Maximum update 240: expected **50,016,738 consumed assistant targets** and
  **105,894,539 consumed input tokens** if completed. Not actual consumption yet.

q8 is numerical/mmap-reload qualification only. Clean exact resumes stop at
updates **64, 128, 240**. Each behavioral milestone requires full 120-task core
and 240-task compositional retention plus the small development panel, in both
saved/x and train/y. Before advancing, **each** mode must meet **116/120** and
**228/240**. Missing evidence, harness failure, identity failure, numerical
failure or a floor violation stops advancement. No automatic failed-epoch retry.
No midrun prompt, LR or mixture adaptation. No acquisition floor before terminal.

## Development panel: deliberately narrow

`dev-panel.json`, SHA-256
`a29094fa9934d83c7f89ce6f4f4cf93fbaa9982aa295bec78684616896d0cb90`,
contains four tasks each in six familiar families: exact instructions, supplied
document extraction, supplied-document arithmetic, one-file reading, two-file
reading, and recovery after an authentic missing-file read. Read-only fresh
fixtures; no shell or other tool execution. The development runner uses the
numerically qualified dense tokenwise engine directly with exact analysis
history serialization, 4,096 output tokens, a 2,048-token analysis limit and at
most six turns. Invalid framing is a scored model failure, not a harness pass.
Identical read repetition is no-progress here because fixtures never change.

This is **fresh-instance development**, not unseen-family/repository evaluation,
HTTP/Pi qualification or an independent final holdout. Exact-marker overlap
checks passed against the sealed private training token payload and consumed
core/compositional/V3/V4 records, with aggregate-only evidence. The audit is not
semantic or family non-overlap. No protected fixture was transformed into a task.
The broader capability panel and independent final holdout remain unfinished.

Novelty receipt SHA-256:
`cf2fa10a6a39b286c766213ea031848bdfc70ea625ce1d316b218e23795a4a11`.
Evaluation source inventory SHA-256:
`c2a4bd795708883adfe27ac9af5eab3b93f4bd0372297eaf3c732d5ff5dc2253`.

Fixed baselines before first candidate development scoring: parent train/y,
prior private 2e-6 saved/x and train/y. Report all milestones/modes. A screening
signal requires at least 12/24 successes, 4/12 tool-task successes, 22/24 valid
first-turn analysis frames and a net gain of at least four successes over the
same-panel parent. This is a screening threshold, not statistical proof or
promotion authority; a signal motivates a separately authorized matched control.

## Validation and execution

- 20 local tests passed: panel/scoring/read confinement, scripted fixture replay,
  exact reasoning history, no-progress, complete/mixed shard handling and
  parent/numerical/behavior-gate fail-closed checks.
- The isolated evaluation snapshot passed its 16 panel/evaluator/aggregator tests.
- A fresh explicit parent train/y legacy-protocol preflight passed **120/120**
  under accepted legacy runtime archive `803f0b790201242cfcc9a02991ffe4f8c4f18f82112c4cc3685f7f63142ba1a9`.
  Run: `evals/e97-lr5e5-screen-parent-train-core-preflight-v1`.
- Managed process `proc_a5bf` completed `run-training.sh q8` in 652 seconds.
  Reload passed for checkpoint `3f1cd3f103a9d60d856e07190a4682be3aabc1431cc5a74c03d264cec4fb4516`:
  8 updates, 806,195 assistant targets, 3,054,944 inputs, reported terminal loss
  1.2513. This is numerical qualification, not behavioral selection.
- The operator then requested investigation of BF16 update resolution. The run
  paused at a clean checkpoint for read-only CPU diagnostics; see
  [BF16 investigation](e97-bf16-update-diagnostics-v1.md). Those audits completed,
  and `proc_2c72` completed clean exact resume to u64 in 3,306 seconds.
  Checkpoint SHA-256:
  `12b2140ff4d299c219af0981ab978ccf2c3135984567941ca244fb57544521b1`.
  Reload passed: **17,984,082 targets / 30,561,243 input tokens**, 64 updates,
  reported terminal loss 1.1434. Loss is not a behavioral gate.
- Managed process `proc_2952` completed all four legacy-retention evaluations,
  then failed during the first development baseline because all eight workers
  selected CUDA device 0. This is a harness failure, not a development score. Sequence wrapper `run-u64-evaluations.sh` SHA-256:
  `45d2bfc7cc37d8dbba8b5d8f07e80ca7ca859cf8dba2ec43fb75249ae6db928f`.
  No u128 launch is authorized by numerical success alone.
  No optimizer or frozen recipe change has been applied. Training source, earlier authorities/checkpoints,
  canonical source registry and authorization allowlists remain unchanged.

## u64 retention and corrected interpretation

| Mode | Core | Compositional |
|---|---:|---:|
| saved/x | 11/120 | 0/240 |
| train/y | **120/120** | **236/240** |

These are results for different actual weight representations of the same
checkpoint. Train/y matches the trusted parent's reported retention scores;
calling this wholesale forgetting or destruction of the trained model was too
broad. Conversely, familiar-task retention is not new capability acquisition.

Schedule-Free maintains a base iterate `z` and live training point `y`, with an
implicit averaged point `x`. In exact arithmetic with beta1=0.9,
`y = 0.9*x + 0.1*z`. Checkpointing calls `optimizer.eval()` to export
`x = (y - 0.1*z)/0.9`; it is not a stale backup or a different run. Legacy
BF16 conversion is lossy, so the evaluated reconstructed train/y is not proven
bit-identical to the resident y immediately before export. Averaging dynamics
and conversion precision remain distinct hypotheses for the behavioral gap.

The predeclared gate required **both** modes to pass. Stop receipt `gates/u64.json`
SHA-256 `050278344760f9b35cf7037b5419861743e317263e890ea3471b7cd04c254055`
therefore remains in force. Stopping advancement does not discard this checkpoint
or establish that its train/y representation is incapable of learning.

## Development harness routing repair

The first attempt failed before completing the parent baseline: missing explicit
CUDA device selection sent all eight torchrun workers to GPU 0, causing OOM and
subsequent cuBLAS failures. NUMA placement alone does not select a CUDA device.
Original failed outputs and the original evaluation snapshot are preserved.

`scripts/eval_e97_lr_screen.py` now validates `LOCAL_RANK`, calls
`torch.cuda.set_device`, and supplies the explicit device to loader and engine.
Ten mocked routing tests plus eight evaluator/aggregator regressions passed
(**18 total**). This changes hardware routing only, not tasks, prompts, budgets,
checkpoint identities or scoring. Corrected read-only snapshot:
`eval-gpu-routing-v2/worktree`; source inventory SHA-256
`9590118f343c5db93e1a386333bf5b62fff9a77e62c8541970a89da1b7db0aff`.

Managed process `proc_b122` runs the fixed three baselines, then candidate
**train/y before saved/x**, prioritizing the operator's question. All eight
rank-to-device assignments (`cuda:0` through `cuda:7`) were observed explicitly.
The corrected **parent train/y baseline** completed: **0/24** successes,
**0/24** valid first-turn analysis frames, no admitted read calls, and 98,304
completion tokens. All 24 failed with `analysis turn requires a canonical
Analysis JSON string`. This measures failure under the new analysis protocol;
it is not a claim that the legacy parent cannot use tools in its trained format.
Aggregate `evals/e97-lr-screen-dev-parent-train-v2/summary.json` SHA-256:
`2b9b33c0f28e060b97300a30d7a12fb407f27b2e465ae6a23aaea19bf57eecf4`.
The prior **2e-6 private-analysis saved/x** control also completed: **0/24**
successes, **0/24** valid first-turn analysis frames, no admitted read calls,
96,553 completion tokens. All 24 failed the same canonical Analysis-frame
requirement. Aggregate `evals/e97-lr-screen-dev-prior-saved-v2/summary.json`
SHA-256: `9e0f5e39d3f6bfccf12bdf182529648147a5e9d1556187b723f6a682ca5bac8c`.
The prior **2e-6 private-analysis train/y** control completed: **0/24**
successes, **0/24** valid first-turn analysis frames, no admitted read calls,
98,304 completion tokens. All 24 failed the canonical Analysis-frame requirement.
Aggregate `evals/e97-lr-screen-dev-prior-train-v2/summary.json` SHA-256:
`e3e505e8787721c59b9a34ba626abb81811c58c2ac3458b1e21fa38f8c57f714`.
All three fixed development baselines are complete. The **5e-5 u64 train/y
candidate** then completed: **0/24** successes, **0/24** valid first-turn analysis
frames, **0/12** tool-task successes, no admitted read calls, 98,304 completion
tokens. All 24 failed the canonical Analysis-frame requirement. The generated
prefixes were legacy `Action: read` calls, often to unrelated or modified paths;
even no-tool instruction tasks generated reads. No observation-dependent
multi-turn rollout was reached, so this result does not isolate an inability to
reason after receiving observations: protocol admission already failed.

Aggregate `evals/e97-lr-screen-dev-u64-train-v2/summary.json` SHA-256:
`4571755f1189f4f96f0bf23de68c7525ba0f9204dce4b5cd40ee68e102b0d61b`.
Thus train/y retained familiar legacy workflows but **demonstrated no new
analysis-protocol capability on this panel**; there was no gain over the parent.
This is a bounded development finding, not proof of zero learning everywhere
or a diagnosis of its numerical/objective cause.

The **5e-5 u64 saved/x candidate** also completed: **0/24** successes,
**0/24** valid first-turn analysis frames, **0/12** tool-task successes, no
admitted read calls, 98,304 completion tokens. Its outputs were predominantly
repetitive descriptive prose about a read tool rather than actual protocol
turns; one began with a noncanonical `<think>` block. These did not reach real
tool observations either. Aggregate
`evals/e97-lr-screen-dev-u64-saved-v2/summary.json` SHA-256:
`64f2972067e69ddf8760f973a172bfe37cb330dd4fc14a2179deb3000fe0d800`.

`proc_b122` completed successfully in 2,118 seconds. This denotes successful
execution of all five development evaluations, **not** successful model
behavior. Final paired result:

| u64 mode | Core | Compositional | New development | Valid analysis frames |
|---|---:|---:|---:|---:|
| train/y | 120/120 | 236/240 | 0/24 | 0/24 |
| saved/x | 11/120 | 0/240 | 0/24 | 0/24 |

All modes and fixed baselines were reported. No new capability was demonstrated
on the 24-instance analysis-protocol development panel. Train/y retention and
saved/x failure are real, separate observations. No continuation, promotion,
parser/prompt relaxation or new training followed. Next diagnostic priority is
correct-target likelihood on explicitly consumed training prefixes versus fresh
prompts, separating a failure to fit from failure to generalize; this has not
yet been measured. The precision candidate remains CPU-qualified only, with
CUDA qualification deferred during this behavioral investigation.

Architecture authority: ADR-003 in
[RESILIENT_DILOCO_COMPUTE_POOL](../RESILIENT_DILOCO_COMPUTE_POOL.md).
Applicable safety intent: **R07/R12** committed atomic save/reload and exact clean
resume; **R14/NDP13** bounded whole-job termination; **R16** immutable evidence;
**NDP15** checkpoint atomicity only. This is local fixed-world K8 SFT, not a
Frontier scale rung. Elastic R02–R06/R08–R11, native no-all-rank/background clauses,
V21S01–V21S17 and ISP01–ISP07 are explicitly unclaimed. Planned milestone pauses
are clean resumes, never communicator shrink or automatic failure recovery.
