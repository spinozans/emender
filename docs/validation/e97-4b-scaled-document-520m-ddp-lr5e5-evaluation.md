# E97 4B scaled-document DDP `5e-5` evaluation

## Verdict

The full-world DDP run was numerically successful but is **not an agent or
chatbot promotion candidate**. The saved Schedule-Free `x` checkpoint achieved
zero strict successes on all three terminal Pi panels and failed an informal
set of generic conversation probes. A matched smoke evaluation at update 64
shows that most of the previously selected Pi policy was displaced within the
first 12.4 million assistant-target tokens.

The terminal checkpoint remains a durable training artifact and a possible
input to controlled recovery research. Its low mixed-objective training loss
must not be presented as instruction-following or agent competence.

## Training artifact

- Run: `e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u2640-d89ea9f8`
- Topology: eight-rank full-world DDP; redundant outer DiLoCo merge disabled
- Learning rate: `5e-5`
- Updates: 2,640
- Input tokens: 1,144,364,830
- Assistant targets: 514,953,509
- Final 80-update mean loss: 1.5294
- Saved checkpoint loss field: 1.5402
- Checkpoint:
  `/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u2640-d89ea9f8/checkpoints/checkpoint_agent_sft_u002640_loss_1.5402.pt`
- SHA-256: `7d0e17afd6dde6b4846c2143580fc87d53f1e8a645bfb87acf1f3d2428e7cf37`
- Bytes: 24,276,129,019
- Mmap reload: passed

No OOM, nonfinite, traceback, or runtime failure occurred during training.
Global gradient clipping at 1.0 affected about 2.4% of updates and did not
explain the behavioral result.

## Terminal saved-`x` evaluation

| Panel | Tasks | Strict success | Arguments | Sequence | Completed | Grounded final | Protocol | No cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pi v2 smoke | 120 | 0 | 0 | 4 | 18 | 18 | 18 | 119 |
| Compositional v2 | 240 | 0 | 0 | 0 | 0 | 0 | 0 | 229 |
| Consumed V4 diagnostic | 240 | 0 | 0 | 0 | 0 | 0 | 0 | 151 |

Schema-valid-call and Pi-exit checks passed on every task in all three panels.
Those checks are necessary transport evidence, not evidence of useful behavior.
Sandbox postconditions passed on 60/120 smoke tasks and 40/240 tasks in each of
the other panels, including tasks whose initial state already satisfied the
postcondition.

Evaluation roots:

- `/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u2640-v2-smoke-d89ea9f8`
- `/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u2640-compositional-v2-d89ea9f8`
- `/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u2640-diagnostic-v4-d89ea9f8`

### Trace characterization

Observed executable calls collapsed toward `read` or no call:

| Panel | Recorded calls | Zero-call tasks | Recorded action distribution |
|---|---:|---:|---|
| Pi v2 smoke | 44 | 82 | 44 `read` |
| Compositional v2 | 70 | 191 | 70 `read` |
| Consumed V4 | 180 | 151 | 180 `read` |

The smoke panel had no exact expected first calls and touched an expected path
in 19 tasks. Compositional v2 had one exact expected first call, any expected
call in 38 tasks, and an expected path in 46 tasks. V4 had none of those
matches.

Many zero-call traces ended with HTTP 422 `agent_protocol_error`. Generated
text commonly began with `Action: read` and then emitted invalid or runaway JSON
with repeated `path`, `content`, or unrelated fields. This is policy and
serialization collapse, not merely selection of the wrong valid tool.

## Earliest matched aggressive milestone

The independently preserved update-64 checkpoint was evaluated on the same
120-task smoke authority:

- Checkpoint SHA-256:
  `4232ee3b60849855d324670d746028bfe4998ba5ca0d19a11883e4c8c7cc0abf`
- Assistant targets: 12,418,566
- Strict success: 5/120
- Arguments: 41/120
- Sequence: 61/120
- Completed: 80/120
- Grounded final: 44/120
- Protocol valid: 80/120
- No identical cycle: 80/120

The stable pre-epoch parent scored 112/120 on smoke. Therefore the destructive
interference was already present by update 64; a search over later aggressive
milestones is not required to locate its onset.

Evaluation root:

`/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/e97-4b-scaled-document-520m-epoch-ddp-lr5e5-u64-v2-smoke-d89ea9f8`

## Schedule-Free `x` versus `y` diagnostic

Training loss was observed on live Schedule-Free `y` weights. Checkpoints were
atomically saved after `optimizer.eval()`, so promotion evaluation correctly
used the stored averaged `x` weights. To determine whether the pathology was
limited to `x`, the terminal optimizer state was used to reconstruct `y` and an
informal six-prompt generic probe was run through the same serializer.

Saved `x` produced no valid answer on five independent prompts or the first
session-memory turn. Every prompt selected a fabricated `read` action, usually
with malformed repeated arguments.

Reconstructed `y` differed but was not competent:

- one of five independent requests returned HTTP 200, but repeated the opaque
  identifier until the token limit and invented a completed workflow;
- the other independent generations were incorrect, repetitive, selected a
  fabricated read, or omitted the required `Final:` prefix;
- the first session-memory turn failed, so no restore/follow-up claim was made.

The `y` diagnostic reported a recurrent state size of 17,795,282 bytes for the
one completed request. This is instrumentation only, not a qualified portable
state artifact.

## Interpretation and next gate

At this learning rate the mixed objective rapidly acquired lower next-token
loss while rapidly overwriting the narrow behavior-selected policy. The 1.8%
Pi target share did not protect literal grounding, action choice, observation
conditioning, valid argument production, or completion discipline. The low
loss also did not yield a usable generic chatbot under the current runtime.

Do not continue the ready 1.35B-target general authority from this terminal
checkpoint without a new bounded experiment. The next training candidate must
use a balanced behavioral curriculum, frequent saved-`x` evaluation, and gates
at 8/32/64/128 updates. Runtime serialization parity and observation-sensitive
next-action tests precede that experiment. GRPO/RLVR remains blocked until a
competent executable baseline exists.
