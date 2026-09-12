# E97 same-trace readback/allocation controls v1

The [observer-isolation diagnostic](e97-observer-effect-v1.md) matched recorded
actors exactly in all five conditions, including all92 hooks. However, each
condition read native probabilities during decoding; the earlier unbound trace
collector did not. This experiment controls that specific execution difference.
Neither observer perturbation nor allocation/synchronization sensitivity is
assumed to be the cause before measurement.

## Frozen scope

Use the **same activation diagnostic `evaluate` actor path**, same full92-site
trace, same actor attributes and disabled-autocast context in every condition.
Add an optional per-step callback before token consumption. Default None adds
no CUDA operation to the original trace path; no model output, activation,
parameter, input or sampled token is replaced.

Five conditions x four previously inspected turns x two repetitions = **40
forced actor replays**, one fresh worker. Reverse condition order on the second
repetition, with Torch seed974223 before each replay:

1. **none:** original trace, no extra per-step work.
2. **read:** independently compute the native one-dimensional FP32 log-softmax
   and selected probability; retain this separately from the observer estimate.
3. **allocate:** allocate and discard two vocabulary-sized FP32 tensors; neither
   is read or supplied to the model.
4. **copy-allocate:** cast native BF16 logits to a temporary FP32 tensor and
   allocate/discard a second FP32 tensor, without softmax or scalar readback.
5. **synchronize:** device synchronization only, without extra probability or
   tensor-allocation work in the callback.

Compare observer-derived probabilities against historical actors, the prior
unbound trace, the no-extra-work condition and repeat/order counterparts. For
`read`, also compare independent native probabilities directly to the observer.
Keep the .0001 diagnostic comparison limit. Explicitly report whether the
`none` baseline reproduces the prior trace; if it does not, do not claim to have
explained that earlier failure. All cases remain post-result diagnostics, not
independent validation. Any observed sensitivity is evidence for further
localization, not proof of a specific kernel bug or permission to relax gates.

Source recipe SHA:
`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`.
Prior trace SHA:
`9f3ef99fd200ea44d70d344848c2bdda0458cf3ac43e4dcc4dbef7309c85e7d3`.
Exact correction-y checkpoint and expected `f57e...` parameter digest unchanged.
Require unchanged parameters/buffers and no gradients. No model numerical code,
precision setting, kernel guard, optimizer, live rollout or training data changes.

## Controls and validation

- `scripts/diagnose_e97_readback_effect.py`
- `scripts/run_e97_readback_effect.sh`
- Optional `step_probe` in `diagnose_e97_activation_alignment.evaluate`

Per callback, require one-dimensional BF16 logits, vocabulary <=65,536 and valid
integer token; at most512KiB of additional transient FP32 storage. Existing
trace bounds remain. All temporary allocation contents are ignored. No FP32
master weights or CPU Adam arithmetic.

Checked single-GPU lease, explicit local device/NUMA, isolated Triton cache,
both expandable allocator flags, hash seed0, highest precision/TF32 disabled,
BF16 reduced-precision reduction disabled. Worker3,600s, outer3,900s, 30s grace,
no retries or communicator reuse. Freeze code/recipe and verify source before
and after, even on child failure. Preserve per-condition immutable private
receipts before continuing. No private generation content is published.

CPU suite: **106 passed**, including callback non-mutation/range tests and a
same-function toy replay demonstrating optional direct readback agrees with
observer probabilities and leaves the activation trace unchanged. CPU tests
are not a GPU probability-path qualification.

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-readback-effect-v1`.
Results pending. Optimizer updates0; original failed gates and RL-not-ready
status remain unchanged.
