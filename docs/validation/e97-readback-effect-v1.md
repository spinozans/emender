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
## Completed: all tested interventions had zero measured effect

`proc_de97` completed all40 replays in **330 seconds**, source `cb30a9bd`.
Every condition, including `none`, matched historical actors exactly. Repetitions,
condition comparisons and native-read/observer comparisons all had maximum
probability difference0. The earlier trace was **not reproduced**:
`baseline_trace_binding_passed:false`, with maximum prior-trace gap
.12353801727294922. This is not evidence that the earlier failure was fixed.
Readback, these temporary-allocation patterns and synchronization remain
unsupported explanations in the tested conditions.

Parameters/buffers remained unchanged, gradients absent, peak allocated HBM
8,529,472,512 bytes. Source inventories passed before/after. All40 immutable
receipts were verified against the terminal measurements. Every captured runtime
field also equaled the failed matrix's runtime preflight.

A CPU inventory audit found that all **36 distinct current E88 PTX bodies** and
the one decay PTX body occur in the failed matrix's cache, with matching compiler
metadata after excluding cache hash. Its matrix cache contains99 distinct E88
bodies (101 files), including additional training/layout variants. Normalization
removes only source-location directives, whole-line comments and debug sections.
Regression tests preserve executable text after debug sections, detect changed
instructions/configuration and reject unterminated debug sections. These are
**PTX/config overlaps, not complete executable/runtime equivalence**: they do
not bind launch order, operands, cuBLAS/PyTorch behavior or SASS identity.

- Recipe SHA: `cb898c6bac6b21c1ed849759cc429c117ecf0d92bee647043c55eb57ac05ce38`.
- Summary SHA: `04c4b022260e48eff1b1e593c99455eb09ba78da789e80771abfc8ef63db83eb`.
- Private measurements SHA:
  `08fe98a74a2405857eb41dba1b619b7c3ffb6fcd06a5ecdf9430a7787a790648`.
- Receipt/runtime audit SHA:
  `6259dd040fcb9c86a8226e17235e475ce0cf695c8f5009ca891a4bcd773220c8`.
- Cache overlap audit SHA:
  `68ff7a8c0a84a2cc100ff89042d22936ffb2d98443b6165b325478693cbdc36f`.

Updated CPU suite: **110 passed**. Next is one exact-source complete matrix
reproduction, not another speculative single-factor change; see
[e97-activation-alignment-v1.md](e97-activation-alignment-v1.md).
Optimizer updates0; original failed gates and RL-not-ready status unchanged.
