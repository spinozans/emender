# E97 upstream activation/layout localization v1

The operator requested continued work after the failed head-only precision
probe. This is a separately frozen, **no-update diagnostic**, not an extension
of either closed SFT budget or authorization to relax probability thresholds.

## Question and selected evidence

Where do actor and training activations first diverge, and which execution
settings introduce the difference? The head-only counterfactual reduced the
maximum probability difference to .05302, still above .05, and found upstream
hidden-state differences. No precision fix has been qualified.

Use the same four previously inspected turns: task000/turn0 stable control,
task004/turn1 FP32-head outlier, task010/turn1 original BF16 outlier, and
task014/turn4 additional observed outlier. All are sample0 training episodes.
This post-result diagnostic selection is **not independent validation** and
cannot pass the original all-57-turn qualification.

Inputs are copied from original recipe SHA
`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`.
Actor/training endpoints bind to complete-panel reproduction measurements SHA
`a2e3399ba8974310db27df9094596237e16a2459050e3f8f8169bd197d6dc9f9`.
Checkpoint remains exact correction-y, SHA
`48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`.

## Frozen profiles and comparisons

Nine profiles, each evaluated twice, on each of four turns: **72 evaluations**
inside one fresh single-GPU worker. Identical forced tokens and Torch seed
974223 each time; no new sampled policy or tool interaction. Verify dropout is
zero and all persistent parameters are BF16.

| Profile | Change relative to comparison reference |
|---|---|
| actor-cache | Baseline: eval, no AMP, full prompt then one token at a time |
| actor-amp | Enable BF16 autocast versus actor-cache |
| full-eval | One causal full sequence versus actor-cache's segmentation |
| full-train | Train mode versus full-eval; checkpointing still disabled |
| full-train-amp | Enable BF16 autocast versus full-train |
| masked-unpadded | Loss path with explicit valid/reset/assistant masks and CE128, no padding |
| masked-padded | Add 128-token alignment/padding |
| checkpointed | Enable group3 checkpointed forward path |
| training-default | Set MLP chunk4096, matching original training endpoint |

The sequence-layout comparison changes projection batching and recurrent
segmentation together; it does not isolate individual kernels. The loss-path
switch introduces explicit masks and head chunking together. Upstream taps
separate hidden-state changes from head-only changes, but these are named
recipe switches, not claims that every low-level operation is orthogonal.
No gradients or checkpoint backward passes run; this only probes forward paths.
Immediate same-profile repeats test local repeatability, not every possible
warm-up/order dependence. Execution order is frozen; no post-result reshuffling.

## Non-mutating observations and binding

Capture only rows that predict the recorded generated tokens at 92 sites:
embedding output; input, mixer output, MLP input, MLP output and wrapper output
for each of 18 layers; and final head input. Handle head chunk offsets explicitly.
In the segmented actor, skip the unused prediction after the last consumed
token. Reject missing stages, row/order/shape mismatches and nonfinite values.
All hooks return None and cannot replace activations, logits, inputs or losses.

Compare per-row relative L2, maximum absolute difference, differing coordinate
counts, dtypes and activation byte hashes against the actor and the named pair.
Report the first observed differing site in execution order; this localizes a
boundary, not necessarily the underlying kernel or mathematical root cause.
Private evidence contains numeric traces/hashes and probabilities, not public
model reasoning. Full activation banks are transient CPU data, not model weights.

Both actor-cache and training-default probabilities must reproduce their bound
references within **.0001**. Same-profile activation repeats must be identical,
with probability repeats within .0001. Otherwise set `localization_ready:false`:
retain measurements but do not claim they explain the original failure.
Model/buffer fingerprints must remain unchanged and gradients absent.
Original qualification thresholds remain unchanged; `rl_optimizer_ready:false`
regardless of any local agreement.

## Bounds, implementation and validation

Maximum prefix16,384, generated512 and feature width3,840. Select prediction
rows only; transient activation-bank budget3GiB, allowing up to six bank-sized
allocations during assembly/comparison. FP32 probability arithmetic is bounded
to128 rows. No FP32 model/master weights or CPU Adam arithmetic.

One checked leased GPU; explicit local device/NUMA placement, isolated Triton
cache, both expandable allocator flags, Python hash seed0, highest matmul
precision/TF32 disabled, BF16 reduced-precision reduction disabled. Worker3,600s,
outer3,900s, 30s kill grace, no restart/retry, no communicator reuse or distributed/
resilient qualification. No source admission, checkpoint promotion or live policy
change. Freeze code/recipe and verify immutable source inventories before/after.

- `scripts/diagnose_e97_activation_alignment.py`
- `scripts/run_e97_activation_alignment.sh`
- `tests/test_e97_activation_alignment.py`

CPU suite: **89 passed**, including eight new tests for segmented/chunked target
alignment, unchanged outputs, hook removal, stage/dtype comparisons, nonfinite
rejection, row/storage bounds and the predeclared profile-pair switches.

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-activation-alignment-v1`.
## v1 failed: unsupported diagnostic training length

`proc_9794` exited1 after **217 seconds**, source `cffeb993`. The existing
recurrence guard correctly rejected the first `full-train` invocation:
`unaligned recurrence padding is forward-only; training lengths must align to the sparse checkpoint interval`.
The sparse checkpoint interval is16, regardless of the outer no-grad context.
The diagnostic's unpadded train-mode condition was invalid; this is not evidence
of a model capability failure or a reason to weaken the training guard.

Three completed-profile log markers survive for task000/turn0: actor-cache,
actor-amp and full-eval. The latter logged `00.mixer` as its first differing
site against actor-cache. **The matrix is incomplete**, and v1 did not publish
per-profile numeric records before failure; no terminal endpoint/repeatability
or final fingerprint verdict is available. Do not promote that partial log
into a complete localization result. No optimizer updates occurred.

Original files remain at `native-activation-alignment-v1`. Its frozen recipe SHA
is `1babe845d2e5f7ba8b9082ca60fce39a596c88796cb49217008e7f1993146914`.
The original source inventory was manually verified after failure. The immutable
`failure-analysis.json` SHA is
`de7572420385e105d93a6ab3df779cbf4ef7126bda46e7b55f41f8d7b8710a98`.

## Separately frozen v2: only supported train-mode shapes

Retain the same four inputs, model, thresholds, hooks, repeated evaluations and
resource limits. Revise the profile order so train/eval is compared **after
explicitly padded, aligned input construction**:

1. actor-cache
2. actor-amp — AMP versus actor-cache
3. full-eval — sequence layout versus actor-cache
4. full-eval-amp — AMP versus full-eval
5. masked-unpadded-eval — explicit masks/loss path/head chunking in eval mode
6. masked-padded-eval — padding versus masked-unpadded-eval
7. masked-padded-train — train mode versus masked-padded-eval
8. checkpointed — group3 checkpointed forward versus masked-padded-train
9. training-default — MLP4096 versus checkpointed

Again **four turns x nine profiles x two evaluations =72**, one fresh GPU
worker. Recipe schema is now `emender-e97-activation-alignment-v2`; incompatible
v1 recipes fail closed. CPU freeze/load preflight records exact forward lengths
and rejects any train-mode length not divisible by16 before GPU execution.
At runtime, verify that the actual kernel interval is16 and every positive
projection chunk is aligned. **No kernel/model guard or numerical operation was
changed.** Unpadded conditions remain eval-only, where existing forward-only
padding is supported. This changes the diagnostic, not the production trainer.

Publish `runtime-preflight.json` and each completed profile's private numeric
receipt immediately. These immutable records survive a later profile failure;
final summaries bind their hashes. Identical receipt publication is idempotent,
while differing bytes are rejected. Preserve original errors and use a new root,
not a retry/resume of a failed child or CUDA context. Source inventories are
verified after execution even if the revised run fails.

Added CPU regressions cover the original invalid profile, actual loss-input
lengths at padding boundaries, the unchanged production-kernel rejection, and
retention/non-overwrite of completed receipts. Full suite: **98 passed**.
New artifacts:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-activation-alignment-v2`.
Results pending; no RL update or threshold relaxation is authorized by a clean
process exit.
