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
## v2 completed, but endpoint binding failed

`proc_d99a` completed all **72 evaluations** in **389 seconds** from `f58a7bee`.
All 36 private profile receipts were retained and independently verified against
the terminal measurements. Source inventories passed before/after. Parameter/
buffer fingerprints stayed unchanged, gradients remained absent, and peak
allocated HBM was 8,553,481,728 bytes. Zero optimizer updates occurred.

Every profile repeated with identical activation hashes and zero probability
delta. All four training-default endpoints matched the prior full-panel
training measurements exactly. **Instrumented actor-cache endpoints did not**:

| Turn | Actor-reference maximum difference |
|---|---:|
| task000 / turn0 | .00062122196 |
| task004 / turn1 | .03957724571 |
| task010 / turn1 | .11893415451 |
| task014 / turn4 | .12353801727 |

All exceed the frozen .0001 endpoint-binding limit. Consequently
`endpoint_binding_passed:false` and **`localization_ready:false`**. The clean
process exit and exact internal repetitions do not qualify the trace as an
explanation of the original actor/training mismatch.

Within this instrumented matrix, sequence-layout changes first differed at
`00.mixer`; autocast and checkpoint-group/MLP switches produced no measured
activation differences at the selected sites, while padding and train/eval
switches did. These are conditional observations, not a proven root cause:
the actor endpoint is unbound. In particular, the trace collector derived its
probabilities from gathered head rows rather than reading the actual native
cache probability directly. We must distinguish observer measurement differences,
observer perturbation and fixture-setting effects before modifying numerical code.

- Recipe SHA: `1aebe6302d42032ca5f9a0af3d63cc5f36d7f401b509801d1a483952cb23d9d5`.
- Summary SHA: `9dd043f18d69479185628e444565300c57ab975a7e6caed94e79dd974468ca2f`.
- Private measurements SHA:
  `9f3ef99fd200ea44d70d344848c2bdda0458cf3ac43e4dcc4dbef7309c85e7d3`.
- `receipt-audit.json` SHA:
  `da1f33af31d9c0f6f4cf3cc130a65adb0f4688e291710d5038dff9335b087160`.

Next is a separately frozen [observer-isolation diagnostic](e97-observer-effect-v1.md).
No RL update, threshold relaxation or model-precision change follows from this
result.

## Frozen complete-matrix reproduction v3

Subsequent [observer](e97-observer-effect-v1.md) and
[readback/allocation](e97-readback-effect-v1.md) controls all matched historical
actors exactly, including the same actor trace path without added per-step work.
The failed matrix's actor discrepancy remains unreproduced. Captured runtime
fields matched, and all subsequent normalized Triton PTX/config variants were
present in the failed matrix cache; that is not complete runtime identity.

Before choosing another numerical intervention, reproduce the **entire original
matrix once** from unchanged numerical source
`f58a7bee340e11a5c765e964fc5585bd6b0d3582`, including its original entry point,
initialization, shape preflight, profile order and CPU trace comparisons. New
artifact root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-activation-alignment-reproduction-v3`.
Controller: `scripts/run_e97_activation_reproduction.sh`.

Require byte-identical original recipe SHA
`1aebe6302d42032ca5f9a0af3d63cc5f36d7f401b509801d1a483952cb23d9d5`
**before GPU acquisition**. Four turns x nine profiles x two repeats =72
evaluations, one fresh checked leased GPU, worker3,600s/outer3,900s plus30s kill
grace, no retries. Keep all original bounds, precision flags, NUMA binding,
isolated Triton cache, .0001 endpoint/repeat limit and unchanged-parameter/no-grad
checks. Verify both controller and original source inventories before/after,
including failure; compose lease release with the original-source EXIT audit.

Evaluate actor and training reference binding independently. If either fails,
retain measurements but do not infer the original mismatch's cause from the
matrix. Even a fully bound/repeatable reproduction does not retroactively explain
or certify the failed run, and four diagnostic turns cannot pass the original
57-turn RL probability gate. Preserve failures and all original thresholds.
No new kernels, model numerics, optimizer updates, admission or promotion.

### Reproduction completed: bound and repeatable, not a probability fix

`proc_ffdb` completed all72 evaluations in **386 seconds**, controller `11f30174`
and unchanged numerical source `f58a7bee`. The recipe was byte-identical before
GPU acquisition; controller and numerical-source inventories passed before/after.
All36 private profile receipts were independently verified. Both **actor and
training reference differences were exactly0** on all four turns; repetitions
had identical activation hashes and zero probability differences.
`endpoint_binding_passed:true`, `repeatability_passed:true`,
**`localization_ready:true`**. Parameters/buffers remained unchanged, gradients
absent, peak HBM8,553,481,728 bytes. Optimizer updates0.

The first measured actor/full-sequence difference is `00.mixer` on every turn.
This is a measured boundary, not identification of an individual operation.
In particular, `00.input` is the wrapper input, **not** a direct observation of
normalized mixer input. Only prediction rows were captured, so differences
elsewhere in the causal prefix are not yet localized.

| Turn | Full-eval vs actor max logp difference | Training-default vs actor |
|---|---:|---:|
| task000 / turn0 | .00001238805 | .00062750932 |
| task004 / turn1 | .01929998398 | .04634499550 |
| task010 / turn1 | .12036108971 | .11980986595 |
| task014 / turn4 | .12353801727 | .00030041765 |

Padding and train/eval switches also produced differences; their effects can
cancel or amplify earlier differences. These maxima may refer to different
tokens and must not be added as contributions. AMP, checkpoint grouping and
MLP chunk changes produced no activation differences at the measured sites.
The original worst training delta **.11980986595 > .05 remains failed**.

The paired CPU audit found **all28 non-segmented profile probabilities and
activation hashes exactly unchanged** from the earlier failed matrix. Only the
eight segmented actor/AMP profiles changed; their first changed measured site
was also `00.mixer`. This narrows the historical variation but does not explain
it or retroactively certify the failed matrix.

- Summary SHA: `04f42be946e9226152e18d18bde252a1a451062745a728e91da642ec49866594`.
- Private measurements SHA:
  `8336d6d5c21b4b4746a44d9e15d7b8978f5a910b3a25ea511257a21c862d02fa`.
- `reproduction-audit.json` SHA:
  `590997c500fbdd85318e07a2da7c2ce500ac59dce493e605a7b7687e2dcaca78`.

Next: [first-mixer component observations](e97-first-mixer-probe-v1.md), including
actual normalized mixer input and raw projections over the entire real prefix.
No kernel/precision change or RL readiness claim follows from localization.
