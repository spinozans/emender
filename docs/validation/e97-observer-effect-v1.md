# E97 actor observer-isolation diagnostic v1

The [activation matrix](e97-activation-alignment-v1.md) completed and repeated
exactly, but its actor endpoint failed historical-reference binding by up to
.12354. Its training endpoint matched exactly. Do not claim upstream numerical
causation until the observer/fixture itself is understood.

## Frozen question and modes

Use the same four diagnostic turns, same exact correction-y checkpoint and
original recorded tokens/probabilities. Source recipe SHA:
`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`.
Prior trace SHA:
`9f3ef99fd200ea44d70d344848c2bdda0458cf3ac43e4dcc4dbef7309c85e7d3`.
Expected parameter digest:
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`.
This is post-result diagnostic selection, not independent evaluation.

Five modes x four turns x two repetitions = **40 forced actor replays**:

1. **native:** original loaded actor attributes, no observer/autocast wrapper.
2. **attributes:** apply the matrix's actor settings (eval, checkpointing off,
   group3 attribute, CE128/FP32-loss flags, MLP chunk0), no observers/context.
3. **disabled-autocast:** those attributes plus the matrix's explicitly disabled
   autocast context. No observers.
4. **head-observer:** add only the matrix collector's head hook.
5. **full-observer:** add all 92 matrix trace sites.

Restore original attribute values **and absence** before each mode, including
MLP chunk settings. Use Torch seed974223 every time. Reverse mode order on the
second repetition to expose some order effects; this does not test all possible
process initializations or shape histories. Assert zero dropout, BF16 persistent
parameters, unchanged parameter/buffer fingerprints and absent gradients.

For every forced prediction, independently read
`log_softmax(cache.next_logits.float(), -1)[token]` **directly from the native
one-dimensional cache output**, exactly as the original probability assay did.
This is the authoritative measurement, never the hook's gathered-row estimate.
When hooks are active, retain their estimate separately and compare it directly
to the native cache. Hooks cannot replace model outputs. Skip unused final-token
predictions and validate exact generated-token coverage.

Report native-vs-recorded, native-vs-prior-trace, observer-vs-native, mode-vs-native
and reversed-order repetition differences. Preserve .0001 as the diagnostic
comparison limit. No successful local comparison qualifies end-to-end RL or
relabels any prior failed experiment. No parameter/precision/kernel changes,
new sampled rollouts, tool calls, data admission or promotion.

## Execution and validation

- `scripts/diagnose_e97_observer_effect.py`
- `scripts/run_e97_observer_effect.sh`
- Observer regressions in `tests/test_e97_activation_alignment.py`

One checked leased GPU, explicit local device/NUMA placement, isolated Triton
cache, both expandable allocator flags, hash seed0, TF32 disabled/highest
precision and BF16 reduced-precision reduction disabled. Limits remain prefix
16,384 / generated512, bounded trace buffers and at most128 FP32 probability
rows. No FP32 master weights or CPU Adam. Worker3,600s, outer3,900s, kill grace30s,
no retries or communicator reuse. Source inventories checked before and after,
including after a failed child. Each completed mode gets an immutable private
receipt before the next mode starts. Private generations remain unpublished.

CPU suite: **101 passed**. New regressions verify attribute restoration,
non-mutating toy cache/head/full observations, and deliberately corrupt an
observer estimate to prove it cannot replace the independently measured native
probability. These are CPU collector checks, not numerical GPU qualification.

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-observer-effect-v1`.
## Completed: no discrepancy in the tested controls

`proc_604f` completed all **40 replays** in **303 seconds** from `98781c3b`.
Every mode matched recorded actors exactly. Observer estimates matched direct
native cache probabilities exactly. Mode comparisons and reversed-order
repetitions all had maximum difference0. Parameters/buffers and original model
attributes remained unchanged; gradients were absent. Peak HBM was
8,529,472,512 bytes. Source inventories passed before/after, and a CPU audit
verified all40 immutable receipts against the terminal measurements.

This does not reproduce or explain the earlier failed trace endpoint. It does
not support blaming the tested attributes, disabled autocast context or hooks
in isolation. **Every condition in this experiment performed a direct native
probability read during decoding**, unlike the original trace collector. That
remaining execution difference must be tested rather than assumed harmless or
causal. No numerical fix or RL qualification is claimed.

- Recipe SHA: `bdef4b1ad0b2de7b8e802af2e76c83d04f3b9b34e1c262559ac2302ea603ec1d`.
- Summary SHA: `99610daa6fef5020de87e2c23dbb059ef7d3eb70fa58654d62f68337e49fdc6b`.
- Private measurements SHA:
  `65bbde3da22678e013ebd64748167d4c8af6b42dd63883bb50eeacd67389503b`.
- `receipt-audit.json` SHA:
  `ee9122a77f8bc09a1adc67e4a0a3cfe331cf80ec4d7596be127f505cbb24b45f`.

Next: [same-trace readback/allocation/synchronization controls](e97-readback-effect-v1.md).
Optimizer updates0; `rl_optimizer_ready:false` throughout.
