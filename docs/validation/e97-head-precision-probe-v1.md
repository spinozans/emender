# E97 read-only head-precision probe v1

## Motivation and frozen scope

The [complete probability assay reproduction](e97-native-rl-logprob-qualification-v1.md)
matched recorded actor probabilities exactly on all 57 turns / 2,711 tokens, but
training-layout maximum difference remained .11981 against the frozen .05 limit.
Training probabilities exactly reproduced the original assay. The earlier actor
replay discrepancy did not reproduce and its cause remains unresolved.

Current `loss_logits_fp32` casts the **BF16 head output** to FP32 for CE; it does
not project hidden states through the head in FP32. Small differences across
actor/training layouts could cross BF16 rounding boundaries and produce much
larger output differences. This is a hypothesis to measure, not a claimed cause.

Keep the original complete selection from recipe SHA
`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`:
all 57 generated turns from sample 0 of all 16 training tasks, no filtering.
Use unchanged correction-y checkpoint SHA
`48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`,
original padded/reset/valid/assistant masks, group3/MLP4096 and FP32 CE128.

## Observer, not policy modification

Add read-only LM-head hooks:

1. At actor prefill/each token, observe only the hidden row predicting the
   corresponding forced sampled token. Skip the unused prediction after the
   final consumed token; require exact N+1 hook-call/N-prediction coverage.
2. Save bounded BF16 hidden rows privately in CPU memory, not model weights.
3. At the corresponding masked training-head rows, verify exact target order
   and measure hidden-state absolute/relative differences.
4. Project those same hidden rows counterfactually in FP32, with autocast
   disabled and TF32 disabled/highest precision. Keep the actual BF16 outputs.
5. Compare counterfactual actor/training FP32 probabilities and selected logits
   alongside the actual BF16 selected logits and unchanged primary assay.

**Hooks return None; they never replace logits, recorded probabilities, token
IDs, model responses, observations or losses.** No inference path or training
head is switched to FP32. The counterfactual probabilities are not captured
behavior probabilities and must not be used as such in RL.

FP32 projection uses at most 4,096 vocabulary rows of temporary head weights
at once, at most 128 hidden rows and one bounded rows-by-vocabulary output.
For this model, the weight-block cast is at most 62,914,560 bytes. No complete
FP32 head-weight copy, persistent FP32 parameters, CPU Adam or optimizer update.
The BF16 hidden bank for 2,711 rows x 3,840 features is about 20.8 MB.

Report original actor/training checks unchanged (.0001 replay max, .05 training
max, .02 p99 and .0001 CE consistency). Report counterfactual FP32 actor/training
max/p99 against the same .05/.02 limits **as diagnostic information only**,
with `changed_policy_qualified:false` and `rl_optimizer_ready:false` regardless
of that comparison. Hash model parameters before/after. Reject missing targets,
nonfinite values, wrong dtypes, unsupported precision or excess probe bounds.

## Execution

- `scripts/e97_head_precision_probe.py`
- `scripts/qualify_e97_native_rl_logprobs.py freeze --head-probe`
- `scripts/run_e97_head_precision_probe.sh`

One checked leased GPU, explicit device/NUMA placement, isolated Triton cache,
both expandable allocator flags, Python hash seed 0, 3,600-second worker and
3,900-second outer limits with 30-second kill grace. No retries, new tool
rollouts, data admission, promotion, distributed/resilient qualification or
extension of closed SFT budgets. Code/recipe/source inventories frozen before
execution and verified after. Private per-token evidence remains private.

CPU suite: **81 passed**, including seven tests of observer output/parameter
preservation, target alignment, bounds, bias/no-bias, vocabulary chunking and
FP32 shadow arithmetic under active autocast.

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-head-precision-probe-v1`.
Results pending.
