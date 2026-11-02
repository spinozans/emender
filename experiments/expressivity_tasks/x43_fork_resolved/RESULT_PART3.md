# X43 part 3: fork-conditioned write-gate retrain

The preregistered prediction fails. Fork-only eligibility worsens both fork and forced loss. This remains true in the fixed step-1000 comparison. The intervention successfully prevents writes at forced consumed-byte positions, but does not improve this model.

Prediction and executable hashes were bound in `receipts/part3/registration.json` before training and evaluation, and committed in `b816217e` before the first evaluation. Registration SHA-256: `d033be56e29a62896c62052254474be03feabb5c514bf845d8452027240df4ca`. No result-driven tuning or retraining was performed.

## Gate implementation receipt

`write_gate.py:masked_writes` supplies a boolean mask to the unchanged Candidate scan: `wire_logits.masked_fill(~mask[..., None], -inf)`. The native policy remains `sigmoid(wire + occupancy_pressure + match_coef * match_evidence + error_coef * error) > enable_threshold`. Ineligible wire values yield probability zero and write case 0: no append, allocation, overwrite, or merge write occurs. Eligible logits are unchanged. This is a constant conjunction with the existing gate, not a learned manager and not an instruction to write at every fork. W_write, W_pop and W_append remain frozen. Parameter count remains 2,000,324 trainable (2,085,588 total). No upstream model, kernel, capacity, optimizer, or read operator was changed.

The consumed input byte at offset t uses fork[t]; its loss predicts t+1. Reads precede writes within a cell. Thus this stores the representation of an already consumed ambiguous decision for later reads, rather than using fork[t+1] to select a context representation. Every layer/head shares the same eligibility mask. POP, decay, expiry and read computations still run at forced positions; write mutation is disabled there. The kernel still computes evidence and routing, so this does not implement a computational fast path or establish a speed gain.

Labels are exactly the retained part1c classification (SHA-256 `294b56a92e38aa723ff2867afac2c25dd0d980c3ef4e548ef1fb3833b449bd35`), whose corpus-wide agreement with the fixed xsa is receipted in part1d. This experiment does not rebuild the index or claim the labels are causal. It explicitly implements the requested label-conditioned probe, including oracle labels during held-out evaluation. The original part-1 training-only-mask requirement is superseded for this probe by PREREGISTRATION_PART3.md; a training-only causal fork-gate experiment remains unrun.

Inference without labels has two separately receipted options: (1) walk a fixed training-only index on observed left context, determine continuation diversity with explicit backoff/abstention for absent contexts, and validate its causal decisions and latency; (2) use a fixed internal confidence proxy from match-evidence/write-strength distributions, with separately registered calibration and classification validation. Neither option is implemented or validated here. Neither may consume future held-out labels. Production semantics require one of them; these oracle-mask results alone do not provide a deployable gate.

## Held-out loss receipts

Full held-out: 111,539 targets, 64,041 fork and 47,498 forced, original nonoverlapping 128-byte resets. Natural-log NLL in nats/byte. Lower is better.

| Arm | Steps | Overall | Forced | Fork | NLL at forks |
|---|---:|---:|---:|---:|---:|
| Retained emender | 5552 | 2.201701 | 1.761594 | 2.528119 | 65.928% |
| Retained transformer | 35853 | 1.829293 | 0.773061 | 2.612681 | 82.004% |
| baseline 180s | 4421 | 2.264813 | 1.844641 | 2.576447 | 65.316% |
| bernoulli_half 180s | 1370 | 2.488181 | 2.147842 | 2.740604 | 63.241% |
| bernoulli_chi 180s | 5894 | 2.457826 | 2.113750 | 2.713021 | 63.377% |
| fork 180s | 5390 | 2.381481 | 1.998269 | 2.665702 | 64.268% |

The quoted 2.223 Emender baseline is part-2 final test, not the complete held-out set used in part1c. The retained full-held-out baseline is 2.201701 with 65.928% of NLL at forks. These denominators are kept separate below.

| Final-test subset (55,770 targets) | Overall | Forced | Fork |
|---|---:|---:|---:|
| Retained emender | 2.222975 | 1.798005 | 2.529165 |
| Retained transformer | 1.924201 | 0.938890 | 2.634118 |
| baseline 180s | 2.283055 | 1.878657 | 2.574424 |
| bernoulli_half 180s | 2.494319 | 2.165487 | 2.731242 |
| bernoulli_chi 180s | 2.466653 | 2.134284 | 2.706126 |
| fork 180s | 2.397326 | 2.033236 | 2.659654 |

Every fresh arm uses the original initialization seed 4301, byte RNG 4301, batch 16, context 128, policy N=2/budgets 32 and 64, width 256/two layers, AdamW lr 3e-4 and weight decay .01, clip 1, bfloat16 training and fp32 evaluation, same corpus/split, and 180-second loop. Physical GPU 2 was acquired exclusively through the broker restricted to 2/4. No forbidden card was selected. The Bernoulli RNG is independent (4303 training, 4304 evaluation), with one mask per occurrence broadcast to all heads.

First-use compilation and checkpoint-save overhead are charged to each 180-second budget, and compilation was observed during training. Preflight validation also compiles some shapes; caches are not reset between arms. Wall-time endpoints have markedly unequal steps and are not a matched-token causal comparison. The separately registered step-1000 checkpoint resolves the step-count comparison at an early training point, without extending or selecting training based on evaluation.

| Fixed 1,000 steps | Overall | Forced | Fork | Δ forced vs baseline | Δ fork vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 2.464841 | 2.111915 | 2.726599 | +0.000000 | +0.000000 |
| bernoulli_half | 2.483494 | 2.139684 | 2.738492 | +0.027769 | +0.011893 |
| bernoulli_chi | 2.482090 | 2.141639 | 2.734595 | +0.029724 | +0.007996 |
| fork | 2.472841 | 2.126162 | 2.729965 | +0.014248 | +0.003366 |

At 1,000 steps fork conditioning is less harmful than either independent mask, but still worse than baseline on both classes. This single run establishes neither a general advantage nor a significant difference among masked policies.

## Mechanism receipts

Training-slice corpus-label fork density chi = 0.57048734; full corpus density = 0.57085389. Telemetry below is measured on held-out evaluation, averaged equally across two layers and two heads; each head statistic is weighted by window length. Full per-head receipts are in the evaluation JSON files. Training eligibility density is measured across sampled occurrences; training event density/occupancy are not separately instrumented.

| Arm | Training eligibility | Eval eligibility | Actual events | Mean gate | Mean live occupancy | Mean prewrite occupancy | Tokens/s including compilation |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.000000 | 1.000000 | 0.462504 | 0.447066 | 0.364668 | 0.360210 | 50299 |
| bernoulli_half | 0.499849 | 0.501233 | 0.104129 | 0.144374 | 0.182879 | 0.180886 | 15586 |
| bernoulli_chi | 0.570444 | 0.572275 | 0.145420 | 0.150845 | 0.180317 | 0.178254 | 67051 |
| fork | 0.570234 | 0.574149 | 0.204370 | 0.205301 | 0.243827 | 0.240846 | 61322 |

Actual event density falls from 0.462504 to 0.204370, rather than rising to approximately the fork density. Eligibility and actual writes are different: a fork must also pass the unchanged evidence/occupancy policy, and sparsifying writes changes future evidence. Mean live occupancy falls from 0.364668 to 0.243827. Thus the measured mechanism is less occupied storage and fewer writes, not evidence of better allocation to decisions. Every evaluated scan, including both checkpoint endpoints and Bernoulli controls, has zero forbidden write events.

## Validation and limitations

Validation passed: all-true gate is bitwise identical to unchanged forward; all-false produces zero write events; alternating masks never write at forbidden positions; checkpoint recomputation/backward has finite gradients; management parameters stay frozen. CPU validation replays all eight full-held-out decompositions, target coverage, registration hashes, density bounds and paired differences exactly. Scientific prediction: failed (both classes regress), not a positive result.

Residual risks: corpus-wide labels leak held-out structure and may use contexts beyond the 128-byte reset window; consumed-decision alignment differs from gating on the next predicted decision; causal index/proxy inference is unimplemented; one corpus, seed and short training; unequal compilation costs and optimizer steps at 180 seconds; the matched-step result is at 1,000 steps only; arithmetic averaging of head occupancy does not weight different capacities; evaluation telemetry does not measure training occupancy; no end-to-end index latency, uncertainty estimate or architecture ranking is established. Bernoulli eligibility masks are independent of bytes but effective write events remain state-dependent. Original checkpoint comparisons differ in training steps and cache conditions.

No .pt files are committed. Local checkpoint SHA-256 hashes are recorded in training receipts. Existing unrelated worktree changes were left untouched. The contact_supervisor tool is unavailable in this session; no coordination messages could be sent through it.

Reproduce registration only before a fresh run (do not overwrite the bound receipt after evaluation):

```bash
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.write_gate --register
bash experiments/expressivity_tasks/x43_fork_resolved/run_write_gate_leased.sh
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.validate_write_gate
```
