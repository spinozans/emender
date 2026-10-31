# X42 mimetic initialization and unfrozen-basis verdict

Mimicry improves materially while unfreezing does not: early-training optimization is the stronger lead at this horizon; geometry learning may require more time.

Standing reference: score 4.802025389; raw NLL 4.742689570. Positive deltas below mean improvement.

| Arm | Score | NLL | Score improvement | NLL improvement | Tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| donor | 3.763635 | 3.763635 | — | — | 146800640 | 54326.3 |
| mimetic | 4.530723 | 4.471388 | 0.271302 | 0.271301 | 43909120 | 16220.5 |
| unfrozen | 4.893406 | 4.834070 | -0.091380 | -0.091380 | 47448064 | 17539.3 |

Donor score is raw NLL. Candidate scores retain the unchanged occupancy/degeneracy penalties. Full per-head telemetry is in summary.json and each arm/result.json. The diagnostic score is for comparison only, excluded from certified promotion.

After mimicry, the raw NLL gap to this fresh donor is 0.707753. The donor consumed 3.343x as many training tokens as the mimetic arm during their respective 2700s runs. Remaining quality differences therefore mix architecture, optimization and throughput; they do not isolate a selectivity cost.

| Arm | Occupancy fraction | Mean gate | Event density | R-CAP/token |
|---|---:|---:|---:|---:|
| baseline | 0.933582 | 0.968164 | 0.999702 | 0.397870 |
| mimetic | 0.933526 | 0.976538 | 0.999697 | 0.397837 |
| unfrozen | 0.933568 | 0.970519 | 0.999668 | 0.397909 |

All three pure arms retain the pressure_runaway degeneracy flag (R-CAP >0.05/token), adding 0.05 to each score. Their near-unit write-event densities and high occupancy persist. The mimetic gain is therefore a raw-NLL gain without relief of the pressure penalty. The observed unfreeze delta is a conditional optimization result under this recipe, not an intrinsic frozen-basis price.

| Mimicry NLL improvement ≥0.1 | Unfreeze NLL improvement ≥0.1 | Read |
|---|---|---|
| yes | yes | Both interventions improve raw NLL materially: early optimization and frozen geometry are both plausible contributors. Their interaction is unmeasured. |
| yes | no | Mimicry improves materially while unfreezing does not: early-training optimization is the stronger lead at this horizon; geometry learning may require more time. |
| no | yes | Unfreezing improves materially while this transplant does not: learned geometry is the stronger lead; partial-transplant mismatch prevents ruling out early-training effects. |
| no | no | Neither intervention improves raw NLL materially: the results are consistent with short-horizon selectivity cost, but do not identify it causally. Lossy transplant and slower geometry training remain alternative explanations. |

## Early training dynamics

| Arm | Objective near 270s | Objective near 1350s | Objective near 2700s |
|---|---:|---:|---:|
| baseline | 7.304088 | 5.757504 | 5.103352 |
| mimetic | 7.104864 | 4.860105 | 4.634161 |
| unfrozen | 7.371040 | 5.617749 | 4.646438 |

Means cover the last up to ten updates at each elapsed-time mark. These are training objectives on different consumed prefixes, not held-out NLL; token positions are retained in summary.json. They describe optimization dynamics without establishing a causal mechanism.

## 1.3B initialization recommendation

At 1.3B, validate a geometry-aware donor warm start on trainable content/value/read-out/MLP paths, preserving the frozen spine, with mapped-representation calibration before a scale run. Keep per-head semantic embeddings and key projections frozen: this baseline-LR diagnostic provides no short-horizon benefit from unfreezing.

## Evidence

protocol.json binds baseline policy and compute sources; donor_provenance.json identifies the fresh donor recipe and corpus hashes; mapping.json lists every copied tensor, geometric projection errors and frozen-parameter checks; unfreeze_scope.json reports added trainable capacity; unfreeze_gradients.json proves finite nonzero key and semantic gradients. Each arm records synchronized training time, final-step overshoot, prefix tokens, score and terminal exit. Donor weights stay local and are not committed.

Exactly 144 read-path tensors were initialized: input weights from the donor K/V right-singular basis and input biases zeroed; shared W_in factors from rank-limited donor K weights (donor K bias omitted); W_v from selected donor V rows projected into that input basis, with their biases; read-out from corresponding donor output columns and full output bias; both MLP linears from donor neurons, replicating neurons and dividing outgoing weights by multiplicity. Queries, semantic embeddings, key projections, management, vocabulary embedding/head and layer norms retained baseline initialization. Detailed head indices and layer widths are in mapping.json. The frozen replay spine source files were untouched.

Diagnostic trainable capacity: 150000144 → 227294736 parameters; total tensor capacity stays 227313264. This adds 77294592 trainable parameters.

The frozen baseline is not fully frozen effective keys: its shared content projection learns coordinates within fixed per-head semantic/key geometry. The diagnostic additionally learns that basis. Management remains frozen. No certified source was edited.

## Residual risks

- One seed, one 2700s horizon; wall-clock comparisons consume different token counts.
- Mimetic arm receives an extra 2700s of donor pretraining; this is not equal end-to-end training compute.
- Partial transplant leaves token embeddings, vocabulary head, queries, norms and frozen semantic basis at baseline initialization; representation mismatch and lossy geometry limit any null conclusion.
- Unfreezing adds trainable semantic/key parameters without shrinking MLP; capacity and throughput confound a pure frozen-basis-price estimate.
- The diagnostic inherits baseline LR/warmup; a null or negative result does not exclude benefits after geometry-specific tuning or longer training.
- No shuffled or scale-matched random donor control was run; a mimetic gain does not isolate semantic transfer from weight scale or conditioning.
- The requested 2x2 is an interpretation matrix, not a completed factorial experiment: joint mimicry+unfreezing was not run.
- No proxy result establishes scale transfer or authorizes 1.3B production training; diagnostic is never production.

No resilient training behavior changed; normative requirement IDs not applicable.
