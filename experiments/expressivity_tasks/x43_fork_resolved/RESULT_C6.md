# X43 C6 — ordinal crossover survives; tested fork-aware recipe fails absolutely

The hash-registered direction survives on this corpus-label diagnostic: Emender minus transformer is **−0.629543 overall nats/byte**, or **−1.083128 fork nats/byte**. This does not establish a sound architecture foundation: all three fork-aware heads are dramatically worse than their retrained controls and historical C3. The explicit OOD channel receives no positive OOD training examples under self-containing training counts. Do not promote the ordinal result to an OOD, causal-generalization, or architectural-superiority claim.

All losses are natural-log nats/byte. The primary all-forced diagnostic has zero forced loss by definition. C3-compatible columns walk only nonempty train support and abstain to the head at empty support, exactly the registered clarification. These columns are the direct C3 comparison; the stronger all-forced diagnostic is separate.

| Head | Fork NLL | Primary all-forced standalone | C3-compatible standalone | C3-compatible fusion | Historical C3 fusion |
|---|---:|---:|---:|---:|---:|
| emender | 10.433393 | 6.064164 | 6.694547 | 6.694321 | 1.673012 |
| transformer | 11.516522 | 6.693707 | 7.324090 | 7.312338 | 1.746996 |
| gdn2 | 11.360357 | 6.602940 | 7.233323 | 7.195213 | not measured |

Retrained neural-only controls (ordinary 256-way softmax, all-position loss, zero conditioning):

| Head | Neural overall | Fork | C3-compatible fusion |
|---|---:|---:|---:|
| emender | 2.231550 | 2.527107 | 1.676996 |
| transformer | 1.858585 | 2.538039 | 1.699410 |
| gdn2 | 1.754910 | 2.380693 | 1.602624 |

The retrained controls retain the Emender/transformer post-hoc fusion crossover (1.676996 < 1.699410). GDN-2 control plus fusion reaches 1.602624; this is a C6 control receipt, not a C7 designed-head or ablation result. Historical C3 standalone was 2.222975/1.924201. New controls share BF16 autocast and a new common initialization stream; they are fresh controls, not exact historical training replays.

| Raw head stratum | Overall (55,770) | Fork (32,415) | Forced (23,355) | Covered (46,721) | Unsupported (9,049) | Nonempty zero-target (17,066) |
|---|---:|---:|---:|---:|---:|---:|
| emender_aware | 6.694550 | 10.433393 | 1.505314 | 6.917163 | 5.545177 | 18.159355 |
| emender_control | 2.231550 | 2.527107 | 1.821338 | 2.186475 | 2.464275 | 2.886512 |
| transformer_aware | 7.324095 | 11.516522 | 1.505319 | 7.668639 | 5.545177 | 20.142119 |
| transformer_control | 1.858585 | 2.538039 | 0.915555 | 1.746939 | 2.435027 | 3.277586 |
| gdn2_aware | 7.233436 | 11.360357 | 1.505577 | 7.560421 | 5.545177 | 19.707956 |
| gdn2_control | 1.754910 | 2.380693 | 0.886368 | 1.652565 | 2.283325 | 3.048477 |

Covered includes zero-target cases. Unsupported includes 2,709 forks and 6,340 forced positions; none are dropped. Complete strata for every standalone/fusion variant are in results.json and c3_comparison.json.

| Arm | Configured trainable parameters | Steps | Input tokens | Training seconds | Initialization seconds | Held-out inference seconds |
|---|---:|---:|---:|---:|---:|---:|
| emender_aware | 2,067,140 | 5,440 | 11,141,120 | 180.007880 | 0.024569 | 1.265495 |
| emender_control | 2,067,140 | 5,668 | 11,608,064 | 180.006443 | 0.021688 | 1.262602 |
| transformer_aware | 2,067,188 | 24,982 | 51,163,136 | 180.007299 | 0.022209 | 0.209562 |
| transformer_control | 2,067,188 | 28,657 | 58,689,536 | 180.002074 | 0.020843 | 0.212403 |
| gdn2_aware | 2,067,638 | 8,944 | 18,317,312 | 180.010553 | 5.037383 | 1.525661 |
| gdn2_control | 2,067,638 | 9,828 | 20,127,744 | 180.014084 | 0.025190 | 0.465158 |

Index preparation took 23.949274 seconds for 517,711 distinct intervals. Same corpus, 90% train prefix, seed 4301, context 128, batch 16, AdamW lr .0003 / decay .01, clipping 1, and 180-second loop per arm. First kernel compilation and synchronization in the loop are charged; compilation cost is not isolated from training. Initialization, index construction, imports, and inference are outside that clock. Curves retain first-step timing and sampled training losses. Controls include 66,816 allocated conditioner/OOD parameters with no gradients, so active parameter counts are correspondingly smaller. Configured cross-head parameter mismatch is below 0.025%.

All GPU work used broker-leased physical card 2 (RTX 6000 Ada, 49,140 MiB); other jobs were left alone and card 7 received no work. No Slurm or Frontier execution occurred. Interpreter: /home/erikg/emender/.venv/bin/python; FLA 0.5.1. Actual GDN-2 is NVIDIA GatedDeltaNet2, external commit 95709fc250357c2dd109361c353192f2aa5913f9, with source hashes in gdn2_source.json.

The initial optional constructor incorrectly selected FLA GatedDeltaNet (GDN-1). The own runner was terminated with exit 143 before any optional-arm training step or outcome receipt, after preserving four completed Emender/transformer rows. The separately registered actual GDN-2 continuation completed both optional arms with exit 0. Original console, terminal failure, interruption reason, and continuation console/terminal are retained. Fresh run_c6_leased.sh now selects the corrected constructor; c6.py remains unchanged as the original hash-bound execution source. Initial pytest collection failed on a relative import, corrected before training; all five final tests pass. No .pt files were produced or committed.

Primary registration: db00edf799d51c2b9f0e877d3a8355ebc0b1a5f6. C3 empty-support clarification, before first inference: 64a8cce690beb7facf95fe5ddedc87db8406f335. Optional-arm correction is separately registered in PREREGISTRATION_C6_GDN2.md. Original first-four execution source: bceb3c92b67506417f85b0cdbda8edb9f24d316b; dependencies.json checks 29 imported project source files against that commit.

Calibration uses exactly 55,769 targets; final scoring uses 55,770. Gates were fitted only on eligible calibration forks, serialized and SHA-bound in calibration_freeze.json before any final loss was calculated. validate_c6.py independently replays 180 loss/denominator checks, hashes, parameter matching, and budget accounting. validate_program.py audits committed bindings, not scientific entailment.

Residual risks and design decision: selectors and fork labels remain corpus-wide and transductive despite train-prefix counts. A causal training-only selector/arm was not executed; that broader C6 ledger requirement remains open. Training has 572,686 forks and zero OOD targets, while final test contains 17,066 nonempty-support zero-target forks. Those receive raw head NLL 18.159355/20.142119/19.707956 for Emender/transformer/GDN-2. Uniform OOD routing with self-containing training counts is a failed recipe; do not scale it or claim an OOD benefit. Relative ranking may reflect different OOD collapse or optimization progress rather than better in-support decisions. Before C7, register a causal selector and leave-one-out/prefix-only OOD exposure or another justified objective, and isolate branch normalization. One seed/corpus establishes no significance or generality. Branch conditioning and interval features were not ablated. C3 remains historical measured evidence: C6 neither establishes a post-hoc artifact nor validates the architecture foundation from its surviving direction.
