# X43 C11 — device inputs with dense loss: predictions refuted

P1, P2 and P3 fail for both heads. Input conditioning removes learned mixture-gate routing from training, but full dense CE does not guarantee held-out OOD quality. Self-containing exact distributions can become shortcut inputs even though gradients still flow through the head. This is a mechanism receipt, not a claim that every device interface fails.

Preregistered in cc137968 before training; protocol SHA256 e682a17d36ec33630096bfbafa89724974eb8282df7d9d175265c46e0424f8e6. Source c11.py frozen in that commit. Exact train-prefix branch counts reconstructed by direct-SA, checked against every C6 target probability, with corpus and feature-array hashes in device.json. 260 input features enter an embedding residual projection; internal memory, write policy and full 256-byte CE are unchanged. No training gate, teacher objective or branch output normalization. The inherited metadata field fork_aware=false denotes absence of C6 branch normalization, not absence of device inputs.

All values nats/byte. Common C3 calibration frozen before final scoring; 55,769 calibration and 55,770 final targets. OOD means empty index support (9,049), not semantic domain shift; fork denominator 32,415. Nonempty zero-target forks (17,066) retained separately in JSON.

| Head | Recipe | Inference | Total | Fork | OOD empty | Nonempty zero-target |
|---|---|---|---:|---:|---:|---:|
| emender | dense | neural | 2.222975 | 2.529165 | 2.444723 | 2.890245 |
| emender | dense | common_c3 | 1.673012 | 2.424886 | 2.444723 | 3.062555 |
| emender | ops | neural | 3.872928 | 5.330632 | 6.551557 | 8.337070 |
| emender | ops | common_c3 | 3.870401 | 5.327873 | 6.551557 | 8.344468 |
| transformer | dense | neural | 1.924201 | 2.634117 | 2.466627 | 3.401529 |
| transformer | dense | common_c3 | 1.746996 | 2.558332 | 2.466627 | 3.463626 |
| transformer | ops | neural | 4.953106 | 7.317413 | 6.060058 | 12.110113 |
| transformer | ops | common_c3 | 4.941959 | 7.298356 | 6.060058 | 12.128367 |

## Swap test — remains registered, unexecuted

| Arm | 0 steps | 100 steps | 200 steps |
|---|---|---|---|
| A-trained ops on B, B index | not run | not run | not run |
| A-trained ops on B, zero device inputs | not run | not run | not run |
| A-trained dense on B | not run | not run | not run |

P4/P5 have no outcome. Corpus B and both swap indexes were not constructed; additive source/split/hash registration is required before executing this deferred arm. No corpus-swappability claim is established.

## Execution and validation

| Head | Active configured parameters | Steps | Tokens | Training seconds |
|---|---:|---:|---:|---:|
| emender | 2133700 | 2015 | 4126720 | 180.016378 |
| transformer | 2133748 | 28381 | 58124288 | 180.000620 |

Broker physical GPU 2 only; lease and terminal exit 0 retained. Interpreter /home/erikg/emender/.venv/bin/python only. No Frontier, Slurm, WorksGood, subagents or checkpoint production. Four CPU tests pass, including dense gradient flow through the actual projection/hook, zero-input equivalence and hook cleanup. Independent validator checks 56 loss/denominator pairs, prediction/protocol/source/index hashes, dense feature target probabilities, feature invariants and budget accounting. Program validator audits pinned committed receipts.

Residual risks: one seed/corpus; corpus-wide context selection and forced/fork features are transductive despite train-prefix counts. Self-containing training distributions expose no zero-target examples. Historical dense controls reused, not precision/initialization/active-parameter matched reruns. Extra active input projection adds 66,560 parameters; cold compilation and unequal steps affect wall-clock comparisons. No fixed-step or causal selector ablation. Input conditioning prevents the specific mixture-gate bypass, not shortcut learning or OOD degradation. P2 is refuted, so “starvation-proof” must not be read as an empirical OOD guarantee. P4/P5 untested. No resilient DiLoCo behavior changed; requirement IDs inapplicable.

Manual notes: requested contact_supervisor tool absent from active tool catalog; no supervisor transport available. No substitute orchestration service used. Unrelated pre-existing worktree changes preserved. No .pt files staged.
