# C7b preregistration — routing constrained residual training

Registered before training. Primary emender, secondary transformer; seed 4301,
identical C7a/C6 corpus, split, backbone/parameter targeting, batch 16, context
128, AdamW lr .0003 decay .01 clipping 1, BF16/float32 probabilities, 180 seconds
per arm including loop synchronization; setup/index reconstruction excluded.

Fixed arm: no training gate parameters. On nonempty support L=.5 CE(y,p)+.5
CE(exact train-prefix count distribution,p). Empty support L=CE(y,p). Both
terms directly train the neural softmax; routing cannot attenuate true-target
weight below .5. Index teacher is reconstructed and checked against C6 ratios.
Floor arm: C7a mixture likelihood with neural g=.3+.7 sigmoid(gate features);
empty support g=1. A probability-weight floor does not guarantee a .3 gradient
floor: responsibility still depends on model target likelihood. This is a
secondary mechanism comparison, not the independent dense-loss guarantee.

Inference unchanged C3: separately fit feature gate on calibration covered forks,
freeze before final test, walk covered forced positions, neural at empty support.
Reuse C1/C3 dense and C7a starved predictions without retraining. Decisive question:
does fixed residual beat dense fused NLL? Report signed fixed-minus-dense deltas
SEPARATELY overall, fork, and empty-support/OOD (9049 final positions, including
2709 forks). Strict column win requires negative delta; no combined win inferred
from disagreeing signs. Also report floor comparisons and fitted calibration
coefficients plus pre-walk neural weights by support/context/count strata.
One seed/corpus, corpus-selected transductive intervals; no causal claim.
No .pt artifacts; broker only physical 2/4; 7 forbidden. Four sequential arms.

Training source SHA256 `69eeb9112b02669c5c5b2157a8a6330aa5d6bc5f4dcfb4be5f890735a152c707`.
