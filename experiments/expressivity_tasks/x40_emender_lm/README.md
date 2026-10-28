# X40 EmenderLM

Six residual pure memory blocks, width256, MLP1024, four semantic-key heads
per block. There is no attention implementation in this model. Frozen X39
geometry/token features precede a learned query projection; shared contextual
W_in reaches every head. X40 owns its error-plus-match enable wire; copied
X39 forward/kernel scaffolding dispatches through the unchanged X36 replay.

Run CPU structure checks:

```sh
/home/erikg/emender/.venv/bin/python -m pytest experiments/expressivity_tasks/x40_emender_lm/tests -q
```

Fresh production-shape policy parity, parameter counts and B64 smoke:

```sh
experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
```

Full ordered run (parity and smoke always repeat):

```sh
experiments/expressivity_tasks/x40_emender_lm/run_leased.sh -m experiments.expressivity_tasks.x40_emender_lm.runner --train
```

The full run writes JSON curves/evaluation/telemetry and performs stage3
fine-tuning in the same process. It does not write model checkpoints. A crash
loses model/optimizer state and cannot resume; an existing curve prevents
accidental reuse. No GPU run uses cards2/3/4/7. No WG services or Frontier work.

Read RECEIPTS.md before interpreting outcomes. CPU structure checks are not
oracle certification. The matched widened base is untrained; existing B64
control quality is the comparison denominator. Historical transformer depth
claims require a matched fine-tuned receipt; X28E native controls are not
transformer substitutes. Depth fine-tuning uses new scalar-token embedding
and binary head, S2944/head, original stage3 30000-update budget, validation
selection every500, then immutable ID/2x/4x/9x pools. This is computationally
substantial despite the requested brief-probe framing; no shorter unregistered
budget silently substitutes for X28E.
