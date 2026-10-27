# X35 preregistration

Registered before X35 experiments (2026-10-27).

- (a) C=64, S=2944: complete pure-chunk forward >=3x X31 sequential.
- (b) Measure ladder/MQAR chunk pure fraction; prediction: a-runs >90%.
- (c) Complete mixed-chunk ladder training loop >=2x X31.
- (d) Zero bitwise trace, functional-state and gradient parity failures.

Do not substitute mutation-only timings for (a), inference for (c), or
1e-5 closeness for (d). Distinguish append-only mutation candidates from
certified entry-state-independent chunks. Density uses recorded X31 traces,
C=64, seed=0 initialized F12g projections, no learning. These are initialization
measurements, not trained-X31 checkpoint measurements. Historical JSONs contain
metrics, not model weights. Exclude incomplete chunks and report excluded tokens.
Separate all-a/all-b windows from whole-data statistics; padding is separate.

## Outcome

See REPORT.md and density.json. Targets (a), (c), and full (d) are NOT RUN:
the requested chunk-entry routing formulation has counterexamples before
performance engineering. Counterexample tests are successful falsification
experiments, not parity passes of an implemented v3 cell. No throughput claim.
