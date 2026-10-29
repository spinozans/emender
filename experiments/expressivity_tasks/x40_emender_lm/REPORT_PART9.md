# X40 part 9 — diagnostic receipt; 1x acceptance failed

**Requested optimization/evidence package is incomplete.** No production change
or speedup is accepted. See [decision memo](receipts/part9/decision.md) and
[machine-readable gate](receipts/part9/gate.json).

Fresh pressure-inclusive production certification passes unchanged at T512 and
T2048. A B1 single-block audit at both scale widths and context lengths observes
four initial scans, four checkpoint-recompute scans, and four reverse calls;
the reverse makes **zero additional native forward calls**. The recomputed scan
already supplies its journal to the reverse. Reverse read score/probability
recomputation remains; removing it needs a new caching/reverse implementation
and exactness/memory measurements. These call counts establish neither sustained
throughput nor impossibility of a future fusion.

The initial T2048 oracle launch failure is preserved. The successful rerun uses
the existing part5 launch-axis adapter, with unchanged floating reference and
unchanged certification assertions. See [raw audit](receipts/part9/call_audit.json).

Part8's historical 1.3B pure/transformer step-time ratios are 1.5854/1.3174 at
T512 B64/B128 and 3.4187/3.4888 at T2048 B64/B128. They are independently
recomputed in [historical ratios](receipts/part9/historical_ratios.json), with
matching envelope metadata retained. No part9 before/after ratio exists.
The 1.2x and 1.0x targets remain unverified expectations; the prior qualified
pilot remains 54,836.10 tok/s against its failed 100k gate.

Missing deliverables: recomputation optimization and bitwise 2M-token curve,
before/after phase profiles and matched throughput, experimental BF16 read
implementation and 5M-token tradeoff evidence, trained-density adjoint
suppression, and a fresh pilot gate. No BF16 adoption recommendation is made.

Validation: both production gates, four call-count assertions, AST parsing and
receipt consistency assertions pass. The task gate remains **failed**. No `.pt`
files, production source changes, protected GPU interference, WG, subagents,
Frontier/Slurm work or resilient DiLoCo changes. Supervisor tool discovery found
no available `contact_supervisor`; no supervisor decision was obtained.
