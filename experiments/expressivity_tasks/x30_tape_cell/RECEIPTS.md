# X30 pre-registration

Registered before any X30 test, training, or timing measurement, 2026-10-26.

1. Independent eager scan versus timestamp store: exact integral routing,
   fp32 outputs/state/gradients atol=rtol=1e-5, multiple seeds and budgets,
   age ties, pop, decay, reuse, cap, and replay with perturbed thresholds.
2. MQAR T=128, vocabulary 65, 64 classes, B=736, AdamW lr=1e-4,
   weight_decay=1e-4, effective batch 64; >=.99 mean per-class recall
   on the frozen validation pool by 5,000 updates. Seed 0.
3. Ladder k<=128; validation-only selection; evaluate frozen bin pools
   once: ID>=.99, 2x/4x>=.90, 9x<=.55 at B=736.
4. H=4 and H=16: replicated independent heads, identical per-head
   outputs and accuracy; measure timing, do not infer speed from FLOPs.
5. Maintenance-only time versus naive scans at S=736 and 11776:
   target >=50x at 11776, amortized O(1) allocation/retirement per token.
   Report content reads separately (O(S*K)); Python/device synchronization,
   snapshots and backward copies must be disclosed, not hidden in read time.

No target is considered met without a recorded measurement. Native published
results are context, not X30 evidence. X28E files must remain untouched.

## Measurements and executable evidence

- `tests/`: CPU op/parity/replay/gradient and exact IEEE decay tests.
- `runs/parity_cpu.json`: exact traces in eight seeded stores through B=11776,
  H=16; functional state and gradient tolerance 1e-5; usage EMA excluded.
- `runs/stage1.json`: full frozen validation pool, H=1, recall 1.000 at
  update 500, B=736, effective batch 64, AdamW lr 1e-4.
- `runs/cost_cpu.json`: isolated inert full-store maintenance 423.0x at 11776;
  complete forward 1.29x at that budget. Timer scopes are explicit.

Analytic power's one-token death-boundary discrepancy was found during
implementation. Exact scalar IEEE interval jumps and a shared discrete
exponential curve replaced it; the adversarial boundary and long-horizon
reference tests now pass. Routing is still recorded and replayed.

Targets for the ladder and complete H=4/H=16 training remain qualification
requirements. A passing isolated maintenance timer is not substituted for
those results. See REPORT.md for detailed limitations and run provenance.

- CPU pytest final run: 51 passed (7.63 s), including the pop-free / reuse /
  decay-free same-slot edge and padding-loss/gradient equivalence.
- `runs/parity_gpu.json`: eight CUDA cases, exact traces, maximum floating
  absolute difference 2.33e-10 at the declared 1e-5 tolerances.
- `runs/cost_gpu.json`: isolated host maintenance tick / CUDA eager baseline
  ratio 676.7x at 11776; complete forward ratio 1.39x. These scopes differ
  and must not be conflated.
- `runs/current_stage1.json`: final-source frozen-checkpoint revalidation,
  full 20,000-example pool, recall 1.000, with source/checkpoint hashes.
- `runs/ladder.json`: the 500-update H=1 pilot selected on validation .786;
  bins ID=.791016, 2x=.597656, 4x=.556641, 9x=.332031. Gate FAILED, exit 2.
  This does not qualify the phase diagram or constitute a full-budget failure.

Full H=4/H=16 stage-1 training has not been performed. Replicated-head output,
trace, pressure, and gradient independence tests pass, but this is partial
head-scaling evidence only. All GPU measurements used broker leases on card 0;
card 7 was excluded. REPORT.md distinguishes launch-time training versions
from final-source frozen-checkpoint revalidation.

Final-source frozen ladder re-evaluation reproduced the pilot scores exactly;
see `runs/current_ladder.json`. Reference tree hashes at both endpoints match;
source hashes and an explicit incomplete-qualification summary are retained in
`runs/source_manifest.json` and `runs/validation_summary.json`.
