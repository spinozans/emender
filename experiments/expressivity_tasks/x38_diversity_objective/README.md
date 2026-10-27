# X38 diversity objective

Arm (e) adds a signed cosine read penalty to X37(c), with unchanged lane
operations. RECEIPTS.md is the pretraining registration and exact objective.
The only imported upstream modules are read-only. No WG or Frontier jobs.

Run sequentially from the repository root:

```bash
experiments/expressivity_tasks/x38_diversity_objective/run_leased.sh parity -q --junitxml=experiments/expressivity_tasks/x38_diversity_objective/receipts/oracle.xml
experiments/expressivity_tasks/x38_diversity_objective/run_leased.sh test -q --junitxml=experiments/expressivity_tasks/x38_diversity_objective/receipts/mechanism.xml
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x38_diversity_objective.cli seal-gate
experiments/expressivity_tasks/x38_diversity_objective/run_leased.sh all
```

`all` sweeps four lambdas through611 steps, selects by the registered rule,
continues the selected optimizer to3052 steps, and runs final probes.
`train --arm 0.01` is a prespecified primary run through3052 steps; it does
not conduct or pretend to conduct a sweep. Each lambda has separate resumable
runs/ and receipts. Source/data fingerprints must match on resume.
Checkpoints are local only. Do not add them to git.

Read vectors are measured before read_out. Nonzero pair coverage accompanies
all correlations. Slot JS uses exact pre-write soft-strength mass from an
evaluation-only one-token shadow scan. It asserts read parity against the
full optimized scan, accumulates mass without usage decay, and normalizes
per head. It compares physical slots, not semantic concepts. It changes
neither training scans nor policy. Full distributions are retained.

Historical X34 control is B8 and4096 tokens shorter than X38. Treat the
quality comparison accordingly. Widened base parameter counts include frozen
geometry tables; this does not create a newly trained quality baseline.

Sealing also archives every imported experiment's top-level Python sources.
The broker wrapper trains against the byte-verified archive, with Triton
source lookup pinned. This retains X37 dependencies and concurrent X36 edits
without modifying or committing upstream files. Own X38 source drift still
fails closed. `receipts/resume_pinned.py --check` checks this loader without GPU.

For the delivered snapshot (including when X37 has not landed separately),
use the archived imports:

```bash
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x38_diversity_objective/receipts/resume_pinned.py --check
experiments/expressivity_tasks/x38_diversity_objective/run_leased.sh --test-snapshot
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x38_diversity_objective/receipts/resume_pinned.py summarize
```

The initial fresh-gate commands require the upstream experiment source tree.
The delivered archive supplies those dependencies without changing that tree.
