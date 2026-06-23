# SF-DiLoCo Scale-Test Rationale

Task: `document-sf-diloco`
Date: 2026-06-23

Related artifacts:

- P5 design: `docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md`
- P6 launcher/analyzer: `scripts/launch_sf_diloco_p5_island_scaling.sh`,
  `scripts/analyze_sf_diloco_p5.py`
- P7 analysis: `docs/experiments/SF_DILOCO_P7_ISLAND_SCALING_ANALYSIS.md`

## Bottom Line

The current default recipe should be:

```text
ScheduleFree inner optimizer + plain avg DiLoCo outer
```

That is a deployment recommendation from the measured P4-P7 evidence, not a
proof that outer `sfsgd` export=`y` is uninteresting. The scale research
candidate remains:

```text
ScheduleFree inner optimizer + outer sfsgd, export_basis=y
```

P4 found both `avg` and `sfsgd_y` plausible, while fixed outer momentum and
`sfsgd` export=`x` were not worth carrying forward. P5 froze the comparison as
an island-count stress matrix over W=2/4/8. P6 executed that matrix, and P7
analyzed all 36 eligible runs under the frozen P5 rule.

The P7 readout is narrow and important:

- Held-out BPB favors `avg` in the local W=2/4/8 evidence, with W=8 declared an
  `avg_win` under the P5 rule.
- The W trend does not show a local held-out-BPB scaling advantage for
  `sfsgd_y`.
- At W=8, `sfsgd_y` shows a merge-shock/recovery hint: smaller average jump,
  smaller positive jump, shorter recovery burden, and one fewer unrecovered
  positive jump than `avg`.

Therefore the right conclusion is not "`sfsgd_y` is dead." The right conclusion
is: `avg` is the current recipe; `sfsgd_y` remains a scale/stability hypothesis
that must be tested at larger true island counts before it is accepted or
discarded for the thousands-of-islands risk.

## Why Test Both At True Scale?

The local P7 matrix measured W=2, W=4, and W=8 on one 8-GPU box. That is useful
for screening optimizer choices, but it does not reproduce the dynamics of
hundreds or thousands of independent islands. Larger true island counts add
risks that are not present, or are only weakly present, at W<=8:

- larger endpoint variance at merge boundaries,
- longer and more failure-prone inter-node communication,
- checkpoint/resume geometry across many workers,
- accumulated drift over many more merge rounds,
- outlier islands and nonuniform progress,
- queue/runtime constraints that change how canaries are resumed and extended.

`avg` should be included because it is the measured winner on held-out BPB in
the current local matrix and is the conservative deployment baseline. A scale
test without `avg` would not answer whether the current default remains stable
as W grows.

`sfsgd_y` should be included because its plausible value is not "better short
local BPB at W<=8"; P7 did not show that. Its plausible value is damping merge
shock or improving long-horizon stability when island count and resume geometry
become the dominant risks. A scale test without `sfsgd_y` would discard the one
remaining outer-optimizer candidate before measuring the regime it was meant to
help.

## Decision Boundary

Use this boundary when interpreting future runs:

- **Deployment recommendation:** choose the arm with the best measured
  held-out BPB and no stability failure at the scale actually being used. Today
  that is `avg`.
- **Research hypothesis:** keep an arm alive when it has a credible mechanism
  and a measured stability signal in a smaller proxy, even if it is not the
  current held-out-BPB winner. Today that is `sfsgd_y`.

Promotion criteria for `sfsgd_y` should be stricter than "shock looks nicer."
It should show no held-out BPB penalty beyond the P5 epsilon/tie band at larger
true W, plus a consistent improvement in merge shock, recovery, unrecovered
positive jumps, resume integrity, or long-horizon drift. If it only improves
shock while leaving a real BPB penalty, it remains a diagnostic or future tuning
candidate, not the default recipe.

## From Scratch Vs Mature Checkpoint

Future tests do not all need to start from a well-trained checkpoint. Scratch
and checkpoint starts answer different questions.

A **from-scratch** scale test answers recipe and trajectory questions:

- whether the optimizer parameterization works across the whole training
  trajectory,
- whether early optimizer transients are acceptable,
- whether the run reaches good held-out BPB without relying on a pre-existing
  basin,
- whether one arm silently trades early stability for worse final quality.

What a from-scratch test can prove: the recipe trains end-to-end under the tested
W, K, LR, beta/export, data, and checkpoint policy.

What it cannot prove cheaply: that a mature model at much lower loss survives a
large jump in island count, checkpoint/resume topology, and long-horizon
multi-merge drift. It may spend most of the budget measuring early training
rather than the mature-model stability failure mode.

A **well-trained-checkpoint warm-start** scale test answers scale and stability
questions cheaply:

- whether a mature model survives larger island count,
- whether merge shock appears when many mature endpoints diverge and rejoin,
- whether checkpoint/resume geometry is coherent for `avg` and `sfsgd_y`,
- whether long-horizon drift appears after repeated large-W merges,
- whether the scale runtime can checkpoint, resume, and continue cleanly.

What a checkpoint canary can prove: the tested arm can preserve and extend an
already-good model under the larger-W system geometry.

What it cannot prove: that the same arm is the best from-scratch training
recipe, or that it reaches the same final BPB when used for the entire training
trajectory.

For thousands-of-islands risk, both tests are useful. If compute is constrained,
the next canary should start from a mature checkpoint because it isolates the
specific risk that P7 leaves open: larger island-count stability. The result
must be labeled as a mature-checkpoint scale/stability result, not as proof of
from-scratch training quality.

## Proposed Ladder Under Compute Constraints

1. **Mature-checkpoint canary at larger true W.** Start from a well-trained
   checkpoint, keep the P5/P6 recipe fixed where possible, and run paired `avg`
   and `sfsgd_y` arms at the next available true island count, such as W=16 or
   W=32. Measure held-out BPB delta, merge-local shock, recovery steps,
   unrecovered positive jumps, sync cost, checkpoint contents, and fresh-process
   resume.
2. **Advance only if the canary is clean.** Continue the ladder only if both
   arms launch, merge, checkpoint, resume, and keep finite loss. `sfsgd_y`
   remains interesting only if its held-out BPB is inside the tie band while its
   shock/recovery or drift behavior is consistently better than `avg`.
3. **Repeat at a larger W before extrapolating.** A single W=16 or W=32 canary
   is not evidence for thousands of islands. Use at least one additional larger
   W point if hardware permits, and keep the language as risk assessment until
   true large-W data exists.
4. **Run from-scratch confirmation if the checkpoint canary is promising.** If
   `sfsgd_y` survives the mature-checkpoint canary with no BPB penalty and a
   clear stability advantage, schedule a from-scratch paired confirmation. That
   run answers whether the candidate is a full training recipe rather than only
   a mature-model stability aid.

Until that ladder changes the evidence, use `avg` as the default recipe and
carry `sfsgd_y` as a scale/stability research arm.
