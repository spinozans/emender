# SF-DiLoCo P7 Island-Count Scaling Analysis

Task: `sf-diloco-p7`

Inputs:

- P5 predeclared rule: `docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md`
- P6 launcher/analyzer: `scripts/launch_sf_diloco_p5_island_scaling.sh`, `scripts/analyze_sf_diloco_p5.py`
- P6 artifacts: `/mnt/nvme1n1/erikg/sf_diloco_p5_island_scaling/{summary.json,paired_by_w.json,scaling_decision.json}`

## Bottom Line

Under the frozen P5 decision rule, the 8-GPU island-count stress does **not**
show that outer `sfsgd` export=`y` is more robust as island count grows.

Observed evidence:

- W=8 is an `avg_win` on final held-out BPB: mean paired delta
  `sfsgd_y - avg = +0.010617` BPB, 95% CI `[+0.005179, +0.016054]`,
  6/6 paired deltas positive.
- W=2 and W=4 are inconclusive, not `sfsgd_y` wins.
- The fitted trend of mean delta versus `log2(W)` is effectively flat
  (`-7.37e-17` BPB per doubling), not the predeclared `sfsgd_y` robustness
  threshold of at least `-0.005` BPB per doubling.
- Shock/recovery at W=8 favors `sfsgd_y` on average jump size and recovery
  burden, but that stability separation is not enough to overturn the primary
  BPB result under the P5 rule.

System-level decision:

`local_evidence_insufficient_for_thousands_of_islands`.

Practical recommendation:

- Current deployment recipe: keep plain `avg` as the conservative default for
  the next recipe, because it wins the largest measured local W point and there
  is no local downward trend in favor of `sfsgd_y`.
- Research follow-up hypothesis: keep `sfsgd_y` as a scale-out canary candidate
  when the test objective is merge-shock damping, resume geometry, and long-run
  stability at true multi-node island counts. Do not claim it is better for
  hundreds/thousands of islands from this 8-GPU result.

## Coverage

The P6 matrix planned 36 runs and produced 36 eligible runs. No completed arm was
dropped and no pair is missing.

| W | seeds | avg eligible | sfsgd_y eligible | missing/failed |
|---:|---|---:|---:|---|
| 2 | 7000-7005 | 6 | 6 | none |
| 4 | 7000-7005 | 6 | 6 | none |
| 8 | 7000-7005 | 6 | 6 | none |

## Frozen Rule Applied

Primary endpoint:

`d_{W,i} = BPB_{W,i}(sfsgd_y) - BPB_{W,i}(avg)`.

Negative deltas favor `sfsgd_y`; positive deltas favor plain averaging. The P5
tie band is `epsilon = 0.005` BPB.

| W | n | mean delta BPB | sd delta | 95% CI | signs neg/pos/zero | dz | P5 per-W decision |
|---:|---:|---:|---:|---|---|---:|---|
| 2 | 6 | +0.010617 | 0.016394 | [-0.006591, +0.027824] | 1/5/0 | +0.648 | inconclusive |
| 4 | 6 | +0.005400 | 0.008298 | [-0.003309, +0.014109] | 2/4/0 | +0.651 | inconclusive |
| 8 | 6 | +0.010617 | 0.005181 | [+0.005179, +0.016054] | 0/6/0 | +2.049 | avg_win |

Scaling rule checks:

- `avg` safe across local W scaling: not declared, because W=2 and W=4 are
  inconclusive and the analyzer's shock trend check flags avg shock as worse
  with W.
- `sfsgd_y` robust to increasing island count: not declared, because W=8 is an
  `avg_win`, `mean_delta_8` is positive, and the fitted trend is not negative by
  at least `0.005` BPB per doubling.
- practical tie across local W scaling: not declared, because W=8 is outside
  the tie rule and is an `avg_win`.
- local evidence insufficient for thousands of islands: declared by the frozen
  rule because there are inconclusive W points and no meaningful local trend in
  favor of `sfsgd_y`.

## Aggregate Metrics By World Size

Held-out BPB and final curve BPB are lower-is-better. Shock columns are measured
from merge-local train-loss jumps. `max_recovery` is in local steps after a
positive merge jump; `unrecovered` counts positive jumps not recovered by the end
of the run.

| W | arm | n | heldout mean | heldout sd | curve1500 mean | max_jump mean | max_jump max | mean_jump mean | mean_pos_jump mean | max_recovery max | unrecovered sum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | avg | 6 | 2.149217 | 0.017384 | 2.149220 | 0.182967 | 0.284900 | -0.017336 | 0.111523 | 125 | 4 |
| 2 | sfsgd_y | 6 | 2.159833 | 0.018255 | 2.159843 | 0.195000 | 0.329400 | -0.000847 | 0.138666 | 425 | 3 |
| 4 | avg | 6 | 2.112467 | 0.004522 | 2.112465 | 0.181100 | 0.240400 | +0.003436 | 0.120590 | 200 | 4 |
| 4 | sfsgd_y | 6 | 2.117867 | 0.006883 | 2.117878 | 0.189633 | 0.410900 | -0.035172 | 0.117374 | 250 | 2 |
| 8 | avg | 6 | 2.099700 | 0.010601 | 2.099696 | 0.325933 | 0.419900 | +0.123903 | 0.193200 | 225 | 7 |
| 8 | sfsgd_y | 6 | 2.110317 | 0.007545 | 2.110324 | 0.298800 | 0.369000 | +0.062342 | 0.171371 | 175 | 6 |

Variance readout:

- Final held-out BPB variance does not show a consistent `sfsgd_y` advantage:
  `sfsgd_y` has slightly higher sd at W=2/W=4 and lower sd at W=8.
- Shock variance improves for `sfsgd_y` at W=8 (`max_jump sd` 0.061475 versus
  0.072859 for avg), but W=4 has a large `sfsgd_y` outlier
  (`max_jump = 0.410900`, seed 7001).
- The paired BPB delta variance narrows with W: sd 0.016394 at W=2, 0.008298 at
  W=4, 0.005181 at W=8. At W=8 the signs are uniformly positive for avg.

## Paired Replicate Metrics

| W | seed | avg BPB | sfsgd_y BPB | delta BPB | delta curve BPB | avg max_jump | sf max_jump | delta max_jump | avg max_recovery | sf max_recovery | avg unrecovered | sf unrecovered |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7000 | 2.128100 | 2.136500 | +0.008400 | +0.008426 | 0.196400 | 0.095500 | -0.100900 | 50 | 50 | 0 | 0 |
| 2 | 7001 | 2.144800 | 2.176700 | +0.031900 | +0.031817 | 0.284900 | 0.108500 | -0.176400 | 125 | 75 | 1 | 1 |
| 2 | 7002 | 2.174800 | 2.158700 | -0.016100 | -0.016077 | 0.032000 | 0.306500 | +0.274500 | 25 | 150 | 0 | 0 |
| 2 | 7003 | 2.164300 | 2.185700 | +0.021400 | +0.021473 | 0.144400 | 0.116300 | -0.028100 | 50 | 25 | 1 | 1 |
| 2 | 7004 | 2.136700 | 2.150800 | +0.014100 | +0.014090 | 0.190400 | 0.329400 | +0.139000 | 25 | 425 | 1 | 0 |
| 2 | 7005 | 2.146600 | 2.150600 | +0.004000 | +0.004010 | 0.249700 | 0.213800 | -0.035900 | 50 | 100 | 1 | 1 |
| 4 | 7000 | 2.115400 | 2.111300 | -0.004100 | -0.004157 | 0.168600 | 0.155800 | -0.012800 | 175 | 75 | 1 | 0 |
| 4 | 7001 | 2.113400 | 2.129900 | +0.016500 | +0.016529 | 0.164500 | 0.410900 | +0.246400 | 50 | 250 | 1 | 1 |
| 4 | 7002 | 2.112300 | 2.121700 | +0.009400 | +0.009349 | 0.150300 | 0.035200 | -0.115100 | 200 | 50 | 0 | 0 |
| 4 | 7003 | 2.118900 | 2.114300 | -0.004600 | -0.004576 | 0.240400 | 0.078000 | -0.162400 | 100 | 50 | 1 | 0 |
| 4 | 7004 | 2.107400 | 2.116600 | +0.009200 | +0.009286 | 0.179200 | 0.136500 | -0.042700 | 175 | 50 | 0 | 1 |
| 4 | 7005 | 2.107400 | 2.113400 | +0.006000 | +0.006042 | 0.183600 | 0.321400 | +0.137800 | 50 | 200 | 1 | 0 |
| 8 | 7000 | 2.093700 | 2.109400 | +0.015700 | +0.015623 | 0.335500 | 0.369000 | +0.033500 | 125 | 100 | 1 | 1 |
| 8 | 7001 | 2.119500 | 2.124300 | +0.004800 | +0.004775 | 0.366200 | 0.201400 | -0.164800 | 225 | 175 | 1 | 1 |
| 8 | 7002 | 2.093900 | 2.110900 | +0.017000 | +0.017022 | 0.296500 | 0.301600 | +0.005100 | 75 | 175 | 2 | 1 |
| 8 | 7003 | 2.100200 | 2.109000 | +0.008800 | +0.008819 | 0.203200 | 0.270700 | +0.067500 | 125 | 100 | 1 | 1 |
| 8 | 7004 | 2.101100 | 2.106400 | +0.005300 | +0.005363 | 0.419900 | 0.359000 | -0.060900 | 225 | 50 | 1 | 1 |
| 8 | 7005 | 2.089800 | 2.101900 | +0.012100 | +0.012162 | 0.334300 | 0.291100 | -0.043200 | 150 | 175 | 1 | 1 |

## Shock And Recovery Deltas

The table reports `sfsgd_y - avg`. Negative max-jump, mean-jump, recovery, or
unrecovered deltas favor `sfsgd_y` stability.

| W | delta max_jump mean | delta max_jump sd | delta mean_jump mean | delta mean_pos_jump mean | delta max_recovery mean | delta unrecovered total |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | +0.012033 | 0.165637 | +0.016489 | +0.027142 | +83.3 | -1 |
| 4 | +0.008533 | 0.155482 | -0.038608 | -0.003217 | -12.5 | -2 |
| 8 | -0.027133 | 0.082488 | -0.061561 | -0.021829 | -25.0 | -1 |

Interpretation:

- At W=8, `sfsgd_y` does damp average shock relative to avg: lower mean
  max-jump, lower mean jump, lower mean positive jump, shorter maximum recovery
  burden, and one fewer unrecovered positive jump.
- At W=2 and W=4, shock evidence is mixed. W=2 favors avg on average shock and
  recovery despite one fewer unrecovered jump for `sfsgd_y`. W=4 has smaller
  mean/positive jumps for `sfsgd_y`, but the seed-7001 `sfsgd_y` outlier raises
  max-jump variance and max recovery.
- The observed W=8 stability signal is a reason to keep `sfsgd_y` in the
  research queue, not a reason to move the primary deployment recipe away from
  avg under the frozen rule.

## Thousands-Of-Islands Statement

The direct measurement here is only W=2, W=4, and W=8 on one 8-GPU stress
matrix. Any claim about hundreds or thousands of independent islands is
extrapolation.

What can be inferred:

- There is no measured local trend indicating that plain avg's W=8 BPB advantage
  is merely a small-island parameterization artifact.
- There is a measured W=8 shock/stability hint that `sfsgd_y` may damp merge
  transients as W grows, but the same local matrix does not show a held-out BPB
  win and does not measure true multi-node/network scale-out effects.
- Hundreds/thousands of islands introduce additional unmeasured dynamics:
  endpoint variance growth, inter-node communication, resume/checkpoint
  geometry, long-horizon drift after many more merge rounds, and Frontier queue
  constraints. These are hypotheses for a scale test, not conclusions from P7.

## Recommendation

Next recipe:

- Use plain `avg` for the next deployment recipe.
- Keep the P5/P6 parameters fixed for comparability unless a separate task
  explicitly reopens tuning: ScheduleFree inner optimizer, K as in P5/P6, and
  no outer `sfsgd` beta/LR sweep folded into this decision.
- Monitor held-out BPB, merge-local shock, recovery steps, unrecovered positive
  jumps, and resume integrity in the next run.

Next scale test:

- Run a true-island follow-up at larger W before making claims about
  hundreds/thousands of islands. The minimum useful ladder is 16 islands and 32
  islands if hardware permits; a Frontier-facing canary should keep avg as the
  primary arm and include `sfsgd_y` as a paired stability arm.
- Treat the research hypothesis as: `sfsgd_y` may be worth using if it reduces
  merge shock, improves resume geometry, or stabilizes long-horizon drift at
  true scale even when short local held-out BPB is tied or slightly worse.
- Promote `sfsgd_y` only if a larger true-island run shows no held-out BPB
  penalty inside the P5 epsilon band and a consistent shock/recovery advantage
  across W.
