# C14-B1 PB verdicts: store door versus passive channel

The primary B1-v2 paired evaluation is complete, already retained in commit
4ad7630c before this harvest. It was not rerun. All 66 table rows agree exactly
with the complete resume sums and frozen-arm denominators; the selected
checkpoint SHA256 is independently rechecked in
`receipts/c14_b_v2/pb_harvest_validation.json`.

| Prediction | B1-v2 final-test NLL | Registered passive reference | Delta |
|---|---:|---:|---:|
| PB1 total | 1.584700014 | 1.509260010 | +0.075440004 |
| PB2 supported target | 0.856875769 | 0.811639725 | +0.045236044 |
| PB2 zero target | 3.679494742 | 3.517122903 | +0.162371839 |
| PB2 fork | 2.085738874 | 2.010901269 | +0.074837605 |

The passive input channel wins all registered comparisons at matched active
time. B1-v2 beats dense 2.276133549 by 0.691433535. Total final-test denominator
is 214,097,022; calibration remains separate. Full support/empty,
nonempty-zero-target, reset/interior and overlapping strata sums and counts are
in `receipts/c14_b_v2/eval_table.json`.

B1-v2 completed 35,419 steps / 71,677,042 real byte targets in 2700.032370 active
seconds (13.118 steps/s), versus passive v2's 179,162,804 targets: 2.499584x
exposure at the same time budget. This confounds an equal-exposure architecture
conclusion. The byte-identical orchestration optimization in 8b233bc2 measures
17.351% higher compute steps/s on repeated batches, not a new trained arm or an
intrinsic 2.5x store ceiling. Initialization, lease wait, checkpoint stalls and
wall time remain separately receipted.

PB3: all four training heads reach but stay within their shared 32/64 budgets;
array and stream ownership, sampled/peak/final occupancy, all operation counts,
stale retirements, insertion requests/refusals and capacity/duplicate merges
are in the full compressed training/verdict receipts. Every head has nonzero
CORRECT, MERGE and stale retirements. Training read opportunities are
42,700,419 / 42,044,952 / 45,555,166 / 45,885,395; found/read equal these
respective denominators (hit rates 1.0). This proves numerical findability,
not causal usefulness; CORRECT combines stream and array events.

PB4 fails graceful degradation: unplugged NLL 2.430144516 versus dense
2.276133549, absolute delta 0.154010968 > 0.02. Reset positions 1–16 have
unplugged NLL 2.462716799. Independent populated native stores empty after
128/513 ticks with zero residual array occupancy. Held-out windows start empty,
o cross-document warmup occurs, and off mode inserts zero array answers.

The first attempt is separate engineering evidence: 237 steps / 479,328 targets
(0.212204% of the registered stream), 0.087607 steps/s. Its complete final NLL
is **2.206163868216956**, improving dense by 0.069969680481, despite starvation.
The task's 2.204625707576865 is approximately the last printed progress value
at shard 26753; the complete table/resume/verdict agree on the value above at
shard 26807. It is not the primary PB1 verdict. Original bytes and all evaluation
resume generations are preserved, with hashes in
`receipts/c14_b/FIRST_ATTEMPT_HARVEST.json`.

PB1 is nonzero, so append-only and unbudgeted conditional ablations remain
required. The existing append-only evaluation completed in its source-bound worktree
on leased card 2: 3,431,242 windows / 434,204,084 targets, exit 0. Final-test
on/off NLL is 2.246020115 / 2.710095493; on is +0.661320101 worse than B1-v2.
It trained 37,254 steps / 75,394,452 targets in 2700.045113 active seconds,
so exposure remains unequal. All strata, lifecycle telemetry and terminal
evidence are harvested under `receipts/c14_b_v2_append_only/`. Unbudgeted completed at **1.716496444** after only 14,455,380 training targets,
with full terminal coverage and all 66 rows verified. No ablation replaces the selected B1-v2.
The triangle is complete: budgeted < unbudgeted < append-only. The roughly 5x
exposure deficit cuts in favor of unbudgeted data efficiency; budget pressure
itself cannot be cleanly separated from lifecycle value and exposure. The
registered causal budget-pressure comparison remains unresolved. See
[RESULT_C14_ABLATION_TRIANGLE.md](RESULT_C14_ABLATION_TRIANGLE.md) for all
strata, limits, and a specified matched-exposure arm (not run).
One seed and unequal exposure limit inference.
No frozen dense/passive arm, production kernel, checkpoint, union service,
card-1 search, sxgc or read-only corpus was modified by this harvest.
