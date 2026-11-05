# C14-B1 unbudgeted control: training complete, evaluation pending

Frozen-source training completed on leased card 2: **7,142 steps, 14,455,380
byte targets, 2700.288256 active seconds**. Source commit c190956d3f970efe18e4377d6d80ac2fab83e5ba
matches every control source/test gate, including replay SHA256
58a8acb855a9659779904665ab510b6ddf08c734302333d31b74fe0eed79bbec.
The registered seed/batch/context/BF16/optimizer/dropout/reset protocol is unchanged.
Training occupancy peaks by head are 81/110/81/96 at cap 257, with zero capacity
merges. At most 256 allocations per independent 128-position window prove the
cap nonbinding. Full telemetry and curves are retained in training.json.gz.

Evaluation was gracefully paused after 80 shards, 10,240 windows and 1,294,694
targets. The unchanged evaluator measured ~1.92 seconds/shard, projecting
~14.27 hours for the full frozen set. Exit 0 is a pause, not acceptance.
Final NLL and the budget-pressure ablation verdict remain pending. No numerical
failure occurred. The partial final-test denominator is not a final result.

| Arm | Final-test NLL | Training targets |
|---|---:|---:|
| Full-ops-budgeted B1-v2 | 1.584700014 | 71,677,042 |
| Append-only | 2.246020115 | 75,394,452 |
| Unbudgeted | Pending | 14,455,380 |

Source bindings, compressed full training/eval states, terminal receipts and
external archive/checkpoint hashes are retained in receipts/c14_b_v2_unbudgeted.
Run resume_eval.sh from that directory to recreate the pinned source worktree
and resume evaluation; never retrain the control or change source hashes.
Full evaluation, all 66 strata/mode rows, empirical sampled occupancy, final
three-arm signs, claim bindings and the requested completion commit remain due.
One seed and unequal exposure prevent an isolated causal budget-pressure claim.

Manual notes: contact_supervisor is absent from the enabled tool catalog;
no supervisor transport was available. No WorksGood service was used. Frozen
arms, union service, card-1 search, sxgc and /mnt/nvme2n1 were not modified.
No resilient DiLoCo behavior changed/tested; gap-matrix IDs do not apply.
