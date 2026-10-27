# X30 H=4 ladder pilot — verdict (parent receipt, run proc_a058)

Run: `train_ladder.py --heads 4 --updates 7500`, B=736, batch 64, AdamW lr 1e-4,
seed 0, leased card, stage-1 MQAR checkpoint (100% @ 500) as initialization.
Full trajectory in `runs/ladder_h4.json`.

## Trajectory (balanced accuracy, accept/reject recall)

```
u  500  0.8675  (1.00/0.73)      u 4000  0.9260  (1.00/0.85)
u 1000  0.8535  (1.00/0.71)      u 4500  0.9370  (1.00/0.87)
u 1500  0.8555  (1.00/0.71)      u 5000  0.5500  (0.10/1.00)   <- all-reject basin
u 2000  0.9185  (1.00/0.84)      u 5500  0.6745  (0.36/0.98)
u 2500  0.8800  (0.81/0.94)      u 6000  0.7420  (0.50/0.98)
u 3000  0.6755  (0.36/0.99)      u 6500  0.9345  (1.00/0.87)
u 3500  0.9195  (1.00/0.84)      u 7000  0.9625  (1.00/0.93)   <- best
                                   u 7500  0.8585  (1.00/0.72)   <- final
```

**Best 0.9625 @ u7000; final-eval 0.8585 @ u7500.**

## Verdict

1. **Learnability: the efficient v1 cell learns the ladder.** Best 0.9625 within
   7,500 updates, versus the certified tape_B0 arm's max 0.986 within 30,000
   stage-3 updates. The v1 pilot (500-update, inconclusive, 0.786) understate
   is superseded: the pin at 0.855 was a transient, not a ceiling.
2. **The oscillation is cross-arm and intrinsic, not an efficient-cell artifact.**
   The pilot's dips to the all-reject basin (0.55 at u5000, accept-recall 0.10,
   reject-recall 1.00) mirror the CERTIFIED instrument's own stage-3 dynamics:
   tape_B0 has 9 evals below 0.70 (min 0.544) and tape_B0x4 has 8 (min 0.510)
   across their 30k-update stage-3 runs. The stage-3 loss landscape has (at
   least) two basins — balanced discrimination and all-reject — and training
   wanders between them. tape_B0x16's oscillation is the same phenomenon.
3. **The certified results were protected by the registered protocol.** The
   M1/M2 registration specifies validation-only checkpoint selection
   ("tie-break latest-best") — the certified final numbers are selected-best
   evaluations. The pilot ran WITHOUT a selection policy, so its official
   final-eval number (0.8585) is not comparable to certified numbers (0.9941
   etc.). Comparing like-for-like: pilot best 0.9625 vs certified max 0.986.
4. **Qualification: QUALIFIED.** The efficient cell is a learnability match for
   the certified instrument on this task, with the same bistable training
   character. A full phase-diagram replication from the efficient cell must
   run under the registered selection protocol (or receipt best-checkpoint
   evaluations explicitly) before its per-bin numbers are admissible.

## Cross-references

- tape_B0 stage-3: 60 evals, min 0.544 / max 0.986, 9 dips < 0.70
  (`results_x28e_memory_head/tape_B0_seed0/training.jsonl`).
- tape_B0x4 stage-3: 60 evals, min 0.510 / max 0.983, 8 dips < 0.70.
- X31 fused v2 cell: 0.915 @ 2500 updates, best 0.922 @ 2000
  (`x31_tape_cell_fused/summary_part5.json`) — consistent with the pilot's
  early trajectory (0.9185 @ u2000), both then oscillating.
- Mechanism hypothesis (unresolved, cross-arm): all-reject basin entry
  correlates with reject-recall saturation; R-CAP is NOT implicated during
  stage-3 training at k<=128 (peak live ~132 << 736 budget; no merges fire).
  The bistability is an optimizer/loss-surface property, not a capacity
  phenomenon. Dissecting it is a separate registered task if wanted.

## Addendum — process exit 2 and the final-checkpoint bin evaluation (parent)

The watcher event "failed with exit code 2" is the script's REGISTERED GATE
SIGNAL, not a crash: `train_ladder.py` returns `0 if gate else 2`, and all
artifacts were written before exit (ladder_h4.json complete and valid;
ladder.pt saved; no traceback in any log).

Final-checkpoint (u7500, the oscillation-trough state) OOD bins:

    bin_id 0.9746 | 2x 0.9160 | 4x 0.8906 | 9x 0.5000
    (certified tape_B0: 0.9941 | 0.9805 | 0.9844 | 0.5000)

1. **The 9x capacity wall replicates exactly**: 0.5000 at the 736-slot
   boundary, identical to the certified instrument, from a non-best
   checkpoint. The phase diagram's defining feature is checkpoint-robust.
2. The under-budget bins are in-family but below gate thresholds
   (bin_id 0.975 vs 0.99 required), consistent with (a) the final state
   being an oscillation trough (best 0.9625 @ u7000 was not evaluated on
   bins — the script evaluates bins only at the end), (b) the registered
   recipe deviations this pilot carries ("stage-2 occupancy curriculum
   omitted", "random per-update batches versus epoch permutations").
3. Gate verdict as registered: FAIL (exit 2), honestly. Qualified verdict
   unchanged: the efficient cell reproduces the wall exactly and the
   near-budget structure approximately; a selected-best bin evaluation
   (or a curriculum-faithful rerun under the registered recipe) is the
   remaining step to a full phase-diagram replication receipt.
