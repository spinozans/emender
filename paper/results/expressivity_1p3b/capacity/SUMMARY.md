# Task A2 Capacity — 1.3B S5 Capacity-Exceeded Experiment — Summary

Honest verdict: **in the capacity-exceeded regime on S5 there is no wall for
either model — both decay in parallel toward the random baseline, and GDN2-MLP
is ahead of E97 at every length.** S5 is a group-action task: adjacent
transpositions are reflections (the GDN update's own form), the faithful 4-dim
Coxeter representation carries them at per-token depth ~1, and group actions
have zero error compounding — the observed length decay is a *learning* failure
(not capacity), so the S5 result cannot exhibit a depth wall. The decisive
depth-wall test lives on iterated nonlinear maps (see
`../iterated-map/`, taskA3).

## Headline: full-vocab argmax accuracy vs length (8 batches/length, seed 42)

| T | E97 acc | E97 seen/held | GDN2 acc | GDN2 seen/held | baseline |
|---:|---:|---:|---:|---:|---:|
| 256 (trained) | 0.397 | 0.439 / 0.226 | 0.685 | 0.735 / 0.484 | 0.0083 |
| 512 | 0.204 | 0.227 / 0.111 | 0.363* | 0.392 / 0.244 | 0.0083 |
| 1024 | 0.109 | 0.122 / 0.055 | 0.190 | 0.207 / 0.120 | 0.0083 |
| 2048 | 0.061 | 0.068 / 0.029 | 0.103 | 0.113 / 0.063 | 0.0083 |
| 4096 | 0.033 | 0.038 / 0.014 | 0.054 | 0.061 / 0.030 | 0.0083 |
| 8192 | 0.021 | 0.025 / 0.008 | 0.034 | 0.039 / 0.016 | 0.0083 |

(*) GDN2's T=512 point: the curriculum escape completed exactly ONE optimizer
step at T=512 before the stage proved OOM-infeasible (see the salvage note in
`cap_gdn2_s5_train.json`); effectively untrained, but flagged.

Both curves fall ~1/T toward the baseline in lockstep (the GDN2/E97 ratio stays
in a narrow 1.6-1.8 band across 512-8192). No length invariance for either
model, no separation reversal, no wall. Held-out-structure deltas at the
trained length are similar (E97 seen-minus-held +0.208, GDN2 +0.251):
pattern-matching does not cover unseen transitions for either model, and
neither composes them at length either.

## Plateau evidence (pre-registered rule)

- **E97**: genuine low-ish plateau at step 5999/9979 (val seen-loss 2.185);
  stage record T=256, stop_reason `plateau`.
- **GDN2**: high plateau at step 4699/9979 (val seen-loss 0.936), which
  triggered the pre-registered curriculum escape to T=512; the escape stage
  OOM'd on hardware even at sub-batch 1 (record `oom_infeasible_on_hardware`),
  so the completed T=256 stage is the training record. Full curves in
  `cap_gdn2_s5_train.json` (both models were still improving slowly when
  their rules fired — memorization was NOT exhausted for either; this is the
  capacity-exceeded-at-cutoff caveat, reported plainly).

## Phase-0 salvage record (taskA3)

Three prior attempts crashed at the curriculum stage without weights ever
being persisted (`*_crash{1,2}_partial.json`, `*_crash3_partial.json`,
logs in `logs/`). The approved salvage re-ran training under the identical
pre-registered protocol (trajectory reproduced bit-for-bit through step 4699)
with allocator hardening (expandable_segments; per-sub-batch empty_cache) and
`--save-weights` (kept permanent in the runner). Hardening carried the
curriculum stage exactly one optimizer step further than any prior attempt;
it remained infeasible at sub-batch 1. Eval phases ran in-process with the
pre-registered batch rule (batch 8 at T=4096, batch 4 at T=8192) and completed
cleanly. No protocol change anywhere: same seed, cap, plateau rule, data.

Files: `cap_{e97,gdn2}_s5_{train,sweep,heldout}.json`, `budget.json`,
`PLAN.md`, crash partials + logs.
