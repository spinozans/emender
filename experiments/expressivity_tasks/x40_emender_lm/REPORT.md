# X40 part 2 — PURE STACK v0: no sustained ignition, honest null

The completed 100,007,936-token run (3052 updates) has held-out perplexity
**36.8497193**, 5.3745 times the existing B64 hybrid reference 6.8564176.
This is a failed pure-stack experiment with nearly absent effective memory,
not a trained demonstration of semantic compression. The 30,000-update
final-EOS depth probe is also a null: balanced accuracy .5 on ID/2x/4x/9x,
all 512 examples in every bin predicted reject; selected update 30,000.
No matched fine-tuned transformer comparison is established.

`receipts/part2_v0_summary.json` binds the original curve, evaluation, quality
and depth hashes and extracts the quality trajectory, both final telemetry
windows for all 24 lanes, context bins and depth confusion matrices. Raw
receipts are retained. Quality PPL at steps 256/512/768/1024/1536/2048/2560/3052:
43.9204 / 41.0163 / 39.9088 / 38.7888 / 37.9813 / 37.8694 / 37.3508 / 36.8497.

A precision correction to the initial lead matters: this was **not literally
zero writes in all six blocks**. Final validation window 0 has block 0 head
peaks 1/1/1/2, mean live .199/.484/.493/.489 and gate means
.00357/.00281/.00238/.00281. Blocks 1–5 have zero live slots in every head,
zero match/error evidence, and sigmoid means approximately 0.00000075 to
0.00000234. The other validation window is retained in the summary. Thus
“no ignition” means failure to sustain useful utilization; initial opening
and a few bottom-block slots must not be erased from the account.

Context PPL at T512/1024/2048/4096/8192 is
35.4438 / 40.7373 / 42.4587 / 40.4998 / 35.4490; endpoint ratio 1.000148.
That passes the registered numerical ratio but proves no long-context use:
five blocks have undefined occupancy slopes, while block 0's scarce-slot
slopes are 0/.200/.417/.675. A stable high PPL in an effectively memoryless
stack is not graceful semantic extrapolation. These are two fixed windows
at each length, not a controlled memory ablation or causal context study.

## Diagnosis and suspect receipts

`part2_diagnosis.json` contains a CPU six-block term audit with explicit
limitations (synthetic vocabulary, zero read stub, single-slot evidence).
`part2_gpu_ignition.json` separately measures actual initial six-block gates
on a TinyStories window at the real vocabulary size. No v0 model checkpoint
exists, so terminal parameter values and a unique causal decomposition
cannot be recovered. The diagnosis is structural plus counterexamples:

- **F12 bias:** the inherited X31 initializer already sets W_write bias +2
  in all 24 lanes. Missing +2 initialization is ruled out. At initialization
  gates open; “closed from step zero” is not established for X40.
- **Match evidence:** unnormalized dot products can exceed +2 even in the
  synthetic init audit (max roughly 6–16), closing individual tokens when
  match_coef=-1. It can suppress repeated content before useful memory has
  formed. It cannot directly explain the terminal upper-block closure:
  those blocks have zero match evidence and zero post-POP error, so their
  very small sigmoid is the base W_write(x) term itself.
- **Lambda and six lanes:** runner loss uses .01 times a mean over six
  blocks, each itself a mean over four heads, not a sixfold sum. Each lane
  receives .01/24 before token averaging. But W_write and match/error
  coefficients have no task-loss gradient through detached routing. Their
  only explicit learning signal is the positive sigmoid penalty, which
  monotonically rewards closing the gate. error_coef starting at zero does
  not freeze it. A memoryless upper block can improve token-local LM loss
  through the residual MLP while its enable controller is paid to close.
- **Decay versus writes:** the actual float32 horizons are receipted as
  128/512/2048/8192. No decay term enters the enable equation. Zero write
  rate yields zero occupancy under every horizon; longer retention cannot
  repair a closed gate. POP and adjacency merges also affect occupancy;
  low occupancy alone does not identify a decay defect. CPU novel-content
  runs with those same four policies retain live stores when enabled.
- **Reproducible signature:** the summary extracts the read-only aborted
  X39 initialization curve's row count and gate_mean set. Its zero signature
  corroborates a failure mode, not proof that the two runs share every cause.

## Registered rescue and validation

The appended part-2 registration in RECEIPTS.md preceded v1 execution.
**Ignition-v1 is an always-enable diagnostic baseline:** freeze W_write
weights=0, bias=+2, match_coef=error_coef=0; lambda=0. All other policies,
content paths and thresholds stay as registered. This deliberately trades
selective enable for a falsifiable guarantee that every nonpadding token
has a write opportunity. It is a joint intervention, not a one-factor causal
identification or a claim that selective memory learning is solved.

`ignition_check.py` receipts a CPU novel-content counterexample under all
four policies and the frozen positive gate in all six blocks. The broker-only
short GPU check verifies the real six-block stack, all 24 live lanes, finite
backward gradients, 24 ReplayBackward nodes, and survival after one optimizer
step; it also runs short independent X31 parity. This small B1/T32 check is
not production-shape certification. Every parent-launched run still executes
the full unchanged-X31 oracle and B64/T512 parity before any training.

Four tests pass: three existing architecture tests plus a cross-stage receipt
directory isolation regression test. The runner now records CE and each
block gate every update and fails if the v1 enable invariant closes. Initial check-harness attempts failed
because the eager CPU API needs an explicit state and returns three values;
those were harness failures, corrected before the successful CPU receipt.
The first GPU wrapper attempt failed in that CPU setup, before a GPU scan.
No long training was launched in this part. No .pt files are included.

Exact command for the parent to launch under its process management, from
`/home/erikg/emender` (the target directory must not exist):

```bash
experiments/expressivity_tasks/x40_emender_lm/run_leased.sh -m experiments.expressivity_tasks.x40_emender_lm.runner --train --ignition-v1 --receipt-dir /home/erikg/emender/experiments/expressivity_tasks/x40_emender_lm/receipts/ignition_v1
```

The wrapper admits only cards 0/1/5/6, excluding certified arms 2/3/4 and
forbidden 7. Original v0 receipts stay in place; v1 gets a fresh directory.
The full command includes the 30,000-update depth probe after LM evaluation.
There is still no checkpoint/resume: a process loss loses trained weights.
Quality rescue, production-shape preflight, memory utility and long-context
benefit remain unmeasured for v1. Always-enabled writes may saturate capacity
or increase cost, and cannot establish semantic compression.

Only X40 is edited. Upstream x31/x34/x36/x37/x38/x39 remain read-only.
No WG machinery, Slurm or Frontier execution was used; resilient DiLoCo
requirement IDs do not apply to this single local GPU task. AGENTS.md and
CLAUDE.md are unchanged. Tool metadata discovery found no contact_supervisor,
supervisor or intercom tool; no substitute orchestration was started.

## Part 3 performance follow-up

[REPORT_PART3.md](REPORT_PART3.md) receipts the v1 decomposition and exact-decay
repair: B64/T512 improves from8.030 to1.065s eager /1.085s with chunk graphs;
5,013,504 training tokens match v1 losses exactly. Pure scale probes remain
blocked by X42-policy pressure parity; transformer controls completed, bf16
store parity failed and stays disabled. This is partial performance closure.

## X40 v1 (ignition-v1) — completed 100M-token run receipts

The always-enable diagnostic baseline completed 3052 updates =
100,007,936 tokens. All six block gates held open at sigmoid(2)=0.8807971
every update: `receipts/ignition_v1/curve.jsonl` gate minimum and maximum are
both 0.8807971477508545; the v1 enable invariant never failed. Final training
loss 2.2746. Held-out quality trajectory (`quality_256..3052.json`):
38.0834 @ 8.4M, 27.9923 @ 16.8M, 22.4253 @ 25.2M, 18.9392 @ 33.6M,
16.5786 @ 41.9M, 14.9374 @ 50.3M, 13.6429 @ 58.7M, 12.6878 @ 67.1M,
12.0683 @ 75.4M, 11.4495 @ 83.9M, 10.8761 @ 92.3M, **10.3745 @ 100M** —
the final 8M-token span still improves 0.50 PPL with no plateau. The
registered ignition-v1 prediction (held-out PPL improves over v0's
36.8497193) passes decisively. The stronger original family band does not:
10.3745 is 1.513x the B64 hybrid reference 6.8564176, an honest 51%
capacity gap — a rescue of utilization, not parity.

### Receipt 1 — long context, the no-KV-cache thesis

`receipts/ignition_v1/long_context.json` (sha-bound digest of the run's
`evaluation.json`, written against the final in-memory model; two fixed
held-out windows per length, store reset per window, 128-token NLL bins,
all 24 head profiles). Held-out PPL at T512/1024/2048/4096/8192:
**10.5736 / 11.9883 / 13.4787 / 15.5788 / 17.0459**; endpoint ratio
**1.6121 < 2: the registered graceful-decay prediction passes.** The decay
is graceful, not cliff-like: X34's B8 base transformer cliffed 6.381 ->
29.738 (4.66x) and the tape hybrid 6.515 -> 29.322 (4.50x) by T4096; X34
arms were never measured at 8192 (missing lengths stay missing; X34
receipts are B8, not B64-matched). The store is the pure model's context:
at 8x training length every lane is pinned — peak live [64, 256, 736, 1472]
= all four budgets full, the archival lane refusing 64% of write requests
(R-CAP route fraction 0.6406, mean pressure 0.635, max 1.0) — while
per-position NLL grows only from 2.54 to ~3.2-3.5 (T8192 max 128-token bin
3.483). Occupancy is block-invariant for heads 1-3 (identical peaks in all
six blocks at every length) and near-invariant for head 0: the frozen
always-open gate plus shared head geometry, not learned per-block
selectivity, sets occupancy. Absolute T512 PPL stays 1.51x the hybrid
control's.

### Receipt 2 — six-lane compression slopes

`receipts/ignition_v1/compression_slopes.json`. Log-log peak-occupancy-vs-
length slopes per block/head are [~0, ~0, 0.357, 0.657] — identical in all
six blocks for heads 1-3 (head 0: peak 62-64 of 64 everywhere; slope 0 in
blocks 0-3, 0.0046/0.0045 in blocks 4/5). **Mean slope 0.2539 <= 0.5: the
registered compression prediction passes.** The hierarchy orders the lanes
exactly as registered, but only past training length: peaks at T8192
[64, 256, 736, 1472] strictly increase with horizon (128/512/2048/8192);
the fast lane is capped at its 64-slot budget from T512 on (turning over
~192 decay-frees per 512 tokens), head 2 reaches its 736 budget at T2048,
head 3 its 1472 at T4096. At T512 heads 1-3 tie at 256 (the horizons do
not separate within training length). Equal four-head-peak stories on the
first 16 validation stories: [2, 1, 1, 0, 0, 0]. The X39 semantic single
lane receipted 0.108 (mean of [0.0945, 0.0550, 0.1275, 0.1535]); the pure
six-lane stack is deeper at 0.2539. The registered caveat binds: heads 2/3
are budget-capped with R-CAP fractions 0.2812/0.6406 at T8192, so the
sublinear slopes are substantially capacity+decay-forced ordering — not
learned semantic compression, for which always-enable removes the
selective-write evidence.

### Receipt 3 — curriculum depth probe (head-to-head control arm complete,
pure arm in flight)

The registered X28E stage-3 protocol (30000 updates, AdamW 1e-4/1e-4, B64
permutation epochs dropping the short remainder, seed 0, validation every
500 with latest-best selection, final-only binary margin BCE, S=2944
budgets) is executed by the run process itself; there is no X40 checkpoint
(the runner has no save path — the final weights exist only in that
process). At commit time the pure arm is at update 1500/30000 with
validation balanced accuracy 0.982/0.949/0.980; `depth_validation.json`
keeps evolving after this commit and is snapshotted here. At the observed
pace (~40 min per 500 updates, ~8x slower than v0's memoryless probe) the
final ID/2x/4x/9x bins (`depth.json`) land roughly 40h after probe start.

The matched transformer control arm is complete
(`receipts/ignition_v1/depth_control.json`, preregistered in
`depth_control_registration.json` written before the run; identical
protocol; X39 exact-exposure B64 hybrid checkpoint, sha256
73f9600154df00ecef14fe26a96395e233ce78536e34fde7fdf795e7ad60218b, 3052
steps, held-out PPL 6.8564176, identity-verified against the X39
`part6_control_training.json` fingerprint; broker card 1): balanced
accuracy **ID 1.0, 2x 1.0, 4x 0.707, 9x 0.5**, per-class accept recall
**1.0 / 1.0 / 0.418 / 0.0** — selected update 30000 (validation 1.0).
The identified X28E fine-tuned transformer baseline (e97_native,
final_eval sha256 4339f44bfe2dfdab90d099fd4db1f64c4a2b43191eacef42f2b36df
c193d4e16, selected update 27000) receipted 0.998 / 0.541 / 0.5 / 0.5. The
matched hybrid control thus clears 2x where e97 collapsed, partially
discriminates at 4x, and fully collapses at 9x (all-accept recall 0).
The registered prediction — pure model > 0.6 at 4x and 9x where the
fine-tuned transformer is <= 0.55 — is decided when the pure arm's bins
land; the honest null (no separation, chance OOD accuracy, or collapse
despite in-range learning) remains a publishable outcome. No pure-arm
claim is made in this commit.

### Process facts

`long_context.json` and `compression_slopes.json` are CPU digests of the
run's `evaluation.json` (source sha-bound inside both receipts); no model
or checkpoint was loaded for them. The control arm ran under the
run_leased.sh broker (lease PID 1358440, card 1). No .pt files are
committed; upstream x28e/x31/x34/x36/x37/x39 imports are read-only.
