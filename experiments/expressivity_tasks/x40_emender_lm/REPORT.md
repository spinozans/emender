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
