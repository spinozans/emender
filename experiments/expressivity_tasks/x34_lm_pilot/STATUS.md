# Part 1 result

Both arms completed **512 optimizer updates / 2,097,152 real TinyStories tokens**
with identical data order, batch size, tokenization and optimizer settings.
Total parameters: tape 21,913,872; baseline 21,914,119 (0.00113% difference).
Trainable parameters: 21,847,052 versus 21,914,119, respectively.
Baseline's 64-step checkpoint was resumed to 512; both final checkpoints were
loaded again without advancing counters. Checkpoints and data stay in `runs/`.

Training loss fell from 10.56 to 3.16 in both arms. At the common 512-step
milestone, fixed held-out perplexity was **25.773 baseline / 25.874 tape**.
Eight common 64-step evaluation milestones and every training update are
preserved in `receipts/smoke_*.json`; `receipts/loss_curves.svg` plots them.
These are smoke observations, not evidence of a language-model advantage.

X31's unchanged numerical forward/routing/gradient parity assertions passed
at H=4, budget=2944, key_dim=64, value_dim=128 with atol=rtol=1e-5. The existing
full X31 parity attestation was checked against source hashes. Every tape
training step asserted a ReplayBackward node; integrated gradients were finite
and the value projection received nonzero gradient. X31/X30/X28E/X33 were not
modified. Nine local tests passed, including causality, shared initialization,
packing order, liveness reconstruction, probe normalization and bin aggregation.

Both complete probe suites ran at the smoke checkpoints: ID depths through
128, OOD 256/512/1152; context lengths 512/1024/2048/4096 with per-position
losses; and 16 separate held-out stories. Both arms scored **0.5 pair accuracy
at every depth**. Report the failure plainly: there is no smoke evidence of
zero-shot counting. The pre-registered full-scale null/verdict is deferred to
part 2, rather than declaring it after only 2M tokens. At k=1152 the tokenizer
produces 865 scored targets; byte depth is not token depth. Only two corruption
pairs per depth are evaluated, so there is no statistical significance claim.

Held-out context perplexities (baseline / tape):

| Context | Baseline | Tape |
| --- | ---: | ---: |
| 512 | 25.535 | 26.098 |
| 1024 | 29.007 | 29.549 |
| 2048 | 31.429 | 32.194 |
| 4096 | 31.682 | 32.893 |

Training peak live slots were 256/head; held-out individual stories peaked at
264/head. The 4096-token windows peaked at 2048/head. Pressure remained zero
throughout these observations. This measures utilization, **not the minimum
state demanded by natural language**. The observed append/correction schedule
and finite context limit occupancy; absence of saturation cannot establish that
2944 slots suffice for general language state tracking.

## Residual risks and continuation

Analytical forward dense/scoring MACs are within 2.61%, but actual median tape
training updates are **23.42x slower** than baseline (0.625 seconds/update for
tape). Routing, replay, memory traffic and launch costs are excluded from that
MAC estimate. Measured GPU compute/time equivalence is NOT established, and a
claim of compute-matched scientific superiority would be premature. Both
parameter and actual-cost receipts are retained, without padding baseline work.

The corpus was successfully downloaded; no synthetic fallback was used. BPE
was fit on training text only, then a deterministic 200M-token prefix was
packed for LM training. Hashes of original files, packed files and tokenizer
are recorded. Initial preparation was deliberately stopped once this prefix
was available and finalized at a token boundary; the default pipeline now
reproduces that cap. Validation is the complete official validation text.

A zero-second censoring check stopped before its first optimizer update and
reported right-censoring, with 1.39 seconds of initialization/checkpoint overhead
shown explicitly. Cumulative time persists across resumes and evaluations;
SIGKILL can still lose work/time since the last durable update, requiring an
external accounting adjustment. Full runs have a 12-hour cumulative cap and
retain checkpoints every 64 updates. The full runner resumes both smoke states,
never starts fresh, and runs sequentially on broker-approved cards 0/1/5/6.

No full run had started at commit time. `run_full.sh` requires committed,
unchanged registration. If launched after this commit, runtime launch metadata
and console/PID live in ignored `runs/`; full-budget results and verdict belong
to a later part. Mutable `pilot_*.json` and `probe_pilot_*.json` receipts are
ignored so continuation does not overwrite committed smoke evidence.
