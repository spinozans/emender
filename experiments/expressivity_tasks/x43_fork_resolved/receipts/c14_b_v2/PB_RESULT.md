# Frozen-window B1-v2 PB verdicts

Complete paired evaluation: 3,431,242 frozen windows / 434,204,084 total targets,
including the separate 214,097,022-target final-test half. No live query or
checkpoint selection occurred. First-shard on/off logits matched the original
metadata/telemetry implementation bit-for-bit; first/middle/last cache shards
matched the independent raw oracle. Evaluation completed in 3743.807 seconds.

| Prediction | Registered measurement | Result |
|---|---|---|
| PB1 | B1-on 1.584700014 vs passive v2 1.509260010 | +0.075440004 NLL; input channel better at the registered fixed budget |
| PB2 supported target | 0.856875769 vs 0.811639725 | +0.045236044 |
| PB2 zero target | 3.679494742 vs 3.517122903 | +0.162371839 |
| PB2 fork | 2.085738874 vs 2.010901269 | +0.074837605 |
| PB4 unplugged | 2.430144516 vs dense 2.276133549 | +0.154010968; fails the registered 0.02 margin |

PB1 still improves on dense by 0.691433535 NLL. These are the registered signs,
not a pure architecture or equal-exposure causal conclusion: B1-v2 trained on
71,677,042 targets, 40.0067% of passive v2's 179,162,804 at equal active time.
The first 237-step attempt's approximately 2.20 evaluation is retained separately
and does not replace these verdicts.

PB3: all four heads reached their physical 32/64-slot budgets. Training array
slot opportunities/found/read were 42,700,419 / 42,044,952 / 45,555,166 /
45,885,395 across heads 0/0, 0/1, 1/0, 1/1; found and read equal opportunities,
so all four measured findability rates are 1.0. Sampled evaluation opportunities
were 6,856,388 / 7,109,001 / 7,924,062 / 7,072,772, also with rates 1.0.
CORRECT and MERGE occurred in every head. Full counters, peaks, denominators
and occupancy samples are retained. Nonzero effective float32 weight is a
numeric findability proxy, not top-1 retrieval or causal-benefit evidence;
CORRECT totals also combine array and stream events.

PB4's independent populated native-store decay test reached empty after
128/513 ticks, with zero residual array slots. This lifecycle pass does not
rescue the failed unplugged NLL prediction. All held-out windows reset empty,
and off mode made zero new array insertions.

PB1 moves, so both registered conditional ablations are required. They remain
pending at this phase; old first-attempt append-only artifacts must not be used
as controls for this retrained B1-v2. verdict_summary.json is the compact receipt;
verdicts.json.gz preserves the full unmodified output and telemetry. No .pt file
is committed. The selected checkpoint remains frozen at SHA256
9c87a6a306b708608d7f466dbd62c70c53263c5322f24963b91b94afc8f4def8.
