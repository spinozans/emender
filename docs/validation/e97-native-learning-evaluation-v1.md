# Matched native learning evaluation v1

Implementation: `scripts/eval_e97_native_learning.py`.
The first panel is frozen before scoring: eight actually consumed native
trajectories, eight native development trajectories, eight no-think
conversation validation records, and eight core/compositional validation
records. All records remain whole. Native problems and legacy record identities
are deduplicated within each cohort. These are development/retention diagnostics,
not independent final holdouts; prior-lineage exposure remains disclosed.

Compare parent live/y, update-4 live/y, update-8 live/y, and update-8 saved/x on
exactly the same records. Report full assistant-token NLL, response-opening NLL
and token accuracy, and native tool-name NLL. Tool-name spans include BPE tokens
touching the name (possibly sharing its leading space). Cohort NLL is the equal
record mean; token accuracy is token-weighted. These are screening metrics, not
the complete decision-balanced D objective.

For two fitting and two development examples per checkpoint, also greedily
generate the first native turn without supplying an Analysis header. Budget:
4,096 tokens with full original prefix and sufficient context reserve. Use
segment prefill and tokenwise recurrent decoding; this is not canonical
chunk-invariant HTTP-serving qualification. Strict native frame/channel/cap
validation is unchanged. Report valid frames, source-matching tool choices and
arguments; no commands are dispatched and matching the demonstration does not
establish task success. Generated private content stays in mode-0400 artifacts,
not progress messages.

No gradients, optimizer updates, tolerance tuning, capability promotion or
learning-rate selection occur in this evaluation. A passing artifact status
means complete validated measurement coverage, not that the model improved.

## Measured result

`proc_56fa` completed in 710 seconds. Evidence:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-learning-eval-v1`.
Panel SHA: `bc64ea9d350038464153b4fe3521db4a4f9955a3980e1d82b67157b118822c48`.
Four clean-export CPU tests passed; eight shards cover all 32 examples for each
of four model representations.

| Metric | Parent y | u4 y | u8 y | u8 x |
|---|---:|---:|---:|---:|
| Native fitting NLL | 1.5423 | 1.3416 | 1.0564 | 1.0532 |
| Native development NLL | 1.5552 | 1.3685 | 1.1510 | 1.1495 |
| Development opening accuracy | 0% | 100% | 99.72% | 99.72% |
| Development tool-name accuracy | 44.08% | 80.30% | 88.24% | 89.02% |
| Conversation-retention NLL | 1.6811 | 1.7256 | 1.7075 | 1.7077 |
| Tool-retention NLL | .00217 | .19328 | .00417 | .00411 |
| Tool-retention token accuracy | 100% | 96.43% | 100% | 100% |
| Valid fitting first-turn frames | 0/2 | 0/2 | 0/2 | 1/2 |
| Valid development first-turn frames | 0/2 | 0/2 | 2/2 | 2/2 |

Both update-8 representations matched the two development source tool names,
but neither matched either complete argument payload. No tools were dispatched;
alternative arguments are not automatically wrong and no task success follows.
The update-4 tool-retention opening dip recovered by update 8 on this panel.
Conversation retention worsened slightly. These are early learning signals after
only 415,097 agent targets (~0.64% of the rebuilt training inventory), not a
complete agent-training result. The frozen exposure-matched LR screen follows;
meaningful larger-exposure training remains necessary.
