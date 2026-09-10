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
