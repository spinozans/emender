# E97 4B Pi v4 post-broad evaluation preflight

**Status:** consumed by the first model evaluation; diagnostic-only from 2026-09-02

V4 is the replacement structural holdout after the first v3 evaluation became
diagnostic. No v4 record, generator output, path, value, payload, or expected
trajectory may enter SFT, teacher prompts, rejection sampling, or RL.

## Authority

- root: `/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout`;
- records: 240, 40 per family;
- expected calls: 960;
- manifest SHA-256:
  `8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8`;
- metadata SHA-256:
  `fc391133e82cc85f05e0e29a00bd521d3d2fdac526f14def1893e166db363dac`.

Families:

1. three-way comparison;
2. pointer-conditioned single-target edit;
3. typed CSV aggregation and JSON write;
4. test-guided implementation rename;
5. search-selected link update;
6. checksum manifest construction and verification.

## Mechanical validation

`scripts/validate_e97_pi_eval_authority.py` reconstructed every sandbox and
executed all 960 declared calls. It checked declared failing-command exit status,
exact writes/edits, successful terminal verifiers, postconditions, and required
final evidence. Result: 240/240 tasks and 960/960 expected calls passed.

The first model evaluation was originally reserved for a behaviorally selected
broad post-training checkpoint that had cleared smoke, v2, broad instruction,
and real-repository development gates. The operator-directed truth-test pivot
consumed V4 before the broad-instruction and real-repository gates were complete.
This ordering deviation is retained explicitly: V4 cannot support another blind
claim, and a new holdout is required for later independent promotion evidence.

## First and only independent evaluation

- checkpoint: complete-64K dual-prompt u256;
- checkpoint SHA-256:
  `2f24db49be7bafb0e155bf3698f20193def8e90efc1bd69bdfbfe2b9661b541b`;
- system variant: `pi-agent-v2`;
- run: `e97-4b-broad-dual-prompt-64k-u256-independent-v4-25faabe5`;
- strict result: **0/240**;
- schema-valid tool calls: 240/240;
- agent completion: 1/240;
- exact tool arguments: 0/240;
- exact tool sequences: 0/240;
- sandbox postconditions: 40/240;
- non-identical-call-cycle check: 16/240;
- Pi exit zero: 239/240.

The dominant failure was not malformed protocol. Calls were schema-valid, but
the model commonly issued the correct first grounded read and then repeated it
instead of using its observation to advance the workflow. The result rejects a
broad unfamiliar-family coding-agent claim for this checkpoint. V4 is now
training-influenced diagnostic evidence even if its records remain excluded.
