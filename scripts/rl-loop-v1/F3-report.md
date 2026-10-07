# F3 — Opt-in receipts block training

## Implemented and default-off safety

Implemented the operator-directed audit C.1 block-training change on **lambda01**.
`BLOCK_MODE=off` is the default. Unflagged lanes retain the 32-receipt window,
one receipts update per eligible cycle, existing anchor schedule, probe gate,
loss floor, and sampled/adopted consumption accounting. The currently running
single lane was not restarted, signalled, or reconfigured. No live bank state,
stream, historical ledger, checkpoint, held-out evidence, coordinator, or supply
implementation was edited. Only the four owned deployed scripts and their
byte-identical repository copies were changed, plus this report.

Code commit, pushed to `origin main`: **ead67e34**
(`feat(rl-bank): opt-in exposure-accounted receipts training blocks`).
Repository report: `scripts/rl-loop-v1/F3-report.md`.

## Block policy and accounting

Run-parser environment defaults (CLI overrides with equivalent dashed names):

| Setting | Default | Meaning |
|---|---:|---|
| `BLOCK_MODE` | `off` | Explicit `off` or `on` |
| `BLOCK_MIN_TARGETS` | 65536 | Trigger at or above unconsumed assistant targets |
| `BLOCK_MAX_TARGETS` | 131072 | Strict whole-record authority target cap |
| `BLOCK_MAX_WAIT_CYCLES` | 48 | Trigger after this many completed collect cycles with inventory |

These are read when the lane process starts. Exporting an environment variable
in a different shell cannot mutate a running Python process: the operator must
apply `BLOCK_MODE=on` to its next explicitly authorized lane launch (or use
`--block-mode on`). No restart or activation was performed by this task.

In block mode, collect/teacher correction/receipt chaining continue as before,
and anchors still train/probe/adopt on their existing schedule. Receipts do not
train during accumulation. After the anchor, the lane re-verifies the stream,
subtracts the E1 consumed ledger, and persists its remaining target inventory.
The threshold or max-wait condition triggers even when that cycle forms no new
receipts, provided it is a completed collect cycle with existing inventory.
Wait age counts cycles, not tasks or wall time; empty inventory resets the age.

The new packer `--all-unconsumed --max-targets N --consumed-ledger PATH` selects
the entire unconsumed inventory unless the cap binds. It preserves era-11
ordering: all teacher-corrected receipts first, FIFO within kind, on-policy
success filler afterward. At the cap it stops before the first non-fitting
whole record; remaining receipts stay eligible. A first priority record larger
than the cap fails closed rather than truncating supervision or exceeding the
cap. Packing only appends the separate packed ledger, never consumption.

Pre-registered update formula in `block_steps()`:

```
steps = min(256, max(8, ceil(authority_assistant_targets / 1400)))
```

Thus 65536 targets request 47 updates and 131072 request 94. One
`rl_train_step.py --steps N --full-pass` invocation constructs one optimizer,
keeps optimizer/sampler state throughout, and publishes only its final
checkpoint. An authority can have more packs than updates; the full-pass
sample planner distributes all packs across updates and serially accumulates
pack gradients, normalized by their combined assistant targets, before each
optimizer step. This avoids silently dropping packs or multiplying peak model
activation memory with a dense multi-pack batch. When packs are fewer than
updates, the epoch-permutation sampler repeats enough packs to meet the
registered update budget. Inventory targets and actual sampled exposure remain
separate: repeated exposure is counted in trainer totals and per-step pack logs,
while the consumed ledger records each adopted sampled receipt once.

Before probing, the lane checks that sampled record membership covers every
packed receipt. The E3 checkpoint-bound valid-frame gate evaluates the final
candidate. A successful block advances receipts lineage/update count **once**,
commits E1 consumption for all sampled records, and recomputes remaining
inventory. Anchor lineage advances independently as before; the receipts block
uses the current adopted anchor child as its parent, retaining E2's chain.

Loss-floor skips, probe rejections and bounded `_Stop` stage failures increment
skip/backoff counters without consuming inventory. The retry policy is an
explicit one-collect-cycle cooldown (`retry_after_cycle = failed_cycle + 2`);
wait age is preserved and retry uses the then-current adopted parent. Hard
stage failures still propagate through existing lane failure/exhaustion bounds.
Successful adoption clears consecutive backoff and wait age.

State additions while enabled: `block_mode`, `block_inventory_targets`,
`block_wait_cycles`, `block_last_inventory_cycle`, `block_attempts_total`,
`block_adopted_total`, `block_skipped_total`, `block_backoff_count`, and
`block_retry_after_cycle`. Greppable lane-output events:
`BLOCK_TRIGGER`, `BLOCK_PACKED`, `BLOCK_TRAINED`, `BLOCK_ADOPTED`, `BLOCK_SKIPPED`.
The off-mode startup does not initialize block accounting on the live lane.

## Validation

Final CPU suite: **120/120 checks**, including all **101 existing checks** and
19 F3 checks in `test_block_training`. Test state exists only under the scratch
path below. Real CPU authority/pack construction, real epoch-permutation pack
materialization, real PyTorch gradient accumulation/optimizer updates, and real
lane probe/adoption/consumption helpers are exercised. The E97 GPU subprocess
and behavioral probe are stubbed; this is not a GPU qualification or capability
claim.

Exact command:

```bash
/home/erikg/emender/.venv/bin/python \
  /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/rl_bank_unit_test.py \
  --scratch /home/erikg/emender-scratch/fix-fanout/F3-unit \
  > /home/erikg/emender-scratch/fix-fanout/F3-tests.log 2>&1
```

Coverage includes default-off/parser environment controls, exact threshold and
below-threshold behavior, no duplicate wait increment within a cycle, empty
inventory, teacher-priority cap/consumed exclusion, bounded density arithmetic,
full-pass planning when packs exceed updates, continuing anchors during
accumulation, a real rejected 36-receipt block preserving all inventory, cooldown
and retry, final adoption advancing once, all 36 receipts consumed only on
adoption, persistent optimizer state inside one invocation, target-normalized
serial accumulation, and low-signal skips. The 36-receipt regression explicitly
exceeds the previous 32-receipt packing window.

Initial validation exposed a missing lane schema in the new test fixture (all
101 pre-existing checks passed); the fixture was corrected. An intermediate
115/115 run passed before five extra parser/wait/skip assertions were added.
The retained final log is the clean 120/120 run. All four modified scripts
compile via Python `compile()`, repository/deployed copies compare byte for
byte, and `git diff --check` passes. No test warning remains in the final run.

Architecture reviewed: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, v1 with ADR-003
amendments, and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable prototype safety
intents: **R07** validated checkpoint adoption, **R12** retained lineage,
**R14/NDP13** bounded stage subprocesses. These tests do not claim ADR-003
production recovery, elastic/native data-plane conformance, async-v2.1/V21S or
ISP overlap qualification. No Frontier Python, Slurm submission, queue binding,
or GPU launch occurred; the Frontier activation requirement does not apply to
this lambda01 CPU run.

## Cycle time, limits, and next step

- `--max-tasks` is unchanged (parser default 2; the live lane currently requests
  4). A cycle is one bounded collect invocation of up to that many tasks plus
  its scheduled anchor; in block mode receipts training/probing happens only
  on trigger cycles. Accumulation cycles omit the receipts checkpoint/probe and
  should be shorter, but no numerical wall-time assumption or benchmark is
  asserted. Increasing max-tasks forms inventory sooner per cycle; decreasing
  cycle duration makes the 48-cycle fallback sooner in wall time. Teacher
  latency, anchor acceptance, task supply and lease contention remain variable.
- Idle/lease-wait polling is not a collect cycle and does not advance max-wait
  age. This is a cycle fallback, not a guarantee of wall-clock firing without
  task supply. Existing `--max-seconds` bounds the lane run.
- The existing `--train-timeout` (3600s default) remains authoritative. No GPU
  timing was measured for 47/94-step blocks; operators must check the bounded
  runtime budget in a separately approved smoke run before enabling production
  blocks. A timeout consumes nothing but invokes existing failure handling.
- The per-invocation optimizer remains fresh between blocks, as requested;
  persistent optimizer state across separate invocations is not claimed.
- Existing filesystem adoption and consumption are not one atomic crash
  transaction. A crash between saved lineage and the consumed-ledger append
  still requires reconciliation from checkpoint/sample logs (F1's retained
  limitation). No recovery-system redesign was introduced.
- Cap selection may leave a small unused tail rather than reorder or split a
  non-fitting priority record. An oversized first record requires operator
  inspection/cap adjustment; it is not silently bypassed.

The required initial `git pull --rebase origin main` in the attended tree was
blocked by the standing-supply daemon's unrelated uncommitted edit to
`configs/pi/e97-firstparty-collection-authorizations-v1.json`. The supervisor
approved an isolated clean detached worktree for the pull/rebase and non-force
`git push origin HEAD:main`. This left the daemon's authorizations edit intact
in the attended tree for its own commit cycle; it was neither staged nor
stashed by F3. The temporary push worktree was removed after the push. Report
publication follows the same scoped workflow if concurrent dirty edits recur.

Recommended next step: required independent review, followed only by an
operator-authorized bounded GPU block checking actual full-pack exposure,
anchor-child ancestry, final-probe binding, runtime and rejection accounting.
Do not infer capability improvement or authorize enabling `BLOCK_MODE` from
CPU tests alone.
