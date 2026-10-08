# F5 — Collector-grid policy-gradient batches

## Critical replay finding and approved contract

The deployed PG attempt path **does not store generation-time per-token logprobs**. Attempt evidence contains the native transcript, generated token ids, sealed policy grade, and generating checkpoint sha in `collect-summary.json`. `rl_pg_batch.py` re-encodes these tokens into prefix/generated rows; it does not transport behavior-policy logprobs. The existing PG step sets `old = gathered.detach().clone()` from the CURRENT training forward, so its importance ratio is exactly one by construction. This is effectively group-relative REINFORCE scored under the training checkpoint, not behavior-policy importance sampling.

The supervisor approved extending ownership to the actual builder, `rl_pg_batch.py`, and specified an explicit **grid-only two-pass replay**. Every retained cross-lane turn is now scored in a no-grad pre-pass under the current lineage's train weights before `group_advantages` is called. Those detached logprobs supply `old` in the update pass. Collector seed checkpoint identities remain provenance only. Multi-update invocations repeat replay at each update start; the parent reference remains frozen at step one. The native qualification's turn layout, FP32 head capture, CE consistency tolerance, coverage gates, candidate loss, clipping, KL beta, optimizer, and fallback bounds are unchanged.

**What the KL guard actually measures:** `realized_kl_vs_old_mean` is the mean across rows of each row's per-token k3 KL between post-update and immediately pre-update current-policy logprobs on replayed tokens. It bounds update displacement. It does NOT measure divergence from a collector seed/generating checkpoint, does not constrain accumulated lineage drift, and does not establish an unbiased off-policy correction. The existing `importance_ratio_mode="recomputed-on-policy"` contract is retained, not relabeled as behavior-policy importance sampling. Grid receipts additionally name the replay mode and clarify anchor semantics.

## Scoped implementation

- `rl_pg_batch.py`: optional `--streams-root` unions standard attempt evidence from every numeric `lane-N` directory. The window applies independently per lane. Train-split isolation, family/task grouping, min-group contrast and all existing re-encode/digest guards remain. Task IDs may collide across lanes without dropping distinct episodes. The one global recency cap sorts by immutable write-once collect-summary `mtime_ns`, then numeric lane/cycle/task; independent lane counters are not comparable wall clocks. Grid-only entry fields bind `source_lane` and `collected_mtime_ns`; verification resolves each episode and grade in its originating lane.
- `rl_bank_lane.py`: PG binds the streams root when explicitly configured through `BLOCK_STREAMS_ROOT`, when block mode supplies its bank-lanes default, or when `COLLECTOR_LANES>0` supplies the bank-lanes default even with block mode off. Unconfigured legacy PG build commands remain unchanged.
- `rl_policy_gradient_step.py`: grid-only current-checkpoint pre-pass before advantages, followed by the unchanged guarded update and post-update KL pass. Legacy single-lane arithmetic and receipt parameter fields remain unchanged.
- `rl_bank_unit_test.py`: 24 new F5 checks; all 138 existing checks still pass.
- `run_bank.sh`: collector launch geometry already works. Its only edit is the preflight suite-count literal/comment, now 162 rather than 138.

The collector's ordinary `bank_collect` path already writes BOTH passed and failed policy attempts to `<lane>/episodes/cycle-NNNN/<task>/attempt/{episode-private.json,grade.json}` and publishes the immutable collect summary before the collector-only training early-out. No collector evidence-storage fix or training enablement was needed. F4's collector non-training/lineage tests remain passing.

The builder and PG step previously existed only in the deployed scripts; this commit tracks their full repo copies for discoverability. All five owned runtime/test scripts are byte-identical between `scripts/rl-loop-v1/` and the nvme scripts directory. **No live bank, checkpoint, state, stream, running process, scheduler submission, or collection was touched.** The operator must relaunch to opt in.

## Validation and retained evidence

Host: lambda01; Python: `/home/erikg/emender/.venv/bin/python` (3.12.3). Frontier module activation and scheduler queue checks are not applicable: no Frontier execution or Slurm submission occurred. The supervisor confirmed the recovered `gpt-6.1-sol` route and waived the unavailable worker route-probing tool; no glm fallback was used.

Repository suite:

```bash
PYTHONPATH=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts:/home/erikg/emender \
  /home/erikg/emender/.venv/bin/python scripts/rl-loop-v1/rl_bank_unit_test.py \
  --scratch /home/erikg/emender-scratch/fix-fanout/F5-unit \
  > /home/erikg/emender-scratch/fix-fanout/F5-tests.log 2>&1
```

Deployed suite:

```bash
PYTHONPATH=/home/erikg/emender /home/erikg/emender/.venv/bin/python \
  /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/rl_bank_unit_test.py \
  --scratch /home/erikg/emender-scratch/fix-fanout/F5-unit-deployed \
  > /home/erikg/emender-scratch/fix-fanout/F5-tests-deployed.log 2>&1
```

Both: **162/162**. Tests cover four lanes including failed attempts, colliding task IDs, family grouping, split isolation, checkpoint provenance, per-lane windows, global cap, deterministic mtime selection and numeric ties, lost contrast after capping, source/split/digest/missing-root guards, and a golden canonical single-lane batch digest captured from the pre-F5 deployed builder. A real builder CLI invocation through `_pg_train_row` verifies lane wiring and guard-triggered fallback. CPU-stubbed PG `main()` executes the real candidate advantages/loss/KL and SGD, asserts all current-parent replay forwards occur before advantages, ignores poisoned stale-logprob fields, checks exact ratio one and legacy loss/KL equality, and checks identical CE, realized-KL, nonfinite advantage/loss-gradient and coverage failures without checkpoint/receipt publication.

Also passed: `py_compile` for all four Python scripts, `bash -n` for both wrapper copies, `git diff --check`, and five source-copy `cmp` checks. The first 138 suite check names remain identical to F4's retained run.

Architecture evidence discipline: R07 atomic committed-publication safety is retained (guards publish no candidate); R15 numerical/reference intent is covered locally by candidate-loss/KL parity; R16 requires honest qualification boundaries. These CPU tests **do not claim a GPU/native logprob qualification, resilient/elastic conformance, Frontier rung, behavioral gain, or measured collection throughput**.

## Commit and remaining risks

Implementation commit `d1afae3c` (`Feed PG batches from collector grids with current-policy replay`) was pushed to `origin/main`. This report is committed separately. No staged files remain after commits.

- The explicit pre-pass costs one extra forward and stores detached per-row logprobs on CPU; existing PG timeouts remain unchanged. Real GPU geometry/performance is not requalified here.
- Global recency depends on filesystem summary clocks. Stable same-state selection and deterministic ties are tested; copied/restored files with altered mtimes can change future selection order.
- Seed-generated trajectories remain off-policy observations; current-policy replay matches the existing arithmetic but is not a new unbiased behavior-policy estimator. The KL guard retains its update-displacement meaning.
- Existing checkpoint/exposure crash-reconciliation limitations and GPU-lease contention remain unchanged.

Recommended next step: required independent review of the scoped change, then operator-authorized collector-grid relaunch and a separately approved GPU replay/throughput smoke. No live relaunch is authorized by this report.
