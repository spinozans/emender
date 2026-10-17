# X5 — curriculum depth frontier (8M scale)

**Headline: deepest stage crossing 90% full-state validation accuracy.** `none (<4)` means no registered stage crossed; it is not a tested T=0 result.

| Architecture | Seed | Frontier T | Steps to threshold (T:steps) | Stopping T | Final accuracy | Total updates |
|---|---:|---:|---|---:|---:|---:|
| e97 | 0 | none (<4) | none | 4 | 0.5371% | 10000 |
| gdn2 | 0 | none (<4) | none | 4 | 0.3906% | 10000 |
| e97 | 1 | none (<4) | none | 4 | 0.2930% | 10000 |
| gdn2 | 1 | none (<4) | none | 4 | 0.8301% | 10000 |
| e97 | 2 | none (<4) | none | 4 | 0.4883% | 10000 |
| gdn2 | 2 | none (<4) | none | 4 | 0.2441% | 10000 |

## Reference beside the frontier

For this invertible map, a predictor restricted to a proper driver suffix with at least two omitted drivers has full-state accuracy 1/3721 (0.026874%) under uniform independent inputs; this is a suffix-only information bound, not a capacity bound on either recurrent model.
Uniform marginal reference: ln(3721) = **8.221748 nats**; suffix floor **0.026874%**.

## Stage loss levels and retention

Loss columns are observed means over the first/final 500 training updates, not a fitted mechanism or an additional early-stop rule. Δ is final training loss minus ln(3721).

| Arm/seed | T | Steps | First-500 CE | Final-500 CE | Δ marginal | Final val CE | Best val accuracy | Retention at exit (T:accuracy) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e97/0 | 4 | 10000 | 34.139728 | 7.794520 | -0.427227 | 7.707434 | 0.6348% | 4:0.5371% |
| gdn2/0 | 4 | 10000 | 33.673335 | 7.629733 | -0.592015 | 7.548789 | 0.8301% | 4:0.3906% |
| e97/1 | 4 | 10000 | 33.950094 | 7.718501 | -0.503247 | 7.669477 | 0.6836% | 4:0.2930% |
| gdn2/1 | 4 | 10000 | 34.875605 | 7.660903 | -0.560844 | 7.550844 | 0.8301% | 4:0.8301% |
| e97/2 | 4 | 10000 | 33.913101 | 7.756801 | -0.464947 | 7.704745 | 0.6836% | 4:0.4883% |
| gdn2/2 | 4 | 10000 | 33.227087 | 7.495137 | -0.726611 | 7.503761 | 0.6836% | 4:0.2441% |

Trailing training-loss OLS slopes (up to 2,000 updates; descriptive, not a stop rule):
- e97/0, T=4: 0.00007262 nats/update over 2000 updates.
- gdn2/0, T=4: 0.00005830 nats/update over 2000 updates.
- e97/1, T=4: 0.00003956 nats/update over 2000 updates.
- gdn2/1, T=4: 0.00011088 nats/update over 2000 updates.
- e97/2, T=4: 0.00002853 nats/update over 2000 updates.
- gdn2/2, T=4: -0.00010152 nats/update over 2000 updates.
Budget exhaustion, not a pre-registered convergence/plateau test, ends a non-crossing stage; the observed loss levels and slopes do not establish an asymptotic plateau.

## What this changes about the X-series reading

Both architectures stop at the initial T=4 stage in all three seeds: none reaches 90% within 10,000 updates. No doubling transition is therefore authorized by the registered rule, and T=8–128 are **unattempted**, not measured failures. There is no lower completed curriculum stage whose retention could be lost; T=4 is evaluated at entry and every 500 updates through the stopping boundary. The experiment moves the observed failure boundary from flat T=128 to the much shallower initial T=4 under this recipe, but does not establish an intrinsic architecture capability limit or test whether a curriculum starting below T=4 could bootstrap composition. These six frontiers support no architectural superiority claim.

## Recipe, receipts and validation

From scratch, seeds 0/1/2; exact X3 E97-MLP (9,900,800 parameters) and real fla-gdn GDN2 (9,887,056; allow_neg_eigval=True). Task vocab 3721; batch 32; matched lr 1e-3; ScheduleFree AdamW β=(.9,.95), wd=.01, clip=1, bf16 autocast, no warmup. Fresh procedural full-state supervision only at the current depth. Same optimizer, weights and training RNG carried across stages; ScheduleFree eval x weights determine threshold crossings. Observed crossings are resolved on the registered 500-step evaluation grid.

T=4→8→16→32→64→128; ≥90% advances; 10K updates without crossing stops; total cap 60K. At entry, every 500 updates and every exit, score current and all lower registered depths on 512 frozen sequences each. Frozen pools use the exact X2/X3 algorithm; aligned archived validation arrays are checked for equality and their file hashes recorded, including T=128 even when unattempted. No test-pool selection.

Results root: `experiments/expressivity_tasks/results_invertible_stage5_curriculum_8m`. `registration.json` predates launch; `receipts/` retains commands, exclusive six-GPU leases, source snapshots, source hashes, package versions, GPU/disk snapshots and exits. Each run retains config, frozen pool arrays/hashes, per-update training.jsonl, validation.json, transitions.jsonl, stages.json, checkpoints.json and result.json. Atomic local checkpoints include model, optimizer and both RNG states; git excludes checkpoint payloads, retaining their paths and SHA-256 receipts. Below 400GiB free the established retention policy preserves stage boundaries, stage best, current and 10K checkpoints; any pruning is audited.

Independent aggregation verifies all six exits, model/recipe counts, exact update sequences, the earliest observed threshold crossing or 10K failure, all lower-depth evaluation panels, stage transitions, loss means, frozen pool hashes and every retained checkpoint hash. Boundary checkpoint payloads additionally verify cumulative optimizer clocks and exact model/optimizer/RNG continuity when a stage advances. CPU tests exercise inclusive thresholds, stop priority, optimizer continuity, retention panels, pool identity and capacity pruning. These are local single-GPU task probes, not resilient DiLoCo training: no R/NDP conformance or Slurm Partition/QOS claim is made (R16 evidence discipline only).

A first launcher preflight stopped before GPU acquisition because the venv lacks pip; package enumeration was switched to stdlib importlib.metadata. That failed preflight and its source snapshot are retained; no training was started or recipe changed.

After launch, a one-line capacity-retention correction explicitly protects stage-entry checkpoints even when a later best supersedes them. No run triggered pruning (free space remained above 400GiB), so this correction does not alter executed training or any frontier. Exact executed source remains in `receipts/full_launch/runner_at_launch.py`; the registration and each run config bind that snapshot, not the later correction. A regression test covers the corrected non-10K stage-entry case.
