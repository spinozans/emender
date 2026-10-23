## Review

**Acceptance: ACCEPT-WITH-CONDITIONS.** Machinery is consistent with the registered size experiment; no scientific outcome is adjudicated.

Paths below use `X27 = experiments/expressivity_tasks/results_stage27_small_architectures`.

- **Correct — size-only comparison:** `run_small_architecture_separators.py:19–44` preserves corresponding X25 architecture features and delegates training to X25’s configuration. The shared loop retains batch64, AdamW/lr/wd, sample draws, supervision, validation selection and test protocol (`run_merrill26_separators.py:199–287`). Dataset linking reuses the actual X23E pools (`run_small_architecture_separators.py:47–55`). “Size” explicitly bundles depth/width/heads/state, not parameter count alone (`X27/REGISTRATION.md:36–41`).

- **Correct — concessions and predictions:** batch64 correction and qualitative-only X26 comparison are explicit (`X27/REGISTRATION.md:9–17`). Size-binding, architecture-binding and nothing-learns are registered, including mixed outcomes, censoring and a separately approved native-tokenization control (`:43–69`). X26 independently registers batch128 (`results_stage26_replication/registration.json:29–30`).

- **Correct — small counts:** tensor-shape receipts support **570,464 / 570,464 / 563,218**, all inside 200k–800k (`X27/receipts/parameter_counts.json:1–291`). Tied embedding/output weights must be counted once; `hybrid_ladder.py:189–192` confirms tying. GDN’s effective one-head configuration is honestly distinguished from ignored generic heads/state arguments.

- **Correct — validation evidence:** all three smoke receipts record N300, **90,904 tokens**, forward/backward plus evaluation, zero optimizer updates and passed status (`X27/receipts/smoke_{e97,e97softplus,gdn2}_graph.json:1–12`; implementation `run_merrill26_separators.py:168–196`). Both test logs record **50 passed** (`X27/receipts/prelaunch_tests.txt:1–14`, `postlaunch_tests.txt:1–14`). These are inspected receipts, not independently rerun tests.

- **Correct — chronology and launch discipline:** supplied history records registration `d3cac70b` before launch-receipt commit `4cc2d1b0`; DAG ancestry is **parent-attested**, not independently Git-verified here. Launch records bind the pushed registration (`X27/receipts/launch.json:2–5`). Tanh’s config and lease bind GPU6. Latest inspected watcher status is **running / queued / queued** (`X27/receipts/watcher_status.json:2–7`); softplus/GDN2 have no per-arm launch-receipt directories. Their smoke receipts do not imply training started.

**No issues found.**

### Required reporting correction

The task’s **“9.9M / 17×” premise is inaccurate**. Actual X25 config totals are:

| Arm | X25 parameters | Reduction |
|---|---:|---:|
| E97 tanh/softplus | 8,950,272 | ≈15.69× |
| GDN2 | 8,936,528 | ≈15.87× |

Evidence: corresponding `results_stage25_recipe_controlled/*_graph_seed0/config.json:21`. Preserve the original directive; annotate subsequent summaries with these actual totals.

### Conditions

1. Parent hash/HEAD/dependency and live-process/GPU audits pass; historical acceptance remains explicitly grounded in the supplied ancestry attestation.
2. Acceptance summaries use the corrected comparator totals/ratios and retain the X26 qualitative-only qualification.

If rerunning validation, supervisor command:

```text
.venv/bin/python -m pytest tests/test_small_architecture_separators.py tests/test_recipe_controlled_separators.py tests/test_merrill26_separators.py tests/test_e97_softplus_graph.py tests/test_merrill26_replication.py
```

- **Merge verdict: OK with notes.**

Complete review returned for parent persistence; no repository files changed.
---

## Parent attestation (2026-10-23)

X27R conditions satisfied: (1) parent audits pass - local HEAD equals origin/main
(pushed through the X27 range); the reviewer's full five-file test command reruns green
under the parent (50 passed); live-process audit: 8 GPUs active with training work
(X25 three graph arms, X26 four replication arms, X27 small-tanh on card 6; the X27
softplus/GDN2 smalls correctly queued, not launched); historical acceptance grounded in
the supplied ancestry attestation (d3cac70b precedes 4cc2d1b0 on the DAG). (2) The
comparator correction is adopted for all subsequent summaries - CONCESSION #20: the
parent's '9.9M / 17x' premise was wrong; the X25 graph arms' actual totals are
8,950,272 (E97 tanh/softplus) and 8,936,528 (GDN2) per their config.json receipts, so
the X27 small arms are ~15.69x / ~15.87x reductions, not 17x. The 9.9M figure was the
parent's memory of the full X-series vocabulary-sized model, not the separator arms'
tied-vocab configs. All summaries use the corrected totals; the X26 qualitative-only
qualification is retained.

Verdict: X27R ACCEPT-WITH-CONDITIONS -> ACCEPTED. X27 size-hypothesis machinery closed.
