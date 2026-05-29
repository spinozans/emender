# QA / Reasoning Stability Check

Date: 2026-05-29 UTC
Task: `qa-reasoning-stability-check-20260529`

## Result

Status: **changed slightly, not materially changed**.

The current release-candidate QA/reasoning checks do not introduce weird or
bad-vibe numbers. The exact 300-item QA values moved downward from the paper's
May 22 artifacts, but the qualitative claim remains stable: all three 1.3B
models remain above the random baseline on the fact/QA panel and remain within
small-sample uncertainty of one another. A short stratified reasoning refresh
also keeps the qualitative result stable: no architecture separates on the
reasoning panel at this training budget.

Recommendation: **no required paper/HF update from this QA/reasoning audit
before proceeding with the Figure 2 / HF metric refresh**. If the paper later
wants all QA numbers to be "current release candidate" values rather than the
existing May 22 full-panel values, create a focused prose-refresh task for
`paper/main.typ` lines 1175-1196 and validate it with a full 300-item QA plus
full 2048-item reasoning rerun. That is optional cleanup, not a blocker found
by this stability check.

## Paper Claims Checked

Current `paper/main.typ` claims at lines 1175-1196:

- The section is headed "QA and reasoning panel at 1.3 B: parity-rate
  evidence".
- The 300-item multiple-choice continuation harness samples ARC-C/E,
  HellaSwag, SciQ, OpenBookQA, and BoolQ.
- Cited QA values: E88 `0.367` with random `~0.29`, GDN `0.380`, and
  M2RNN-CMA `0.367`; all three are described as within one standard error.
- Cited reasoning qualitative claims: all three collapse on hard multi-step
  object tracking, FOLIO/ReCLor are near-random, GDN leads formal fallacies and
  web-of-lies in the old panel, and E88 reasoning accuracy `0.319` is within
  one standard error of M2RNN-CMA `0.336`.
- The section's conclusion is that standard benchmark capability is acquired at
  the same rate across the linear-recurrent, Emender, and raw-write nonlinear
  baselines at this training budget.

## Existing Artifacts Located

Committed QA/reasoning artifacts:

- `paper/results/qa_reasoning/racer_panel_300item_progression.csv`
- `paper/results/qa_reasoning/fact_panel_latest.csv`
- `paper/results/qa_reasoning/reasoning_panel_latest.csv`
- `paper/results/qa_reasoning/knowledge_probe_40item.csv`
- `paper/results/qa_reasoning/SOURCES.md`
- `paper/results/qa_reasoning/section_draft.md`

Those artifacts point to `~/racer_eval_runs/` snapshots from 2026-05-21 and
2026-05-22. The paper's cited QA values come from the 300-item snapshot at
approximately E88 step 942k, GDN step 1251k, and M2RNN step 861k. The cited
reasoning values come from the 2048-item snapshot at approximately E88 step
957k, GDN step 1272k, and M2RNN step 879k.

The current release-candidate checkpoints are much later than those artifacts,
so the existing artifacts were stale for this release-candidate stability
question. I reran the minimum useful checks instead of changing paper/source/HF
files.

## GPU Check

Before rerun, `nvidia-smi` showed:

- GPU 0: idle, `0%` utilization, `2 MiB / 49140 MiB`, no `pmon` process.
- GPUs 1-7: active training/inference jobs with high utilization.

Selected GPU: **GPU 0**.

All refresh commands used:

```bash
CUDA_VISIBLE_DEVICES=0
```

No active training/inference jobs were disrupted.

## Fresh Check Commands

Panels were rebuilt deterministically from the existing scripts and seeds.

```bash
mkdir -p /home/erikg/racer_eval_runs/qa_reasoning_stability_20260529 /tmp/racer_eval_panels

python scripts/build_racer_eval_panel.py \
  --out /tmp/racer_eval_panels/qa_stability_fact_300_20260529.jsonl \
  --per_task 50 \
  --seed 20260521

python scripts/build_reasoning_eval_panel.py \
  --out /tmp/racer_eval_panels/qa_stability_reasoning_2048_20260529.jsonl \
  --per_task 160 \
  --limit_total 2048 \
  --seed 20260522

python scripts/build_reasoning_eval_panel.py \
  --out /tmp/racer_eval_panels/qa_stability_reasoning_280_20260529.jsonl \
  --per_task 20 \
  --limit_total 0 \
  --seed 20260522
```

The 2048-item reasoning panel was built for parity with the old artifact, but
not evaluated. To keep this task a stability check rather than a broad
benchmark sweep, the fresh reasoning eval used the 280-item stratified panel
covering the same 14 reasoning categories at 20 prompts per category.

Exact fact-panel eval command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/racer_eval_suite.py \
  --probes /tmp/racer_eval_panels/qa_stability_fact_300_20260529.jsonl \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --primary_score avg_nll \
  --out /home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/fact_300_current_rc.json \
  --report /home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/fact_300_current_rc.md \
  --checkpoint /tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1404000_loss_2.7211.pt \
  --label E88 \
  --checkpoint /tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1848000_loss_2.7201.pt \
  --label FLA-GDN \
  --checkpoint /tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1344000_loss_2.7155.pt \
  --label M2RNN-CMA
```

Exact reasoning-panel stability eval command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/racer_eval_suite.py \
  --probes /tmp/racer_eval_panels/qa_stability_reasoning_280_20260529.jsonl \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --primary_score avg_nll \
  --no_pmi \
  --out /home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/reasoning_280_current_rc.json \
  --report /home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/reasoning_280_current_rc.md \
  --checkpoint /tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1404000_loss_2.7211.pt \
  --label E88 \
  --checkpoint /tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1848000_loss_2.7201.pt \
  --label FLA-GDN \
  --checkpoint /tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1344000_loss_2.7155.pt \
  --label M2RNN-CMA
```

Outputs were written outside the repository under:

- `/home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/fact_300_current_rc.json`
- `/home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/fact_300_current_rc.md`
- `/home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/reasoning_280_current_rc.json`
- `/home/erikg/racer_eval_runs/qa_reasoning_stability_20260529/reasoning_280_current_rc.md`

## Batching And Throughput

The eval is continuation-NLL scoring, not free-form text generation. There is
no sampling and no generated continuation text. The script scores the answer
choices deterministically and selects the lowest average NLL.

Batching behavior:

- `--batch_size 32` was used for both fresh evals.
- No separate microbatch argument exists in `scripts/racer_eval_suite.py`.
- E88 and M2RNN-CMA used `score_mode=stateful-prefix`, the script's default
  high-throughput path for recurrent models with simple tensor states. It runs
  the prompt prefix once per prompt, then batches answer-choice tails up to
  `batch_size`.
- GDN used `score_mode=full-sequence`, which batches full prompt-plus-choice
  sequences up to `batch_size`, grouped by exact input length to avoid
  padding-sensitive score drift.
- The existing batched/high-throughput path is the one used here. Using scalar
  unbatched scoring should not change results, only throughput. The script's
  length grouping and stateful-prefix paths are intended to remain numerically
  equivalent to the scalar continuation-NLL definition.

Observed throughput from the JSON outputs:

| Panel | Model | Prompts | Batch size | Score mode | Elapsed s | Prompts/s |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Fact 300 | E88 | 300 | 32 | stateful-prefix | 108.33 | 2.77 |
| Fact 300 | GDN | 300 | 32 | full-sequence | 14.10 | 21.28 |
| Fact 300 | M2RNN-CMA | 300 | 32 | stateful-prefix | 87.82 | 3.42 |
| Reasoning 280 | E88 | 280 | 32 | stateful-prefix | 124.66 | 2.25 |
| Reasoning 280 | GDN | 280 | 32 | full-sequence | 11.57 | 24.19 |
| Reasoning 280 | M2RNN-CMA | 280 | 32 | stateful-prefix | 108.12 | 2.59 |

## Results

### 300-item QA / fact panel

| Model | Paper value | Fresh RC value | Change | Current step |
| --- | ---: | ---: | ---: | ---: |
| E88 | 0.367 | 0.353 | -0.014 | 1,404,000 |
| GDN | 0.380 | 0.343 | -0.037 | 1,848,000 |
| M2RNN-CMA | 0.367 | 0.340 | -0.027 | 1,344,000 |

Current category values:

| Model | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 | 0.280 | 0.460 | 0.420 | 0.300 | 0.160 | 0.500 |
| GDN | 0.280 | 0.480 | 0.420 | 0.280 | 0.220 | 0.380 |
| M2RNN-CMA | 0.280 | 0.520 | 0.360 | 0.260 | 0.200 | 0.420 |

Interpretation:

- The exact QA values changed slightly and moved downward.
- The qualitative parity claim is stable. All three are still above the
  combined random baseline of roughly `0.29`.
- Pairwise current model gaps are small: E88 exceeds GDN by 1.0 pp and
  M2RNN-CMA by 1.3 pp, well inside the uncertainty implied by a 300-item panel
  and 50-item category slices.
- The paper's "within one standard error" statement still holds.

### Reasoning stability panel

The fresh reasoning check is a 280-item stratified panel, not a replacement for
the old full 2048-item paper artifact.

| Model | Old full-panel value cited in paper | Fresh 280-item stability value | Current step |
| --- | ---: | ---: | ---: |
| E88 | 0.319 | 0.311 | 1,404,000 |
| GDN | 0.350 | 0.307 | 1,848,000 |
| M2RNN-CMA | 0.336 | 0.311 | 1,344,000 |

Selected current category checks:

| Category | E88 | GDN | M2RNN-CMA |
| --- | ---: | ---: | ---: |
| formal fallacies | 0.600 | 0.550 | 0.500 |
| web of lies | 0.550 | 0.300 | 0.200 |
| logical deduction 7 objects | 0.050 | 0.200 | 0.250 |
| tracking shuffled objects 7 objects | 0.050 | 0.100 | 0.100 |
| FOLIO | 0.200 | 0.400 | 0.300 |
| ReCLor | 0.050 | 0.100 | 0.050 |

Interpretation:

- The overall reasoning result remains a tie within small-sample uncertainty.
- Hard object-tracking variants remain collapsed or near-random.
- FOLIO/ReCLor remain weak in this short panel.
- The old paper's fine-grained "GDN leading formal fallacies and web-of-lies"
  sentence is not revalidated by this 20-per-category subset. Because this was
  a short stability check rather than a full-panel replacement, this is not
  treated as a material contradiction. It is a reason not to strengthen that
  sentence unless a full reasoning rerun confirms it.

## Materiality Decision

Classification: **changed slightly**.

Not material:

- No model collapses on QA.
- No model newly separates on reasoning.
- The paper's central QA/reasoning conclusion, namely parity-rate capability
  evidence at this training budget, is still supported.
- The current values are not a blocker for proceeding with the Figure 2 / HF
  metric update.

Slight changes:

- The exact 300-item QA values are lower than the paper's stale May 22 values.
- The current 300-item ordering is E88, GDN, M2RNN-CMA, but all gaps are tiny.
- The short reasoning subset does not reproduce the old category-leader wording
  for GDN on formal fallacies and web-of-lies.

## Validation Checklist

- Existing paper QA/reasoning claims and cited values were identified from
  `paper/main.typ` lines 1175-1196.
- Existing QA/reasoning artifacts were located under
  `paper/results/qa_reasoning/` and traced to 2026-05-21/22
  `~/racer_eval_runs/` snapshots.
- Fresh GPU checks were run because the existing artifacts predate current
  release-candidate checkpoints.
- GPU availability/utilization was checked first; GPU 0 was idle and selected;
  all reruns used `CUDA_VISIBLE_DEVICES=0`.
- Exact commands, checkpoint paths, panel sizes, batch size, batching modes, and
  observed throughput are recorded above.
- Results are classified as slightly changed, not materially changed.
- Recommendation is explicit: no required paper/HF update from this audit before
  proceeding; optional focused prose cleanup only if exact current QA values are
  desired later.
- No paper/source/HF/model artifact changes were made. No generated PDFs,
  checkpoints, safetensors, HF caches, Docker layers, or tokens were committed.
