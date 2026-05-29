# Racer / Hugging Face Freshness Audit

Date: 2026-05-29 UTC
Task: `racer-hf-freshness-check-20260529-full`

## Recommendation

Recommendation: **update both, but in two gates**.

1. Refresh Figure 2, paper metric text, PDF, and metric-only release/card
   references now. The current 10K-smoothed tail labels changed at the
   paper's 3-decimal precision for all three public models.
2. Proceed toward public HF checkpoint refresh as **v0.2**, not by moving
   `v0.1`, after local conversion and Docker smoke revalidation. The local
   retained checkpoints are newer than the public `v0.1` checkpoint basis and
   have better smoothed checkpoint-step BPB for all three models, with the
   clearest gain for GDN. Because the repos are already public, the actual
   public upload/tag step should remain approval-gated after the local v0.2
   release-candidate smoke passes.

Downstream workgraph tasks created from this audit:

- `refresh-figure-2-pdf`: refresh Figure 2/PDF and metric references from the
  2026-05-29 endpoints.
- `validate-v0-2-racer`: select, convert, and local/Docker-smoke v0.2
  checkpoint candidates; no public HF upload in that task.

No paper/source/HF/model artifacts were modified by this audit.

## Prior State Checked

Recent completed task logs and tracked reports establish the public/release
baseline:

- `release-v01-racer-checkpoint-pin`: refreshed Figure 2 on 2026-05-27 from
  live racer logs to E88 `0.979`, GDN `0.975`, M2RNN-CMA `0.984`; pinned the
  original v0.1 checkpoint basis.
- `release-v01-private-hf-staging-upload`: converted/uploaded the three model
  artifacts and recorded staging commits.
- `release-v01-model-card-docs-polish` and `release-v01-final-v01-docker-smoke`:
  recreated `v0.1` at docs-only descendants and passed CPU/GPU Docker smoke.
- `hf-rename-127b-to-13b-link-sync`: moved the public repos to the `1.3b`
  slugs while preserving `v0.1` SHAs.
- `figure2-caption-replication-pointer`: latest paper task in the inspected
  logs; uploaded `http://hypervolu.me/~erik/ndm/Garrison_2026_Emender-626c5e8a.pdf`
  and left Figure 2 values at E88 `0.979`, GDN `0.975`, M2RNN-CMA `0.984`.

The GitHub release asset target
`https://github.com/poietic-pbc/emender/releases/download/v0.1/Garrison_2026_Emender.pdf`
returned HTTP 404 during this audit. The latest concrete public PDF URL found
in recent paper logs is the `626c5e8a` hypervolu URL above, which returned
HTTP 200 over plain HTTP.

## Current Local Racer Endpoints

The current endpoints below were recomputed from the active logs with the same
10K-step smoothing convention used by `paper/results/figure_2/smooth.py`, and
with the pinned tokenizer conversion from
`scripts/estimate_tokenizer_bytes_per_token.json`:

```text
bits/byte = nats/token * log2(e) / 3.918625
```

The wallclock hours are the stitched monotonic wallclock used by the Figure 2
pipeline, not only the current resume segment's `elapsed_h`.

| Model | Active log | Log tail time UTC | Log mtime UTC | Tail step | Raw tail loss | 10K-smoothed loss | 10K-smoothed BPB | Rounded BPB | Stitched wallclock h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log` | 2026-05-29T17:25:46+00:00 | 2026-05-29T17:25:46.815+00:00 | 1,403,700 | 2.6256 | 2.652980 | 0.976731 | **0.977** | 512.210 |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log` | 2026-05-29T17:25:30+00:00 | 2026-05-29T17:25:30.497+00:00 | 1,844,750 | 2.5225 | 2.604058 | 0.958719 | **0.959** | 517.153 |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log` | 2026-05-29T17:25:33+00:00 | 2026-05-29T17:25:33.689+00:00 | 1,341,200 | 2.6537 | 2.661638 | 0.979918 | **0.980** | 478.372 |

Searched artifact roots:

- `/tmp/pile_convergence_3arch`
- `/tmp/pile_convergence_m2rnn`
- `/tmp/pile_convergence_hybrid_m2rnn`
- `~/elman`, including `~/elman/run_pile_convergence_3arch.sh` and
  `~/elman/benchmark_results`

The current release-trio logs/checkpoints are under `/tmp/pile_convergence_*`.
The recent `~/elman` hits were CMA-ES/baseline warm-start evaluation artifacts,
mostly 100M/Mamba2 evals, not newer E88/GDN/M2RNN-CMA public-release
checkpoints.

## Figure 2 Comparison

| Model | Public paper / v0.1 snapshot BPB | Current local BPB | Rounded label changes? |
| --- | ---: | ---: | --- |
| E88 / NDM | 0.979277 -> `0.979` | 0.976731 -> `0.977` | yes |
| GDN | 0.974841 -> `0.975` | 0.958719 -> `0.959` | yes |
| M2RNN-CMA | 0.984356 -> `0.984` | 0.979918 -> `0.980` | yes |

Ordering remains unchanged at the current tail:

```text
GDN < E88 / NDM < M2RNN-CMA
```

The narrative should still say all three are sub-1 BPB, but the current tail
does materially shift wording. The prior text says E88 and GDN are nearly
co-linear and separated by only a small fractional BPB at the reported point;
the current endpoint has GDN about `0.018` BPB below E88. The wallclock text
should also be refreshed: E88/GDN are now about 21.3-21.5 stitched GPU-days,
while M2RNN-CMA is about 19.9 days.

Decision criterion result: **plot refresh is required** because every rounded
3-decimal tail label changed.

## Local Checkpoints Versus Public v0.1

The public `v0.1` checkpoint basis from
`docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md` was:

| Model | Public v0.1 basis path at pin time | Pin step | Raw checkpoint loss | Raw checkpoint BPB | Pin mtime UTC |
| --- | --- | ---: | ---: | ---: | --- |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt` | 1,281,000 | 2.6850 | 0.988519 | 2026-05-27T20:18:43.401+00:00 |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt` | 1,686,000 | 2.6105 | 0.961091 | 2026-05-27T19:59:32.707+00:00 |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt` | 1,212,000 | 2.6870 | 0.989256 | 2026-05-27T19:51:20.514+00:00 |

Those exact `.pt` files are no longer retained locally as of this audit, but
the converted public `v0.1` HF artifacts still resolve at the expected SHAs.

Current retained checkpoint directories each contain six newer checkpoint files.
The `latest.pt` symlinks now point to newer steps than the public v0.1 basis:

| Model | Current checkpoint directory | Current `latest.pt` target | Latest checkpoint mtime UTC | Latest raw loss / raw BPB | 10K-smoothed BPB at latest checkpoint | Best retained candidate by smoothed BPB |
| --- | --- | --- | ---: | ---: | ---: | --- |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832` | `checkpoint_step_1401000_loss_2.6855.pt` | 2026-05-29T16:26:40.015+00:00 | 2.6855 / 0.988703 | 0.976366 | `checkpoint_step_1386000_loss_2.6205.pt` at 0.975616 BPB |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832` | `checkpoint_step_1842000_loss_2.6497.pt` | 2026-05-29T16:38:59.532+00:00 | 2.6497 / 0.975523 | 0.961105 | `checkpoint_step_1842000_loss_2.6497.pt` at 0.961105 BPB |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023` | `checkpoint_step_1341000_loss_2.7485.pt` | 2026-05-29T17:21:21.081+00:00 | 2.7485 / 1.011898 | 0.980318 | `checkpoint_step_1341000_loss_2.7485.pt` at 0.980318 BPB |

Checkpoint-release interpretation:

- The raw loss in the checkpoint filename is a single save-step training loss
  and is noisy. It disagrees with smoothed progress for GDN and M2RNN-CMA.
- The smoothed checkpoint-step BPB is better than the public v0.1 paper
  snapshot for all three models.
- Gains by smoothed checkpoint-step BPB are approximately E88 `-0.0037`, GDN
  `-0.0137`, and M2RNN-CMA `-0.0040` BPB versus the public v0.1 snapshot.
- This is enough to justify a v0.2 RC validation pass, especially because the
  paper refresh would otherwise report current training endpoints that no
  public checkpoint release represents.

Decision criterion result: **HF checkpoint refresh is justified only as a new
v0.2 release path after local conversion and smoke validation**. Existing
public `v0.1` tags should remain immutable.

## Public HF Checks

Unauthenticated Hugging Face checks used `huggingface_hub` with `token=False`
for `model_info`, `list_repo_refs`, and `README.md` readback. No weights were
downloaded.

| Model | Public repo | `v0.1` resolved SHA | Private? | Tags | Required files present? | README value/link check |
| --- | --- | --- | --- | --- | --- | --- |
| E88 / NDM | <https://huggingface.co/poietic-pbc/emender-e88-1.3b> | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | false | `v0.1` | yes | mentions `0.979`, `0.975`, `0.984`, GitHub repo, paper PDF target, release hub |
| GDN | <https://huggingface.co/poietic-pbc/gdn-1.3b> | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | false | `v0.1` | yes | mentions `0.979`, `0.975`, `0.984`, GitHub repo, paper PDF target, release hub |
| M2RNN-CMA | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b> | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | false | `v0.1` | yes | mentions `0.979`, `0.975`, `0.984`, GitHub repo, paper PDF target, release hub |

The `v0.1` tree URLs resolve for all three repos:

- <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.1>
- <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.1>
- <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.1>

The model cards are coherent with the current public `v0.1` state, but they
will become stale if the paper is refreshed without a metric-reference sync.

## Proposed Follow-up Scope

### `refresh-figure-2-pdf`

Scope:

- Re-run Figure 2 smoothing/plot generation from live logs.
- Update paper Figure 2 CSVs/plot, caption date, paper prose/conclusion values,
  release docs, and metric-only model-card references as needed.
- Build paper, visually inspect Figure 2, run Lean gate, upload PDF, and push
  source/docs only.

Validation:

- Recomputed current values are recorded with exact paths, times, steps,
  smoothed losses, BPB, and wallclock.
- Paper source and Figure 2 assets agree at 3 decimals.
- `paper/build.sh` has zero Typst warnings.
- Visual inspection is clean.
- `cd formal/lean && ./scripts/check_paper_core.sh ElmanProofs/PaperCore.lean`
  passes.
- No generated PDFs, checkpoints, safetensors, HF caches, Docker layers, or
  tokens are staged or committed.

### `validate-v0-2-racer`

Scope:

- Select exact v0.2 release-candidate checkpoints from the retained `/tmp`
  directories.
- Compute SHA256, raw loss/BPB, and smoothed checkpoint-step BPB for each.
- Convert locally to safetensors/config/tokenizer/modeling artifacts.
- Run local and Docker CPU/GPU smoke tests.
- Do not public-upload or retag in this task.

Validation:

- Exact selected paths, steps, mtimes, sizes, SHA256, and metric rationale are
  recorded.
- Local `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  loads all three candidates without missing/unexpected/mismatched keys.
- Docker CPU smoke passes for all three; Docker GPU smoke passes if CUDA is
  available or unavailability is logged.
- If validation passes, create or propose a separate public HF `v0.2` publish
  task with an explicit human-approval gate.

### Approval-gated public HF v0.2 publish

Scope:

- Upload validated v0.2 artifacts to the three existing public `1.3b` repos.
- Create `v0.2` tags; do **not** move `v0.1`.
- Update model cards and release hub to distinguish `v0.1` and `v0.2`.
- Run unauthenticated public resolver checks and Docker smoke against
  `revision="v0.2"` after upload.

Validation:

- Explicit human approval for public v0.2 upload is present in task context.
- `v0.2` tags resolve to intended commits for all three repos.
- `v0.1` tags still resolve to the original public SHAs.
- Public `README.md`, `config.json`, custom modeling files, tokenizer files,
  and `model.safetensors` resolve under `v0.2`.
- CPU/GPU Docker generation smoke passes from `revision="v0.2"`.
- No tokens, raw checkpoints, local safetensors, HF caches, Docker layers, or
  generated PDFs are staged/committed.

## Guardrail Result

This audit created only this Markdown report and downstream wg task metadata.
It did not modify paper source, figure assets, Hugging Face repositories,
GitHub release assets, local checkpoints, safetensors, generated PDFs, Docker
state, or token-bearing files. The git commit for this audit should stage this
report only.
