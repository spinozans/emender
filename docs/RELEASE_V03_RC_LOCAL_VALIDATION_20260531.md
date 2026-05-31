# v0.3 Racer RC Local Validation (V3 paper)

Date: 2026-05-31 UTC
Task: `v3-validate-current`

Local-only validation of the three racer checkpoints selected to **match the
committed V3 paper's reported endpoints**, for a prospective v0.3 Hugging Face
release. **No** Hugging Face upload, repo/branch/tag creation, visibility change,
or public model-card mutation was performed by this task. No weights,
safetensors, tokenizers, HF caches, Docker layers, or PDFs were staged or
committed; the only committed files are this report and two read-only
selection/conversion helper scripts.

Large generated artifacts and evidence live outside the git checkout:

- Selection evidence: `/tmp/v3-racer-selection.json`
- Local converted artifacts:
  `/tmp/release-v03-local-hf-candidates-agent-672/{e88,gdn,m2rnn}`
- Local validation manifest:
  `/tmp/release-v03-local-hf-candidates-agent-672/validation_manifest.json`
- Docker smoke transcript + JSON evidence:
  `/tmp/release-v03-docker-local-hf-artifact-smoke-agent-672/`

---

## ⚠️ Important: ticket AS_OF steps are stale; selection matches the *committed* paper

The task ticket listed AS_OF endpoints **E88 ~1,405,450 / GDN ~1,847,050 /
M2RNN-CMA ~1,343,050**. These are the **pre-recompute** draft steps. They were
superseded by task `v3-data-recompute` (commit `5381651`) and `v3-prose-update`
(commit `3c52a9b`), and the V3 synthesis gate (`paper/review/V3_GATE.md`) signed
**GO** on the recomputed numbers. The ticket's AS_OF numbers appear in
`paper/review/V3_NUMBERS.md` change #14 explicitly as the **OLD** values.

The **committed** V3 paper (`paper/main.typ` §5 L906–910, `AS_OF.md`,
`V3_NUMBERS.md`) reports the recomputed endpoints from the
**2026-05-31T13:49:33Z** active-log snapshot, using the canonical `trail_100k`
(100K-step trailing average) convention:

| Model | Committed paper endpoint step | Paper `trail_100k` nats | Paper BPB |
| --- | ---: | ---: | ---: |
| E88 / NDM | 1,523,250 | 2.644925 | **0.974** |
| GDN | 1,999,300 | 2.653617 | **0.977** |
| M2RNN-CMA | 1,466,400 | 2.661439 | **0.980** |

Per the task's controlling principle — *"The HF checkpoints must match the
paper, NOT the latest training step"* — I selected against the **committed
paper** endpoints. Matching the stale ticket steps would make HF **disagree**
with the published paper, which is exactly what the task forbids.

### Availability of the ticket AS_OF checkpoints (STOP-condition check)

The task asks me to stop and report if the on-disk checkpoint nearest the AS_OF
step was overwritten by ongoing training. Checkpoint directories retain only a
rolling window of the most recent ~6 saves (3,000-step spacing):

| Model | Retained on-disk steps | Ticket AS_OF step | On disk? |
| --- | --- | ---: | :--: |
| E88 / NDM | 1,509,000 … 1,524,000 | 1,405,450 | ❌ overwritten |
| GDN | 1,986,000 … 2,001,000 | 1,847,050 | ❌ overwritten |
| M2RNN-CMA | 1,452,000 … 1,467,000 | 1,343,050 | ❌ overwritten |

**All three ticket AS_OF checkpoints are gone** (only newer saves survive).
However, the **committed paper's actual endpoint** steps *are* covered by the
retained window for every architecture, so HF *can* be made to agree with the
paper without substituting an off-paper checkpoint. This is the opposite of the
failure mode the STOP clause guards against: the selected checkpoints' smoothed
BPB reproduce the paper's published labels (below).

---

## Selection Rule

For each architecture, select the retained on-disk checkpoint `.pt` **nearest
the committed V3 paper endpoint step**, then record its `trail_100k` smoothed
loss/BPB at the selected step. The smoothed metric is recomputed with the
**canonical** pipeline — `paper/results/figure_2/smooth.py`'s `MODELS` config and
`trailing_average` (idx window = 100,000 / `log_every`=50 = 2,000 log entries) —
run against the current live logs (which extend past every selected step). This
is the same method that produced the paper numbers. BPB conversion uses the
pinned tokenizer constant
`bits_per_byte_per_nat_per_token = 0.3681635882200934`
(`scripts/estimate_tokenizer_bytes_per_token.json`, `bytes_per_token = 3.918625`).

Helper: `scripts/select_v03_racer_checkpoints.py` (read-only; writes
`/tmp/v3-racer-selection.json`).

---

## Selected Checkpoints

| Model | Selected checkpoint | Step | Δ vs paper step | Δ vs ticket AS_OF | Raw ckpt loss | Raw ckpt BPB | `trail_100k` loss @ step | `trail_100k` BPB @ step | Paper BPB | mtime UTC | Size bytes | SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| E88 / NDM | `…/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1524000_loss_2.6143.pt` | 1,524,000 | +750 | +118,550 | 2.6143 | 0.962490 | 2.644967 | 0.973780 | 0.974 (0.973765) | 2026-05-31T14:06:01.315Z | 7,639,217,707 | `090c743f72cb4e28fbe9be05402d51a5353dd5fcfdc98c75628a537543f96c74` |
| GDN | `…/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1998000_loss_2.6148.pt` | 1,998,000 | −1,300 | +150,950 | 2.6148 | 0.962674 | 2.653895 | 0.977067 | 0.977 (0.976965) | 2026-05-31T13:27:18.385Z | 8,114,430,987 | `b002fc98ca053c3125a30a0c7329aadbcef216981b5685eac8826870b61344ed` |
| M2RNN-CMA | `…/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1467000_loss_2.6277.pt` | 1,467,000 | +600 | +123,950 | 2.6277 | 0.967423 | 2.661404 | 0.979832 | 0.980 (0.979845) | 2026-05-31T14:02:16.500Z | 7,842,766,221 | `72c03692ca62762b4fad07017f03e0b147b0cc8b79716c02ff3f08356b56676d` |

Each selected checkpoint's `trail_100k` BPB rounds to the paper's published
label (0.974 / 0.977 / 0.980), with sub-0.0002 BPB residual versus the paper's
exact endpoint value (E88 +0.000015, GDN +0.000102, M2RNN −0.000013). The
±750/1,300/600 step offset versus the paper tail step is negligible against the
100K-step trailing window. Note the **raw** checkpoint-filename loss is a single
noisy save-step training loss and is *not* the paper/Figure-2 metric; the
smoothed `trail_100k` column is the paper metric.

---

## Local Conversion

Command:

```bash
python -u scripts/prepare_v03_local_hf_candidates.py \
  --workdir /tmp/release-v03-local-hf-candidates-agent-672 \
  --force
```

The helper wrote, per model: `README.md`, `config.json`, `configuration_ndm.py`,
`modeling_ndm.py`, `generation_config.json`, `requirements.txt`,
`special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`,
`tiktoken/tokenizer.model`, `model.safetensors`. Raw `.pt` checkpoints were read
in place from `/tmp` and were not copied into git; safetensors were written only
under `/tmp`.

| Model | Artifact directory | `model.safetensors` size | Safetensors keys |
| --- | --- | ---: | ---: |
| E88 / NDM | `/tmp/release-v03-local-hf-candidates-agent-672/e88` | 2,713,728,024 | 87 |
| GDN | `/tmp/release-v03-local-hf-candidates-agent-672/gdn` | 2,975,047,820 | 297 |
| M2RNN-CMA | `/tmp/release-v03-local-hf-candidates-agent-672/m2rnn` | 2,807,297,000 | 150 |

---

## Local AutoModel Load

```python
AutoModelForCausalLM.from_pretrained(
    str(artifact_dir), trust_remote_code=True,
    torch_dtype=torch.bfloat16, output_loading_info=True, local_files_only=True,
)
```

| Model | Loaded class | Core class | Missing | Unexpected | Mismatched | Param count | Result |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| E88 / NDM | `NdmForCausalLM` | `ndm.models.ladder_lm.LadderLM` | `[]` | `[]` | `[]` | 1,273,191,856 | PASS |
| GDN | `NdmForCausalLM` | `ndm.models.ladder_lm.LadderLM` | `[]` | `[]` | `[]` | 1,352,352,498 | PASS |
| M2RNN-CMA | `NdmForCausalLM` | `ndm.models.m2rnn_baseline.M2RNNLM` | `[]` | `[]` | `[]` | 1,307,101,140 | PASS |

---

## Docker Smoke

Command (reusing the path-agnostic v0.x local-HF-artifact harness):

```bash
SMOKE_IMAGE=ndm-release-v03-local-hf-artifact-smoke:20260531 \
SMOKE_ARTIFACT_BASE=/tmp/release-v03-local-hf-candidates-agent-672 \
SMOKE_OUTPUT_DIR=/tmp/release-v03-docker-local-hf-artifact-smoke-agent-672 \
SMOKE_GPU_DEVICE=0 \
scripts/docker_local_hf_artifact_smoke.sh
```

- Image: `ndm-release-v03-local-hf-artifact-smoke:20260531`
  (`sha256:a82ac4bad3fa87806252bde9367823a5c2523ec7d1662be38d219e789a49da19`,
  8,654,573,484 bytes)
- Runs used `--network none`, bind-mounted only the three converted artifact
  dirs (read-only) and the small output dir, and used fresh per-run Docker cache
  volumes that were removed by the harness exit trap (`docker volume ls` shows
  none remaining).
- CUDA evidence:
  `{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}`

Result: `ok=True  gpu_status=available`. Greedy generation (`max_new_tokens=2`,
prompt `"The theorem states"`) for all three models on both CPU and CUDA, all
logits finite, no missing/unexpected/mismatched keys:

| File | ok | model | device | step | core class | new token ids | text |
| --- | :--: | --- | --- | ---: | --- | --- | --- |
| e88_cpu.json | True | e88 | cpu | 1,524,000 | LadderLM | `[35944, 35944]` | `']).]).'` |
| e88_cuda.json | True | e88 | cuda | 1,524,000 | LadderLM | `[35944, 35944]` | `']).]).'` |
| gdn_cpu.json | True | gdn | cpu | 1,998,000 | LadderLM | `[318, 318]` | `' is is'` |
| gdn_cuda.json | True | gdn | cuda | 1,998,000 | LadderLM | `[318, 318]` | `' is is'` |
| m2rnn_cpu.json | True | m2rnn | cpu | 1,467,000 | M2RNNLM | `[6059, 48968]` | `' prin Angola'` |
| m2rnn_cuda.json | True | m2rnn | cuda | 1,467,000 | M2RNNLM | `[6059, 48968]` | `' prin Angola'` |

---

## Validation Checklist (task `## Validation`)

- [x] Selected checkpoint per arch matches the **committed paper** AS_OF step
  (deltas recorded: E88 +750, GDN −1,300, M2RNN +600); availability of the
  **ticket** AS_OF steps confirmed as overwritten and reported above.
- [x] Exact paths, steps, mtimes, sizes, SHA256, raw + smoothed (`trail_100k`)
  BPB recorded.
- [x] Local `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  loads all three with no missing/unexpected/mismatched keys.
- [x] Docker CPU smoke passes for all three; GPU smoke passes (CUDA available on
  NVIDIA RTX 6000 Ada).
- [x] No upload/retag; no tokens/weights/safetensors/PDFs staged or committed.

## Next step (downstream)

`v3-approval-gated` ("V3: approval-gated public HF publish of validated current
checkpoints") is the human-approval-gated publish consumer of this validation.
This task performs **no** HF write. Any publish must reuse the exact selected
checkpoints/SHA256 above so the public v0.3 repos agree with the committed paper.
