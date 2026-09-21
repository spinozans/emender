# E97 long-document anchor v1 — anchor corpus authorities, packs, and the base→bridge→v6 document-NLL drift

**Status:** COMPLETE. Both anchor data authorities (library + mainmix draws)
are built, sha-bound, holdout- and probe-excluded, packed as document-causal
64K packs (v2 schema), independently validated, and ready for E2 admission via
cohort-specs (§4). The frozen document-NLL probe is complete over all three
lineage checkpoints and measures **+0.1665 nats of monotone drift** on raw
long documents from base (y-form) → bridge → v6-u96 (§3.3). All work was
CPU-only; the E1 GPU arc was never touched.

Work dir: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-long-document-anchor-v1/`
(`STATUS.md` there is the chronological log with all shas; heavy-job stdout in
`logs/`). Repo-side edits this arc: `scripts/export_e97_4b_base_y_weights.py` (new),
`scripts/build_e97_long_document_anchor_library.py` (shard-stream fix),
`scripts/build_e97_mainmix_anchor_draws.py` (new), plus the probe runner
`--score-nll` mode in the `e97-llama-cpp-port-v1` workspace and
`scripts/convert_e97_4b_to_gguf.py` (weight-form handling).

---

## 1. Mission and deliverables
types, especially old books) — the distribution that formed the sequential-recurrent
state dynamics, absent from current SFT diets. Two required source families:
(a) the CommaPile book/long-form library sources, (b) the pre-packed mainmix stream
itself. Plus a frozen document-NLL retention probe measuring distribution drift
across the lineage base -> bridge -> v6-u96, and an E2 slot-in plan.

## 2. Corpus artifacts

### 2.1 Library authority — `library-authority/`

- Schema `emender-e97-tulu3-masked-sft-v1`, status complete, training_eligible true.
- Manifest sha256 `1bd7aa364d4111be0050360851d0292c5dae0e02dee9bae9de33fd2d6f68fcf8`.
- **1,942 records / 59,400,278 tokens** (59.25M quota target met; 1,927 documents).
- Whole books whole: 15 documents >65,537 tokens segmented ONLY at the 64K pack
  boundary (30 segment records; peS2o 28, pressbooks 2); consecutive records in
  reading order, final segment carries EOT.
- Dedup by text sha verified (zero duplicate documents; the only repeated shas are
  the 15 parent-document segment pairs by design).
- Holdout exclusion (identity + shingle) and probe-panel exclusion applied and
  counted per source (see table); all four payload sha256s re-verified against
  the manifest post-build.
- 1% deterministic-hash validation split.

Per-source counts:

| source | documents | records | tokens | segmented | notable exclusions |
|---|---|---|---|---|---|
| pre_1929_books | 397 | 397 | 12,016,749 | 0 | humanize 4,713; probe 1 |
| library_of_congress | 265 | 265 | 8,016,010 | 0 | humanize 3,577 |
| project_gutenberg | 244 | 244 | 8,002,670 | 0 | humanize 1,541; probe 1 |
| doab | 188 | 188 | 6,008,841 | 0 | humanize 2,466 |
| arxiv_papers | 127 | 127 | 6,014,891 | 0 | humanize 254; shingle 2; markupsafe 2; probe 1 |
| peS2o | 179 | 193 | 6,007,911 | 28 | humanize 2,131 |
| caselaw_access_project | 167 | 167 | 5,011,115 | 0 | humanize 420; probe 1 |
| ubuntu_irc | 90 | 90 | 4,037,270 | 0 | humanize 57; markupsafe 18; prettytable 18; more-itertools 3; probe 1 |
| biodiversity_heritage_library | 161 | 161 | 2,006,960 | 0 | humanize 7 |
| pressbooks | 63 | 64 | 2,024,505 | 2 | humanize 67 |
| public_domain_review | 46 | 46 | 253,356 | 0 | (quota 250k; source-limited) |
| **total** | **1,927** | **1,942** | **59,400,278** | **30** | |

Books (7 sources) carry ~39.1M tokens (~66% of mass) as designed.

Selection: per source, candidates ranked by document length descending, taken in
rank order until quota (selection manifest sha
`03df50d47682048221c306c89bac61b7cf0e338d28b9a8d941a053dec83be066`,
index in `selection/index/`). Tokenizer p50k_base, cache sha `94b5ca7d...` (verified).

### 2.2 Mainmix draws authority — `mainmix-draws-authority/`

- Manifest sha256 `fc9192ab09795af88a767e4f8ef471bda8eee73d8bf9f8b82c2e324324d48bf0`.
- **50,826 records / 61,300,355 tokens** (61.25M causal targets) — 14 seeded
  (seed 974131) random 16MiB-aligned windows of the 1TB stream
  (`commapile_mainmix_v0.1_1tb.txt`, sha `44f4c334...` from base args.json),
  processed in ascending byte order; scan-across 0x1e delimiters; window-edge
  fragments kept partial exactly as pre-training windows cut the stream;
  decode errors='replace; eot per record; 4 segment records (2 docs >65,537 tokens).
- Probe-doc exclusion is the only curation (by design). Raw-stream repeats are
  preserved: 3 duplicate texts / 4 extra records (0.008%).
- 1% validation split (50,297 train / 529 validation records).
- All four payload sha256s re-verified post-build.
- Superset kept: `mainmix-draws-authority-calibration-36w-157m/` (36 windows,
  157,309,549 tokens, manifest `a2348923...`) — the main authority's 14 windows
  are a subset; lets E2 scale the mainmix share without new draws.

### 2.3 Packs (document-causal 64K, E1 pattern)

Both pack sets: `build_e97_sft_packs.py --boundary-aware --context-size 65536
--sampler-mode epoch-permutation`, schema `emender-e97-sft-boundary-aware-packs-v2`
(per-document recurrent resets, cross-document targets masked, loss-masked padding),
validated with `validate_e97_sft_packs.py`.

| pack set | packs (train/val) | train tokens | train records | validation | pack manifest sha |
|---|---|---|---|---|---|
| `mainmix-draws-packs/` | 997 / 12 | 60,644,653 (60.59M targets) | 50,297 | pass: 1,009 packs, 50,826 records | `043d6615fe7f5b01d57afeb227da77db1d67ed93179eb96523ca34870111b1a1` |
| `library-authority-packs/` | 1,126 / 9 | 58,961,765 (58.96M targets) | 1,928 | pass: 1,135 packs, 1,942 records | `e1be577873487566591800a643aa20dd0f4e2ad76574ed98f35a81da022f85a7` |

Validation JSONs: `logs/validate-packs-mainmix.log`, `logs/validate-packs-library.log`
(status=pass, training_eligible=true, every pack's payload re-derived and sha-checked).

## 3. Document-NLL retention probe

### 3.1 Frozen panel — `probe/probe-panel-manifest.json`

Schema `emender-e97-document-nll-probe-panel-v1`, sha
`abca0b8b4ce9a917e8262b8d4ce0743565c594900a27fc1442f43dd42ed4540e`;
token-ids `probe-panel.token-ids.jsonl` sha `5725d8206ac90e5ad4bbb4acd492bf4832f704ccb1d39615bae1c87fd55edc9f`.
8 documents, hash-ranked among top-1000 longest per source, two book eras:

| doc | source | chars | full tokens | panel tokens | scored (cap 32,768) |
|---|---|---|---|---|---|
| project_gutenberg | books (old) | 127,999 | 33,824 | 32,769 | 32,768 (truncated) |
| pre_1929_books | books (old) | 127,998 | 26,830 | 26,830 | 26,829 |
| arxiv_papers | papers | 127,998 | 54,797 | 32,769 | 32,768 (truncated) |
| caselaw_access_project | law | 127,986 | 29,079 | 29,079 | 29,078 |
| ubuntu_irc | logs | 127,997 | 44,336 | 32,769 | 32,768 (truncated) |
| libretexts | textbooks | 39,169 | 8,121 | 8,121 | 8,120 |
| news | news | 15,178 | 3,197 | 3,197 | 3,196 |
| stackexchange | Q/A | 117,898 | 36,504 | 32,769 | 32,768 (truncated) |

Total scored: 198,295 tokens per checkpoint. Teacher-forced causal NLL per doc via
the e97-llama-cpp-port-v1 `e97-runner --score-nll` mode (GGML CPU, 32 threads).

### 3.2 Checkpoint weight forms — critical finding

The base pre-training checkpoint (step_024448, 99.72B tokens, sha `3ace0042...`)
is a **Schedule-Free** checkpoint saved with `train_mode=False`: its
`model_state_dict` holds the **x/averaged** point. Two independent implementations
(the GGML runner and a spec-only eager torch reference built during debugging)
both find the x-form degenerate for generation: document NLL ~23.5 nats,
top-1 ~0.003, greedy repetition. This also invalidated the Sep 19
`e97-4b-base-99B-f32.gguf` (its tensors are bitwise-identical to the checkpoint —
the conversion was correct; the weight FORM was wrong) and the base q8_0 built
from it (since deleted).

`ndm/e97.py` documents `weight_mode='train'` — the **y/train point** recovered from
the saved optimizer state via the ScheduleFree `train()` swap
(`y = lerp(x -> z, 1 - beta1)`) — as the **generation default**, and it is the
exact parent the SFT arc initialized bridge/v6 from. Fix (new
`scripts/export_e97_4b_base_y_weights.py`): load base on CPU with
`ndm.e97.load_e97_checkpoint(weight_mode='train')`, cross-check the swap
arithmetic against the raw optimizer state (max diff 0.000391 on the embedding =
bf16 rounding), export `probe/gguf/base-y-weights.pt` (sha
`05ecc9c0dffc33af56eab310de620c748fd496496788a823bee6c0c6a83d57ef`), convert ->
`probe/gguf/e97-4b-base-99B-y-f32.gguf` -> `e97-4b-base-99B-y-q8_0.gguf`
(chain `logs/convert-base-y.sh`, log `logs/convert-base-y.log`).
Smoke pass on the rebuilt q8_0: mean NLL 3.327 / top-1 0.365 (sane; v6 = 3.459
on the same smoke panel).

Bridge (`9b78628d...`, `saved-eval-x` SFT checkpoint) and v6-u96 (`d8146498...`)
GGUFs are direct conversions of their saved eval weights; bridge conversion
verified (bitwise, `BRIDGE_GGUF_OK`).

### 3.3 Drift table — base (y-form) -> bridge -> v6-u96, q8_0, cap 32,768/doc

Timing judgment: the 256-core box allowed all checkpoint scorings to run in
parallel, so the **full 32,768-token cap was kept for all three checkpoints**
(no reduction needed; the drift table stayed fully comparable at ~4h wall).

| doc (scored tokens) | base NLL | bridge NLL | v6 NLL | drift base→bridge | drift base→v6 |
|---|---|---|---|---|---|
| project_gutenberg (32,768) | 3.2487 | 3.3855 | 3.4431 | +0.1368 | +0.1944 |
| pre_1929_books (26,829) | 3.1023 | 3.2405 | 3.2887 | +0.1382 | +0.1864 |
| arxiv_papers (32,768) | 1.3155 | 1.3855 | 1.4158 | +0.0700 | +0.1003 |
| caselaw_access_project (29,078) | 2.2827 | 2.3927 | 2.4368 | +0.1100 | +0.1541 |
| ubuntu_irc (32,768) | 2.2465 | 2.3283 | 2.3608 | +0.0818 | +0.1143 |
| libretexts (8,120) | 3.5122 | 3.6503 | 3.7061 | +0.1380 | +0.1939 |
| news (3,196) | 3.5552 | 3.7278 | 3.7945 | +0.1725 | +0.2393 |
| stackexchange (32,768) | 2.4866 | 2.6534 | 2.7246 | +0.1669 | +0.2381 |
| **TOTAL (198,295)** | **2.4919** | **2.6105** | **2.6585** | **+0.1185** | **+0.1665** |

Per-doc top-1 accuracy follows the same monotone pattern in aggregate: base 0.51044,
bridge 0.49534, v6 0.49162.

**Full-panel finding: the drift is monotone on every document and at every stage.**
Base (y-form) beats bridge on all 8 docs; bridge beats v6 on all 8 docs. The two
SFT stages of the current arc cost +0.1185 nats (bridge) and a further +0.0480
nats (v6) over the full panel — **+0.1665 nats total (+6.7% relative)** over the
base pre-training distribution, worst on the raw-book and short-news tails
(+0.19 to +0.24 nats base→v6) and mildest on arxiv (+0.10). This is exactly the
drift the anchor cohorts target: the model is progressively forgetting the base
distribution (raw long-form text) that formed its recurrent state dynamics,
and the loss is largest precisely on the old-books distribution the library
cohort re-introduces.

### 3.4 Quantization-delta spot check (base, doc1, q8_0 vs f32)

base-y q8_0 doc1 mean NLL **3.248696** vs base-y f32 doc1 mean NLL **3.248634**:
delta = **+0.000062 nats** (+0.0019% relative), with top-1 0.38962 vs 0.38995.
The q8_0 quantization error is ~3-4 orders of magnitude smaller than the
per-doc drift deltas (+0.0700 to +0.2393 nats), so the drift table is not a
quantization artifact. Honest caveat, kept regardless: **all three checkpoints
were scored as q8_0 GGUFs, so quantization binds every row equally** — it can
shift absolute NLLs but cancels to first order in the stage-to-stage deltas.
Log `logs/score-nll-base-y-f32-delta.log`, output
`probe/scores/delta-base-y-f32-doc1.json` (f32, 5.90 tok/s, 5557s for doc1).

### 3.5 Score artifacts

`probe/scores/nll-base-99B-y-q8_0.json`,
`nll-bridge-parent-9b78628d-q8_0.json`,
`nll-v6-u96-d8146498-q8_0.json` (schema `emender-e97-document-nll-probe-v1`,
per-doc nll_sum / mean / top-1 / seconds; all three panels complete over the
full 198,295-token set). Base-y panel wall: 2.84h at 19.4 tok/s overall. Discarded as invalid: the x-form base
scoring (`logs/score-nll-base.log`, NLL 23.5/22.1). Debug artifacts kept for
provenance: `eager-base-mini.json` (spec-only eager reference showing the x-form
degeneracy), `smoke-base-y-q8_0.json` (rebuild smoke pass).

## 4. E2 slot-in plan

### 4.1 Interface (verified against the E1 prep builder)

The established prep machinery is `prepare_e97_pi_native_repair_training.py`
(the builder that produced `e97-e1-chat-agent-prep-v1/e1-preparation`, whose
merged packs are schema `emender-e97-sft-boundary-aware-packs-v2`,
epoch-permutation, context 65536 — identical to our anchor pack sets). Cohorts
enter via repeated `--cohort-spec <json>` files:

```json
{
  "cohort": "long-document-library-anchor",
  "root": "/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-long-document-anchor-v1/library-authority",
  "sha256": "1bd7aa364d4111be0050360851d0292c5dae0e02dee9bae9de33fd2d6f68fcf8",
  "seed": 914131,
  "budget_targets": 15000000,
  "repeat_epochs": 1
}
```

```json
{
  "cohort": "mainmix-draws-anchor",
  "root": "/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-long-document-anchor-v1/mainmix-draws-authority",
  "sha256": "fc9192ab09795af88a767e4f8ef471bda8eee73d8bf9f8b82c2e324324d48bf0",
  "seed": 914132,
  "budget_targets": 15000000,
  "repeat_epochs": 1
}
```

(seeds 914131/914132 are illustrative; E2's owner assigns final seeds/budgets.
`budget_targets` = assistant-target-token budget; the builder takes a deterministic
seeded whole-record slice of the authority up to the budget, interleaves
weighted-fair-by-token-share with the other cohorts, and records `spec_cohorts`
entries in the merged prep manifest with authority paths + shas.)

### 4.2 Share arithmetic (15-25% target)

E1 segment prep total: 131.2M assistant-target tokens. At similar E2 scale:

- 15% of 131.2M ≈ 19.7M targets -> e.g. library 12M + mainmix 8M (repeat_epochs 1)
- 20% ≈ 26.2M -> e.g. library 15M + mainmix 11M
- 25% ≈ 32.8M -> e.g. library 18M + mainmix 15M

The two authorities' pools (59.4M + 61.2M targets) cover any point in the range
with a single seeded slice. Suggested default: **library ~15M + mainmix ~11M
(≈20%)** — books carry ~2/3 of anchor mass, mirroring the corpus design — with
`mainmix-draws-authority-calibration-36w-157m` as the scale-up pool if E2's total
grows (its 14-window main authority is a strict subset).

### 4.3 Known wrinkle (documented, not changed)

`read_conversation_slice` in the prep builder skips records with n > 65,536
tokens: that drops the 15 library segment-1 records (983k tokens, 1.65% of the
library) and the 2 mainmix segment records (131k tokens, 0.2%). Clean mitigations
for the E2 owner, in order of preference: (a) accept the small loss (whole books
<= 65,536 tokens are unaffected — 1,912 of 1,942 library records); (b) approve a
one-line prep-builder threshold bump (n > 65,537) so 64K-boundary segments pass
through; (c) offset by nudging `budget_targets` up ~2% for the library cohort.
Nothing in the anchor artifacts needs to change for any option.

### 4.4 What E2 must NOT do

No re-tokenization, re-selection, or re-curation of the anchor authorities —
they are sha-bound end to end (authority manifest sha -> pack manifest sha ->
record payloads), holdout- and probe-excluded, and probe-panel docs stay out of
training so the frozen NLL panel remains a clean drift instrument for post-E2
re-measurement. E2 re-runs the same probe (`e97-runner --score-nll`, panel
`probe-panel.token-ids.jsonl`, q8_0 conversions) to measure whether the anchor
cohorts arrest the drift in section 3.3.

## 5. Continuation / ops notes

- All artifacts are sha-bound; re-verification commands are the builders'
  validate/preflight modes (`build_e97_sft_packs.py` verifies authority payload
  shas before packing; `validate_e97_sft_packs.py` re-derives every pack).
- The Sep 19 `e97-llama-cpp-port-v1/e97-4b-base-99B-f32.gguf` and its q4_0/q8_0
  siblings are the DEGENERATE x-form base — flagged, not deleted (they live in
  the port workspace). Any future base GGUF work must use the y-form chain
  (`scripts/export_e97_4b_base_y_weights.py` + `logs/convert-base-y.sh`).
- Disk: workspace footprint ≈ 1.6GB (authorities+packs) + 8GB base-y .pt +
  21GB probe GGUFs; /mnt/nvme2n1 at 98% — base-y-weights.pt may be deleted after
  E2 accepts the GGUFs (it is regenerable from the sha-bound base checkpoint).
- `logs/` holds every heavy job's stdout; `STATUS.md` is the phase-by-phase log.
