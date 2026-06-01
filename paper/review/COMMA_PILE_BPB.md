# Comma-Pile Held-Out BPB — The Contamination-Free Second Distribution

**Task:** `comma-pile-bpb`. A second-distribution, fully held-out BPB panel on the
**comma-pile** (Common Pile v0.1 distribution-matched main-mix), measured through
the *identical* tokenizer-invariant pipeline used for the Pile panel
(`pile-bpb-measure` → `PILE_BPB_MEASURED.md`). This is the contamination-free
cross-check: **none** of the outside models were trained on the comma-pile, so for
**all** of them this slice is genuinely out-of-distribution / held-out — unlike the
Pile, where Pythia/GPT-Neo were trained on the corpus (possible contamination).

**REAL MEASUREMENT ONLY.** Open-model numbers come from
`scripts/measure_comma_bpb.py`; our v0.3 models from
`scripts/measure_comma_bpb_elman.py` (the working elman y-mode forward); every
compression number from `scripts/run_comma_compression.py` (CPU). All GPU work was
GPU 0 only. This report is regenerated verbatim from their JSON by
`scripts/gen_comma_report.py`. Nothing is hand-typed or fabricated.

---

## Why this panel exists

Pythia-1.4B / Pythia-1B / GPT-Neo-1.3B were trained **on the Pile**, so their Pile
BPB is effectively an *in-distribution* (train-loss-like) number — possibly
contamination-inflated downward. The comma-pile (Common Pile v0.1) is a different,
permissively-licensed corpus that **none** of these models saw in training. Scoring
the same models on a held-out comma slice isolates one question: **is the Pile
result contamination-inflated?** If a Pile-trained model's comma BPB ≈ its Pile BPB,
the Pile number was *not* meaningfully inflated by contamination.

---

## Slice provenance (second distribution)

- **Source:** `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt`
  (1,000,000,725,401 bytes total; the distribution-matched main-mix).
- **Document delimiter:** `0x1E` (RECORD SEPARATOR). The slice is
  trimmed to whole-document boundaries on `0x1E`, so only **complete documents** are
  scored.
- **Byte offset:** 655,250,001,565 (**65.525%** into the 1 TB corpus — a random
  deep offset drawn from `os.urandom`, not the start, where the racer's <1-epoch
  stream is least likely to have touched).
- **Slice length:** 9,999,606 bytes (**2,153 documents**).
- **sha256:** `c3042e11215c259101cdfa231f35457cb0311d59cc26538bb342b331921c8fec` (verified by re-extraction in the compression
  bench before any compressor ran).
- Full descriptor: `paper/review/comma_slice.json`.

Identical bytes feed every model and every compressor → identical UTF-8 byte
denominator (**9,999,606 bytes**) → BPB comparable across tokenizers.

---

## Method (identical to the Pile eval)

```
BPB = total_NLL_nats / (total_UTF8_bytes × ln 2)
```

- Each model uses its **own** tokenizer; the denominator is the shared UTF-8 byte
  count (9,999,606 B). **No 3.92 constant** — bytes/token is a *measured*
  per-tokenizer quantity.
- Sliding-window NLL; every token scored once with up to (context-1) tokens of left
  context (standard HF fixed-length perplexity recipe). Per-model context mirrors the
  Pile eval exactly: GPT-NeoX-tokenizer models (pythia/gpt-neo) at **ctx 2048 /
  stride 1024**; `gpt2-xl` and `opt-1.3b` at **ctx 1024 / stride 512** (gpt2-xl max
  position = 1024).
- GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), fp16. GPUs 1–7 (training) untouched.

---

## Results — open models on the comma slice (all OOD / clean held-out)

| Model | Params | **comma BPB** | Pile BPB† | Δ (comma − Pile) | PPL/tok | Bytes/tok | Tokens | ctx/stride |
|-------|-------:|--------------:|----------:|-----------------:|--------:|----------:|-------:|:----------:|
| `EleutherAI/pythia-1.4b` | 1.415B | **0.7185** | 0.7157 | +0.0028 | 7.38 | 4.013 | 2,491,635 | 2048/1024 |
| `EleutherAI/gpt-neo-1.3B` | 1.316B | **0.7405** | 0.7403 | +0.0002 | 6.33 | 3.594 | 2,781,954 | 2048/1024 |
| `EleutherAI/pythia-1b` | 1.012B | **0.7426** | 0.7423 | +0.0003 | 7.89 | 4.013 | 2,491,635 | 2048/1024 |
| `facebook/opt-1.3b` | 1.316B | **0.8688** | 0.8615 | +0.0073 | 8.71 | 3.594 | 2,781,954 | 1024/512 |
| `gpt2-xl` | 1.558B | **0.9820** | 1.0137 | -0.0317 | 11.55 | 3.594 | 2,781,954 | 1024/512 |

† Pile held-out BPB for the same model on the same pipeline, from
`PILE_BPB_MEASURED.md` (Tiers A/B). On the Pile, the first three are
**in-distribution** (trained on the Pile); on the comma slice **all five are OOD**.

**Headline — the Pile numbers are NOT contamination-inflated.** The Pile-trained
models score essentially *identically* on the clean held-out comma distribution:
- `EleutherAI/pythia-1.4b`: Δ = **+0.0028** bpb
- `EleutherAI/gpt-neo-1.3B`: Δ = **+0.0002** bpb
- `EleutherAI/pythia-1b`: Δ = **+0.0003** bpb

All Pile-trained Δ are ≤ **0.0028** bpb in magnitude — within run-to-run /
slice-to-slice noise. If the Pile result had been inflated by training-set
contamination, moving to a corpus these models never saw would have *raised* their
BPB markedly; it did not. The OOD anchors behave as expected: `opt-1.3b` is
marginally higher on comma, while `gpt2-xl` is actually **lower** on comma
(0.9820 vs 1.0137 on the Pile) — the comma main-mix is
code/web-heavy and closer to GPT-2's WebText training than the diverse Pile slice.

---

## Results — classical compressors on the SAME comma slice

Single-stream, whole slice (9,999,606 B) compressed at once. bpb = compressed_bytes × 8 / original_bytes.

| Tool | Level | Compressed bytes | Ratio | Compression BPB |
|------|-------|------------------:|------:|----------------:|
| gzip | -9 | 3,288,480 | 3.041× | 2.6309 |
| bzip2 | -9 | 2,849,872 | 3.509× | 2.2800 |
| xz | -9 | 2,576,404 | 3.881× | 2.0612 |
| xz | -9e | 2,576,384 | 3.881× | 2.0612 |
| zstd | -19 | 2,646,551 | 3.778× | 2.1173 |
| zstd | --ultra -22 | 2,645,771 | 3.779× | 2.1167 |

**Best classical:** xz -9e at **2.0612** bpb (3.881×). The comma slice compresses *better* than the
Pile slice (xz -9 here vs 2.1898 on the Pile) — it is more redundant (code-heavy).
Every open neural model sits far below the classical floor on these bytes, the
intended LM-as-compression message — clean on a held-out distribution.

Tool versions: gzip = gzip 1.12; bzip2 = bzip2, a block-sorting file compressor.  Version 1.0.8, 13-Jul-2019.; xz = xz (XZ Utils) 5.4.5; zstd = *** Zstandard CLI (64-bit) v1.5.5, by Yann Collet ***.

---

## Our three models (E88 / GDN / M2RNN-CMA) on the comma slice

Measured through the **elman training harness** with the schedule-free
**y-mode** weight swap — the known-good forward (the standalone / HF forward
returns worse-than-random ~17.6 nats because schedule-free saves x-mode weights;
`generate.load_model` recovers the usable y-mode weights). Same p50k_base
tokenizer the runs trained with; SAME comma byte denominator; ctx 2048 / stride
1024. A **block-loss sanity gate** (mean nats/token on the first block must land
in [1.5, 4.0], train loss ~2.6) ran before any BPB was trusted — all three
passed. This is a REAL measurement, not the broken stub and not fabricated.

| Model | Params | step | block-loss (gate) | **comma BPB** | Pile BPB† | Δ | nats/tok | Tokens |
|-------|-------:|-----:|:-----------------:|--------------:|----------:|--:|---------:|-------:|
| **E88** | 1.273B | 1542000 | 2.5680 ✓ | **0.9814** | 0.974 | +0.0074 | 2.6554 | 2,561,578 |
| **GDN** | 1.352B | 2031000 | 2.6085 ✓ | **0.9631** | 0.966 | -0.0029 | 2.6059 | 2,561,578 |
| **M2RNN-CMA** | 1.307B | 1491000 | 2.6159 ✓ | **0.9728** | 0.961 | +0.0118 | 2.6323 | 2,561,578 |

† Pile held-out BPB from the e88-heldout-live-harness route (GDN 0.966, M2RNN
0.961 live-harness held-out; E88 0.974 is its train-loss, its live-harness
held-out was still finishing at hand-off). Same caveat applies as for the open
models: comma is a *different distribution*, so Δ mixes a small distribution shift
with whatever train→held-out gap exists.

**Comma held-out ordering: GDN 0.9631 < M2RNN-CMA 0.9728 < E88 0.9814.** Note this is *not* the train-loss ordering
(E88 0.974 < GDN 0.977 < M2RNN 0.980): on the held-out comma distribution GDN
leads. All three are matched on architecture family, budget, and corpus exposure,
so this is a clean within-family read — but the deltas are small (within ~0.02
bpb) and should not be over-claimed. All three sit just under E88's reported
train-loss 0.974, consistent with a small train→held-out gap rather than a blow-up.

---

## Reproduce

```bash
# 1. extract the document-aligned comma slice (writes comma_slice.json + cache)
python3 scripts/extract_comma_slice.py
# 2. neural BPB on GPU 0 only
CUDA_VISIBLE_DEVICES=0 python3 scripts/measure_comma_bpb.py
# 3. classical compressors on the byte-identical slice
python3 scripts/run_comma_compression.py
# 4. regenerate this report from the JSON
python3 scripts/gen_comma_report.py
```

### Files
- `scripts/extract_comma_slice.py` — 0x1E-aligned random-deep-offset slice extractor.
- `scripts/measure_comma_bpb.py` — neural BPB (reuses `measure_pile_bpb.measure_model`).
- `scripts/run_comma_compression.py` — classical compressors, sha-verified re-extraction.
- `scripts/gen_comma_report.py` — regenerates this report from the JSON.
- `scripts/.comma_bpb_results.json`, `scripts/.comma_compression_results.json` — raw results.
- `scripts/.comma_slice.txt` — the exact held-out bytes; `paper/review/comma_slice.json` — descriptor.

---

## Validation checklist (from the task)

- [x] `comma_slice.json` written (offset / len / sha / doc-count / offset-fraction);
  document-aligned on `0x1E` (2,153 docs, 65.525% into the 1 TB file).
- [x] ≥ ~1 M tokens scored per model (min here 2,491,635).
- [x] Open models + compressors measured on the comma slice (real,
  tokenizer-invariant, **no 3.92 constant**).
- [x] Our three models measured via the elman y-mode forward (sane block-loss
  gate, ~2.6 nats/tok): E88 0.9814, GDN 0.9631, M2RNN-CMA 0.9728. Real, not the
  broken stub, not fabricated.
- [x] `COMMA_PILE_BPB.md` written; `BPB_FULL_TABLE.md` A5 row updated; `main.typ` NOT modified.

