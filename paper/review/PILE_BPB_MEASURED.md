# Measured Bits-Per-Byte on a Held-Out Pile Slice

**Task:** `pile-bpb-measure`. Empirically anchor E88's reported **0.974 bpb** on
The Pile by MEASURING, through ONE identical tokenizer-invariant pipeline on the
SAME held-out byte slice, the bits-per-byte of open Pile-trained models and
reader-familiar out-of-distribution models, and to attempt the same for E88 and
our other v0.3 checkpoints.

**REAL MEASUREMENT ONLY.** Every number is produced by `scripts/measure_pile_bpb.py`
on GPU 0 and transcribed into this report by `scripts/gen_bpb_report.py` directly
from `scripts/.hf_results.json` (no hand-typed numbers). What could not be run is
reported as an exact blocker, not fabricated.

---

## Headline

**1. Where 0.974 sits among real same-corpus models.** The open Pile-trained
references span **0.7157–0.7423** BPB on this slice — below 0.974.
This is expected and *not* a like-for-like loss: Pythia/GPT-Neo were trained on
the **entire** Pile, so this slice is **in-distribution (effectively training
data) for them**, while 0.974 is E88's own train-loss. On identical bytes,
dedicated ~1.3B Pile transformers compress this text to ~0.72–0.74
bpb; E88's 0.974 train-loss is a higher number, but the two are on different
distributions and slightly different byte-normalizations — indicative context,
not a leaderboard.

**2. E88 / GDN / M2RNN-CMA held-out BPB — attempted, BLOCKED (reported honestly).**
The production checkpoints were located on local disk and **load strict (0
missing / 0 unexpected keys)** — e.g. E88 at
`/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1530000_loss_2.5965.pt`
rebuilds to exactly 1.273B params via `elman.models.ladder_lm.LadderLM(level="E88",
dim=1664, depth=12, n_heads=370, n_state=32, gate_activation="silu",
e88_decay_mode="mamba", state_expansion=2)`. **But the available eval code did not
reproduce a valid forward pass:** both Triton and PyTorch-fallback paths, in
fp16/bf16/fp32, returned ~17.6 nats/token — *worse than uniform-random
(ln 50281 ≈ 10.83 nats)* — on a slice block where the model's own training loss
is ~2.6. A strict state-dict load matches key names and shapes, not the
computation graph; the `E88FLAHybrid` recurrence forward as invoked standalone is
structurally mismatched to how these weights were trained (a config flag that
does not change parameter shapes but does change compute — e.g. `use_silu`,
`use_l2_norm`, decay parameterisation). Rather than publish a fabricated or
worse-than-random number, **E88/GDN/M2RNN-CMA held-out BPB is reported as NOT
MEASURED — blocked on reproducing the `E88FLAHybrid` eval forward.** Their
reported train-loss bpb (E88 0.974, GDN 0.977, M2RNN-CMA 0.980) stands as the
paper's own figure. A follow-up task should run the eval inside the live training
harness (`/home/erikg/elman` validation path / the exact `create_ladder_model`
config the racer used) where the forward is known-good.

> **RESOLVED (2026-06-01, task `measure-e88-gdn`):** that follow-up ran. The
> blocker was the schedule-free optimizer saving x-mode (eval) weights, not a
> forward-graph mismatch; applying the y-mode swap yields a sane forward.
> **Measured held-out BPB: E88 = 0.9661, GDN = 0.9661, M2RNN = 0.9613.** See the
> "UPDATE (2026-06-01)" section at the end of this document.

---

## Results — measured tiers (identical pipeline, identical bytes)

### Tier A — open Pile-trained reference (~1–1.4B)

| Model | Params | Held-out Pile BPB | PPL/token | Bytes/token | Tokens |
|-------|-------:|------------------:|----------:|------------:|-------:|
| `EleutherAI/pythia-1.4b` | 1.42B | **0.7157** | 7.26 | 4.00 | 2,501,813 |
| `EleutherAI/gpt-neo-1.3B` | 1.32B | **0.7403** | 6.35 | 3.60 | 2,777,009 |
| `EleutherAI/pythia-1b` | 1.01B | **0.7423** | 7.82 | 4.00 | 2,501,813 |

These models trained on the full Pile, so this slice is in-distribution for them
(effectively a train-loss measurement, not held-out).

### Tier B — reader-familiar, NOT Pile-trained (out-of-distribution anchor)

| Model | Params | Held-out Pile BPB | PPL/token | Bytes/token | Tokens |
|-------|-------:|------------------:|----------:|------------:|-------:|
| `facebook/opt-1.3b` | 1.32B | **0.8615** | 8.59 | 3.60 | 2,777,009 |
| `gpt2-xl` | 1.56B | **1.0137** | 12.55 | 3.60 | 2,777,009 |

Included as recognizable familiarity anchors; a higher BPB for OOD models is
expected and is not a quality knock. gpt2-xl and facebook/opt-1.3b measured at context=1024/stride=512 (gpt2-xl max position = 1024); Pile-trained tier at context=2048/stride=1024.

**Cross-check against published numbers.** Measured **gpt2-xl = 1.0137** on this slice vs the Pile paper's published GPT-2 XL zero-shot Pile *test* BPB of 1.0468 (Gao et al. 2020, Table 2): ours is ~0.03 lower, quantifying how much easier this single contiguous slice is than the full diverse Pile test set (plus a protocol difference — sliding-window byte-normalized NLL here vs their per-document protocol). Absolute levels here read low across the board; the **relative, identical-bytes comparison is the robust result.**

**Failures / models that could not run:**
- none

---

## Method (fully reproducible)

- **Hardware:** GPU 0 only (`CUDA_VISIBLE_DEVICES=0`, hard-pinned before importing
  torch). GPUs 1–7 (racer training) untouched. dtype float16.
- **Corpus:** `/mnt/nvme2n1/erikg/pile.txt` (~1.3 TB UTF-8). No official Pile val/test split is
  mirrored locally (only train shards), and **no comma-pile / common-pile corpus
  was located** — held-out source is an offset slice of the master corpus.
  **comma-pile sample not located.**
- **Held-out slice:** byte offset 1,000,000,000,000 (~77% into the file), whole-line
  slice → **9,999,511 bytes**, sha256 `3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a`. Exact byte range and
  reproduction recorded in `paper/review/heldout_slice.json` (shared verbatim with
  the parallel compression task). SAME bytes feed every model → identical byte
  denominator → BPB comparable across tokenizers.
- **Why held-out:** the racer training streams <1 epoch from the file start; a
  slice ~1 TB in is the region least likely consumed. Best-effort; the Pile-trained
  references saw all of the Pile in training, so for them it is in-distribution.
- **Protocol:** context 2048, sliding window stride 1024
  (every token scored once with up to 2047 tokens of left context;
  standard HF fixed-length-model perplexity recipe). Each model uses its OWN
  tokenizer; the denominator is the shared UTF-8 byte count.
- **Tokenizer-invariant BPB:** `bpb = (sum NLL nats / ln2) / total_utf8_bytes`.

### Files
- `scripts/measure_pile_bpb.py` — measurement pipeline.
- `scripts/gen_bpb_report.py` — regenerates this report from the JSON.
- `scripts/.hf_results.json` — raw machine-written results.
- `scripts/.pile_heldout_slice.txt` — the exact held-out bytes.
- `paper/review/heldout_slice.json` — slice descriptor for cross-task reuse.

### Reproduce
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/measure_pile_bpb.py \
  --models EleutherAI/pythia-1.4b EleutherAI/gpt-neo-1.3B EleutherAI/pythia-1b \
  gpt2-xl facebook/opt-1.3b \
  --cache scripts/.pile_heldout_slice.txt --out scripts/.hf_results.json
CUDA_VISIBLE_DEVICES=0 python3 scripts/gen_bpb_report.py
```

---

## Validation checklist (from the task)

- [x] Ran on **GPU 0 only**; training GPUs (1–7) untouched.
- [~] E88 held-out bpb vs 0.974: **attempted** — checkpoint located and loads strict
  (1.273B), but the `E88FLAHybrid` eval forward could not be reproduced through the
  available code (worse-than-random ~17.6 nats); reported as exact blocker.
- [x] **3 open Pile-trained ~1–1.4B models** measured through the identical
  pipeline.
- [x] Reader-familiar OOD tier measured (author req).
- [x] Every number really measured; the unmeasurable models reported with the exact blocker.
- [x] Tokenizer-invariant byte-normalized bpb; method fully documented.
- [x] `paper/review/heldout_slice.json` written for the parallel compression task.
- [x] comma-pile second distribution: **not located**.
- [x] `paper/review/PILE_BPB_MEASURED.md` written; `paper/main.typ` NOT modified.

---

## UPDATE (2026-06-01, task `measure-e88-gdn`) — E88 / GDN / M2RNN held-out BPB: **MEASURED** (blocker resolved)

The `pile-bpb-measure` blocker ("`E88FLAHybrid` eval forward returns
worse-than-random ~17.6 nats") is **resolved**. The cause was **not** a
structural mismatch in the forward graph — it was the **schedule-free
optimizer's saved weights**. These runs use `AdamWScheduleFree`, which saves the
**x-mode** (eval-extrapolated) parameters. Those x-mode weights are catastrophic
at inference (~17–20 nats); the usable **y-mode** (training) weights are
recovered by loading the optimizer state and calling `optimizer.train()` — the
exact swap `generate.load_model` performs (`generate.py:130–147`). The prior
attempt loaded only `model_state_dict` (the x-mode weights) → garbage.

These held-out numbers were produced **inside the elman harness** with the
**live training kernel**: each model is rebuilt exactly as `train.py` does
(same per-architecture branch, all flags from the run's `args.json`,
`use_triton=1` for E88 / XMA Triton backend for M2RNN, `r_h_mode` auto-resolved
to `none` as `train.py` does), `model_state_dict` loaded **strict (0 missing /
0 unexpected)**, then the **y-mode swap** applied. A **block-loss sanity gate**
(mean NLL on the first 2048-token block) was required to pass **before** any BPB
was trusted.

**Sanity gate (passed for all three; vs the prior ~17.6 nats):**

| Model | Block loss (nats/token) | Ckpt train loss (nats) |
|-------|------------------------:|-----------------------:|
| E88     | **1.809** | 2.597 |
| fla-gdn | **1.602** | 2.730 |
| m2rnn   | **1.711** | 2.735 |

### Held-out BPB — our v0.3 checkpoints (identical slice, identical denominator)

Same protocol and **same 9,999,511-byte slice** (sha256 `3e4241a9…`,
`paper/review/heldout_slice.json`) as the tiers above: **context 2048, stride
1024**, each model's own **p50k_base** tokenizer, byte denominator shared, dtype
bf16, **GPU 0 only**.

| Model | Level | Params | Ckpt step | Held-out Pile BPB | PPL/token | nats/token | Bytes/token | Tokens |
|-------|-------|-------:|----------:|------------------:|----------:|-----------:|------------:|-------:|
| E88     | E88     | 1.273B | 1,542,000 | **0.9661** | 12.93 | 2.5598 | 3.822 | 2,616,009 |
| GDN     | fla-gdn | 1.352B | 2,031,000 | **0.9661** | 12.93 | 2.5597 | 3.822 | 2,616,009 |
| M2RNN   | m2rnn   | 1.307B | 1,491,000 | **0.9613** | 12.77 | 2.5470 | 3.822 | 2,616,009 |

**E88 train(0.974) vs held-out delta (the task's headline number):**

| Model | Paper train BPB | Held-out BPB (this slice) | Δ (held-out − train) |
|-------|----------------:|--------------------------:|---------------------:|
| E88     | 0.974 | 0.9661 | **−0.0079** |
| GDN     | 0.977 | 0.9661 | −0.0109 |
| M2RNN   | 0.980 | 0.9613 | −0.0187 |

All three held-out BPBs land **slightly below** (better than) the paper's own
train-loss BPB figures. This is consistent and **not** a contradiction: the same
cross-check that validated this pipeline showed this single contiguous slice is
**~0.03 BPB easier** than the full diverse Pile test set (measured
**gpt2-xl = 1.0137** here vs the Pile paper's published **1.0468**). A held-out
slice that is ~0.03 easier than the corpus average comfortably explains a
~0.008–0.019 improvement over the reported train figure. Two independent
caveats on the Δ: (i) the 0.974/0.977/0.980 figures are the paper's own
train-loss-derived BPB and may use a slightly different bytes/token
normalization than the **3.822 bytes/token** measured on this slice; (ii) these
are live checkpoints from the still-running convergence races (E88 train loss
2.597, GDN 2.730, M2RNN 2.735 at the steps used), not the final converged
weights. The robust statement is the **identical-bytes, identical-protocol**
measurement: **E88 = 0.9661, GDN = 0.9661, M2RNN = 0.9613 held-out Pile BPB.**

### How these relate to the tiers above

- vs **open Pile-trained references** (pythia-1.4b 0.7157, gpt-neo-1.3B 0.7403,
  pythia-1b 0.7423): those models trained on the **entire** Pile, so this slice
  is **in-distribution** (effectively train data) for them; for our models,
  trained <1 epoch from the file start, a slice ~1 TB in is genuinely
  **held-out**. The gap is expected and is the in-distribution-vs-held-out gap,
  not purely an architecture-quality gap.
- vs the **OOD anchor** facebook/opt-1.3b (0.8615, measured at ctx 1024): not a
  clean comparison (different context); reported for orientation only.

### Method notes / reproducibility

- **Schedule-free y-mode swap is mandatory** for any inference/eval on these
  checkpoints. Loading `model_state_dict` alone yields ~17.6 nats.
- **Checkpoint provenance.** The task originally named E88 step `1,530,000`
  (loss 2.5965); that file was **rotated out by the still-running training job**
  mid-eval (checkpoint rotation). Step **1,542,000** (loss **2.5970**, the same
  convergence point, Δloss = 0.0005) was used instead. All three checkpoints
  were **copied to a stable directory** before eval to be immune to rotation.
- **Batched-window validation.** Windows are independent (each a fresh forward
  with its own left context), so the sweep batches 8 windows/forward to use the
  GPU. This was verified bit-identical to batch=1: GDN 0.9661 (batched) =
  0.9661 (batch=1); M2RNN 0.9613 = 0.9613.
- **GPU isolation.** `CUDA_VISIBLE_DEVICES=0` hard-pinned before importing
  torch; training GPUs 1–7 untouched.

#### Files
- `scripts/measure_pile_bpb_elman.py` — in-harness eval (faithful `train.py`
  construction + y-mode swap + identical sliding-window BPB protocol).
- `scripts/.elman_bpb_results.json` — raw machine-written results (the numbers
  in the tables above are transcribed verbatim from this file).

#### Reproduce
```bash
# E88 (use_triton=1) / GDN / M2RNN (XMA), GPU 0 only:
CUDA_VISIBLE_DEVICES=0 XMA_PATH=/home/erikg/xma python3 scripts/measure_pile_bpb_elman.py \
  --checkpoint <ckpt.pt> --name <e88|fla-gdn|m2rnn> --batch-size 8 \
  --slice /home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt \
  --out /tmp/<name>.json
# args.json must sit alongside <ckpt.pt> (the run's exact config).
```
