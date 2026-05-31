# The Bits-Per-Byte Landscape — One Consolidated, Honest Table

**Task:** `bpb-full-table`. Read-only synthesis of the neural eval
(`pile-bpb-measure` → `paper/review/PILE_BPB_MEASURED.md`) and the classical
compression baselines (`pile-compression-bench` → `paper/review/COMPRESSION_BPB.md`)
onto the **single byte-identical held-out Pile slice**
(`paper/review/heldout_slice.json`). **`paper/main.typ` is NOT modified by this task.**

Every number below is traced to one of those three source files. Nothing here is
re-measured or hand-fitted; where a number could not be produced it is reported as
an exact blocker, not fabricated.

---

## READ THIS FIRST — the table is TWO different things

This document deliberately presents the numbers as two separate objects. Conflating
them is the single most likely misreading, so they are labelled explicitly and the
load-bearing one is named.

- **(A) THE ABSOLUTE LANDSCAPE** — every model and every compressor on the held-out
  Pile slice, lined up by BPB. **CONTEXT ONLY.** Heavily caveated below. The absolute
  ordering here is **NOT** an architecture ranking — it is dominated by *training
  budget* and *train/test distribution*, two confounds that swamp the update-rule
  signal.

- **(B) THE MATCHED-BUDGET THREE-WAY** — E88 vs GDN vs M2RNN-CMA, identical
  architecture family, identical protocol, identical token budget. **This is the
  load-bearing result.** It is the only comparison in this document where the *only*
  variable is the update rule, so it is the only one that speaks to an architecture
  question. Everything in (A) exists to give (B) a scale, not to compete with it.

### Two caveats that must be read before the absolute column (A)

1. **UNDERTRAINING CAVEAT.** E88 (and its GDN / M2RNN-CMA siblings) saw on the order
   of **~15–16 B tokens** of training. The outside baselines in tier A2/A3 below
   (Pythia, GPT-Neo, OPT, GPT-2 XL) saw **10–20× more** (Pythia/Neo: ~300 B Pile
   tokens; GPT-2/OPT: hundreds of B). **Therefore any absolute-BPB gap between our
   models and the outside models is a TRAINING-BUDGET difference, not an architecture
   verdict.** A reader must not read "E88 0.974 sits above Pythia 0.7157" as "E88's
   update rule is worse than Pythia's" — they are not on the same training budget and
   not even on the same train/test footing (next caveat).

2. **CONTAMINATION / DISTRIBUTION ASYMMETRY.** Pythia-1.4B, Pythia-1B and
   GPT-Neo-1.3B were trained **on the Pile**. This held-out slice is an offset region
   of the same master corpus, so for those models it is effectively
   **in-distribution training data** (best-effort held-out; possible direct
   contamination). Their 0.72–0.74 is therefore close to a *train-loss*, not a true
   held-out generalization number. The reader-familiar models (OPT-1.3B, GPT-2 XL)
   are **out-of-distribution** w.r.t. the Pile, so their higher BPB is *expected and
   is not a quality knock*. **Only the matched three-way (B) is simultaneously
   contamination-clean and budget-clean** (all three share the exact same corpus
   exposure and the exact same budget), which is precisely why it is the load-bearing
   comparison.

### METHOD — what "BPB" means here (no 3.92 constant)

Every **measured held-out** BPB in this document is computed tokenizer-invariantly
from the model's **actual NLL** over the slice and the slice's **actual UTF-8 byte
count**:

```
BPB = (Σ NLL_nats / ln 2) / total_UTF8_bytes
    = actual_NLL_nats / (actual_UTF8_bytes × ln 2)
```

with the **shared denominator `total_UTF8_bytes = 9,999,511`** for every model
(identical bytes → identical denominator → BPB comparable across tokenizers). This is
**NOT** the paper's `0.974`-style normalization, which uses a *fixed global*
`bytes_per_token = 3.918625` ("the 3.92 constant", per `BPB_CONTEXT.md` L51–54). The
held-out column uses each model's **own** measured tokens over the real bytes, so the
"bytes/token" you see varies per tokenizer (4.00 for the GPT-NeoX BPE, 3.60 for the
GPT-2 BPE) and is a *measured* quantity, not a constant. Exact tokens scored
(all ≫ 1 M) and bytes/token are reported per model in the table.

The **0.974 / 0.977 / 0.980** figures for E88 / GDN / M2RNN-CMA are the paper's own
**train-loss** bpb (100 K-step trailing training loss under the fixed-3.92
normalization). They are flagged as train-loss everywhere they appear and are **not**
held-out measurements — see the blocker below.

---

## (B) THE MATCHED-BUDGET THREE-WAY — the load-bearing result

Same architecture family (`LadderLM`, ~1.27 B params), same protocol, same ~15–16 B
token budget; the **only** variable is the update rule. This isolates the question the
paper actually asks.

| Rank | Model | Update rule | Params | Train-loss BPB | Held-out BPB | Tokens (held-out) |
|-----:|-------|-------------|-------:|---------------:|:-------------|:------------------|
| 1 | **E88** | E88 (mamba-decay hybrid) | 1.273 B | **0.974** | **NOT MEASURED** (blocked) | — |
| 2 | GDN | gated-delta | ~1.27 B | 0.977 | NOT MEASURED (blocked) | — |
| 3 | M2RNN-CMA | M2RNN + CMA | ~1.27 B | 0.980 | NOT MEASURED (blocked) | — |

**Matched ordering (train-loss): E88 0.974 < GDN 0.977 < M2RNN-CMA 0.980.** Because
all three are matched on budget *and* corpus *and* normalization, this train-loss
ordering is a *clean* read of the update-rule variable — the 3.92 constant and the
train-vs-test confound apply **equally to all three**, so they cancel in the
within-family comparison. E88 is ahead by **0.003 bpb** over GDN and **0.006** over
M2RNN-CMA. (These deltas are small; the paper should not over-claim them, but the
ordering is the variable-isolating result.)

> **HELD-OUT THREE-WAY IS BLOCKED — stated plainly.** The production checkpoints were
> located on local disk and **load strict (0 missing / 0 unexpected keys, rebuild to
> exactly 1.273 B params)**, but the available eval code **did not reproduce a valid
> forward pass**: both Triton and PyTorch-fallback paths, in fp16/bf16/fp32, returned
> **~17.6 nats/token — worse than uniform-random** (ln 50281 ≈ 10.83 nats) on a block
> whose train loss is ~2.6. A strict state-dict load matches key *names and shapes*,
> not the *computation graph*; the `E88FLAHybrid` recurrence forward as invoked
> standalone is structurally mismatched to how the weights were trained (a config flag
> — e.g. `use_silu` / `use_l2_norm` / decay parameterisation — that changes compute
> without changing parameter shapes). Rather than publish a worse-than-random or
> fabricated number, the three-way held-out BPB is reported as **NOT MEASURED**. A
> follow-up must run the eval **inside the live training harness** (the exact
> `create_ladder_model` config the racer used) where the forward is known-good.

### Consequence for the headline "0.974 vs measured held-out delta"

The task asks for "E88 train-0.974 vs measured held-out delta." **That delta cannot
currently be computed**, because E88's held-out BPB is NOT MEASURED (blocker above).
This is the honest core finding: we have E88's **train-loss** (0.974) but **not** its
held-out generalization, so the train→held-out gap for E88 is **unknown**. What we
*can* bound it with is the analogous gap for the in-tree open models, where the same
pipeline produced both an in-distribution (≈train-loss-like) number and the
literature's held-out number — see the gpt2-xl cross-check (1.0137 measured vs 1.0468
published) as a calibration that this pipeline reads ~0.03 bpb *lower* than the full
Pile test set on this single contiguous slice. So when the E88 eval forward is fixed,
expect its single-slice held-out number to read slightly *optimistic* relative to a
full-Pile-test number, not pessimistic.

---

## (A) THE ABSOLUTE LANDSCAPE — context only, heavily caveated

All numbers on the **same** held-out slice (sha `3e4241a9…`, 9,999,511 bytes). **Read
top-to-bottom as "what compresses these exact bytes best," NOT as an architecture
ranking** (see the two caveats above: budget + contamination confounds dominate).

| Tier | Name | Type / params | Held-out BPB | Trained on Pile? | Tokens scored | Bytes/tok | Notes |
|------|------|---------------|:------------:|:----------------:|:-------------:|:---------:|-------|
| **A1 — our models (matched family)** | **E88** | LadderLM E88, 1.273 B | **0.974†** | yes (~15–16 B tok) | n/a (train-loss) | 3.92 (fixed) | †**train-loss**, fixed-3.92 norm; held-out NOT MEASURED (blocked). Load-bearing comparison is (B). |
| A1 | GDN | LadderLM gated-delta, ~1.27 B | 0.977† | yes (~15–16 B tok) | n/a (train-loss) | 3.92 (fixed) | †train-loss; held-out NOT MEASURED. |
| A1 | M2RNN-CMA | LadderLM M2RNN+CMA, ~1.27 B | 0.980† | yes (~15–16 B tok) | n/a (train-loss) | 3.92 (fixed) | †train-loss; held-out NOT MEASURED. |
| **A2 — open Pile-trained ref** | `EleutherAI/pythia-1.4b` | transformer, 1.42 B | **0.7157** | **yes (full Pile, ~300 B tok)** | 2,501,813 | 4.00 | In-distribution → ≈train-loss for it, NOT held-out. Possible contamination. ctx2048/stride1024, fp16. |
| A2 | `EleutherAI/gpt-neo-1.3B` | transformer, 1.32 B | **0.7403** | **yes (full Pile)** | 2,777,009 | 3.60 | In-distribution; same caveat. |
| A2 | `EleutherAI/pythia-1b` | transformer, 1.01 B | **0.7423** | **yes (full Pile)** | 2,501,813 | 4.00 | In-distribution; same caveat. |
| **A3 — reader-familiar OOD** | `facebook/opt-1.3b` | transformer, 1.32 B | **0.8615** | **no (OOD)** | 2,777,009 | 3.60 | OOD anchor; higher BPB expected, not a quality knock. ctx1024/stride512. |
| A3 | `gpt2-xl` | transformer, 1.56 B | **1.0137** | **no (OOD)** | 2,777,009 | 3.60 | OOD anchor + pipeline validator: 1.0137 here vs Pile-paper published 1.0468 (Gao 2020 Tbl 2) → pipeline reads ~0.03 low on this easy slice. ctx1024/stride512. |
| **A4 — classical compressors** | xz -9 | LZMA, model-free | **2.1898** | n/a | whole slice | n/a | **Best classical.** 2,737,100 B, 3.653×. |
| A4 | xz -9e | LZMA extreme | 2.1902 | n/a | whole slice | n/a | 2,737,672 B — *marginally larger* than -9; reported as measured. |
| A4 | zstd --ultra -22 | FSE/Huffman | 2.2494 | n/a | whole slice | n/a | 2,811,634 B; ~no gain over -19. |
| A4 | zstd -19 | FSE/Huffman | 2.2500 | n/a | whole slice | n/a | 2,812,348 B, 3.556×. |
| A4 | bzip2 -9 | BWT, model-free | 2.3683 | n/a | whole slice | n/a | 2,960,226 B, 3.378×. |
| A4 | gzip -9 | DEFLATE, model-free | 2.8029 | n/a | whole slice | n/a | 3,503,416 B, 2.854×. |
| **A5 — comma-pile** | — | second distribution | — | — | — | — | **NOT LOCATED.** No comma-pile / common-pile corpus on disk; second-distribution check not run (reported by both upstream tasks). |

**Reading the tiers (the only safe reads):**
- *Within A4* (compressors): a fully like-for-like ranking — xz beats zstd beats bzip2
  beats gzip on these bytes.
- *Within A2* and *within A3*: like-for-like (same budget regime, same train/test
  footing) — Pythia-1.4B leads the Pile-trained group; OPT leads the OOD group.
- *Within A1 = comparison (B)* — the only architecture-isolating read.
- **Across tiers A1↔A2↔A3: NOT a clean comparison.** The budget confound (caveat 1)
  and the contamination/distribution confound (caveat 2) both point the same way and
  both inflate the apparent gap. Do not read it as architecture quality.

---

## Compression-ratio framing — E88 vs the best classical coder

Bits-per-byte converts directly to an equivalent whole-corpus compression ratio
(`ratio = 8 / bpb`, i.e. how many× smaller than the raw 8-bit bytes):

| Object | BPB | Equiv. compression ratio (8 / BPB) |
|--------|----:|----------------------------------:|
| E88 (train-loss) | 0.974 | **8.21×** |
| Best open Pile-trained (pythia-1.4b, held-out/in-dist) | 0.7157 | 11.18× |
| Best classical (xz -9) | 2.1898 | **3.65×** |
| gzip -9 | 2.8029 | 2.85× |

**Framing:** E88's learned model captures the slice at an equivalent **8.21×** vs the
best general-purpose compressor's **3.65×** — i.e. E88 models roughly **2.25× more of
the structure** than xz, the strongest model-free coder tested (8.21 / 3.65 ≈ 2.25; or
equivalently 2.1898 / 0.974 ≈ 2.25× fewer bits/byte). This is the intended
"LM-as-compression" message: a learned model captures far more statistical structure
than a dictionary/entropy coder. **Caveat:** this is **not** a like-for-like contest —
E88's 0.974 is a *train-loss* under the fixed-3.92 normalization while xz is a
*streaming* general compressor over the raw bytes; the classical numbers are a
**model-free floor**, not a competitor on the same protocol. Read the 2.25× as "the
learned model is well clear of the generic-redundancy floor," not as a benchmark
victory.

---

## Slice consistency check (sha256) — neural ⟂ compression

The task requires confirming both pipelines scored the **same bytes**. They did:

| Source file | Reports slice sha256 | Bytes | Offset |
|-------------|----------------------|------:|-------:|
| `heldout_slice.json` | `3e4241a946e76c31…77fe25a` | 9,999,511 | 1,000,000,001,956 |
| `PILE_BPB_MEASURED.md` (neural) | `3e4241a946e76c31…77fe25a` | 9,999,511 | (offset 1e12 → newline start) |
| `COMPRESSION_BPB.md` (classical) | `3e4241a946e76c31…77fe25a` | 9,999,511 | 1,000,000,001,956 |

**RESULT: CONSISTENT.** All three files carry the identical sha256
`3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a` and the identical
byte length `9,999,511`. The compression task **independently re-extracted** the slice
from `/mnt/nvme2n1/erikg/pile.txt` (seek 1e12 → next newline → read 10 MB → trim to
last newline) and its re-hash matched the neural eval's on-disk cache and run-log
sha **before** any compression ran. The neural and classical BPB numbers are therefore
on byte-identical input.

> **One transcription mismatch FLAGGED (resolved, non-blocking).** The
> `pile-bpb-measure` WG progress message at 20:44 reported sha **`ddafac3e…`**. This
> does **not** match the authoritative `3e4241a9…`. Per `COMPRESSION_BPB.md` L116–119,
> `ddafac3e…` is a **transcription slip in the progress message only**; the
> authoritative `scripts/.bpb_run.log` and the on-disk cache file both read
> `3e4241a9…`, which is what was actually scored and what was compressed. The
> committed artifacts are internally consistent; the slip lives only in a chat log.
> Flagged here as required.

---

## Provenance — every number to a source

| Number(s) | Source file |
|-----------|-------------|
| E88/GDN/M2RNN train-loss 0.974/0.977/0.980; held-out NOT-MEASURED blocker; ~17.6-nats forward; ckpt load-strict 1.273 B | `PILE_BPB_MEASURED.md` §Headline 2, L28–48, L125–127 |
| pythia-1.4b 0.7157 / gpt-neo-1.3B 0.7403 / pythia-1b 0.7423 (+ tokens, bytes/tok, PPL) | `PILE_BPB_MEASURED.md` Tier A, L56–60 |
| opt-1.3b 0.8615 / gpt2-xl 1.0137; gpt2-xl vs published 1.0468 cross-check | `PILE_BPB_MEASURED.md` Tier B, L67–75 |
| Method formula `BPB = (ΣNLL/ln2)/bytes`; ctx/stride; tokenizer-invariance | `PILE_BPB_MEASURED.md` Method L82–102 |
| 3.92 fixed-norm vs per-doc; train-loss-vs-test confound | `BPB_CONTEXT.md` L46–62 |
| gzip 2.8029 / bzip2 2.3683 / xz -9 2.1898 / xz -9e 2.1902 / zstd-19 2.2500 / zstd-22 2.2494 (+ compressed bytes, ratios, tool versions) | `COMPRESSION_BPB.md` result table L15–22 |
| Slice sha / bytes / offset; re-extraction provenance; ddafac3e slip | `heldout_slice.json`; `COMPRESSION_BPB.md` L75–119 |
| comma-pile NOT LOCATED | `PILE_BPB_MEASURED.md` L88–89, L134 |

No number in this document originates anywhere but these files. Nothing is fabricated;
the two unmeasurable objects (E88/GDN/M2RNN held-out, comma-pile) are reported as
explicit blockers, not filled in.

---

## Bottom line

1. **The architecture question is answered only by (B), the matched three-way:** on
   train-loss, **E88 (0.974) < GDN (0.977) < M2RNN-CMA (0.980)** — a clean,
   confound-cancelling read of the update-rule variable, by small margins.
2. **E88's held-out BPB is NOT MEASURED** (eval-forward blocker), so the
   train→held-out **delta is currently unknown** and must not be invented. Fix the
   `E88FLAHybrid` eval forward in the live harness to close this.
3. **The absolute landscape (A) is context, not a ranking.** E88's 0.974 sitting above
   the Pile-trained 0.72–0.74 reflects **10–20× less training budget** and those
   models' **in-distribution** advantage — a budget/contamination artifact, not an
   architecture verdict.
4. **vs the model-free floor**, E88 compresses ~**2.25×** better than the best
   classical coder (xz -9), the intended LM-as-compression message — read as
   "well clear of generic redundancy," not a like-for-like win.
5. **Slice integrity holds:** all neural + classical numbers are on the byte-identical
   slice `3e4241a9…` (9,999,511 B); the only mismatch is a chat-log transcription slip
   (`ddafac3e…`), flagged and resolved.

*`paper/main.typ` was NOT modified by this task. This is a read-only synthesis.*
