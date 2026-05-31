# BPB Context: Where does 0.974 bpb on The Pile sit?

Contextualizes the paper's headline **E88 = 0.974 bits-per-byte (BPB) on The Pile**
(1.273 B params, after ~23 stitched GPU-days) against real, cited reference numbers.
All numbers below are quoted from their sources; none are fabricated. Unverifiable
items are listed explicitly at the end.

## 1. Reference BPB numbers on The Pile

The canonical source is **The Pile paper** (Gao et al. 2020, *"The Pile: An 800GB
Dataset of Diverse Text for Language Modeling"*, arXiv:2101.00027 — `@thepile2020`
in `paper/refs.bib`). Its **Table 2** reports overall Pile **test BPB** for the
released GPT-2 and (API) GPT-3 models, evaluated **zero-shot** (none fine-tuned on
the Pile), **per-document**, as **bits per UTF-8 encoded byte**, on **one-tenth** of
each test set.

| Model | Params | Pile test BPB | Source | Normalization note |
|---|---|---|---|---|
| GPT-2 Small | 124 M(1) | **1.2253** | Pile paper Table 2 | zero-shot; per-doc bpb = (L_T/L_B)*l/ln2; GPT-2 BPE |
| GPT-2 Medium | 355 M(1) | **1.0928** | Pile paper Table 2 | " |
| GPT-2 Large | 774 M(1) | **1.0828** | Pile paper Table 2 | " |
| GPT-2 XL | 1.5 B(1) | **1.0468** | Pile paper Table 2 | " |
| GPT-3 ada | 2.7 B(2) | **0.9631** | Pile paper Table 2 | zero-shot; per-doc bpb; GPT-2/GPT-3 BPE |
| GPT-3 babbage | 6.7 B(2) | **0.8718** | Pile paper Table 2 | " |
| GPT-3 curie | 13 B(2) | **0.7980** | Pile paper Table 2 | " |
| GPT-3 davinci | 175 B(2) | **0.7177** | Pile paper Table 2 | " |
| **E88 (this paper)** | **1.273 B** | **0.974** | `paper/main.typ` §5, `paper/results/figure_2/` | **train-loss** bpb; fixed 3.918625 B/tok under `p50k_base` |
| GDN (this paper) | 1.352 B | 0.977 | `paper/main.typ` §5 | same protocol/normalization as E88 |
| M²RNN-CMA (this paper) | 1.307 B | 0.980 | `paper/main.typ` §5 | same protocol/normalization as E88 |

(1) GPT-2 per-size parameter counts (124M/355M/774M/1.5B) are the standard public
GPT-2 release sizes (Radford et al. 2019); the Pile paper labels them
small/medium/large/XL but does **not** restate the counts. Treat the param column
for GPT-2 as standard architecture spec, not a Pile-paper claim.

(2) GPT-3 parameter counts are the Pile paper's own stated **assumption**, verbatim:
*"While the sizes of GPT-3 models on the OpenAI API have not been publicized, we
assume here that ada, babbage, curie and davinci models correspond to 2.7B, 6.7B,
13B and 175B parameter models respectively."* This API->size mapping is now widely
believed to be inaccurate (later sources put ada~350M, babbage~1.3B, curie~6.7B,
davinci=175B), so do **not** lean on the GPT-3 param column. The BPB values
themselves are exact API-measured Table-2 figures and stand regardless.

### Methodology / comparability caveat (important)

The Pile paper computes BPB with the **true per-document** token-to-byte ratio
`bpb = (L_T / L_B)*l/ln(2)`, where `L_T` = length in tokens and `L_B` = length in
UTF-8 bytes, using the GPT-2 BPE tokenizer. BPB is "preferred ... due to its
invariance to different tokenization schemes."

This paper's 0.974 uses a **fixed global** conversion: `bpb = nats/token * log2(e) /
bytes_per_token` with `bytes_per_token = 3.918625` (mean over a 2000-window sweep at
`chunk_tokens=2048` under the `p50k_base` BPE; pinned in
`scripts/estimate_tokenizer_bytes_per_token.json`, ~3.92 in the text). `p50k_base`
is the same byte-level GPT-2-family BPE (~50k vocab), so the normalization is
*close* but not identical: (a) a single mean ratio vs per-document ratios, and (b) a
slightly different vocab than GPT-2's tokenizer.

**Two further reasons the cross-paper comparison is only indicative, not a
leaderboard:**
1. **Train-loss vs test BPB.** 0.974/0.977/0.980 are 100K-step trailing **training**
   loss converted to bpb (model trained *on* the Pile). The GPT-2/GPT-3 figures are
   held-out **zero-shot test** BPB (models *not* trained on the Pile). Training loss
   on the training distribution is systematically optimistic relative to held-out
   test loss.
2. **Subset.** Pile-paper numbers used ~10% of the test set, per-document.

## 2. Verdict: is 0.974 "good for the class"?

**Within its own matched-protocol class (the only apples-to-apples comparison),
0.974 is the best of the three.** Against GDN (0.977) and M²RNN-CMA (0.980) — same
corpus, same `p50k_base` normalization, same Schedule-free AdamW / 2048-context
protocol, same train-loss measurement — E88 is the lowest-BPB endpoint, and all
three are sub-1-bpb. This is the comparison that actually licenses a ranking, and it
supports the paper's "good for the class" framing for a ~1.3 B pure-recurrent LM.

**Against GPT-2/GPT-3 the comparison is loose but favorable as order-of-magnitude
context:** 0.974 sits below *every* GPT-2 size's zero-shot Pile test BPB (best GPT-2
is XL at 1.0468) and lands around the GPT-3 "ada"-level (0.9631) on the Pile paper's
scale. Read this as "a 1.3 B single-GPU recurrent model trained ~23 days reaches the
sub-1-bpb regime, below all GPT-2 sizes and near the smallest GPT-3 API tier" —
**not** as a claim of beating those models, because the 0.974 is train-loss under a
fixed-ratio normalization while the GPT-2/GPT-3 numbers are zero-shot per-document
test BPB. The honest one-line summary: **0.974 is a strong, genuinely sub-1-bpb
result that is clearly best-in-class among the paper's matched 1.3 B recurrent
baselines, and broadly competitive with small-to-mid reference transformers on the
Pile once the train-vs-test and normalization caveats are stated.**

## 3. Numbers that could NOT be fully verified / carry caveats

- **GPT-2 per-size parameter counts (124M/355M/774M/1.5B):** standard public GPT-2
  spec, **not** stated in the Pile paper itself. Verified as common GPT-2 knowledge,
  not from `@thepile2020`.
- **GPT-3 ada/babbage/curie/davinci = 2.7B/6.7B/13B/175B:** explicitly the Pile
  paper's *assumption*, and now believed inaccurate. The **BPB values** are verified
  exact; the **param mapping** is flagged unreliable.
- **Direct rank of 0.974 vs GPT-2/GPT-3:** NOT a like-for-like comparison
  (train-loss + fixed-ratio normalization here vs zero-shot per-document test BPB
  there). No verified source places this paper's E88 on the Pile-paper Table-2
  measurement protocol.
- **No other published ~1.3 B recurrent-LM Pile BPB** was located within this task's
  search to serve as an external same-class baseline; the only same-class,
  same-protocol numbers are the paper's own GDN (0.977) and M²RNN-CMA (0.980). If an
  external comparable is needed, it remains unverified/uncollected here.

## Sources

- The Pile paper, Gao et al. 2020 — arXiv:2101.00027 (`@thepile2020`):
  - [abstract](https://arxiv.org/abs/2101.00027) ·
    [HTML/Table 2](https://ar5iv.labs.arxiv.org/html/2101.00027) ·
    [PDF](https://arxiv.org/pdf/2101.00027)
- This paper's numbers: `paper/main.typ` §5 (`<fig_lm_racers>`),
  `paper/results/figure_2/`, `scripts/estimate_tokenizer_bytes_per_token.json`.

*main.typ was NOT modified by this task.*
