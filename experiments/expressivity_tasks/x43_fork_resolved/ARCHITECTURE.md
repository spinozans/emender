# EmenderLM — the architecture as it stands
(Nov 6, 2026. Every factual claim cites the receipt or commit that binds it.
This document describes; PROGRAM.md binds. Where a fact is in flight rather
than settled, it is marked IN FLIGHT with its open receipt.)

## 1. The block

A two-branch residual block — a **memory branch and an MLP branch** — the
same skeleton as a transformer block (attention branch + MLP branch), with
the memory branch's read operation being attention-shaped: the current
hidden state queries stored keys, a similarity-weighted blend of stored
contents is projected and added back into the residual stream.

- Byte-level: vocabulary 256; no tokenization (standing constraint; the
  theory note Part III explains why it is also the correct choice).
- C14 fork-resolved class: embedding width 256, two layers, **2,133,700
  trainable parameters** (emender) / 2,133,748 (transformer control) —
  receipt `receipts/c14_byte/model_shapes.json`.
- Production classes: the velocity program's matched-proxy family (~150M
  class; searches converged and nominated), the terminal 1.3B class
  (nominations committed for transformer 3.1907 / GDN2 3.1675; pure in
  flight; `receipts/velocity_1p3b_nominations/`).
- BF16; AdamW lr 3e-4 / decay .01, clip 1; document-bounded windows; no
  state crosses document boundaries (tested; the standing C14+ constraint).

## 2. The store (the memory branch's substrate)

A bounded set of **slots**, each a (key, content) pair:

- **Keys**: content-derived, through a **frozen semantic basis** — the
  key-embedding basis is fixed at initialization and permanent
  ("init permanent for addressing"). Queries carry **learned coordinates**
  over that frozen basis. Removing the frozen basis hurts (+0.09 NLL;
  mimetic-program receipts); donor-initialized weights help (−0.27;
  `x42_pile_scale/MIMETIC_RECIPE.md`).
- **Content**: slot payloads pass through a **shared trainable content
  projection**.
- **Reads**: content-addressed by similarity — no positional addressing;
  the blend is the memory branch's contribution to the residual stream.
  Per-head budgets (the 32/64 class at the C14 scale; search-space bounds
  32–128 × ratio ≤ 2.85 in the velocity registrations).

A transformer is a degenerate point of this design space: store = the
entire sequence, write-once immutable, learned keys, no budget, no ops
(the C14-B ablation measured the write-once regime inside the emender at
**+0.66 nats** — the ops' causal value: `RESULT_C14_ABLATION_TRIANGLE.md`).

## 3. The five operations (current laws)

| Op | Law (receipt-bound) |
|---|---|
| **APPEND** | Copy state/content into a slot under budget pressure; capacity refusals and capacity merges are receipted events (PB3 telemetry) |
| **CORRECT** | On observed contradiction, update the slot (residual learning). Fires nonzero on every training head (PB3: read hit-rate 1.0, all op classes firing) |
| **POP** | Ownership expiry: a slot dies when its pattern no longer anchors the running context (the suffix-ownership component, `c14_b_suffix.py`, 8 tests) |
| **DECAY** | Staleness weighting — the soft layer on top of ownership expiry |
| **MERGE** | Consolidation of indistinguishable slots; capacity merges under pressure |

## 4. The two consultation doors (external exact knowledge)

The external source: a read-only xsa container (suffix array + BWT; branch
queries return exact next-byte distributions — 256 probabilities plus
count, entropy, fork indicator, and supported-suffix length = the 260
scalars). Serving: 22.7k contexts/s, zero corpus reads at query time
(sxgc 88d4e07 receipts, verified independently by our brute-force passes).

- **The passive input channel (C14)**: the 260 scalars fed as inputs at
  every position, with independent per-position keep=.5 feature dropout.
  Measured: emender 2.276→1.509; transformer 1.404→1.266 (stacked 1.2636);
  the full first-pass table in `RESULT_C14_FIRST_PASS.md`.
- **The store door (C14-B)**: answers APPEND into the store via **native
  Cell keying at the query position** (an array slot is keyed exactly as a
  stream slot would be — same filing system), with the queried suffix's raw
  bytes carried as **slot metadata** for op identity: CORRECT matches
  contradictions by metadata; MERGE consolidates by metadata; POP/DECAY
  run per ownership. Registration + amendments: `PREREGISTRATION_C14_B.md`
  (8196246b, 060445bc, 189a031a).

## 5. The unbounded configuration — concretely

"Unbounded" is the **same five ops with the cap set nonbinding** (257 at
the C14 class; never reached; zero capacity merges). Nothing else changes:
CORRECT/POP/DECAY/MERGE still run; the store settles at its **natural
churn steady state** (~110 live slots at the 2.13M class vs ~48 for the
budgeted arms).

What the receipts say about it:
- **Data efficiency**: 2.3× the passive channel at matched exposure
  (normalized from the raw 9×; `RESULT_C14_DATA_EFFICIENCY.md`).
- **Better at matched exposure** (single seed): the matched twin 1.5653
  vs the budgeted 1.5847 (−0.0194; `RESULT_C14_B_MATCHED.md`); the
  second-seed hardening IN FLIGHT (seed 3402, budgeted arm done,
  unbudgeted eval running).
- **Cost**: ~1.6× the budgeted arm's step cost at 2.13M shapes (8.14 vs
  ~13 steps/s sustained; phases 1–3 receipts) — better than linear in
  live slots. At 1.3B long-context shapes the wall remains: 280 ms/head
  at T1024 after round 2, projection 557 native-only days; the kernel
  program is in round 4 (streaming-within-pages synthesis)
  IN FLIGHT (`RESULT_UNBOUNDED_SCALABILITY_PHASE5*.md`).

## 6. The deployment machinery

- **Fusion/stacking**: soft-gate calibration over the array's
  distributions. Receipted on both heads: fusion adds ~nothing after
  consultation training (1.509→1.506; 1.266→1.2636) — the training
  channel absorbs it. Exact forced walks FAIL on both heads (zero
  probability on 30.75M real targets) — walks need backoff.
- **The spines**: Wikipedia 100 GB (container 42.35 GB, χ/n = 0.156);
  pile-head100 (container 63.16 GB, **χ = 24.19 G, χ/n = 0.242** — the
  Pile is 55% denser in decision contexts; build certificate + our
  independent query verification, 17/17 brute-force exact).
- **The transfer receipt**: on held-out documents beyond the 100 GB
  mark — text the array never saw — every arm's array gain EXCEEDS its
  in-region gain (survival 1.34–1.99×). **The spine's value is
  transfer, not coverage** (`receipts/p100_heldout/decomposition.json`;
  not causally identified — stated in the receipt).

## 7. The training protocol (the standing rules)

- Matched-time: 2700 active seconds per arm at the C14 class; actual
  steps/bytes reported as first-class columns (the stopping-rule ruling,
  `stream_audit.json` lineage); matched-exposure twins where the science
  requires them.
- One seed per arm unless hardened; single-seed limits stated wherever a
  sign is claimed.
- Registrations hash-bound before measurements; the validator defines
  "documented" (PROGRAM.md, 459+ bindings).

## 8. The theory correspondence

The store is a **particle filter over the corpus's decision space**: the
weighted interval set, with the five ops as its set-dynamics (APPEND =
hypothesis enters; CORRECT = evidence reweighting; POP/DECAY = expiry;
MERGE = component collapse), the budget as the beam width, and the xsa
making propagation exact. Full formalism, the LLM-families-as-
approximations mapping, tokenization, and multi-loop-before-emit:
`THEORY_INTERVAL_OPERATORS.md` (Parts I–IV).

## 9. State classes (the 1.3B matrix forming)

| Family | State class | Standing |
|---|---|---|
| Transformer | unbounded KV (per-sequence, write-once) | nominated 3.1907 |
| GDN2 | fixed state | nominated 3.1675 |
| Emender (budgeted) | bounded managed store | search in flight (gen 5+, ~Nov 9) |
| Emender (unbounded) | unbounded managed store | launch decision pending the second-seed sign + kernel program |

## 10. What this document is

A faithful map of the receipted state at the commit where it lands. The
authoritative artifacts are PROGRAM.md (bindings), the PREREGISTRATION
files (protocols), the RESULT files (measurements), and the code. If this
document and a receipt ever disagree, the receipt wins — and this
document gets corrected in the same commit that notices.
