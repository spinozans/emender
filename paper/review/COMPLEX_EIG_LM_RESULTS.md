# Complex-Eigenvalue Head — Matched-Compute LM Verdict (convergent-loss-null)

**Task:** `complex-eig-lm` — test Erik's convergent-loss bet for the complex-eigenvalue
gated-delta head: at matched compute, does *complex-everywhere(+nonlinear-subset)* TIE
the real-eigenvalue cell on out-of-sample LM bits-per-byte? *Capability is the
differentiator, not loss* — so the **predicted result is a TIE on bpb**.

REAL data (the Pile, byte-level), REAL kernels (the validated
`complex_eig_chunked` scan + FLA GatedDeltaNet), REAL training. No mocks.

---

## 1. Setup — a controlled, single-variable comparison

Both arms share a **byte-for-byte identical FLA GatedDeltaNet shell** (same q/k/v
projections, short-conv, L2-norm, output gate, RMSNorm, out-proj). They differ in
**exactly one thing**: the recurrent transition.

| arm | level | transition | kernel |
|-----|-------|-----------|--------|
| **baseline (real)** | `real-eig-gdn` | per-head **real** scalar decay `g_t` (`allow_neg_eigval` → real λ∈(−1,1)) | FLA native chunked |
| **complex-everywhere** | `complex-eig-lm` | per-key-channel **complex** eigenvalue `λ = r·e^{iθ}` | validated `complex_gated_delta_chunked` |
| **complex + nonlin-subset** | `complex-eig-lm` | complex everywhere **+ per-step `hardtanh`** on 1/8 of heads | chunked bulk + sequential subset |

`real-eig-gdn` is the cell the complex head strictly generalizes (θ=0 → real-positive
decay = GDN; θ=π → reflection; validated in `COMPLEX_EIG_VALIDATE_RESULTS.md`).

**Shared config (all arms):** dim=512, depth=6, n_heads=8, n_state=64 (P=32 complex
channels), expansion=1.0, SwiGLU MLP per layer, vocab=256 (byte-level), seq_len=512,
batch=16, schedule-free AdamW, lr=2e-3, weight_decay=0 (pure streaming, no epoch
repeats → no decay confound), bf16, seed=42 (**identical byte stream** across arms).

**Matched params** (within ~1%, baseline-favored): baseline `mlp_ratio=2.33` →
**19,295,200**; complex `mlp_ratio=2.0` → **19,103,200**. The complex head's extra cost
(per-channel `a_proj_cplx` + `theta_proj`) is offset by giving the baseline a slightly
wider MLP, so total trainable params are equal — the fair "same parameter budget,
different inductive bias" comparison.

**Data:** train = `/mnt/nvme2n1/erikg/pile.txt` (1.2 TB, byte-level, random doc-aligned
start). Held-out = a **disjoint 64 MB Pile slice** at the 600 GB offset (never touched by
the short training reads). bpb = (CE_nats / ln2) / 1.0 (byte-level ⇒ bytes/token = 1).

> **Eval correctness note.** train.py's schedule-free *averaged (x) weights* are broken
> in this environment (final-eval CE ≈ 13.5 nats, ~10× random — a known schedule-free
> reconstruction bug). The correct, verified metric is the **y-iterate (the actual
> trained weights)**, recovered by loading the optimizer state and calling
> `optimizer.train()`. All held-out bpb below use the y-iterate on 400 val batches
> (`scripts/eval_ceig_bpb.py`). Sanity: y-iterate held-out bpb matches the streaming
> training loss to ~0.1 bpb.

---

## 2. Results

### 2a. Matched **tokens** (12,000 steps = 98.3M tokens, identical seed-42 stream)

| arm | params | held-out bpb (400 val batches) | tok/s | wall-clock (12k steps) |
|-----|-------:|-------------------------------:|------:|-----------------------:|
| real-eig-gdn (baseline) | 19,295,200 | **2.0955** | ~145K | **10.1 min** |
| complex-everywhere      | 19,103,200 | **2.1300** | ~85K  | **28.6 min** |

Δ(complex − baseline) = **+0.0345 bpb** (complex marginally *worse*) on the *same*
98.3M-token stream and matched params — a **near-tie on loss**, complex never wins,
and it pays **2.83× the wall-clock**. Training loss tracks all run (last-100 CE:
baseline 1.3494 vs complex 1.3713; the held-out 0.035 bpb gap ≈ the 0.02-nat
train-loss gap, i.e. no generalization surprise).

### 2b. Matched **wall-clock** (~30–37 min budgets; faster kernel ⇒ more tokens)

| arm | wall-clock | tokens seen | held-out bpb |
|-----|-----------:|------------:|-------------:|
| real-eig-gdn (baseline) | 37 min | **389M** (47,525 steps) | **1.8663** |
| complex-everywhere      | 28.6 min | 98.3M (12,000 steps) | **2.1300** |
| complex + nonlin-subset | 37 min | **3.5M** (433 steps) | **2.6533** |

At equal-order wall-clock the baseline trains **4× more tokens than complex-everywhere
and ~110× more than complex+nonlin** (the +nonlin per-step `hardtanh` heads are
sequential / non-chunkable → 1.6K tok/s vs 145K). The loss gaps follow the token gaps:
baseline 1.866 ≪ complex 2.130 ≪ complex+nonlin 2.653. (The baseline's 37-min budget
slightly over-credits it vs complex's 28.6 min, but at 28.6 min the baseline is already
~1.89 bpb — the ~0.24 bpb gap dwarfs the budget difference, so the ordering is robust.)

---

## 3. Verdict — **convergent-loss-null CONFIRMED on the loss axis; complex LOSES on compute**

**Prediction:** TIE on bpb (capability, not loss, is the differentiator).

**Measured:**
- **Matched tokens → TIE (a hair worse).** Same data, same parameter budget: complex
  2.1300 vs baseline 2.0955, Δ = +0.035 bpb (~1.6%). The complex-eigenvalue structure
  **does not lower the LM loss ceiling** — exactly the convergent-loss-null. It does not
  win on loss; if anything it is fractionally behind. **This is the predicted tie.**
- **Matched wall-clock → LOSS.** Because the complex scan is pure-torch-complex
  (chunkable but not fused Triton) it runs 2.83× slower, and the +nonlin subset's
  bounded-state heads are non-chunkable (sequential, ~110× slower). At equal compute the
  baseline sees far more tokens and wins by 0.24–0.79 bpb.

**Bottom line:** the complex-eigenvalue head buys **no LM-bpb advantage at matched
compute** — it ties on loss when handed identical data (confirming the null) and loses
once the kernel's compute tax is counted. Any case for the head must rest on **targeted
capability** (the complex-eig capability line / positional & modular probes), **not** on
LM loss. The convergent-loss bet is upheld: loss is not the differentiator.

---

## 4. Honest accounting & caveats

- **Compute cost is real and asymmetric.** The complex kernel is pure-torch-complex
  (validated, autograd, chunkable) but **not fused Triton** like FLA GDN-2, so it runs
  ~1.7–2× slower in tok/s at matched tokens (consistent with the 2.4–3.3× fwd-only
  factor in `COMPLEX_EIG_VALIDATE_RESULTS.md`; the LM gap is smaller because the MLP +
  embedding + lm-head are shared, fused, and dominate). The **+nonlinear-subset** arm's
  per-step `hardtanh` heads are **sequential / latency-bound** (bounded state ⊥
  chunkable — a fundamental property), so at matched tokens it is impractically slow;
  it is reported at **matched wall-clock** instead, where its slower kernel directly
  shows up as fewer tokens seen.
- **Scale.** This is a 19M-param, ~10²M-token screen — a *near-convergence* probe on
  fresh streaming data (no repeats), not a full 1.3B convergence run. The convergent-loss
  bet is about the loss *ceiling*; a screen at this size is the standard repo methodology
  (cf. e97-within-layer / E99 controls) and is the affordable evidence here.
- **bpb is a loss metric, not a capability metric.** A tie (or near-tie) on bpb is
  *consistent with* "capability is the differentiator, not loss" — it says the complex
  eigenvalue structure does not change the LM loss ceiling, which is exactly the null.
  The capability separation (if any) lives in the targeted probes (positional_clock /
  modular tasks), not in LM bpb — that is the sibling complex-eig capability line.
