# The Recurrent-State Nonlinearity: Saturating vs Non-Saturating — Literature, Capability Mapping, and a Recommendation for the e97 Cell

**Task:** `recurrent-nonlin-research` · **Date:** 2026-06-08 · research + synthesis (NO training run; grounds every empirical claim on committed project results). `paper/main.typ` NOT edited.

**Question.** We default the recurrent state map to `tanh`. `tanh` **saturates**. Is that the right elementwise nonlinearity for our target capabilities — counting, latching, state-tracking, nonlinear composition — and if not, what is? Is a *single* recurrent nonlinearity fundamentally limited, and does the e97 cell need a **two-pathway** (saturating gate + non-saturating accumulator) design — and is that compatible with our fused/chunked kernels?

**Bottom line (stated up front).**
1. The honest axis is **not** "linear vs nonlinear." It is the **shape of the state map**: *saturating* (tanh), *non-saturating-rectifying* (relu/softplus), *affine-signed* (identity/linear). Each buys a **different** computational regime, and on a **single** matrix-state cell they **conflict** — exactly our E88 finding, and exactly the saturated-RNN hierarchy of Merrill et al. 2020.
2. A recurrently-applied **elementwise nonlinearity of any shape (tanh OR relu OR softplus) breaks chunkability** — it has no chunked-matmul form because pointwise-then-mix is not associative. This is a hard, measured constraint in our kernels, not an opinion. So "swap tanh→relu on the state map" is a **double penalty** for an LM cell: it costs finite-state expressivity (E88) *and* it costs throughput (no chunked kernel).
3. The LSTM/GRU resolution — and the modern linear-attention resolution — is the same: **keep the state-to-state map LINEAR (chunkable) and put all the nonlinearity in input-driven gates + a bounded readout.** The unbounded *counter* lives in a linear additive cell; the *latch/control* lives in saturating gates that are functions of the input, never composed recurrently through the state. State-*tracking* is recovered not by a nonlinear state map but by **negative/complex eigenvalues** in the linear map (Grazzi et al. 2025) — which our `gdn-neg` head already exploits.
4. **Recommendation for the e97 cell: do NOT put the nonlinearity on the recurrent state.** Keep the chunkable linear `e97_delta` backbone, recover track via `gdn-neg`, and add the LSTM motif as a **bounded readout** (`tanh`/`silu` on the *output* `Sᵀq`, post-scan) plus optional **dedicated unbounded "counter" heads** (forget-gate pinned near 1). Concrete configs in §7. A per-step relu/softplus state map should be kept **only as a sequential expressivity probe** to measure the counting ceiling — it is not a production LM lever (the project already measured the chunkable-shell compromise flat 0.88× throughput with no token-matched edge and no capability uplift, losing wall-clock to `gdn2-mlp`).

---

## 1. The saturation trade-off, precisely

Write the matrix-state update in the form our code uses (`ndm/models/e88_fla_hybrid.py:1877`):

```
S_t = f( decay_t · S_{t-1} + outer(write_t, read_t) )
```

The elementwise map `f` is the lever. Three shapes, three regimes:

| `f` | shape | bound on \|S\| | what it is good for | what it costs |
|---|---|---|---|---|
| `tanh` | **saturating** | bounded (≤1) | bistable latching, fixed-point attractors | cannot count (state can't grow); squashes the signed structure groups need |
| `identity`/`linear` | affine, **signed** | unbounded (decay-limited) | finite-state / group word-problems (S5/S3); chunkable | counts worst; cannot, alone, do inherently-sequential tracking unless eigenvalues go negative |
| `relu`/`softplus` | **non-saturating, rectifying** | unbounded above | unbounded counting / accumulation | clamps the negative half-space → destroys reflection structure (S5); not chunkable |

The two physical facts that drive everything:

- **Counting needs an unbounded state.** To recognize `aⁿbⁿcⁿ` the model must hold a count that grows with `n`; a bounded (saturating) state has a fixed ceiling and *cannot* count past it. This is the Weiss–Goldberg–Yahav 2018 separation (§2).
- **Latching needs a bounded, attracting state.** To hold a flag indefinitely you want a stable fixed point; saturation gives you exactly that, but Bengio–Simard–Frasconi 1994 proved the price: robust latching forces the dynamics into a regime where **gradients vanish**.

So saturation is simultaneously *what you want* for latching and *what you must avoid* for counting. A single elementwise `f` cannot be both bounded and unbounded. That is the trade-off, and it is structural, not a tuning artifact.

---

## 2. Literature (all citations web-verified — title/authors/year/arXiv|DOI confirmed)

### 2.1 Counting needs NON-saturating / unbounded state

- **Weiss, G.; Goldberg, Y.; Yahav, E. (2018). "On the Practical Computational Power of Finite Precision RNNs for Language Recognition." ACL 2018 (Short), pp. 740–745. arXiv:1805.04908, DOI 10.18653/v1/P18-2117.**
  The headline separation: **LSTM and the ReLU/IRNN simple-RNN can implement unbounded counting** and recognize `aⁿbⁿ`, `aⁿbⁿcⁿ`; **GRU and squashing (sigmoid/tanh) simple-RNNs cannot count** — they are limited to bounded, finite-state behavior in finite precision. This is the canonical "non-saturating ⇒ counting, saturating ⇒ no counting" result.
- **Le, Q. V.; Jaitly, N.; Hinton, G. E. (2015). "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units." arXiv:1504.00941.**
  The **IRNN**: a ReLU recurrent net with the recurrent matrix initialized to the (scaled) identity. The non-saturating ReLU activation plus identity-init gives the unbounded, accumulation-friendly dynamics Weiss 2018 leans on — and matches LSTM on several long-range tasks. This is the "identity/linear init" fix that makes a non-saturating recurrence trainable.
- **Suzgun, M.; Gehrmann, S.; Belinkov, Y.; Shieber, S. M. (2019). "LSTM Networks Can Perform Dynamic Counting." ACL 2019 Workshop (Deep Learning and Formal Languages). arXiv:1906.03648.**
  Empirically, LSTMs learn to emulate real-time **k-counter machines** (Dyck-1 and shuffles), a *single* LSTM unit sufficing for Dyck-1 — the cell-state pathway is literally a learned counter.
- **Merrill, W.; Weiss, G.; Goldberg, Y.; Schwartz, R.; Smith, N. A.; Yahav, E. (2020). "A Formal Hierarchy of RNN Architectures." ACL 2020, pp. 443–459. arXiv:2004.08500.**
  The **saturated-RNN capacity hierarchy**: ranks architectures by space complexity and rational recurrence. Formalizes that the **LSTM is a strict counter** (not rational-recurrent), separating it from QRNN/GRU. This is the theory backbone for "which nonlinearity ⇒ which class."

### 2.2 Latching / stability / gradients favor SATURATING — but at a cost

- **Hochreiter, S.; Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation 9(8):1735–1780. DOI 10.1162/neco.1997.9.8.1735.**
  The **two-pathway** archetype: a *constant-error-carousel* **additive (linear, non-saturating) cell state** `c` for memory/counting, gated by **multiplicative saturating gates** (sigmoid) and read out through a **saturating squash** (`tanh`) — `o_t ⊙ tanh(c_t)`. The non-saturating part counts; the saturating part controls and bounds the readout.
- **Bengio, Y.; Simard, P.; Frasconi, P. (1994). "Learning Long-Term Dependencies with Gradient Descent is Difficult." IEEE Trans. Neural Networks 5(2):157–166. DOI 10.1109/72.279181.**
  Proves the **latching ⊥ trainability** tension: to robustly latch (store a bit over long lag) the recurrence must sit where its Jacobian contracts, which is **exactly where gradients vanish**. The reason you cannot just crank up `tanh` saturation to get long memory.
- **Pascanu, R.; Mikolov, T.; Bengio, Y. (2013). "On the difficulty of training Recurrent Neural Networks." ICML 2013 (PMLR v28:1310–1318). arXiv:1211.5063.**
  The exploding/vanishing-gradient analysis via the recurrent Jacobian spectral radius, and gradient-norm clipping. Explains why a *non-saturating* (relu) state — unbounded magnitude — is prone to **exploding** state/gradients (the mirror failure mode of tanh's vanishing).
- **Cho, K.; van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; Bengio, Y. (2014). "Learning Phrase Representations using RNN Encoder–Decoder…" EMNLP 2014. arXiv:1406.1078.** and **Chung, J.; Gulcehre, C.; Cho, K.; Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv:1412.3555.**
  Define the **GRU** and show gated cells (LSTM/GRU, with a non-saturating carry) **beat a plain `tanh` recurrent unit**. (Note: GRU lacks LSTM's *separate* unbounded cell, which is why Weiss 2018 finds GRU *cannot* count while LSTM can — the carry is convex-interpolated and bounded.)

### 2.3 Init / normalization fixes — make a non-saturating or norm-preserving recurrence trainable

- **Arjovsky, M.; Shah, A.; Bengio, Y. (2016). "Unitary Evolution Recurrent Neural Networks." ICML 2016 (PMLR v48:1120–1128). arXiv:1511.06464.** Recurrent matrix parameterized **unitary** (eigenvalues of magnitude exactly 1) ⇒ no vanishing/exploding gradients **without** a saturating squash (uses `modReLU`). The "norm-preserving instead of saturating" route.
- **Henaff, M.; Szlam, A.; LeCun, Y. (2016). "Recurrent Orthogonal Networks and Long-Memory Tasks." ICML 2016 (PMLR v48). arXiv:1602.06662.** Explicit orthogonal-RNN constructions for long-memory; how orthogonal init stores information over many steps.
- **Vorontsov, E.; Trabelsi, C.; Kadoury, S.; Pal, C. (2017). "On orthogonality and learning recurrent networks with long term dependencies." ICML 2017 (PMLR v70:3570–3578). arXiv:1702.00071.** **Hard** orthogonality can *hurt*; **soft/bounded** deviation trades stability for optimization speed — i.e. you want eigenvalues near, but not pinned to, the unit circle.
- **Ba, J. L.; Kiros, J. R.; Hinton, G. E. (2016). "Layer Normalization." arXiv:1607.06450.** Per-step normalization of recurrent pre-activations — the standard way to keep a non-saturating (relu) state well-conditioned and bounded *in scale* without bounding it in *range*. (Our cells use RMSNorm in the same role.)

### 2.4 Modern linear-attention: the resolution is "linear chunkable state + gates," not a nonlinear state map

- **Gu, A.; Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752 (COLM 2024).** Selective (input-dependent) **linear** SSM with a hardware-aware **chunked parallel scan**. The nonlinearity is in the input-dependent `Δ, B, C` *projections*, not in a recurrent state map.
- **Yang, S.; Wang, B.; Shen, Y.; Panda, R.; Kim, Y. (2023/24). "Gated Linear Attention Transformers with Hardware-Efficient Training" (GLA). arXiv:2312.06635 (ICML 2024).** Data-dependent **gates** on a **linear** state, with the chunked FlashLinearAttention kernel. Same recipe.
- **Yang, S.; Kautz, J.; Hatamizadeh, A. (2024). "Gated Delta Networks: Improving Mamba2 with Delta Rule." arXiv:2412.06464 (ICLR 2025).** The **gated-delta** cell — gate (erase) + delta rule (targeted write) inside *one chunkable linear pathway*. This is our GDN-2 backbone and our `gdn-neg` recall head.
- **Beck, M.; Pöppel, K.; Spanring, M.; Auer, A.; …; Hochreiter, S. (2024). "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517 (NeurIPS 2024).** The cleanest modern statement of the **two-pathway split at architecture scale**: **sLSTM** (scalar memory, exponential gating, memory-mixing, *sequential / non-parallelizable*) vs **mLSTM** (matrix memory, covariance update, *fully parallelizable / chunkable linear*). xLSTM literally interleaves a sequential nonlinear-control block and a parallel linear-memory block — because no single block does both well. **This is the LSTM two-pathway lesson, re-derived for the kernel era, and it is the template for our recommendation.**

### 2.5 State-tracking expressivity — linearity, not saturation, is the binding limit; negative eigenvalues fix it

- **Siegelmann, H. T.; Sontag, E. D. (1995). "On the Computational Power of Neural Nets." J. Comput. Syst. Sci. 50(1):132–150. DOI 10.1006/jcss.1995.1013.** Finite recurrent **sigmoidal** nets with unbounded precision are **Turing-complete** — recurrent nonlinearity + unbounded precision is maximally expressive *in principle* (the practical-precision results above are the realistic refinement).
- **Merrill, W.; Petty, J.; Sabharwal, A. (2024). "The Illusion of State in State-Space Models." ICML 2024. arXiv:2404.08819.** Linear/diagonal SSMs (S4, Mamba) are stuck in **TC⁰** and **cannot** solve inherently-sequential state-tracking like **S5** permutation composition. Crucially, the limit is **linearity of the transition**, not lack of saturation.
- **Grazzi, R.; Siems, J.; et al. (2025). "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues." ICLR 2025. arXiv:2411.12537.** Widen the transition eigenvalue range from `[0,1]` to **`[-1,1]`** and a **linear** RNN solves parity and S5 — **state-tracking without any nonlinear state map.** This is precisely the mechanism behind our `gdn-neg` head.
- **Siems, J.; Carstensen, T.; Zela, A.; Hutter, F.; Pontil, M.; Grazzi, R. (2025). "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products." arXiv:2502.10297.** Multiple delta steps per token ⇒ diagonal-plus-rank-`n` transitions (products of Householder reflections) ⇒ richer state-tracking, still a **structured linear** transition (and still chunkable).

**Reading of 2.4 + 2.5 together:** the field's answer to "how do I get counting AND latching AND tracking AND parallel kernels" is **not** "find a better recurrent nonlinearity." It is **(a)** keep the state-transition **linear** so it chunks, **(b)** carry the nonlinearity in **input-driven gates** and a **bounded readout**, **(c)** buy state-tracking with **negative/complex eigenvalues and richer (Householder) linear transitions**, and **(d)** where you truly need a sequential nonlinear recurrence, **isolate it in a separate pathway/block** (sLSTM) rather than making the whole state nonlinear.

---

## 3. Which nonlinearity for which capability — mapped to our OWN measured data

The cleanest experiment in the repo is **E88** (`paper/review/E88_NONSAT_RESULTS.md`): four arms identical except `state_activation` ∈ {linear, tanh, relu, softplus}, plus an LSTM, param-matched ~8M, fp32, trained T=128, evaluated to T=1024. The state nonlinearity is the *only* isolated variable. T=1024 extrapolation (the discriminator):

| arm | state map | **counting `aⁿbⁿcⁿ`** (rand 0.5) | **finite-state S5** (rand 0.008) | **finite-state S3** (rand 0.167) |
|---|---|---|---|---|
| e88-linear | affine / signed | 0.812 (worst E88) | **0.087 (best E88)** | **0.857 (best E88)** |
| e88-tanh | **saturating** | 0.836 | 0.065 | 0.524 |
| e88-relu | **non-sat rectify** | **0.893 (best E88)** | 0.050 | 0.589 |
| e88-softplus | non-sat smooth | 0.872 | 0.042 (worst) | 0.434 |
| **lstm** | **two-pathway** | **0.951** | **1.000** | **1.000** |

This *is* the capability map, measured on our stack:

- **COUNT (unbounded accumulation):** ordering is **relu > softplus > tanh ≈ linear**, gap **grows with length**. Non-saturating wins, exactly as Weiss 2018 predicts. The verified unit test (`tests/test_e88_state_activation.py`) shows why: with `decay=1` and a constant `+0.5/step` write, after 500 steps `tanh` saturates at \|S\|=0.88 while `relu` accumulates the exact count 250.0.
- **TRACK (finite-state / group word-problem, S5/S3):** ordering is the **mirror image**: **linear > tanh > relu > softplus**. Rectifying the state (relu/softplus) clamps the negative half-space and **destroys the signed/reflection structure** S5 needs. The *affine signed* map is best; the most aggressively non-saturating (softplus) is worst.
- **LATCH (bistable hold):** saturation's home turf in principle; but on a *single* matrix-state cell both `tanh` and the rectifying maps solve the capped Dyck/flag-hold probes once length is bounded, so it is a weak discriminator here (E88 §3.2). Latch is the capability the *gates* (not the state map) own in the LSTM.
- **NONLINEAR COMPOSITION (iterated map):** from the E97 within-layer study (`E97_WITHIN_LAYER_SYNTHESIS.md`), the raw-write backbone reaches 0.68 alone and 0.93 mixed; nonlinear composition is helped by *some* nonlinearity but is not the axis that separates the cells.

**The sharpest single fact:** on a single-state cell, **`tanh` is the worst of both worlds** — bounded enough to lose counting (0.836, barely above linear) yet squashed enough to underperform the affine map on S5 (0.065 vs 0.087) and badly on S3 (0.524 vs 0.857). **Our default state nonlinearity is the one shape that wins nothing.** Swapping `tanh→relu` does **not** Pareto-dominate; it *slides* the cell from the finite-state corner toward the counting corner — you trade S5 for `aⁿbⁿcⁿ`.

---

## 4. Is a SINGLE recurrent nonlinearity fundamentally limited? — Yes (E88 = LSTM lesson)

**Yes, and the limit is structural.** A single elementwise `f` must commit to **one** shape and therefore **one** regime:
- *affine/signed* → groups/tracking, **cannot count**;
- *non-saturating/rectifying* → **counts**, loses tracking;
- *saturating* → latches in principle, but is dominated at both ends by the other two on a single state.

This is the E88 finding and it is the **same** statement as the saturated-RNN hierarchy (Merrill 2020): capability classes are *separated*, and one cell sits at one point. The **LSTM escapes** the trade-off (S5 1.0 **and** count 0.95) for one reason — it has **two state pathways**:
- a **non-saturating additive cell** `c_t = f_t⊙c_{t-1} + i_t⊙g_t` — the *counter*, unbounded;
- a **saturating, gated, signed hidden** `h_t = o_t⊙tanh(c_t)` — the *controller/readout*, bounded.

The nonlinearity that matters (the gates `f,i,o` and the readout squash) is **input-driven and elementwise per step**, applied to *gates* and *outputs* — **not composed recurrently through the state**. The state-to-state map of the cell `c` is **linear** (a per-step input-dependent diagonal gate). That is the whole trick, and §5 shows it is also what makes it fast.

**Corollary that the project's E97 line already confirms:** you do not even need the *nonlinear* pathway to recover **state-tracking**. Our `gdn-neg` head — a **linear** gated-delta state with a **negative along-key eigenvalue** (the Grazzi 2025 mechanism) — drives S5 track to **1.00** and recall to **0.96** (`E97_WITHIN_LAYER_SYNTHESIS.md` Q1). Track is bought by eigenvalue sign in a linear map, not by saturation. So of the four capabilities, **three (recall, track, latch) are reachable in a linear+gated+signed cell**; the one that genuinely wants unbounded non-saturating dynamics is **counting**.

---

## 5. Two-pathway vs the fused/chunked kernels — the binding engineering constraint

This is where the recommendation is forced, because it is **measured in our kernels**, not argued.

### 5.1 A recurrent elementwise nonlinearity has NO chunked form

From `ndm/triton/e97_chunked.py` / `E97_CHUNKED_KERNEL_NOTE.md`: the `e97_delta` recurrence, **when the state is linear**, is an asymmetric gated-delta **affine** recurrence
`S_t = (decay_t·I − k_t·read_keyᵀ) S_{t-1} + k_t·write_valᵀ`,
which has a **chunked-parallel matmul form** (per-chunk UT-inverse via Newton–Schulz; only the `T/C` cross-chunk state thread is sequential). The fused Triton fwd+bwd kernel runs at **GDN-2-class throughput (97–100% util at scale dims, ≤1.02× GDN-2)** — parity-verified 13/13.

But the note states the constraint explicitly:

> "Engages only for linear-state e97_delta — per-step `tanh` is a pointwise nonlinearity on the whole state every step, **not associative, so it has no chunked-matmul form**. With `e97_state_nonlin='tanh'` the head keeps the sequential kernel."

This generalizes to **any** elementwise `f` (tanh, relu, softplus): `f(A·S + b)` does not factor across a chunk, so there is no parallel scan. Confirmed by our memory line *"bounded-state ⊥ chunkable = FUNDAMENTAL"* — and it is broader than *bounded*: **any nonlinear state map ⊥ chunkable.** A non-saturating relu state map is therefore **just as un-chunkable as tanh** — it does not even buy back the throughput it costs.

### 5.2 The "apply the nonlinearity sparsely" compromise was tested — flat throughput, no token edge, no capability uplift

The obvious escape — chunk linearly, apply `f(S)` only at **chunk boundaries** (every `C` steps) — is implemented as `gdn2_nonlin_shell` (`ndm/models/gdn2_nonlin_shell.py`, fused Triton). It was run at **1.3B** with chunk-size `C` as a free wall-clock axis (`E97_WALLCLOCK_CMA_RESULTS.md`): **it lost wall-clock.** Fused tanh throughput was **flat 0.88× for all `C`** (`C` is not a speed lever), there was **no token-matched edge** (a TIE that does not grow at longer budget), and **no capability uplift**; `gdn2-mlp` won wall-clock. So even the principled compromise that preserves *most* chunkability does not earn its place.

### 5.3 The two-pathway design IS kernel-compatible — *if you place the nonlinearity correctly*

The LSTM/mLSTM lesson maps onto a chunkable design exactly:

| LSTM piece | recurrence type | chunkable? | e97/our analogue |
|---|---|---|---|
| cell `c_t = f_t⊙c_{t-1} + i_t⊙g_t` | **linear**, input-dep diagonal gate | **YES** (GLA/Mamba/mLSTM class) | linear `e97_delta` / GDN-2 state |
| gates `f,i,o = σ(W·x)` | elementwise of **input**, not of state | **YES** (precomputed per step) | the silu/sigmoid input & output gates already in the head |
| readout `h_t = o_t⊙tanh(c_t)` | elementwise of state but **post-scan, not recurrent** | **YES** (apply after chunked `Sᵀq`) | a bounded readout on `out_t = Sᵀq` |
| state-tracking | negative/complex eigenvalues of the **linear** map | **YES** | `gdn-neg` (Grazzi 2025) |

The key realization: **the saturating nonlinearity in a two-pathway cell never sits on the state-to-state map.** It sits on (i) **gates** that are functions of the current input (so they are just per-step coefficients of a *linear* recurrence — this is precisely why GLA/Mamba/mLSTM chunk), and (ii) the **readout** `tanh(c_t)`, which is applied *after* the scan produces `c_t` (or our `Sᵀq`) and so never enters the recurrence. Both are chunk-safe. The *only* un-chunkable placement is the one we currently default to: `f` wrapped around the recurrent state itself.

**Therefore the two-pathway design is not only compatible with our fused/chunked kernels — it is the *only* way to get latch+count+track behavior while staying chunkable.** The single-state nonlinear map is the incompatible option.

---

## 6. Synthesis: nonlinearity placement, not nonlinearity choice

| placement of nonlinearity | counts? | tracks (S5)? | latches? | **chunkable?** | verdict |
|---|---|---|---|---|---|
| `tanh` on recurrent state (current default) | no | weak | yes-ish | **NO** | worst of all worlds (E88 §5; kernel §5.1) |
| `relu`/`softplus` on recurrent state | **yes** | no | partial | **NO** | counting probe only; double penalty for an LM cell |
| **linear state + neg eigenvalue** (`gdn-neg`) | partial | **yes (1.0)** | yes | **YES** | track+recall+latch, fast — current backbone |
| **linear state + input gates + bounded readout** (LSTM/mLSTM motif) | **yes** (unbounded cell) | yes (neg eig) | **yes** (gates) | **YES** | the target design |
| nonlinearity at chunk boundaries (`gdn2_nonlin_shell`) | no uplift | no uplift | — | partial | **lost wall-clock** (flat 0.88×, no token edge; measured, §5.2) |

The question "which recurrent nonlinearity?" dissolves: **the recurrent state map should be linear.** What varies is *where else* the nonlinearity goes — gates (always), readout (for a bounded counter readout), eigenvalue sign (for tracking). The single capability that strictly wants a non-saturating *recurrent* map is unbounded counting, and even there the LSTM gets it from a **linear additive cell with a separate bounded readout**, not from `relu(state)` — which is why LSTM (0.951) beats e88-relu (0.893) on counting *and* keeps S5 at 1.0.

---

## 7. Recommendation + concrete configs for the e97 cell

**Headline recommendation.** Stop treating the recurrent-state nonlinearity as the lever. **Keep the e97 state map LINEAR (chunkable), recover tracking with `gdn-neg` (negative eigenvalue), and import the LSTM two-pathway motif as (a) a bounded readout and (b) optional dedicated unbounded "counter" heads — never as a nonlinear state map.** This is the only configuration that is simultaneously capability-broad (count+latch+track+recall) and kernel-fast.

The knobs already exist in the code: `typed_head_mixture.py` exposes `e97_state_nonlin` (default `'tanh'`, line 205) and `use_chunked_e97_delta` (default True); `e88_fla_hybrid.py` `state_activation` ∈ {tanh, identity/linear, relu, softplus}; `gdn2_nonlin_shell.py` `state_nonlin` ∈ {identity, tanh, relu, softplus, softplus_c}. No new kernel is required for configs A–C below; D reuses the existing sequential path.

### Config 0 — Baseline (current chunkable winner), the control
`level='typed-gdn2-lm'`, 50/50 `e97_delta` (linear) + `gdn-neg`, `e97_state_nonlin='linear'`, `gdn_allow_neg_eigval=1`, `use_chunked_e97_delta=True`, SwiGLU MLP. **This is the reference** — it already does recall 0.96 / track 1.00 / count 1.00 (within-layer study) at GDN-2-class throughput. Everything else must beat it on counting/length-extrapolation *without* losing throughput.

### Config A — Bounded readout (LSTM `o⊙tanh(c)` motif), chunk-safe — **try first**
Same as Config 0, but apply a **saturating squash on the output, post-scan**: `out_t = silu_gate ⊙ tanh(Sᵀq_t)` (the squash is on the read-out value, not the recurrent state). Implement as a flag on the e97_delta head's output projection; it runs **after** the chunked kernel, so chunkability is untouched.
- *Hypothesis:* recovers the LSTM's "unbounded clean cell + bounded readout" separation — better length-extrapolation on counting than linear-readout, at zero throughput cost.
- *Validation:* `aⁿbⁿcⁿ` and `dyck_depth` at T∈{128,256,512,1024} vs Config 0; held-out BPB tie-or-better; kernel util unchanged (≥97% at scale dims).

### Config B — Dedicated unbounded "counter" heads (placement, not pressure)
Split heads three ways within the layer: `gdn-neg` (recall+track) + `e97_delta` linear (backbone) + a **fraction of "counter" heads** with the **forget/decay gate pinned near 1 and input gate near 1** (`decay_t≈1`), i.e. near-pure linear integrators — the explicit `aⁿbⁿcⁿ` accumulators. This is the "fixed-population floor / placement beats pressure" result from our unified-cell work: give counting its own heads rather than asking one map to do everything.
- *Validation:* counter-head fraction sweep {0, 1/8, 1/4}; measure counting T=1024 extrapolation and check it does not regress S5 (the counter heads are linear-signed, so they should not).

### Config C — Exponential input gating (xLSTM/mLSTM motif), chunk-safe
Replace the sigmoid input gate with **exponential gating** (mLSTM-style, with the max-stabilizer) on the linear state. Still a per-step input coefficient ⇒ chunkable. Sharper write control for accumulation/counting.
- *Validation:* counting + BPB vs Config 0/A; numerical stability of the exp-gate stabilizer at 1.3B.

### Config D — Per-step `relu`/`softplus` state map — **EXPRESSIVITY PROBE ONLY, not an LM cell**
`e97_state_nonlin='relu'` (or `'softplus'`), which **forces the sequential kernel** (the loud no-eager guard is intact). Run **only** on the counting/expressivity battery to measure the *ceiling* a non-saturating recurrent map reaches, as the upper reference for Configs A–C.
- *Explicit non-goal:* this is **not** a candidate production cell. §5.1 (no chunked form) + §5.2 (`gdn2_nonlin_shell` lost wall-clock) + E88 (relu *hurts* S5) make a recurrent relu/softplus state map a measured dead end for an LM backbone. Use it to bound what A–C should aim for on counting, then discard.

### What we are explicitly NOT recommending
- **Do not** change the default to `relu`/`softplus` on the recurrent state. It is un-chunkable (§5.1, same as tanh) *and* it demotes S5/S3 (E88 §4). Double penalty.
- **Do not** revisit `gdn2_nonlin_shell` (nonlinearity at chunk boundaries). Already lost wall-clock at 1.3B (§5.2).
- **Do not** keep `tanh` on the recurrent state as the default. It is the worst-of-both-worlds shape (E88 §5) and breaks chunkability. If a head must stay sequential-tanh for a targeted latch probe, scope it to that probe, not the LM backbone.

### Priority
**A (bounded readout)** is the cheapest, most LSTM-faithful, fully chunk-safe change → try first. **B (counter heads)** is the placement fix for the one capability (counting) a linear state under-serves. **C (exp gating)** is a refinement. **D** is a measurement, not a product. All four leave the proven `gdn-neg` track/recall mechanism intact.

---

## 8. Honest caveats

- **E88 absolute S5 numbers are low (under-trained at the shared 8M/short budget);** but the *ordering* is seed-stable and reproduces prior CMA-tuned findings, and the LSTM hitting 1.0 at the same budget proves the task is learnable — so the relative capability map is sound (E88 §4 caveat).
- **Counting in a linear gated cell is not free** — e88-linear counts *worst* (0.812). The claim is not "linear counts well," it is "the *LSTM's* linear cell + bounded readout + separate controller counts well (0.951) and chunks." Config A/B test whether the e97 head reproduces that with the readout/counter-head separation; if A/B do **not** lift counting toward LSTM levels, that is a real negative result and the counting capability may require the sequential pathway (Config D ceiling) — to be settled empirically, not assumed.
- **The within-layer LM verdict independently loses at scale on throughput grounds for the *raw-write* heads** (`E97_WITHIN_LAYER_SYNTHESIS.md`); this document concerns the **state-nonlinearity** axis specifically and recommends the chunkable linear `e97_delta`/GDN path, which is exactly the path that survived that audit.
- **`hochreiter1991`** (the 1991 diploma thesis) was *not* separately web-verified; the latching-gradient claim rests on the verified **Bengio–Simard–Frasconi 1994** paper, which is the standard citation for it.

---

## 9. One-paragraph answer to the brief

We default to `tanh` on the recurrent state; the literature and our own E88 data agree this is the **one shape that wins nothing** — bounded enough to lose counting (Weiss 2018), squashed enough to lose the signed structure S5 needs. The trade-off is real and a **single** elementwise state nonlinearity is **fundamentally limited** to one regime (Merrill 2020; E88). The resolution is the LSTM's, re-derived by xLSTM/GLA/Mamba/gated-delta for the kernel era: **make the recurrent state map linear (so it chunks), put the nonlinearity in input-driven gates and a bounded readout, and buy state-tracking with negative eigenvalues (Grazzi 2025 ⇒ our `gdn-neg`).** A relu/softplus *recurrent* map counts but is **un-chunkable (just like tanh) and S5-hostile** — a double penalty we have already measured the chunk-boundary compromise (`gdn2_nonlin_shell`) to fail. **Recommended e97 cell: linear `e97_delta` + `gdn-neg`, plus the LSTM bounded-readout (Config A) and optional dedicated linear counter-heads (Config B); keep the per-step relu/softplus state map only as a sequential counting probe (Config D).**

---

*Deliverable for `recurrent-nonlin-research`. Every empirical number is from a committed project doc (E88_NONSAT_RESULTS, E97_CHUNKED_KERNEL_NOTE, E97_WALLCLOCK_CMA_RESULTS, E97_WITHIN_LAYER_SYNTHESIS) or a verified unit test; every citation's title/authors/year/arXiv|DOI was web-verified (5-cluster parallel check, 18/18 confirmed). `paper/main.typ` untouched.*
