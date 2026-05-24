# Related Work: Large-Scale Nonlinear Recurrent Language Models

## CHANGELOG

- **2026-05-24 (`reframe-paper-narrative`):** Dropped the "first pure
  nonlinear recurrent LM at ≥500M (or ≥1B) on a Pile-class corpus"
  priority framing throughout. M2RNN (`arXiv:2603.14360`, March 2026)
  also trained a pure-recurrent variant at 410M, so the priority claim
  is contested at best and not the strongest version of the story. The
  document is now organized around the **three-pillar reframe** used in
  `paper/OUTLINE.md`: (1) generality of multi-programming for pure
  nonlinear recurrent LMs at scale, (2) the update rule matters — the
  delta-correcting write `v − Sᵀk` separates from raw-write matrix RNNs
  and approaches the NC1 limit via S5 tracking, (3) FLOPs-per-bit
  convergence under CMA-ES (N = 4). Entries 16 (M2RNN pure 410M) and 17
  (M2RNN 7B MoE hybrid) are repositioned as **related-work peers** that
  demonstrate a similar pure-recurrent capability at smaller scale with
  a different (raw-write) update rule, not as priority threats. The
  former "Closest Prior Art" section is renamed "Related-Work Peers,"
  and the "Summary Verdict on NDM Novelty Claim" is replaced with
  "Summary: Where the Three Pillars Land in the Literature." Per-entry
  verdicts that contained "first at scale" or similar priority language
  are rewritten to compare on the update-rule, recipe-generality, or
  FLOPs-per-bit axes instead. NC1 wording is held at "approaches the
  NC1 limit via S5 tracking" pending `lean-formalization-gap-audit`.

---

**Purpose:** Catalogue the prior and concurrent literature on large-scale
recurrent language models, classify each entry by recurrence type
(linear-state vs nonlinear-state vs hybrid), and identify which entries
are peers under the three-pillar framing — i.e., which entries
demonstrate pure or near-pure nonlinear recurrence at scale, which
entries are the comparison cohort for the FLOPs-per-bit convergence
finding under CMA-ES, and which entries fall outside the relevant
class. The paper itself (see `paper/OUTLINE.md`) makes no priority
claim; this document collects the evidence base for the comparison.

**NDM update equation (reference):**
```
r_t     = S_{t-1}^T k_t
delta_t = v_t - r_t
S_t     = tanh(d_t S_{t-1} + k_t delta_t^T)
```
`S_{t-1}` appears inside `tanh` twice (via scaling and via the delta correction), making the temporal state update **nonlinear in state**. The 1.27B-parameter NDM model is fully recurrent (no attention, no linear-recurrent layers).

**Linearity criterion used throughout:**
A model is *linear-state* if the recurrence can be written as `h_t = A_t h_{t-1} + b_t` where `A_t` and `b_t` depend only on the current input `x_t`, not on `h_{t-1}`. A model is *nonlinear-state* if `h_{t-1}` (or a nonlinear function of it) appears inside a nonlinearity that governs the state update itself.

---

## Comparison Table

| # | Model | Params (largest) | Training tokens / dataset | Recurrence type | Citation |
|---|-------|-------------------|--------------------------|-----------------|----------|
| 1 | Mamba (S6) | ~3B | ~300B / The Pile | Linear-state (diagonal SSM) | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 2 | Mamba-2 (SSD) | ~2.7B | ~300B / Pile subset | Linear-state (scalar-times-identity SSM) | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| 3 | RetNet | ~3B | ~100B / The Pile | Linear-state (decayed linear attention) | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |
| 4 | GLA (Gated Linear Attention) | ~3B | ~100B / Pile-class | Linear-state (data-dependent linear recurrence) | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) |
| 5 | DeltaNet | 1.3B | 100B / Pile-class | Linear-state (delta rule, linear in S) | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 6 | Gated DeltaNet | 1.3B | 100B / Pile-class | Linear-state (gated delta rule) | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 7 | RWKV-4 | 14B | ~330B / The Pile | Linear-state (WKV linear recurrence) | [arXiv:2305.13048](https://arxiv.org/abs/2305.13048) |
| 8 | RWKV-5/6 (Eagle/Finch) | 7.5B | multilingual | Linear-state (matrix-state WKV, linear) | [arXiv:2404.05892](https://arxiv.org/abs/2404.05892) |
| 9 | RWKV-7 (Goose) | 2.9B | 3.1T multilingual | Linear-state (generalized delta rule) | [arXiv:2503.14456](https://arxiv.org/abs/2503.14456) |
| 10 | HGRN / HGRN2 | ~3B | ~100B / Pile-class | Linear-state (gated linear RNN) | [arXiv:2404.07904](https://arxiv.org/abs/2404.07904) |
| 11 | S5 | small (~50M) | LRA benchmarks | Linear-state (parallel scan SSM) | [arXiv:2208.04933](https://arxiv.org/abs/2208.04933) |
| 12 | MinGRU / MinLSTM | small / not scaled to >500M | competitive small-scale | Linear-state (gates input-only, parallel scan) | [arXiv:2410.01201](https://arxiv.org/abs/2410.01201) |
| 13 | **sLSTM** (in xLSTM) | 1.3B (mixed with mLSTM) | 300B / SlimPajama | **Nonlinear-state** (gates depend on h_{t-1} via memory mixing) | [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) |
| 14 | mLSTM / xLSTM-7B | 7B (pure mLSTM) | 2.3T / DCLM | Linear-state (covariance update, parallelizable) | [arXiv:2503.13427](https://arxiv.org/abs/2503.13427) |
| 15 | Griffin / Hawk (RG-LRU) | 9B (RecurrentGemma) | 2T / web data | Linear-state (diagonal gated linear recurrence) | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) |
| 16 | **M2RNN** (pure) | **410M** (pure recurrent) | 100B / Nemotron-CC-v2 | **Nonlinear-state** (matrix state, tanh-gated) | [arXiv:2603.14360](https://arxiv.org/abs/2603.14360) |
| 17 | M2RNN (hybrid 7B MoE) | 7B MoE (1.1B active) | 100B / Nemotron-CC-v2 | Hybrid (nonlinear recurrent + attention) | [arXiv:2603.14360](https://arxiv.org/abs/2603.14360) |
| 18 | Titans (MAC/MAG) | unspecified (experiments ~125M–1.3B range) | not Pile-class at scale | Hybrid (nonlinear MLP memory + attention) | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| 19 | Liquid AI LFM-7B | 7B | not publicly disclosed | Hybrid (recurrence + conv + attention) | [liquid.ai](https://www.liquid.ai/research/liquid-neural-networks-research) |
| 20 | Classical LSTM/GRU at scale | none at ≥500M on Pile | n/a | Nonlinear-state (h_{t-1} inside gates) | — |
| 21 | LSTM ZOO scaling (proof-of-concept) | 1B | small-scale proof-of-concept | Nonlinear-state (LSTM, ZOO training) | [arXiv:2505.17852](https://arxiv.org/abs/2505.17852) |

---

## Per-Entry Verdict

### 1. Mamba (S6) — `arXiv:2312.00752`
**Recurrence type:** Linear-state. The selective SSM is `h_t = A_t h_{t-1} + B_t x_t` with diagonal `A_t`, `B_t` depending on `x_t` only (input-selective, not state-selective). Despite the "selection" mechanism, `h_{t-1}` never enters a nonlinearity in the state update itself.  
**Scale:** ~3B; trained to convergence on Pile-class data.  
**Verdict: Does NOT contradict NDM's claim.** Mamba is linear-state; NDM's claim is specifically about nonlinear-state recurrence.

---

### 2. Mamba-2 (SSD) — `arXiv:2405.21060`
**Recurrence type:** Linear-state. SSD (State Space Duality) restricts `A_t` to scalar-times-identity, maintaining a strictly linear state update. Computationally equivalent to a structured form of linear attention.  
**Scale:** ~2.7B; Pile-class training.  
**Verdict: Does NOT contradict NDM's claim.** Same linearity argument as Mamba-1.

---

### 3. RetNet — `arXiv:2307.08621`
**Recurrence type:** Linear-state. Recurrence is `S_t = \gamma S_{t-1} + k_t v_t^T` — a decayed outer-product accumulation, purely linear in `S_{t-1}`.  
**Scale:** ~3B; 100B tokens.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 4. GLA (Gated Linear Attention) — `arXiv:2312.06635`
**Recurrence type:** Linear-state. `S_t = G_t \odot S_{t-1} + k_t v_t^T` where `G_t` depends only on `x_t`. The gating is input-dependent but the recurrence is linear in `S_{t-1}`.  
**Scale:** ~3B; Pile-class.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 5. DeltaNet — `arXiv:2406.06484`
**Recurrence type:** Linear-state. The delta rule `S_t = S_{t-1} + k_t(v_t - S_{t-1}^T k_t)^T` expands to `S_t = (I - k_t k_t^T) S_{t-1} + k_t v_t^T`, which is an affine (linear) function of `S_{t-1}`. The correction term `v_t - S_{t-1}^T k_t` is linear in `S_{t-1}`. No nonlinearity wraps the state.  
**Scale:** 1.3B; 100B tokens on Pile-class data. This is the closest linear-state model architecturally to NDM (shared delta-correction lineage) but remains linear-state.  
**Verdict: Does NOT contradict NDM's claim.**  
*Note:* NDM's update also has a delta correction but wraps everything in `tanh`, making NDM nonlinear-state and DeltaNet linear-state.

---

### 6. Gated DeltaNet — `arXiv:2412.06464`
**Recurrence type:** Linear-state. Adds a gating term `G_t` (dependent on `x_t` only) to the delta rule recurrence; still linear in `S_{t-1}`.  
**Scale:** 1.3B; 100B tokens.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 7. RWKV-4 — `arXiv:2305.13048`
**Recurrence type:** Linear-state. The WKV mechanism accumulates key-value pairs with exponential decay; the recurrence is `wkv_t = e^{-(w+u)} wkv_{t-1} + e^{k_t} v_t`, linear in the state. No nonlinearity acts on `wkv_{t-1}` in the state update.  
**Scale:** 14B; ~330B tokens of The Pile (RWKV-4 World). This is the largest scale in the RWKV family.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 8. RWKV-5/6 (Eagle/Finch) — `arXiv:2404.05892`
**Recurrence type:** Linear-state. Eagle/Finch introduce multi-headed matrix-valued states and data-dependent decay, but the recurrence remains a gated linear scan (no nonlinearity applied to the matrix state itself). The paper introduces LoRA-augmented decay vectors but the temporal state update is still `S_t = A_t \odot S_{t-1} + k_t v_t^T`.  
**Scale:** 7.5B (Eagle); ~1.1T multilingual tokens.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 9. RWKV-7 (Goose) — `arXiv:2503.14456`
**Recurrence type:** Linear-state. RWKV-7 introduces a generalized delta rule with vector-valued gating and in-context learning rates (a diagonal-plus-low-rank transition matrix), which the authors show subsumes both the standard delta rule and a forget-gate approximation. The state update is still linear in the previous state — `h_t` does not appear inside a nonlinearity that governs `h_t`'s own computation. Importantly, the RWKV-7 authors note that related work (Titans, Miras) has introduced *nonlinear* deep memory, contrasting it with RWKV-7's approach.  
**Scale:** 2.9B; 3.1T multilingual tokens.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 10. HGRN / HGRN2 — `arXiv:2404.07904`
**Recurrence type:** Linear-state. HGRN2's recurrence is `h_t = \text{Diag}(f_t) h_{t-1} + i_t \otimes (1 - f_t)` where `f_t` and `i_t` depend only on `x_t`. State expansion yields a matrix state but the update is a linear function of the previous matrix state. The paper explicitly labels itself "Gated Linear RNNs."  
**Scale:** ~3B; ~100B tokens on Pile-class.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 11. S5 — `arXiv:2208.04933`
**Recurrence type:** Linear-state. A simplified SSM with multi-input/multi-output structure; `h_t = A h_{t-1} + B x_t` with fixed or learned diagonal `A`. Used primarily on Long Range Arena; no large-scale LM training.  
**Scale:** Small (~50M); not trained on Pile-class data at scale.  
**Verdict: Does NOT contradict NDM's claim** (wrong scale and wrong recurrence type).

---

### 12. MinGRU / MinLSTM — `arXiv:2410.01201`
**Recurrence type:** Linear-state. The key simplification in this paper is removing hidden-state dependence from gates, yielding `h_t = (1-z_t) h_{t-1} + z_t \tilde{h}_t` where `z_t` depends only on `x_t`. This collapses to a linear scan with input-dependent coefficients, making it fully parallelizable. The "Min" prefix denotes that this is precisely the minimal (linear) variant of the original (nonlinear) GRU/LSTM.  
**Scale:** Competitive only at small scale; no Pile-class training at ≥500M published.  
**Verdict: Does NOT contradict NDM's claim.**

---

### 13. sLSTM (in xLSTM) — `arXiv:2405.04517`
**Recurrence type:** **Nonlinear-state.** sLSTM uses exponential gating for input and forget gates, with *memory mixing* — the gates depend on the previous hidden state `h_{t-1}`, which in turn is derived from the cell state `c_{t-1}` via a nonlinearity (output gate applied to `tanh(c_{t-1})` or similar). This creates a nonlinear feedback loop: `c_t = f_t \odot c_{t-1} + i_t \odot z_t` where `f_t = \exp(W_f [h_{t-1}; x_t])` and `h_{t-1}` is nonlinearly coupled to `c_{t-1}`. The xLSTM paper explicitly states that memory mixing (hidden-to-gate dependency) is what enables sLSTM to solve state-tracking tasks that purely linear models cannot.
**Scale:** The original xLSTM paper trained a **1.3B-parameter mixed model** (7:1 mLSTM:sLSTM block ratio) on **300B tokens of SlimPajama** (a Pile-class corpus). The model is *not* pure sLSTM — most blocks use mLSTM (linear-state), with a minority of sLSTM (nonlinear-state) blocks.
**Position under the three-pillar reframe:** Related-work peer for **Pillar 1**. xLSTM-1.3B is the closest scale band among prior nonlinear-recurrent results, but it is a *mixed* nonlinear+linear stack (87.5% linear mLSTM blocks). It is informative as evidence that nonlinear recurrent blocks are usable at the 1.3B band when interleaved with linear blocks; it is *not* an example of pure nonlinear recurrence at scale. For the update-rule comparison (Pillar 2), sLSTM operates on a scalar cell with exponential gating rather than a matrix state with delta correction, so it is not a direct comparable on that axis.

---

### 14. mLSTM / xLSTM-7B — `arXiv:2503.13427`
**Recurrence type:** Linear-state. The mLSTM covariance update is `C_t = f_t \odot C_{t-1} + i_t \odot (v_t k_t^T)` where the gates `f_t, i_t` and input projections `v_t, k_t` depend only on `x_t`. The paper emphasizes this makes mLSTM "fully parallelizable" — a hallmark of linear-state recurrence. The xLSTM-7B model uses *only* mLSTM blocks.  
**Scale:** 7B; 2.3T tokens of DCLM.  
**Verdict: Does NOT contradict NDM's claim.** Despite being the largest xLSTM variant and branded as LSTM-based, mLSTM is linear-state.

---

### 15. Griffin / Hawk (RG-LRU) and RecurrentGemma — `arXiv:2402.19427`, `arXiv:2404.07839`
**Recurrence type:** Linear-state. The Real-Gated Linear Recurrent Unit (RG-LRU) has a recurrence where gates use the current input only (not the recurrent state): `h_t = \bar{a}_t h_{t-1} + \bar{b}_t x_t`, with `\bar{a}_t = \exp(-\text{softplus}(r_t \cdot \log \lambda) \exp(-x_t \cdot r_t))`. The gate `r_t` depends on `x_t` but not on `h_{t-1}`. The paper explicitly names this a "linear recurrence." Griffin additionally hybridizes RG-LRU with local attention.  
**Scale:** RecurrentGemma-9B (2B and 9B variants) trained on 2T tokens of web/code data.  
**Verdict: Does NOT contradict NDM's claim.** RG-LRU is linear-state; Griffin is a hybrid.

---

### 16. M2RNN (pure recurrent, 410M) — `arXiv:2603.14360`
**Recurrence type:** **Nonlinear-state.** M2RNN's update is:
```
Z_t = tanh(H_{t-1} W + k_t v_t^T)
H_t = f_t H_{t-1} + (1 - f_t) Z_t
```
Since `H_{t-1}` appears inside `tanh` (via `H_{t-1}W`), `H_t` is a nonlinear function of `H_{t-1}`. This is a matrix-state nonlinear RNN.
**Scale:** **410M parameters** for the largest pure-recurrent M2RNN evaluation; 7B is MoE hybrid (entry 17 below). Training data: 100B tokens of Nemotron-CC-v2 (a high-quality web corpus, not The Pile but in the same class).
**Publication:** March 2026, concurrent with NDM development.
**Position under the three-pillar reframe:** **Related-work peer.** M2RNN
is the closest peer for both Pillar 1 (it demonstrates that a pure
nonlinear matrix-state recurrent LM can be trained to useful loss on a
large web corpus — at 410M with a different systems setup) and Pillar 2
(it is the head of the *raw-write* nonlinear matrix RNN family, against
which the delta-correcting write `v − Sᵀk` is compared). The key
contrasts:
- **Update rule.** M2RNN's nonlinear write is a linear map of the state,
  `H_{t-1} W`, followed by `tanh` — a *raw-write* matrix RNN. NDM's
  nonlinear write is a delta correction, `v − S^T k`, followed by `tanh`
  — bounded prediction-error feedback. The empirical state-tracking
  separation between these two update families at matched parameter
  count is the central Pillar 2 result; the trusted Lean core formalizes
  a one-step update-family resource separation between them. *(TODO:
  the surrounding NC1 wording is held pending
  `lean-formalization-gap-audit`.)*
- **Scale.** M2RNN's pure-recurrent variant is 410M; the NDM
  pure-recurrent stack documented in the paper is 1.27B. The
  multi-programmed systems recipe under Pillar 1 is the enabling
  difference; the recipe is general and would also apply to the M2RNN
  update family (the M2RNN-CMA variant is the concrete instantiation
  used in the experiments).
- **Training corpus.** NDM trains on The Pile; M2RNN on Nemotron-CC-v2.
  Both qualify as Pile-class web corpora; neither result is corpus-bound.

This entry is **not** treated as a priority threat. M2RNN and the work
documented in the paper are concurrent investigations of overlapping
questions, with different update rules and different scale bands, and
the paper credits M2RNN explicitly as such.

---

### 17. M2RNN (7B MoE hybrid) — `arXiv:2603.14360`
**Recurrence type:** Hybrid (nonlinear recurrent layers interleaved with attention).
**Scale:** 7B total parameters (1.1B active); hybrid outperforms pure-recurrent Gated DeltaNet by 0.4–0.5 perplexity points in the M2RNN paper's reported comparisons.
**Position under the three-pillar reframe:** Related-work peer, but
weakly so — the hybrid 7B M2RNN sits outside the pure-recurrent class
addressed by Pillar 1 and is therefore not a direct comparable for
either pillar 1 or pillar 2. It is included for completeness of the
M2RNN family.

---

### 18. Titans (MAC/MAG) — `arXiv:2501.00663`
**Recurrence type:** Hybrid (nonlinear MLP-based memory module + attention). The Neural Long-Term Memory Module (LMM) uses an MLP as a recurrent state and updates it via gradient-based "surprise" (online meta-learning). This is deeply nonlinear in the memory state. However, Titans is explicitly designed as a hybrid: short-term attention + long-term MLP memory.  
**Scale:** Largest experiments reported in the ~125M–1.3B range; no large-scale pure-recurrent Pile-class training.  
**Verdict: Does NOT contradict NDM's claim.** Titans is hybrid, not pure recurrent, and the MLP-memory update mechanism is a qualitatively different design from NDM's matrix-state delta correction.

---

### 19. Liquid AI LFM-7B
**Recurrence type:** Hybrid (recurrence + convolution + attention). The Liquid Foundation Model architecture combines multiple sequence mixing mechanisms; it is not a pure recurrent model.  
**Scale:** 7B; training data not publicly disclosed.  
**Verdict: Does NOT contradict NDM's claim.** LFM is hybrid and lacks a published paper with reproducible equations.

---

### 20. Classical LSTM/GRU trained at scale (historical)
**Recurrence type:** **Nonlinear-state.** Classical LSTM cell state: `c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W x_t + U h_{t-1} + b)` where `h_{t-1} = o_{t-1} \odot \tanh(c_{t-1})`. The gate `f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)` depends on `h_{t-1}`, which depends nonlinearly on `c_{t-1}`. So the full recurrence is nonlinear in `c_{t-1}`.
**Scale:** **No published model at ≥500M parameters trained to near-convergence on a Pile-class corpus.** Large-scale language model training shifted to Transformers before anyone trained billion-parameter classical LSTMs on modern corpora. Mikolov's RNN-LM work (2010–2013) operated at word-level with ≪1M parameters. Google's GNMT (2016) used stacked LSTM encoders/decoders but for translation, not LM, and at ~380M parameters. No known Pile-class convergence result for classical LSTM/GRU at ≥500M.
**Position under the three-pillar reframe:** Background, not peer.
Classical LSTM/GRU is the historical reason the field assumed pure
nonlinear recurrence was impractical at scale — and the unsuccessful
training of these architectures at large scale is what motivated the
move to linear-state and hybrid alternatives surveyed in entries 1–15.
Neither the systems recipe (Pillar 1) nor the update-rule comparison
(Pillar 2) makes a direct comparison against these architectures.

---

### 21. Scaling RNNs with ZOO — `arXiv:2505.17852`
**Recurrence type:** Nonlinear-state (classical RNN/LSTM architecture implied, trained via zero-order optimization rather than BPTT).  
**Scale:** Demonstrates scaling to 1B parameters as a proof-of-concept, but training used Zero-Order Optimization (ZOO), not standard gradient descent. The training corpus and convergence metrics are not comparable to full Pile-class training.  
**Publication:** May 2025.  
**Verdict: Does NOT contradict NDM's claim.** The ZOO approach is a different training regime; near-convergence on a Pile-class corpus using standard gradient training has not been demonstrated. This result is also concurrent with or later than NDM's main experimental work.

---

## Related-Work Peers

These entries deserve dedicated treatment in the paper's related-work
section. They are positioned as **peers** that share part of the
problem space — pure or near-pure nonlinear recurrent language models —
not as priority threats.

### 1. M2RNN (arXiv:2603.14360) — Peer on pure-nonlinear pure-recurrent training
M2RNN is the closest peer in the literature. It:
- Is nonlinear-state (matrix state with tanh inside the update).
- Was trained pure-recurrently at 410M.
- Has a similar design motivation (state tracking, nonlinear matrix memory).
- Is concurrent (March 2026).

Relationship to the three-pillar reframe:
1. **Pillar 1 (multi-programming generalizes):** M2RNN demonstrates a
   pure-recurrent variant at 410M with a different systems setup; the
   present paper extends the same general capability to 1.27B with a
   multi-programmed recipe that is shown (via the M2RNN-CMA variant in
   the experiments) to apply across the nonlinear matrix-state family.
2. **Pillar 2 (update rule matters):** M2RNN is the head of the
   *raw-write* nonlinear matrix RNN family. The paper's headline
   expressivity comparison is between this raw-write update and the
   delta-correcting update `v − S^T k`. The one-step update-family
   resource separation is formalized in the trusted Lean core.
3. **FLOPs-per-bit (Pillar 3):** Under matched CMA-ES search, M2RNN and
   NDM converge to nearly the same FLOPs-per-bit learning rate (N = 4
   together with FLA-GDN and Mamba2). The training-speed axis is not
   what separates the two.

**Action required:** The paper cites M2RNN and explicitly distinguishes
the delta-correcting mechanism from M2RNN's raw-write approach. No
priority is claimed against M2RNN.

### 2. xLSTM-1.3B with sLSTM blocks (arXiv:2405.04517) — Peer at scale, mixed-recurrence
xLSTM at 1.3B uses nonlinear-state sLSTM blocks (with memory mixing,
gates depend on `h_{t-1}`), trained on 300B tokens of SlimPajama. It is a
nonlinear-state model at the same scale band as the paper's headline
NDM stack.

Relationship to the three-pillar reframe:
1. **Pillar 1:** xLSTM-1.3B is the closest scale-band peer for the
   "pure nonlinear recurrent LM at scale" capability, but it is a
   *mixture* of nonlinear (sLSTM) and linear (mLSTM) blocks (7:1 ratio,
   87.5% linear). It supports the conclusion that nonlinear recurrent
   blocks are trainable at this scale; it is not a demonstration of
   pure nonlinear recurrence.
2. **Pillar 2:** sLSTM uses scalar cell state with exponential gating;
   NDM uses matrix state with delta correction. These are qualitatively
   different update structures, so xLSTM is not a direct comparable on
   the update-rule axis.

**Action required:** The paper acknowledges xLSTM-1.3B as a peer
nonlinear-recurrent result at the same scale band, while noting that
its mixed architecture and scalar cell state place it outside the
narrower comparison set used for Pillars 1 and 2.

### 3. Titans (arXiv:2501.00663) — Nonlinear memory, different regime
Titans uses a deeply nonlinear memory module (MLP with online gradient
updates) and is qualitatively nonlinear-state. However, it is designed
as a hybrid and has not been evaluated as a pure-recurrent model at
Pile-class scale. The update mechanism (test-time MLP gradient update)
is substantially different from a matrix delta correction.

---

## Open Questions

1. **M2RNN training scale (pure):** The paper reports 410M for homogeneous M2RNN. Was a larger pure-recurrent M2RNN trained internally and not reported? This should be confirmed via author correspondence or future arxiv versions.

2. **xLSTM sLSTM-only models:** The xLSTM paper evaluates architectures in mixed configurations. Was a *pure sLSTM* model (all blocks nonlinear) ever trained at ≥500M parameters? If so, it would be a stronger peer for Pillar 1 than the published xLSTM-1.3B mixed stack. The xLSTM-7B uses only mLSTM (linear), suggesting the team moved away from sLSTM at large scale.

3. **Mikolov-scale LSTM LM revival:** EleutherAI and others have experimented with recurrent architectures on The Pile. It is worth checking whether any unreported or unpublished classical LSTM experiments at ≥500M parameters on The Pile exist in gray literature (GitHub, blog posts, tech reports).

4. **Liquid AI architecture details:** LFM uses recurrence as a component but the full architecture is not public. If the recurrent component is nonlinear-state and can be ablated as a pure-recurrent model at ≥500M, this could be relevant.

---

## Summary: Where the Three Pillars Land in the Literature

The paper makes no priority claim. Each of the three pillars is
characterized below by which prior or concurrent entries it engages
with and what the literature evidence supports.

### Pillar 1 — Multi-programming generalizes (pure nonlinear recurrent LM at scale)

| Question | Evidence from this survey |
|---|---|
| Has any prior or concurrent work trained a *pure-recurrent* nonlinear matrix-state LM on a Pile-class corpus? | Yes — M2RNN (entry 16) at 410M on Nemotron-CC-v2, concurrent (March 2026). |
| Has any prior work done so at ≥1B parameters? | Not on the public record before the present work, but M2RNN is a peer and may scale further; no priority is claimed. |
| Has any near-pure result been trained at ≥1B? | Yes — xLSTM-1.3B (entry 13), but only 12.5% of blocks are nonlinear sLSTM; the rest are linear mLSTM. |
| Is the multi-programming recipe specific to any one update rule? | No — the recipe (per-head bounded memory programs, fused Triton recurrence kernel, sparse checkpointing) is structural and applies to the M2RNN family as well. The CMA-reshaped M2RNN variant in the paper's experiments is the concrete instantiation. |

### Pillar 2 — The update rule matters (delta correction vs raw-write, approaches the NC1 limit via S5 tracking)

| Question | Evidence from this survey |
|---|---|
| Which prior architectures are in the *raw-write* nonlinear matrix RNN family? | Primarily M2RNN (entry 16) — `Z = tanh(H·W + k v^T)`. |
| Which prior architectures are in the *delta-correcting* family but not in the nonlinear-state class? | DeltaNet (entry 5) and Gated DeltaNet (entry 6) — both use delta correction but the recurrence is linear in state. RWKV-7 (entry 9) generalizes the delta rule but remains linear in state. |
| Has any prior pure-recurrent nonlinear matrix RNN been compared against linear-state baselines on a state-tracking probe at matched parameter count? | Not in the surveyed literature — the S3/S5 separation reported in `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a is the comparison evidence for this pillar. |

*(TODO: the "approaches the NC1 limit via S5 tracking" wording is
held pending `lean-formalization-gap-audit`; the trusted Lean surface
as of 2026-05-23 covers S5 tracker construction, S5
non-solvability, and a fixed-precision finite-state ceiling, not a
formal NC1 lower bound for linear-scan recognizers.)*

### Pillar 3 — FLOPs-per-bit convergence under CMA-ES (N = 4)

| Question | Evidence from this survey |
|---|---|
| Which architectures are in the N = 4 CMA-ES comparison cohort? | NDM (this paper), M2RNN-class (entry 16), FLA-GDN (entry 6 lineage), Mamba2 (entry 2). |
| Where is the supporting evidence documented? | `docs/CMA_FLOP_RATE_FINDING.md` — *forthcoming*, produced by `harvest-cma-flop-rate-finding`. |
| Does the paper claim a FLOPs-per-bit advantage for any update rule? | No. The finding is the *opposite*: under matched CMA-ES search, the four families converge to nearly the same FLOPs-per-bit learning rate. |

### Suggested framing line for the paper's related-work section

> Concurrent work M2RNN [arXiv:2603.14360] demonstrates pure-recurrent
> nonlinear matrix-state language modeling at 410M with a raw-write
> update `tanh(H·W + k v^T)`. The present work extends this line in
> three respects: (i) a 1.27B pure-recurrent model trained under a
> multi-programmed systems recipe; (ii) a delta-correcting update
> `v − S^T k` that separates on state-tracking expressivity from the
> raw-write family at matched parameter count and approaches the NC1
> limit via S5 tracking; and (iii) a CMA-ES (N = 4) FLOPs-per-bit
> comparison showing that, once shape and hyperparameters are
> matched, the recurrent families converge to nearly the same
> learning rate — so the contribution is the architectural option and
> the mechanism, not raw training speed.
