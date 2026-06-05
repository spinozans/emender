# NOVELTY + POSITIONING — how new is the horizontal-hybrid unified-knob mixture-of-specialists?

**Task:** `novelty-positioning` · model claude:opus · literature positioning, no GPU.
**`paper/main.typ` was NOT edited.** Committed, not pushed.

**Citation discipline.** Every reference below is a real, web-verifiable work (arXiv id /
DOI / venue). The 2024–2026 hybrid/MoE/parallelism references were located and their
arXiv abstract pages confirmed in this session (titles + ids checked); the
state-tracking / precision references reuse the bibliography independently verified in
[`PRECISION_NONLINEARITY_RESEARCH.md`](PRECISION_NONLINEARITY_RESEARCH.md) §9 (each 2024–26
arXiv id fetched at `arxiv.org/abs/<id>` there). **No citation, author, year, or result
below is fabricated.** Where I could not confirm an author list I cite title + id only and
say so. Two items I deliberately do **not** cite as confirmed are flagged in §9.

---

## 0. What is being positioned (the artifact, in one sentence)

A **single width-multiprogrammed (serial-in-time, parallel-over-width) recurrent layer**
whose per-head matrix state updates by one parameterized cell

```
S_t = φ_γ( λ·S_{t-1}  −  β·k(kᵀS_{t-1})  +  i·k vᵀ )
```

where each head occupies a **learnable point in a (gain λ, correction β, state-nonlinearity
φ_γ) space**. The four classical recurrent capabilities —
**non-solvable-group tracking (S₅), unbounded counting (aⁿbⁿcⁿ), bistable latching, and
state-nonlinear computation** — are **corners** of that space, and the heads
**self-organize (init-spread + knob-specific LR) into a stable HORIZONTAL
mixture-of-specialists inside ONE layer** that covers all four
([`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md),
[`UNIFIED_LEARNABILITY_RESULTS.md`](UNIFIED_LEARNABILITY_RESULTS.md)). E88 is the single
point (λ=decay∈(0,1), β=1, φ=tanh); the "(0,1) decay clamp cribbed from SSMs" is the lock
that forecloses counting and latching.

The five components, each labelled **NOVEL / NOVEL-SYNTHESIS / KNOWN** with the precise
delta vs the closest real prior art, follow.

---

## 1. Component 1 — HORIZONTAL hybridization: heterogeneous recurrent *update rules* across heads in ONE layer

**Claim under test.** Mixing *heterogeneous capability/head types* within a single
layer/timestep, rather than across depth.

### Closest prior art and the exact delta

| prior work | what it does | how it differs from ours |
|---|---|---|
| **Hymba** — Dong, Fu, Diao, …, Kautz, Molchanov, *A Hybrid-head Architecture for Small Language Models*, arXiv:**2411.13676** (ICLR 2025) | **The nearest neighbor.** Puts **softmax-attention heads and Mamba-SSM heads in the SAME layer, in parallel**, fusing their outputs. Explicitly "integrates attention heads and SSM heads within the same layer" instead of stacking layer types. | Hymba establishes the *horizontal-mixing principle* but with exactly **two fixed, architecturally-distinct module types** (attention + one SSM), one of which is **not a recurrence**. Ours mixes a **continuum of heterogeneous recurrent update rules** (track/count/latch/nonlin) **drawn from one parameterized cell**, not two bolted-together modules. |
| **Mixture-of-Memories (MoM)** — *Linear Sequence Modeling with Mixture-of-Memories*, arXiv:**2502.13685** (2025) | Multiple independent **memory states** inside one linear-attention layer; a **router** sends each token to a subset of memories (sparse, MoE-style). Reduces memory interference. | Memories are **homogeneous** — all use the **same** update rule; heterogeneity is in *contents* and *routing*, not in *dynamics*. Ours: every head runs a **different recurrence regime** (negative-eigenvalue reflection vs λ=1 integrator vs λ>1 bistable latch vs nonlinear φ), all **active every step** (no routing). |
| **MoE-Mamba** (arXiv:**2401.04081**), **BlackMamba** (arXiv:**2402.01771**), **Mixture-of-Mamba** (arXiv:**2501.16295**), **Routing Mamba** (arXiv:**2506.18145**) | Sparse **mixture-of-experts** over the MLP / projections / modality-specific params of an SSM stack. | The **SSM dynamics are shared/homogeneous**; MoE selects *parameters/projections*, not *recurrence regime*. Token/modality-routed and sparse. Ours is **dense heterogeneous dynamics**, no router. |
| **Multi-head attention head specialization** (e.g. *Attention heads of LLMs: a survey*, Patterns 2025; classic Voita/Clark probing work) | Heads specialize in *function* (positional, syntactic, induction…). | All heads run the **identical** attention operation; specialization is in learned weights, **not in the update rule**. Ours varies the **rule** (λ, β, φ) per head. |

### Verdict: **NOVEL-SYNTHESIS**

The horizontal/within-layer mixing principle is **not new** — **Hymba owns it** (attention+SSM in one layer). What is new is mixing **heterogeneous *recurrent* dynamics spanning distinct capability regimes** (not attention+SSM, but track/count/latch/nonlin) and drawing them from **one parameterized family** rather than assembling distinct modules. **Do NOT claim "first to mix head types within a layer" — that is false (Hymba).** The defensible statement: *"first within-layer mixture of heterogeneous recurrent update rules drawn from a single cell, spanning the four classical capability regimes."*

---

## 2. Component 2 — a SINGLE parameterized cell whose knobs continuously interpolate count / track / latch / nonlinear

**Claim under test.** One cell, (gain × correction × φ), whose knobs move it between the
four capability corners; E88 is one point in it.

### Closest prior art and the exact delta

| prior work | the lever it parameterizes | delta vs the four-corner cell |
|---|---|---|
| **xLSTM** — Beck, Pöppel, …, Hochreiter, arXiv:**2405.04517** (NeurIPS 2024) | sLSTM (scalar memory-mixing, state-tracking) + mLSTM (matrix memory, parallelizable) | These are **two separate block types stacked vertically**, each a distinct cell. Ours is **one cell** whose **continuous knobs** reach both behaviours; no block-type switch. |
| **DeltaProduct** — Siems et al., arXiv:**2502.10297** (2025) | **n_h Householder reflections per token** → diagonal-plus-rank-n_h transition; more reflections ⇒ more state-tracking | A genuine *parameterized capability lever*, but along **one axis** (state-tracking richness) and a **discrete** one. It does **not** address counting or latching. Ours adds the **gain axis** that reaches count (λ=1) and latch (λ>1). |
| **Gated DeltaNet** — Yang, Kautz, Hatamizadeh, arXiv:**2412.06464** (ICLR 2025) | gate (decay) + delta-rule in one cell | One cell, but decay is **clamped to (0,1)** and β∈(0,1) by default (`allow_neg_eigval=False`) — structurally **one regime**; it is **the E88-class point**, not the span (see [`GDN_VS_E88_TRANSITION.md`](GDN_VS_E88_TRANSITION.md)). |
| **Cirone et al. 2024** (arXiv:**2402.19047**); **Fixed-Point RNNs / Movahedi et al. 2025** (arXiv:**2503.10799**) | **diagonal → dense** transition interpolation | One continuous axis (transition density) for **state-tracking**. Again **no counting/latching corner**. |
| general SSM params (Mamba arXiv:2312.00752; S5 arXiv:2208.04933) | continuous A/decay/Δ | stay **inside one regime** (diagonal, decay∈(0,1)). |

### Has the (gain × correction) → capability-corners unification been stated? **No (as far as verified).**

The **individual levers are all known**: eigenvalue **sign** (Grazzi; DeltaProduct),
transition **density** (Cirone; Movahedi), **gating/decay** (GDN; Griffin's RG-LRU,
arXiv:2402.19427). What is **not** in the literature is the framing that **one cell
`φ(λS − βk(kᵀS) + i·kvᵀ)` has the four *classical* capabilities as the corners of a
(λ, β, φ) cube**, with **E88 as a single point** in it, and in particular that **counting
and latching join tracking in the same knob space once the gain is freed**. The
expressivity is demonstrated conclusively by the pinned-preset arms
([`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md) §6.1: track 0.83 where every other
preset is at baseline; latch 1.00 to T=1024; count best-extrapolating linear arm).

### Verdict: **NOVEL-SYNTHESIS**

Component levers are **KNOWN** individually and must be attributed; the **unified
four-corner parameterization** — and specifically **folding counting + latching into the
state-tracking knob space via the free gain** — is the novel synthesis, and is the
strongest single contribution candidate (§6).

---

## 3. Component 3 — GAIN/DECAY range as the capability lever; the (0,1) clamp as the lock

**Claim under test.** Forcing decay into (0,1) ("stable forgetting", cribbed from SSMs) is
the lock that makes a cell leaky (no count, no latch); **freeing the gain** — λ=1 integrate
→ count, λ>1 self-bounded → latch, β>λ → negative eig → track — is the lever.

### What is KNOWN (must attribute, do NOT claim)

- **Negative / reflection eigenvalue unlocks state-tracking.** Grazzi et al.,
  *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, arXiv:**2411.12537**
  (ICLR 2025): extending the eigenvalue range [0,1]→[−1,1] unlocks parity, modular
  arithmetic, permutation composition. **This is exactly the β>λ corner.** Khavari et al.
  (arXiv:2508.07395): parity needs **both** input-dependence **and** negative eigenvalues.
  The artifact's own [`GDN_VS_E88_TRANSITION.md`](GDN_VS_E88_TRANSITION.md) and
  [`S5_MECHANISM_SYNTHESIS.md`](S5_MECHANISM_SYNTHESIS.md) confirm this **causally** — but
  the *mechanism* is Grazzi's, not ours.
- **Counting needs a non-saturating additive accumulator.** Weiss, Goldberg & Yahav,
  *On the Practical Computational Power of Finite Precision RNNs*, arXiv:**1805.04908**
  (ACL 2018): LSTM/ReLU cells count (aⁿbⁿ, aⁿbⁿcⁿ) via an additive non-saturating cell;
  GRU/squashing RNNs cannot. **The λ=1 (no-decay) count corner is this classical result.**
- **Bistable latching** via a self-loop gain ≥ 1 with a saturating nonlinearity is the
  classical Hopfield/attractor mechanism (textbook), and exponential-gate magnitude freeing
  is in **xLSTM** (mLSTM exp gates, arXiv:2405.04517).

### What is NOVEL here

The **sign** lever (Grazzi) and the **counting** result (Weiss) exist *separately*. The new
move is treating **the scalar self-loop gain λ — its magnitude, not only the
eigenvalue sign — as ONE continuous lever** that walks a *single cell* through
leaky → integrate/**count** (λ=1) → bistable/**latch** (λ>1+φ) → reflection/**track**
(β>λ), and **naming the SSM-inherited (0,1) decay clamp as the specific architectural lock**
that forecloses the count and latch corners. The causal "un-cribbing" demo (same kernel,
only λ changes: hold-a-bit 0.43 at λ=0.9 → 1.00 at λ=1.3; counting 0.83 at λ<1 → 0.92 at
λ=1, [`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md) §6.3) is, to my knowledge, not a
restatement of any single prior result.

### Verdict: **KNOWN (sign/tracking + counting halves) + NOVEL-SYNTHESIS (the unified gain-magnitude lever and the "decay-clamp = lock" framing)**

**Overclaim risk is highest here.** The eigenvalue-range-controls-tracking sub-claim is
**Grazzi 2025 and must be cited as theirs**; "counting needs an un-decayed accumulator" is
**Weiss 2018**. The defensible novel residue is the **single gain knob unifying both ends +
the explicit identification of the (0,1) clamp**.

---

## 4. Component 4 — LEARNED per-head specialization of recurrence dynamics via init-placement + knob-specific LR

**Claim under test.** A single layer can be *trained* to hold a heterogeneous
mixture-of-specialists across distinct capability regimes — and it takes **init-spread +
knob-specific LR**, because a **generic-init** learned cell collapses to one regime.

### Closest prior art and the exact delta

| prior work | what it learns | delta vs ours |
|---|---|---|
| **Multi-timescale Representation Learning in LSTM Language Models**, arXiv:**2009.12727** (ICLR 2021) | LSTM forget-gate biases self-organize to an **inverse-Gamma distribution of timescales**; units specialize to different decay rates from data. | Specialization is **within ONE regime** (all leaky decay, different *rates*) — not across **distinct capability regimes** (count vs latch vs track). And it *emerges* from generic training; **ours shows generic training does NOT cross regimes** (knobs stay at center, [`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md) §6.4). |
| **Mamba** per-channel decay (A_log init ~U(0,16), arXiv:2312.00752) | heterogeneous **initialization** of decay across channels, learned input-dependent. | Spreads init across one regime's rates; **no corner-placement**, no regime heterogeneity. |
| **MoM / MoE-SSM routing** (2502.13685; 2401.04081) | a **router** learns to assign tokens to homogeneous experts/memories. | Learns *assignment*, not *heterogeneous dynamics*; experts are identical-rule. |
| **DeltaProduct / Fixed-Point RNNs** | n_h or density is a **fixed hyperparameter**, not learned per-head. | We **learn** the per-head regime placement. |

### What is NOVEL here

Two coupled findings, both backed by controlled experiments:
1. **A negative result:** with **generic init**, a single learned cell at fixed train length
   **does not self-organize** onto the corners — λ never reaches 1, the along-key eigenvalue
   never goes negative, 0% of heads reflect; the projections absorb the task and the
   recurrence knobs stay near init ([`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md)
   §6.4). **Expressivity ≠ learnability.**
2. **The fix:** **init-spread across the four corners + a knob-specific learning rate
   (~10–20×)** makes **100% of heads HOLD their corner, 0% drift to center**, even on a
   *mixed* task that demands all four at once — and the mixture **beats both the best single
   preset and an LSTM** on that mixed task (0.92 @128, 0.72 @1024;
   [`UNIFIED_LEARNABILITY_RESULTS.md`](UNIFIED_LEARNABILITY_RESULTS.md) §2–3). The latch
   heads keep λ>1; the track heads keep negative along-key eig — regimes the generic cell
   never reached.

Per-*unit timescale* specialization is **KNOWN** (2009.12727; Mamba), but those stay in one
regime and emerge for free. A **learned, stable mixture across distinct *capability*
regimes in one layer**, plus the **finding that it requires deliberate inductive placement
(generic init fails)**, appears to have no direct precedent.

### Verdict: **NOVEL** (within-regime timescale specialization is KNOWN; cross-regime learned mixture-of-recurrent-specialists + the init-spread/knob-LR recipe is novel)

This is the **second** strongest defensible claim, because the generic-vs-spread contrast is
a clean controlled ablation rather than a framing.

---

## 5. Component 5 — the "serial-in-time + width-multiprogramming, so we can afford nonlinearity-in-time" systems argument

**Claim under test.** Because the layer is serial-in-time but parallel-over-width (per-head
matrix states, a fused kernel), it is **not scan-bound**, so it can afford a free gain and a
genuine state-nonlinearity-in-time — against the parallel-scan orthodoxy.

### Closest prior art and the exact delta

- **The parallel-scan orthodoxy itself** — S4/S5 (arXiv:2208.04933), Mamba/Mamba-2
  (2312.00752, 2405.21060), GLA (2312.06635), DeltaNet (2406.06484): the entire SSM/linear-
  attention program **constrains the recurrence to be (near-)linear precisely to enable a
  parallel/chunkwise scan**. This is the thesis the artifact pushes against.
- **"Were RNNs All We Needed?"** — Feng, Tung, Ahmed, Bengio, Hajimirsadeghi,
  arXiv:**2410.01201** (2024): minGRU/minLSTM **strip state-dependence from the gates
  specifically to regain the parallel scan** — i.e. they **sacrifice nonlinearity-in-time
  to keep the scan**. This is the *direct opposite* design choice and the cleanest foil.
- **ParaRNN** (arXiv:**2510.21450**) and **DEER / "Towards Scalable and Stable
  Parallelization of Nonlinear RNNs"** (arXiv:**2407.19115**): try to **parallelize
  *nonlinear* RNNs over length** via Newton / fixed-point iteration — they **agree
  nonlinearity-in-time is worth having** and that the tradeoff "isn't fundamental," but pay
  an iterative solve to keep length-parallelism. **Martin & Cundy** (arXiv:**1709.04057**)
  is the canonical linear-scan-over-length reference.

### The delta — and the honest caveat

The artifact's move is **orthogonal**: *don't parallelize over length at all* — parallelize
over **width/heads/batch**, run serial O(T) in time, and the nonlinearity-in-time is then
"free" because length-parallelism was never the throughput source. The throughput numbers
back that it is *competitive* (count variant 40.5k tok/s > LSTM 38.0k > GDN 23.7k @T=128;
within 4% of GDN @T=512, [`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md) §6.5).

**But this is how every serial RNN (Elman/LSTM) has *always* trained — parallel over batch,
serial over time.** The width-multiprogramming framing is a *re-statement of that old fact
as a deliberate design stance*, supported by a throughput measurement at one scale (~8M
params, fp32) with an **8× memory footprint** vs LSTM/GDN. It is a reasonable *positioning*
argument, **not a new systems algorithm**, and the throughput edge is modest and
scale-limited.

### Verdict: **KNOWN (mechanism) / NOVEL framing only — flag as overclaim risk**

The mechanism (serial-time RNN, batch/width parallelism) is as old as RNNs; the "nonlinearity-
in-time is affordable" point is the same one ParaRNN/DEER/Were-RNNs are circling from the
other side. **Do not present this as a systems contribution.** Present it as a *design
rationale* that licenses the free gain and φ — defensible only if paired with the throughput
table and the disclosed memory cost.

---

## 6. The single most defensible novel claim

> **A single parameterized recurrent cell exposes the four classical recurrent
> capabilities — non-solvable-group tracking, unbounded counting, bistable latching, and
> state-nonlinear computation — as corners of a (gain λ, correction β, nonlinearity φ)
> space, and ONE width-multiprogrammed layer can be *trained* (init-spread + knob-specific
> LR) to hold a stable HORIZONTAL mixture of all four specialists at once — beating both the
> best single-corner preset and an LSTM on a mixed all-capabilities task.**

The defensible core is the **integration**: not any one lever (each is anticipated), but
(i) the **unified four-corner parameterization with E88 as one point** (§2), plus (ii) the
**learned, stable, within-layer heterogeneous mixture** with the controlled
generic-fails / spread-holds evidence (§4). That conjunction has **no direct precedent**:
Hymba mixes only attention+SSM; MoM/MoE-SSM mix homogeneous dynamics; xLSTM separates the
cells across blocks; Grazzi/DeltaProduct/Cirone each parameterize **one** axis (tracking)
and none reach the count/latch corners.

---

## 7. Claims that are NOT defensible / are anticipated by prior work (do NOT overclaim)

1. **"We discovered that negative/reflection eigenvalues enable state-tracking."** **FALSE
   to claim — Grazzi et al. 2025 (arXiv:2411.12537); Khavari et al. 2025; DeltaProduct/Siems
   2025.** The artifact *confirms it causally on real models* — frame it as confirmation,
   cite them.
2. **"We discovered that counting needs a non-saturating accumulator."** Anticipated by
   **Weiss, Goldberg & Yahav 2018 (arXiv:1805.04908)** (and Merrill's automata hierarchy).
3. **"First to mix heterogeneous head types within one layer."** **FALSE — Hymba
   (arXiv:2411.13676)** already mixes attention + SSM heads in one layer. Restrict the claim
   to *heterogeneous recurrent rules from one parameterized cell*.
4. **"First mixture-of-experts / multiple-memory recurrence."** Anticipated by **MoM
   (2502.13685), MoE-Mamba (2401.04081), BlackMamba (2402.01771)** — though all are
   homogeneous-dynamics + routed; our distinction is *heterogeneous dense dynamics*.
5. **"Per-head specialization of recurrent timescales is new."** Anticipated for the
   *within-regime* case by **Multi-timescale LSTM (2009.12727)** and Mamba per-channel decay.
   Only the *cross-regime* learned mixture is new.
6. **"Diagonal→dense / single-cell capability interpolation is new."** Anticipated as a
   one-axis lever by **Cirone 2024 (2402.19047), Movahedi 2025 (2503.10799),
   DeltaProduct (2502.10297)**.
7. **"Serial-in-time nonlinearity is a novel systems contribution."** No — serial RNNs are
   classical; the nonlinearity-vs-scan tradeoff is the explicit subject of **ParaRNN
   (2510.21450), DEER (2407.19115), Were-RNNs-All-We-Needed (2410.01201)**. Framing only.
8. **"A single learned cell auto-discovers all four corners."** **The artifact's own data
   refutes this** ([`UNIFIED_CELL_RESULTS.md`](UNIFIED_CELL_RESULTS.md) §6.4: generic init
   stays at center). The honest claim requires the init-spread caveat — do not drop it.

---

## 8. Nearest-neighbor papers we MUST cite and contrast

**Tier 1 — must cite, must contrast explicitly (the reviewer will know these):**
- **Hymba**, arXiv:2411.13676 — within-layer attention+SSM head mixing (the horizontal foil).
- **Mixture-of-Memories (MoM)**, arXiv:2502.13685 — multiple memory states in one layer (homogeneous).
- **xLSTM**, arXiv:2405.04517 — sLSTM+mLSTM as separate blocks (the "one cell vs two blocks" foil).
- **Grazzi et al. 2025**, arXiv:2411.12537 — negative eigenvalues ⇒ state-tracking (the §3 attribution).
- **DeltaProduct / Siems et al. 2025**, arXiv:2502.10297 — n_h reflections, parameterized tracking lever.
- **Gated DeltaNet**, arXiv:2412.06464 — the gate+delta single cell = the E88-class point.
- **Weiss, Goldberg & Yahav 2018**, arXiv:1805.04908 — counting needs a non-saturating cell.

**Tier 2 — must cite (positioning / lineage):**
- Vertical hybrids: **Jamba** (2403.19887), **Griffin/Hawk** (2402.19427), **Samba**
  (2406.07522), **Zamba2** (Zyphra technical report; GitHub `Zyphra/Zamba2`) — the
  vertical-vs-horizontal contrast.
- MoE-over-SSM: **MoE-Mamba** (2401.04081), **BlackMamba** (2402.01771),
  **Mixture-of-Mamba** (2501.16295), **Routing Mamba** (2506.18145).
- Single-axis interpolation: **Cirone 2024** (2402.19047), **Fixed-Point RNNs / Movahedi 2025**
  (2503.10799), **Khavari 2025** (2508.07395).
- Parallel-scan axis: **Were RNNs All We Needed?** (2410.01201), **ParaRNN** (2510.21450),
  **DEER / Lim et al.** (2407.19115), **Martin & Cundy** (1709.04057).
- Per-unit timescale specialization: **Multi-timescale LSTM** (2009.12727).
- The TC⁰/NC¹ anchor: **Merrill, Petty & Sabharwal 2024** (2404.08819), **Barrington 1989**.

---

## 9. Honest gaps — not cited as confirmed (no fabrication)

- **Multi-timescale LSTM (arXiv:2009.12727)** — title and id confirmed via web search; I did
  **not** re-fetch the abstract to confirm the exact author list, so I cite it by **title +
  id + venue (ICLR 2021)** only, not by author names.
- **Zamba2** — cited by name + the Zyphra GitHub/technical-report; I did not locate a stable
  arXiv id this session, so no arXiv id is asserted for it (BlackMamba 2402.01771 is the
  arXiv-anchored Zyphra reference).
- **Samba (2406.07522), Jamba (2403.19887), Griffin (2402.19427)** author lists are the
  widely-known ones (Ren et al.; Lieber, Lenz et al.; De et al.) but where I was not 100%
  certain of the full list I cite **title + arXiv id**, which is sufficient to verify.
- No reference here is asserted that was not surfaced with a matching arXiv/DOI/URL this
  session or independently verified in
  [`PRECISION_NONLINEARITY_RESEARCH.md`](PRECISION_NONLINEARITY_RESEARCH.md) §9.

---

## 10. Honest verdict — how novel is the whole, and the strongest framing that survives scrutiny

**One paragraph.** *Every individual lever in this artifact is anticipated by 2024–2025 prior
work — negative-eigenvalue tracking (Grazzi), n_h-reflection / diagonal→dense interpolation
(DeltaProduct, Cirone, Movahedi), gated delta-rule cells (Gated DeltaNet), counting via
non-saturating accumulators (Weiss 2018), within-layer head-type mixing (Hymba), multiple
memory states (MoM), per-unit timescale specialization (multi-timescale LSTM), and the
nonlinearity-vs-parallel-scan tradeoff (Were-RNNs / ParaRNN / DEER). The artifact is
therefore **not** a new primitive; it is a **NOVEL-SYNTHESIS**. Its genuine, defensible
contribution is the **unification**: showing that one parameterized matrix-recurrence cell
places the four *classical* capabilities — group-tracking, counting, latching, and
state-nonlinearity — as **corners of a single (gain, correction, nonlinearity) space** (with
E88 as one cribbed point), and that **one width-multiprogrammed layer can be trained, via
init-spread plus a knob-specific learning rate, to hold a stable heterogeneous
mixture-of-specialists across all four corners at once** — a within-layer mixture of
*heterogeneous recurrent dynamics* (not attention+SSM as in Hymba, not homogeneous memories
as in MoM, not separate blocks as in xLSTM) that beats both the best single-corner preset and
an LSTM on a mixed task.* The strongest framing that survives a hostile reviewer is exactly
that synthesis claim, **stated with its attributions intact and its three overclaims
dropped**: do not claim the eigenvalue or counting mechanisms (cite Grazzi/Weiss), do not
claim first-within-layer-mixing (cite Hymba), and do not sell serial-time as a systems
result. Positioned that way — "a single trainable cell that unifies the four classical
recurrent capabilities as a learned horizontal mixture-of-specialists in one layer" — the
contribution is real, narrow, and defensible.
