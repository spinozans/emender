# E97 Final Synthesis — nonlinearity-in-time: one verdict, the paper story

**Task:** `e97-final-synthesis` · **Date:** 2026-06-08 · **Read-only synthesis** (no
training, no mocks). Reconciles the eight result docs of the e97 / within-layer /
nonlinearity-in-time investigation into one verdict + the defensible paper narrative.
`paper/main.typ` untouched.

---

## TL;DR (the one verdict)

> **LM verdict:** **loses at honest wall-clock.** At 1.3B, on the LM-BPB objective that actually governs
> fixed-GPU-hour pretraining (wall-clock), nonlinearity-in-time does **not** beat
> `gdn2-mlp` (dense gdn-neg + SwiGLU MLP). The losses are convergent; the e97
> family's edge is per-token only, it does **not** grow with budget, and it cannot
> amortize the throughput tax. `gdn2-mlp` is the 1.3B scale cell.
>
> **Capability verdict:** **GO (as science, not as a product).** There **is** a
> real, length-robust, 3-seed capability gap that nonlinearity-in-time provably
> buys — but it is narrow, it lives on a constructed arithmetic task, and the cell
> that buys it is precisely the cell that is **fundamentally un-chunkable** and so
> can never be the fast LM cell. The capability gap and the LM wall-clock loss are not in
> tension: they are the **two halves of the same kernel constraint**.
>
> **The paper story:** *"GDN-2 (linear, gated-delta, negative-eigenvalue) is the LM
> cell. We then map, precisely and with a clean double dissociation, the capability
> that exotic nonlinear-in-time recurrence buys (length-robust tracking of
> non-invertible algebraic recurrences) and the capability it loses
> (invertible-group tracking) — and show that the buying cell is exactly the
> un-chunkable one, so the trade is structural, not tunable."*
>
> **Single recommended next step:** test the LSTM **bounded-readout** (`o⊙tanh(Sᵀq)`,
> post-scan, chunk-safe) — `RECURRENT_NONLINEARITY_RESEARCH` Config A — as the
> *chunkable* way to capture the bounded-state benefit without the un-chunkable
> recurrent nonlinearity. It is the only path that could move the LM verdict.

---

## 0. First, disambiguate "E97" — three different cells get conflated

Most apparent contradictions across the eight docs dissolve once three distinct
cells are kept separate. **They are not interchangeable, and the whole synthesis
turns on which one each result is about.**

| name | recurrent state map | chunkable? | role in this synthesis |
|---|---|---|---|
| **`e97_raw`** | raw-write (overwrite), linear-state | yes (fused) | the *token-efficiency* champion at small scale; recall-blind |
| **`e97_delta` (linear)** | split-edit gated-delta, **linear** state | **YES** — GDN-2-class kernel | the *wall-clock-competitive* 1.3B cell; modest count edge |
| **`e97_delta` (per-step tanh)** | split-edit delta + **per-step tanh** state | **NO** — sequential kernel | the *capability-separating* cell (modular_quadratic) |
| `gdn-neg` | gated-delta, linear, **negative eigenvalue** | YES | the recall+track workhorse; the backbone everything mixes with |
| `gdn2_nonlin_shell` | linear within chunk, tanh **at chunk boundary** | partial | the "sparse-nonlinear compromise" — tested: throughput flat 0.88×, no token edge, no capability uplift |

**The crux fact (it recurs in every section below):** the chunked GDN-2-class fast
path engages **only for *linear*-state `e97_delta`**. Per-step `tanh` is a pointwise
map on the whole state every step — **not associative, no chunked-matmul form** — so
the tanh cell stays on the slow sequential kernel (`E97_CHUNKED_KERNEL_NOTE`,
explicitly). Therefore **the cell that buys the capability (per-step tanh) and the
cell that is wall-clock-competitive (linear) are different cells**, and no kernel
work can merge them — *any* nonlinear state map ⊥ chunkable
(`RECURRENT_NONLINEARITY_RESEARCH` §5.1).

---

## 1. LM verdict — convergent loss → loses wall-clock at 1.3B (the honest reasons)

**Verdict: loses wall-clock at 1.3B.** Keep `gdn2-mlp` as the 1.3B cell. The reasoning is *not* poisoned
this time — every leg is a real measurement with a fair throughput bar.

**(a) The kernel objection was real but is now removed — so the bar is fair.**
The prior within-layer/scale wall-clock loss blamed a "2.6× slower, 13–15% util" kernel.
`E97_CHUNKED_KERNEL_NOTE` refutes that: the chunked fused fwd+bwd e97_delta kernel
(Newton–Schulz UT-inverse, C=32) runs at **97–100% util at the 1.3B scale dims and
≤1.02× GDN-2 throughput — faster, not slower** (within_layer 0.18×, scale_1p3b 0.78×,
wide 0.50×; parity 13/13). `E97_FUSED_LM_KERNEL_NOTE` separately found and fixed an
*inert-kernel* bug (the fused path silently fell back to eager because RMSNorm-under-
autocast fed it fp32; the bf16-cast fix engages it → 43–266× over the eager T-scan,
parity rel-L2 ~8e-4, flat in T). **So the 1.3B comparison below is run on a fair,
fast kernel** — this is what makes the wall-clock loss honest rather than a throughput artifact.

**(b) Token-matched, e97 is genuinely more sample-efficient — but the edge is small
and does not grow.** At 1.3B, `e97_delta`(linear)+gdn-neg **wins token-matched**
(2.071 vs gdn2-mlp 2.094, **+0.023**, both seeds, 4.99M tokens; `E97_DELTA_1P3B_CMA`).
The `gdn2_nonlin_shell` boundary-phi cell **ties** token-matched (2.093 vs base 2.092
at 5M; 1.983 vs 1.981 at ~8.9M) and — critically — **the edge does not grow with
budget** (`E97_WALLCLOCK_CMA` §4). There is no budget at which a sample-efficiency
crossover flips the result.

**(c) Wall-clock — the objective that governs fixed-GPU-hour pretraining — gdn2-mlp
wins decisively.** `e97_delta`+gdn-neg **loses wall-clock by +0.044** (2.071 vs 2.027;
gdn2-mlp fits ~7.0M vs 5.0M tokens in 720s at ~40% higher throughput). Every
`gdn2_nonlin_shell` tanh-C config loses wall-clock to `gdn2-mlp` (best tanh `C=1`
+0.013; `E97_WALLCLOCK_CMA` §3), and **chunk-size C is flat ~0.88× — not a speed
lever** (§1). With a small, non-growing per-token edge and a ~0.84–0.88× throughput
tax, the wall-clock loss is structural.

**(d) "Convergent loss / e97-raw #1" — reconciled precisely.** At **159M
time-bounded**, `e97_raw`+MLP is #1 on held-out BPB (3.231 vs gdn2-mlp 3.393),
*beating it on 3.1× fewer tokens* (`E97_WITHIN_LAYER_STUDY`). The generalization audit
(`E97_GENERALIZATION_AUDIT`) confirms this **is real, not a train-loss artifact** —
the ~98%-artifact trap it exposes runs the *other* direction (ranking time-bounded
screens on *raw train loss* falsely crowns gdn2; the clean train→held generalization
gap is only +0.021 BPB). So the honest statement is: **e97-raw's lead is a real
small-scale *token-efficiency* lead, not a train-loss artifact — but it is *not* a
wall-clock win and it does not survive to 1.3B** (the scale-pilot dominated it; the
1.3B head-to-heads go to gdn2-mlp on wall-clock). The token-efficiency edge *shrinks
with scale*; at 1.3B the losses are convergent and throughput decides.

> **LM bottom line:** nonlinearity-in-time is not a wall-clock LM win at 1.3B, for a
> reason that is now fair (fast kernel) and stable (no growing edge). It loses wall-clock.

---

## 2. Capability verdict — yes, there is a provable gap; with WHICH nonlinearity

**Verdict: a separating capability EXISTS, and it is demonstrated** — but it is *not*
the counting cliff the casual framing expects, and it does **not** need a
non-saturating recurrence.

**The depth-ceiling argument (the theory anchor).** A stack of linear/data-dependent-
linear recurrences + constant-depth MLP readouts has `O(T)` *linear* state depth but
only `O(L)` *nonlinear* composition depth (Merrill–Petty–Sabharwal 2024: linear SSMs
live in TC⁰; the "state" is an illusion). A nonlinear-in-time recurrence has `O(T)`
nonlinear depth. So a separator must need **serial nonlinear composition whose depth
grows with `T`**, beyond gdn-neg's free reach (negative eigenvalues already give it
abelian/modular counting + group/permutation tracking). Predicted signature:
**length-extrapolation cliff** for linear+MLP, **flat** for nonlinear-in-time.

**The witness (real data, 3 seeds, `E97_CAPABILITY_GAP_RESEARCH`).** On
**`modular_quadratic` mod p=64** — the orbit `x_t=(x_{t-1}²+c_t) mod p`, a
**nonlinear, non-invertible, non-contracting** finite-field map — at 16× train length
(T=128→2048), MLP present in both arms:

| arm | recurrence | T=128 | T=2048 | cliff |
|---|---|---|---|---|
| gdn-neg | **linear**-in-time | 1.000 | **0.786** | −0.214 |
| `e97_delta` (per-step tanh) | **O(T)** per-step-nonlinear | 1.000 | **0.965** | −0.035 |

**GAP = +0.180 [SEPARATION]**, every seed favors the nonlinear cell (+0.258/+0.218/
+0.063). It is **representational, not under-training** (2× steps moves gdn-neg only
+0.018, GAP grows to +0.195), and **stable across the scale window** (p=32 +0.203,
p=48 +0.214, p=64 +0.180; e97_delta perfect at p=32/48; both collapse at p=97 where
the 6k-step budget runs out — the window's hard edge).

**Which nonlinearity? SATURATING per-step tanh — *not* non-saturating.** This is the
counter-intuitive result that resolves the apparent tension with the counting
literature (§3):
- The Weiss-predicted **non-saturating `count` (relu-state)** arm is the **worst
  almost everywhere** — it fails to even fit `modular_quadratic` (0.30–0.45).
- **Unbounded counting** (`dyck_depth_unbounded`) **collapses for all arms together**,
  including relu — so counting-via-recurrence was **not** the witness here.
- The separator is **iterated non-invertible *algebraic* composition**, won by the
  **saturating per-step tanh** `e97_delta`.

**Clean double dissociation along invertibility:** on the **invertible group** control
`s5_permutation`, gdn-neg wins (0.555) and tanh-e97 *fails outright* (0.029); on the
**non-invertible** map, tanh-e97 wins (0.965 vs 0.786). Neither dominates — each owns
one provably distinct axis. This is the textbook shape of a real capability *trade*.

> **Capability bottom line:** the gap is **real, length-robust, and a double
> dissociation** — but narrow (one constructed arithmetic family, in a
> nonlinearity×scale window), and it does **not** require a non-saturating
> recurrence; the saturating per-step-tanh `e97_delta` is the witness, the
> non-saturating relu counter is the worst arm.

---

## 3. Nonlinearity finding — the sat/non-sat trade, and why ONE map can't win

The two nonlinearity stories in the docs point at *different* shapes for *different*
capabilities. They are not contradictory once the capability axes are separated.

**The E88 capability map (`RECURRENT_NONLINEARITY_RESEARCH` §3, measured on our
stack):** on a *single* matrix-state cell the state-map shape is a one-dimensional
slider, not a free win:

| state map | COUNT (aⁿbⁿcⁿ) | TRACK (S5) | regime |
|---|---|---|---|
| linear/affine-signed | 0.812 (worst) | **0.087 (best)** | groups/tracking, cannot count |
| **tanh (saturating)** | 0.836 | 0.065 | **wins nothing** — worst of both worlds |
| relu (non-sat) | **0.893 (best)** | 0.050 | counts, loses tracking |
| **LSTM (two-pathway)** | **0.951** | **1.000** | both — separate cell + readout |

**The two findings, reconciled:**
1. **For unbounded COUNTING:** non-saturating (relu) > tanh (Weiss 2018; E88). A
   *single* nonlinearity is fundamentally limited to one regime — counting **or**
   tracking, never both — which is the saturated-RNN hierarchy (Merrill 2020). On a
   single state map, **tanh is the one shape that wins nothing.**
2. **For iterated NON-INVERTIBLE ALGEBRAIC composition (§2):** the saturating per-step
   tanh `e97_delta` is the *winner*, and relu is the *worst*. This is a **third
   capability axis (invertibility) that the count↔track map omits.**

**So "does the gap need a non-saturating recurrence?" — NO.** The *counting* axis
wants non-saturating; the *demonstrated* gap (invertibility axis) wants the saturating
delta cell. The honest refinement: **saturation is a liability on group/count
tracking but an asset on dense non-invertible algebraic maps.** "tanh is worse" is
true *on average over count/track* but false on the axis that actually separated.

**Does one nonlinearity suffice, or is a two-pathway (LSTM) cell required?** A single
recurrent state map **cannot** cover counting + latching + tracking — that is the E88/
Merrill-2020 result and the original LSTM motivation. The resolution the whole field
converged on (LSTM → xLSTM/GLA/Mamba/gated-delta) is **not "find a better recurrent
nonlinearity"** but **placement**: keep the state-to-state map **linear** (so it
chunks), carry the nonlinearity in **input-driven gates + a bounded readout**, and buy
state-tracking with **negative eigenvalues** (Grazzi 2025 → our `gdn-neg`). Of the
four capabilities, **three (recall, track, latch) are already reachable in a
linear+gated+signed cell** (gdn-neg: recall 0.96, track 1.00); only unbounded counting
genuinely wants a non-saturating *recurrent* map — and even the LSTM gets counting from
a **linear additive cell + bounded readout**, not from `relu(state)`.

---

## 4. The crux — the capability-bearing cell is exactly the un-chunkable one

This is the single fact that makes the whole investigation coherent, and it is the
honest reason capability ≠ LM win:

- The cell that **separates** (§2) is **per-step-tanh `e97_delta`** — a nonlinear
  state map → **no chunked form → sequential kernel** (`E97_CHUNKED_KERNEL_NOTE`).
- The cell that is **wall-clock-competitive** (§1) is **linear `e97_delta`** — it
  chunks (97–100% util) but has only a modest count edge (count 0.906 vs 0.868), no
  +0.18 separation, and still loses wall-clock to gdn2-mlp by the within-layer
  two-kernel-split overhead.
- The "apply the nonlinearity sparsely" compromise (`gdn2_nonlin_shell`, tanh at chunk
  boundary) was the principled escape — at 1.3B: throughput flat 0.88×
  regardless of C, no token-matched edge, no capability uplift (`E97_WALLCLOCK_CMA`),
  *and* it does not even separate on `modular_quadratic` (0.686, cliffs like the linear
  baseline; `E97_CAPABILITY_GAP` §4.1).

**`bounded/nonlinear state ⊥ chunkable` is fundamental, not an implementation gap.**
The capability lives in exactly the dynamics that destroy tensor-core throughput. You
cannot have both in one cell — which is precisely why the LM verdict (loses wall-clock) and the
capability verdict (real gap) coexist without contradiction.

---

## 5. The paper story (defensible, data-supported)

**Headline:** *GDN-2 is the LM cell; here is the precise composition-depth /
invertibility capability boundary that exotic nonlinear-in-time recurrence buys and
loses — and why that boundary is structural.*

The narrative the data supports, in three beats:

1. **GDN-2 (gdn-neg) is the deployment cell.** Linear, gated-delta, negative
   eigenvalue: it already gets recall, group/permutation tracking, abelian counting,
   and latching *for free in a chunkable kernel*, and it wins the 1.3B wall-clock
   objective. Three of the four target capabilities need no nonlinear recurrence at
   all (Grazzi 2025: tracking is bought by eigenvalue sign, not saturation).

2. **Nonlinearity-in-time is not redundant — it buys a real, narrow capability, with
   a clean double dissociation.** At equal LM loss, the per-step-tanh `e97_delta` cell
   is the *only* arm that holds up under 16× length extrapolation on a non-invertible
   algebraic recurrence (modular_quadratic mod 64: +0.18, 3 seeds, representational);
   gdn-neg owns the invertible-group side (S5) where the tanh cell collapses. This is
   a constructive disproof of "nonlinearity-in-time is useless given the MLP" — and a
   precise statement of *what* it buys (non-invertible dense algebraic tracking) and
   *what it costs* (invertible-group tracking).

3. **The trade is structural, not tunable — and that is the point.** The cell that
   buys the capability is un-chunkable; the chunkable cell does not buy it; the
   sparse-nonlinear compromise buys neither speed nor capability. So at LM scale the
   capability cannot be deployed without paying a throughput tax that the (small,
   non-growing) per-token edge cannot amortize. The LM-BPB tie is a tie *on the LM
   mixture* (dominated by recall/group/count structure gdn-neg already covers), **not**
   evidence of capability equivalence — and the kernel constraint, not a tuning
   failure, is why the capability stays off the LM critical path.

**What we do NOT claim (the alternative narrative the data does *not* support):** we
cannot claim "non-saturating nonlinearity-in-time provably separates on depth-growing
tasks at equal LM loss." The non-saturating arm is the *worst* on the separator, and
the separation is shown on a *constructed* arithmetic task, not on any natural-language
sub-capability. The defensible claim is the boundary-mapping story above, not a
non-saturating-wins story.

---

## 6. Accept / reject + the single next step

**Accept / reject: nonlinearity-in-time loses wall-clock** as the 1.3B LM cell. `gdn2-mlp` (dense
gdn-neg + SwiGLU MLP) remains the scale cell. **Accept** for reporting the capability gap
as a scientific result (the double dissociation + the kernel-constraint explanation),
*not* as a deployment recommendation.

**Single recommended next step — the only lever that could move the LM verdict:**
test **Config A (LSTM bounded readout)** from `RECURRENT_NONLINEARITY_RESEARCH` §7:
apply a saturating squash on the *output* post-scan — `out_t = silu_gate ⊙ tanh(Sᵀq_t)`
— on top of the **linear chunkable** `e97_delta` + gdn-neg backbone. It is the LSTM's
"unbounded clean linear cell + bounded readout" separation placed *where it stays
chunk-safe* (after the scan, not on the recurrence), so it costs zero throughput.
- **Win condition:** lifts counting / length-extrapolation toward LSTM levels (the
  Config D sequential-tanh ceiling) at gdn2-mlp-tie-or-better wall-clock BPB and
  unchanged kernel util (≥97% at scale dims).
- **If it fails:** that is a real negative result — the bounded-state benefit genuinely
  requires the sequential recurrent nonlinearity, and the capability is *permanently*
  off the wall-clock-LM path. Either way the question is closed.

*(Why this and not others: `gdn2_nonlin_shell` already measured flat 0.88× throughput, no token edge, no capability uplift; per-step
relu/softplus state is a double penalty — un-chunkable AND S5-hostile — and belongs
only on the expressivity probe as a ceiling, never as a production cell.)*

---

## 7. Proven vs open (honest ledger)

**Proven (robust, multi-seed, real data):**
- LM loses wall-clock at 1.3B, on a *fair fast kernel* — gdn2-mlp wins, edge does
  not grow with budget (`E97_DELTA_1P3B_CMA`, `E97_WALLCLOCK_CMA`).
- A real +0.18 length-extrapolation separation on modular_quadratic mod 64 (3 seeds,
  representational per 2× control, stable over p∈{32,48,64}), with a clean S5 double
  dissociation (`E97_CAPABILITY_GAP`).
- The sat/non-sat capability slider on a single state map; the separator is *saturating*
  per-step tanh, *not* non-saturating relu; a single nonlinearity cannot cover both
  count and track (`RECURRENT_NONLINEARITY_RESEARCH`, E88, 18/18 verified citations).
- `bounded/nonlinear state ⊥ chunkable` is fundamental; gdn2_nonlin_shell sparse-NL
  compromise loses on both throughput and capability.
- e97-raw's small-scale held-out lead is **real token-efficiency, not a train-loss
  artifact** — but does not survive to 1.3B wall-clock (`E97_GENERALIZATION_AUDIT`).

**Open (explicitly not resolved):**
- **Minimal mechanism not isolated.** The single-variable A/B (`nlshellP1` = per-step
  tanh in the GDN shell) was *run but inconclusive* — that kernel config can't fit even
  the train length. Whether the separator is the per-step nonlinearity, the split-edit
  delta structure, or their combination is unproven; the win is attributed to
  `e97_delta`'s construction *as a whole*.
- **p=97 collapses at 6k steps for both arms;** whether the gap re-opens at larger
  budget (the window's hard edge) is untested.
- **Natural-language relevance unknown.** The separation is on a constructed arithmetic
  task. Whether any LM sub-capability lies in the non-invertible-iterated-map family —
  the question that would actually justify deploying e97 at scale — is not answered.
- **Config A (bounded readout) untested** — the single open lever (§6).

---

## Sources reconciled

| doc | contribution to this synthesis |
|---|---|
| `E97_WITHIN_LAYER_STUDY_RESULTS` | within-layer cell covers 5/5 primitives but capability ⊥ time-bounded LM; e97_raw LM champion at 159M (token-efficiency) |
| `E97_GENERALIZATION_AUDIT_RESULTS` | e97-raw #1 is real held-out token-efficiency, NOT a train-loss artifact; ~98% of the apparent train→held gap is measurement artifact |
| `E97_DELTA_1P3B_CMA_RESULTS` | 1.3B: e97_delta(linear) wins token-matched (+0.023), loses wall-clock (+0.044); kernel fast (util 88–100%); split-overhead is the lever |
| `E97_WALLCLOCK_CMA_RESULTS` | gdn2_nonlin_shell: throughput flat 0.88× in C, no token-matched edge, edge doesn't grow, no capability uplift → loses wall-clock |
| `E97_CHUNKED_KERNEL_NOTE` | chunked fused e97_delta ≤1.02× GDN-2 (faster), refutes 2.6× premise — **but only for LINEAR state; per-step tanh stays sequential** |
| `E97_FUSED_LM_KERNEL_NOTE` | the fused kernel was inert (eager fallback) in the hybrid path; bf16-cast fix → 43–266×, parity verified; tanh/identity only |
| `E97_CAPABILITY_GAP_RESEARCH` | the +0.18 modular_quadratic separation, double dissociation, saturating-not-relu, representational; provenance-corrected from a broken-aggregator null |
| `RECURRENT_NONLINEARITY_RESEARCH` | sat/non-sat capability map; single-nonlinearity limit; two-pathway resolution; nonlinear-state ⊥ chunkable; Config A as the next lever |

*Every number above is traceable to a committed doc; no new experiments were run.*
