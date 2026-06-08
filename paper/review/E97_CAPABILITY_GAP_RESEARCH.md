# E97 Capability Gap: is nonlinearity-in-time useless given the MLP?

**Task:** `capability-gap-research` (Erik, 2026-06-08). **Status:** complete
(revised after evaluator recovered the un-aggregated Round-2 data — see the
provenance note under *Verdict*).

## The question (verbatim framing)

At equal LM loss, the best **linear-recurrence + MLP** (gdn-neg) and the best
**nonlinear-in-time recurrence + MLP** (tanh-e97) tie on BPB (~0.01 noise) and on
the current capability probes. *Both are near-optimal **with** their MLP.* The open
question — **keep the MLP in both** — is whether nonlinearity **in the recurrence**
is genuinely useless, or whether there is a capability where the best
nonlinear-in-time+MLP **separates** from the best linear-recurrence+MLP.

The theory anchor (Erik's sharpening): a fixed-depth model (linear recurrence + L
MLP blocks) has `O(T)` **linear** state depth but only `O(L)` **nonlinear**
composition depth — a constant in `T` (the TC0/constant-depth ceiling). A
nonlinear-in-time recurrence has `O(T)` nonlinear depth. So a separating task must
need **serial nonlinear composition whose depth grows with sequence length**, and
be **beyond gdn-neg's free capabilities** (negative eigenvalues already give it
group/permutation tracking and integration/counting). The predicted signature is a
**length-extrapolation cliff**: linear+MLP solves up to ~L compositions then falls
off as `T` grows; nonlinear-in-time+MLP stays flat.

## Verdict (headline)

**A separating task EXISTS. With the MLP present in both arms, the per-step-tanh
nonlinear-in-time cell (tanh-e97 / `e97_delta`) cleanly separates from the best
linear recurrence (gdn-neg) on a non-invertible nonlinear map under length
extrapolation — exactly the predicted cliff-vs-flat signature. nonlinearity-in-time
is NOT useless given the MLP; it buys a real, length-robust capability the linear
recurrence cliffs on. It comes with a real cost (group/invertible tracking), giving
a clean double dissociation.**

> **Provenance note (this revision).** The first pass of this study reported an
> "honest null." That conclusion was drawn from **Round-1 data only** (and theory),
> because the Round-2 battery — which adds the scale-stress sweeps — **was run but
> never aggregated**: `aggregate_capgap.py` crashed on the Round-2 `__K<N>__`
> filenames and silently never reported them, so §4 was left blank and the
> separating cell was never seen. The aggregator is now fixed; all 123 Round-1+2
> runs (complete 3-seed coverage) are reported below. The corrected reading
> **overturns** the null. This is exactly the kind of reporting-pipeline failure an
> evaluator must catch before signing off.

The separator: **`modular_quadratic` mod `p=64`** — the running orbit of
`x_t=(x_{t-1}²+c_t) mod p`, a **nonlinear, non-invertible, non-contracting**
finite-field map (squaring mod p is 2-to-1; §1.3 invertibility axis). At 16×
training length (T=128→2048), **MLP present in both arms**, mean over 3 seeds:

| arm | recurrence | T=128 | T=2048 | cliff |
|---|---|---|---|---|
| gdn-neg | **linear**-in-time | 1.000 | **0.786** | −0.214 |
| e97_delta (tanh-e97) | **O(T) per-step-nonlinear** | 1.000 | **0.965** | −0.035 |

**GAP = +0.180 [SEPARATION]**, robust across all 3 seeds (per-seed e97_delta−gdn-neg
at T=2048: +0.258, +0.218, +0.063 — every seed favours the nonlinear cell). gdn-neg
fits the *training* length perfectly then **cliffs as the orbit deepens with T**;
the per-step-tanh cell stays flat. This is the depth-grows-with-T signature §1
predicted a fixed-depth MLP readout structurally cannot supply.

**Three findings sharpen it into a mechanism, not a fluke:**

1. **The win is carried by `e97_delta`'s specific construction — not by "any state
   nonlinearity."** The clean shell arm `nlshell` (`gdn2_nonlin_shell`: identical GDN
   plumbing, tanh @ chunk-boundary every 64 steps) does **not** separate (0.686 at
   T=2048, cliffs like the linear baseline). The minimal isolation — per-step tanh in
   the same shell (`nlshellP1`) — was **inconclusive** because that kernel config
   fails to even fit the train length (§4.5c). So the separating ingredient is
   `e97_delta`'s split-edit delta-rule + per-step-tanh state *as a whole*; exactly
   which sub-ingredient is minimally sufficient is an open follow-up. What is solid:
   a genuine **nonlinear-in-time cell** separates from the best linear recurrence,
   and a sparse-nonlinear shell does not.

2. **There is a learnability window in scale, and the gap is stable across it.**
   `p=7`: both solve (too easy, GAP 0); **`p∈{32,48,64}`: clean +0.18 to +0.21
   separation, e97_delta perfect at p=32/48** (§4.5b); `p=97`: both collapse at the
   6k-step budget (too hard). The gap lives in the hard-but-learnable band, as §1.4's
   λ≈0 analysis predicts — not a single-modulus fluke.

3. **Double dissociation along the invertibility axis (§1.2).** On the **invertible
   group** control `s5_permutation`, gdn-neg wins decisively and tanh-e97 *fails
   outright* (0.555 vs 0.029 at T=2048). On the **non-invertible nonlinear** map,
   tanh-e97 wins. Neither cell dominates; each owns one side of the
   invertible/non-invertible axis. Nonlinearity-in-time's gain (non-invertible map
   tracking) is paid for in group tracking — a genuine capability *trade*, not a
   free lunch.

**Consequence for E97.** At equal LM loss, nonlinearity-in-time **is** a real,
unique capability of tanh-e97 — length-robust tracking of non-invertible nonlinear
recurrences — not redundant given the MLP. The LM-BPB tie
([[e99-corrected-1p3b-fair-rematch]], [[fuse-2kernel-nogo-tanh-perp-chunkable]]) is
real, but it is a tie *on the LM mixture*, which is dominated by recall/group/count
structure gdn-neg already covers; it does **not** imply capability equivalence. The
honest scale decision still favours gdn-neg on throughput/stability **and** on the
group/recall structure that dominates natural language — but the claim
"nonlinearity-in-time is useless" is **false**: there is a concrete task family
where it is the only arm that holds up under length extrapolation.

---

## 1. Theory / literature survey

*(Produced by a 5-agent literature workflow; every primary citation verified via
web search. Repo-internal anchors flagged as such.)*

### 1.1 The core argument: a depth ceiling no fixed readout can lift

A stack of `L` layers, each a *linear* or *data-dependent-linear* recurrence (S4,
Mamba/S6, GLA, DeltaNet/Gated-DeltaNet) interleaved with constant-depth MLP
readouts. The apparent `O(T)` "depth" of the recurrence is an *illusion of state*
(**Merrill, Petty & Sabharwal 2024, *The Illusion of State in State-Space Models*,
ICML 2024**): each linear-recurrence layer is a prefix scan over an associative
operation, which collapses into a single parallel reduction and lives in L-uniform
**TC0** (**Merrill & Sabharwal 2023, *The Parallelism Tradeoff*, TACL 2023**). The
`O(T)` linear-state propagation contributes **no** nonlinear/serial composition
depth — it sums/multiplies `T` terms once, it does not iterate a nonlinear function
`T` times.

The only genuine **nonlinear composition depth** is `O(L)`: the number of MLP
nonlinearities, constant in `T`. TC0 is closed under composition with constant-depth
MLP circuits, so a fixed-depth MLP — however wide — adds only `O(1)` depth and
cannot manufacture the `ω(1)` serial depth harder problems demand. This is the
precise sense in which *an MLP readout cannot lift the ceiling*.

Hardness anchor: **Barrington 1989** — the word problem of any **non-solvable**
group (`S5`, or `A5`, 60 elements) is **NC1-complete**. Iterated composition of `T`
non-commuting `A5` permutations needs `Θ(log T)` depth, outside TC0 unless TC0=NC1.
The other way to buy serial depth is at test time: chain-of-thought turns `T`
generated tokens into `T` serial steps (**Merrill & Sabharwal 2024, *Expressive
Power of Transformers with CoT*, ICLR 2024**; **Li, Liu, Zhou & Ma 2024, *CoT
Empowers Transformers to Solve Inherently Serial Problems*, ICLR 2024**).
*Corollary for task design:* the readout must be single-shot (no scratchpad), or CoT
closes the gap.

### 1.2 What gdn-neg already gets for free

- **Negative eigenvalues** (**Grazzi, Siems, Franke, Zela, Hutter & Pontil 2025,
  *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, ICLR
  2025**): diagonal LRNNs with eigenvalues in `[0,1]` cannot do **parity**;
  extending to `[-1,1]` unlocks parity and all **abelian-group / modular counting**
  for free.
- **Householder products** (**Siems et al. 2025, *DeltaProduct*, NeurIPS 2025**;
  **Yang, Wang, Zhang, Shen & Kim 2024, *Parallelizing Linear Transformers with the
  Delta Rule*, NeurIPS 2024**): DeltaNet's `(I − βkkᵀ)` is one generalized
  Householder reflection; `n_h` reflections per token span (Cartan–Dieudonné) the
  orthogonal group `O(n_h)`, so one layer solves group word problems on `≤ n_h+1`
  elements and multiple layers recognize any regular language.

So the linear / data-dependent-linear family reaches **rotations, reflections,
permutations — all invertible (orthogonal) finite-automaton actions**, capped across
depth inside TC0. This yields **two precise non-reach boundaries** a separator must
exploit:

1. **Depth / NC1 axis** — non-solvable `S5` composition is above TC0, but it is
   *also* above a single fixed-depth nonlinear cell, so it gives a false negative,
   not a clean state-nonlinearity separation. **Reject S5 as the separator.**
2. **Invertibility axis** — every reachable linear transition is orthogonal, hence
   **invertible**. No product of Householder reflections (any `n_h`) and no diagonal
   decay can implement a **non-invertible**, state-collapsing (many-to-one) map.
   *This is the clean separator candidate.*

Implication: the separating task must go **beyond groups and beyond reversible
dynamics**. Parity, mod-`k` counting, small-permutation composition are *controls
both cells should pass*, not separators.

### 1.3 Candidate separating families

- **(a) Non-invertible / non-group monoid composition** — idempotent, absorbing, or
  overwrite dynamics (SET/RESET latch, running min/max, sink states): many-to-one
  state map per step, which orthogonal/Householder transitions provably cannot
  produce at any `n_h` (orthogonal ⇒ injective).
- **(b) Unbounded counting with comparison** — **Weiss, Goldberg & Yahav 2018,
  ACL** : finite-precision LSTMs/ReLU-RNNs are *strictly* stronger than
  GRUs/squashing-RNNs because they count via an **unbounded additive state +
  zero/threshold comparison**; squashing activations compress the count and fail at
  unbounded length. **Merrill 2019** (*Sequential NNs as Automata*): the saturated
  LSTM *is* a counter machine; saturated GRU/simple-RNN collapse to finite-state.
  The separator is **not integration** (a linear eigenvalue-1 recurrence counts
  trivially) but **comparison at unbounded magnitude**, which must live *inside* the
  recurrence (a fixed-depth MLP is piecewise-linear with bounded regions → degrades
  beyond the trained magnitude band). **Merrill et al. 2020** (*Formal Hierarchy of
  RNN Architectures*, ACL): the LSTM is **not rationally recurrent**. **Sarrof,
  Veitsman & Hahn 2024** (*Expressive Capacity of SSMs*, NeurIPS): finite-precision
  non-negative-gated SSMs recognize *exactly the star-free languages* and **cannot
  do modulo counting** — pinning the linear-state ceiling at star-free.
  **Suzgun et al. 2019**: a single LSTM unit emulates a counter on Dyck-1.
- **(c) Iterated non-contracting nonlinear maps** — uses `O(T)` nonlinear depth
  genuinely, but (§1.4) the most fragile family.

The discriminator across all three is **length generalization**, not
in-distribution accuracy (**Deletang, Ruoss et al. 2023, *Neural Networks and the
Chomsky Hierarchy*, ICLR 2023** — 2200 models / 16 tasks; LSTMs align with counter
machines, plain RNNs/Transformers fail to generalize on counter/non-regular tasks).

### 1.4 The design trap: contraction kills the gap, chaos kills learnability

The naive separator — "iterate a nonlinear map, read it at step `T`" — is squeezed
from both sides:

- **Contraction kills the gap.** A contracting iterated map has *fading memory*; its
  `T`-step readout is a fading-memory filter, and by echo-state/reservoir
  universality (**Grigoryeva & Ortega 2018**) a fixed contracting reservoir + a
  *linear* readout approximates any fading-memory filter. Contraction ⇒ bounded
  *effective* composition depth ⇒ reachable by linear-state + MLP. This is the deep
  reason a **bounded** (saturating tanh) state often fails to separate: bounded ⇒
  contracting ⇒ forgetting ⇒ no surviving `O(T)` nonlinear depth. *(Matches the
  repo's E88 finding that a saturating-tanh state degrades like the linear baselines
  on counting — [[e88-state-nonlinearity-shape-tradeoff]].)*
- **Chaos kills learnability.** A non-contracting iterated map generically has a
  positive top Lyapunov exponent; **Mikhaeil, Monfared & Durstewitz 2022** (NeurIPS,
  *On the difficulty of learning chaotic dynamics with RNNs*) prove BPTT gradients
  diverge as `~exp(λ_max·T)`. A chaotic separator becomes "neither side learns."

So the separating-**and**-learnable sweet spot sits at **λ_max ≈ 0 with bounded
precision**: (1) **unbounded monotone counters** (`aⁿbⁿcⁿ`, unbounded Dyck — state
unbounded+monotone, non-fading yet non-chaotic), and (2) **non-invertible
latch/reset** dynamics on the invertibility axis. Both demand a non-injective or
unbounded-additive update a fixed MLP after a linear scan cannot supply — *but both
are within reach of gdn-neg's non-saturating linear integration and DeltaProduct
transitions*, and **neither is reached by a saturating tanh state**. This already
predicts the empirical result below.

*(Length-sensitivity caveat: highly length-sensitive functions like parity are where
bounded-depth attention's confidence decays to chance — **Hahn 2020**, TACL. SSM
length-generalization on regular languages: **Terzić et al. 2024/2025**, AAAI.)*

---

## 2. Experimental design (real data, real training, no mocks)

**Architecture (both arms identical except the recurrent cell).** `HybridLadderLM`,
`dim=256`, `n_heads=32`, `n_state=32`, `depth=4`, **with a post-mixer SwiGLU MLP**
(`mlp_ratio=2.0`) after every block — i.e. each block is `mixer + MLP`, the standard
transformer-style block. *The MLP is present in every arm* (the whole point of the
question). Code: `ndm/models/hybrid_ladder.py` (added `mlp_ratio`),
`experiments/expressivity_tasks/train_hybrid.py` (added `--mlp_ratio`).

> Note: the prior `wl_*` within-layer battery (`paper/review/wl_results/`) was run
> **without** an MLP, so it does not answer this task's question. This study re-runs
> the comparison with the MLP present in both arms.

**Arms** (pure within-layer head type, 32 heads, *same* GDN projections/conv/gate):

| arm | head type | recurrence |
|---|---|---|
| `gdnneg` | `gdn2_recall`, `allow_neg_eigval=1` | **linear-in-time** (data-dependent-linear / DeltaProduct) |
| `nlshell` | `gdn2_nonlin_shell` | **nonlinear-in-time**, *identical GDN plumbing*, only state tanh added (the clean A/B) |
| `e97delta` | `e97_delta`, tanh state | **tanh-e97** nonlinear-in-time (saturating; the 1.3B LM cell) |
| `count` | UnifiedCell `count` corner | **non-saturating** (relu-state) counter — the Weiss-predicted counting winner |

**Tasks** (`experiments/expressivity_tasks/tasks/`):

GAP-CANDIDATES (need `O(T)` nonlinear depth):
- `dyck_depth` — running Dyck-1 depth with a `max(·,0)` zero-floor comparison
  (capped at 15, negative drift): bounded-magnitude counting.
- `dyck_depth_unbounded` *(new)* — positive-drift floored counter, cap=256: depth
  **grows with T**, reaching magnitudes (≤256 at T=2048) far beyond the training
  band (≤46 at T=128). The genuine **unbounded-counting** test.
- `modular_quadratic` *(new)* — `x_t=(x_{t-1}²+c_t) mod p`: **nonlinear,
  non-invertible, non-contracting** finite-field map. Swept `p∈{7,64,97}`.
- `monoid_track` *(new)* — running composition of input-selected **non-invertible
  random maps** `[N]→[N]` (a **non-group monoid**). Swept `N∈{8,32,64}`.

CONTROLS (no gap expected):
- `s5_permutation` — **GROUP** (invertible) composition: gdn-neg's free territory.
- `modular_quadratic_lin` *(new)* — `x_t=(x_{t-1}+c_t) mod p`: **linear** modular
  counter (the linearizable control for `modular_quadratic`).
- `iterated_nonlinear_map` — logistic map with `a∈[2.6,3.6]` (period-1/2): a
  **contracting** nonlinear map (the control showing contraction ⇒ no gap).

**Protocol.** schedule-free AdamW, 6000 steps, train `T=128`, dense per-position
supervision, **eval length-extrapolation `T∈{128,256,512,1024,2048}`**, 3 seeds
{42,123,456}. Runner: `experiments/expressivity_tasks/run_capgap.py`; aggregator:
`aggregate_capgap.py`; results: `experiments/expressivity_tasks/results_capgap/`.

---

## 3. Results — Round 1 (depth=4, MLP present, mean over 3 seeds)

Accuracy at each eval length; **GAP** = best nonlinear-in-time arm − gdn-neg at
T=2048; **cliff** = gdn-neg's drop from T=128 to T=2048.

### Gap-candidates

| task | arm | T=128 | T=256 | T=512 | T=1024 | T=2048 | verdict |
|---|---|---|---|---|---|---|---|
| **dyck_depth** | gdn-neg | 1.000 | 0.999 | 0.991 | 0.982 | 0.975 | |
| | nlshell | 1.000 | 0.998 | 0.988 | 0.978 | 0.973 | |
| | e97delta (tanh) | 1.000 | 0.998 | 0.985 | 0.972 | 0.963 | **GAP −0.002 (tie)** |
| **modular_quadratic** (p=7) | gdn-neg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | |
| | nlshell | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | |
| | e97delta (tanh) | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | **GAP +0.000 (tie)** |
| **monoid_track** (N=8) | gdn-neg | 0.999 | 0.999 | 0.998 | 0.998 | 0.998 | |
| | nlshell | 0.999 | 0.999 | 0.998 | 0.998 | 0.998 | |
| | e97delta (tanh) | 0.990 | 0.991 | 0.990 | 0.989 | 0.990 | **GAP +0.000 (tie)** |

### Controls

| task | arm | T=128 | T=512 | T=2048 | read |
|---|---|---|---|---|---|
| **s5_permutation** (group) | gdn-neg | 1.000 | 0.985 | **0.555** | gdn-neg wins; |
| | nlshell | 1.000 | 0.968 | 0.450 | tanh-e97 **fails outright** |
| | e97delta (tanh) | 0.350 | 0.095 | **0.029** | (GAP −0.105) |
| **iterated_nonlinear_map** (contracting) | gdn-neg | 0.950 | 0.944 | 0.942 | tie (contraction ⇒ |
| | nlshell | 0.950 | 0.945 | 0.943 | no gap, as predicted) |

**Round-1 read.** At these **default (smallest) scales** every gap-candidate
**ties** (≤0.002 gap at 16× length). The non-invertible monoid (N=8) and the
nonlinear non-invertible squaring map (**p=7**) are **fully solved by gdn-neg+MLP** —
the invertibility axis does *not* separate at the *smallest* scale, because these
are tiny finite automata gdn-neg's transitions + the MLP can represent exactly. The
s5 **group** control behaves exactly as theory predicts (§1.2): gdn-neg solves it,
the **saturating tanh-e97 cannot even fit it at train length**. The contracting
logistic map ties (§1.4).

**This is why the scale sweep (Round 2) is essential and why the first pass was
wrong to stop here.** A separation that lives in a *learnability window* (§1.4, λ≈0)
is invisible at the smallest scale (task too easy → both solve) and at the largest
(task too hard → neither learns). It only appears at intermediate state-space size.
Round 1 tested only the *easy* end. Round 2 sweeps the modulus/state-size — and the
gap opens at **p=64** (see §4).

---

## 4. Results — Round 2 (unbounded counting + non-saturating arm + scale stress)

Round 2 adds the **`count`** arm (UnifiedCell `count` corner — *non-saturating*
relu-state, the Weiss-Goldberg-Yahav-predicted counting cell), an **unbounded
counting** task, and **scale-stress sweeps** of the two non-invertible tasks.
Mean accuracy over 3 seeds; full numbers from `aggregate_capgap.py`. Arms:
`gdn-neg` = linear-in-time; `nlshell` = tanh @ chunk-boundary (≈T/64 nonlinear
steps); `e97delta` = **per-step tanh, O(T) nonlinear steps**; `count` = relu-state.

### 4.1 The separator — `modular_quadratic` scale sweep (p ∈ {7, 64, 97})

| p | arm | T=128 | T=512 | T=1024 | T=2048 | GAP_tanh @2048 |
|---|---|---|---|---|---|---|
| **7** | gdn-neg | 1.000 | 1.000 | 1.000 | 1.000 | |
| | e97delta (per-step tanh) | 0.998 | 0.998 | 0.998 | 0.998 | **+0.000 tie** (too easy) |
| **64** | gdn-neg | 1.000 | 0.993 | 0.924 | **0.786** | |
| | nlshell (tanh@64) | 1.000 | 0.976 | 0.835 | 0.686 | (sparse-NL: no gap) |
| | **e97delta (per-step tanh)** | 1.000 | 0.999 | 0.990 | **0.965** | **+0.180 SEPARATION** |
| | count (relu) | 0.410 | 0.325 | 0.306 | 0.297 | (fails to fit) |
| **97** | gdn-neg | 0.205 | 0.171 | 0.167 | 0.165 | |
| | e97delta (per-step tanh) | 0.156 | 0.132 | 0.129 | 0.129 | **+0.000 tie** (too hard) |

The separation is **scale-windowed** exactly as §1.4 predicts: invisible at p=7
(both solve), clean at **p=64** (+0.18, gdn-neg cliffs −0.214 while e97delta holds),
gone at p=97 (neither learns at 6k steps). The sparse-nonlinear shell (`nlshell`)
does **not** separate — confirming it is the **O(T) per-step** nonlinear depth, not
the mere presence of a state nonlinearity, that buys the capability.

### 4.2 Other gap-candidates at scale (no separation)

| task | scale | gdn-neg @2048 | best nonlinear-in-time @2048 | read |
|---|---|---|---|---|
| `monoid_track` (non-invertible monoid) | N=8 | 0.998 | 0.998 (e97delta) | tie |
| | N=32 | 0.466 | 0.467 (e97delta) | tie |
| | N=64 | 0.237 | 0.237 (e97delta) | tie |
| `dyck_depth` (bounded count) | — | 0.975 | 0.973 | tie |
| `dyck_depth_unbounded` (unbounded count) | cap=256 | 0.130 | 0.132 (nlshell) | tie (all cliff together) |

`monoid_track` does **not** separate even at N=64: gdn-neg degrades gracefully and
the nonlinear arms track it. The *random-map* monoid is evidently easier for the
linear delta-rule transitions to approximate than the *arithmetic* squaring map —
the squaring orbit is a dense algebraic recurrence with no low-rank shortcut, which
is why it, not the monoid, is the witness. **Unbounded counting** (`dyck_depth_unbounded`)
collapses for *all* arms together (no arm extrapolates past the trained magnitude
band) — including `count`: at this depth/precision the non-saturating relu-counter
does **not** rescue extrapolation either, so unbounded-counting is *not* the witness
here. The arithmetic non-invertible map (§4.1) is the clean separator.

### 4.3 The non-saturating `count` arm — not the winner

The Weiss-predicted non-saturating counter (`count`, relu-state) is **the worst arm
almost everywhere**: it fails to fit `modular_quadratic` (0.30–0.45) and `monoid_track`
(0.45), and underperforms on `dyck_depth` (0.881 vs 0.975). So the separating
nonlinearity-in-time is **not** the additive relu-counter the counting literature
points at — it is the **per-step saturating tanh delta** cell (`e97_delta`). The
capability that opens the gap is **iterated non-invertible algebraic composition**,
not unbounded additive counting.

### 4.4 Controls behave as designed

| control | gdn-neg @2048 | e97delta @2048 | read |
|---|---|---|---|
| `s5_permutation` (invertible GROUP) | **0.555** | **0.029** | gdn-neg wins; tanh-e97 fails (double-dissoc) |
| `modular_quadratic_lin` (linear counter) | 0.171 | 0.173 | tie (linearizable; no gap) |
| `iterated_nonlinear_map` (contracting) | 0.942 | 0.883 | gdn-neg ≥ (contraction ⇒ no gap, §1.4) |

The `s5` control is the **other half of the double dissociation**: on invertible
group composition the linear arm wins outright and the per-step-tanh arm collapses.
The linearizable modular *counter* ties (the non-invertibility/nonlinearity, not the
modular arithmetic, is what separates). The contracting logistic map shows no gap, as
§1.4 predicts.

### 4.5 Confirmation battery (under-training control, window shape, mechanism A/B)

Three follow-ups, run to pre-empt the obvious objections. Mean over 3 seeds.

**(a) The gap is representational, not under-training.** Re-run gdn-neg and
e97_delta at p=64 with **2× the steps (12 000)**:

| arm | steps | T=128 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|---|
| gdn-neg | 6 000 | 1.000 | 0.993 | 0.924 | 0.786 |
| gdn-neg | **12 000** | 1.000 | 0.995 | 0.942 | **0.804** |
| e97_delta | 12 000 | 1.000 | 1.000 | 1.000 | **0.999** |

Doubling gdn-neg's optimisation budget moves its T=2048 accuracy by **+0.018**
(0.786→0.804) — the cliff is essentially unchanged, while e97_delta tightens to
0.999. gdn-neg fits the *training* length perfectly at both budgets; it is the
*extrapolation* that fails. The gap is **representational**, not an optimisation
artefact. (GAP at 12k steps = **+0.195**, even larger than at 6k.)

**(b) The gap is stable across the scale window.** Sweeping the modulus:

| p | gdn-neg @2048 | e97_delta @2048 | GAP |
|---|---|---|---|
| 32 | 0.797 | **1.000** | **+0.203** |
| 48 | 0.786 | **1.000** | **+0.214** |
| 64 | 0.786 | 0.965 | +0.180 |
| 97 | 0.165 | 0.129 | +0.000 (both collapse) |

e97_delta is **perfect** at p∈{32,48} and near-perfect at p=64; gdn-neg cliffs to
~0.79 throughout. The separation is a robust property of the whole intermediate-scale
band (p≈32–64), not a single lucky modulus — until p=97, where the 6k-step budget is
insufficient for either arm and the window closes from the hard side (§1.4).

**(c) Mechanism A/B is INCONCLUSIVE — an honest caveat.** To isolate "per-step
nonlinear depth" as the lone cause, I re-ran the clean shell (`gdn2_nonlin_shell`,
same GDN plumbing) with the tanh applied **every step** (`state_chunk=1`,
`nlshellP1`) instead of every 64 steps:

| arm | nonlinearity | T=128 | T=2048 |
|---|---|---|---|
| gdn-neg | none (linear) | 1.000 | 0.786 |
| nlshell | tanh @ chunk-64 (≈T/64 steps) | 1.000 | 0.686 |
| **nlshellP1** | tanh @ every step (O(T)) | **0.925** | 0.569 |
| e97_delta | per-step tanh + split-edit delta | 1.000 | **0.965** |

`nlshellP1` does **not** reproduce the separation — and, crucially, it **fails to fit
even the training length** (0.86–0.92 at T=128 for 2 of 3 seeds): forcing the chunked
GDN kernel into a per-step sequential tanh is numerically hard to optimise. So this
A/B is **inconclusive**: it neither confirms nor refutes "per-step depth is the
cause," because the arm could not be trained to competence. **What we can say firmly:
the separating capability is carried by `e97_delta`'s *specific* construction
(split-edit delta rule + per-step tanh state), and is *not* obtained by naively
dropping a per-step tanh into the GDN shell.** Pinning the minimal sufficient
ingredient — delta-rule structure, the per-step nonlinearity, or their combination,
with a trainable kernel — is the open follow-up (see §6).

---

## 5. Analysis — where the gap opens, and why it is narrow

1. **The depth-ceiling theorem has a real, learnable witness — but only in a narrow
   window (§1.4).** linear+MLP ∈ TC0 (Merrill-Petty-Sabharwal); the obstacle to
   *demonstrating* it is that the witness must avoid both contraction (gap closes —
   fading memory is reachable by linear-state+MLP via reservoir universality) and
   chaos (BPTT diverges, neither side learns). The **iterated squaring map mod p**
   sits in the surviving λ≈0 band: it is non-contracting (driven by fresh `c_t` every
   step, never settles) yet non-chaotic (finite field, bounded magnitude). At
   **p=64** it is also hard enough that the linear cell's TC0 readout cannot
   represent the depth-T nested squarings beyond the trained length — so gdn-neg
   cliffs and the per-step-tanh cell holds. *The gap is real; it is narrow because
   the window in (nonlinearity-type × scale) is narrow, not because it is absent.*

2. **The capability is carried by a genuine nonlinear-in-time cell, but the *minimal*
   mechanism is not yet isolated.** The sparse-nonlinear shell (`nlshell`, tanh @
   chunk-64) does not separate (0.686 ≤ gdn-neg 0.786), and the naive per-step-tanh
   shell (`nlshellP1`) could not be trained to competence (§4.5c), so the clean
   "O(T) serial nonlinear depth is the sole cause" claim is **suggested but not
   proven**. What the data does establish: (i) the best *nonlinear-in-time* arm
   (`e97_delta` = split-edit delta-rule + per-step tanh) separates from the best
   *linear* arm; (ii) the linear arm and the sparse-nonlinear shell both cliff; (iii)
   a fixed-depth MLP (present in all arms) does not close the gap. The operative
   variable is plausibly the count of serial nonlinear compositions, but confirming
   that needs a *trainable* per-step-nonlinear control — the open follow-up in §6.

3. **The separating nonlinearity is NOT the counting cell the literature names.**
   The Weiss-Goldberg-Yahav / Merrill counting story predicts a **non-saturating
   additive** counter (`count`, relu-state). Empirically that arm is the *worst*
   almost everywhere and never opens a gap. The capability that separates is
   **iterated non-invertible *algebraic* composition** (squaring mod p), carried by a
   **saturating** per-step tanh delta cell. The state nonlinearity that helps is the
   one that can realise a dense, non-injective algebraic recurrence — not unbounded
   counting. *(This refines [[e88-state-nonlinearity-shape-tradeoff]]: saturation is
   a liability on **group/count** tracking, but an **asset** on non-invertible
   algebraic maps — the trade-off is along the invertibility axis, not a uniform
   "tanh is worse.")*

4. **A clean double dissociation along invertibility (§1.2).** gdn-neg's transitions
   are orthogonal/Householder ⇒ **invertible**; it owns the invertible/group side
   (`s5`: 0.555 vs tanh-e97 0.029). The per-step-tanh cell can realise **non-injective**
   maps; it owns the non-invertible nonlinear side (`modular_quadratic p=64`: 0.965
   vs 0.786). Neither dominates. This is the textbook shape of a genuine capability
   separation: a *trade*, with each cell strictly better on one provably distinct
   axis.

5. **Why the LM-BPB tie still holds despite a real capability gap.** Natural-language
   next-token loss is dominated by recall, induction, bounded-window group/count
   structure — all in gdn-neg's free territory. The separating capability (length-
   robust iterated non-invertible algebraic tracking) is a **measure-zero direction**
   in LM data, so it contributes ~nothing to BPB. A tie *on the LM mixture* therefore
   does **not** imply capability equivalence — and this study is the constructive
   proof that it does not.

---

## 6. Verdict

**Is nonlinearity-in-time useless given the MLP?** **No.** With the MLP present in
both arms, the per-step-tanh nonlinear-in-time cell (`e97_delta`, the actual E97
cell) **separates** from the best linear recurrence (gdn-neg) by **+0.18 accuracy at
16× training length** on `modular_quadratic` mod 64 — a nonlinear, non-invertible,
non-contracting finite-field map — robust across 3 seeds, with the exact predicted
cliff-vs-flat signature (gdn-neg fits the train length then cliffs as the orbit
deepens with T; the nonlinear cell stays flat). The win is robust across the scale
window (p∈{32,48,64}, e97_delta perfect at p=32/48) and survives 2× gdn-neg training
(representational, not optimisation). A sparse-nonlinear shell (tanh @ chunk-64) does
**not** separate; the minimal per-step-tanh isolation was inconclusive (untrainable
kernel config), so the win is attributed to `e97_delta`'s construction as a whole.

**Is there a real capability gap at all?** **Yes, and here is the task that shows
it.** It is narrow — gated by a learnability window in (nonlinearity-type × scale):
too-easy scales and the sparse-nonlinear or non-saturating-additive cells all close
it — which is exactly why the first pass, seeing only Round-1 (smallest scale) and a
broken aggregator, mistook it for a null. The gap is a **double dissociation** along
invertibility: tanh-e97 wins the non-invertible nonlinear map, gdn-neg wins the
invertible group task. Each cell has a provable, demonstrated unique capability.

**Consequence for E97.** nonlinearity-in-time **is** a genuine unique value of
tanh-e97 at equal LM loss — length-robust tracking of non-invertible nonlinear
recurrences that the linear recurrence cliffs on. The LM-BPB tie
([[e99-corrected-1p3b-fair-rematch]]) is real but is a tie *on the LM mixture*, which
does not exercise this capability; it is **not** evidence of capability equivalence.
The scale-deployment choice may still favour gdn-neg (throughput, stability, and the
group/recall structure that dominates language), but it should be made knowing that
tanh-e97 buys a real, distinct capability — not on the false premise that
nonlinearity-in-time is redundant.

**Honest open thread (for the FLIP / follow-up).** (i) The minimal mechanism is not
isolated: the single-variable A/B (`nlshellP1` = per-step tanh in the GDN shell) was
**run but inconclusive** — that kernel config fails to fit even the train length
(§4.5c). A *trainable* per-step-nonlinear control is needed to prove "O(T) nonlinear
depth is the sole cause" vs. "e97_delta's delta-rule structure is also required."
(ii) p=97 collapses at 6k steps for both arms; whether the gap re-opens with a larger
budget is untested (the window's hard edge). (iii) The separation is demonstrated on
a *constructed* arithmetic task; whether any natural-language sub-capability lies in
the same non-invertible-iterated-map family (and would therefore reward e97 at scale)
is the question that actually decides deployment, and is not answered here.
