# ARCHITECTURE ZOO CLASSIFICATION — is E97 / GDN-2 "enough" (counting AND group-tracking)?

**Task:** `zoo-classify` · **Model:** claude:opus · **Type:** READ-ONLY code analysis (no GPU, no
training). **`paper/main.typ` was NOT edited.** Every classification below is read from the **real
state-update in the repo**, with `file:line` citations. Where an architecture's runnable update is
**not** in this repo (external dependency), it is flagged **NOT FOUND (external)** and classified
only from whatever equation *is* in-repo (Lean formalism or a documented kernel derivation), never
guessed.

The broad E-series/Mamba/baseline sweep was produced by a 52-agent classification workflow (one
agent per model file, each reading the per-token state update from source); the headline
architectures (E88, **E97**, GDN, **GDN-2**, M²RNN, LSTM) were read and verified by hand. Where an
agent's eigenvalue read conflicted with the source or the empirical anchor, the hand-verified value
is used and the discrepancy is noted.

---

## 0. The predictive axis (two independent levers), with its anchors

Prior diagnostics established that the expressivity of a recurrent state map is governed by **two
independent properties of the per-token update `S_t = f(A_t·S_{t-1} + B_t)`**, *not* by
"linear-vs-nonlinear state":

**Lever 1 — MEMORY SHAPE (predicts COUNTING).**
- **squashed** (`tanh`/`sigmoid` on the state) → `|S|` bounded → finite-state → **cannot count**.
- **corrective / delta** (read-modify-write toward a target: `delta = v − S@k`, or convex blend
  `h = f·h + (1−f)·z`) → erases → **cannot count**.
- **additive / accumulating** (unbounded: `S += outer`, `relu`/`softplus` non-saturating state, or
  an LSTM additive cell `c = f·c + i·g`) → **can count**.
- **affine / linear** (identity state map, decay-limited) → partial at best.

**Lever 2 — TRANSITION EIGENVALUE RANGE (predicts S5 / non-solvable-group tracking).**
- Can the **along-key eigenvalue** of `A_t` reach **negative** (a reflection, e.g. `decay·I − k̂k̂ᵀ`
  has along-key eig `decay−1 < 0`)? → **can track S5**.
- Or is it **pinned positive** (e.g. `g·(I − β k̂k̂ᵀ)` with `β∈(0,1)` → `g(1−β) > 0`)? → **cannot**.
- A flag may toggle it (`allow_neg_eigval` multiplies `β` by 2 → `β∈(0,2)` → negative reachable;
  `pos_eigval_clamp` forces eig `≥ 0`).
- A **fully nonlinear** elementwise RNN (LSTM/ReLU-RNN) can reach S5 through the **nonlinearity
  itself** (Merrill–Petty–Sabharwal: *nonlinearity OR input-dependent transition*), independent of
  the linear-eigenvalue lever.

**Anchors (causally established; do not re-derive):**
- **A single squashed/corrective state sits in ONE regime.** Hyperparameter/CMA search moves
  *within* a regime, not across it (`S5_MECHANISM_SYNTHESIS.md`; `S5_CONFIG_FLIP.md`).
- **The negative/reflection along-key eigenvalue is causally necessary AND sufficient for S5.**
  Two models × five configs, perfect separation: every config that can reach a negative along-key
  eigenvalue solves S5, every one that cannot fails (`EIGENVALUE_CAUSAL_TEST.md`,
  `GDN_VS_E88_TRANSITION.md`). e88-linear `decay·I − k̂k̂ᵀ` (eig `decay−1`, 100% negative, min −1.0)
  solves S5 1.0; GDN `g(I − βk̂k̂ᵀ)` (eig `g(1−β)`, 100% positive) fails (0.54); `allow_neg_eigval`
  flips GDN to 1.0.
- **Counting needs an additive non-saturating cell; the LSTM (two compartments) is the only arm
  strong at BOTH** (`PROBE1_COUNTING_RESULTS.md`, `E88_NONSAT_RESULTS.md`): on unbounded `aⁿbⁿcⁿ`
  the LSTM holds 0.95 @ T=1024 while every single-state delta/linear/tanh arm falls to ~0.81–0.84;
  swapping E88's `tanh→relu` (single state) *slides along* the trade-off (counting 0.81→0.89) but
  *demotes* S5 (0.60→0.34) — it never occupies both ends.

**The single-state trade-off, in one line:** *signed/affine* → S5 not counting; *non-saturating
rectifying* → counting not S5; *saturating tanh* → worst of both; *escape requires a second,
additive compartment alongside the signed pathway (LSTM).*

---

## 1. HEADLINE ANSWER — is E97 enough? Is GDN-2 enough?

### 1.1 E97 — **NO, not "enough."** Tracks S5 (≥ E88); **cannot count.** Single corrective+squashed state.

**Where E97 lives in the repo (two real sources, not guessed):**
- **Runnable:** E97 is the `use_split_edit=True` configuration of `E88FLAHybrid`
  (`ndm/models/e88_fla_hybrid.py:856` declares the flag "E97: separate key-axis erase/read and
  value-axis write gates"; gates built at `:1042-1047`; computed as **sigmoid** at `:1482-1486`;
  applied in the recurrence at `:1784-1816`). There is **no `e97_*.py` model file** (the E-series
  files stop at `e94.py`) — E97 *is* E88-with-split-gates.
- **Formal:** the same update is defined in Lean,
  `formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean:98-128`:
  `e97UpdateExpanded = tanh((λI − k(b⊙k)ᵀ)·H + k(w⊙v)ᵀ)`.

**Per-token state update (real code, `e88_fla_hybrid.py:1784-1816`):**
```
read_key    = k̂ ⊙ erase_gate          # erase_gate = sigmoid(·) ∈ (0,1)   (:1785, :1483)
write_value = v ⊙ value_write_gate     # value_write_gate = sigmoid(·) ∈ (0,1)  (:1786, :1484)
retrieved   = S @ read_key                                                   (:1796)
delta       = write_value − retrieved                                        (:1807)
S           = f( decay·S + delta ⊗ k̂ )   # f = state_activation, DEFAULT tanh (:1816)
```
Affine form (from the scan path the code itself builds, `:1959-1973`):
`A_t = decay·I − k̂ (b⊙k̂)ᵀ`,  `B_t = k̂ ⊗ (w⊙v)`.

**Memory shape:** single **corrective** (delta read-modify-write) + **squashed** (default `tanh`).
Single matrix state.

**Eigenvalue:** along-`k̂` eigenvalue `= decay − Σᵢ bᵢ k̂ᵢ²`. Since `b = sigmoid ∈ (0,1)` and
`Σ k̂ᵢ² = 1`, this lies in `(decay−1, decay)` — **negative reachable** (when the learned erase gate
drives `Σ bᵢk̂ᵢ² > decay`). So E97 **can reach the reflection** that tracks S5. Two nuances:
(i) the split gate also lets the **read direction `b⊙k̂` differ from the write direction `k̂`** — a
strictly *richer* transition than E88's `decay·I − k̂k̂ᵀ` (the Lean `e88_cannot_realize_splitWitness_transition`
proves E88's coupled transition cannot match a nonparallel split, `SplitGatedDelta.lean`); setting
`b→1` recovers E88 exactly. (ii) Unlike E88, the negative eig is **not free**: E88 gets `decay−1<0`
by construction, whereas E97 must *learn* a high erase gate + low decay (at init `decay≈0.98,
b≈0.5` gives a *positive* along-key eig). So E97 is **≥ E88 in S5 capacity** but its reflection is
gate-conditional.

**Prediction:** **S5 = YES** (capacity ≥ E88; the reflection lever is reachable). **Counting = NO**
— it is a **single corrective+squashed** matrix state; the split gates add transition richness but
**do not add an additive/accumulating compartment**. By the anchor, a single corrective state
*cannot* count. (You could set `state_activation='relu'` to make E97 non-saturating, but per
`E88_NONSAT_RESULTS.md` that buys counting at the cost of S5 — the same single-state trade-off; and
with `raw_write` you'd remove the delta-correction that gives S5. No single E97 config gets both.)

**Verdict:** E97 is a **better S5 model than E88** (richer, sign-reachable transition) but is **on
exactly the same side of the counting trade-off** — one squashed corrective compartment. **Not
"enough" for both.**

### 1.2 GDN-2 — **NO, not "enough."** S5 only with `allow_neg_eigval`; **cannot count.** Single corrective state.

**Where GDN-2 lives in the repo:**
- **Runnable kernel: NOT FOUND (external).** `ndm/models/external_gdn2.py` is a *wrapper* that
  loads NVIDIA's `GatedDeltaNet2` from a non-vendored checkout (`:22` default
  `/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py`; raises ImportError if absent, `:122-125`). The
  numeric per-token update is **not in this repo**. The wrapper *does* expose
  **`allow_neg_eigval`** (`:178`, passed through at `:207`).
- **Formal (in-repo equation):** Lean `SplitGatedDelta.lean:137-143`:
  `gdn2LinearCore D H k b w v = splitGatedLinearCore 1 (D·H) k b w v`
  `= (I − k(b⊙k)ᵀ)·(D·H) + k(w⊙v)ᵀ` — i.e. **E97's split-gated linear core applied to a
  pre-decayed state `D·H`, with NO state nonlinearity** (resource signature:
  `gdn2_split_erase_write_is_scan_compatible_linear_state`, temporal-nonlinearity = false,
  scanCompatible = true; `E97_GDN2_FORMALISM_FINDINGS.md`).

**Memory shape:** single **corrective** (split-gated delta) + **affine/linear** (no `tanh`),
scan-compatible. Single matrix state.

**Eigenvalue:** decay multiplies the **whole** operator (like GDN), so the along-key eigenvalue
keeps the sign of `(1 − Σ bᵢk̂ᵢ²) · decay > 0` → **pinned positive** by default → **S5 NO**. The
`allow_neg_eigval` flag (the same Grazzi-2025 `β×2` lever that flips GDN, `GDN_VS_E88_TRANSITION.md`
§4) makes the along-key eig reach negative → **S5 conditional-YES**.

**Prediction:** **S5 = conditional** (YES iff `allow_neg_eigval=True`; NO by default).
**Counting = NO** — single corrective delta state; the `(I − βk̂k̂ᵀ)` erase prevents unbounded
accumulation (same reason GDN cannot count).

**Verdict:** GDN-2 is the **linear/scan-compatible** sibling of E97 (E97 minus the temporal
nonlinearity, decay folded into the whole operator). It can be *flagged* into S5 but **still cannot
count** — one corrective compartment. **Not "enough" for both.**

### 1.3 Why neither is enough — the structural reason

Both E97 and GDN-2 are **single delta-rule (corrective) compartments**. The split erase/read +
write gates are a genuine improvement to the **transition operator** (richer, sign-reachable
reflections; the Lean witnesses prove they exceed E88's coupled transition), so they help **S5**.
But counting is a **memory-shape** property — it needs an **additive, non-saturating accumulator**,
which the split gates do **not** add. To get counting these models must squash less
(`relu`/`softplus`) or stop correcting (`raw_write`), and either move *loses* S5. **One compartment
⇒ one regime.** This is precisely the LSTM-escape result (`E88_NONSAT_RESULTS.md` §5): only a model
with a **second additive compartment alongside the signed pathway** occupies both ends. E97/GDN-2
do not have that second compartment, so the predicted answer is **"enough for S5, not for
counting."**

---

## 2. THE FULL ZOO TABLE

Memory shape × eigenvalue range × single/multi-state → predicted (counting?, S5?), with
`file:line`. **C** = counting, **S5** = group-tracking. "(refl)" = negative/reflection along-key
eigenvalue reachable; "(+)" = pinned positive. Predictions marked **†** were corrected against the
empirical E88-tanh anchor where an agent's eigenvalue read missed the `−k̂k̂ᵀ` reflection.

### 2.1 Headline architectures (hand-verified)

| arch | update (real code) | memory shape | along-key eig | state | C | S5 | cite |
|---|---|---|---|---|---|---|---|
| **E88** (tanh, default) | `tanh(decay·S + (v−S@k̂)⊗k̂)` | squashed+corrective | `decay−1 < 0` **(refl)** | single | no | **yes** | `e88_fla_hybrid.py:1807-1816,1959-1973` |
| **E88-linear** | `decay·S + (v−S@k̂)⊗k̂` | corrective/affine | `decay−1` **(refl)** | single | no | **yes** | `e88_fla_hybrid.py:1262-1263` (`identity`) |
| **E88-relu/softplus** | `relu/softplus(decay·S + …)` | additive non-sat + corrective | `decay−1` but rectified | single | **partial** | no† | `e88_fla_hybrid.py:1266-1269` |
| **E88 raw_write** | `f(decay·S + v⊗k̂)` | additive/squashed | `decay > 0` **(+)** | single | depends on `f` | no | `e88_fla_hybrid.py:1793-1794` |
| **E88 pos_eigval_clamp** | `f(decay·(S−r⊗k̂)+v⊗k̂)` | corrective | `decay·(1−1)=0` **(+)** | single | no | no | `e88_fla_hybrid.py:1797-1805,1968-1971` |
| **E97** (`use_split_edit`) | `tanh(decay·S + (w⊙v − S@(b⊙k̂))⊗k̂)` | squashed+corrective | `decay−Σbᵢk̂ᵢ²` **(refl, learned)** | single | **no** | **yes** | `e88_fla_hybrid.py:1784-1816`; `SplitGatedDelta.lean:98-128` |
| **GDN** (fla, default) | `g(I−βk̂k̂ᵀ)S + βk̂vᵀ` | corrective/linear | `g(1−β) ∈ (0,1)` **(+)** | single | no | **no** | `GDN_VS_E88_TRANSITION.md §1.2`; `fla_gated_delta.py:100-111` |
| **GDN +`allow_neg_eigval`** | same, `β∈(0,2)` | corrective/linear | `g(1−β)` **(refl)** | single | no | **yes** | `fla_gated_delta.py:60,108` |
| **GDN-2** | `(I−k(b⊙k̂)ᵀ)(D·H) + k(w⊙v)ᵀ` | corrective/linear | `(1−Σbᵢk̂ᵢ²)·decay` **(+)**, refl iff flag | single | **no** | **cond.** | `SplitGatedDelta.lean:137-143`; `external_gdn2.py:178,207` **(kernel external)** |
| **M²RNN / M²RNN-CMA** | `z=tanh(hW+kvᵀ); h=f·h+(1−f)z` | squashed+corrective (raw-write, **no `−kk̂ᵀ`**) | `≈ f + (1−f)·eig(W) > 0` **(+)** | single | no | **no** | `m2rnn_baseline.py:308-317` |
| **LSTM** (cuda/baseline/counter) | `c=f·c+i·tanh(g); h=o·tanh(c)` | **additive cell** + gated hidden | `f∈(0,1)` (+) but fully nonlinear | **multi** | **yes** | **yes** | `cuda_lstm.py:220-226`; `counter_baseline.py:75-76` |

### 2.2 E-series lineage (E63 → E94) and `mom_e88` (workflow-classified, real code)

| arch | memory shape | along-key eig | state | C | S5 | cite |
|---|---|---|---|---|---|---|
| e63_nonlinear_delta | squashed (blend/residual + tanh) | `α∈(0,1)` (+) | single | no | no | `e63_nonlinear_delta.py:158-397` |
| e63m_matrix_nonlinear | corrective (tanh in read only) | `α∈(0,1)` (+) | single | no | no | `e63m_matrix_nonlinear.py:208-214` |
| e64_additive_h | corrective (convex blend to tanh) | `α∈(0,1)` (+) | single | no | no | `e64_additive_h.py:137-149` |
| e65_diagonal_h | corrective+squashed | `α∈(0,1)` (+) | single | no | no | `e65_diagonal_h.py:147-152` |
| e66_lowrank_h | corrective+squashed | `α∈(0,1)` (+) | single | no | no | `e66_lowrank_h.py:154-162` |
| e67_h_gated | corrective (state-dep gate) | `α∈(0,1)` (+) | single | no | no | `e67_h_gated.py:158-163` |
| e68_self_gating | squashed+corrective | `α∈(0,1)` (+) | single | no | no | `e68_self_gating.py:145-159` |
| e70_matrix_linear | **additive** (`S=decay·S+v⊗k`, no tanh) | `decay>0` (+) | single | **yes** | no | `e70_matrix_linear.py:196-199` |
| e71_delta | corrective (Widrow-Hoff delta) | `1−β ∈ (0,1)` (+) | single | no | no | `e71_delta.py:107-110` |
| e71_matrix_gated | corrective (convex blend) | `α∈(0,1)` (+) | single | no | no† | `e71_matrix_gated.py:258-272` |
| e72_matrix_selfgate | corrective | `α∈(0,1)` (+) | single | no | no | `e72_matrix_selfgate.py:189` |
| e73_matrix_nonlinear | squashed (`tanh(S·z+v⊗k)`) | no `−kk̂ᵀ` (+) | single | no | no | `e73_matrix_nonlinear.py:199` |
| **e74_fixed_decay** | corrective (delta, **no tanh**) | **`α−1 < 0` (refl)** | single | no | **yes** | `e74_fixed_decay.py:169-172` |
| e74_v2 | squashed+corrective (`tanh` default) | `≈0` / (+) | single | no | no | `e74_v2.py:428-456` |
| e75_gated_delta | squashed+corrective | **`β−1 < 0` (refl)**† | single | no | **yes†** | `e75_gated_delta.py:188` |
| e75_multihead | squashed+corrective | **`β−1 < 0` (refl)** | single | no | **yes†** | `e75_multihead.py:302` |
| e75_vector_gate | corrective (per-row decay) | `g∈(0,1)` (+) | single | no | no | `e75_vector_gate.py:175-177` |
| e76_logspace_delta | corrective (+`tanh` default) | `decay∈(0,1)` (+) | single | no | no | `e76_logspace_delta.py:245-250` |
| **e77_linear_matrix** | corrective/affine (**no tanh**) | **`decay−1 < 0` (refl)** | single | no | **yes** | `e77_linear_matrix.py:166-180` |
| **e78_projected_matrix** | corrective (**no tanh**, projected) | **`decay−1 < 0` (refl)** | single | no | **yes** | `e78_projected_matrix.py:138-162` |
| e79_coupled_matrix | mixed (additive+gated decay) | `sig·sig ∈ (0,1)` (+) | **multi (S,M)** | partial | no | `e79_coupled_matrix.py:319-343` |
| e80_full_rank_gate | mixed (additive+gated) | `sig∈(0,1)` (+) | **multi (S,M)** | partial | no | `e80_full_rank_gate.py:181-204` |
| e81_gate_as_state | corrective (mutual sig gates) | `sig∈(0,1)` (+) | **multi (S,G)** | no | no | `e81_gate_as_state.py:201-223` |
| e82_self_gate | corrective (sig gate) | `sig∈(0,1)` (+) | single | no | no | `e82_self_gate.py:166-178` |
| e83_circular_tower | corrective (row/col sig decay) | `sig·sig ∈ (0,1)` (+) | single (×3 slots) | no | no | `e83_circular_tower.py:385-386` |
| e84_neural_ode | corrective (decay-to-zero, RK4) | `∈[0.75,1)` (+) | **multi (S,G)** | no | no | `e84_neural_ode.py:96-167,279` |
| e85_input_as_matrix | squashed (additive then L2-norm) | normalized (+) | single | no | no | `e85_input_as_matrix.py:169-175` |
| e86_input_matrix_delta | squashed+corrective | **`β−1 < 0` (refl)** | single | no | **yes†** | `e86_input_matrix_delta.py:215-216` |
| e87_sparse_block | squashed+corrective (top-k blocks) | `β∈(0,1)` (+) | single/block | no | no | `e87_sparse_block.py:258-260,412-415` |
| e89_residual_state | mixed (`S += tanh(decay·S+…)`) | `decay∈(0,1)` (+) | single | partial | no | `e89_residual_state.py:313-318` |
| e90_dual_rate | squashed+corrective (fast+slow) | **`decay−1 < 0` (refl)** | **multi (fast,slow)** | no | **yes** | `e90_dual_rate.py:1170-1189` |
| e91_matmat | squashed+corrective (rank-r) | **`α−1 < 0` (refl)** | single | no | **yes†** | `e91_matmat.py:225-228` |
| e92_matmat | squashed+corrective (`W_h·S`) | `α·eig(W_h) ∈ (0,1)` (+) | single | no | no | `e92_matmat.py:169` |
| e93_minimal | squashed+corrective | `α·eig(W_h) > 0` (+) | single | no | no | `e93_minimal.py:178-184` |
| e94 | squashed+corrective (`W_h_time·S`) | `eig(W_h)` (sign free, but squashed) | single | no | no | `e94.py:381-388` |
| mom_e88 | squashed+corrective (MoE slots) | `decay∈(0,1)` (+) | single/slot | no | no | `mom_e88.py:547-550` |
| **e45_pure_accumulation** | **additive** (`h=x+h` / `x+α·h`) | `1.0` / `α∈(0,1)` (+) | single | **yes** | no | `e45_pure_accumulation.py:163-166` |
| gated_delta_net (local) | corrective (`α(I−βk̂k̂ᵀ)`) | `α(1−β)∈(0,1)` (+) | single | no | no | `gated_delta_net.py:138` |

### 2.3 Mamba / SSM / Elman / baselines (workflow-classified)

| arch | memory shape | along-key eig | state | C | S5 | cite |
|---|---|---|---|---|---|---|
| mamba2_baseline | **NOT FOUND (external `mamba_ssm`)** — linear-state (scalar·I) per docs | `∈(0,1)` (+) | n/a | no | no | wrapper `mamba2_baseline.py:59`; `docs/related_work_nonlinear_rnns.md` |
| mamba3_baseline | **additive** (selective scan) — **kernel external** | `α=exp(A·dt)∈(0,1)` (+) | **multi** | **yes** | no | `/home/erikg/mamba3/.../mamba3_step_fn.py:785-795` **(external path)** |
| mamba2_informed_elman | **additive** (`H=decay·H+x⊗B`) | `decay∈(0,1)` (+) | single | partial | no | `mamba2_informed_elman.py:90-91` |
| mamba_gated_elman | squashed (Elman `tanh`) | `eig(W_h)<1` (+) | single | no | no | `mamba_gated_elman.py:188-190` |
| gru_baseline | corrective+squashed (convex blend) | `z∈(0,1)` (+) | single | no | partial‡ | `gru_baseline.py:132` |
| cuda_gru | corrective+squashed | `1−z∈(0,1)` (+) | single | no | no | `cuda_gru.py:101-150` |
| min_rnn_baseline (minGRU/minLSTM) | corrective (convex blend) | `1−z∈(0,1)` (+) | single | no | no | `min_rnn_baseline.py:72-75,119` |
| lstm_baseline | **additive cell** + gated hidden | `f∈(0,1)`, fully nonlinear | **multi** | **yes** | **yes** | `lstm_baseline.py:152` (nn.LSTM) |
| counter_baseline (LSTM + ReLU-RNN) | **additive** | `f∈(0,1)` / ReLU | multi / single | **yes** | yes / no | `counter_baseline.py:75-76` |
| dual_memory_elman | squashed (tanh core; tape = external mem) | n/a (+) | single + tape | no | no | `dual_memory_elman.py:75-81` |
| hybrid_elman | mixed (tanh core + additive diag memory) | `decay∈(0,1)` (+) | **multi** | partial | no | `hybrid_elman.py:170-176` |
| hybrid_ladder | **NOT FOUND** — composite wrapper, delegates to layers | n/a | n/a | — | — | `hybrid_ladder.py` (no recurrence) |
| stock_elman | squashed (vanilla Elman) | `eig(W_h)<1` (+) | single | no | no | `stock_elman.py:167-170` |
| diagonal_state_elman | squashed (diag Elman) | `A_t∈(0,1)` (+) | single | no | no | `diagonal_state_elman.py:151-154` |

‡ GRU: along-key eig pinned positive, but as a *fully nonlinear* RNN it can in principle reach S5
via the nonlinearity (Merrill's "OR"); marked partial pending a probe. Single-state, so no counting.

---

## 3. THE EXCEPTIONS — additive and/or two-compartment (the interesting ones)

These are the architectures that are **not** single squashed/corrective states; they are where the
trade-off *could* be escaped. Flagged explicitly per the task:

**(a) ADDITIVE single-state → the COUNTING corner (count YES, S5 NO):**
`e45_pure_accumulation`, `e70_matrix_linear`, `mamba2_informed_elman`, `mamba3_baseline`,
`mamba2_baseline` (linear-state SSM). These accumulate (no squash, no delta-correction) so they
count, but their decay is pinned positive → no reflection → no S5. Mirror image of E88.

**(b) REFLECTION-reachable corrective → the S5 corner (count NO, S5 YES):** `e74_fixed_decay`,
`e77_linear_matrix`, `e78_projected_matrix` (all *linear, no tanh*, `decay−1<0`), plus the **E88-tanh
family** whose `−k̂k̂ᵀ` gives a reflection under `tanh`: `e75_gated_delta`, `e75_multihead`,
`e86_input_matrix_delta`, `e90_dual_rate`, `e91_matmat`. (Several of these were marked S5=no by the
workflow on the grounds that `tanh` "eliminates" the negative eigenvalue — **corrected to yes** here:
the empirical anchor E88-tanh *does* track S5, 0.47–0.99, with exactly this structure.)

**(c) MULTI-COMPARTMENT (the candidate escapes):**
- **TRUE escape (count AND S5): `lstm_baseline`, `cuda_lstm`, `counter_baseline` LSTM.** An
  *additive, non-saturating cell* `c=f·c+i·g` (counts) **plus** a gated nonlinear hidden
  `h=o·tanh(c)` (full nonlinearity → S5). Empirically the only arm strong at both (S5 1.0, count
  0.95). **This is the architecture E97/GDN-2 are NOT.**
- **PARTIAL / NOT a true escape:** `e79_coupled_matrix`, `e80_full_rank_gate` (two coupled matrices,
  but both gated-positive → counting partial, S5 no); `e81_gate_as_state`, `e84_neural_ode` (two
  compartments, both *corrective* → neither regime); `e90_dual_rate` (fast+slow, but **both** are
  tanh-squashed+corrective → S5 yes via reflection, counting no — not an escape); `hybrid_elman`
  (tanh core + additive diagonal memory, but memory decay positive → counting partial, S5 no);
  `mamba3_baseline` (additive multi-compartment → counts, but positive eig → no S5).
  **Lesson:** two compartments help only if they are in *different* regimes (one additive
  non-saturating, one signed/nonlinear). Most multi-compartment zoo members put both compartments
  in the *same* regime and therefore do **not** escape — exactly the E97/GDN-2 situation, one
  regime.

---

## 4. PROBE-CONFIRMATION SHORTLIST

Which predictions are **already confirmed** by prior runs, and which are **worth confirming** to
validate the axis's predictive power.

**Already covered (no new run needed):**
- E88-tanh, E88-linear, E88-relu, E88-softplus on S5/S3 + counting (`E88_NONSAT_RESULTS.md`,
  `PROBE1_COUNTING_RESULTS.md`, `S5_SYMMETRIC_RESULTS.md`).
- GDN default (S5 fail) and **GDN +`allow_neg_eigval`** (S5 1.0) — the eigenvalue causal flip
  (`EIGENVALUE_CAUSAL_TEST.md`).
- E88 `raw_write` (S5 destroyed) and `pos_eigval_clamp` (S5 destroyed) (`EIGENVALUE_CAUSAL_TEST.md`).
- M²RNN / M²RNN-CMA (S5 fail, counting worst) (`S5_MECHANISM_SYNTHESIS.md`, `PROBE1`).
- LSTM (S5 1.0 AND counting 0.95) — the two-compartment escape (`E88_NONSAT_RESULTS.md`).

**Worth confirming (high-value, directly tests this classification):**
1. **E97 (`use_split_edit=True`) on S5 + counting — UNPROBED, top priority.** Prediction: **S5 ≥
   E88-linear** (richer, sign-reachable transition; should match or beat 0.60→), **counting ≈
   E88-tanh** (single corrective+squashed; should *not* reach the LSTM). The split gates' value
   would show as an S5 gain *without* a counting gain — the cleanest test of "transition richness
   helps S5, not counting." (One-flag run on the existing E88 harness; `train_hybrid.py` already
   wires the flag.)
2. **E97 + `state_activation='relu'` — UNPROBED.** Prediction: it slides toward counting and *loses*
   S5, like e88-relu — confirming the split gates do **not** add a counting compartment.
3. **GDN-2 (external) with vs. without `allow_neg_eigval` on S5 — UNPROBED.** Prediction: positive
   → S5 fail; flag on → S5 succeed; **counting fails either way**. Mirrors the GDN flip and tests
   the GDN-2 verdict. (Requires the external checkout — flag it as a dependency.)
4. **An additive single-state arm (e70_matrix_linear or mamba2_informed_elman) on `aⁿbⁿcⁿ` + S5.**
   Prediction: counts (≈ E88-relu or better), fails S5 — confirms the "counting corner."
5. **A reflection-linear arm (e77_linear_matrix) on S5.** Prediction: S5 like e88-linear — confirms
   that the negative eigenvalue, not the FLA-GDN packaging, is the lever.

---

## 5. NOT FOUND (honest — no runnable per-token update in this repo)

- **GDN-2 numeric kernel** — `external_gdn2.py` wraps a **non-vendored** NVIDIA checkout
  (`gdn2.py`, `:22,122-125`). Classified only from the **in-repo Lean equation**
  (`SplitGatedDelta.lean:137-143`) and the wrapper's `allow_neg_eigval` flag.
- **mamba2_baseline** — wrapper to external `mamba_ssm.Mamba2` (`:59`). Classified from
  `docs/related_work_nonlinear_rnns.md` (linear-state, scalar·I), not from in-repo recurrence.
- **mamba3_baseline** — update lives at an external path `/home/erikg/mamba3/...mamba3_step_fn.py`
  (not under this repo's tree). Classified from that source but flagged external.
- **fla GDN / fla GatedDeltaNet** — the kernel is in the external `fla` library; the per-token
  update is **derived with line citations in `GDN_VS_E88_TRANSITION.md §1.2`** (and `fused_recurrent.py`),
  which is what is used here.
- **hybrid_ladder** — composite container, **no recurrence of its own** (delegates to E88/GDN/M²RNN).
- **E97 is NOT not-found:** it exists in-repo twice (runnable `use_split_edit` path + Lean
  formalism), as documented in §1.1.

---

## 6. VALIDATION CHECKLIST

- [x] **E97 and GDN-2 state-updates located + classified from real code** (E97 = `use_split_edit`
  path of E88, `e88_fla_hybrid.py:1784-1816` + Lean `SplitGatedDelta.lean:98-128`; GDN-2 = Lean
  `:137-143` + `external_gdn2.py` flag, kernel flagged **external**). **No guessing.**
- [x] **Full zoo table**: memory shape + along-key eigenvalue range + single/multi-state → predicted
  counting & S5, per architecture, with `file:line` (§2; ~50 architectures).
- [x] **Explicit "is E97/GDN-2 enough?" answer with the exact code reason** (§1): **both track S5,
  neither counts** — single corrective compartment; split gates enrich the *transition* (S5) but add
  no *additive accumulator* (counting). Exceptions (additive / two-compartment) flagged (§3),
  including the LSTM as the only true escape and the partial multi-compartment cases.
- [x] **Probe-confirmation shortlist** (§4): already-covered runs listed; the unprobed **E97** and
  **GDN-2** S5/counting checks are the high-value tests of this axis. **NOT FOUND honesty** (§5).
- [x] **`paper/main.typ` untouched.** This document committed; **not pushed** (per task).
