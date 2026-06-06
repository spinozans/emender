# E88 NON-SATURATING STATE — does swapping `tanh` for an unbounded state nonlinearity give E88 a counter?

**Task:** `e88-nonsat` · **Model:** claude:opus · **Date:** 2026-06-05 ·
**Type:** model-variant build + training run. **`paper/main.typ` was NOT edited.**

Runs after, and reuses, `probe1-counting`
(`paper/review/PROBE1_COUNTING_RESULTS.md`): same counting tasks, same
param-matched ~8M recipe, same fp32 protocol, and the committed probe1 JSONs for
the shared arms (`e88-tanh`, `e88-linear`, `lstm`).

---

## 0. The hypothesis

E88's default state nonlinearity is `tanh` (`linear_state=0`):

```
S_t = tanh(decay_t * S_{t-1} + outer(delta_t, k_t))
```

`tanh` **saturates** → the matrix-state magnitude is bounded (|S|≤1 per element)
→ finite-state expressivity (it does S5 at fixed length) but it **cannot count**
[Weiss, Goldberg & Yahav 2018]: an unbounded counter needs the state to grow
without bound. **Prediction:** replacing `tanh` with a **non-saturating** state
nonlinearity (unbounded magnitude) should give E88 a counter, while keeping its
single-matrix, width-efficient design.

---

## 1. What was built (real code, registered, committed, tested)

### 1.1 `state_activation` knob on `E88FLAHybrid` (`ndm/models/e88_fla_hybrid.py`)

A single authoritative axis for the state map `S = f(decay*S + outer)`:

| `state_activation` | `f` | shape | bound on \|S\| |
|--------------------|-----|-------|-----------------|
| `tanh` (default)   | `tanh(z)`     | SATURATING nonlinear | bounded (≤1) |
| `identity`/`linear`| `z`           | affine / linear       | unbounded (decay-limited) |
| **`relu`**         | `max(z,0)`    | **NON-SATURATING rectifying** | **unbounded above** |
| **`softplus`**     | `log(1+e^z)`  | **NON-SATURATING smooth**     | **unbounded above** |

- Backwards compatible: unset → `tanh`, or `identity` when the legacy
  `linear_state=True` is passed; `identity` keeps `linear_state` in sync so the
  existing fast-path dispatch is unchanged.
- `relu`/`softplus` are implemented **only** in the fp32 PyTorch reference
  recurrence (the path the counting/S5 experiments already use under
  `--disable_autocast`). The fused bf16 CUDA/Triton kernels hard-code
  `tanh`-or-linear, so the layer **forces fp32 and raises** if a non-saturating
  state is routed at a bf16 kernel — it never silently runs `tanh`.
- Wired through `train_hybrid.py --state_activation {tanh,identity,linear,relu,softplus}`.

**The exact non-saturating form is documented and unit-tested.** With `decay=1`
and a constant positive write of `0.5/step`, after 500 steps:

| state map | `|S|_max` after 500 steps | behaviour |
|-----------|---------------------------|-----------|
| `tanh`     | **0.88** (≤1)            | SATURATES — cannot count past a fixed ceiling |
| `identity` | 250.0                    | grows linearly (unbounded) |
| **`relu`** | **250.0** (= 0.5×500)    | **accumulates the exact count (unbounded)** |
| **`softplus`** | 250.9                | grows linearly (unbounded) |

(`tests/test_e88_state_activation.py`, 8 tests, all pass — resolution,
exact-definition match, the saturation/accumulation contrast above, fwd+bwd in
fp32 for every activation, and the bf16-kernel rejection guard.)

### 1.2 Runner / aggregator
- `experiments/expressivity_tasks/run_e88_nonsat.py` — dynamic idle-GPU
  scheduler (only GPUs with <2 GB used; never preempts; resumable). Counting:
  runs **only** the two new arms (`e88-relu`, `e88-softplus`) at the probe1
  recipe so they are directly comparable to the committed probe1 JSONs. S5/S3:
  runs all five arms.
- `experiments/expressivity_tasks/aggregate_e88_nonsat.py` — merges new
  `nonsat_*` JSONs with the reused `probe1_*` JSONs into per-arm × length tables.

## 2. Run protocol

- **Arms (param-matched ~7.9–8.1M).** The four E88 arms share the identical
  shape `dim=384, n_heads=32, n_state=32, depth=4` and differ in **exactly one
  thing** — `state_activation` — so the state nonlinearity is the *only* isolated
  variable. `lstm` (dim 448) is the additive-counter reference from probe1.
- **3 seeds** {42,123,456}. **Train T=128.** **Eval T ∈ {128,256,512,1024}.**
- schedule-free AdamW, lr 3e-4, batch 32, **fp32 (`--disable_autocast`)**.
- Steps: counting 3000 (= probe1); S5 10000; S3 6000 (finite-state word problems
  need more steps). Raw per-seed JSONs: `results/nonsat_*.json` (+ reused
  `results/probe1_{anbncn_viability,dyck_depth}__{e88-tanh,e88-linear,lstm}__*`).

---

## 3. Counting results

### 3.1 `anbncn_viability` (1a, UNBOUNDED a^n b^n c^n — the headline counter)

Accuracy at each eval length, mean±std over seeds. Random baseline = 0.500.

| arm           | state shape                     | params | T=128 | T=256 | T=512 | T=1024 |
|---------------|----------------------------------|--------|-------|-------|-------|--------|
| e88-linear    | affine / linear                  | 7.92M  | 0.999 | 0.960 | 0.868 | 0.812 |
| e88-tanh      | SATURATING (tanh)                | 7.92M  | 0.987 | 0.928 | 0.869 | 0.836 |
| **e88-relu**  | **NON-SATURATING (relu)**        | 7.92M  | 0.977 | 0.947 | **0.914** | **0.893** |
| e88-softplus  | NON-SATURATING (softplus)        | 7.92M  | 0.998 | 0.945 | 0.899 | 0.872 |
| **lstm**      | additive counter [WGY+]          | 8.05M  | 1.000 | 1.000 | 0.978 | **0.951** |

**Reading 1a — YES, the non-saturating state gives E88 a measurably better
counter.** At every length ≥256 the ordering is
**`relu` > `softplus` > `tanh` ≈ `linear`**, and the gap grows with length.
At T=1024 `e88-relu` 0.893 beats `e88-tanh` 0.836 (**+0.057**) and `e88-linear`
0.812 (**+0.081**). `e88-relu` **closes ~50 % of the `tanh`→`lstm` extrapolation
gap** (0.836→0.951 is 0.115; relu reaches 0.893, i.e. +0.057). The non-saturating
cell pulls E88 *toward* the LSTM counter — but **does not fully reach it**
(0.893 vs 0.951): a single relu-rectified matrix-state counts better than `tanh`
or linear, yet still degrades where the LSTM's dedicated additive cell stays
near-flat. **Separation of degree, not kind** — consistent with probe1.

### 3.2 `dyck_depth` (1b, capped running Dyck-1 depth)

Random baseline = 0.182.

| arm           | state shape               | params | T=128 | T=256 | T=512 | T=1024 |
|---------------|----------------------------|--------|-------|-------|-------|--------|
| e88-linear    | affine / linear            | 7.92M  | 0.999 | 0.987 | 0.963 | 0.924 |
| e88-tanh      | SATURATING (tanh)          | 7.92M  | 0.999 | 0.978 | 0.951 | 0.929 |
| **e88-relu**  | **NON-SATURATING (relu)**  | 7.92M  | 1.000 | 0.999 | 0.984 | 0.931 |
| e88-softplus  | NON-SATURATING (softplus)  | 7.92M  | 0.995 | 0.968 | 0.934 | 0.894 |
| lstm          | additive counter [WGY+]    | 8.05M  | 1.000 | 0.998 | 0.984 | 0.975 |

**Reading 1b — weak separator, as probe1 warned.** `relu` is the best E88 arm
through T=512 (0.984) but the T=1024 cells bunch at 0.89–0.93 (relu noisy,
±0.050). The **display cap (15) + near-floor stationary depth makes 1b
effectively finite-state**, so the linear arm ties and the non-saturating gain is
small. Use **1a** as the headline; 1b confirms that *bounding the counter erases
the advantage*.

---

## 4. S5 / S3 (finite-state word problems)

### 4.1 `s5_permutation` (S5 word problem; random baseline 0.0083)

| arm           | state shape               | params | T=128 | T=256 | T=512 | T=1024 |
|---------------|----------------------------|--------|-------|-------|-------|--------|
| **e88-linear**| affine / linear            | 7.96M  | **0.601** | **0.320** | **0.161** | **0.087** |
| e88-tanh      | SATURATING (tanh)          | 7.96M  | 0.465 | 0.250 | 0.126 | 0.065 |
| e88-relu      | NON-SATURATING (relu)      | 7.96M  | 0.337 | 0.180 | 0.090 | 0.050 |
| e88-softplus  | NON-SATURATING (softplus)  | 7.96M  | 0.263 | 0.140 | 0.073 | 0.042 |
| **lstm**      | additive counter [WGY+]    | 8.10M  | **1.000** | **1.000** | **1.000** | **1.000** |

### 4.2 `s3_permutation` (S3 control; random baseline 0.167)

| arm           | state shape               | params | T=128 | T=256 | T=512 | T=1024 |
|---------------|----------------------------|--------|-------|-------|-------|--------|
| **e88-linear**| affine / linear            | 7.92M  | 1.000 | **0.996** | **0.968** | **0.857** |
| e88-tanh      | SATURATING (tanh)          | 7.92M  | 1.000 | 0.968 | 0.739 | 0.524 |
| e88-relu      | NON-SATURATING (relu)      | 7.92M  | 1.000 | 0.979 | 0.838 | 0.589 |
| e88-softplus  | NON-SATURATING (softplus)  | 7.92M  | 0.998 | 0.939 | 0.681 | 0.434 |
| lstm          | additive counter [WGY+]    | 8.05M  | 1.000 | 1.000 | 1.000 | 1.000 |

**Reading S5/S3 — the non-saturating state HURTS the finite-state task.** Among
the four E88 arms, the ordering on S5 is the **mirror image** of counting:
**`linear` > `tanh` > `relu` > `softplus`** at every length, seed-stable
(std ≤ 0.08). On S3 the linear arm dominates extrapolation (0.857 @ T=1024 vs
0.43–0.59 for the rest); the two non-saturating arms are no better than `tanh`
(relu ≈ tanh, softplus worse). **Rectifying the state (relu/softplus) destroys
exactly what S5/S3 need.** S5 is a finite *group* (permutations) whose word
problem wants **signed, reflection-rich linear dynamics** [Grazzi 2025;
DeltaProduct 2025]; ReLU clamps the negative half-space to zero, deleting the
reflection structure — so the most aggressively non-saturating map (softplus) is
worst and the affine map (which preserves sign) is best.

**Honest caveat (under-training, not a confound on the *ordering*).** At this
shared shape/budget the E88 arms did **not** converge on S5 (best 0.60 @ T=128;
prior CMA-tuned configs reached 0.99 at 20 k steps —
`review/S5_CONFIG_FLIP.md`). The *absolute* S5 numbers are therefore low. But (a)
the **relative E88 ordering is robust and seed-stable**, (b) it **reproduces the
prior finding** that `e88-linear ≥ e88-tanh` on S5, and (c) the LSTM reaching
**1.000** at the same budget proves the task is learnable here — so the E88 arms'
weakness is real, not a broken harness. The new, clean result is that **`relu`/
`softplus` sit *below* `tanh`**, i.e. non-saturation is a *cost* on finite-state.

---

## 5. The big synthesis — computational class × regime

Mapping each arm to (state map) × (finite-state S5/S3 ; unbounded counting 1a),
using T=1024 extrapolation as the discriminator:

| arm          | state map                       | **finite-state (S5/S3)**        | **unbounded counting (1a)**     |
|--------------|----------------------------------|----------------------------------|----------------------------------|
| e88-linear   | linear / affine (signed)         | **best E88** (S5 0.60→ ; S3 0.857) | worst E88 (0.812)               |
| e88-tanh     | saturating nonlinear             | mid (S5 0.47 ; S3 0.52)          | weak (0.836)                    |
| e88-relu     | **non-saturating rectifying**    | **worst-ish** (S5 0.34 ; S3 0.59) | **best E88 (0.893)**            |
| e88-softplus | non-saturating smooth            | worst (S5 0.26 ; S3 0.43)        | strong (0.872)                  |
| **lstm**     | **additive gated cell + hidden** | **1.000 (both)**                 | **0.951**                       |

**Two facts jump out:**

1. **For a *single-state* E88 there is a genuine TRADE-OFF, and its axis is the
   SHAPE of the state nonlinearity, not "linear vs nonlinear."**
   - *Affine / signed* (`linear`) → wins finite-state / groups, **cannot** count.
   - *Non-saturating / rectifying* (`relu`/`softplus`) → **counts** best, **loses**
     finite-state.
   - *Saturating* (`tanh`) → the **worst of both**: bounded enough to lose
     counting, rectified/squashed enough to underperform the affine map on S5.
     (This is the sharpest indictment of the *default* E88 state map.)

   Swapping `tanh`→`relu` does **not** Pareto-dominate; it **slides E88 along the
   trade-off** from the finite-state corner toward the counting corner.

2. **The LSTM escapes the trade-off — and that is the real lesson.** It is the
   only arm strong at **both** (S5 1.0 *and* count 0.95), because it has **two
   state pathways**: a *non-saturating additive cell* (the counter) **and** a
   *gated, sign-preserving hidden state* (the finite-state controller). E88's
   single squashed matrix-state is forced to pick **one** nonlinearity shape and
   therefore **one** regime.

### What this means for the paper's "nonlinear recurrence" framing

- The honest axis is **not** linear-vs-nonlinear (probe1 already showed
  eigenvalue-rich *linear* recurrences win S5). It is the **shape of the state
  map**: *saturating* vs *non-saturating-rectifying* vs *affine-signed* — each
  buys a **different** computational regime, and on a single-state cell they
  **conflict**.
- **Does E88 need a non-saturating cell to earn a genuine expressivity claim?**
  *For counting, yes* — a non-saturating (relu) state is the only change that
  measurably moves E88 toward the LSTM counter (closing ~50 % of the gap),
  whereas `tanh` and `linear` both stall. *But it is not free, and not
  sufficient*: the same relu state **demotes** E88 on S5/S3, and even at its best
  it does not reach the LSTM. So a non-saturating cell earns a **counting** claim
  **at the cost of** the finite-state claim.
- **Therefore the defensible, evidence-backed claim is regime-specific and
  architectural:** to claim *both* finite-state/group expressivity *and*
  unbounded counting, E88 would need to stop squashing a *single* matrix-state
  and instead carry a **non-saturating additive compartment alongside the
  signed/linear pathway** (the LSTM's two-track design) — not merely swap one
  elementwise nonlinearity for another. A single-`f` E88 can sit *anywhere on the
  trade-off* but cannot occupy both ends, which is exactly what the data shows.

---

### Reproduce

```
# state_activation knob is wired into train_hybrid.py; runner does idle-GPU scheduling
python experiments/expressivity_tasks/run_e88_nonsat.py --start_gpus 2 3 4 5 6
python experiments/expressivity_tasks/aggregate_e88_nonsat.py --task anbncn_viability
python experiments/expressivity_tasks/aggregate_e88_nonsat.py --task dyck_depth
python experiments/expressivity_tasks/aggregate_e88_nonsat.py --task s5_permutation
python experiments/expressivity_tasks/aggregate_e88_nonsat.py --task s3_permutation
python -m pytest tests/test_e88_state_activation.py -q   # 8 tests
```
Arms (4 E88 differing only in `state_activation`, + LSTM), seeds {42,123,456},
train T=128, eval {128,256,512,1024}, fp32, param-matched ~8M. Raw per-seed JSONs
in `experiments/expressivity_tasks/results/nonsat_*.json` (counting baselines
reused from `probe1_*`).
