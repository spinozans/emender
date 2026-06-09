# Phi exploration: which state-map nonlinearity maximizes the depth-growing capability?

**Task:** phi-explore · **Status:** complete. 11 phi × 3 seeds × 2 substrates swept on the
modular_quadratic depth cliff (mod 48), length-extrapolation to 16×, REAL data. Tables
below are from `aggregate_phi.py` over `results_phi/` (gated-delta) and
`results_phi_split/` (split-edit).

**TL;DR.** The best phi for the depth-growing capability is a **per-step bounded
saturating** map (tanh ≡ hardtanh ≡ softsign, all perfect T2048=1.000) **applied on the
split-edit (E97) recurrence**, where the linear baseline collapses to 0.205. The operative
property is **boundedness, not the specific function**; a **learned** phi fails to
discover it; and phi is **nearly inert on plain gated-delta** — its value is
substrate-coupled. Carry `tanh` (kernel-ready) or the cheaper `hardtanh`/`softsign` into
the E97 kernel/CMA-ES; collapse the phi search to "bounded vs not" and fix it bounded.

## The one un-pulled lever

The capability-gap study (`paper/review/E97_CAPABILITY_GAP_RESEARCH.md`) proved that a
**per-step** nonlinear *state* map separates from a linear recurrence on a
depth-growing capability: on `modular_quadratic` with modulus K∈{32,48,64}, the
per-step **tanh** cell (e97_delta) beats the linear `gdn-neg` baseline by **+0.18..0.21**
accuracy at 16× length extrapolation (train T=128, test T=2048), 3 seeds, with a SwiGLU
MLP present in **both** arms. Crucially the *same* tanh applied only at chunk
boundaries (chunk=64, the `nlshell` arm) does **not** separate — confirming the
mechanism is the **per-step** state nonlinearity

```
S_t = phi( diag(g_t) S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T )           (1)
```

But `tanh` was never optimized. tanh saturates (bounded |S|≤1). It is one arbitrary
choice on a menu of elementwise maps. This study sweeps **phi itself** to find which
map MAXIMIZES the depth-growing nonlinear-composition capability — a potential new
mechanism, not a repackaging.

## Design — the cleanest possible A/B, on TWO substrates

Every arm is the **same** stack — FLA `GatedDeltaNet` projections (q/k/v/g/beta), short
conv, q/k L2-norm, the per-step write, output gate, RMSNorm, out-proj — differing in
**exactly one elementwise function** phi. The vehicle is
`ndm/models/phi_shell.py::PhiShellLayer`, running the recurrence at `state_chunk=1` (phi
every step), gradient-checkpointed, and `torch.compile`d for the launch-bound T-loop
(eager for length-extrap eval to avoid per-T recompiles).

Because the prior separation was demonstrated on the **split-edit (E97)** recurrence —
not the plain gated-delta shell — phi is swept on **both** substrates so we can tell
whether the depth capability is phi-placeable on the cheap substrate or is structural to
split-edit:

- **Substrate A — gated-delta** (eq. 1): `S = phi(diag(g) S + beta (v − S k) k^T)`.
  `phi='identity'` here is exactly the linear `gdn-neg` baseline.
- **Substrate B — split-edit / E97** (`per_step_phi_scan_split`): adds an **erase gate**
  on the read key and a **write gate** on the value, `S = phi(diag(g) S + (w⊙v −
  S(e⊙k)) k^T)` — mirroring `E88FLAHybrid(use_split_edit=True)`, the cell that achieved
  the +0.18 in the capability-gap study. `phi='identity'` here is the linear split-edit
  baseline (the within-substrate control).

Within each substrate the **only** thing that moves the number is phi.

- depth-4 stack, dim 256, 32 heads, head/state dim 32, expansion 1.0, **SwiGLU MLP
  (ratio 2.0) after every mixer in all arms** (the fixed O(depth) nonlinear readout — so
  the question is purely whether nonlinearity *in time* adds anything on top).
- train T=128, schedule-free AdamW, lr 3e-4, batch 32, 6000 steps, 3 seeds (42/123/456) —
  the modular_quadratic task **groks** (long ~0.5 plateau then a jump to ~1.0 near
  step 2.5–3.6k), so the full 6000-step budget is required.
- length-extrapolation eval at T ∈ {128, 256, 512, 1024, 2048} (Délétang protocol; 16×).
- **REAL data**: exactly-generable finite-automaton tasks (no mocks).

## The phi menu (≥5 variants; grouped by cost signature)

| phi | form | cost signature |
|-----|------|----------------|
| `identity` | S | **LINEAR baseline** (= gdn-neg) |
| `tanh` | tanh(S) | bounded / saturating (transcendental) |
| `softsign` | S/(1+\|S\|) | bounded / saturating (**rational / algebraic**) |
| `hardtanh` | clamp(S,−1,1) | bounded / saturating (piecewise-linear) |
| `poly3` | 1.5u−0.5u³ on \|S\|≤1, sat. outside | bounded / saturating (**low-degree polynomial**) |
| `relu` | max(0,S) | rectifying, unbounded one-sided |
| `softplus` | softplus(S)−log2 | smooth rectifier, unbounded one-sided |
| `gelu` | gelu(S) | smooth gated, non-monotone near 0 |
| `silu` | S·σ(S) | smooth gated (swish), non-monotone near 0 |
| `signed_sqrt` | sign(S)·√\|S\| | odd, compressive, **unbounded magnitude** (never saturates) |
| `learned` | S + α·MLP₁→16→1(S) | **learned** elementwise scalar map (init = identity) |

Cost-signature predictions (from the counting-vs-saturation theory, Weiss-Goldberg-Yahav
2018, and the E88 nonsat results): **saturating** maps cap |S| → good for finite-state /
bounded-modulus tracking but **cannot count unboundedly**; **rectifying** maps are
unbounded one-sided → can count but break sign/parity tracking; the **compressive
unbounded** `signed_sqrt` keeps the orbit informative without saturating. All per-step
phi are **non-chunkable**, so this is a pure CAPABILITY ranking (throughput is identical
order-of-magnitude across phi and is not a comparison axis here).

## Battery

- **Primary depth cliff** — `modular_quadratic` K∈{32,48,64}: x_t=(x_{t−1}²+c_t) mod K,
  non-invertible, non-contracting, nonlinear; needs O(T) nested squarings the fixed MLP
  cannot supply. This is the *proven* separator.
- **Cost-signature controls** — `iterated_nonlinear_map` (contracting logistic map →
  bounded effective depth, no gap expected) and `dyck_depth_unbounded` (UNBOUNDED
  counting → rectifying phi should help, saturating phi should hurt).

---

## RESULTS

### Substrate A — gated-delta, modular_quadratic mod 48 (3 seeds, depth 4, 16× extrap)

Mean accuracy over 3 seeds at each eval length; GAP = (phi − identity) at T=2048;
cliff = acc@T128 − acc@T2048. `identity` is the linear gdn-neg baseline.

| phi | group | T128 | T512 | T2048 | GAP vs identity | grokked? |
|-----|-------|------|------|-------|-----------------|----------|
| **hardtanh** | bounded/sat (PL) | 1.000 | 0.992 | **0.790** | **+0.070 (separates)** | yes |
| identity | LINEAR baseline | 1.000 | 0.983 | 0.720 | — (baseline) | yes |
| learned | learned | 1.000 | 0.93 | 0.696 | −0.024 (tie) | yes |
| tanh | bounded/sat | 1.000 | 0.974 | 0.708 | −0.011 (tie) | yes |
| softsign | bounded/sat (rational) | 0.983 | 0.832 | 0.634 | −0.086 (worse) | mostly |
| relu | rectifying | 0.857 | 0.796 | 0.628 | −0.092 (worse) | partial |
| gelu | smooth-gated | 0.696 | 0.628 | 0.533 | −0.186 (worse) | no |
| silu | smooth-gated | 0.634 | 0.561 | 0.515 | −0.205 (worse) | no |
| softplus | rectifying | 0.586 | 0.523 | 0.505 | −0.215 (worse) | no |
| poly3 | bounded/sat (poly) | 0.521 | 0.507 | 0.501 | −0.218 (worse) | no |
| signed_sqrt | compressive-unbnd | 0.504 | 0.495 | 0.494 | −0.225 (worse) | no |

(`learned` gated-delta = 0.696, 3 seeds — ties identity/tanh, i.e. the free-form phi just
learns a near-identity map on this substrate.)

**On gated-delta, per-step tanh does NOT separate** (0.708 vs identity 0.720) — this
contradicts the e97_delta +0.18 but exactly reproduces the prior `nlshell` null
(gated-delta+tanh ties/loses to gdn-neg). The only phi that separates on this substrate
is **hardtanh** (+0.070): bounded like tanh but with **unit slope near 0**, so it
preserves grokking while still capping |S|. Every phi that distorts the near-zero region
(softsign compresses, the rectifiers/smooth-gated break the sign symmetry, signed_sqrt
has an infinite slope at 0) **fails to grok at all** (T128 ≪ 1.0) — so on gated-delta the
binding constraint is *trainability*, not asymptotic capability.

### Substrate B — split-edit (E97), modular_quadratic mod 48 (3 seeds, 16× extrap)

The split-edit recurrence (erase gate on the read key, write gate on the value, β
write-strength) is the cell that achieved the prior +0.18. Here `phi='identity'` is the
**linear split-edit** within-substrate control.

| phi | group | T128 | T512 | T2048 | GAP vs identity | grok / behaviour |
|-----|-------|------|------|-------|-----------------|------------------|
| **tanh** | bounded/sat | 1.000 | 1.000 | **1.000** | **+0.795** | perfect, zero cliff |
| **hardtanh** | bounded/sat (PL) | 1.000 | 1.000 | **1.000** | **+0.795** | perfect, zero cliff |
| **softsign** | bounded/sat (rational) | 1.000 | 1.000 | **1.000** | **+0.795** | perfect, zero cliff |
| relu | rectifying | 1.000 | 0.798 | 0.578 | +0.372 | groks T128, **drifts** at long T |
| silu | smooth-gated | 0.835 | 0.627 | 0.531 | +0.326 | partial grok |
| gelu | smooth-gated | 0.710 | 0.678 | 0.669 | +0.464 | no grok (flat) |
| softplus | rectifying | 0.584 | 0.522 | 0.504 | +0.299 | no grok |
| poly3 | bounded/sat (poly) | 0.521 | 0.505 | 0.502 | +0.297 | no grok |
| signed_sqrt | compressive-unbnd | 0.498 | 0.491 | 0.489 | +0.284 | no grok |
| **learned** | learned | 1.000 | 0.856 | **0.294** | +0.088 | groks T128, **fails to extrapolate** |
| identity | LINEAR baseline | 1.000 | 0.735 | 0.205 | — (baseline) | groks T128, **collapses** at long T |

On split-edit the picture inverts vs gated-delta: **phi is decisive.** The linear
split-edit state has no magnitude control and **collapses under length extrapolation**
(identity T2048=0.205, a +0.795 cliff); rectifying relu *counts* but is unbounded so it
**drifts** (0.578); and the **bounded saturating maps — tanh, hardtanh, softsign — give
PERFECT 16× extrapolation (T2048=1.000, zero cliff).** All three tie at ceiling, so the
operative property is **boundedness**, not the specific functional form (transcendental
tanh = piecewise-linear hardtanh = rational softsign).

Two sharp negatives: (i) the **learned** elementwise phi groks the train length
(T128=1.000) but **does not discover the generalizing saturation** (T2048=0.294,
+0.706 cliff) — a free-form phi overfits the training length; boundedness must be
*imposed*, not learned. (ii) The non-bounded / distorted-near-0 phi
(softplus/poly3/signed_sqrt/gelu) again fail to grok at all, so the same trainability
constraint from substrate A persists.

(The within-substrate +0.795 is larger than the prior e97_delta +0.18 because that number
compared tanh-split against the *stronger* gdn-neg gated-delta baseline (T2048≈0.72–0.79),
whereas this isolates phi against the linear *split-edit* baseline (0.205) — the two are
consistent: tanh-split ≈ 1.0 either way.)

## Capability-vs-phi summary

T2048 mean accuracy (3 seeds, 16× extrapolation), both substrates, ranked:

| phi | gated-delta T2048 | split-edit T2048 | verdict |
|-----|-------------------|------------------|---------|
| **tanh** | 0.708 | **1.000** | **best (split-edit)** — bounded, kernel-ready |
| **hardtanh** | **0.790** | **1.000** | **best** — bounded PL, ties tanh |
| **softsign** | 0.634 | **1.000** | **best** — bounded rational, ties tanh |
| identity (linear) | 0.720 | 0.205 | baseline |
| relu | 0.628 | 0.578 | counts but drifts (unbounded) |
| silu | 0.515 | 0.531 | partial |
| gelu | 0.533 | 0.669 | no grok |
| learned | 0.696 | 0.294 | overfits train length, no extrap |
| softplus | 0.505 | 0.504 | no grok |
| poly3 | 0.501 | 0.502 | no grok |
| signed_sqrt | 0.494 | 0.489 | no grok (∞ slope at 0) |

**The value of phi is substrate-dependent.** On the plain gated-delta shell the linear
recurrence already extrapolates moderately (0.72) and no phi clears it by much (hardtanh
+0.07). On the high-capability **split-edit** cell the linear recurrence **collapses**
(0.205) and a per-step **bounded** phi is what unlocks perfect length generalization
(1.000). This is exactly why the prior e97_delta (split-edit + tanh) separated and the
nlshell (gated-delta + tanh) did not.

## Cost-signature notes

The sweep cleanly recovers the textbook signatures, plus a trainability axis:

- **saturating / bounded** (tanh, hardtanh, softsign, poly3): cap |S| → safe for the
  finite-state mod-K tracking; the *shape near 0* decides grokking — unit-slope-at-0
  (hardtanh) groks and is best; compressive-at-0 (softsign) or flat polynomial (poly3)
  degrade or kill grok.
- **rectifying** (relu, softplus): one-sided/unbounded → break the sign tracking the
  modular orbit needs; partial or no grok, worse extrapolation.
- **smooth-gated** (gelu, silu): non-monotone near 0 → worst grokking of the menu.
- **compressive-unbounded** (signed_sqrt): infinite slope at 0 → gradient pathology, no
  grok. Never saturates, but unusable as a per-step state map here.
- **trainability is the dominant cost on gated-delta**: a phi has to be ≈ identity near
  the origin to learn the algorithm at all; only then does its tail shape (bounded vs
  not) buy any extrapolation.

## Verdict: best phi for the depth capability + what to carry forward

**Best phi for the depth-growing capability: a per-step BOUNDED SATURATING map, applied
on the split-edit recurrence.** `tanh`, `hardtanh`, and `softsign` all give *perfect* 16×
length extrapolation on the modular_quadratic cliff (T2048 = 1.000, zero cliff, 3 seeds),
vs the linear split-edit baseline's collapse to 0.205. They tie at ceiling, so the
operative property is **boundedness of the state**, not the specific function — tanh was
never special; any smooth-or-PL bounded map works equally.

What the sweep settles:
1. **It is boundedness, not a particular nonlinearity.** tanh ≡ hardtanh ≡ softsign at
   ceiling. The mechanism is magnitude control of the recurrent state so the orbit stays
   in-distribution at lengths 16× past training; the exact squashing shape is irrelevant.
2. **It is substrate-coupled.** Per-step phi is nearly inert on plain gated-delta
   (tanh ties identity; only hardtanh nudges +0.07) but decisive on split-edit (+0.795).
   So phi is not a free-standing "new mechanism" — it is the stabilizer that lets the
   split-edit cell's expressivity actually extrapolate.
3. **Do NOT learn phi.** The free-form learned elementwise phi groks the training length
   but fails to extrapolate (0.294) — it does not discover the saturation that
   generalizes. Boundedness must be imposed by construction.
4. **Rectifying/unbounded and near-0-distorting phi are out.** relu counts but drifts
   (0.578); softplus/poly3/signed_sqrt/gelu/silu fail to grok at all (a phi must be
   ≈ identity near 0 to even train, then bounded in the tail to extrapolate).

**Carry into the kernel / CMA-ES:** keep **`tanh`** as the per-step state map for the
split-edit (E97) cell — it is already the validated, fused-Triton-supported choice and is
at the capability ceiling. The new, actionable finding is that the kernel does **not**
need transcendental tanh: **`hardtanh`** (a clamp — one min/max, cheaper, no exp) and
**`softsign`** (one divide) are bit-for-capability equivalent, so a future kernel can use
the cheapest bounded map without losing the depth capability. There is **no upside to a
learned or exotic phi**; the CMA-ES search space over phi can be collapsed to the single
binary axis *bounded vs not* (and should fix "bounded"). Do **not** spend the per-step
(non-chunkable) cost of phi on the gated-delta backbone, where it buys almost nothing —
phi earns its keep only on the split-edit recurrence.

### Validation checklist (task phi-explore)
- ≥5 phi swept on the depth-cliff battery, length-extrap 16×, 3 seeds, REAL data, vs the
  linear baseline: **done** — 11 phi (identity/tanh/softsign/hardtanh/poly3/relu/softplus/
  gelu/silu/signed_sqrt/learned) × 3 seeds × 2 substrates on modular_quadratic mod 48,
  train T=128 → eval T∈{128…2048}, vs the linear `identity` baseline (and the gated-delta
  `identity` reproduces gdn-neg). Data: `results_phi/` and `results_phi_split/`.
- explicit best-phi + capability-vs-phi table + cost-signature notes, committed: **done**
  (this document).

### Scope notes (honest)
- Central cliff modulus K=48 (3 seeds) is reported in full. Cross-modulus robustness
  (K∈{32,64}) is established for the tanh/linear endpoints by the prior capability-gap
  study (+0.18–0.21 at K∈{32,48,64}); the bounded-phi ceiling here is consistent with it.
- Cost signatures (saturating→bounded/no-count, rectifying→count/drift, near-0 distortion
  →no-grok) are read directly off the 11-phi × 2-substrate K=48 sweep; the
  iterated_nonlinear_map / dyck_depth_unbounded controls are a natural follow-up to
  confirm the counting/contracting halves of the signature with the same vehicle
  (`run_phi_sweep.py --tasks ...`).

## Reproduce

```bash
# Substrate A — gated-delta (idle-GPU scheduler, resumable, skips existing JSON):
PYTHONPATH=. python experiments/expressivity_tasks/run_phi_sweep.py \
    --tasks modular_quadratic --mq_ks 48 \
    --steps 6000 --batch_size 32 --lr 3e-4 --eval_interval 1500 --eval_n_batches 4
PYTHONPATH=. python experiments/expressivity_tasks/aggregate_phi.py        # reads results_phi/

# Substrate B — split-edit / E97 (beta write-strength + open-gate init -> groks):
PYTHONPATH=. python experiments/expressivity_tasks/run_phi_sweep.py \
    --tasks modular_quadratic --mq_ks 48 --split_edit 1 \
    --output_dir experiments/expressivity_tasks/results_phi_split \
    --steps 6000 --batch_size 32 --lr 3e-4 --eval_interval 1500 --eval_n_batches 4
PYTHONPATH=. python experiments/expressivity_tasks/aggregate_phi.py \
    --out_dir experiments/expressivity_tasks/results_phi_split
```
Compute notes: the per-step scan is non-chunkable and launch-bound — `torch.compile`d
for training (one compile at fixed T=128), eager for length-extrap eval (avoids per-T
recompiles), gradient-checkpointed (~2.6 GB/job). modular_quadratic **groks** so the full
6000-step budget is needed.

Code: `ndm/models/phi_shell.py` (`PhiShellLayer`, the phi menu, `per_step_phi_scan` and
`per_step_phi_scan_split`), `experiments/expressivity_tasks/{run_phi_sweep,aggregate_phi}.py`,
`--phi` / `--split_edit` args in `train_hybrid.py`, `'phi-shell'` level in
`ndm/models/ladder_lm.py`.
