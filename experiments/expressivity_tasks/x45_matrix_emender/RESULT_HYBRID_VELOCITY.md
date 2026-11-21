# X45 K20 — THE HYBRID VELOCITY MATRIX: measured interleave-ratio costs before any hybrid registration

Result document for the K20 task (the hybrid program's first instrument).
**Zero training ran in this lane** — timing probes only, one card (lease 0,
RTX 6000 Ada, `receipts/k20_hybrid_velocity.json`). The registration-style
pre-state held: no directional prediction was made, and the measurement IS
the result. No hybrid engine was built (per the task: the projection table
is the deliverable; a hybrid build is a later, sign-off-gated lane).

**The question this lane exists for.** The board comparison is
matched-TIME (2700 active seconds). A hybrid of our slot layers + GDN2's
dense layers pays our scan's cost on every slot layer. The banked receipt
said our composed scan forward is **22.96 ms per (layer, window)** at the
registered shape (`receipts/k18_gates.json` gate D); unknown were GDN2's own
mixer cost at the same shape, the backward ratios, and the end-to-end
wall-clock of mixed stacks. **Measured, not estimated** — every number below
is a same-session CUDA-synced median with its spread, on the committed
engines imported verbatim and never modified.

**Headline, as narrowly as the evidence supports.**

* **Per-mixer cost ratio: 7.28×** (fwd+bwd, medians). Our composed mixer with
  the conv included costs **47.98 ms** per (layer, window-group) — forward
  **23.40 ms** (consistent with the banked 22.96 ms composed-scan receipt,
  which is the mixer's dominant term), backward **24.55 ms**. GDN2's
  committed FLA mixer costs **6.59 ms** (forward 2.35, backward 4.25) at its
  own committed 150M shape. The task's poles were ~10× (⇒ 3-of-24 ≈ 2.1×
  pure-GDN2 wall, board-fatal) and ~3× (⇒ ≈ 1.25×, viable); **the measured
  7.3× lands between them, nearer the middle**.
* **The honest step walls, both banked rows re-measured same-session at the
  probe step shape (64 windows, microbatch 8):** ours **10.62 s/step** (the
  banked row's own 10.74 s, −1.1%), GDN2 **3.12 s/step** (banked 3.25, −3.9%).
  Per-step optimizer: our machine's HybridMuonX45 **0.169 s**, GDN2's fused
  AdamW **0.015 s** — the optimizer is 1.6 % / 0.5 % of the two steps, a
  minor term in the projection, included anyway.
* **The projected hybrid wall-clock at 3-of-24 (mixer swap, the banked
  protocol's Muon): 4.27 s/step = 1.37× pure-GDN2 ⇒ the hybrid sees 73 % of
  GDN2's exposure in 2700 s.** Whole-layer swap: 4.34 s = 1.39×, 72 %.
  4-of-24: 1.47×/1.50× (68 %/66 %). 6-of-24: 1.69×/1.73× (59 %/58 %).
  8-of-24: 1.90×/1.96× (53 %/51 %) — at the edge of the board-fatal pole.
* **No bracket reaches the task's ~1.25× "viable" example, and no bracket is
  anywhere near the 2.1× "board-fatal" pole either.** The verdict scale is
  the task's own pre-state; on it, **3-of-24 is the only near-viable bracket**
  (concedes ~27 % of GDN2's matched-time exposure), 4-of-24 the outer edge,
  6- and 8-of-24 marginal-to-fatal at parity quality. The go/no-go on
  registering a ~1.4× time handicap is the operator's, and the receipts are
  what they should hear it from.

---

## 1. Method (the boundary, stated once)

Everything at the probe protocol's batch/window regime and the registered
150M shapes; both engines imported verbatim from the committed code
(`velocity_ladder._build_k10` + `convert_model_to_composition` +
`convert_model_to_short_conv` for ours; `matched_arms.build_gdn2` — the
registered `lattice_gdn2` nomination — for GDN2). Nothing was modified; the
receipt hash-binds every timed source (`bindings` in the receipt).

* **The unit** is one (layer, window-group): batch 8 × 2048 tokens at each
  machine's own committed 150M width — **ours 576, GDN2 640**
  (`part8_surface.gdn_config`'s own capacity solve; forcing a common width
  would have meant modifying a committed engine, which this lane refuses).
  Same batch, same window, same stream, same seed (3401), bf16 autocast
  compute with fp32 params — the training regime.
* **Real activations, not fixtures:** one real microbatch group of the
  flagship stream driven through each machine with hooks, capturing each
  measured layer's own mixer/MLP input; every timed layer runs on the
  activation it would actually see in training. Five layers per machine
  (0, 6, 12, 18, 23).
* **Warmup ≥ 20, medians not means, CUDA-synced, spread stated:** 25 warm-up
  iterations, then 25 measured iterations per direction, `torch.cuda.synchronize()`
  around every timed region. Forward is the real training form (autocast,
  graph built). Backward is `out.pow(2).sum().backward()` into **both** the
  parameters and the input (the input carries `requires_grad`, so the
  measured adjoint includes the `dx` the real model owes) — the same
  reduction for both machines.
* **The step walls** are each committed loop's own honest clock, re-run
  same-session: ours under the banked probe protocol's step body
  (`protocol_correction._train_corrected`'s `do_step`: HybridMuonX45 with the
  K18 arm's own optimizer split), GDN2 under `matched_arms.train`'s gdn2 body
  (fused AdamW at the registered nomination's knobs). 3 warm-up steps, then
  5 measured steps, median reported. The corrected protocol's EMA update
  sits **outside** that clock in the committed loop and is excluded here too.
* **One card, one session** (lease 0, `lambda01`, torch 2.9.1+cu128, triton
  3.5.1), `OMP_NUM_THREADS=4`, the approved venv Python.

## 2. Arm 1 — pure-machine per-(layer, window) costs

Medians over 25 CUDA-synced iterations after 25 warm-up, per layer, on that
layer's own captured activation (`per_layer_ours` / `per_layer_gdn2` in the
receipt; every layer's p25/p75/min/max/std is in the receipt).

| machine (fwd / bwd, ms per (layer, window-group)) | layer 0 | layer 6 | layer 12 | layer 18 | layer 23 | machine median |
| --- | --- | --- | --- | --- | --- | --- |
| **ours** — composed mixer, conv included | 23.17 / 24.18 | 23.40 / 24.51 | 23.37 / 24.62 | 23.47 / 24.55 | 23.46 / 24.57 | **23.40 / 24.55** |
| **GDN2** — committed FLA mixer | 2.59 / 4.55 | 2.74 / 4.97 | 2.34 / 4.25 | 2.35 / 4.24 | 2.34 / 4.24 | **2.35 / 4.25** |
| **ours** — committed SwiGLU MLP (hidden 3352) | 2.00 / 5.24 | 2.13 / 5.24 | 2.14 / 5.25 | 2.06 / 5.27 | 2.02 / 5.30 | **2.06 / 5.25** |
| **GDN2** — committed SwiGLU MLP (hidden 1648) | 1.28 / 3.09 | 1.29 / 3.07 | 1.30 / 3.09 | 1.30 / 3.09 | 1.30 / 3.11 | **1.30 / 3.09** |

* **Fwd+bwd medians:** ours mixer **47.98 ms** (cross-layer half-spread
  **0.48 %**), GDN2 mixer **6.59 ms** (half-spread 8.6 % — its layers 0
  and 6 run ~10–17 % above 12/18/23 on both directions; the receipt keeps
  all five rows). **Per-mixer ratio 7.28×.** MLPs: ours 7.32 ms, GDN2
  4.39 ms.
* **Cross-check against the banked receipt:** our mixer's forward (23.40 ms)
  is the banked composed-scan forward (22.96 ms, same shape, different
  session) plus the projections and the conv — **the scan remains ~98 % of
  the mixer's forward, consistent across sessions.** The conv is still
  ~0.7 % of the mixer (banked: 0.17 ms against the scan's 22.96 ms), i.e.
  the cost is the scan's, as the K18 receipt said.
* **GDN2's own backward path includes its committed activation
  checkpointing** (the FLA core's `torch.utils.checkpoint`; its backward
  recomputes forward states) — timed as-committed, never modified.

## 3. Arm 2 — the honest per-step wall-clock of the two pure machines

The probe protocol's step shape: 64 windows/step, microbatch 8 ⇒ 8
window-groups of (batch 8 × 2048) per step, 24 layers ⇒ **192 mixer calls
per step per machine**. 3 warm-up steps then 5 measured; medians, spreads in
the receipt (`steps_ours` / `steps_gdn2`).

| machine | step wall (median) | p25–p75 | banked row's own step | per-step optimizer | optimizer median |
| --- | --- | --- | --- | --- | --- |
| **ours** (K18 machine, HybridMuonX45 4×) | **10.62 s** | 10.60–10.68 | 10.74 s (`k18_runs/conv_seed3401`, step 180: 1,932.7 s / 180) | Muon split (matrices on Muon, gates+conv on AdamW) | **0.169 s** (1.6 %) |
| **GDN2** (registered nomination, fused AdamW) | **3.12 s** | 3.10–3.21 | 3.25 s (`velocity_runs/gdn2_probe_seed3401`, step 180: 584.4 s / 180) | fused AdamW | **0.015 s** (0.5 %) |

Same-session re-measurement lands within 1.1 % (ours) and 3.9 % (GDN2) of
the banked rows' own honest walls — **the probe protocol's time-exposure
asymmetry, receipted**: the probe is matched-TARGETS (20,022,866 bytes), so
the two banked rows trained the **same bytes in different wall-clocks**
(ours 1,932.7 s vs GDN2 584.4 s — a 3.31× asymmetry, almost exactly the
step-wall ratio). The 2700 s **board** flips the rule to matched-TIME, where
the faster machine sees more bytes and the hybrid's exposure fraction is
the step-cost ratio — that is the number the registration needs, below.

## 4. Arm 3 — the projected hybrid wall-clock table

**No hybrid was built or run.** The projection is computed from the
measured per-(layer, window-group) costs:

```
T_hybrid(s) = T_pure_gdn2 + s × 8 × Δper-layer  [+ optimizer term]
```

with s slot layers of 24, 8 window-groups per step, Δ the measured
per-(layer, window-group) swap delta, and the optimizer term the measured
Muon-vs-AdamW difference (+0.154 s) when the hybrid runs the banked
protocol's Muon — which is what updates both machines' params in the banked
protocol. Two composition assumptions, both from measured numbers:
**mixer swap** (our composed mixer inside an otherwise-GDN2 stack, GDN2's
MLPs everywhere — Δ = 47.98 − 6.59 = 41.39 ms) and **whole-layer swap**
(our mixer + our MLP on the slot layers — Δ = 41.39 + 2.93 = 44.32 ms).
The AdamW-bracket rows (hybrid keeps GDN2's registered AdamW) are in the
receipt; they differ from the Muon rows by only the 0.154 s optimizer term.

| ratio | mixer swap: step, wall vs pure-GDN2, exposure in 2700 s | layer swap: step, wall, exposure |
| --- | --- | --- |
| **3-of-24** (1:7-like) | 4.27 s, **1.37×**, **73.1 %** | 4.34 s, **1.39×**, **71.9 %** |
| **4-of-24** (1:5) | 4.60 s, **1.47×**, **67.8 %** | 4.69 s, **1.50×**, **66.5 %** |
| **6-of-24** (1:3) | 5.26 s, **1.69×**, **59.3 %** | 5.40 s, **1.73×**, **57.8 %** |
| **8-of-24** (1:2) | 5.92 s, **1.90×**, **52.7 %** | 6.11 s, **1.96×**, **51.0 %** |
| 24-of-24 (calibration) | 11.22 s (predicted) vs **10.62 s measured** | 11.78 s vs 10.62 s measured |

**The calibration row is the projection's own honesty check:** rebuilding
the pure-ours step from GDN2's measured step plus 24×8 measured swap deltas
over-predicts the measured pure-ours step by **5.6 %** (mixer swap) /
**10.9 %** (layer swap). The residual is per-call sync/launch overhead that
isolated probes pay per invocation but a real pipelined step partially
hides — so **the hybrid projections above are conservative in the same
direction** (the true hybrid steps should land slightly *below* the table,
its exposure slightly *above*). Stated, not hidden.

## 5. Arm 4 — the feasibility verdict per ratio

In the board registration's units (matched-TIME 2700 s, both machines
eating 64-window steps, so exposure ∝ 1/step-cost):

* **3-of-24:** the hybrid sees **73 % / 72 %** of GDN2's exposure in 2700 s
  (mixer / layer swap). Wall-clock **1.37×/1.39×** pure-GDN2.
* **4-of-24:** **68 % / 66 %**; wall **1.47×/1.50×**.
* **6-of-24:** **59 % / 58 %**; wall **1.69×/1.73×**.
* **8-of-24:** **53 % / 51 %**; wall **1.90×/1.96×** — bordering the
  task's 2.1× board-fatal pole.

On the task's own pre-stated scale (~1.25× viable, ~2.1× board-fatal at
parity quality): **no bracket reaches the 1.25× viable example — the
measured per-mixer ratio (7.3×) sits above the 3× pole that example assumed
— and no bracket approaches the 2.1× fatal pole.** 3-of-24 is the only
near-viable bracket, 4-of-24 its outer edge, 6- and 8-of-24
marginal-to-fatal.

## 6. The design recommendation (honest, from the receipts)

1. **A 3-of-24 hybrid is registrable only if the hybrid's quality case can
   absorb a ~27 % matched-time exposure handicap** (73 % of GDN2's bytes in
   2700 s, conservatively projected — the calibration says the true number
   is slightly better). If the operator's bar is the ~1.25× example
   strictly, **no ratio is viable under matched-time** and the honest
   registration statement is that one.
2. **4-of-24 is the outer edge** (a ~32–34 % handicap); anything at
   **6-of-24 or beyond concedes ≥ 40 % of the exposure and should not be
   registered** — at parity quality those brackets are board-fatal
   (1.7–2.0× wall).
3. The per-layer costs say **where any future hybrid lane should spend its
   kernel effort before registration**: our mixer's backward (24.6 ms) is
   now as expensive as its forward (23.4 ms), and both are the scan's — the
   composed scan is ~98 % of the mixer's forward and the mixer is ~7.3×
   GDN2's per layer. Cutting the per-mixer swap delta in half (a block-
   kernel lane's natural target) moves 3-of-24 from 1.37× to ≈ 1.21×
   (≈ 83 % exposure) — inside the task's viable band; no optimizer or MLP
   lever moves the number materially (the optimizer is 1.6 % of our step,
   the MLP swap delta 0.9 ms/layer).
4. **Disclosed design-space facts the registration will need:** the two
   committed 150M machines do **not** share a width (ours 576, GDN2 640);
   a real hybrid must pick one, a sign-off-gated decision this lane does
   not make (the scan's dominance makes the 7.3× ratio robust to that
   choice, but the absolute table would shift). The probe protocol's own
   asymmetry (§3) is the standing deconfound any matched-time comparison
   must state.

## 7. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k20_hybrid_velocity \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k20_hybrid_velocity.json
# receipt: receipts/k20_hybrid_velocity.json (session, git + engine hash
# bindings, every layer's p25/p75/min/max, per-step rows, the projection,
# the calibration, the verdict)
```

## 8. What this lane did NOT do

* **No hybrid engine was built, registered, or trained** — the projection
  table is the deliverable, per the task; any hybrid build is a later,
  sign-off-gated lane.
* **No engine was modified** — both mixers are the committed builds,
  hash-bound in the receipt; GDN2 was timed at its own committed width
  (640) rather than forced to ours, per the never-modify rule.
* **No board arm, no 2700 s run, no quality claim** — velocity only; the
  parity-quality assumption in the viability scale is the task's pre-state,
  not a measurement here.
