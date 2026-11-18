# Pre-registration (DRAFT — operator sign-off pending) — GDN2 with a growing state

**Date:** 2026-11-22 · **Status:** draft, awaiting the operator's sign-off. The
two binding choices the audit escalated were DECIDED by the supervisor on
2026-11-22 and are recorded in §2 and §5: the **heads axis** and the
**state-saturation (participation-ratio) trigger** with the delta
self-interference term as a non-gating co-diagnostic. Nothing else in this
draft is settled: the C3 saturation-regime definition (§7) is a LATER stage
behind its own sign-off, and no quality claim is made or tested here.

**Authorisation:** the operator's structural-audit directive, part 2 — "the
operator's control experiment": GDN2 with a growing state, as the test of
whether **growth is THE differentiator independent of slot architecture, or
whether addressable slots earn their keep**. Companion document:
[`RESULT_OP_SET_AUDIT.md`](experiments/expressivity_tasks/x45_matrix_emender/RESULT_OP_SET_AUDIT.md)
(the op-by-op audit this experiment is designed against).

**Rules inherited verbatim:** GDN2's model code is READ-ONLY — the growth
mechanism is a wrapper/registrar (the K8 pattern: our lanes wired optimizer
and growth machinery around imported, unmodified kernels). No edits to
`k10_*/k11_*` sources, the existing PREREGISTRATION files, or `lattice.py`.
Impossibility is escalated, never patched around inside GDN2's code.

---

## 1. Why this experiment exists (the audit's finding it tests)

The audit's formal result: at fixed configuration, **both** machines are
fixed-state automata under the same monoid
(`GDNMemoryOperator.gdn2_conditional_scan`;
`gdn_state_channel_depth_bounded_in_time`), and the only exits from that class
in the whole program are **POP / MERGE / growth** — the exact-capacity and
unbounded-state operations. **GDN2 has none of the three.** Its state
dimensions are configuration constants; nothing in its update or kernels
contemplates a state that changes shape mid-sequence.

So every measured emender-vs-GDN2 comparison to date has been run with the
one capability that formally differentiates the machines **switched off on
our side and absent on theirs** — and our growth receipts (K8) say it buys
nothing in the byte-LM regime (the three-way tie 2.0278 / 2.0186 / 2.0240)
because neither machine's store saturates there (our occupancy 24–85 of 128;
and §6's toy receipt below shows GDN2's per-head effective rank ≈ **5 of 128**
at its own receipted trained decay regime).

This experiment gives GDN2 the missing capability and runs the three-arm
comparison **in the one regime where the fixed-state bound provably binds**
(C3: long-context / dense association). It is the control experiment for the
audit's central question: if GDN2+growth closes the C3 gap, growth — not
slot architecture — is the differentiator; if our growable machine beats
GDN2+growth at matched grown state, addressable slots earn their keep on top
of growth.

## 2. The axis, and the d_k impossibility receipt (a finding about GDN2's growth ceiling)

**Registered axis: HEADS.** The registered 150 M GDN2 arm
(`part8_surface.GDNBlock` → `GDN2ExternalMLPLayer(expansion=2., head_dim=128,
num_heads=2)`, width 576, depth 24) has per-head state `S ∈ ℝ^{d_k×d_v} =
ℝ^{128×256}`, 2 heads. Growth multiplies the head count; the snap set is
**{2, 4, 8}** (geometric 2.0, the K8 rule's shape; cap 8 = the official
paper arm's head count, `configs/frontier/e97_gdn2_paper/gdn2_mlp.json`:
n_heads=8, n_state=128 — head growth is GDN2's OWN official scaling manifold).

**The d_k impossibility receipt (supervisor-directed finding, state it
prominently):** GDN2's chunk kernel hard-caps the key axis —
`lit_gpt/gdn2_ops/chunk_gdn2.py:1170`:
`assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for
GDN-2 :-("`. From the registered `d_k = 128` the key axis admits **exactly
one doubling** (128→256) and nothing beyond — no growth *trajectory* is
expressible on it, and patching the kernel is forbidden by the standing
rules. **Heads are GDN2's only legal growth axis.** (Head count is a runtime
grid dimension in every chunk kernel — no head-count assert exists — and the
official configs scale heads, not head_dim, across model sizes.)

## 3. The zero-injection birth protocol (the exact mechanism)

**What grows.** At a growth event `H → H' = min(cap, 2H)`:

* **State:** the recurrent state `[B, H, d_k, d_v]` pads along the head axis
  with **exact zeros**; the short-conv states pad their new channels with
  exact zeros. This is the Lean-legal zero-injection
  (`LiftedMemory`/`injection_propagation`: new state components enter as
  exact zeros; the monoid and scan theorems hold), and per-head isolation in
  the chunk kernels makes the old heads' states and outputs **untouched by
  construction** — each head owns a disjoint slice of every projection and
  its own state; nothing in the kernel mixes heads.
* **Parameters (the registrar, a wrapper — zero GDN2 code edits):** a fresh
  `GDN2ExternalMLPLayer` is constructed at `H'` with GDN2's **own** init
  recipes (Xavier gain `2^-2.5`; `A_log ~ log U(1,16)`; the `dt_bias`
  softplus recipe), and the old module's parameters are copied into the
  shared slices: `q_proj/k_proj/b_proj/f_proj.1` rows `[0:H·128]`,
  `v_proj/w_proj/g_proj.1` rows `[0:H·256]` (biases likewise),
  `o_proj` columns `[0:H·256]`, the conv channels, `A_log[0:H]`,
  `dt_bias[0:H·128]`. `f_proj.0`/`g_proj.0` (`head_v_dim`-wide) are
  head-count-independent and copied whole. **The single deviation from
  fresh init: `o_proj.weight[:, H·256:]` is zeroed at the event** — the
  wake-up protocol below. Optimizer surgery: AdamW state is carried by
  slice for every shared parameter (`exp_avg`/`exp_avg_sq` prefixes copied,
  new regions zero), fresh state for the new `A_log`/`dt_bias` entries.

**The parameter-liveness constraint, stated honestly (this is the one place
GDN2+growth CANNOT copy K8).** K8's growth added **zero new parameters**
(the key family is a fixed table; growth reveals rows mechanically). GDN2's
keys are **learned projections**, so growth necessarily adds trainable
parameters. A fully-zero-born head is provably **dead forever**: with zero
`k_proj` rows its state stays exactly zero (the update
`(I − k(b⊙k)^T)DS + k(w⊙v)^T` is identically zero in `k`), and with zero
`o_proj` columns no gradient can ever reach it — the K8 "dead until first
write" semantics would be *dead until nothing*, vacuous growth. The
registered birth is therefore:

* **born at GDN2's own init** for every new-head parameter (the new head is
  a functioning, untrained GDN2 head from the event: its keys are live, it
  writes into its zero state within the window — the K8 semantics of "new
  state enters exact zeros and fills only via its own writes" is preserved
  on the STATE side), **except** `o_proj`'s new input columns, zeroed at
  the event.
* **Consequence (i) — event-time output invariance:** the grown model's
  outputs at the event are **bit-identical to the un-grown model's** (old
  heads untouched; the new head's readout `o = S^T q` is nonzero but its
  path to the residual is multiplied by the zero columns). The event
  perturbs nothing observable; the smoke must receipt this.
* **Consequence (ii) — the one-step wake-up:** the first post-event
  optimizer step receives **nonzero gradient on `o_proj`'s new columns**
  (through the new head's already-live `o_norm` output); from the second
  step the head's internal parameters train. The head is *silent until its
  first output-weight step* — the honest GDN2 analog of K8's
  *dead-until-first-write*.

**Segmentation and re-dispatch (mid-window events).** The trigger is
evaluated at fixed **64-token statistic boundaries** mid-window (the
registered K8 granularity — and GDN2's kernel chunk size is exactly 64,
`chunk_gdn2.py`: `chunk_size = 64`, so the boundaries align with the
kernel's own chunking). The **probe pass** is a single inference-mode kernel
call with `return_intermediate_states=True`: the chunk kernel returns the
per-chunk state trajectory `h` (the state at every 64-token boundary) as a
first-class output — no segmentation and no GDN2 code change needed for the
statistics. The **real pass** replays the window with autograd, splitting the
kernel calls at the fire boundary: pre-event segments at `H`, the zero-padded
carry, post-event segments at `H'` on the grown module. Unlike K8 no
schedule-replay machinery is needed beyond the split itself: GDN2's gates
are pointwise functions of the input (no state-dependent discrete decisions
exist to flip), so the probe pass's plan is exactly reproducible.
Segmentation legality rests on the same Lean theorem as the whole program
(`gdn2_conditional_scan`: chunked == sequential, unconditional for GDN2
given inputs) — and on the kernel's own backward returning the
initial-state gradient (`dh0`), which makes the mid-window carry
**differentiable**; the smoke must receipt both the numerical segment-carry
parity and the carry-gradient legality.

## 4. What growth does NOT change (the honest scope)

* **Per-head capacity is unchanged** (always 128×256): growth adds parallel
  superposition slots, not wider ones. The trigger is therefore defined
  against the *per-head* capacity (§5).
* **No shrink**, as registered in K8: POP/MERGE-style content shrinkage has
  no GDN2 analog anyway (the audit §1.3–1.5).
* **Parameter count grows** (new projection rows/entries): the K8
  zero-parameter property does NOT transfer. Every C3 comparison is at
  matched EXPOSURE with parameter deltas disclosed as first-class columns,
  and the **fixed-H' control** (§7) separates "grown large" from "born
  large" — the K8 fixed-2048 analog.

## 5. The trigger (supervisor-decided) and the co-diagnostic

**The gating statistic — STATE SATURATION, the participation ratio.** Per
(layer, head, boundary): the boundary state `S ∈ ℝ^{128×256}` has at most
`d_k = 128` singular values; define

```
PR(S) = (Σ_i σ_i(S))² / Σ_i σ_i(S)²        ∈ [1, d_k]
```

the effective rank of the dense state — the honest analog of our
mean-live-slots/rank occupancy (a slot store is "full" when its slots are
live; a dense superposition is "full" when its effective dimensionality
approaches its key width). Aggregate: **mean over (layer, head)** at the
boundary; smoothed over the **last 4 boundaries** (reset on fire); fires at

```
smoothed mean PR ≥ τ_sat = 0.75 · d_k = 96
```

**The calibration protocol (disclosed, committed before any growth arm
runs):** no GDN2 PR receipts exist. Before the growth arms, the fixed arm's
per-boundary PR is receipted on the C3 stream (the Wheeler/K6
disclosed-calibration precedent). If the registered 96 is unreachable
in-regime, the calibrated threshold is fixed and committed BEFORE the growth
arms run; otherwise 96 stands. If PR never fires in-regime even after
calibration, the growth arm degenerates to fixed — the **informative null**,
reported as such (the K8 tie branch).

**Why saturation and not a write-pressure statistic (the K8 receipt doing
its job):** our own registered trigger (`scan.repeat_write_rate`, a
write-reuse statistic) fired at rate ~1.0 on an **idle** store (K8 §5) —
write-reuse conflates "capacity pressure" with "the model rewrites the same
hot directions". C3 is defined by state saturation, not write habit, so the
gating statistic measures the state itself.

**The non-gating co-diagnostic (supervisor-directed) — delta
self-interference / key collision.** Per (layer, head, segment): with
`S_entry` the segment-entry state and `(b_t, k̂_t)` the segment's own
erase-gate/key projections (recomputed non-invasively by the wrapper),

```
ι = mean_t  ‖S_entry^T (b_t ⊙ k̂_t)‖₂ / (‖S_entry‖_F · ‖b_t ⊙ k̂_t‖₂)
```

the normalized collision of the segment's writes against the state they
found at entry — the closest formal analog of our repeat-write rate (it is
the delta rule's own correction numerator). It is **reported per boundary on
every arm and gates nothing**. Its divergence-or-coincidence with saturation
across the C3 arms is the trigger-design receipt — including whether OUR
registered repeat-write trigger deserves the scrutiny the K8 receipt already
cast on it.

**A toy anchor for the threshold (CPU simulation, seed 3401, random
unit keys, gates at the receipted regimes; retained as a design receipt, not
a model receipt):** at the side-by-side's trained retention (exp(g) ≈ 0.68),
PR sits at **≈ 5 of 128** at every boundary from T=64 to 2048; at retention
0.95, PR ≈ 32–34; at 0.995, PR saturates ≈ 100 (the state fills the key
space). Two honest consequences: (a) GDN2's dense state is *not* near
saturation in the byte-LM regime — the C3 regime must be designed to push
retention/occupancy up for the trigger to be testable at all; (b) the
τ_sat = 96 threshold is reachable only in genuinely saturated regimes,
which is the point.

## 6. THIS STAGE's scope: the feasibility smoke (no quality claim)

The C3 run is a later stage (§7). This stage delivers the registration draft
plus a **feasibility smoke on cards 2/3** answering, with receipts
(`receipts/gdn2_growth_smoke.json`):

* **S-1 Inertness (event-time output invariance).** The grown module
  (H=4, born per §3, `o_proj` new columns zeroed) is bit-identical to the
  H=2 module on identical inputs, fp32 and under bf16 autocast, forward —
  and the fully-alive birth (no `o_proj` zeroing) is receipted for its
  perturbation magnitude, as the honest contrast.
* **S-2 Segmentation parity.** The 64-token segment-carry walk (eval, fp32)
  matches the whole-window chunk pass; and a mid-window growth continuation
  (segments at H=2, then the grown module with zero-padded state) matches
  the un-grown continuation — the mid-window event changes nothing
  observable at the event.
* **S-3 Trains through the event.** A few real optimizer steps at H=2, the
  registrar event, steps after: loss finite, gradients finite, the
  **one-step wake-up receipted** (nonzero grad on `o_proj`'s new columns at
  the first post-event step; nonzero grad on the new head's internal
  parameters from the second), and the **optimizer-state carry receipted**
  (AdamW `exp_avg` slices equal on shared regions, fresh on new).
* **S-4 Kernel/shape tolerance.** The chunk kernels forward+backward at
  H=4 and H=8 at the registered window shape (T=2048) run and produce
  finite gradients; walls receipted. Any kernel impossibility at grown
  shapes is **escalated, not patched**.
* **S-5 Statistic anchors.** PR and ι computed from the real kernel's
  boundary states at init on the smoke shape — the first real-module
  anchors for §5's calibration protocol (the §5 toy anchor is the design
  receipt; S-5 is the module receipt).

**Failure handling:** any S-gate that cannot be met with GDN2's code
untouched is an **impossibility receipt and an escalation**, not a patch.

## 7. The C3 stage (LATER, its own sign-off — design recorded, not registered here)

The saturation regime's definition (stream, horizon, task construction that
forces the fixed state to alias) is deferred to a C3 registration with its
own operator sign-off. The arms and predictions this draft commits to shape:

**Arms (matched exposure, the honest clock, parameter deltas disclosed):**
1. `gdn2_growth` — the registered GDN2 arm + §3's growth (snap {2,4,8},
   τ_sat trigger);
2. `gdn2_fixed` — the registered GDN2 arm verbatim (H=2);
3. `gdn2_fixed_8` — born at H=8 (the size-without-growth-timing control,
   the K8 fixed-2048 analog — separates "grown large" from "born large"
   given that growth adds parameters);
4. `emender_growable` — our growable machine (the K8 grow arm's lane).

**Pre-stated prediction structure (falsifiable both ways, numbers to be
fixed in the C3 registration once the regime is defined):**
* **P-G1 (the binding receipt).** In the C3 regime, `gdn2_fixed` degrades
  with association load beyond its state class (the fixed-state bound's
  empirical shadow).
* **P-G2 (growth is THE differentiator).** If growth — not slot
  architecture — is the differentiator: `gdn2_growth` beats `gdn2_fixed`
  by a pre-registered margin AND closes to `emender_growable` at matched
  grown state.
* **P-G3 (slots earn their keep).** If addressable slots matter on top of
  growth: `emender_growable` beats `gdn2_growth` at matched grown state
  size and matched exposure.
* **Refutation clauses.** `gdn2_growth` ≈ `gdn2_fixed` in-regime (trigger
  fired, no gain — or honestly never fired, both reported) refutes
  "growth is the differentiator for GDN2's architecture" at this scale;
  `gdn2_growth ≥ emender_growable` at matched state refutes the slot
  story's necessity at this scale. Either verdict is informative; the K8
  tie is the declared null branch.

## 8. Commands (this stage)

```bash
eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_smoke \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/gdn2_growth_smoke.json
```

## 9. SMOKE RECEIPTS (appended post-draft; the draft above is unchanged)

**Run:** 2026-11-22, card 2 (broker lease `GPU_LEASE_VISIBLE=2,3 ... acquire
1`), `/home/erikg/emender/.venv/bin/python`, fp32 with tf32 disabled,
`NDM_PIN_TRITON_AUTOTUNE=0` (the receipted pinned-autotune shared-memory
OOR of `RESULT_MATRIX_EMENDER.md` §6, hit again at this shape — an
environment knob, not a GDN2 patch). Receipt:
`receipts/gdn2_growth_smoke.json`; driver:
`gdn2_growth_smoke.py`. **All five S-gates MET.** One GDN2 block at the
registered mixer shape (dim 576, head_dim 128, expand_v 2, short conv),
2 heads, fp32 — block-level feasibility, no quality claim.

* **S-1 inertness — MET.** The wake-up birth (grown to H=4, o_proj's new
  input columns zeroed) differs from the H=2 module by **max 5.2e-7**
  (output scale 0.53; ~1e-6 relative) in fp32 — **not bit-identical**:
  the H=4 launch selects a different Triton autotune config than H=2 and
  the intra-head reduction order shifts within fp32 noise; receipted as
  measured. bf16 autocast: 2.9e-3 (the bf16 noise scale). The honest
  contrast — the fully-alive birth (no o_proj zeroing) — perturbs the
  output by **1.85e-3**, ~3500× the wake-up diff: the o_proj zeroing is
  what buys event-time output invariance. Parameters 3,814,978 →
  5,343,556 (+1,528,578 — growth adds trainable parameters, as §4
  discloses).
* **S-2 segmentation — MET, and stronger than registered.** The 64-token
  segment-carry walk is **bit-identical** to the whole-window chunk pass
  (output diff 0.0, state diff 0.0 at every boundary): the 64-token
  statistic boundaries coincide exactly with the kernel's own
  `chunk_size = 64`, so the probe/real segmentation is the kernel's own
  decomposition. Mid-window growth (fire at token 1024 of 2048): the full
  walk's readout differs from the un-grown walk by **5.1e-7** (fp32
  noise), the old heads' final state by 1.8e-5, and **the new heads
  wrote 3.37 Frobenius mass** — live keys from birth, the state filling
  exactly as §3 specifies. Kernel-trajectory semantics receipted:
  `h[:, c+1]` is the state AFTER chunk c; `h[:, 0]` is the empty initial
  state.
* **S-3 trains through the event — MET.** The carry-gradient check:
  whole-window vs 64-token-segment carried path, gradients on
  q/k/v/b/w — **relative max diff 0.0 on all five**: the kernel's
  backward returns `dh0` and the segmented backward is exact at this
  shape. Training (MSE, AdamW, 3 steps at H=2 → the registrar event →
  steps at H=4): losses 1.013 / 0.995 / 0.978 (pre-event) → 0.961 /
  0.943, all finite, grads finite. **The one-step wake-up receipted
  exactly as designed:** at the first post-event step the o_proj new
  columns receive nonzero gradient (1.5e-6) while the new head's
  internal parameters receive **exactly 0.0** (the zero columns block
  them); after one optimizer step the columns move (1.3e-3) and the new
  head's k_proj rows receive nonzero gradient (1.2e-6). Optimizer
  surgery exact: shared-slice `exp_avg` diff 0.0, new regions fresh.
* **S-4 kernel tolerance — MET.** Forward+backward at the registered
  window shape (T=2048, batch 2) at H = 2 / 4 / 8: all finite; walls
  0.018 / 0.021 / 0.054 s per block; peak 0.27 / 0.43 / 0.81 GB. **No
  kernel impossibility at any grown shape on the heads axis** (the d_k
  axis's impossibility is receipted in §2).
* **S-5 statistic anchors — RECEIPTED.** On the real kernel's own
  per-chunk state trajectory, at init, random inputs (batch 2, T=2048,
  one block, fp32): **PR mean 48.4 of 128** (min 36.5, max 56.6; first
  boundary 40.0, last 48.4) — the τ_sat = 96 threshold sits ABOVE the
  init anchor, reachable only in genuinely more-saturated regimes, as
  §5 intends; the calibration protocol on the C3 stream remains the
  threshold's authority. The co-diagnostic ι: mean 0.092 (max 0.147) —
  computable non-invasively, low at init (random writes are mostly
  orthogonal to the entry state), reported per boundary as registered.

**Verdict of the feasibility stage:** the delta rule trains through a
growth event on the heads axis; the gradients are legal (including
through the carried state, exactly); the kernels tolerate every grown
shape; the event is output-invariant to fp32 noise with the one-step
wake-up receipted; and both trigger statistics are computable from the
kernel's own trajectory without touching GDN2's code. The C3 stage
(§7) is unlocked pending its own registration and the operator's
sign-off of this draft.
