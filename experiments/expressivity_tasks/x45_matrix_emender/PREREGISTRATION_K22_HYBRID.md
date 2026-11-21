# K22 — THE HYBRID: registration draft (committed BEFORE any arm)

**Status: DRAFT, committed before any probe measurement (the K11B lesson).
The engines are imported verbatim and never modified; every new file is a
`k22_*` file. No arm runs before this draft is committed.**

## The machine (the operator's architecture go: 3 slot layers of 24, mid-stack)

GDN2's committed 150M arm — `matched_arms.build_gdn2` verbatim: the
registered `lattice_gdn2` nomination, 24 dense mixers at width 640, 149,855,670
trainable parameters, the banked **1.3522 machine** (`protocol_correction_runs/
gdn2_speedrun_rescored`: post-cooldown **1.3521734476089478**, constant-LR
1.3950541530336653, EMA-180 1.3803383282252721, seed 3401) — with **3 of its
24 mixer positions carrying our committed composed slot mixer**: the
K15/K18 engine VERBATIM (`ShortConvComposedMixer`: the K13 additive read,
the K11B p3 per-channel decay/write/read gates, the K10 separated keys at
the mimetic birth, the K18 identity-born short conv), instantiated at the
host's width **640** with the registered store geometry unchanged (heads 2,
value_dim 128, rank/rank_max 128, key_dim 16, ctx 4, segment_tokens 64,
the K19-confirmed incumbent thresholds tau_match 0.45 / tau_merge 0.85 /
tau_pop 0.10 / read_scale 2.7870458737509636). **Independent stores per
slot layer** (the shared-pool variant is a named follow-up, NOT this lane).

### The seam — settled by the non-negotiable falsifier

The task text says both "3 of its 24 mixers REPLACED" and "the machine is
born bit-identical to the committed GDN2 arm … a hybrid frozen at zero-read
must be bit-identical to pure GDN2's banked trajectory FOREVER". A literal
removal of the dense mixers at the slot layers makes the birth-parity bar
impossible BY CONSTRUCTION (a 21-mixer GDN2 is not the banked 24-mixer
machine at step 1). The falsifier is non-negotiable, so the only reading
that satisfies the birth discipline is the additive one it dictates, and
this lane registers it plainly:

> **The slot layer KEEPS GDN2's dense mixer and GAINS the slot read, added
> to the dense mixer's output through a ZERO-BORN READ GATE** (the K13
> additive-read convention at the layer seam). At birth the slot
> contribution is exactly zero; the machine is bit-identical to the
> committed GDN2 arm; the extra parameters are present but inert; any
> measured delta belongs to the slot reads opening.

The seam's expression (one new module, `k22_hybrid.HybridSlotGDN2Layer`,
which HOLDS the committed `GDN2ExternalLayer` untouched):

```
out, cache = self.gdn2(x, h0, **kwargs)          # the committed layer, verbatim
return out + self.read_gate.to(out.dtype) * self.slot(x), cache
```

**The exact parameter delta, quoted honestly (computed, not hardcoded —
`hybrid_slot_parameter_delta`)**: purely ADDITIVE, nothing removed:

| component | per slot layer | x3 |
|---|---|---|
| `value` (640x256) | 163,840 | 491,520 |
| `key` (2560x32) | 81,920 | 245,760 |
| `query_key` (2560x32) | 81,920 | 245,760 |
| `out` (256x640) | 163,840 | 491,520 |
| `decay` (+bias) | 1,282 | 3,846 |
| `f_proj.0/1` | 114,688 | 344,064 |
| `b_proj` | 163,840 | 491,520 |
| `g_proj.0/1` (+bias) | 114,944 | 344,832 |
| the 3 channel scalars | 6 | 18 |
| `v_conv` (256x4) | 1,024 | 3,072 |
| the K22 read gate (1 scalar) | 1 | 3 |
| **total added** | **887,305** | **2,661,915** |

**+2,661,915 parameters (+1.776%) on 149,855,670 → 152,517,585.** Near-params
in the honest sense (the delta is smaller than the machines' own
backbone-level differences and 3.4x smaller than the composed machine's
own p3 gate stack); stated in every receipt, never hidden.

### The placement (picked and receipted)

**Slot layers {6, 12, 18}.** Receipts: (a) Memorizing Transformers (Wu et
al., arXiv:2203.08913): a single kNN-memory layer in a dense stack is
optimal placed **mid-stack** (their own sweep; our closest published
precedent for a memory layer inside a dense stack); {6, 12, 18} is the
even 6-spacing centered on the middle (layer 12), the Jamba/Samba
minority-interleave pattern at the 1:7 ratio. (b) Our own decorative
sweep's load-bearing positions: the composed machine's own banked sweeps
(`k15_runs/composition_seed3401`, `k18_runs/conv_seed3401`) put the
load-bearing mass (>= 0.02 dNLL) in the FIRST half — layers 0-4, decaying
with depth, layer 6 the last cell at half the bar — so the earliest
mid-stack spread {6, 12, 18} covers the sweep's load-bearing decline
while honoring the operator's mid-stack go. (c) Independent stores per
slot layer: the three stores' write/read cycles do not interact by
construction.

## The birth discipline (the K13/K11B machinery inverted)

Each slot mixer's **READ gate born at ZERO**: `read_gate` is one scalar
per slot layer (K22's own seam lever — the cleanest attribution that can
exist), born at exact 0.0, so `out + 0.0 * slot(x) == out` bitwise (the
only legal escape is the IEEE sign-of-zero corner — a +0.0 contribution
added to a -0.0 dense entry flips the sign of zero; counted and receipted
by the gate receipt, invisible to every value comparison and to the loss).
The dense GDN2 mixers, their parameters, their routing and their
optimizer treatment are the banked row's own VERBATIM. **The store
machinery runs from step 1** (APPEND/CORRECT fill the store; the ops fire
inside the dense stack) **but nothing reads out until a gate opens**; the
gate's gradient (dLoss/dgate = <upstream, slot output>) is the open
signal, and while a read gate is exactly zero every OTHER slot parameter
receives an exactly-zero gradient (inert at birth — receipted). Disclosed
dynamics, stated not patched: while the gates are closed, decoupled
weight decay still shrinks the inert slot parameters (~10% over the
204-step probe at the banked recipe's lr*wd); the engine's own per-channel
r_vec read gate stays at the composed machine's committed birth (r_vec ==
1, untouched — the K22 seam gate is the only new birth lever).

## The falsifier (non-negotiable)

`hybrid_frozen`: the same machine with the whole slot branch frozen
(`requires_grad=False` on every slot parameter and the read gates; the
branch sits outside the optimizer, the grad clip and the EMA by
construction, and the parameter split then equals the banked GDN2 row's
own exactly). **It must be bit-identical to the banked GDN2 trajectory
FOREVER**: every one of its 204 steps equal to
`protocol_correction_runs/gdn2_speedrun_rescored/training.jsonl`
(max per-step delta 0.0 — the K18 perfect-receipt precedent is the
expectation to beat), landing at **1.3521734476089478** post-cooldown bit
for bit, and the step-1 train loss equal to the banked row's own
5.948033220866492 (the protocol-identical start receipt — the optimizer
cannot touch the first forward). **Any divergence is a birth-parity
conflict: STOP and escalate.**

## The arms and the protocol

* `hybrid` — born zero-read, the gates LEARN to open. Seed **3401** first.
* `hybrid_frozen` — the falsifier, seed 3401.

Matched-exposure probe protocol: train until 20,000,000 targets (landing
20,022,866 on the registered flagship stream), the corrected
cooldown-primary scoring (the 24-step linear cooldown, EMA-0.99 secondary,
the step-180 constant-LR eval retained for banked comparability), the
document-disjoint calibration eval, the honest clock with the optimizer
seconds receipted separately, grad clip 1.0. **The recipe is the banked
GDN2 row's own VERBATIM**: HybridMuonX45 at the 5x speedrun point (peak LR
0.0027100927623701265), option-B routing (gates on AdamW), warmup 69,
microbatch 8 — the ONLY difference from the banked row is the machine.
The slot branch routes per the K18 convention mirrored: the hidden-map
matrices (value/key/query_key/out) on Muon; the gate channels (decay,
f_proj, b_proj, g_proj, the three scalars), the depthwise conv kernel and
the K22 read gate on AdamW — with the routing cross-checked equal to the
banked GDN2 row's own split on the normalized dense names against a
pure-GDN2 twin.

**Second seed 3402 on any win >= 0.01 per the standing rule.**

Velocity honesty (the K20/K21 receipts, cited): the hybrid step is
projected 1.37x pure-GDN2 (3-of-24 at the measured 7.28x per-mixer ratio),
and the committed block checkpointing re-runs the slot scan in backward,
so the honest wall is reported first-class — matched-EXPOSURE means the
arms eat the SAME bytes in different wall-clocks.

## The pre-state (directional, per the literature, with refutation clause)

* **H-1**: hybrid post-cooldown < 1.3521734476089478 — the hybrid beats
  pure GDN2 outright at matched exposure (the Jamba/Samba/Meta
  matched-param minority-interleave pattern). A margin >= 0.01 is a
  BANKABLE win; a sub-band delta gets the parity phrasing. No
  band-worship.
* **H-2**: hybrid <= 1.3811104467936925 (the composed machine's own
  same-seed row; 1.383232252938407 at 3402) — the slot store pays inside
  a dense stack even though the pure slot machine lost (the Meta
  complementarity receipt).
* **REFUTATION**: if the hybrid lands >= pure GDN2's row, the
  literature's minority-interleave gain did NOT transfer to byte-LM at
  150M/20M — receipt it plainly; the uniformity finding would then extend
  to the hybrid regime.

## The mechanism duties (the decisive instruments)

* **THE DECORATIVE SWEEP IS THE HEADLINE INSTRUMENT**: the slot-read mask
  per slot layer post-cooldown — do the slot layers become load-bearing in
  the hybrid (does the uniformity finding invert — more than 4-5 layers
  mattering)? The banked pure-machine regime is 4-5 of 24 layers >= 0.02
  dNLL; the hybrid's question is whether its 3 slot layers clear that bar.
* **The gate trajectories**: which slot layers open their read gates, how
  far (the read gate values at every checkpoint, init / 1 / 5 / 10 / 20 /
  40 / 80 / 120 / 160 / 180 / final / EMA-180).
* **The op-mix rows on the slot layers**: APPEND/CORRECT/POP/MERGE shares
  inside a dense stack (the C5 instrument's own walk).
* **Occupancy / effective rows / identity share vs the pure machine's**
  banked mechanism anchors (cited, never re-run).
* **The C5 instrument per arm** (the K15/K18 mechanism receipts on the
  slot mixers: effective rows, occupancy, identity share, the p3 gate
  telemetry, the conv telemetry).
* **fp64 gradcheck through the hybrid seam** (the slot-in-dense-stack
  boundary), both gated combinations, and the kernel path's seam gradients
  within the measured fp32-vs-fp64 floors (gate B of
  `k22_gate_receipt.py`).
* **K10/K13/K11B/K15/K17/K18 suites stay green with the module loaded**
  (the standing receipt; the console committed).

## The gate receipts (`k22_gate_receipt.py`, run BEFORE any arm)

* **Gate A** — born as GDN2, bit for bit: the hybrid vs a pure-GDN2 twin
  of the same build on one real production microbatch; the full-model loss
  and every slot layer's combined output asserted bit-identical (flips
  counted bitwise; the sign-of-zero corner counted separately, flips
  outside exact-zero entries must be ZERO).
* **Gate A2** — the seam's dtype discipline under the production bf16
  autocast regime (the residual stream's dtype is the committed machine's
  own).
* **Gate B** — the fp64 gradcheck through the hybrid seam (above).
* **Gate C** — the open signal (dL/dgate nonzero at birth), the inert
  birth (every other slot gradient exactly zero), and the store fills.
* **Gate D** — the honest wall (the slot branch's own forward cost,
  stated against the K20/K21 receipts).

## The rules

Cards 2/3 (one leased card at a time, `GPU_LEASE_VISIBLE=2,3 … acquire 1`);
`/home/erikg/emender/.venv/bin/python`; `OMP_NUM_THREADS=4`. Registration
draft committed BEFORE any arm (this document). Commit per arm; detached
chains for anything long (the PPID=1 playbook). The committed engines are
never touched (a parallel lane K23 owns speed work in scratch; any engine
change would break the hash-bound receipts). **Escalate on: birth-parity
conflicts (if born-as-GDN2 cannot be made bit-identical, STOP), placement
ambiguity the receipts cannot settle, and any clause verdict that inverts
a standing conclusion.**
