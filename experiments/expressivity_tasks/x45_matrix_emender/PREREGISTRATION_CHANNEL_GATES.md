Pre-registration draft — PER-CHANNEL GATES ON THE EMENDER (K11B): GDN2's
gate granularity — per-channel, per-token, input-dependent DECAY, write
admission and read-out — through the existing recorded-schedule channels

**Date:** 2026-11-23 · **Status: DRAFT, operator sign-off pending.** This is
a NEW registration (it changes the store's CONTINUOUS gate parameters to
GDN2's granularity; the algebra's discrete op selection is untouched),
written before any probe measurement is claimed. Probe-scale engineering
checks (parity gates, gradchecks, memory, timing) are pre-registration
evidence and are in scope now; **no board-scale (2700 s) claim is run
before the operator signs this document off.** It does not touch any
`k10_*`/`k11_*`/`k12_*` source, any committed PREREGISTRATION, `lattice.py`,
or GDN2 model code — those are imported or read, never edited.

## 0. The chain this stage closes, and the enabling fact

K11 tested the per-head SCALAR write-strength + read-out gate surface and
receipted the honest null (`RESULT_LEARNED_GATES.md`): every arm at parity,
the refutation clause applied verbatim, and the null **scoped to that form**
— "the per-head scalar gate surface does not own the residual"; the
registration's own §7.2 named the per-slot/per-channel variant as the
natural next arm. The op-set audit (`RESULT_OP_SET_AUDIT.md` §1.4) sharpens
why that arm is mandatory before any terminal reading: our DECAY is "the
strictly poorer instance — per-head scalar vs GDN2's per-channel
input-driven." GDN2's actual gates are **per-channel, per-token,
input-dependent** (`/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py`):
`f_proj` a small bias-free two-layer MLP to a per-token decay
pre-activation, `b_proj` a single bias-free Linear write/erase gate,
`g_proj` a two-layer read gate — together the bulk of its 27.4 M gate
parameters (18.3 % of the machine). **The per-channel form on our machine
is untested; this lane is it.**

The enabling fact is K11's own, verbatim: gates are SCHEDULE DATA. The Lean
algebra's APPEND writes `(key, gate·value)` and MERGE blends
`old + beta·(new−old)` with all gate magnitudes part of the fixed
`TokenDecision`; the affine-in-values scan theorems hold for ANY supplied
gate values, and the discrete op SELECTION stays thresholded and recorded
(P2). Making the magnitudes per-channel adds zero new math: only the
PRODUCER of the vector changes.

**Supervisor rulings received before this document was committed** (the
task's escalate-before-commit rule, cited at each choice below):

* **R1 — the per-channel DECAY birth form, APPROVED as proposed.**
  `lam_vec = clamp(lam_inc · (1 + m_f·σ(f_proj(x))), DECAY_FLOOR, 1−1e−6)`
  per (token, head, channel), with `lam_inc = σ(decay(x)).clamp(...)` the
  incumbent's REGISTERED per-head scalar projection **kept verbatim and
  still trained**; `f_proj` GDN2's own shape transplanted
  (`Sequential(Linear(width, d_v, bias=False), Linear(d_v, heads·d_v,
  bias=False))`, all weights born at exact zero) and `m_f` a per-head
  scalar born at EXACT zero. At birth the factor is exactly 1.0, so
  `lam_vec` is BIT-EQUAL to the registered scalar on every channel (IEEE
  mul-by-1.0 is exact; clamp of an in-range value is the identity) — the
  0-flip bar holds at probe scale. The honest deviation disclosure, ruled
  into the registration verbatim: the per-channel input-DEPENDENCE and
  channel width are transplanted whole; the ABSOLUTE form is a deviation
  around the registered scalar because mimetic-neutral birth requires it
  (GDN2 has no incumbent scalar to be born equal to). The channel's reach
  is bounded: `lam_vec ∈ λ_inc·(1±|m_f|) ∩ [DECAY_FLOOR, 1−1e−6]`. The
  GDN2-closer log-space form (`λ_inc·exp(m_f·σ(f))`) was considered and
  rejected as primary (no birth-parity advantage, an extra per-channel
  exp in the walk).
* **R2 — the axis interpretation, CONFIRMED: d_v (the VALUE channel) is the
  primary.** GDN2's f_proj decays its FIXED key axis; our machine has no
  fixed address axis — slots are content-addressed and dynamic — so the
  only axis a learned projection can transplant onto whole is the value
  channel: per-channel memory timescales on values, the same class of
  expressivity GDN2 realizes on keys. The rejected strict analog is named
  in §7: PER-SLOT (address-axis) decay, "decay-by-address-match" — the
  write key's similarity to each slot entering the decay — a genuinely
  different mechanism, not a projection transplant, and the NAMED FOLLOW-UP
  if arm 1 nulls. Not built in this lane.
* **R3 — the engine limitation, APPROVED.** K4's imported
  `persistent_scan_trajectory` (the K11 backward's segment value replay)
  takes per-head SCALAR decay only; the per-channel form cannot ride it.
  K11B carries lane-specific kernels in the NEW file
  `k11b_channel_gates.py` at K4's defined-order spec (a per-channel
  trajectory replay, a forward walk with vector decay/gate loads, a
  value-walk backward with per-channel dlam/dgate adjoints); K10's
  bwd-prep and K11's key-traj/key-walk kernels are imported verbatim. No
  committed engine is touched; the new kernel path carries the full
  bit-equality burden in §3.
* **R4 — the parameter accounting, APPROVED.** K11's 0.1 % capacity
  tolerance is DROPPED for this lane (a real per-channel transplant cannot
  hold it); the deltas are stated in §1.5 and asserted by the probe, the
  capacity asymmetry vs GDN2's 18.3 % gate share is the finding (even arm
  3 is ~3× leaner), and the FROZEN-GATES falsifier control is MANDATORY in
  every arm's receipt (§2.4).
* **P2/P5 carried verbatim from K11**: decisions are inviolable (no gate
  value feeds op selection; the 0-flip receipt), and every new channel is
  born mimetic-neutral (the arm's step-0 behavior IS the banked winner's,
  bit for bit; the gates grow by training).

## 1. The architecture under test

**The K10 winner (per-token + separated keys, the tuned config) with its
continuous gate parameters moved to GDN2's granularity — per-channel,
per-token, input-dependent — at the matched training surface (the banked
winner lineage: hybrid Muon at 4× the tuned AdamW LR, the gate projections
on the AdamW side, the DECAY gate's own convention).** Same five-op
algebra, same thresholds, same key channel, same per-token granularity,
same defined-order arithmetic spec, same discrete op selection. Three arms,
cumulative per the task's own staging:

### 1.1 Arm P1 — per-channel input-dependent DECAY (the f_proj analog; the
priority, the audit's own "strictly poorer" row)

A per-token `d_v`-dim decay VECTOR replacing the per-head scalar in the
DECAY channel (values only; keys are never decayed — K10's registered
order, DECAY then lanes, unchanged):

```
lam_vec[b,t,h,c] = clamp( lam_inc[b,t,h] · (1 + m_f[h]·σ(f_proj(x))[b,t,h,c]),
                          DECAY_FLOOR, 1−1e−6 )
```

(R1's approved form, restated.) P2 verbatim: the decay vector never feeds
op selection — the decision chain runs on the detached key channel against
the PRE-decay state exactly as registered. The channel is consumed
everywhere the scalar was: the whole-state decay `S ← lam_vec ⊙ S` (per
column), the CORRECT blend's old-row term, and the backward's per-channel
adjoint `dlam[c] = ⟨dS[:,c], S_t[:,c]⟩`.

### 1.2 Arm P2 — + per-channel write admission (the b_proj analog)

`g_vec = 1 − m_w[h]·σ(b_proj(x))` per (token, head, channel), with
`b_proj = Linear(width, heads·d_v, bias=False)` zero-born (GDN2's b_proj is
a single bias-free Linear) and `m_w` per-head born at exact zero. Consumed
through the existing write/injection channel, at the two content-carrying
ops, per channel: **APPEND admits `g_vec ⊙ p`** (the Lean APPEND's
`gate·value`), **CORRECT blends `g_vec ⊙ p + (1−g_vec) ⊙ (lam_vec ⊙ v_pre)`**
(the Lean MERGE arithmetic at the CORRECT write enable — the delta rule on
the dominant op, now at GDN2's granularity). MERGE/POP/NONE stay VERBATIM.

### 1.3 Arm P3 — + per-channel read gates (the g_proj analog), combined

`r_vec = 1 + m_o[h]·σ(g_proj(x))` with `g_proj = Sequential(Linear(width,
d_v, bias=False), Linear(d_v, heads·d_v, bias=True))` zero-born (GDN2's
g_proj shape, second layer with bias), applied `reads ⊙ r_vec` in the mixer
before the output projection (outside the scan; K11's read-out
convention). Born at exact 1 per channel.

### 1.4 The arithmetic spec (inherited verbatim, elementwise in the channel)

K6/K10/K11's defined-order spec applies unchanged (balanced binary tree
folds, fp32 decision/read arithmetic, correctly-rounded `sqrt.rn`/`div.rn`,
libdevice `exp`, `mul.rn`/`add.rn`/`sub.rn` — no FMA contraction). The
per-channel extensions: the DECAY is `mul.rn(acc, lam_vec)` broadcast per
value column (at birth the incumbent's scalar decay per element, bit for
bit), and the CORRECT blend's defined-order per-channel form
`g·p + (1−g)·lam·v_pre` (one `sub.rn`, three `mul.rn`, one `add.rn` — the
same value as `v_dec + g·(p−v_dec)` in exact arithmetic, chosen so the
register walk keeps K10's registered lane order AND the unity-gate birth
is the incumbent's overwrite bit for bit, the −0.0 corner K11 receipted).
The recorded lane-0 payload is the BLENDED lane value, so the recorded
schedule replays through ordinary scheduled writes — the gates never
appear in the replay. The recorded schedule is K10's own 2-lane layout; no
dense lane (K11's G2 receipted the dense-write channel's null separately).

### 1.5 Parameter accounting (stated, not hidden; R4)

Per mixer, GDN2's own projection shapes transplanted: the decay channel
`f_proj` 576·128 + 128·256 + 2 = **106,498**; the write channel `b_proj`
576·256 + 2 = **147,458**; the read channel `g_proj` 576·128 + 128·256 +
256 + 2 = **106,754**. At depth 24, cumulative:

| arm | channels | params added | total | share | GDN2's gate share |
|---|---|---:|---:|---:|---:|
| P1 | decay | +2,555,952 | 152,537,760 | +1.70 % | 18.3 % |
| P2 | decay+write | +6,094,944 | 156,076,752 | +4.06 % | 18.3 % |
| P3 | decay+write+read | +8,657,040 | 158,638,848 | +5.77 % | 18.3 % |

The 0.1 % capacity tolerance is DROPPED for this lane (R4): a real
per-channel transplant cannot hold it, and even arm 3 is ~3× leaner than
GDN2's own gate share — the asymmetry is the finding, not a caveat. The
comparison remains config-matched and protocol-matched to the banked
winner; the parameter asymmetry is quoted in every receipt.

## 2. The protocol

### 2.1 The probe

Every arm is the banked winner config VERBATIM (the K10 winner build,
hybrid Muon at 4× the tuned AdamW LR, the banked `banked_gates_adamw`
routing with the per-channel gate projections on the AdamW side — the
DECAY gate's own convention, cross-checked name-for-name against the
banked split on the un-converted twin) plus the arm's per-channel channels
at the mimetic-neutral birth, at the matched-exposure probe protocol: the
20,022,866-target landing, 180 constant-LR steps, seed 3401, the registered
flagship stream, the document-disjoint calibration eval (7 windows /
14,336 targets). Primary seed 3401 on every arm.

### 2.2 The scoring

The corrected protocol, imported verbatim (`protocol_correction.py`): the
24-step linear cooldown PRIMARY, EMA-0.99 at step 180 SECONDARY
(ranking-only), constant-180 RETAINED for banked comparability. Both
anchors quoted everywhere: the constant-LR banked best **1.4899048124040877**
(seed 3402; the same config's seed-3401 row **1.4907409463609969**) and
the corrected post-cooldown banked best **1.4338248457227434** (seed 3402).
GDN2's rows (constant 1.3951, post-cooldown **1.3522**) are the gap
reference.

### 2.3 The pre-stated question (no directional prediction for arm 1's
magnitude)

**Is the "write geometry" residual actually GATE GRANULARITY?**

* **(i) per-channel decay closes ≥ 0.02 of the gap** (the arm's
  post-cooldown primary score ≤ 1.4138 vs the corrected anchor 1.4338,
  or the same-seed constant-180 improving ≥ 0.02 on the banked
  1.4907/1.4899) ⇒ the C5 attribution refines to "gate granularity", the
  quality program has one more real lever, and the leaner-op-set
  registration gains the per-channel decay gate as a required component.
* **(ii) null at parity** (every arm inside the 0.075-nat
  endpoint-variance band of the seed-matched banked rows on the primary
  scoring) ⇒ the residual is the DENSE SUPERPOSITION ITSELF — values
  written at every token vs events — a form our machine cannot adopt
  without becoming GDN2; the honest TERMINAL reading of the quality
  program, receipted as such.

Both outcomes are results. The **win bar for the second seed is 0.01**
(seed 3402 runs on any arm whose primary score beats the seed-matched
comparison by ≥ 0.01); the **gap-closure bar for reading (i) is 0.02**.

### 2.4 The frozen-gates falsifier control (MANDATORY, R4)

One control run per lane (on the priority arm): the gate parameters frozen
at birth (requires_grad=False, excluded from the optimizer). **The bar,
stated exactly:** (a) the control's step-1 train loss MUST be BIT-IDENTICAL
to the banked same-seed row's (the forward-at-birth plumbing receipt, the
same 0-flip bar every live arm asserts); (b) the control's landing MUST be
at parity with the banked same-seed row (inside the winner's own measured
seed-delta scale on the primary scoring). **The honest limitation, stated
not hidden:** full-trajectory bit-identity is NOT achievable for this lane
by construction — the per-channel DECAY adjoint reduces
`d(lam_inc) = Σ_c dlam[c]` over the value axis in autograd's order, while
the incumbent's backward fuses the same sum over (rank, channel) in the
kernel's block order; the two differ in ULPs, and the difference compounds
chaotically through 180 optimizer steps even with every gate frozen. The
control therefore measures everything-except-gate-learning at the band's
scale — module creation, optimizer routing, dtype promotion, the new
kernels' arithmetic — with the per-step loss divergence against the banked
row measured and quoted (expected: exact 0 at step 1, ULP-scale growth
after). The step-1 bit-identity receipt (K11's 0-flip-at-birth bar) is
asserted in every live arm too.

### 2.5 The mechanism duties

Per arm, at BOTH the step-180 constant state and the post-cooldown final:
the K11 mechanism instrument (identity-rewrite share via the per-channel
trajectory replay, read entropy / effective rows, occupancy, op mix), the
read-mask sweep (the side-by-side's C4 diagnostic, imported verbatim), and
the gate telemetry — the born-identity drift: per-layer mean
`|lam_vec − lam_inc|` (the decay gate's distance from the registered
scalar), the write gate's `|1−g_vec|`, the read gate's `|1−r_vec|`, and the
two-step wake-up receipt (the scalars move first; the projections' step-0
gradient is exactly zero by construction, receipted so the step-0 zero is
not read as a dead channel). No directional mechanism prediction is
pre-stated for arm 1 (the task's own rule); the K11 anchors (identity
0.118–0.122, entropy 1.515–1.600, effective rows 4.55–4.95, occupancy
33.5–35.2, sweep 5–6/24 layers ≥ 0.02) are quoted as the parity references
any granularity effect would have to move.

## 3. The parity gates (committed before any probe measurement)

* **Gate A** — oracle-vs-kernel bit-equality including the per-channel
  path: the 2-lane schedule fields, the payload lanes, the read WEIGHTS,
  the pre-decision liveness, the exit state/liveness/key table
  (`torch.equal`) in both GATED combinations and the bf16 autocast regime;
  the read at the measured fp32-vs-fp64 floor; the 0-flip decision-
  invariance receipt (P2); the birth-parity receipt vs K10's OWN launch
  (P5), and the mixer-level birth receipt (the converted mixer's output
  bit-identical to the K10 mixer's at birth).
* **Gate B** — the fp64 gradcheck of the oracle through the payload, the
  PER-CHANNEL DECAY vector, the PER-CHANNEL GATES, the key channel, the
  value state and the key table; the seven kernel gradients vs the
  fp32-vs-fp64 floor (NOISE_FACTOR 8×); the per-channel trajectory replay
  bit-exact against the forward's own segment checkpoints; the zero-born
  scalars' step-0 gradient receipt and the two-step wake-up.
* **Gate D** — walls at the registered shape vs K10's and K11's OWN
  launches on the same box, the honest per-channel cost statement.
* The committed K10/K11 test suites re-run green with the new module
  loaded.

## 4. The refutation clause (pre-stated)

If the per-channel arms land at parity — no mechanism number moves and the
primary score does not beat the seed-matched banked rows beyond the band —
then GATE GRANULARITY does not own the residual at probe scale, and the
honest reading is (ii): the residual is the dense superposition itself.
That is reported as the TERMINAL reading of the quality program's
write-geometry attribution (C5 refined to its final form), not excused —
with the probe-scale caveat stated (180 steps; the gates may need a longer
horizon to open) and the named follow-ups (§7) held open. If an arm wins
by ≥ 0.02, reading (i) is reported with the second seed's confirmation
required at the 0.01 bar before any claim is banked.

## 5. Honest scope

* Probe scale only (the 20 M-target matched exposure, 180+24 steps); no
  board-scale claim.
* The arms ADD parameters (§1.5) — the 0.1 % tolerance is dropped by
  ruling; the comparison is config- and protocol-matched, and the
  parameter asymmetry vs GDN2's 18.3 % gate share is quoted in every
  receipt.
* The absolute decay form is a bounded deviation around the registered
  scalar (R1's disclosure): the per-channel reach is
  `λ_inc·(1±|m_f|) ∩ [floor, 1−1e−6]` — GDN2's own decay is unbounded in
  rate. The lever tested is the per-channel input-dependent STRUCTURE, not
  GDN2's exact parameterization.
* The per-channel form costs walls: the vector DECAY load and the
  per-channel gate loads add registers to a walk already at the K11
  ceiling (255 regs / 2 spills); gate D receipts the honest cost.

## 6. Files

`k11b_channel_gates.py` (the engine; new, imports the committed kernels),
`test_k11b_channel_gates.py` (the parity suite), `k11b_gate_receipt.py`
(gates A/B/D at the registered shape), `K11B_CHANNEL_GATES.md` (the gate
receipts), `k11b_probe_channel_gates.py` (the probe + the assemble),
`run_k11b_channel_gates_leased.sh` (the detached leased driver), the
per-arm receipts under `k11b_runs/`, and `RESULT_CHANNEL_GATES.md` (the
result, written only after the arms land).

## 7. Open registration questions

1. **Per-slot (address-axis) decay — the NAMED FOLLOW-UP (R2).** The true
   algebraic f_proj analog on dynamic addresses: the write key's
   similarity to each stored slot entering that slot's decay
   ("decay-by-address-match"). Not a projection transplant; a genuinely
   different mechanism. If arm 1 nulls and the audit's "strictly poorer
   instance" claim is to be closed, this is the remaining reading.
2. **The dense-write channel at per-channel granularity.** K11's G2
   (per-head dense deposits) nulled; a per-channel dense deposit (the
   `w ⊙ v` analog) was NOT run in this lane — the task's arms isolate the
   f/b/g-projection triad. Named, not run.
3. **The interaction of the per-channel DECAY with the read-mask sweep's
   near-decorative stores.** If per-channel timescales open, the sweep's
   17/24 near-decorative anchor is the first number to move; no
   prediction is pre-stated.
4. **Board-scale.** Whether any winning arm graduates to the 2700 s board
   is the operator's call, pending this lane's outcome and sign-off.
