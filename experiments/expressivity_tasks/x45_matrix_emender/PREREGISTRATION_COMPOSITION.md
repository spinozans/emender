Pre-registration draft — THE COMPOSITION (K15): the two proven-positive
quality levers of the session, composed — the K13 ADDITIVE READ on the K11B
p3 PER-CHANNEL GATE STACK

**Date:** 2026-11-24 · **Status: DRAFT, operator sign-off pending.** This is
a NEW registration (it composes two committed, green engines into a new
machine-variant — the read form AND the gate granularity change together),
written and committed BEFORE any probe measurement is claimed. Probe-scale
engineering checks (parity gates, gradchecks, memory, timing) are
pre-registration evidence and are in scope now; **no board-scale (2700 s)
claim is run before the operator signs this document off.** It does not
touch any `k10_*`/`k11*`/`k12_*`/`k13_*`/`x46_*` source, any committed
PREREGISTRATION, `lattice.py`, or GDN2 model code — those are imported or
read, never edited. New files only: `k15_*.py`, `test_k15_*.py`, the
receipts, this document.

## 0. The question this stage exists for

K13 receipted the **additive read** as the session's largest single lever:
post-cooldown **1.3928** (seed 3401) / **1.3845** (seed 3402) vs the banked
corrected anchor **1.4338** — −0.041 / −0.049, ~60 % of the 0.0816-nat
corrected gap closed, the |w|-mass effective-rows instrument at
**3.61× / 3.37×** the banked anchors (16.42 / 16.69), the collapse detector
quiet (top-row share 0.156 / 0.154), no mechanism collapse. K11B receipted
the **p3 per-channel gate stack** (the f_proj decay + b_proj write + g_proj
read channels) as seed-stable on the SOFTMAX read: post-cooldown
**1.42353** (seed 3401) / **1.42343** (seed 3402) vs the same 1.4338 anchor
— **+0.0103 / +0.0104**, a positive but small, exactly-replicated effect.
The two levers touch DIFFERENT sub-mechanisms of the same store: the read
form is a value-channel change that touches neither the store nor the ops;
the gate stack scales the DECAY/write/read magnitudes per channel,
input-dependently, and never op selection. **Do they compose?** The
mechanism stories do not obviously conflict (nothing in the additive form
needs the softmax the gates were tuned under), but they also do not
obviously add (the gates' +0.010 was measured ON the softmax read — a
read-form-dependent effect would vanish, halve, or invert under the
additive read). This lane is the composition arm at the matched-exposure
probe protocol, both seeds. **No directional prediction is pre-stated on
the headline.**

### 0.1 The birth discipline (the supervisor's composition rulings, as
carried from the two engines' own registrations)

* **The K11B gates born at the registered scalar equivalents** —
  bit-equal at birth through the committed `channel_decay` wrapper (the
  falsifier-fixed construction, receipt `57e1f942`: the clamp bounds in
  the incumbent's carrier dtype): `lam_vec == lam_inc` per channel,
  `g_vec == 1` per channel, `r_vec == 1` per channel, exactly. At birth
  **the composition IS the K13 additive machine bit for bit** — this is
  the composition's own birth receipt (engine-side: every channel of the
  K15 forward launch vs K13's own committed kernel launch at identical
  inputs; probe-side: the step-1 train loss must be BIT-IDENTICAL to the
  K13 additive arm's own same-seed row, 5.782775391046648 at 3401 /
  5.720172563202926 at 3402, with the K11-receipted −0.0 IEEE corner of
  the unity-gate CORRECT blend the only legal escape — escalated, not
  excused, if it fires).
* **The additive read has no birth parity vs the BANKED winner** — a
  different function; the K13 decisions-bit-identical receipt stands (0
  flips in 315 M lanes: the read cannot feed op selection). The
  composition's step-1 is NOT banked-parity and **none is asserted** —
  this is a machine-variant arm like K13 was. The composition's
  decisions-bit-identical receipt runs against the K13 additive kernel
  itself (the composition's nearest kin), under random per-channel
  gate/decay values.

### 0.2 The parameter accounting

The composition's gate stack is **p3's own**: the per-channel DECAY
(`f_proj`, 106,498/mixer) + write (`b_proj`, 147,458/mixer) + read
(`g_proj`, 106,754/mixer — the SAME construction the K13 additive machine
already carries, **instantiated once, never double-added**) channels; the
additive read itself is parameter-free. At depth 24 (width 576, heads 2,
value_dim 128): **+8,657,040** parameters → **158,638,848** total,
**+5.77 %** over the audited 149,981,808 — the additive machine's own
152,543,904 plus exactly p3's decay+write channels (6,094,944). K11B's R4
precedent applies verbatim: the 0.1 % capacity tolerance is DROPPED for a
real GDN2-granular gate transplant (GDN2's own gate matrices are 18.3 % of
its 149.9 M; even the full composition at +5.77 % is ~3× leaner — the
asymmetry is the finding). The delta is computed by the committed k11b
accounting (`arm_parameter_delta("p3", ...)`), stated, asserted by the
probe, not hidden.

## 1. The architecture under test

**The K10 winner (per-token + separated keys, the tuned config) with the
K13 additive read under the K11B p3 gate stack** — at the matched training
surface (the banked winner lineage: hybrid Muon at 4× the tuned AdamW LR,
every gate projection on the AdamW side, the DECAY gate's own "a gate, not
a hidden map" convention). Same five-op schedule, same store, same
thresholds, same key channel, same projections, same recorded-schedule
discipline. Stated exactly once, the composed machine:

```
lam_vec = clamp(lam_inc·(1 + m_f·sigmoid(f_proj(x))), floor, 1-1e-6)   # per channel, born == lam_inc
g_vec   = 1 - m_w·sigmoid(b_proj(x))                                    # per channel, born == 1
score_j = read_scale · cos(q_key(t), K_j)        # the banked score channel, VERBATIM (R13-1)
w_j     = score_j  if slot j is live else 0       # raw score; NO softmax, NO ||S_j|| (R13-2)
y_raw   = Σ_j w_j · S[j]                          # in-kernel, the floor-channel rank sum
y       = r_vec ⊙ y_raw                           # the g_proj read gate, outside the scan
r_vec   = 1 + m_o·sigmoid(g_proj(x))               # zero-born, exact 1 at birth
```

with the per-channel DECAY applied to the values only (`S ← lam_vec ⊙ S`)
and the gated lane-0 arithmetic (APPEND admits `g_vec ⊙ p`, CORRECT blends
`g_vec ⊙ p + (1-g_vec) ⊙ (lam_vec ⊙ v_pre)` per channel — K11B's committed
pre-decay form). Everything else — decisions, the MERGE blend (with its
own detached strength weights), the key lane, POP/MERGE zeroing — is the
K10 winner VERBATIM through both engines' own commitments. Decisions remain
inviolable (P2): no gate value and no read form feeds op selection.

### 1.1 The engine composition (committed engines imported, never edited)

`k15_composition.py` composes the two committed engines; the ONLY new
kernel is the forward walk:

* **`_k15_seq_fwd_kernel`** — K11B's persistent whole-window walk with the
  read block replaced by K13's additive block (`w = where(live, score, 0)`;
  the softmax's exp/max/denominator GONE). Everything else is K11B's
  verbatim: the per-channel DECAY load, the gated lane-0 arithmetic, the
  2-lane recorded schedule, the per-lane-guarded MERGE-source close (the
  K12-flagged multi-lane corner of the K10/K11/K13 oracles is NOT copied —
  the composed oracle takes K11B's committed fix, so the oracle matches
  the kernel bit for bit on heterogeneous MERGE fixtures).
* **the backward composes the committed kernels VERBATIM**: K11B's
  per-channel trajectory replay (`channel_persistent_trajectory`), K10's
  key-traj replay (the K13 backward's own construction), **K13's additive
  prep** (`_k13_bwd_prep_kernel`: `dscores = dw` on live rows, the
  strength channel EXACT ZEROS), **K11B's per-channel value walk**
  (`_k11b_bwd_value_walk_kernel`: the per-channel DECAY and gate adjoints
  plus the read adjoint's value terms — the strength path dies by the
  prep's zero `dstrength` channel, IEEE add of 0.0 exact, the K13 receipt
  carried over), and K10's key walk. No committed kernel is edited.
* **the mixer** `ComposedReadMixer` — the committed `ChannelGatedSeparatedMixer`
  (arm p3) with `mix()` swapped to the composed scan: the committed
  `channel_decay` birth, the committed `apply_gate_birth`, the K11/K11B/K13
  read-out convention, all inherited VERBATIM by subclassing.

## 2. The protocol, the arms, the birth receipts

### 2.1 The arms and the protocol

**One arm — `composition` — at BOTH seeds (3401, 3402), both primary** (no
birth parity vs the banked winner exists; the seed-matched K13 additive
rows are the anchors). Every arm is the banked winner build VERBATIM
(`velocity_ladder._build_k10`, the tuned winner config, imported) then
`convert_model_to_composition`, at the matched-exposure probe protocol:
the 20,022,866-target landing, 180 constant-LR steps, the registered
flagship stream, the document-disjoint calibration eval, the honest clock —
then the CORRECTED scoring (the standing amendments, imported verbatim):
24-step linear cooldown PRIMARY, EMA-0.99 at step 180 SECONDARY
(ranking-only), constant-180 retained for banked comparability.

**The anchors, both quoted everywhere:**

* the composition's OWN read-form baseline — the K13 additive rows:
  post-cooldown **1.3928245476314** (seed 3401) / **1.384450146130153**
  (seed 3402); constant-180 1.439220769064767 / 1.435692889349801; EMA
  1.4283077546528407 / 1.437227657863072. **The composition must beat ITS
  OWN read-form baseline to count as composing.**
* the banked corrected anchor **1.4338248457227434** (seed 3402) and the
  banked constant rows (1.4907409463609969 / 1.4899048124040877), quoted
  for the ladder.
* **GDN2's post-cooldown 1.3521734476089478 — the gap reference: the
  composition's post-cooldown row vs 1.3522 is the HEADLINE** (the honest
  rematch: how much of the 0.0816-nat corrected gap remains after the two
  levers compose).
* the K11B p3 rows on the softmax read (post-cooldown 1.42353 / 1.42343)
  and the K13 mechanism rows (effective rows 16.42 / 16.69, top-row share
  0.156 / 0.154, identity 0.1254 / 0.1171, occupancy 26.21 / 26.53) — the
  mechanism anchors.

### 2.2 The birth receipts (the machine-variant's own, run BEFORE any
training on the first train batch)

1. **BORN AS THE ADDITIVE MACHINE**: layer for layer, at the arm's own
   inputs, the composed scan's every channel — schedule, payload lanes,
   read weights, reads, state, liveness, pre-decision liveness, key table —
   is `torch.equal` to the K13 additive kernel's own committed launch
   (the gates at the registered scalar equivalents). Probe-side: the
   step-1 train loss is BIT-IDENTICAL to the K13 additive arm's own
   same-seed row (quoted above) — the composition's nearest-kin parity,
   asserted loudly, escalated not excused if it breaks.
2. **DECISIONS-BIT-IDENTICAL**: the composed scan's recorded schedule is
   bit-identical to the K13 additive kernel's under random per-channel
   gate/decay values (no gate value feeds op selection; the read form
   cannot either).
3. **The gates born at the registered values**: `lam_vec == lam_inc` per
   channel bit for bit (through the production `channel_decay` expression
   under autocast, the falsifier's standing remedy), `g_vec == 1`,
   `r_vec == 1` per mixer on the first batch.
4. **No banked step-1 parity, disclosed**: the step-1 loss is a different
   function's value vs the banked row (5.785070835425993 at 3401 /
   5.768619428012168 at 3402) and none is asserted; receipts 1–3 carry the
   birth honesty instead.

### 2.3 The mechanism instrument, per arm, at BOTH states

The K13 instrument (the additive read's own channels) UNDER the K11B
mechanism instrument (the gates' own readout), at the step-180 constant
snapshot AND the post-cooldown final:

* **the K13 instrument**: |w|-mass effective rows (R13-4) — **the
  interesting instrument: does the per-channel decay gate stack change the
  additive read's mass distribution?** — with the top-row |w|-mass share
  (the collapse detector) and the mean live-row count; the birth-magnitude
  diagnostic is the birth receipt's own row (bit-equal to K13's at birth by
  construction, receipt 1);
* **the K11B instrument**: the identity-rewrite share through the
  per-channel trajectory replay, occupancy, op mix, and the gate telemetry
  for ALL THREE gate channels (per-layer mean |lam_vec − lam_inc|,
  |1 − g_vec|, |1 − r_vec| — the learned per-channel controller's own
  readout under the additive read);
* **the read-mask sweep** (`side_by_side.k10_read_mask_sweep`, imported
  verbatim) at both states.

### 2.4 The engine gates (pre-registration evidence, in scope now)

* **gate A**: the K15 kernel vs the composed oracle — every channel
  bit-exact (`torch.equal`), 0 flips, at the registered shape, BOTH gated
  combinations, bf16 autocast corners included; the read at the measured
  fp32-vs-fp64 floor; the signed-weight sub-receipt (the additive
  signature under the gates).
* **gate A2 (the composition's own)**: the BORN-AS-THE-ADDITIVE-MACHINE
  receipt (0 differing elements vs K13's own launch, both gated
  combinations), the decisions-bit-identical bar under random
  per-channel gates/decay, and the birth-magnitude/dilution rows.
* **gate B**: the **fp64 gradcheck of the oracle THROUGH THE COMPOSED PATH
  mandatory** in the suite (every adjoint branch: the additive read's
  direct term into the state, the score path into the query keys and the
  key table, the gated lanes, the per-channel DECAY vector and write
  gate); the kernel path's SEVEN gradients vs the measured
  fp32-vs-fp64 floors, `NOISE_FACTOR` 8×; the segment-recompute
  bit-equality on both channels (K11B's per-channel value replay + K10's
  key replay).
* **gate D**: block-call walls at the registered shape against K13's AND
  K11B's own same-session launches (the composition carries BOTH engines'
  costs: the dropped softmax and the per-channel vector loads).
* **the committed suites stay green with the module loaded**: K10, K13 and
  K11B's own suites re-run in the same session as the K15 commit (the
  K11B lane's own precedent).

## 3. The pre-stated predictions (with refutation clauses)

Stated BEFORE any probe measurement. **No directional prediction is
pre-stated on the headline** — both directions are live outcomes (the
gates' +0.010 was measured on the softmax read; the additive read sums
exactly the channels the decay gates modulate).

* **C-1 — THE COMPOSITION CLAUSE**: if the composition beats the
  additive-read baseline (its OWN read-form anchor, the K13 rows above)
  by **≥ 0.01 in the post-cooldown primary at BOTH seeds**, the two levers
  **COMPOSE**, the new best machine is the composition, and the honest
  rematch row is the composition's post-cooldown vs **GDN2's 1.3522** —
  the headline quoted with the fraction of the 0.0816-nat corrected gap
  closed (banked 1.4338 → composition) and the remaining gap to GDN2.
* **C-2 — THE READ-FORM-DEPENDENCE CLAUSE (parity)**: if the composition
  lands at parity with the additive baseline — both seeds' post-cooldown
  deltas inside the ±0.01 band around the K13 rows, or the seeds
  disagreeing across it — **the gates' +0.010 was READ-FORM-DEPENDENT**
  (the softmax read needed the channel gates; the additive read does not),
  receipted as the interaction finding, with the measured asymmetry
  reported as measured.
* **C-3 — THE INTERFERENCE CLAUSE**: if the composition is WORSE than the
  additive baseline by > 0.01 at BOTH seeds, **the gates interfere with
  the additive read** (the per-channel decay gates modulate exactly the
  channels the additive read now sums — the write admission scales the
  magnitudes the read no longer renormalises), receipted as the
  interference, with the mechanism instrument (the |w|-mass distribution
  under the gates, the gate drifts) attached as the inspection entry.
* **The adjudication rule (the K13 clause-conflict precedent, pre-stated)**:
  the specific pre-stated clause above governs the verdict; any band
  shapes the phrasing of the claim, never the experiment's adjudication.
  Mixed seeds (one clause at 3401, another at 3402) adjudicate to the
  CONSERVATIVE clause (the weaker of the two readings), reported with both
  rows next to each other, escalated per the task's rules.
* **M-1 (mechanism, no win/lose bar, reported per arm)**: the |w|-mass
  effective rows vs the K13 additive rows (16.42 / 16.69) — did the gates
  change the read mass distribution? — with the collapse detector, the
  identity share, occupancy, op mix, the three gate channels' drifts, and
  the read-mask sweep at both states.

**The refutation clause, stated verbatim before the arms:** if the
composition lands at parity or worse vs ITS OWN additive baseline (C-2 or
C-3), the two levers do not compose and that is reported as the finding,
not excused — the interaction (read-form-dependence or interference) is
receipted with the mechanism instrument attached, and the additive read
ALONE remains the best machine of the program.

## 4. Claim set (registered, pending sign-off)

1. **Probe scale only** (20 M-target matched exposure, 180+24 steps), one
   protocol, both seeds; no board-scale (2700 s) claim before the operator
   signs this document off.
2. Any NLL delta vs the K13 additive rows is seed-matched at both seeds;
   the composition-vs-GDN2 row is the headline and is quoted with the
   remaining gap; nothing inside the pre-stated ±0.01 band is banked as
   composing.
3. The engine claims (gates A/A2/B/D) are the lane's own receipts, in
   `receipts/k15_gates.json` + `K15_COMPOSITION.md`; the arms' receipts
   land per arm under `k15_runs/composition_seed<seed>/`, committed per
   artifact (the ceiling is real; a committed partial beats an
   uncommitted complete).
4. The card discipline: cards 0/1 (this lane's own pool); cards 2/3 are
   the signed X46 sweep's, card 4 the detached emender run — untouched;
   cards 5/6 left as buffer; card 7 forbidden.

## 5. Deliverables

| file | role |
| --- | --- |
| `PREREGISTRATION_COMPOSITION.md` | this DRAFT (committed before any probe measurement) |
| `k15_composition.py` | the composed oracle, the ONE new forward kernel, the composed backward body (committed kernels imported verbatim), the op pair, the mixer and its conversion |
| `test_k15_composition.py` | gates A and B (the fp64 gradcheck through the composed path mandatory), the composition's birth receipts' fixtures, the carrier-corner remedy, refusals, determinism |
| `k15_gate_receipt.py` | the gate receipts into `receipts/k15_gates.json` |
| `k15_probe_composition.py` | the arm runner + the assemble (the C-1/C-2/C-3 adjudication across seeds) |
| `run_k15_composition_leased.sh` | the detached leased driver (cards 0/1) |
| `K15_COMPOSITION.md` / `RESULT_COMPOSITION.md` | the engine document / the arms' result document |

## 6. Commands

```bash
# the suite (the fp64 gradcheck through the composed path is mandatory):
/home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k15_composition.py -x
# the committed suites stay green with the module loaded (the K11B precedent):
/home/erikg/emender/.venv/bin/python -c \
  "import experiments.expressivity_tasks.x45_matrix_emender.k15_composition, pytest, sys; \
   sys.exit(pytest.main(['-q', \
  'experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py', \
  'experiments/expressivity_tasks/x45_matrix_emender/test_k13_additive_read.py', \
  'experiments/expressivity_tasks/x45_matrix_emender/test_k11b_channel_gates.py']))"
# the gate receipts:
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)" && \
CUDA_VISIBLE_DEVICES=$GPU_LEASE_HELD OMP_NUM_THREADS=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k15_gate_receipt
# one arm (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k15_composition_leased.sh composition 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k15_runs/\
composition_seed3401.console.txt 2>&1 < /dev/null &
# the assemble:
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.\
x45_matrix_emender.k15_probe_composition assemble
```
