# RESULT — K15 THE COMPOSITION: the two proven-positive levers, composed

**Date:** 2026-11-24 · **The registration:** `PREREGISTRATION_COMPOSITION.md`
(DRAFT, committed before any probe measurement). **Receipts:** every number
below is from `k15_runs/composition_seed{3401,3402}/result.json`,
`receipts/k15_composition_summary.json` (the assemble) and
`receipts/k15_gates.json` (the engine gates). **Verdict adjudicated under
the supervisor's ruling of 2026-11-24** (the pre-stated conservative
mixed-seed rule governs; the ruling is quoted in §5).

**The question:** do the session's two proven-positive quality levers
COMPOSE — the K13 additive read (the banked score channel verbatim, softmax
removed, strength dropped, the g_proj gate zero-born) and the K11B p3
per-channel decay/write/read gate stack (born at the registered scalar
equivalents)? No directional prediction was pre-stated; the C-1/C-2/C-3
clauses were.

## 1. The arms' rows (matched-exposure probe, corrected scoring)

| arm (post-cooldown PRIMARY) | seed 3401 | seed 3402 |
| --- | ---: | ---: |
| **the composition** (K13 read + p3 gates) | **1.3811104467936925** | **1.383232252938407** |
| its OWN baseline — the K13 additive arm | 1.3928245476314 | 1.384450146130153 |
| **improvement vs own baseline** | **+0.01171410083770752** | **+0.0012178931917461444** |
| the composition, constant-180 (banked-comparable) | 1.433674931526184 | 1.4341776371002197 |
| the K13 additive arm, constant-180 | 1.439220769064767 | 1.435692889349801 |
| the composition, EMA-180 (ranking-only) | 1.424389226096017 | 1.4385861498968942 |
| the K11B p3 arm on the SOFTMAX read | 1.4235332352774483 | 1.4234301022128885 |
| the banked winner (corrected anchor, seed 3402) | — | 1.4338248457227434 |
| **GDN2 (the gap reference)** | — | **1.3521734476089478** |

**The headline (the honest rematch):** the composition's post-cooldown rows
are **+0.028937 (seed 3401) / +0.031059 (seed 3402) remaining vs GDN2's
1.3522**, closing **64.6% / 62.0%** of the 0.0816-nat corrected gap
(banked 1.4338 → GDN2 1.3522) — the program's best absolute rows at both
seeds.

**The clause adjudication (the registration's 3, ruled in §5):**
C-1 (the compose bar, ≥ +0.01 at BOTH seeds) **did not replicate**:
+0.0117 at 3401 clears it, +0.0012 at 3402 is deep inside the winner's own
seed delta (0.0119) — **C-2 PARITY (read-form-dependence) is the verdict
under the pre-stated conservative mixed-seed rule**. C-3 (interference)
did not fire: the composition is BETTER than its own baseline at both
seeds.

## 2. The birth receipts — the composition is born as the additive machine

The session's cleanest composition receipt, at probe scale on the first
train batch, before any training:

* **BORN-AS-THE-ADDITIVE-MACHINE**: the composed scan's every channel is
  bit-identical to the K13 additive kernel's own committed launch, layer
  for layer — **0 flips in 516,734,976 lanes**, the gates born at the
  registered scalar equivalents through the production `channel_decay`
  expression under autocast (all 24 layers: `lam_vec == lam_inc` per
  channel bit for bit, `g_vec == 1`, `r_vec == 1`).
* **The nearest-kin step-1 parity, asserted and held at BOTH seeds**: the
  composition's step-1 train loss is **BIT-IDENTICAL to the K13 additive
  arm's own same-seed row** — 5.782775391046648 (3401) /
  5.720172563202926 (3402). The banked winner's step-1 rows
  (5.785070835425993 / 5.768619428012168) are a DIFFERENT function's
  values — quoted, never asserted.
* Engine-side (the gate receipts): 0 differing elements in 43,061,248 lanes
  vs K13's kernel at the registered shape; the fp64 gradcheck THROUGH THE
  COMPOSED PATH green; all seven gradients ≤ 1.5× the measured floors
  (`K15_COMPOSITION.md` §2).

## 3. The mechanism instrument (M-1, reported as measured, no bar)

| instrument (post-cooldown state) | seed 3401 | seed 3402 | the K13 additive rows |
| --- | ---: | ---: | ---: |
| \|w\|-mass effective rows (R13-4) | 16.83 | 15.07 | 16.42 / 16.69 |
| top-row \|w\|-mass share (collapse detector) | 0.155 | 0.175 | 0.156 / 0.154 |
| identity-rewrite share | 0.1266 | 0.1220 | 0.1254 / 0.1171 |
| occupancy (mean live slots) | 27.20 | 25.12 | 26.21 / 26.53 |
| decorative-store sweep: layers ≥ 0.02 dNLL | 4 | 5 | (K13's own: 4–5 of 24) |
| gate drift — decay (\|lam_vec − lam_inc\|) | 0.0107 | 0.0109 | (K11B p3: 0.0069 / 0.0075) |
| gate drift — write (\|1 − g_vec\|) | 0.0233 | 0.0280 | (K11B p3's own scale) |
| gate drift — read (\|1 − r_vec\|) | 0.0113 | 0.0116 | (K13's g_proj: 0.0109 / 0.0127) |

**The interesting instrument's answer: the gates did NOT change the
additive read's |w|-mass distribution** (16.83/15.07 vs the additive
machine's own 16.42/16.69 — the same instrument, the same regime, no
collapse: the top-row share is unchanged). All three gate channels WOKE and
moved under the additive read (drifts at or above their K11B softmax-arm
scales — the per-channel controller is live), and the identity share,
occupancy and op mix are the additive machine's own within noise. The
mechanism moved where the gates act (the write/decay channels) and did not
move where the read acts.

## 4. The finding, stated per the supervisor's ruling

**(a) What banks the program's best machine.** The composition lands as
the program's best absolute machine at both seeds (1.3811 / 1.3832 — better
than the K13 additive rows, the K11B p3 softmax rows and the banked
winner), with ~63% of the corrected gap closed and +0.029/+0.031 remaining
vs GDN2. But the load-bearing, two-seed, mechanism-receipted causal result
is **the ADDITIVE READ's** (K13: −0.041/−0.049, effective rows 3.6×, no
collapse); the gates ride at **parity-grade evidence** — better absolute
rows at both seeds, no interference (C-3 did not fire), a single-seed
+0.012 improvement at 3401 reported as measured and NOT banked as
composing (the pre-registered bar needed both seeds).

**(b) The read-form-dependence finding (a real interaction receipt, not a
failure to measure).** The gates' +0.0103/+0.0104 was measured on the
COMPETITIVE (softmax) read; under the additive read it shrinks to seed
noise (+0.0117/+0.0012). The honest reading: the channel gates and the
read law draw from the same well — both loosen the read/write interface's
rigidity (the softmax's simplex was the thing the per-channel read/write
gates had to compensate for; remove the simplex and the compensation is
already bought) — so the two levers **partially substitute rather than
add**. That is the pre-named C-2 clause's content, receipted with the
mechanism instrument attached: the gates act (all three channels drift),
the read's mass distribution is unmoved, and the NLL composes to parity.

**(c) The named follow-ups for the remaining ~0.03 (mandatory before any
final gap claim, per the config-symmetry discipline; neither runs in this
lane):**
1. **an LR/optimizer bracket ON THE COMPOSED MACHINE** — the composed
   machine has NEVER been tuned: it inherits the banked winner's 4×-Muon
   point, tuned for the softmax machine, while the read law changed the
   machine substantially; the remaining gap may partially yield to config
   search, and the symmetric-claim rule requires our best-vs-best
   (ours searched, GDN2 searched) to be search-symmetric;
2. **the write form (event writes vs dense superposition)** — named as the
   LAST structural candidate owner of whatever remains after (1).

## 5. The adjudication, quoted from the supervisor's ruling (2026-11-24)

> Ruling: APPROVED as proposed — C-2 per the pre-stated conservative
> mixed-seed rule. [...] 3402's +0.0012 is deep inside the winner's own
> seed delta, and banking C-1 on one seed would be the kind of
> over-reading this program does not do. [The best-machine claim is
> carried by] the ADDITIVE READ's two-seed receipted gain [...], with the
> gates INCLUDED at parity-grade evidence [...]. The gates' +0.010 was
> measured on the competitive read; under the additive read it shrinks to
> seed noise — i.e., the channel gates and the read law draw from the same
> well [...], so they partially substitute rather than add. That's a real
> interaction receipt, not a failure to measure. [Follow-ups:] an
> LR/optimizer bracket ON THE COMPOSED MACHINE [... the symmetric-claim
> rule requires our best-vs-best to be search-symmetric]; the write form
> (event writes vs dense) named as the LAST structural candidate owner of
> whatever remains after.

## 6. The protocol and provenance

Both arms: the banked winner build verbatim (`velocity_ladder._build_k10`,
the tuned winner config) + `convert_model_to_composition` (the K15
conversion; the gates born at the registered scalar equivalents), the
20,022,866-target matched-exposure landing at 180 constant-LR steps on the
registered flagship stream, the hybrid Muon 4× lineage with the p3 gate
tensors on the AdamW side (cross-checked against the banked split,
name-for-name), THEN the corrected scoring: the 24-step linear cooldown
PRIMARY (the post-cooldown calibration eval), EMA-0.99 at step 180
SECONDARY (ranking-only), constant-180 retained for banked comparability.
Parameters: 149,981,808 + 8,657,040 (p3's own gate stack — the read
channel instantiated once) = **158,638,848**, disclosed and asserted.
Training walls: 1920.8 s / 1924.2 s constant-phase (~13% slower than the
K13 arms' 1707/1719 — the per-channel vector loads, gate D's own
prediction). Both seeds ran detached-leased on this lane's own card pool
(0/1), committed per arm. No `k10_*`/`k11*`/`k12_*`/`k13_*` source, no
committed PREREGISTRATION, no `lattice.py` was touched.

**Scope caveats, standing:** probe scale only (no board-scale claim before
the operator signs the registration off); single-protocol rows; the
endpoint-variance band (0.075) and the winner's seed delta (0.0119) are
quoted with every delta; the composition-vs-GDN2 row is a cross-machine
context row at the same probe protocol, not a controlled comparison.
