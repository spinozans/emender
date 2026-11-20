# RESULT — K13 THE ADDITIVE READ: the read-dilution finding's owner found — the C5 attribution refines to THE READ FORM, the terminal reading amended

**Date:** 2026-11-24 · **Status: complete — both arms landed. THE HEADLINE: the additive read (GDN2's form, transplanted into the banked K10 winner at the matched protocol) is AHEAD at every scoring, both seeds — post-cooldown 1.3928/1.3845 vs the corrected anchor 1.4338 (−0.0410/−0.0494, clearing the pre-stated 1.4138 close-bar at both seeds; the gap to GDN2's 1.3522 shrinks from 0.0816 to 0.0323–0.0406, 60.5% closed at the primary row), constant-180 −0.0515/−0.0542, EMA −0.112/−0.104 — and the pre-registered read-dilution instrument MOVED exactly as the read-form hypothesis predicted (|w|-mass effective rows 16.4/16.7 = 3.6x/3.4x the seed-matched banked 4.55/4.95, no collapse). Under the supervisor's ruling on the pre-stated clause conflict (quoted in §2), the C5 attribution REFINES to THE READ FORM (competitive softmax dilution), the terminal reading is amended, and the read form joins the leaner registration. The band caveat is stated verbatim in §2: the single-protocol primary delta alone sits inside the 0.075 endpoint-variance band and would not bank — the amendment rests on the clause structure plus the cross-scoring and two-seed coherence plus the mechanism receipt.** ·
**Authorisation:** the operator's K13 task (the additive read, the follow-up to the C5 read-dilution finding), with the supervisor's pre-build rulings R13-1..R13-5 (the score form, the strength drop, the g_proj gate, the |w|-mass instrument, the REQUIRED birth-magnitude diagnostic) and the post-arms ruling on the M-R2 clause conflict. The registration is the DRAFT `PREREGISTRATION_ADDITIVE_READ.md`, committed BEFORE any probe measurement (the K11B lesson); its predictions are evaluated below verbatim. The engine, its gates and the suite are `K13_ADDITIVE_READ.md` + `receipts/k13_gates.json` (all green; the fp64 gradcheck through the additive path mandatory and passed).

## 0. The protocol, as run

Every arm is the banked winner build VERBATIM (`velocity_ladder._build_k10`, the tuned winner config, the banked bake-off build — imported, not rebuilt) then `convert_model_to_additive_read` (the K13 conversion: the K10 mixer's own parameters shared object-for-object, the read swapped to the additive form per the registration, the committed k11b `g_proj` gate added zero-born at exact 1), at the matched-exposure probe protocol (the 20,022,866-target landing, 180 constant-LR steps, the registered flagship stream, the document-disjoint calibration eval at 7 windows / 14,336 targets, the honest clock), under the corrected scoring: the 24-step linear cooldown PRIMARY, EMA-0.99 at step 180 SECONDARY (ranking-only), constant-180 retained for banked comparability.

**The machine-variant's birth receipts, green at probe scale (no birth parity is possible — a different function — and none was asserted; the registration's 2.2 receipts carried the honesty instead):**

* **DECISIONS-BIT-IDENTICAL**: 0 flips in **315,408,384 lanes** — the additive scan's recorded schedule is `torch.equal` to the BANKED read's own committed kernel launch's at identical inputs, every layer, on the first training batch (the read form cannot feed op selection). The conversion's parameter sharing is asserted bit-for-bit against a fresh twin build first.
* **The birth-magnitude diagnostic (R13-5)**: per-head read-output RMS additive/banked = **1.81x/1.86x** at birth (the near-empty store; the gate fixture's fuller store receipts 2.47x/2.52x) — the honest ~O(live-rows)x consequence of removing the simplex, stated not guessed. The step-1 losses landed 5.7828 (3401) / 5.7202 (3402) vs the banked rows' 5.7851/5.7686 — a different function's values, disclosed, never asserted.
* **The gate born at exact 1**: `r_vec == 1` bit for bit at the production expression under autocast, every layer.

## 1. The corrected ladder

| arm (the additive read) | constant-180 | **post-cooldown (PRIMARY)** | EMA-180 (secondary) | Δ vs anchor (PRIMARY) |
|---|---:|---:|---:|---:|
| **additive, seed 3401** | 1.4392 | **1.3928** | 1.4283 | **−0.0410** |
| **additive, seed 3402** | 1.4357 | **1.3845** | 1.4372 | **−0.0494** |
| banked winner (3401) | **1.4907** | — | — | — |
| banked winner (3402) | **1.4899** | **1.4338** | 1.5408 | — |
| banked GDN2 (context) | 1.3951 | **1.3522** | 1.3803 | — |

* **The primary read (seed 3402, the corrected anchor's own seed):** post-cooldown **1.3845** vs **1.4338** — **−0.0494**, clearing the pre-stated 1.4138 close-bar by 2.5x the bar; **60.5% of the 0.0816 corrected gap to GDN2 closed** (the residual to GDN2: 0.0323). Constant-180 **1.4357** vs the banked **1.4899** (−0.0542, 4.5x the winner's own measured seed delta 0.0119). EMA **1.4372** vs **1.5408** (−0.1036 — outside the 0.075 band, ranking-only).
* **The second seed agrees:** post-cooldown 1.3928 (−0.0410 vs the cross-seed anchor; −0.0515 on constant-180; −0.1125 on EMA). The additive arm's own seed delta **0.0084** — inside the winner's own 0.0119.
* **The honest clock**: the arms trained **FASTER** than the banked rows — 1707.5 s / 1719.1 s constant phase vs the banked 1904/1906 (11,726 / 11,647 bytes/s) — the softmax's exp/max/denominator are gone from the walk, ~10% cheaper for free (gate D: forward 16.43 ms vs K10's banked 19.7 per scan).

## 2. The verdict — the pre-stated clauses, applied under the supervisor's ruling

**The M-R2 clause conflict, escalated and ruled (the task's pre-authorized escalation):** the primary deltas (−0.0410/−0.0494) BOTH clear the registration's specific win bar (arm ≤ 1.4138; "closes ≥ 0.02 of the corrected gap") AND sit numerically inside the 0.075-nat endpoint-variance band that the claim set names as the bankability bar. The supervisor's ruling, quoted for the record:

> Clause A governs the EXPERIMENT's verdict; the band governs the MACHINE-RANKING claim's phrasing, not this experiment's adjudication. State both, precisely separated. A pre-stated specific win bar is the instrument that defines what counts as a win for ITS hypothesis; the 0.075 endpoint-variance band is the protocol-level guard for cross-machine best-vs-best rankings with search noise. Applying the band mechanically to a pre-stated clause that cleared at both seeds — with the constant-180 deltas at 4.5x the winner's own seed delta, the EMA deltas OUTSIDE the band at both seeds, the two seeds agreeing within 0.0084, all three scorings pointing the same way, and the pre-registered mechanism instrument (M-R1) firing exactly as the read-form hypothesis predicted with no collapse — would be band-worship, not discipline. The clause structure + coherence + mechanism is the evidence; the single primary number is not asked to carry it alone.

**THE VERDICT: M-R2's close clause is MET at both seeds. The C5 attribution REFINES to THE READ FORM — competitive softmax dilution.** The residual decomposes:

> The READ FORM (competitive softmax dilution) owned **~0.04–0.05** of the original ~0.08 corrected gap — now closed by the additive read at both seeds, seed-coherently, mechanism-receipted. The terminal reading amends from *"the dense superposition itself owns ~0.07"* to *"the read law owned ~0.04–0.05 (closed, seed-coherent, mechanism-receipted); the residual is ~0.03–0.04, with the write form and the management form as the leading remaining candidates — K14 in flight tests the latter."*

**The read form JOINS the leaner registration** (the additive read is now banked structure in this lane's lineage, pending the operator's sign-off of the DRAFT registration).

**The band caveat, stated verbatim as ruled:** *the single-protocol primary delta (−0.041/−0.049) alone sits inside the 0.075 endpoint-variance band and would not bank; the amendment rests on the clause structure plus cross-scoring and two-seed coherence plus M-R1; a board-scale or second-protocol confirmation banks the machine-level ranking claim.* Nothing in this document claims the additive-read machine has beaten the banked winner machine at board scale — this document's claim is the pre-registered hypothesis-level adjudication at probe scale, both seeds, stated with its evidence class.

## 3. M-R1 — the read-dilution instrument MOVES: MET at both seeds, no collapse

| | seed 3401 | seed 3402 |
|---|---:|---:|
| \|w\|-mass effective rows (post-cooldown state) | **16.42** | **16.69** |
| seed-matched banked anchor (e^H of its own entropy) | 4.55 | 4.95 |
| **factor** | **3.61x** | **3.37x** |
| the pre-stated bar | ≥ 2x | ≥ 2x |
| top-row \|w\|-mass share (the collapse detector) | 0.156 | 0.154 |

The competitive→additive signature fired exactly as pre-stated: nothing competes for a simplex, and the read's mass spread from ~4.6–5 banked rows to ~16.4–16.7 — toward the live-row count (the occupancy is 26.2/26.5 live rows at the same state), with the top row holding only ~15% of the |w| mass. **The refutation clause does NOT fire**: the additive read does NOT collapse to a few dominant rows. The step-180 states agree (18.12/16.69 effective rows).

## 4. M-R3 — the mechanism instrument, reported per arm (no bar)

* **Identity-rewrite share**: 0.1254 / 0.1171 post-cooldown (banked same-config: 0.1222 / 0.1180) — unchanged within noise; the write-side identity regime is the read form's own, as it should be (decisions are inviolable and the decision chain is untouched).
* **Occupancy**: **26.21 / 26.53 mean live slots — BELOW the banked 33.49 / 35.15.** Reported as measured, no bar pre-stated: the additive arm runs a leaner live set at the same op mix (CORRECT 0.980/0.980 per token, APPEND 0.0175/0.0175, MERGE 0.0021/0.0023, POP 0.0000 — the banked mix within noise).
* **The g_proj gate**: opened **0.0109 / 0.0127** mean distance from its born identity (per-layer range up to ~0.03) — the gate is live and mildly used; the arm's own verdict is that the raw additive read needs little output gating at probe scale.
* **The decorative-store sweep**: load-bearing layers (ΔNLL ≥ 0.02 when one layer's read is masked) = **4 of 24** (3401) / **5 of 24** (3402) — FEWER than the banked-era anchor of 7, not more. Stated as measured: **the additive read CONCENTRATES load-bearing reads rather than broadening them** (the top masked-layer deltas: 0.073/0.071/0.039/0.033 at 3401; 0.126/0.069/0.035/0.026 at 3402), while the all-layers-masked floor stays high (1.60 / 1.56 above base) — the stores still carry most of the model. The pre-stated "may make more stores load-bearing" direction is NOT confirmed; the concentration direction is the receipt.
* **The refutation clause does not fire anywhere**: mechanism numbers moved (the instrument, the occupancy), and the NLL improved beyond parity — a coherent positive, not a dissociation.

## 5. The engine receipts (the lane's own, all green — `receipts/k13_gates.json` + `K13_ADDITIVE_READ.md`)

Gate A bit-parity 0 flips in 17,336,320 lanes vs the additive oracle at the registered shape (read at 1.14x its measured floor; 255 registers / 2 spills); the signed-weight receipt (1.34M negative entries, gain-bounded on live rows, exact zero on dead, NOT a simplex); the fp64 gradcheck THROUGH the additive path OK; the six gradients at **≤ 1.0x** the measured floor (several exactly 0.0 — the adjoint without the softmax jacobian is measurably cleaner); the segment recompute bit-equal through the imported replays; gate D **cheaper than the banked read** (16.43 vs 19.7 ms forward). The suite: 19 tests green, including the DECISIONS-BIT-IDENTICAL receipt against K10's own committed kernel and the strength-path-gone receipt. The parameter delta: **+2,562,096 (+1.70%)**, the k11b R4 precedent's own read channel — the read-form change itself parameter-free.

## 6. Honest scope, walls, and the named next arm

* **Probe scale only** (20 M-target matched exposure, 180+24 steps), one protocol, both seeds; the machine-level ranking claim needs a board-scale or second-protocol confirmation (§2's caveat, verbatim).
* The arm ADDS parameters (+1.70%, the k11b R4 precedent); the comparison is the clause structure's adjudication, not a parameter-matched machine ranking.
* The occupancy dip (26.2/26.5 vs banked 33.5–35.2) and the sweep's concentration (4–5 load-bearing of 24 vs the anchor 7) are reported as measured with no pre-stated bar — named as instruments for the follow-ups, not explained post-hoc here.
* **The named next arm (named, not run):** K14's arms (in flight) run on the BANKED (softmax) read, so their routing verdict is on the old read law. The composition — **the additive read + the learned router, and whether the p3 gate stack composes with the additive read** — is the natural next arm once both land. Not run in this lane.
* The birth-magnitude diagnostic is attached to every early-divergence reading: at birth the additive read is ~1.8–2.5x the banked read's RMS (near-empty to gate-fixture stores), and the first training steps adapt the downstream scale — no divergence occurred (the step-1 losses are within 0.003/0.048 of the banked rows' own, and the training is ~10% faster).

## 7. Commands

```bash
# the arms (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k13_additive_read_leased.sh additive 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k13_runs/\
additive_seed3401.console.txt 2>&1 < /dev/null &
# the assemble (the ladder + the M-R1..M-R3 adjudication):
.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.\
k13_probe_additive_read assemble
```
