# RESULT_GDN2_LR_BRACKET.md — GDN2's LR bracket: the last config-symmetry arm, and the config-symmetry program's final verdict

**Date:** 2026-11-21 · **Status: complete — the config search is now symmetric on BOTH axes (routing and LR) on BOTH machines, and the symmetric best-vs-best is settled: GDN2 AHEAD by 0.0949, outside the endpoint-variance band. The residual gap is structural — C5, the write/read form — exactly what K11 is attacking.** ·
**Authorisation:** the operator's GDN2-LR-bracket directive — the routing-symmetry lane gave OUR winner an LR bracket and found 4x beats its inherited 3x point (1.4899/1.4907 vs 1.5146/1.5027, two seeds), but GDN2's best arm (option B, gates-on-AdamW, 1.4151) still sat at its inherited 3x point, unbracketed. The symmetric claim rule (no 'leads'/'behind' except best-vs-best with the config search symmetric) had exactly one asymmetry left — in GDN2's favor — and this lane closes it: GDN2 gets the same upward bracket ours got, at its best routing, on the muon_control driver's conventions verbatim.

## 0. The verdict, stated first

**GDN2's optimum moves up too — by the same small order ours did — and the
symmetric best-vs-best lands where the matched-point reads always pointed:
GDN2 AHEAD.** At its best routing (option B, gates-on-AdamW), GDN2's LR
bracket lands **4x = 1.3992** and **5x = 1.3951** (seed 3401) against the
banked inherited 3x row's **1.4151** — both below it by more than the
0.0119 seed band (−0.0159 / −0.0200), sign-stable at the second seed
(5x at 3402: **1.4049**, −0.0102, inside the band but the same sign), and
**5x vs 4x differs by only 0.0041 — the bracket is flattening: GDN2's
optimum saturates in the 4x–5x region**, at the same small distance above
its inherited point that ours did (our gain 0.013–0.024, GDN2's
0.016–0.020).

| the bracket table (GDN2, option B routing, the exact probe protocol, 20,022,866 targets) | final NLL | vs the banked 3x row 1.4151 |
|---|---:|---:|
| GDN2 + Muon-3x, option B (the control's arm, banked — the inherited point) | 1.4151 | — |
| **GDN2 + Muon-4x, option B** (seed 3401) | **1.3992** | **−0.0159** |
| **GDN2 + Muon-5x, option B** (seed 3401) | **1.3951** | **−0.0200** |
| GDN2 + Muon-5x, option B (seed 3402 — the clear new best owes the second seed, pre-stated rule) | 1.4049 | −0.0102 |
| *(reference: GDN2 option A 3x 1.4702 · GDN2's AdamW 1.5837 · our best 1.4899 at 4x, B-analog routing, seed 3402)* | | |

**The new symmetric best-vs-best — the config search now symmetric on both
axes, both machines:** our best measured surface (the B-analog routing at
4x, seed 3402: **1.4899**) vs GDN2's best measured surface (the option-B
routing at 5x, seed 3401: **1.3951**) — **margin 0.0949, GDN2 AHEAD,
outside the ~0.075-nat endpoint-variance band.** The matched-point reads
agree everywhere: at the matched 3x point 1.5027 vs 1.4151 (GDN2 +0.0876);
at the matched 4x point 1.4899 vs 1.3992 (GDN2 +0.0907); same-seed 3401
4x-vs-4x 1.4907 vs 1.3992 (+0.0915). **The parity-within-band caveat the
routing lane owed (0.0748, asymmetric in OUR favor because GDN2's LR axis
was unmeasured) is now resolved honestly: GDN2's bracket mirrors ours,
buys it the same ~0.02, and the gap re-opens to ~0.09 — outside the band,
in GDN2's favor, at every matched or best-vs-best comparison.**

**The config-symmetry program is complete, and its final verdict: no
surface lever on either machine — routing (both machines bracketed: ours
+0.016 inside the band, GDN2's 0.055 bought by its option B), or the LR
point (both machines bracketed: ours −0.013/−0.024 at 4x, GDN2's
−0.016/−0.020 at 4x–5x) — inverts or closes the machine ranking. What
remains, ~0.09 at best-vs-best, sign-stable at every matched point
measured, is structural: the C5 write/read form — GDN2's 27.4M learned
gate-matrix params against our 27.7K decay pair and three hyperparameter
thresholds — exactly the lever the K11 LEARNED-GATES pre-registration
targets, running now. The gap is owned by C5.**

## 1. The arms and the conventions (the muon_control driver's, verbatim)

Every arm runs the EXACT probe protocol verbatim: the 20,000,000-target
matched-exposure rule (the 20,022,866 landing, 180 steps, every arm
identical), seed 3401 first (the second seed is the pre-stated rule's own
call), the registered flagship stream (identity asserted), the
document-disjoint calibration eval at 7 windows / 14,336 targets
(recorded-read path), the honest clock, the protocol's own grad-clip (1.0,
error_if_nonfinite), and the bake-off's own checkpoint plan (steps
1/5/10/20/40/80/120/160/180; CPU state snapshots inside the loop, every
eval on restored snapshots after training — the training path unperturbed
by construction). GDN2's model code is UNTOUCHED: `matched_arms.build_gdn2`
(the registered lattice_gdn2 arm, its searched nomination's knobs asserted
at launch against the banked row), the control's own
`split_parameters_gdn2` (option B: the gate-producing projections
b/w_proj, f_proj.0/1, g_proj.0/1 on AdamW — 191 Muon tensors /
122,255,360 params, the receipted counts), the control's own
`_train_with_snapshots_gdn2_hybrid`, and `bakeoff_optimizers.HybridMuonX45`
are imported VERBATIM. No prior-lane source, PREREGISTRATION or
`lattice.py` was edited; all new code is `gdn2_lr_bracket.py` + its leased
launcher + tests.

**The LR convention:** the bake-off's own, applied optimizer-for-optimizer
and bracketed upward the way our machine's bracket went — 4x and 5x GDN2's
OWN banked AdamW LR (5.4201855e-4): **4x = 2.1680742e-3, 5x =
2.7100927e-3** — with GDN2's own betas (0.9, 0.9392), weight decay
(0.20059), warmup 69 on BOTH hybrid groups (the prior line's
both-groups convention), microbatch 8, all asserted at launch. The ONLY
difference vs the banked option-B arm is the peak Muon LR.

**The pre-stated second-seed rule** (receipted in every arm's result.json,
applied exactly as stated): an arm below the banked option-B final
(1.4151) by MORE than the winner's measured seed band (0.0119) is a CLEAR
new best and owes a second seed; below by less: parity-within-band, the
operator's call; neither arm beats 1.4151: no second seed owed. **Both
bracket arms cleared the bar at seed 3401; the best-of-table (5x) paid
the second seed (3402).**

**Protocol-identical starts, receipted:** every seed-3401 arm's step-1
train loss is bit-identical to the banked gdn2 probe's own first row
(5.948033220866492) — asserted in-code. The seed-3402 arm has NO banked
same-seed GDN2 row to assert against (none exists beyond 3401), and its
receipt says so honestly (`step1_reference: null`, the note recorded)
rather than silently passing; its step-1 loss (5.858114336923005) is the
seed's own init draw, as the winner's seed-3402 rows were.

**Receipts:** `gdn2_lr_bracket_runs/<label>_seed<seed>/` (each: `start.json`
with the full convention block and the routing inventory, `training.jsonl`
live per-step, `checkpoints.jsonl`, `result.json` with the curve, walls,
GDN2's own gate telemetry at the final checkpoint — the
mechanism-unchanged receipt — and the Muon update RMS), the assemble
summary `receipts/gdn2_lr_bracket_summary.json`, and the lane's sources
(`gdn2_lr_bracket.py`, `run_gdn2_lr_bracket_leased.sh`,
`tests/test_x45_gdn2_lr_bracket.py`). Session lambda01, cards 2/3 (the
lane's pool; cards 0/1 — the K11 lane's — untouched), RTX 6000 Ada, torch
2.9.1+cu128. Committed per arm as each landed.

## 2. The overlaid curves (THE artifact)

Calibration eval NLL (nats/byte, 7 windows / 14,336 targets) at matched
checkpoints — this lane's arms against the banked curves (the control's
GDN2 rows and our machine's bracket rows, cited, unmodified).

| step | MB | gdn2 AdamW (banked) | gdn2 B 3x (banked) | **gdn2 B 4x** | **gdn2 B 5x** | gdn2 B 5x s3402 | our B-analog 3x s3402 (banked) | our 4x s3402 (banked) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.7045 | 5.8977 | 5.8882 | 5.8786 | 5.8423 | 5.7726 | 5.7759 |
| 5 | 0.57 | 3.8999 | 5.4941 | 5.3533 | 5.2153 | 5.1891 | 5.5616 | 5.4873 |
| 10 | 1.19 | 3.1024 | 4.3434 | 4.0016 | 3.8159 | 3.7744 | 4.9560 | 4.7356 |
| 20 | 2.37 | 2.6675 | 2.9845 | 2.7815 | 2.6485 | 2.6756 | 3.8099 | 3.5657 |
| 40 | 4.54 | 2.2309 | 2.0999 | 2.0095 | 1.9543 | 1.9710 | 2.7592 | 2.6298 |
| 80 | 8.88 | 1.8528 | 1.5686 | 1.5369 | 1.5374 | 1.5267 | 1.9244 | 1.8327 |
| 120 | 13.28 | 1.6290 | 1.4150 | 1.4252 | 1.4149 | 1.4338 | 1.6236 | 1.5761 |
| 160 | 17.72 | 1.5056 | 1.3638 | 1.3778 | 1.3688 | 1.3637 | 1.4921 | 1.4704 |
| 180 | 20.02 | 1.5837 | 1.4151 | **1.3992** | **1.3951** | 1.4049 | 1.5027 | 1.4899 |

**What the curves say:**

* **The bracket moves GDN2's early descent faster and its endpoint lower —
  but its steady read only slightly.** The 4x/5x arms leave init faster
  than the 3x row at every early checkpoint (step 10: 4.0016/3.8159 vs
  4.3434) and cross GDN2's own AdamW at step 40 (1.95–2.01 vs 2.2309),
  while the 3x row crossed at step 40 too. The steady deficits vs GDN2's
  AdamW (steps 80–160 means): 3x −0.213, 4x −0.216, 5x −0.222/−0.221 —
  the bracket's steady gain is −0.003 to −0.009, inside the noise; the
  endpoint gain (−0.016/−0.020) is the larger read, and it is honest to
  note the 3x row's endpoint jump was unusually large (+0.051 from step
  160 to 180, vs the bracket arms' +0.021 to +0.041): the step-160 reads
  (1.3638 / 1.3778 / 1.3688 / 1.3637) sit within ~0.014 of each other.
  The pre-stated rule keyed on finals and is satisfied at both seeds run;
  the honest statement is that GDN2's LR gain above 3x is real,
  sign-stable, and SMALL — the same order as ours, with the steady read at
  the band's edge and the endpoint read outside the seed band.
* **5x does not clearly separate from 4x** (finals 1.3951 vs 1.3992,
  Δ 0.0041; steady −0.222 vs −0.216; step-160 1.3688 vs 1.3778) — the
  bracket saturates in the 4x–5x region; 6x+ is untested (outside the
  approved bracket, exactly as 10x was on our side).
* **The 5x seed pair is tight** (1.3951 / 1.4049, delta 0.0098 — inside
  the winner's 0.0119 band class), and its seed-3402 row lands below the
  banked 3x row's endpoint at the same seed class (no same-seed 3x row
  exists for GDN2 — the banked GDN2 rows are all seed 3401; stated, not
  compared across seeds).

## 3. The readouts per arm

| arm | final NLL | vs the banked 3x option-B 1.4151 | steady (80–160 vs gdn2 AdamW) | train s | Muon cost s/step | step-1 receipt |
|---|---:|---:|---:|---:|---:|---|
| gdn2 B 3x (banked, the control's arm) | 1.4151 | — | −0.213 | 593.0 | +0.189 | banked row |
| **gdn2 B 4x, s3401** | **1.3992** | −0.0159 | −0.216 | 611.7 | +0.196 | bit-identical |
| **gdn2 B 5x, s3401** | **1.3951** | −0.0200 | −0.222 | 586.3 | +0.192 | bit-identical |
| gdn2 B 5x, s3402 | 1.4049 | −0.0102 | −0.221 | 608.9 | +0.193 | no banked 3402 row (recorded) |

Walls are honest-clock class-matched to the banked GDN2 rows (586–612 s vs
the control's 593–616). The Muon update RMS is healthy and in the same
class on every arm (0.177–0.178, vs the control's 0.168–0.184).

**The mechanism is receipted UNCHANGED in every arm** (this lane's scope
rule: optimizer wiring only, GDN2's source untouched) — GDN2's own gate
telemetry at the final checkpoint, read non-invasively from its own
projections, stays in the banked regime everywhere (state retention
exp(g) 0.48–0.83 per layer, erase gate b 0.44–0.53, write gate w
0.51–0.67, implied horizons 1.9–6.1 tokens — the control's banked rows:
0.675/0.486/0.561, 2.1–7.4). Better weights on the same always-on dense
gated-delta machinery; only the weights moved.

## 4. The pre-stated second-seed rule, adjudicated

Applied exactly as stated (the rule is receipted in every arm's
result.json and the assemble): **both bracket arms landed below the banked
option-B row by more than the 0.0119 seed band at seed 3401 (−0.0159 /
−0.0200) — both are clear new bests; the best-of-table (5x) paid the
second seed.** Its seed-3402 row (1.4049) confirms the sign (−0.0102 vs
the 3x row, inside the band at this seed) and the tight seed pair
(Δ 0.0098). The 4x arm is single-seed, like every GDN2 row it is compared
against except its own 5x pair; the seed band is the only seed evidence
and no conclusion here rests on a margin smaller than it except where
stated.

## 5. The honest verdict of the config-symmetry program, and the C5 read

*What is GDN2's best LR point?* — **4x–5x of its own AdamW LR (finals
1.3992/1.3951 vs the inherited 3x's 1.4151), saturating there — 5x is the
best measured point and 4x is not distinguishable from it outside the
band.** GDN2's config axis bought it ~0.016–0.020 over its inherited
point, the same small order our bracket bought us (0.013–0.024). The
inherited 3x point under-served BOTH machines by the same ~0.02; neither
machine's bracket is a ranking-mover.

*Is anything settled between the machines now?* — **Yes, and it settles
the way every matched read has pointed all along: GDN2 AHEAD.** With the
config search symmetric on both axes — routing bracketed on both machines
(our spread +0.016 inside the band; GDN2's option B bought 0.055), LR
bracketed on both machines (our 4x, GDN2's 4x–5x) — the symmetric
best-vs-best is **our 1.4899 (B-analog routing, 4x, seed 3402) vs GDN2's
1.3951 (option-B routing, 5x, seed 3401): margin 0.0949, outside the
~0.075 endpoint-variance band, GDN2's favor.** The matched-point reads
converge on the same number: 3x-vs-3x bests 0.0876, 4x-vs-4x 0.0907
(same-seed 3401: 0.0915). The routing lane's one honest open door —
"parity-within-band at our 4x vs GDN2's unbracketed 3x, asymmetric in our
favor" — is closed by this lane the only way it could be: GDN2's axis
mirrors ours, and the margin re-opens beyond the band.

*The C5 read, final form:* the side-by-side attributed ~0.16 nats
(AdamW-era steady) to the write/read form; the bake-off showed the surface
masks most of it; the control showed the ranking survives the matched
surface; the routing lane showed routing owns none of the residual on our
side; this lane shows **the LR axis owns ~0.02 on each side — equal, and
therefore ranking-neutral: the last config alibi is spent.** What remains
— **~0.09 at best-vs-best, ~0.09 at every matched point, sign-stable
across every seed and routing and LR point measured — is structural, and
its architectural face is receipted in the inventory enumeration: GDN2
writes through 27.4M params of learned gate machinery (18% of the
machine); our machine writes through 27.7K (0.018%) and three
hyperparameter thresholds.** Any structural arm must run under the
receipted best surfaces both machines now have (ours: B-analog at 4x;
GDN2's: option B at 4x–5x) or it will mis-attribute surface to structure.
That arm is K11 LEARNED-GATES, pre-registered and running now on cards
0/1 — the config-symmetry program hands it a clean table: **the residual
gap is C5's, and only the write form can move it.**

## 6. Residuals, stated

* **The endpoint-jump caveat, applied to this lane's own headline:** the
  banked 3x row's endpoint jump (+0.051 from step 160 to 180) was the
  largest of the four option-B rows (bracket arms +0.021 to +0.041); the
  step-160 reads sit within ~0.014 of each other (1.3638 / 1.3778 /
  1.3688 / 1.3637). The pre-stated second-seed rule keyed on finals and is
  satisfied; the honest statement of the bracket's gain is "real,
  sign-stable at both seeds run, ~0.02 at the endpoint and ≤~0.009 in the
  steady mean" — no conclusion here rests on the last 0.01 of any
  endpoint, and the best-vs-best margin (0.0949) does not rest on it
  either (step-160 best-vs-best: 1.4704 vs 1.3688 = 0.102).
* **The 4x arm is seed-3401 single-seed** (the second seed went to the
  best-of-table arm, 5x, per the pre-stated rule); the 5x seed pair
  (Δ 0.0098) is the only GDN2 seed evidence.
* **6x+ is untested on either machine** (outside the approved brackets;
  the 5x-vs-4x read — Δ 0.0041, inside the band — says GDN2's curve is
  flat there, but the far side beyond 5x is unmeasured, exactly as 6x+
  is on ours).
* **Probe scale, one protocol** — everything above is the 20 M probe (180
  steps, 7-window eval, the ~0.075-nat endpoint-variance band); the
  2700 s board and C14+ scale are untested, as every lane has stated for
  its own numbers.
* **The seed-3402 step-1 receipt has no banked same-seed GDN2 row** to
  assert against (none exists beyond 3401); the receipt records this
  honestly instead of silently passing — the winner's seed-3402 rows had
  a banked same-seed row, GDN2's do not, and that is a property of the
  banked corpus, not a weakness of this arm.

## Commands

```bash
# one arm, detached on a leased card from the diagnosis lane's pool (cards 2/3;
# the K11 lane's cards 0/1 untouched)
setsid bash -c 'OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=2,3 \
  experiments/expressivity_tasks/x45_matrix_emender/run_gdn2_lr_bracket_leased.sh \
  gdn2_muon4x_gates_adamw 3401 > …/gdn2_lr_bracket_runs/gdn2_muon4x_gates_adamw.console.txt 2>&1' &

# the far-side bracket, and the second seed the clear new best owes
setsid bash -c '… run_gdn2_lr_bracket_leased.sh gdn2_muon5x_gates_adamw 3401 …' &
setsid bash -c '… run_gdn2_lr_bracket_leased.sh gdn2_muon5x_gates_adamw 3402 …' &

# the bracket table + the second-seed rule + the symmetric best-vs-best
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.gdn2_lr_bracket assemble

# the bracket/routing tests (CPU; includes the real-inventory option-B counts
# and the banked-artifact receipt checks)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_x45_gdn2_lr_bracket.py -q
```
