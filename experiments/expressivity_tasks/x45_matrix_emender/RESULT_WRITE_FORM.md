# RESULT — K17 THE WRITE FORM UNDER THE ADDITIVE READ: the last structural
transplant, tested on the machine that changed

**Date:** 2026-11-25 · **The registration:** `PREREGISTRATION_K17_WRITE_FORM.md`
(DRAFT, committed before any probe measurement, 365ab3a5). **Receipts:** every
number below is from `k17_runs/<arm>_seed3401/result.json`,
`receipts/k17_write_form_summary.json` (the assemble) and
`receipts/k17_gates.json` (the engine gates). **The adjudication is
ESCALATED per the lane's rules** (§5): no directional prediction was
pre-stated, the clauses resolved cleanly on the measured rows, and the
final gap-quote phrasing is the supervisor's (the K13 clause-conflict
precedent).

**The question:** does the dense gated write — G2's form, which HURT on the
SOFTMAX machine (+0.007) — COMPOSE with the additive read on the composed
machine (the K15 adjudication's named LAST structural candidate owner of
the remaining ~0.03), and does the additive read's one never-re-tuned
constant (the score-channel GAIN) own any of it?

## 1. The arms' rows (matched-exposure probe, corrected scoring, seed 3401)

| arm (post-cooldown PRIMARY) | post-cooldown | constant-180 | EMA-180 | vs the composed machine's same-seed 1.3811 |
| --- | ---: | ---: | ---: | ---: |
| **the dense gated write** (the b_proj-delta form, born as the composed machine) | **1.377752116748265** | 1.4345858097076416 | 1.4276997872761317 | **+0.0034** |
| **the frozen-write falsifier control** (the dense channel frozen at birth) | **1.3811104467936925** | 1.433674931526184 | 1.424389226096017 | **+0.0000 — BIT-IDENTICAL to the banked row** |
| the read-gain bracket, `read_scale` = 1.0 (a config arm) | 1.3799265623092651 | 1.4341617311750139 | 1.4112982409340995 | +0.0012 |
| the read-gain bracket, `read_scale` = 2.0 (a config arm) | 1.384902102606637 | 1.446018968296675 | 1.4168705940246582 | −0.0038 |
| the read-gain bracket, `read_scale` = 4.0 (a config arm) | 1.4033683368137904 | 1.4406403814043318 | 1.4551541634968348 | **−0.0223** (outside the band, the harmful side) |
| **the composed machine's own row** (the K15 arm, CITED — the baseline; the inherited 4x point AND the inherited gain 2.7870458737509636) | **1.3811104467936925** | 1.433674931526184 | 1.424389226096017 | — |
| **GDN2 (the gap reference)** | **1.3521734476089478** | | | |

**The bands on every margin:** the 0.075-nat endpoint-variance band and the
winner's 0.0119 seed delta (the composed machine's own 3401/3402 pair delta
is 0.0021 — the tightest pair any lane has measured on this machine).

## 2. The verdicts, per the pre-stated clauses (applied exactly as stated)

* **The dense gated write: W-2 PARITY-GRADE at seed 3401** (+0.0034, deep
  inside the seed band). The pre-stated 0.01 win-bar rule owes NO second
  seed, and the conditional combined arm does NOT fire. **The single honest
  nuance: the dense arm's 1.377752116748265 is the program's BEST ABSOLUTE
  ROW** (−0.0034 vs the composed machine's own 1.3811, +0.0256 remaining vs
  GDN2's 1.3522, 68.8% of the 0.0816-nat corrected gap closed at this seed)
  — reported as measured, NOT banked as composing (the pre-registered bar
  was 0.01, and the prior G2 lesson was that this lever HURTS; here it does
  NOT hurt — the G2-under-softmax result does NOT replicate under the
  additive read (no W-3 interference) — but the single-seed margin is
  one-third of the seed band).
* **The frozen-write falsifier control: PASSED, PERFECTLY.** The control's
  step-1 train loss is bit-identical to the composed machine's own row, AND
  — beyond the K11B §2.4 precedent's honest expectation (ULP-scale
  divergence) — **the full 204-step trajectory is BIT-IDENTICAL to the
  composed machine's own banked training log** (max per-step |Δloss| =
  0.0, first divergence: none), and the landing is bit-identical to the
  banked row on every scoring. The dense lane's exact-zero contributions
  never flip a single gradient bit: the falsifier receipts the whole
  machinery (module creation, optimizer routing, dtype promotion, the new
  kernels' arithmetic) as a NO-OP at birth, at the strongest possible bar.
* **The read-gain bracket: G-1 NULL on the near side, harmful on the far
  side.** The bracket's near side is flat within the band (1.0: +0.0012;
  2.0: −0.0038), and the far side past the inherited point TURNS UP outside
  the band (4.0: −0.0223) — the same flat-top-then-cliff shape the K16 LR
  bracket's muon6x cell showed. The machine's ACTUAL inherited gain
  (**2.7870458737509636**, the tuned softmax point — the registration's
  factual correction, §3 of `K17_WRITE_FORM.md`) stands, receipted; the
  read-gain axis's config-symmetry duty is discharged by receipt.

**No second seeds owed, no combined arm, no GDN2 contact** (the closest row
is the dense arm's +0.0256 remaining), **no quality-conclusion inversion,
`escalation_flags` empty** in the assemble receipt.

## 3. The mechanism receipts (the interesting instrument's answer)

* **The C5 write-channel instrument — the dense channel did not learn to
  carry mass at probe scale.** The TRAINED dense deposit's mean |value| is
  0.00483, **0.82 % of the payload scale** (per layer: 0.3 %–1.7 %), with
  the lane firing at 0.9995 (every token's live guards admit it) and the
  zero-born per-head scalars drifting only to 0.048 max. This is the SAME
  shape K11's per-head G2 arm receipted on the softmax machine (deposit
  0.0015–0.024 of payload scale): the channel OPENS barely, at probe
  scale, under the additive read too — the per-channel granularity did not
  change it. The dense channel carries 0.83 % of the event writes' mass:
  the write form is INERT at this exposure, and the NLL composes to parity
  accordingly.
* **The K15 mechanism instrument under the dense write: unmoved.** |w|-mass
  effective rows 15.87 (the composed machine's own same-seed row: 16.83),
  the top-row collapse detector 0.165 (vs 0.155), occupancy 26.2 (vs 27.2),
  the identity-rewrite share 0.124 (vs 0.127), all three p3 gate channels'
  drifts at the composed machine's own scales (decay 0.0110 vs 0.0107,
  write 0.0245 vs 0.0233, read 0.0108 vs 0.0113), and the decorative-store
  sweep in the banked regime (5 of 24 layers ≥ 0.02 dNLL at the
  post-cooldown state). The mechanism moved where the dense channel acts
  (it carried a little mass) and moved nothing else.
* **The dissociation clause, applied verbatim:** the dense channel's mass
  is inert (0.82 %) AND the NLL is at parity — a CONSISTENT null, not a
  dissociation: the write-form lever does not own the remaining ~0.03 at
  probe scale. The honest caveat (K11's own, carried): 180 constant-LR
  steps; the channel may need a longer horizon to open — what a
  board-scale decision must weigh.

## 4. The honest scope of this result

* Probe scale only (the 20,022,866-target matched exposure, 180+24 steps),
  seed 3401 (the pre-stated win-bar rule owed no second seed anywhere);
  single-protocol rows.
* The dense arm ADDS parameters (+3,538,992 = +2.36 % over the composed
  machine; GDN2's own gate share is 18.3 %) — quoted in every receipt.
* The composed-machine-vs-GDN2 row is a cross-machine context row at the
  same probe protocol, not a controlled comparison.
* The engine's own receipts (the birth chain, the gradcheck, the walls)
  are `K17_WRITE_FORM.md` / `receipts/k17_gates.json`: born as the
  composed machine 0 flips in 516,734,976 lanes at probe scale, the fp64
  gradcheck through the dense-write path green, the dense lane's honest
  wall cost +23 % on the forward block call (training walls 2088.6 s
  constant vs the composed arm's 1920.8).

## 5. The adjudication (ESCALATED, per the lane's rules — the K13
clause-conflict precedent governs)

The measured rows for the supervisor's ruling: **the last named structural
candidate is TESTED and does not own the composed machine's remaining gap
at probe scale.** The dense gated write — G2's form reborn at the
per-channel granularity on the additive interface — lands at
parity-grade evidence (+0.0034 at 3401, one seed, inside the 0.0119 band;
the program's best absolute row; NO interference, unlike G2 on the
softmax machine), with the C5 instrument receipting the channel inert
(0.82 % of payload mass). The read-gain bracket is null on the near side
and harmful past the inherited point (4.0: −0.0223). The falsifier control
is bit-identical to the composed machine's own landed row — the strongest
plumbing receipt the lane structure has produced.

The gap quote the adjudication must phrase: **ours-best is now the dense
arm's 1.377752116748265 at seed 3401 (single-seed, parity-grade), vs GDN2's
1.3521734476089478 — +0.0256 remaining, sign-stable against every arm and
every cell measured this lane, INSIDE the 0.075-nat endpoint-variance
band.** The K16 ruling's own phrasing ("the gap's magnitude is owned at
probe scale by the last named structural candidate — the WRITE FORM — OR
by protocol-scale noise the band already prices in; distinguishing those
two is beyond the probe's resolution") now has its second half receipted:
the write form was tested and does not own it at probe scale. What remains
is either protocol-scale noise (the band), or a lever the probe cannot see
(the 180-step horizon on a channel that opens at 0.8 % of payload scale) —
and the named-but-untested residuals are the horizon question and the
cross-machine scale question, both of which belong to the board-scale
comparison, not to another probe arm.

## 6. Commands

```bash
# the arms (the detached leased driver; cards 0/1, one card per arm):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k17_write_form_leased.sh dense 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k17_runs/\
dense_seed3401.console.txt 2>&1 < /dev/null &

# the assemble + the clause adjudication:
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.k17_probe_write_form assemble

# the lane's tests (CPU):
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_k17_write_form_probe.py -q
```
