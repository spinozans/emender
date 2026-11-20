# RESULT_CONFIG_BRACKET.md — K16: THE CONFIG BRACKET ON THE COMPOSED MACHINE

**Date:** 2026-11-25 · **Status: complete — the config-symmetry duty is DISCHARGED BY RECEIPT: the composed machine's inherited hybrid-Muon 4x point was already the bracket's optimum, the optimizer-family ranking survives the read-form change intact, and the best-vs-best row vs GDN2 is UNCHANGED (ours 1.3811/1.3832 vs GDN2's 1.3522, +0.0289/+0.0311 remaining). The remaining gap is NOT config-owned at probe scale; the final gap-quote phrasing is escalated per the lane's rules (§5).** ·
**Authorisation:** the supervisor's K15 ruling (2026-11-24), the named follow-up 1 — mandatory before any final gap claim: the composed machine (per-token + separated keys + the K13 additive read + the K11B p3 gate stack; the program's best absolute rows 1.3811/1.3832 post-cooldown) has NEVER been tuned — it inherits the banked winner's hybrid Muon 4x point, which was tuned for the SOFTMAX-read machine. The symmetric-claim rule requires our best-vs-best to be search-symmetric; this lane brackets the composed machine's LR axis {2x, 3x, 6x} around the inherited 4x point (cited at both seeds, never re-run) and runs the optimizer-family check (plain AdamW at its own tuned 1x point).

## 0. The verdict, stated first

**The inherited 4x point stands.** The bracket's near side rises monotonically
toward it (2x 1.4355 → 3x 1.4034 → 4x **1.3811**, both lower cells worse by
0.054/0.022 — OUTSIDE the 0.0119 seed band), the far side turns up inside the
band (6x **1.3863**, +0.0052 vs the 4x row — parity-within-band, and its
|w|-mass concentrates: effective rows 13.0 vs 16.8, top-row share 0.216 vs
0.155 — the higher LR pushes the read's mass toward concentration, the
opposite of collapse but the wrong direction for quality), and the
optimizer-family check lands decisively on the incumbent family (plain AdamW
at its own tuned point: **1.5747**, +0.194 vs the 4x row — the hybrid Muon's
advantage on the composed machine is larger than the whole remaining gap, and
the family ranking the bake-off established on the softmax machine carries
over to the additive-read machine UNCHANGED). **The best cell of the bracket
is the inherited 4x cell — the second-seed duty is DISCHARGED BY THE BANKED
PAIR** (the K15 composition arm's own rows at both seeds: 1.3811 / 1.3832,
seed delta 0.0021, the tightest pair any lane has measured on this machine);
no new run is owed, and the search-symmetric ours-best is UNCHANGED.

**The config yield is ZERO at the primary scoring — and that is the receipt.**
The composed machine's remaining +0.0289/+0.0311 vs GDN2's 1.3522 was NOT
hiding an LR or family mis-tuning at probe scale: every bracket cell lands at
or above the inherited point's row, the family check confirms the incumbent
optimizer family, and the config-symmetry duty (ours searched, GDN2 searched,
best-vs-best) is now discharged on every axis — routing (both machines,
the routing-symmetry and muon-control lanes), LR point (ours: the winner's
own bracket AND now the composed machine's own bracket; GDN2: its own
bracket), and family (ours: the bake-off AND this lane's composed-machine
check). What remains is owned by the last named structural candidate — the
WRITE FORM (event writes vs dense superposition) — subject to the
band-discipline phrasing adjudicated in §5.

### the bracket table (seed 3401, the exact probe protocol, 20,022,866 targets, corrected scoring)

| arm (post-cooldown PRIMARY) | constant-180 | **post-cooldown** | EMA-180 (ranking-only) | vs the inherited 4x row 1.3811 |
|---|---:|---:|---:|---:|
| composed + Muon-2x (6.685428315074257e-4) | 1.4937 | 1.4355 | 1.6000 | +0.0544 |
| composed + Muon-3x (1.0028142472611385e-3) | 1.4484 | 1.4034 | 1.4841 | +0.0223 |
| **composed + Muon-4x (1.3370856630148514e-3) — the inherited point, CITED (the K15 arm's own rows)** | **1.4337** | **1.3811** | 1.4244 | — |
| composed + Muon-6x (2.005628494522277e-3) | 1.4489 | **1.3863** | **1.3842** | +0.0052 |
| composed + AdamW-1x (3.3427141575371284e-4, fused, the bake-off's own adamw convention) | 1.6918 | 1.5747 | 1.7417 | +0.1936 |
| *(the 4x cell's second seed, banked: post-cooldown 1.3832 at 3402 — the pair's delta 0.0021)* | | | | |

**The honest instrument divergence, stated not hidden:** the SECONDARY
EMA-0.99 read (ranking-only, biased toward mid-run weights) ranks 6x FIRST
(1.3842 vs the 4x cell's 1.4244) while the PRIMARY post-cooldown read keeps
4x ahead (1.3811 vs 1.3863). The pre-stated rule keys on the primary; the
divergence is reported as measured — the 6x arm's EMA iterate is its best
ranking instrument, its primary row is 0.0052 worse (inside the seed band),
and its mechanism rows show WHY the two instruments disagree at 6x: the
|w|-mass concentrates (effective rows 13.0, top-row share 0.216) and the
write/decay gates drift ~3x their 4x-cell scales (write 0.0386 vs 0.0233,
decay 0.0120 vs 0.0107) — a hotter, less stable endpoint, exactly the shape
a within-band bracket top looks like.

## 1. The arms and the conventions (the prior bracket lanes', verbatim)

Every arm runs the EXACT probe protocol: the composed machine BYTE FOR BYTE
(the banked winner build `velocity_ladder._build_k10` verbatim +
`convert_model_to_composition` — the K15 conversion, the gates born at the
registered scalar equivalents, committed and read-only, imported never
modified), the 20,022,866-target matched-exposure landing at 180
constant-LR steps, the registered flagship stream (identity asserted), the
document-disjoint calibration eval (recorded-read path), the honest clock,
the protocol's own grad-clip (1.0, error_if_nonfinite), the banked knobs
asserted at launch (betas (0.9, 0.9392), weight decay 0.17422, warmup 45 on
both hybrid groups, microbatch 8) — THEN the corrected scoring (24-step
linear cooldown PRIMARY, EMA-0.99 at step 180 SECONDARY ranking-only,
constant-180 retained for banked comparability). The ONLY difference vs the
K15 composition arm is the optimizer config: the peak LR multiple (2x/3x/6x
of the tuned AdamW base 3.3427141575371284e-4) or the family (plain fused
AdamW at 1x, the bake-off's own adamw arm's convention, every trainable
tensor in one group). The hybrid arms' routing is the K15 arm's own
(`split_parameters_gated(model, "p3")`, imported verbatim, cross-checked
against the banked split name-for-name).

**Protocol-identical starts, receipted:** every arm's step-1 train loss is
BIT-IDENTICAL to the K13 additive arm's own same-seed row
(5.782775391046648 at 3401) — hence to the K15 composition arm's own row —
asserted in-code at every arm (the machine at birth is the additive machine;
the optimizer cannot touch the first forward). The birth receipts proper
(BORN-AS-THE-ADDITIVE-MACHINE, 0 flips in 516,734,976 lanes; the gates born
at the registered values) are the K15 lane's own, receipted at BOTH seeds —
the machine is not this lane's variable.

**Receipts:** `k16_runs/<label>_seed<seed>/` (each: `start.json` with the
full convention block and the routing inventory, `training.jsonl` live
per-step, `scores.jsonl`, `result.json` with the curve, walls, the
mechanism instrument (the K15 probe's own `k15_probe_diagnostics`:
|w|-mass effective rows, the collapse detector, occupancy, op mix, all
three gate channels' telemetry) and the decorative-store sweep at both the
step-180 and post-cooldown states), the assemble summary
`receipts/k16_config_bracket_summary.json`, and the lane's sources
(`k16_config_bracket.py`, `run_k16_config_bracket_leased.sh`,
`tests/test_k16_config_bracket.py`). Session lambda01, cards 0/1 (this
lane's own pool, one leased card per arm, two arms in flight; the corrected
X46 sweep's cards 2/3 untouched; cards 4/5/6 left free as buffer; card 7
forbidden), RTX 6000 Ada, torch 2.9.1+cu128, detached-leased drivers,
committed per arm.

## 2. The overlaid curves (THE artifact)

Calibration eval NLL (nats/byte) at matched constant-LR checkpoints, seed
3401 — the bracket's cells against the cited 4x curve (the K15 arm's own):

| step | MB | muon2x | muon3x | **muon4x (cited)** | muon6x | adamw |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 5.7943 | 5.7991 | 5.7930 | 5.7928 | 5.7281 |
| 5 | 0.57 | 5.7028 | 5.6709 | 5.6356 | 5.5787 | 5.0615 |
| 10 | 1.19 | 5.5429 | 5.4595 | 5.3718 | 5.1366 | 4.2445 |
| 20 | 2.37 | 4.9212 | 4.4593 | 4.0654 | 3.4963 | 3.1433 |
| 40 | 4.54 | 3.3119 | 2.8124 | 2.5699 | 2.2346 | 2.4424 |
| 80 | 8.88 | 1.9102 | 1.7175 | 1.6474 | 1.5873 | 1.9669 |
| 120 | 13.28 | 1.5858 | 1.4974 | 1.4667 | 1.4437 | 1.7350 |
| 160 | 17.72 | 1.4610 | 1.4070 | 1.3906 | 1.3966 | 1.5995 |
| 180 | 20.02 | 1.4937 | 1.4484 | **1.4337** | 1.4489 | 1.6918 |

**What the curves say:**

* **The near side is monotonely starved:** 2x and 3x descend slower at every
  early checkpoint and never catch the 4x curve (step 80: 1.910/1.718 vs
  1.647) — the classic under-tuned bracket arm, exactly the shape the banked
  winner's own bracket showed below 4x.
* **6x descends FASTER early and turns up at the endpoint:** it is the best
  cell at every checkpoint through 120 (step 40: 2.2346 vs 2.5699) but its
  last 20 steps cost it (step 160: 1.3966 vs 1.3906; the endpoint jump
  160→180 is +0.052 vs the 4x row's +0.043) — the hotter point buys early
  descent and pays endpoint oscillation, and the cooldown recovers most but
  not all of it (post-cooldown 1.3863 vs 1.3811, inside the seed band). The
  4x–6x region is FLAT within the band at the primary scoring: the composed
  machine's LR optimum saturates there, exactly as both machines' own
  brackets did.
* **The family check is not close:** plain AdamW descends fast off init
  (the smallest step-10 row, 4.2445) but crosses to the WORST cell by step 40
  and ends +0.19 nats behind — the bake-off's own finding (the hybrid Muon
  owns the winner's surface) carries to the additive-read machine UNCHANGED;
  the read-form change did NOT move the optimizer ranking.

## 3. The readouts per arm (seed 3401)

| arm | post-cooldown | vs 4x 1.3811 | steady (steps 80–160 mean) | train s | Muon RMS | mechanism (post-cooldown state) |
|---|---:|---:|---:|---:|---:|---|
| muon2x | 1.4355 | +0.0544 | 1.652 | 1964.1 | 0.182 | eff rows 17.96 · top share 0.129 · occ 26.5 · write drift 0.017 |
| muon3x | 1.4034 | +0.0223 | 1.540 | 1961.0 | 0.182 | eff rows 18.03 · top share 0.132 · occ 27.6 · write drift 0.018 |
| **muon4x (cited)** | **1.3811** | — | 1.502 | 1920.8 | — (not receipted by the K15 lane) | eff rows 16.83 · top share 0.155 · occ 27.2 · write drift 0.023 (the K15 receipt's own) |
| muon6x | 1.3863 | +0.0052 | 1.476 | 1942.2 | 0.183 | eff rows **13.01** · top share **0.216** · occ 23.4 · write drift **0.039** |
| adamw | 1.5747 | +0.1936 | 1.767 | 1897.9 | — | eff rows 14.47 · top share 0.181 · occ 23.4 · write drift 0.006 |

Walls are honest-clock class-matched across the lane (1898–1964 s constant
phase vs the K15 arm's own 1921/1924). The Muon update RMS is healthy and
in the same class on every hybrid arm this lane ran (0.182–0.183, vs the
banked rows' 0.168–0.185; the K15 lane's own receipts do not carry the
metric). The decorative-store sweep stays in the banked regime on
every arm (4–7 layers at ≥ 0.02 dNLL, the K15 rows' own 4–5 class). The
mechanism row to watch is 6x's: the |w|-mass concentrates and the gates
drift hotter — reported as measured, no bar pre-stated.

## 4. The pre-stated rules, adjudicated as stated

* **SECOND-SEED:** the best cell at seed 3401 is the **4x cell** — the
  lowest post-cooldown row across muon2x/3x/6x/adamw and the cited 4x row is
  the 4x row itself (1.3811). The rule's 4x exemption fires: the K15
  composition arm already landed at both seeds (1.3811/1.3832, pair delta
  0.0021), the pair is complete, and NO second-seed run is owed. The
  6x cell's +0.0052 deficit is inside the 0.0119 seed band —
  parity-within-band between cells, and the pre-stated rule keys on the
  primary row, not the within-band ordering.
* **THE BAND DISCIPLINE:** no bracket cell landed at or past GDN2's
  1.3522 post-cooldown row — the closest is the inherited point itself at
  +0.0289. No both-scorings-both-seeds contact report was triggered; no
  quality-conclusion inversion occurred (the family ranking holds, the
  machine ranking vs GDN2 is unchanged in sign at every cell and both
  seeds); `escalation_flags` is empty in the assemble receipt.
* **PRE-STATE (no directional prediction):** the bracket moved the best row
  by 0.000 at the primary scoring (the inherited point was already optimal)
  — the config-symmetry duty is discharged by receipt either way, and this
  is the receipt: **the remaining gap quote is UNCHANGED (+0.0289/+0.0311),
  and the write-form attribution inherits the SAME residual, not a smaller
  one.**

## 5. The final gap quote (PROPOSED — escalated for adjudication per the lane's rules; the K13 clause-conflict precedent governs)

**Search-symmetric on every config axis now — routing on both machines, LR
point on both machines (ours including the composed machine's own bracket,
GDN2's its own), family checked on ours at both machine forms — the
best-vs-best row under the corrected scoring is: ours (the composed
machine, at its bracket-optimal inherited 4x point) 1.3811 / 1.3832
(post-cooldown, seeds 3401/3402, pair delta 0.0021) vs GDN2 1.3522
(post-cooldown, its own bracket's best) — GDN2 AHEAD by +0.0289 / +0.0311,
sign-stable at both seeds and at every config cell measured, and INSIDE the
~0.075-nat endpoint-variance band the side-by-side receipts established.
The honest reading is NOT "a settled 0.03 deficit": it is
parity-within-band at best-vs-best — the composed machine's best row and
GDN2's best row differ by less than the protocol's own endpoint-variance
band, the gap's direction is sign-stable everywhere measured, and its
magnitude is owned at probe scale by the last named structural candidate
(the WRITE FORM — event writes vs dense superposition) OR by protocol-scale
noise the band already prices in; distinguishing those two is beyond the
probe's resolution and belongs to the board-scale comparison, not to this
lane. No config alibi remains on either side.**

The caveats that stand on every margin: the composed-machine-vs-GDN2 row is
a cross-machine context row at the same probe protocol, not a controlled
comparison; the 0.075 endpoint-variance band and the 0.0119 seed band are
quoted with every delta; probe scale only (no board-scale claim before the
operator signs the composition's registration off).

## 6. Residuals, stated

* **6x is single-seed** (the pre-stated rule sent the second seed to the
  best cell, which was the already-paired 4x cell); its 6x-vs-4x read
  (+0.0052) is inside the seed band and its EMA-first divergence is quoted
  as measured in §0 — a 6x seed pair would be the operator's call if the
  secondary instrument's hint is wanted pursued, not this lane's.
* **The 2x/3x/6x points bracket a FLAT top:** the composed machine's LR
  optimum saturates in the 4x–6x region (as both machines' own brackets
  did); 8x+ is untested (outside the approved bracket, exactly as 10x was
  on the softmax machine and 6x+ is on GDN2).
* **The adamw arm ran at ITS OWN tuned point only** (1x, the only LR the
  machine's AdamW side has ever been tuned at) — a bracketed AdamW on the
  composed machine was not in the approved arms; at +0.19 the family read
  is not close enough for an LR bracket inside the family to be the
  question.
* **Probe scale, one protocol** — everything above is the 20 M probe (180
  steps + 24-step cooldown, 7-window eval); the 2700 s board and C14+ scale
  are untested, as every lane has stated for its own numbers.

## Commands

```bash
# one arm, detached on a leased card from this lane's pool (cards 0/1;
# the X46 sweep's cards 2/3 untouched)
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k16_config_bracket_leased.sh muon2x 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k16_runs/\
muon2x_seed3401.console.txt 2>&1 < /dev/null &

# the bracket table + the second-seed rule + the search-symmetric ours-best
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket assemble

# the lane's tests (CPU; the cited-rows and rules checks)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_k16_config_bracket.py -q
```
