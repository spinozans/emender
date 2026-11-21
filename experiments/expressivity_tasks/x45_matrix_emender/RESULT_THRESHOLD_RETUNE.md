# RESULT_THRESHOLD_RETUNE.md — K19: THE THRESHOLD RE-TUNE ON THE COMPOSED MACHINE

**Date:** 2026-11-21 (the search completed 12:04 UTC; this document is the
post-lane-death harvest) · **Status: complete — THRESHOLDS-ROBUST PARITY. The
re-tune found NOTHING: the incumbent won its own generation on the search's own
scoreboard (anchor 1.3811104467936925 bit-for-bit, the harness gate; all 8
generation samples worse), the win bar (≥ 0.01 at seed 3401 vs the composed
machine's own same-seed banked row) was NOT met by any challenger, so nothing is
banked and the search-internal scoreboard stays what it always was — a search
receipt, never a ladder row. The tuned thresholds (tau_match 0.45, tau_pop 0.10,
read_scale 2.787, tau_merge 0.85 held) TRANSFERRED across the K13 read-law +
K11B gates interface change; the K16 config-symmetry discharge now extends to
the thresholds; searched routing joins learned routing (K14) in the rejected
bin; the residual (+0.0289/+0.0311 vs GDN2 1.3522) belongs to the board
scale.** ·
**Authorisation:** the registration commit a246afac (the machinery committed
BEFORE any measurement, the standing lesson), the pre-stated adjudication in
`PREREGISTRATION_K19_THRESHOLD_RETUNE.md` §3 (the supervisor's addition of
2026-11-25), cards 5/6 only (the K18 lane's 0/1 and the X46 re-run's 2/3
untouched).

## 0. The harvest receipt (how this document came to exist)

The original attended worker lane died at its 6-hour session ceiling mid-search.
The detached leased runner (`run_k19_threshold_search_leased.sh`, the
long-session survival playbook: setsid + redirected stdio, the lease owner and
heartbeat keeper inside the detached process) survived the session death and
completed the full registered program cleanly, exiting 12:04 UTC. **Contrary to
the harvest expectation, the late receipts were NOT left uncommitted:** the
runner's own registered design is commit-per-eval ("the ceiling is real: a
committed partial beats an uncommitted complete"), and it carried that through
the whole pass — every eval receipt, the winner's confirm, and the assemble were
committed and pushed by the detached process itself (last commit a6b88a29,
12:02:27 UTC). The only artifact left uncommitted was the search console
(`k19_runs/k19_search.console.txt`), committed by this harvest together with
this document. The harvest itself spent zero GPU; the lease `.wg/gpu_leases/5`
was released on exit (the lease directory is empty except its lock; no stale
lease file remains).

Commit trail of record (all by the detached runner unless noted):

| stage | commit |
|---|---|
| the registration (before any measurement) | `a246afac` |
| the smoke shakedown (isolated paths; caught a `write_result` kwargs bug before any full exposure) | `b944d595`, `70deaf95`, `8c7786a5`, `3df583ea`, `b9dafffb`, `a1914610` |
| eval 0 (incumbent anchor, the harness gate) | `13435196` |
| evals 1–8 (g1_0 … g1_7) | `ca7b1444`, `5da92470`, `bd4fa34f`, `c1d0f455`, `af22d77e`, `ca429842`, `dfd78bac`, `6d453060` |
| the bounded pass complete (9 evaluations) | `66cab000` |
| the winner's protocol-confirm at seed 3401 | `c4a123a6` |
| the assemble (`receipts/k19_threshold_search_summary.json`) | `a6b88a29` |
| this document + the search console (the harvest) | this commit |

## 1. The verdict, stated first

**THRESHOLDS-ROBUST PARITY.** The composed machine's op-routing thresholds —
tuned on the PRE-read-law machine per `CMAES_TUNING.md` and never re-searched
after K15's read/write interface change — were re-searched on the interface
that actually runs them, under the original search's own parameterization
VERBATIM, and the re-tune found nothing:

* **The incumbent won its own generation.** The anchor re-run (eval 0)
  reproduced the composed machine's banked seed-3401 post-cooldown row
  **1.3811104467936925** bit for bit (`anchor_reproduced_banked_row: true`) and
  the K15 arm's own step-1 row (5.782775391046648, the nearest-kin parity,
  `step1_k13_parity: true`) — the harness gate passed. All 8 generation samples
  are worse (table in §2); the best of them is +0.0014, deep inside the 0.0119
  winner-seed band, and the worst is +0.0199.
* **The win bar was not met, by construction of the outcome:** the search's
  best config IS the incumbent, so the measured "win" vs the composed machine's
  own same-seed row is **0.0** against a bar of ≥ 0.01 — PARITY, per the
  pre-stated clause quoted in §5. The second seed (3402) is NOT owed: that
  clause fires only on STALE (a challenger meeting the bar).
* **No confirmation run is owed — and the strongest form of it already
  exists.** The incumbent's own full-protocol row is banked
  (receipts/k15_composition_summary.json; 1.3811104467936925 at seed 3401) and
  was re-validated bit-for-bit TWICE beyond the banked row itself: once by the
  search's anchor eval, and once by the registered detached design's
  winner-confirm pass (`k19_runs/winner_seed3401/`), which — the winner being
  the incumbent — re-ran the banked row at the full probe protocol with the
  full mechanism receipts and reproduced it **bit for bit**
  (`determinism_reproduced: true`; post-cooldown 1.3811104467936925,
  constant-180 1.433674931526184, EMA-180 1.424389226096017 — every digit equal
  to the K15 arm's own landed rows). Nothing further is owed, and nothing new
  is banked: there is no challenger.
* **Escalation flags: EMPTY.** No row at or past GDN2's 1.3521734476089478 (the
  best sample 1.382519 is +0.0303 away); the anchor reproduced; the confirm
  passed its determinism receipt; no standing conclusion inverted (§5).
* **Every margin is read against BOTH bands** — the 0.075 endpoint-variance band
  and the winner's 0.011948040553501738 seed delta — and all anchors are quoted
  throughout: composed **1.3811104467936925 / 1.383232252938407** (seeds
  3401/3402), GDN2 **1.3521734476089478**.

## 2. The 9-eval table (the full search-internal scoreboard — a receipt, never a ladder row)

One eval = one matched-exposure probe at seed 3401 (180 constant-LR steps + the
24-step linear cooldown, 20,022,866-target matched exposure, the
document-disjoint calibration eval, the corrected cooldown-primary scoring;
22,745,198 total targets incl. the cooldown's ~2.7M, identical across evals).
The PRIMARY score (post-cooldown) is the table's NLL.

| eval | label | tau_match | tau_pop | read_scale | NLL (post-cooldown) | Δ vs incumbent | vs 0.0119 band | occupancy (mean live slots) | effective read rows | top-row w-share | identity share |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | **incumbent_anchor** | 0.45 | 0.10 | 2.787046 | **1.3811104467936925** | — | — | 27.20 | 16.83 | 0.1553 | 0.12663 |
| 1 | g1_0 | 0.333377 | 0.189707 | 2.038027 | 1.3825189556394304 | +0.0014085 | inside | 15.28 | 9.84 | 0.2222 | 0.10953 |
| 2 | g1_1 | 0.385508 | 0.165678 | 3.287821 | 1.394933717591422 | +0.0138233 | **outside** | 17.71 | 10.88 | 0.2172 | 0.11378 |
| 3 | g1_2 | 0.489572 | 0.036505 | 2.718021 | 1.3885233742850167 | +0.0074129 | inside | 32.10 | 19.63 | 0.1361 | 0.12899 |
| 4 | g1_3 | 0.458321 | **−0.057480** | 2.938353 | 1.4010401112692696 | +0.0199297 | **outside** | 27.42 | 17.42 | 0.1445 | 0.12540 |
| 5 | g1_4 | 0.354812 | 0.116793 | 2.455912 | 1.3860350166048323 | +0.0049246 | inside | 16.52 | 10.33 | 0.2221 | 0.10991 |
| 6 | g1_5 | 0.379797 | 0.174476 | 3.074112 | 1.3870552948543005 | +0.0059448 | inside | 17.53 | 10.87 | 0.2134 | 0.11311 |
| 7 | g1_6 | 0.457628 | 0.192498 | 3.474968 | 1.3897779669080461 | +0.0086675 | inside | 25.89 | 15.23 | 0.1800 | 0.12311 |
| 8 | g1_7 | 0.325014 | 0.181219 | 3.405611 | 1.3934331621442522 | +0.0123227 | **outside** | 13.50 | 8.54 | 0.2543 | 0.10356 |

Best search-internal NLL: **1.3811104467936925** (the incumbent anchor itself).
Per-eval receipts: `receipts/k19_probes/eval_000..008.json`; the assembled
scoreboard, trajectory and adjudication:
`receipts/k19_threshold_search_summary.json`.

## 3. The parameterization receipt (the original search's own, verbatim)

From `receipts/k19_threshold_search_summary.json` → `parameterization`, which
restates the registration commit a246afac, which restates `CMAES_TUNING.md`
phase 2 verbatim:

* **Searched dims and bounds:** `tau_match` linear **0.2–0.8**; `tau_pop`
  linear **−0.2–0.4**; `read_scale` linear **0.5–4.0**.
* **tau_merge HELD at 0.85**, per the original search's own receipted reason
  (CMAES_TUNING.md: "tau_merge was NOT searched: the diagnosis below shows
  merges fire plenty but mostly on near-duplicate rows, so threshold-tuning the
  merge gate had no evidence demanding it") — supervisor-approved 2026-11-25,
  nothing invented.
* **x0** the composed machine's incumbent (tau_match 0.45, tau_pop 0.10,
  read_scale 2.787046, tau_merge 0.85; lr 3.3427141575371284e-4, weight decay
  0.17422218730922470, warmup 45 — the K15 composed machine's inherited banked
  point, held fixed); **sigma0 0.15; popsize 8; ONE generation; CMA-ES seed
  20261125.**
* Every eval at seed 3401 on the same deterministic stream prefix (the
  original's own convention: the search tunes the seed and the data draw as
  well as the thresholds); the search's eval IS the registered probe protocol —
  the same horizon the composed anchors and the GDN2 row live at. **No
  board-scale (2700 s) claim is made or implied; the probe-horizon optimum is
  receipted as a probe-horizon optimum.**

## 4. The mechanism telemetry (the op-mix instrument per eval)

The K15 mechanism instrument (`k15_probe_diagnostics`, imported verbatim) ran
at every eval's post-cooldown state. Lane-0 op rates per token, all 9 evals:

| eval | append | correct | pop | merge | none |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.018005 | 0.979841 | **0.0** | 0.002154 | 0.0 |
| g1_0 | 0.009672 | 0.989395 | 0.0 | 0.000933 | 0.0 |
| g1_1 | 0.011965 | 0.986422 | 0.0 | 0.001612 | 0.0 |
| g1_2 | 0.022481 | 0.974080 | 0.0 | 0.003438 | 0.0 |
| g1_3 | 0.018389 | 0.979256 | 0.0 | 0.002355 | 0.0 |
| g1_4 | 0.010880 | 0.987801 | 0.0 | 0.001320 | 0.0 |
| g1_5 | 0.011599 | 0.986931 | 0.0 | 0.001470 | 0.0 |
| g1_6 | 0.017797 | 0.979713 | 0.0 | 0.002490 | 0.0 |
| g1_7 | 0.008626 | 0.990494 | 0.0 | 0.000880 | 0.0 |

**What the samples' op mixes show about why they lost:**

* **No sample's threshold move changed the op mix qualitatively.** Every
  sample stayed correct-dominant (97.4–99.0%); `none` is exactly 0 everywhere;
  merges stay ≤ 0.34%; and **`pop` never fired in ANY eval** — 0 lane-0 rate
  and 0 counts in the all-lane histogram — *including g1_3's tau_pop =
  −0.0575*, the only sampled point on the permissive side of the incumbent's
  0.10. The routing op mix is not a live degree of freedom at this budget:
  threshold moves within the original bounds reshape how much state the machine
  keeps, not which op it does.
* **The knobs moved the state-use instruments coherently, and neither direction
  paid.** Occupancy ranged 13.50–32.10 (incumbent 27.20), effective |w|-mass
  read rows 8.54–19.63 (16.83), top-row w-share 0.136–0.254 (0.155), identity
  share 0.104–0.129 (0.127). The losses came from BOTH directions of instrument
  movement: **less state use** (the most-concentrated sample g1_7: top-row
  0.254, effective rows 8.54, +0.0123 outside the band) and **more state use**
  (the highest-churn sample g1_2: append 2.25%, merge 0.34%, occupancy 32.10 —
  the most slots live and the most rows read of any eval, +0.0074). The
  incumbent sits between the extremes; the search probed both flanks and both
  flanks cost.
* **The winner's confirm vs the K15 arm's own rows: identical, every digit.**
  op-mix delta winner-minus-incumbent is exactly 0.0 on all five ops; identity
  share 0.126627025427666, occupancy 27.198343912760418, effective rows
  16.82564353942871, top-row share 0.15532034520042232 — bit-for-bit the
  banked machine's own instrument rows (source: `k15_runs/composition_seed3401/
  result.json`), as it must be: same config, same seed, same protocol,
  determinism receipted.
* **The gate telemetry was unmoved at every sample** (decay-gate mean distance
  from the registered scalar 0.0094–0.0113, write-gate distance from 1
  0.0221–0.0288, read-gate distance from 1 0.0093–0.0138 — the same scales as
  the incumbent's own rows): the searched thresholds did not disturb the K11B
  p3 gate stack.
* **The decorative-store sweep** ran at the confirm's both states (the receipt
  convention of K15/K16): 4 of 24 layers ≥ 0.02 dNLL post-cooldown — the
  banked regime's own 4–5, unchanged (the full per-layer rows are in
  `k19_runs/winner_seed3401/result.json` and the search console).

## 5. The adjudication (pre-stated in the registration §3, quoted and executed)

* **"THE SEARCH-INTERNAL SCOREBOARD IS NEVER A LADDER ROW"** — held. Nothing
  was banked from the search; §2's table is a receipt only.
* **"THE WIN BAR: ≥ 0.01 at seed 3401 vs the composed machine's own
  same-seed banked row (1.3811104467936925)"** — the winner's post-cooldown row
  had to be ≤ 1.3711104467936925 to meet it. The winner IS the incumbent:
  measured 1.3811104467936925, **win = 0.0, bar NOT met** → per the
  pre-stated clause, **"PARITY: the thresholds were ROBUST across the interface
  change — a receipt; the search-symmetry duty is discharged either way."**
* **The second seed (3402) clause** — fires only "If met → the thresholds were
  STALE and the second seed is owed." Not met; `confirm_seed_3402: null`; no
  second seed owed or run.
* **The determinism receipt** — the confirm "must reproduce the search eval's
  own NLL bit-for-bit": reproduced (`determinism_reproduced: true`), and the
  search eval's own NLL is itself the banked row reproduced by the anchor
  eval. Two independent bit-for-bit reproductions of the composed machine's
  banked full-protocol row, both already committed.
* **The escalation clauses (a)–(d)** — all clear: no GDN2 contact (best row
  1.3811, +0.0289 vs 1.3522; best *sample* 1.3825, +0.0303); no standing
  conclusion inverted — the K14 receipts said the model REJECTS LEARNED
  routing (the structural zero-gradient wall on the op channel); K19 now
  receipts that the model rejects SEARCHED routing too, on the new interface,
  under the original search's own parameterization: **searched routing joins
  learned routing (K14) in the rejected bin.** The anchor reproduced; the
  confirm passed.
* **The pre-state, honored:** "NO directional prediction was pre-stated; stale
  thresholds, robust thresholds and an incumbent win are ALL receipts." The
  outcome is the robust one, receipted as such.
* **Banking:** nothing to bank — there is no challenger and no new row; the
  ours-best quotation is UNCHANGED (composed 1.3811/1.3832, cited).

## 6. The read_scale trajectory, stated honestly

The supervisor's own registered request (the single most interesting
dimension — the read law changed its mechanical role from softmax temperature
to raw magnitude scale under K13). The samples ranged **2.038–3.475**; the
incumbent's **2.787** sits at the middle of the sampled cloud (cloud midpoint
2.757) — **a local optimum, not a boundary artifact** (the search bounds
0.5–4.0 were nowhere near touched). The honest nuance: the single best sample
(g1_0, +0.0014 — inside the 0.0119 band) sat at the cloud's LOW edge
(read_scale 2.038), so the generation carries a faint within-band tilt toward
lower read scale; but that delta is parity-grade at band resolution, the
registered win bar (0.01) cannot see it, and the pre-stated rule banks nothing
from it. There is no evidence the incumbent's read_scale is mistuned, and no
evidence the K13 read-law change moved its optimum anywhere the original
bounds would find.

## 7. The closing state: the config-symmetry discharge extends to the thresholds

K16 closed with "the remaining gap is NOT config-owned at probe scale; the gap
quote is now search-symmetric in every axis the program knows how to search" —
with the op-routing thresholds as the one receipted-stale exception (tuned on
the PRE-read-law machine, never re-searched after the interface change). K19
closes that exception at parity. Every config axis is now receipted at
parity-or-worse on the composed machine:

| axis | lane | receipt |
|---|---|---|
| LR point (hybrid-Muon multiplier) | K16 | the inherited 4x point is the bracket's optimum at BOTH seeds |
| optimizer family | K16 | plain AdamW at its own tuned point +0.194; the family ranking carries over UNCHANGED |
| op-routing thresholds | **K19** | **the incumbent is its own generation's optimum; robust across the interface change (this document)** |
| routing policy, learned | K14 | REJECTED (the structural zero-gradient wall) |
| routing policy, searched | **K19** | **REJECTED (this generation: all 8 samples worse; no regime change in the op mix)** |

**The remaining gap vs GDN2 is +0.0289 (seed 3401) / +0.0311 (seed 3402)
against 1.3521734476089478 — parity-within-band per the K16 ruling, not
config-owned at probe scale; the residual belongs to the board scale** (the
standing 2700 s horizon question every lane has named for its own numbers).

## Receipts of record

* `receipts/k19_probes/eval_000..008.json` — the 9 per-eval receipts (config,
  searched knobs, step log, op mix, mechanism post-cooldown, routing receipt,
  anchors, both bands, versions, git).
* `receipts/k19_threshold_search_summary.json` — the assembled scoreboard,
  parameterization, confirm receipt, mechanism comparison, adjudication,
  escalation flags (empty).
* `k19_runs/winner_seed3401/{result.json,scores.jsonl,start.json,training.jsonl}`
  — the winner's protocol-confirm (the incumbent re-run, bit-for-bit).
* `k19_runs/k19_search.console.txt` — the detached search console (committed by
  this harvest).
* `receipts/k15_composition_summary.json`,
  `k15_runs/composition_seed3401/result.json` — the banked composed-machine rows
  and instrument rows cited throughout.
* `PREREGISTRATION_K19_THRESHOLD_RETUNE.md` — the registration and the
  pre-stated adjudication; `CMAES_TUNING.md` — the original search's
  parameterization and the tau_merge receipted reason.

## Commands (the harvest's own validation)

```bash
git log --oneline | grep -i k19          # the commit trail of record (§0)
git status --porcelain                  # the k19 tree clean post-harvest
ls .wg/gpu_leases/                      # empty except .lock: lease 5 released on exit
```
