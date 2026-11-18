# RESULT_PROTOCOL_CORRECTION.md — the literature-backed scoring fixes applied to the velocity probe, and the recipe-correct (Moonlight) Muon arms

**Date:** 2026-11-22 · **Status: complete — all four arms landed. The headline: the ~0.09 machine gap SURVIVES the instrument fix (0.0949 banked → 0.0816 corrected, still outside the 0.075-nat endpoint-variance band, sign-stable under both recipes and both corrected scorings), and the Moonlight recipe LOSES at the probe horizon on BOTH machines — the probe-optimal "speedrun" LR points remain the best measured points at 20M targets, and the horizon-transfer rule is now a standing obligation, not a caveat.** ·
**Authorisation:** the operator's protocol-correction directive (three deliverables), with the two escalated conventions ruled by the supervisor before any run: (1) the corrected scoring — a 24-step linear cooldown (13.3% of the 180-step budget, Hägele's band) as the PRIMARY score plus EMA-0.99 at step 180 as the RANKING-ONLY secondary, the step-180 constant-LR eval retained for banked comparability; (2) the Moonlight implementation — the pinned hybrid's Moonshot `0.2·√max(shape)` scaling IS the RMS match (no invented rescaling), so the recipe's only unmet element was the LR point (~1× the machine's own AdamW-tuned LR), with the RMS receipt MEASURED on both groups per step. Machine-seed pairing inherited from the banked best rows (ours 3402, GDN2 3401); no directional prediction pre-stated for any arm. The research brief: `research.md` findings #10 (Moonlight), #14–15 (river-valley / constant-LR+cooldown), #18 (Bjorck horizon scaling).

## 0. The corrected ladder, stated first

All four arms at the exact probe protocol verbatim (the 20,000,000-target matched-exposure rule → the 20,022,866 landing, 180 constant-LR steps, the registered flagship stream, the document-disjoint calibration eval at 7 windows / 14,336 targets, the honest clock), then the corrected scoring. **Both speedrun re-runs reproduced their banked best rows BIT-IDENTICALLY end-to-end** — the step-180 constant-LR evals equal the banked finals to the last digit (rerun−banked = 0.0 on both), the strongest protocol-equivalence receipt any lane has banked: the corrected scores below differ from the banked rows ONLY by the instrument.

| arm (machine · recipe · LR point) | constant-180 (banked-comparable) | **post-cooldown (PRIMARY)** | EMA-180 (secondary, ranking-only) | cooldown drop | wall (const / total) |
|---|---:|---:|---:|---:|---:|
| **ours** (K10 winner, B-analog routing, seed 3402) · speedrun · 4× | **1.4899** (= banked, bit-identical) | **1.4338** | 1.5408 | 0.0561 | 1,897 / 2,147 s |
| ours · Moonlight · 1× | 1.7360 | 1.6207 | 2.0629 | 0.1152 | 1,901 / 2,151 s |
| **GDN2** (option B, seed 3401) · speedrun · 5× | **1.3951** (= banked, bit-identical) | **1.3522** | 1.3803 | 0.0429 | 620 / 704 s |
| GDN2 · Moonlight · 1× | 1.4550 | 1.3960 | 1.5766 | 0.0589 | 616 / 699 s |

**The gap, re-measured under every instrument (margin ours-minus-GDN2, positive = GDN2 ahead):**

| comparison | constant-LR (the banked instrument) | post-cooldown (PRIMARY) | EMA-180 (secondary) |
|---|---:|---:|---:|
| speedrun recipe (the banked best rows' configs) | **0.0949** (banked) | **0.0816** | 0.1605 |
| Moonlight recipe (1× both machines) | 0.2810 | 0.2247 | 0.4864 |

Every cell is GDN2 ahead, outside the 0.075-nat endpoint-variance band. **The 0.09 gap survives the instrument fix** — narrowed from 0.0949 to 0.0816, by exactly the differential cooldown drop (ours 0.0561 − GDN2's 0.0429 = 0.0132; 0.0949 − 0.0132 = 0.0817 ✓): the river-valley penalty is real (Wen et al.), it is machine- and LR-dependent, and the constant-LR probe was overstating OUR best row's loss by ~0.013 more than GDN2's — but removing it does not close, let alone invert, the machine ranking. The secondary instrument agrees (EMA margins 0.1605 speedrun / 0.4864 Moonlight, same sign), as does the step-160 steady read (1.4704 vs 1.3688 = 0.1016). **The gap is recipe-sensitive (0.08 speedrun / 0.22 Moonlight) but sign-stable under every combination measured — the ranking conclusion of the config-symmetry program is INSTRUMENT-ROBUST.**

## 1. The instrument fix (Deliverable 1): what the corrected scoring changes, and what it does not

**The convention (supervisor-approved, pre-stated, receipted in every arm):** after the constant-LR phase completes the banked protocol verbatim, a LINEAR cooldown of 24 steps (13.3% of 180, inside the brief's 10–15% band) continues the same registered stream/data order, LR factor (25−k)/24 for cooldown step k (1.0 → 1/24, never zero), both hybrid groups scaling equally; the post-cooldown calibration eval is the PRIMARY corrected score. The SECONDARY is the EMA iterate at step 180 (decay 0.99, window ≈ 100 steps, constant-phase only, no extra data) — RANKING-ONLY, its absolute number biased toward mid-run weights, stated not smoothed.

**What it changes:**

* **The constant-LR score was a biased estimator, exactly as the brief predicted.** Every arm's score improves under the cooldown (drops 0.043–0.115), and the bias is differential: it penalized our 4× arm more than GDN2's 5× arm (0.0561 vs 0.0429), moving the best-vs-best margin by 0.013. The 1× Moonlight arms — slower descenders, still far from their valley floor — drop the most (0.115 ours / 0.059 GDN2). **Honest conflations, stated:** the cooldown score includes ~13% more data (2.7M extra targets), identical across arms, so cross-arm margins stay valid while per-arm "drops" are NOT pure oscillation-penalty measurements (extra-budget descent is in them); and the EMA's absolute numbers are mid-run-biased by construction — only its rankings are read.
* **The probe's fine-grained LR-bracket margins are demoted.** The bracket lanes' conclusions — our 4× over 3× by 0.013–0.024, GDN2's 4×–5× over 3× by 0.016–0.020, 5×-vs-4× by 0.0041 — sit INSIDE the band the instrument moves scores by (per-arm drops 0.04–0.06 here, machine-differential ~0.013). Under the corrected protocol those margins are NOT load-bearing: the probe-optimal points (ours 4×, GDN2's 4×–5×) remain the best measured points at the probe horizon, but the ~0.02 separations between adjacent LR points are probe-instrument artifacts until re-bracketed under the corrected scoring. No re-bracket was run (outside the approved four arms).
* **The "0.09" headline number is instrument-dependent and must be quoted as ~0.08–0.09.** The honest current statement: GDN2 ahead by 0.0816 under the corrected primary score, 0.0949 under the legacy constant-LR score, sign-stable everywhere.

**What it does NOT change:**

* **The machine ranking** — GDN2 ahead under both recipes × both corrected scorings (four combinations, all outside the band; §0).
* **The structural margins are UNAFFECTED** — the granularity/untangling dial (−0.3895, chunk 128 → per-token) and the separated-keys result (−0.1860) are 3–6× the entire measured instrument bias (≤0.115 per arm, and their compared arms shared one optimizer/LR, so the differential bias between them is second-order). Both survive as banked directional conclusions; neither was re-run here (outside scope). The C5 attribution — the residual gap is the write/read form, GDN2's 27.4M learned gate-matrix params vs our 27.7K decay pair and three hyperparameter thresholds — is REINFORCED: after removing the optimizer-recipe and instrument alibis, ~0.08 remains, structural.
* **Muon's win over AdamW on both machines** — even the Moonlight-1× arms beat the banked AdamW rows by margins outside the band (ours 1.6207 vs 1.8227 = −0.20; GDN2 1.3960 vs 1.5837 = −0.19), and the speedrun arms by 0.23–0.39 under the corrected score (ours 1.4338 vs 1.8227 = −0.39; GDN2 1.3522 vs 1.5837 = −0.23). Caveat stated: the AdamW rows were not re-scored here (outside the four approved arms); their own cooldown drops cannot close a 0.19–0.20-nat margin.

## 2. The Moonlight-recipe arms (Deliverable 2): the recipe-correct comparison

**The key codebase fact (the lane's own finding, supervisor-approved):** our pinned hybrid Muon is ALREADY RMS-matched by construction — `hybrid_muon.scaled_update` scales the orthogonalized update by `0.2·√max(shape)`, the Moonshot scaling whose docstring says it "matches AdamW's approximate .2 update RMS"; weight decay is already in the hybrid (decoupled, each machine's own tuned value). So the Moonlight recipe's only unmet element was the LR point. The Moonlight arms are therefore `HybridMuonX45` VERBATIM at ~1× the machine's own tuned AdamW LR (ours 3.3427e-4, GDN2 5.4201855e-4), own betas/weight-decay/warmup, same routings and machine-seed pairings as the re-scored best rows — a clean isolation of the LR point under matched update scale.

**The measured RMS receipt (new instrumentation, metrics-only, on OUR machines):**

| arm | Muon update RMS (measured, steady) | aux AdamW update RMS (measured) |
|---|---:|---:|
| ours · speedrun 4× | 0.1818 | 0.4586 |
| ours · Moonlight 1× | 0.1743 | 0.4515 |
| GDN2 · speedrun 5× | 0.1803 | 0.3272 |
| GDN2 · Moonlight 1× | 0.1795 | 0.3268 |

Two receipts in one table: (a) the Muon updates sit at Moonlight's ~0.2 target on every arm — the RMS matching holds, by construction, and did not need re-implementing; (b) **each machine's OWN aux-AdamW update RMS measures 0.33–0.46, ABOVE the 0.2–0.4 paper class the Moonshot scaling targets** — so "RMS-matched to AdamW" is matched to the paper's class, not to these machines' own AdamW updates. A strictly local RMS-matching would scale the Muon update UP ~1.8–2.5×. This is the nor-blog caveat (RMS matching can undertrain by width), measured on our machines — receipted as a FINDING, not acted on: no arm was run at a locally-matched RMS (that would be invention outside the brief's citation), and it is named as the natural follow-up cell.

**The recipe verdict at the probe horizon:** the Moonlight 1× point LOSES on both machines, under every scoring — post-cooldown, ours 1.6207 vs its speedrun 1.4338 (the recipe costs us +0.187), GDN2 1.3960 vs 1.3522 (+0.044). No directional prediction was pre-stated, and the asymmetry measured is itself informative: the recipe's cost is 4× larger on our machine than on GDN2. **But the honest scope of this verdict is the probe horizon only.** Bjorck et al. (#18): the optimal LR shrinks with token horizon — a 20M-target / 180-step probe is exactly the regime where hot points win and the long-horizon-correct 1× point has not yet had room to pay off; Moonlight's evidence is at 100B-token scale. The probe CANNOT adjudicate which recipe wins at the 2700s board or beyond; what it can and does say is: (i) the machine ranking does not change under either recipe (GDN2 ahead by 0.08 speedrun / 0.22 Moonlight — if anything the recipe-correct comparison WIDENS GDN2's lead), and (ii) any claim that the speedrun points transfer to longer horizons must now pass the horizon rule below, not inherit the probe's ranking.

**The mechanism is receipted UNCHANGED in every arm** (the lane's scope rule: scoring/optimizer-surface only): our arms stay in the banked identity-CORRECT hot-set regime (32.8–37.6 live slots, read entropy 1.54–1.84 nats, identity-rewrite share 0.115–0.118, span_key_std min 0.83–1.81 vs the 0.05 guard floor — the KEY-COLLAPSE guard holds throughout); GDN2's own gate telemetry at the post-cooldown final state stays in the banked regime (state retention exp(g) 0.60–0.74, erase gate b 0.45–0.50, write gate w 0.57–0.58, implied horizons 2.6–4.1 tokens). Better weights on the same machinery; only the LR point and the scoring instrument moved.

## 3. The horizon-transfer rule for the board runs (Deliverable 3, pre-stated)

The board runs (~2700 s ≈ 91.7M targets at gdn2's banked 33,958 bytes/s ≈ 4.6× the probe exposure) do NOT inherit probe-optimal LRs raw. The standing rule:

1. **Downscaled transfer (Bjorck-style power law):** `eta_board = eta_probe_opt × (D_probe / D_board)^alpha` — with alpha FIT AT OUR SCALE from a two-budget bracket (re-run the top config at 2× probe budget and take the local optimum shift), **never Bjorck's published exponent imported** (it is fitted at far larger scale). **The alpha fit is DEFERRED to the board-prep stage and labelled as not-yet-run:** a two-budget LR bracket is a multi-arm search this lane's four-arm budget did not admit (the supervisor's own scope note). At D_board/D_probe ≈ 4.6, any alpha in the plausible 0.1–0.5 range scales the probe-optimal LR down by ×0.43–0.84 — i.e. the board LRs will sit BELOW the probe optima by construction of the rule.
2. **The hard gate:** top-k configs must hold RANK STABILITY at 2× probe budget UNDER THE CORRECTED SCORING before any board transfer — the brief's rule #5, upgraded from post-hoc observation (the 1.2×-budget failure the lanes noted) to a hard gate.
3. **Recipe discipline:** a probe-selected raw LR multiple (a speedrun point) is SUSPECT for board transfer until it passes the gate; the Moonlight point is the standing long-horizon-correct recipe but is NOT banked as better at any horizon we have measured — the honest status is "unresolved below the 2×-budget gate".

## 4. The protocol amendment for the registration (the new standing probe discipline)

1. **Cooldown-or-EMA scoring:** every probe score that feeds a selection or a claim is the post-cooldown score (24-step linear, 13.3% of the constant-LR budget, same stream, both groups scaled); the EMA-0.99 iterate is the ranking-only cross-check; the constant-LR step-180 number is retained ONLY as the banked-comparability column and is no longer quotable as a config's "score" on its own.
2. **RMS-matched Muon, with a measured receipt:** any Muon arm keeps the RMS-matched update scale (the pinned Moonshot `0.2·√max(shape)` scaling) and weight decay, and receipts BOTH groups' measured update RMS per run (the new metrics-only instrumentation in `protocol_correction.py`); raw LR multiples are labelled speedrun-recipe points. A locally-RMS-matched point (~1.8–2.5× the Moonshot scale, per the measured aux-AdamW RMS) is named as an open follow-up, not a change.
3. **Horizon-corrected LR transfer:** no probe-optimal LR reaches a board run except through the downscaled power-law rule (alpha fit locally) and the 2×-budget rank-stability gate.

## 5. Residuals, stated

* **Single seed per arm** — the four arms run at the approved machine-seed pairing (ours 3402, GDN2 3401, the banked best rows' own); the winner's two-seed band (0.0119) and the 0.075 endpoint band are the only seed/variance evidence; the headline margins (0.08–0.22) sit far above them, the recipe deltas (0.044 GDN2) sit inside the endpoint band.
* **The probe cannot adjudicate long-horizon recipe questions** — the Moonlight verdict above is a probe-horizon statement; the 2×-budget gate and the board runs are where the recipe question settles.
* **The alpha fit and the 2×-budget gate are pre-stated but not run** (budget; labelled in §3) — the rule is the deliverable, the fit is board-prep.
* **The AdamW rows were not re-scored** (outside the four approved arms); all Muon-vs-AdamW statements above carry that caveat explicitly.
* **One LR point per recipe** — the Moonlight arm is 1× only; no search around it (a 1.5×–2× intermediate, or the locally-RMS-matched point, is unmeasured).
* **The LR brackets at 3× were not re-run under the corrected scoring** — the demotion of the ~0.02 bracket margins (§1) is an instrument-band argument, not a re-measurement.
* **Probe scale, one protocol** — everything above is the 20M probe + the 24-step cooldown extension; the 2700 s board and C14+ scale remain untested, as every prior lane stated for its own numbers.

## Provenance and receipts

No model/semantics change; scoring/optimizer-surface only. New files only: `protocol_correction.py`, `run_protocol_correction_leased.sh` (this lane's own cards 4/5; the K11 lane's cards 0/1 and the op-audit lane's cards 2/3 untouched), `tests/test_x45_protocol_correction.py`. No `k10_*`/`k11_*` source, PREREGISTRATION, `lattice.py` or GDN2 model code was edited — the banked builders (`velocity_ladder._build_k10`, `matched_arms.build_gdn2`), the prior lanes' routing splits and `bakeoff_optimizers.HybridMuonX45` are imported VERBATIM. Receipts: `protocol_correction_runs/<label>/` (each: `start.json` with the full convention block, `training.jsonl` live per-step with phase + LR factor, `scores.jsonl`, `result.json` with all three scorings, the measured RMS pair, the mechanism diagnostics and the step-1/step-180 banked-equivalence receipts), the assemble summary `receipts/protocol_correction_summary.json`, the two smoke receipts (`protocol_correction_runs/smoke_*`). Session lambda01, cards 4/5, RTX 6000 Ada, torch 2.9.1+cu128. Committed per arm as each landed.

## Commands

```bash
# one arm, detached on a leased card from this lane's pool (cards 4/5)
setsid bash -c 'OMP_NUM_THREADS=4 \
  experiments/expressivity_tasks/x45_matrix_emender/run_protocol_correction_leased.sh \
  winner_speedrun_rescored \
  > …/protocol_correction_runs/winner_speedrun_rescored.console.txt 2>&1' &

# the assemble: the corrected ladder + the gap re-measurement + recipe deltas
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
  experiments.expressivity_tasks.x45_matrix_emender.protocol_correction assemble

# the lane's tests (CPU; the conventions, the four arms' configs, the RMS
# receipt, the banked-artifact bindings)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
  tests/test_x45_protocol_correction.py -q
```
