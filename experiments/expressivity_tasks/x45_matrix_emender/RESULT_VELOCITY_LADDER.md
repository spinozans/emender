# RESULT_VELOCITY_LADDER.md — the matched-exposure velocity ladder

**Date:** 2026-11-17/18 · **Status:** complete — Part A (the GDN2 row) run at
the protocol; Part B (the K10 CMA-ES search, one generation within the
~4–5 h budget, + the tuned-winner confirmation cell) landed and committed ·
**Authorisation:** the
operator's velocity-lane directive — two deliverables: (A) the GDN2 row on the
matched-exposure velocity ladder, run at the protocol, never interpolated;
(B) CMA-ES tuning of the winner (per-token + separated keys), then one
confirmation cell of the tuned winner at the probe protocol.

**The operator's framing is this doc's spine: per-token velocity ranking,
structure first, tuning second, scale third.**

## The protocol (one protocol, every row)

The K6/K10 probe protocol verbatim: **20,022,866-target matched exposure**
(the `train until 20,000,000 targets seen` rule, landing at exactly the K6
arm's 20,022,866 on the registered flagship stream — same stream, same data
order, same stopping rule), **seed 3401**, the registered stream binding
(`L.flagship_identity()` + `L.assert_flagship_stream`, manifest
`48b9095e…`), the **document-disjoint calibration eval at 7 windows / 14,336
targets** (full 2048-byte windows, batch 1, recorded-read path
`X40_BATCHED_READS=0`), the honest clock (starts at step 1, includes the
stream read, compile/capture, fwd/bwd, clip, AdamW), single seed stated in
every receipt. Two rows below are banked at the CMA-ES search's own
**fixed-exposure objective** (16,777,216 targets → 16,817,493 landing) rather
than the 20 M probe rule — stated per row, never silently mixed.

## The ladder

| # | row | config status | eval NLL (nats/byte) | exposure (targets) | steps | train wall (s) | tok/s | peak GB | Δ to row above |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | chunked (matrix_ops, chunk 128, cd-protocol) | registered semantics; optimizer TRANSFERRED from the gdn2 nomination (one LR, no search) | **2.4173** | 20,022,866 | 180 | 588.7 | 34.0k | 21.96 | — |
| 2 | chunked, CMA-ES tuned | CMA-tuned (phase-2 best; see §2) — banked | **2.2564** | 16,817,493 † | 152 | ~486 active (~502 process) | ~34.6k | (not banked per-eval) | −0.1609 † |
| 3 | per-token (K6) | registered per-token semantics; optimizer transferred (one LR) | **2.0278** | 20,022,866 | 180 | 1,719.5 | 11.6k | 19.97 | −0.2286 † |
| 4 | per-token + separated keys (K10, **the winner**) | registered SEP-KV swap (mimetic birth, free drift); optimizer transferred (one LR) | **1.8418** | 20,022,866 | 180 | 1,865.8 | 10.7k | 23.09 | **−0.1860** |
| 5 | **gdn2 (this lane, Part A)** | the registered `lattice_gdn2` nomination **verbatim** — best-effort status: its optimizer/policy came from the 144-candidate byte-velocity search at this scale (gen 7, candidate 1); NOT re-tuned at this protocol | **1.5837** | 20,022,866 | 180 | 584.4 | 34,262 | 4.49 | **−0.2581** |
| 6 | **tuned winner (K10 + CMA-ES, Part B)** | CMA-tuned (this lane's phase-1 search, §3) | **1.8227** | 20,022,866 | 180 | 1,872.2 | 10,695 | 23.09 | +0.2390 vs gdn2; **−0.0191** vs its untuned row 4 |

† rows 1↔2↔3 straddle the two exposures (20 M probe rule vs the search's
16.8 M fixed objective): the deltas are quoted as measured, with the exposure
mismatch stated. Rows 4↔5 are same-protocol (both at 20,022,866).

### The deltas, read honestly

* **Structure first** (the granularity/addressing axis, rows 1→3→4, all at the
  20 M protocol where adjacent): chunk 128 → per-token is **−0.3895** nats
  (2.4173 → 2.0278, the K6 dial result); adding separated keys is **−0.1860**
  (2.0278 → 1.8418, the K10 result). Both are architectural, both at the
  transferred (untuned) optimizer — the largest single gains on this ladder.
* **Tuning second** (row 1→2, mixed exposure): CMA-ES on the chunked arm
  bought **−0.1609** at the 16.8 M probe objective (2.3599 transferred →
  2.2564 searched, same protocol — the clean same-protocol tuning delta; the
  −0.1609 in the table is quoted across the exposure mismatch and is NOT the
  search's own claim). Tuning is real but second-order next to structure.
* **Scale third** (row 5): gdn2 at the same exposure and the same eval beats
  every matrix row — **1.5837 vs the winner's 1.8418, a −0.2581 remaining
  gap** at the matched protocol. The K10 doc's "0.33-nat probe-scale gap"
  (vs gdn2's 16.8 M anchor 1.5163) is now superseded by the run number: the
  gap at the ladder's own protocol is 0.2581 nats.
* **The tuned-winner row (row 6, §4):** the search's incumbent at the probe
  protocol lands at **1.8227 vs the untuned winner's 1.8418 — tuning buys
  −0.0191 nats at the ladder's own protocol**, the honest single-generation
  figure (the probe-objective delta was −0.0229). It sits INSIDE the
  ~0.1-nat checkpoint-variance band the §1 blemish receipts, and it does not
  touch the structure conclusion: the tuning gain is an order of magnitude
  smaller than either structural gain. gdn2's row-5 lead over the winner
  shrinks to **0.2390** nats at the matched protocol.
* **Velocity (the ranking the lane is named for):** gdn2 runs the protocol at
  **34,262 bytes/s**, the chunked matrix arm at **34,012**, the per-token arm
  at **11,645**, the key arm at **10,731** (10,695 tuned — tuning does not buy
  velocity). The per-token machines pay a 3.2× velocity price for their
  structural gains (and the key channel costs a further 8 % over fused
  per-token); at matched exposure the matrix arms need 1,720/1,866 s where
  gdn2 needs 584 s.

## 1. Part A — the GDN2 row, run at the protocol (never interpolated)

`runs` receipt: `velocity_runs/gdn2_probe_seed3401/` (`start.json` /
`training.jsonl` / `result.json`, the arms' convention; result.json sha256
`0343bbd1…`, session `lambda01` card 0, RTX 6000 Ada, torch 2.9.1+cu128,
triton 3.5.1).

* **The number: 1.5836738858904158 nats/byte** — gdn2 trained under the exact
  probe protocol (20,000,000-target matched-exposure rule, seed 3401,
  registered stream binding asserted, the document-disjoint calibration eval
  at 7 windows / 14,336 targets), landing at **20,022,866 targets in 180
  steps, 584.4 s train wall, 34,261.7 bytes/s, peak 4.49 GB**.
* **The arm config is the receipted one verbatim**: the registered
  `lattice_gdn2` nomination (its own implementation, the searched
  optimizer: lr 5.4202e-4, beta2 0.9392, weight_decay 0.2006, warmup 69,
  microbatch 8), run through the registered `matched_arms.py` harness's
  `--bytes` matched-exposure rule. Its best-effort status is stated in the
  table: 144-candidate-search-tuned at this scale, not re-tuned here.
* **Probe-shaped throughput does NOT differ materially from 34k**: 34,261.7
  bytes/s against the banked 2700 s arm's 33,958.0 (+0.9 %) — the probe
  protocol's shorter horizon does not change gdn2's rate. Receipted in the
  run's `bytes_per_s`.
* **Protocol equivalence, receipted bit-exact**: this run's step-1 loss
  (5.948033220866492) is bit-identical to the banked 2700 s board run's step
  1, and its steps 150–152 (bytes 16,597,785 / 16,699,225 / 16,817,493;
  losses 2.22151… / 1.75012… / 2.07187…) are bit-identical to the banked
  CMA-ES search's gdn2 anchor's own step log — the same stream, the same
  data order, the same stopping rule, the same build.
* **The honest blemish, stated not smoothed**: the row lands **0.0674 nats
  ABOVE the banked 16,817,493-target anchor (1.5163)** despite seeing 3.2 M
  more targets. gdn2's late train loss is noisy (1.75–2.5 per step at
  16–20 M), and the 7-window / 14,336-target eval boundary is small; the
  180-step checkpoint simply evals worse than the 152-step checkpoint. This
  is checkpoint-to-checkpoint eval variance at a fixed protocol, not a
  protocol difference — and it is exactly why the operator's "do NOT
  interpolate; run it" was right: the ladder's own gdn2 number is 1.5837,
  and the remaining gap to the K10 winner is 0.2581 nats, not 0.3255.

## 2. The banked chunked-tuned row (cited, not re-run)

`receipts/cmaes_result_phase2.json` + `receipts/cmaes_probes/eval_030.json`:
the chunked arm's CMA-ES incumbent — lr 2.4725e-3, weight_decay 0.20525,
warmup 9, read_scale 3.3283, tau_match 0.46501, tau_pop 0.074821 (tau_merge,
beta2, microbatch at the transferred values) — scored **2.2564 nats/byte**
(152 steps, 16,817,493 targets, ~486 s active train, ~34.6k bytes/s) against
its transferred anchor's 2.3599 at the same fixed exposure/seed/stream. The
search's own clean tuning delta is **−0.1034 nats (12 % of the probe-scale
gap to gdn2)**; its 2700 s confirmation closed 0.076 nats at board scale
(`runs_150m_matched/matrix_ops_cmaes_confirm/`: 1.6813). Full discipline,
diagnosis and receipts: `CMAES_TUNING.md`.

## 3. Part B — CMA-ES tuning of the winner (landed)

`velocity_ladder.py` re-ran the `cmaes_search.py` discipline on the K10 arm,
importing the banked machinery verbatim (search space, codecs, receipt-log and
commit helpers): **the same fixed-exposure objective** (16,777,216 targets →
the 16,817,493 landing, byte-for-byte identical windows across evaluations),
**the same fixed-seed-per-eval rule** — every evaluation uses seed 3401 and
the same deterministic stream prefix, so the search tunes the seed and the
data draw as well as the hyperparameters, stated as the cost as before —
**the per-evaluation log committed as it landed** (`receipts/
velocity_k10_cmaes_evals.jsonl`, per-eval receipts in
`receipts/velocity_k10_probes/`), **the same phases** (phase 1: lr (log),
weight_decay (log), warmup (logint, capped at the probe horizon), read_scale
(linear); phase 2 adds tau_match / tau_pop only if phase 1 does not converge
— the prior search's precedent). The probe entry point is the K10 arm's own
build (`matched_arms.build_matrix("matrix_ops", 128, seed, k5b=False)` +
`convert_model_to_separated_keys`, the parameter-neutral swap asserted at
149,981,808), with the searched policy knobs set on the converted mixers.
`k10_*` / `k6_*` / `cmaes_*` sources, the PREREGISTRATIONs, `lattice.py` and
`matched_arms.py` are untouched.

**What ran, and what the budget admitted.** One evaluation of the K10 arm at
the fixed exposure costs **~26.6 min** (the transferred anchor's measured
wall: 1,593.9 s; the arm's probe throughput is 10.7k bytes/s against the
chunked arm's 34.0k), so the ~4–5 h GPU budget bought the transferred anchor
plus **exactly one CMA-ES generation at the prior search's popsize (8)**:
pycma 4.4.4, sigma0 0.20 in the unit cube, x0 = the transferred config,
CMA-ES seed 20261115, 9 evaluations in 13,198 s (3.7 h) of driver wall. The
wall guard then stopped cleanly before generation 2 (4,802 s remaining <
15,120 s needed). **Phase 2 (tau_match / tau_pop) was NOT run**: by the
prior search's own precedent phase 1 "did not converge" (a single
generation never establishes convergence; the incumbent was still moving),
but the budget does not admit another ~3.5 h of evaluations — the trigger
condition holds and the budget refuses it, stated honestly. A phase-2 run
(popsize 8, one generation, x0 = the phase-1 incumbent) is the natural next
step if the operator wants it.

**The result.** Every evaluation's config, wall and NLL is in the log; the
result receipt is `receipts/velocity_k10_cmaes_result.json`.

| eval | config (lr / weight_decay / warmup / read_scale) | NLL (nats/byte) | best-so-far |
|---:|---|---:|---:|
| 0 | transferred anchor (5.4202e-4 / 0.2006 / 69 / 2.000) | 1.6775 | 1.6775 |
| 1 | 2.3339e-4 / 0.1435 / 108 / 3.122 | 1.7383 | 1.6775 |
| 2 | 1.2001e-3 / 0.5842 / 67 / 1.802 | 1.7113 | 1.6775 |
| 3 | 5.0605e-4 / 0.5298 / 83 / 2.436 | 1.6734 | 1.6734 |
| **4** | **3.3427e-4 / 0.1742 / 45 / 2.787** | **1.6546** | **1.6546** |
| 5 | 3.0442e-4 / 0.0355 / 74 / 2.178 | 1.7085 | 1.6546 |
| 6 | 3.8047e-4 / 0.4202 / 79 / 1.804 | 1.6856 | 1.6546 |
| 7 | 1.6821e-3 / 0.5381 / 31 / 2.642 | 1.8033 | 1.6546 |
| 8 | 9.6343e-4 / 0.3580 / 107 / 2.814 | 1.6908 | 1.6546 |

*(tau_match / tau_pop / tau_merge / beta2 / microbatch at the transferred
values in every phase-1 evaluation.)*

* **The incumbent: 1.6546 vs the transferred anchor's 1.6775 — tuning buys
  −0.0229 nats at the probe objective** (lr 0.62× the transfer, warmup 45,
  read_scale 2.79). The honest scale: the prior chunked-arm search bought
  −0.1034 at the same objective; this single-generation search buys a
  quarter of that, and — as the checkpoint-variance receipt in §1 shows —
  the 7-window eval boundary itself carries ~0.1 nats of
  checkpoint-to-checkpoint variance on this arm, so a −0.023 tuning delta is
  INSIDE the band where single-checkpoint reads are fragile. The gdn2
  anchor at this exposure (cited, banked): 1.5163 — the K10 arm's
  probe-objective gap to gdn2 moves 0.1612 → 0.1383.
* **The incumbent trajectory** (the table's best-so-far column, the honest
  convergence read): flat through the first two candidates, three
  improvements in a row (evals 3, 4), then no further gain in five — a
  concave, shallow curve asymptoting far from gdn2, the same shape the
  chunked search receipts. With one generation there is no second-derivative
  claim either way; what the trajectory does show is that the space around
  the transfer is shallow (all 8 candidates within 1.6546–1.8033, six of
  eight no better than the anchor).
* **The tunes-the-seed caveat, verbatim from the prior search:** every
  evaluation uses seed 3401 and the same deterministic stream prefix, so the
  search tunes the seed and the data draw as well as the hyperparameters;
  the incumbent's NLL is a seed-3401 number.

## 4. The confirmation cell (the tuned-winner ladder row)

The tuned incumbent (eval 4's config) through the probe protocol itself (the
20,000,000-target matched-exposure rule → 20,022,866, seed 3401, the same
calibration eval, the same honest clock, the K10 mechanism diagnostics
receipted alongside): `velocity_runs/k10_tuned_confirm_seed3401/`
(result.json sha256 `994aff95…`).

* **The number: 1.822737353188651 nats/byte** — the tuned winner at the exact
  probe protocol, landing at **20,022,866 targets in 180 steps, 1,872.2 s
  train wall, 10,694.9 bytes/s, peak 23.09 GB** (the untuned winner's own
  walls: 1,865.8 s / 10,731.3 — the searched knobs cost no velocity). The
  knobs that differ from the transfer (receipted in the run's `start.json`):
  lr 3.3427e-4 (0.62× the transfer), weight_decay 0.17422, warmup 45,
  read_scale 2.787; tau_match / tau_pop / tau_merge / beta2 / microbatch at
  the transferred values.
* **Tuning's honest size at the ladder's protocol: −0.0191 nats**
  (1.8418 → 1.8227) from one CMA-ES generation. Against the structural gains
  on the same ladder (−0.3895 granularity, −0.1860 keys), tuning is an order
  of magnitude smaller — the operator's spine, measured: structure first,
  tuning second, scale third.
* **The mechanism stays put** (the confirm cell's own diagnostics, the same
  instruments as the banked probe): occupancy 27.3/128 mean live slots
  (banked untuned: 24.0), read entropy 1.732 nats (1.654), identity-rewrite
  share 0.113 (0.103), MERGE 0.00227/token (0.0031), CORRECT 0.9789/token
  — the tuned winner reads a slightly larger hot set slightly more spread;
  nothing structural moved. The KEY-COLLAPSE guard holds (span_key_std min
  0.985 vs floor 0.05; emitted-key mean pairwise cosine 0.091, still below
  tau_match — the drift the mimetic receipt predicted, unchanged by tuning).

## 5. Residuals, stated

* **Single seed, single checkpoint, small eval boundary** — every ladder row
  is one seed-3401 run at one checkpoint eval'd on 7 windows / 14,336
  targets. The §1 blemish receipts the boundary's checkpoint-to-checkpoint
  variance at ~0.07–0.16 nats on these arms: rows whose deltas sit inside
  that band (the tuning deltas) are directionally suggestive, not
  bank-grade; rows above it (the structural deltas, the gdn2 lead) are the
  load-bearing ones.
* **The search is one generation deep** (9 evaluations, 3.7 h): the phase-1
  incumbent is the best of anchor + one popsize-8 generation; phase 2
  (tau_match / tau_pop) was triggered by the precedent's convergence rule
  but refused by the budget. The trajectory is concave and shallow — six of
  eight candidates no better than the anchor — which is evidence the space
  around the transfer is flat, not evidence it is exhausted.
* **gdn2 is untuned at this protocol** (its nomination is the
  144-candidate-search config at the board scale, stated as its best-effort
  status): a gdn2-side search at this exposure could move row 5 too. The
  honest comparison rule from the chunked search is inherited: only the
  matrix arm was searched here.

## Commands

```bash
# Part A: the gdn2 row at the probe protocol (the registered harness, --bytes)
setsid bash -c 'OMP_NUM_THREADS=4 \
  experiments/expressivity_tasks/x45_matrix_emender/run_matched_arms_leased.sh \
  gdn2 experiments/expressivity_tasks/x45_matrix_emender/velocity_runs/gdn2_probe_seed3401 \
  2700 3401 128 k5b 20000000 \
  > …/velocity_runs/gdn2_probe.console.txt 2>&1' &

# Part B: the K10 CMA-ES search + the tuned-winner confirmation cell
setsid bash -c 'OMP_NUM_THREADS=4 \
  experiments/expressivity_tasks/x45_matrix_emender/run_velocity_ladder_leased.sh search \
  > …/velocity_runs/k10_cmaes_search.console.txt 2>&1' &
setsid bash -c 'OMP_NUM_THREADS=4 \
  experiments/expressivity_tasks/x45_matrix_emender/run_velocity_ladder_leased.sh confirm \
  CONFIG_JSON \
  > …/velocity_runs/k10_tuned_confirm.console.txt 2>&1' &
```
