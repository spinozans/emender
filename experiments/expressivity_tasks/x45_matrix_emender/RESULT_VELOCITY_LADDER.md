# RESULT_VELOCITY_LADDER.md — the matched-exposure velocity ladder

**Date:** 2026-11-17/18 · **Status:** Part A (the GDN2 row) landed and committed;
Part B (the K10 CMA-ES search + the tuned-winner confirmation cell) in flight —
this doc is committed per landing, as directed · **Authorisation:** the
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
| 6 | **tuned winner (K10 + CMA-ES, Part B)** | CMA-tuned (this lane's search, §3) | *in flight* | 20,022,866 | — | — | — | — | — |

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
* **Velocity (the ranking the lane is named for):** gdn2 runs the protocol at
  **34,262 bytes/s**, the chunked matrix arm at **34,012**, the per-token arm
  at **11,645**, the key arm at **10,731**. The per-token machines pay a 3.2×
  velocity price for their structural gains (and the key channel costs a
  further 8 % over fused per-token); at matched exposure the matrix arms get
  the same 20,022,866 targets in 1,720/1,866 s where gdn2 needs 584 s.

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

## 3. Part B — CMA-ES tuning of the winner (in flight)

`velocity_ladder.py` re-runs the `cmaes_search.py` discipline on the K10 arm,
importing the banked machinery verbatim (search space, codecs, receipt-log and
commit helpers): **the same fixed-exposure objective** (16,777,216 targets →
the 16,817,493 landing, byte-for-byte identical windows across evaluations),
**the same fixed-seed-per-eval rule** — every evaluation uses seed 3401 and
the same deterministic stream prefix, so the search tunes the seed and the
data draw as well as the hyperparameters, stated as the cost as before —
**the per-evaluation log committed as it lands** (`receipts/
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

**Budget honesty (the sizing constraint that shapes this search):** the K10
arm's probe throughput is 10.7k bytes/s against the chunked arm's 34.0k, so
one evaluation costs ~25–30 min where the chunked search's cost ~8.6 min. A
~4–5 h GPU budget buys the transferred anchor plus roughly ONE CMA-ES
generation at the prior search's popsize (8); the driver's wall guard stops
cleanly before any generation that does not fit, and the result receipt
(`receipts/velocity_k10_cmaes_result.json`) states exactly what ran. The
gdn2 anchor is CITED from the banked search (1.5163 at the identical fixed
exposure/seed/stream), not re-run. Whether phase 1 "converges" — and whether
a phase 2 is therefore triggered within the budget — is read from the
incumbent trajectory when it lands, reported as measured.

*This section is filled in as the search and the confirmation cell land.*

## 4. The confirmation cell (in flight)

The tuned incumbent through the probe protocol itself (the 20,000,000-target
matched-exposure rule → 20,022,866, seed 3401, the same calibration eval, the
same honest clock, the K10 mechanism diagnostics receipted alongside):
`velocity_runs/k10_tuned_confirm_seed3401/`. *Filled in when it lands.*

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
