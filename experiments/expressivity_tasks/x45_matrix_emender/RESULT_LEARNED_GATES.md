# RESULT — K11 LEARNED GATES: the honest null at the matched-exposure probe

**Stage:** the K11 learned-gates probe (gate C), run to completion under the
standing amendments.  **Registration:** `PREREGISTRATION_LEARNED_GATES.md`
(the M-1..M-5 predictions and the refutation clause were committed before
any probe measurement).  **Receipts:** `k11_runs/<arm>_seed<seed>/result.json`
(this document quotes them; the assemble summary is
`receipts/k11_learned_gates_summary.json`).  **Engine:** committed at
`4cb6623a`, untouched; the gate receipts are `receipts/k11_gates.json` +
`K11_GATES.md` (gates A/E/B/D, all green; the committed suite re-run green
in this session, 21 passed).

## 0. The protocol, as run

Every arm is the banked winner config VERBATIM (the K10 winner build, hybrid
Muon at 4× the tuned AdamW LR, the banked `banked_gates_adamw` routing with
the gate projections on the AdamW side — the DECAY gate's own convention,
cross-checked name-for-name against the banked split in-process) plus the
arm's gate channels at the mimetic-neutral birth, at the matched-exposure
probe protocol (the 20,022,866-target landing, 180 constant-LR steps, seed
3401, the registered flagship stream, the document-disjoint calibration eval
at 7 windows / 14,336 targets), scored under the corrected protocol
(`RESULT_PROTOCOL_CORRECTION.md`): the 24-step linear cooldown PRIMARY,
EMA-0.99 at step 180 SECONDARY (ranking-only), constant-180 retained for
banked comparability.  **Both comparison anchors are quoted everywhere:**
the constant-LR banked best **1.4899048124040877** (seed 3402; the same
config's seed-3401 row is **1.4907409463609969**) and the corrected
post-cooldown banked best **1.4338248457227434** (seed 3402).  The
**0-flip-at-birth receipt held at probe scale in every run**: each arm's
step-1 train loss is BIT-IDENTICAL to the banked same-seed row's
(5.785070835425993 at 3401, 5.768619428012168 at 3402) — the arms are born
as the banked winner exactly, and every difference below is the gates'
training, not plumbing.

## 1. The corrected ladder

| arm (seed 3401 unless stated) | params added | constant-180 | post-cooldown (PRIMARY) | EMA-180 (secondary) | cooldown drop |
|---|---:|---:|---:|---:|---:|
| **banked winner (3401)** | — | **1.4907** | — | — | — |
| **banked winner (3402)** | — | **1.4899** | **1.4338** | **1.5408** | 0.0561 |
| G1 write+read-out (3401) | +55,488 | 1.4995 | **1.4260** | 1.5155 | 0.0736 |
| G2 dense (3401) | +27,744 | 1.4957 | 1.4412 | 1.5252 | 0.0544 |
| G3 combined (3401) | +83,232 | 1.4837 | 1.4283 | 1.5131 | 0.0554 |
| **G1 second seed (3402)** | +55,488 | **1.4837** | **1.4304** | 1.5268 | 0.0533 |
| banked GDN2 (3401, context) | 27.4M gate params | 1.3951 | 1.3522 | 1.3803 | 0.0429 |

**The corrected-ladder row (the best arm by the primary score):**
G1 at seed 3401 — post-cooldown **1.4260**, vs the corrected anchor
1.4338 (−0.0079, cross-seed), vs the banked constant same-seed 1.4907
(+0.0088), vs GDN2's post-cooldown 1.3522 (+0.0738).

**The second-seed read (seed-matched, the decisive comparison):** G1 at
seed 3402 — post-cooldown **1.4304** vs the seed-matched corrected anchor
**1.4338** (−0.0035), constant-180 **1.4837** vs the banked **1.4899**
(−0.0063), EMA 1.5268 vs 1.5408 (−0.0140).  The primary-score seed delta on
the arm (+0.0044) is inside the winner's own measured seed delta (0.0119).

**The read: PARITY, on every scoring, in every arm.**  Every seed-matched
delta is ≤ 0.0063 in magnitude — 12–20× inside the 0.075-nat
endpoint-variance band — and the EMA cross-check agrees in sign and size.
No arm beats the banked winner outside the band under any of the three
scorings; no arm loses to it either.  GDN2's post-cooldown row remains
0.074–0.111 nats ahead of every gate arm: **the ~0.08 machine gap is
untouched by this lever.**

## 2. The mechanism adjudication — M-1..M-5, honestly

| prediction | the pre-stated bar | measured (step-180 state) | verdict |
|---|---|---|---|
| M-1 NLL improves on the winner's matched-surface row | better than the banked row (band −0.02..−0.25 vs the registration's 1.5146 anchor; amended: the banked winner config) | all arms within ±0.009 of the seed-matched banked rows on every scoring | **NOT BANKABLE — parity inside the 0.075 band** (nominally below the corrected anchor on G1/G3, above on G2; the second seed confirms: −0.0035) |
| M-2 identity share falls ≥ 0.03 below the 0.123 anchor | ≤ 0.093 | G1 0.1200, G2 0.1233, G3 0.1212, G1@3402 0.1154 (banked same-config: 0.1222/0.1180) | **NOT MET — unchanged** (all within 0.008 of the seed-matched banked anchor) |
| M-3 read entropy > 1.8 nats (effective rows up from ≈5.3–5.6) | > 1.8 | G1 1.5239, G2 1.5083, G3 1.5072, G1@3402 1.6123 (banked: 1.5150/1.6002) — effective rows 4.5–5.0 vs the banked 4.55–4.95 | **NOT MET — unchanged** |
| M-4 ≥ 10 of 24 layers at ΔNLL ≥ 0.02 in the read-mask sweep | ≥ 10 (anchor: 7 layers ≥ 0.02, 17 near-decorative) | G1 5/24, G2 5/24, G3 6/24 at ≥ 0.02; near-decorative 17/17/16 of 24 | **NOT MET — unchanged** (the AdamW-era anchor's own caveat stands: the muon4x-era sweep was never banked) |
| M-5 op mix | NO PREDICTION (P2) | CORRECT 0.9746/token, APPEND 0.023, MERGE 0.0023–0.0026, POP 0.0 — the banked mix within noise | reported |

The mechanism numbers are **dead flat**: the identity-rewrite share, the
read entropy / effective rows, the occupancy (33.5–34.9 vs the banked
33.5–35.2) and the read-mask sweep all sit within noise of the banked
winner's own same-config measurements, on every arm, at both the step-180
and post-cooldown states.

## 3. The gate telemetry — the controller is live, and it stays near identity

The learned admission controller WORKS and is per-layer heterogeneous — this
is not a dead channel (the gradient-reaches-the-zero-born-scalar receipts are
the committed tests', and the telemetry moved off birth on every arm):

* **write gate** `g_write = 1 − m_w·σ(w_w·x+b_w)`: mean distance from 1
  per layer 0.010–0.076 (G1/G3 mean 0.034); the most-open layer (layer 0)
  writes some tokens at 0.877 strength, the most-closed (layer 5) at 0.995;
* **read-out gate**: mean distance from 1 ≈ 0.024, layers 0.017–0.031;
* **dense deposit scale** (G2/G3): 0.0015–0.024 of payload scale, mean
  ≈ 0.022–0.024 — the dense channel fires on every token (P3's live-guard
  receipt) but at a small learned magnitude.

Given per-head admission control at probe scale, **the model opens the gates
only ~3% off identity** — and the store's statistical structure follows the
gates: still byte-identity CORRECT on ~97.5% of tokens, still ~4.6 effective
read rows, still 17 near-decorative stores.  The honest dissociation note
(the registration's own clause): NLL moved (inside noise) while NO
mechanism number moved — reported as measured.

## 4. The verdict — the refutation clause, applied verbatim

The registration's clause: *"if NO mechanism number moves and NLL does not
improve, the learned-gate lever does not own the residual and that is
reported as refuted, not excused."*  No mechanism number moved; NLL did not
improve beyond parity.  **The learned-gate lever — the per-head scalar
write-strength / read-out / dense-deposit surface — does not own the
~0.08-nat residual at probe scale.**  Per the task's own pre-stated reading,
this null is a result: it **confines the gap to GDN2's denser write
geometry** (per-CHANNEL gate matrices, ~18% of its parameters, a dense
channel-gated rank-1 delta at every token from step 1) rather than to our
per-head scalar gate surface (0.02–0.06% of parameters).  The registration's
§7.2 named variant — per-slot / per-channel gates — is the natural next arm
if the operator wants the lever retried at GDN2's granularity; the
probe-scale caveat (180 steps; the gates may need a longer horizon to open)
is stated and is exactly what a board-scale sign-off would have to weigh.

## 5. Walls (gate D + the arms' own clocks)

Block-call walls at the registered shape (same session, backward-only on
both sides): k11 24.81 ms fwd / 19.12 ms bwd vs K10's own launch
18.02/17.13 — **1.38× forward, 1.12× backward** (the 3-lane schedule
traffic and the blend chain; the kernel's 255-register / 2-spill report is
in `K11_GATES.md`).  The arms' own honest training clocks vs the banked
rows' (same box; the banked rows' own receipts): G1 2089 s (banked 3401:
1904 s, 1.10×), G2 2110 s (1.11×), G3 2254 s (1.18×), G1@3402 2087 s
(banked 3402: 1906 s, 1.10×) — the gate channel's cost is stated, not
hidden, and is the price of the 1.38× forward block wall plus the read-out
gate's mixer path.

## 6. Honest scope

* Probe scale only (20M-target matched exposure, 180+24 steps); no
  board-scale (2700 s) claim — the registration's own gate, pending the
  operator's sign-off.
* The arms ADD parameters (G1 +0.037%, G2 +0.018%, G3 +0.055% — inside the
  0.1% capacity tolerance, asserted in every receipt); the comparison is
  parameter-matched to within tolerance, not to the parameter, and the
  asymmetry to GDN2's 18% gate-matrix share is the finding, not a caveat to
  be dropped.
* The mechanism anchors are era-mixed and stated as such: M-2/M-3's
  registration anchors are one surface back (the bake-off muon3x rows:
  identity 0.123, entropy 1.664); M-4's sweep anchor is two surfaces back
  (the side-by-side's AdamW-era sweep: 7 layers ≥ 0.02, 17/24
  near-decorative).  The same-config seed-matched anchors (identity
  0.1222/0.1180, entropy 1.5150/1.6002, measured on the banked winner
  config itself) are quoted alongside every read, and every verdict holds
  against both anchor sets.
* The K12-flagged multi-lane MERGE corner in the k11 oracle was assessed
  and receipted before the arms ran (gate E, `K11_GATES.md` §2): unreachable
  by every committed and receipt fixture, inert in production (the kernel
  guards per lane) — no receipt here is touched by it.

## 7. Commands

```bash
# one arm (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k11_learned_gates_leased.sh g1 3401 \
  > .../k11_runs/g1_seed3401.console.txt 2>&1 < /dev/null &
# the assemble (the corrected ladder + the mechanism adjudication):
.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.\
k11_probe_learned_gates assemble
```
