# K11B — the per-channel-gates receipts: gates A, E, B and D at the registered shape

**Stage:** the K11B per-channel-gates lane's gate-receipt stage — the one
casualty of the prior lane (the sweep was in flight when the lane died at
the 6 h ceiling; this stage ran it to completion and committed it).
**Receipts:** `receipts/k11b_gates.json` (this document quotes them; every
number below is from that file).  **Engine:** `k11b_channel_gates.py`
(committed at `0cec7202` with the 20/20 parity suite, read-only here —
engine bugs escalate with receipts, never patch silently).
**Registered shape:** batch 8, 2048 tokens, 2 heads, rank 128, value_dim 128,
key_dim 16, 64-token segments, bf16 inputs / fp32 defined-order math, **2
recorded lanes** (K10's own layout; no dense lane — K11's G2 receipted the
dense-write channel's null separately).

## 0. The prior lane's casualty, closed

The prior lane died at the session ceiling with the engine and its 20/20
test suite finished and committed (`0cec7202`) but the gate-receipt sweep
only in flight.  This stage ran `k11b_gate_receipt.py` to completion
detached-leased on this lane's own card pool (53 minutes end to end, card 1)
and committed the receipt (`16698e3b`).  The committed K10/K11 suites were
re-verified green **with the new module loaded** in the same session as the
engine commit (40 passed in 40:28; `k11b_runs/k10_k11_suites_console.txt`),
and the lane's own 20-test suite green at 32:54
(`k11b_runs/test_suite_console.txt`).

## 0.5 The falsifier's job receipt and the corner it caught (the ledger)

**The frozen-gates falsifier control is the instrument that caught a real
defect every parity receipt missed, and its lesson is now standing law in
the suite.**  The control's first run FAILED its step-1 bar: the frozen
p1 arm's step-1 train loss `5.785136661889908` vs the banked same-seed
row's `5.785070835425993` (delta 6.6e-5) — at exact-zero gate parameters,
so a forward-at-birth parity break, not gate learning
(`k11b_runs/p1_frozen_seed3401_prefixbreak/`, root-caused and receipted in
`k11b_runs/birth_diagnostic.json`, commit `57e1f942`):

* **Why every prior receipt missed it:** gate A/B's fixtures hand BOTH
  kernels *explicit* decay tensors — the parity comparison never went
  through `channel_decay`'s production expression under the bf16 autocast
  regime — and the committed mixer-level birth fixture's data never
  saturates the sigmoid to bf16 1.0.  **The lesson, stated plainly:
  bit-parity fixtures must go through the production expression under
  autocast.**  The new suite test
  (`test_gate_b_birth_survives_the_clamp_bound_carrier_corner` — a
  heterogeneous fixture with head 0's decay bias saturating the bf16
  sigmoid to exactly 1.0) is the standing remedy, and the 21-test suite is
  green with it (`a856362a`).
* **The corner, a receipt-worthy micro-finding:** the incumbent's clamp
  bound `1-1e-6` rounds to EXACTLY 1.0 in its bf16 carrier (its own
  effective ceiling), while `channel_decay`'s mul-by-factor promoted the
  per-channel product to fp32, whose clamp preserves `0.999999`.  At the
  probe's layer 4, ONE saturating bf16 position in 32,768 diverged at
  birth by exactly `1.0133e-6 = 1.0 - fp32(1-1e-6)`; that single-position
  ULP difference produced 11,044 differing reads and amplified through
  twenty more layers to the 6.6e-5 step-1 loss delta.  **ULP-scale birth
  defects do not stay ULP-scale — which is WHY the bar is bit-identity and
  not a tolerance.**
* **The fix (the supervisor's R1 ruling, option A):** the clamp bounds
  now act in the incumbent's carrier dtype (`9630f5c6`) — at birth
  `lam_vec` is bit-equal to the registered scalar on every channel,
  corners included; beyond birth the effective ceiling under autocast is
  the incumbent's own 1.0 (the disclosed deviation), and in fp32 contexts
  the paper bound survives.  The postfix proof: the converted p1 build's
  step-1 loss is `5.785070835425993` — BIT-IDENTICAL to the banked row
  (`k11b_runs/birth_diagnostic_postfix.json`).  The control then PASSED
  on the fixed engine (`f7051b72`): step-1 bit-identical, the per-step
  divergence exactly as registration 2.4 discloses (ULP-scale
  compounding, max |delta| 0.0434 at step 180), and the landing at
  parity (post-cooldown 1.4371 vs the 1.4338 anchor).
* The receipts in this document were RE-RUN on the fixed engine and are
  green throughout (only the timing fields moved); the engine change
  touches the birth expression only — the kernels, the oracle and the
  recorded-schedule channels are untouched, and their receipts are
  unchanged in substance.

## 1. Gate A — oracle parity, the 0-flip bar, and the birth parity (P2/P5)

The k11b kernel path vs the NEW eager per-token oracle
(`channel_gated_oracle_scan`) at the registered shape, in **both GATED
combinations**, on the K5A-bar fixture (512 fixed payload AND write-key
directions — the identity-cache regime), random per-channel gates in
(0.05, 0.95), the DECAY channel a per-channel vector (0.85–1.0 spread):

* **every channel bit-equal** (`torch.equal`) in both combinations: the
  2-lane schedule fields (slot/erase/op/live_after), the payload lanes (the
  BLENDED lane-0 value included — the recorded lane-0 payload is the
  post-gate write, so the recorded schedule replays through ordinary
  scheduled writes), the read weights, the pre-decision liveness, the exit
  state/liveness/key table — **16,908,288 lanes compared, 0 flips**;
* **the read** at the measured fp32-vs-fp64 floor: gated floor 1.91e-06,
  kernel deviation 9.54e-07; ungated floor 3.81e-06, deviation 1.91e-06 —
  inside the 8× noise factor in both combinations (the kernel's `tl.sum` vs
  the oracle's tree-fold, K10/K11's own floor convention);
* **the 0-flip decision-invariance receipt (P2):** the registered scalar
  broadcast per channel + unity gates vs random per-channel gates AND
  random per-channel decay vectors — the discrete op sequence bit-identical,
  0 flips (no gate value, scalar or vector, feeds op selection);
* **the birth-parity receipt (P5):** k11b at the registered scalar broadcast
  per channel + unity gates vs **K10's OWN launch** on the same fixture —
  bit-identical on every channel (schedule, key table, liveness, state,
  reads, weights; 0 differing elements everywhere) — the mul-by-1.0/clamp
  birth is the incumbent bit for bit at the registered shape;
* the fixture's own op histogram: NONE 32,768 / APPEND 1,645 / CORRECT
  31,123 / POP 0 / MERGE 0 (the honest statement that this random fixture
  exercises the two content-carrying ops and the decision chain; the POP
  and MERGE branches are exercised by the committed suite's crafted
  fixtures and gate E below);
* the compiled forward kernel's own metadata: 255 registers, **6 spills**
  (at `num_warps=8`) — the walk is at the register ceiling (K11's own gate
  A: 255 registers / 2 spills); the per-channel vector loads cost 4 more
  spills and the cost is measured in gate D, not hidden.

## 2. Gate E — the K12-flagged multi-lane MERGE corner: NOT COPIED

The K10/K11 eager oracle's latent multi-lane MERGE bug — the
`if sel["merge"].any():` block index-puts zeros at `second_index` for
EVERY (batch, head) lane whenever ANY lane merges (the K12 placement lane's
flag; `K11_GATES.md` §2) — is **NOT copied** into the k11b oracle: the new
file guards the MERGE-source close PER LANE, the kernel's own semantics.
No committed engine is touched.  On the K11 gate-E fixture verbatim (a
crafted heterogeneous two-lane fixture: head 0 MERGEs — second-key
similarity 0.9 ≥ tau_merge; head 1 CORRECTs with a live second row):

* **(a) the committed K11 engine's corner, reproduced:** the K11 oracle vs
  the K11 kernel diverges exactly at the NON-merging lane's live second row
  — state 1 / live 1 / keys 1 differing (and 7 differing schedule elements)
  — K11_GATES.md §2's receipt, reproduced on this fixture;
* **(b) the k11b oracle vs the k11b kernel: BIT-IDENTICAL** on the same
  fixture — 0 differing elements on state, live, keys, the schedule, and
  the payload lanes;
* the fixture's MERGE structure is exercised on the KERNEL path (histogram
  NONE 15 / CORRECT 15 / MERGE 2) — the per-lane guard is measured where it
  matters, not only in the oracle.

## 3. Gate B — gradient parity within measured floors

* **The fp64 gradcheck of the oracle passes** (`eps=1e-6, atol=rtol=1e-4`)
  through the payload, the **PER-CHANNEL DECAY vector**, the
  **PER-CHANNEL GATES**, the key channel, the value state and the key
  table, with the branch histogram NONE 192 / APPEND 12 / CORRECT 180
  (POP/MERGE 0 on this fixture — the same branches the committed suite's
  gradcheck exercises; POP/MERGE adjoints are pinned by the committed
  structural tests).
* **The seven gradients vs the fp32-vs-fp64 floor** (fp32 kernel path vs
  the oracle's fp32 autograd, each read against the floor measured on the
  same fixture, 8× noise factor): every one within the factor — `dpayload`
  1.91e-06 dev at a 1.91e-06 floor, `dwrite_keys` 2.15e-06 at 1.43e-06,
  `dquery_keys` 3.81e-06 at 3.81e-06, `ddecay` 3.81e-06 at 4.77e-06,
  `dgates` 4.77e-06 at 7.63e-06, `dstate` 3.81e-06 at 3.81e-06, `dkeys`
  exactly 0.0 at a 0.0 floor.  The per-channel DECAY adjoint
  (`dlam[c] = ⟨dS[:,c], S_t[:,c]⟩`) and the per-channel write/read gate
  adjoints are inside the same floor discipline as the incumbent channels.
* **The replay receipt:** the 2-lane recorded schedule rebuilds the
  forward's own segment checkpoints BIT-EXACTLY through the lane-specific
  PER-CHANNEL trajectory replay kernel (K4's defined-order spec, the
  per-channel form K4's imported scalar replay cannot ride — R3).
* **The two-step wake-up receipt:** at birth the zero-born per-head scalars
  receive step-0 gradient (decay 2995.90, write 1646.81, read 827.42 —
  through σ(0) = 0.5) while the projections' step-0 gradient is EXACTLY
  zero while m == 0 (the K11 two-step wake-up pattern) — receipted so the
  step-0 zero is not read as a dead channel — and the born decay vector is
  BIT-EQUAL to the registered scalar on every channel.

## 4. Gate D — walls at the registered shape (the honest same-session read)

Block-call walls, measured against **K10's AND K11's OWN launches on the
same fixture and box** (not citations), the backward timed backward-only on
ALL sides (the schedule computed once outside the timed loop, K10/K11's own
gate-D pattern):

| path | forward ms/scan | backward ms/scan | fwd vs | bwd vs |
|---|---:|---:|---:|---:|
| k11b (per-channel, 2-lane) | 23.10 | 18.29 | **1.274× K10** / 1.071× K11 | **1.057× K10** / 1.006× K11 |
| k11 (gated, 3-lane) | 21.57 | 18.17 | 1.190× K10 | 1.050× K10 |
| k10 (same session) | 18.13 | 17.30 | — | — |

Forward throughput 709,318 tok/s at the registered shape (the post-fix
re-run; the pre-fix run measured 23.32/18.26 ms and 702,668 tok/s — the
walls are the same within session noise, as expected for an engine change
that touches only the birth expression).  The per-channel
form adds per token: one (VP,) DECAY vector load and one (VP,) gate load
(vs K11's two scalars), the per-channel CORRECT blend (elementwise in the
value dim), and in the backward one (BV,) dlam and one (BV,) dgate store
per (lane, block, token) instead of the per-block scalar partials; gate A's
6-spill register report is the cost's mechanism, and the measured overhead
is the cost's number — the forward carries the vector-load cost (1.286×
K10) while the backward is nearly free (1.053× K10), which is why the arms
remain single-card probe-affordable.

## 5. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k11b_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k11b_gates.json
```

The committed engine suite (`test_k11b_channel_gates.py` — 20/20 green at
the engine commit `0cec7202`, re-run green in the harvest session at
32:54) and the committed K10/K11 suites re-run green with the module
loaded (40 passed, 40:28).  On the FIXED engine the lane's suite is
**21/21** (the falsifier's corner guard included, 39:26) and the K10/K11
suites **40/40** with the fixed module loaded (49:56)
(`k11b_runs/*_postfix.console.txt`).
