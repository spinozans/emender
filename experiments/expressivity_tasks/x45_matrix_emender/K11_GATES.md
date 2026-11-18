# K11 — the learned-gates receipts: gates A, E, B and D at the registered shape

**Stage:** the K11 learned-gates lane's gate-receipt stage (the one casualty of
the prior lane, finished).  **Receipts:** `receipts/k11_gates.json` (this
document quotes them; every number below is from that file).  **Engine:**
`k11_learned_gates.py` (committed at `4cb6623a`, read-only here — engine bugs
escalate with receipts, never patch silently).  **Registered shape:** batch 8,
2048 tokens, 2 heads, rank 128, value_dim 128, key_dim 16, 64-token segments,
bf16 inputs / fp32 defined-order math, 3 recorded lanes.

## 0. The prior lane's one casualty, closed

The prior lane died at the session ceiling while debugging this instrument;
the flagged failure was gate B's fp64 reference path passing a Double
`key_lane` into an fp32 `keys.index_put` inside `gated_oracle_scan`
(`k11_learned_gates.py:326`) — a CALLER-side leaf-dtype bug (the fp64
reference leaves must be fp64 throughout the oracle), not an engine bug.
The working tree's `k11_gate_receipt.py` already carries the one-line
caller-side fix (`s64 = state.double()`, `k64 = keys.double()`, all six
fp64 reference leaves doubled) — the prior lane evidently applied it before
dying uncommitted.  This stage proves it green end-to-end rather than
asserting it: the fp64 reference path was exercised standalone (all leaves
fp64, no dtype error) and the full receipt below ran to completion — the
fp64 gradcheck passes and every fp32-vs-fp64 floor is measured, so the fp64
reference is fp64 throughout by receipt, not by inspection.

## 1. Gate A — oracle parity, the 0-flip bar, and the birth parity (P2/P5)

The kernel path vs the eager per-token oracle at the registered shape, in all
four GATED/DENSE combinations, on the K5A-bar fixture (512 fixed payload AND
write-key directions — the identity-cache regime), gates at random in
(0.05, 0.95), a random dense payload:

* **every channel bit-equal** (`torch.equal`) in all four combinations: the
  3-lane schedule fields (slot/erase/op/live_after), the payload lanes (the
  blended lane-0 value included), the read weights, the pre-decision
  liveness, the exit state/liveness/key table — **33,816,576 lanes compared,
  0 flips**;
* **the read** at the measured fp32-vs-fp64 floor: floor 1.91e-06 … 3.81e-06,
  kernel deviation ≤ the floor itself, inside the 8× noise factor in every
  combination;
* **the 0-flip decision-invariance receipt (P2):** identity gates + zero
  dense vs random gates + random dense — the discrete op sequence (lanes
  0/1) bit-identical, 0 flips;
* **the birth-parity receipt (P5):** k11 at unity gates and a zero dense
  payload vs **K10's OWN launch** on the same fixture — bit-identical on
  every channel (schedule, key table, liveness, state, reads, weights);
* the fixture's own op histogram at the registered shape: NONE 32,784 /
  APPEND 34,397 / CORRECT 31,123 / POP 0 / MERGE 0 (the honest statement
  that this random fixture exercises the two content-carrying ops and the
  decision chain; the MERGE and POP branches are exercised by the committed
  test suite's crafted fixtures);
* the compiled forward kernel's own metadata: 255 registers, **2 spills**
  (at `num_warps=8`; the spill count is receipted — the walk is at the
  register ceiling and the cost is measured in gate D, not hidden).

## 2. Gate E — the K12-flagged multi-lane MERGE corner: UNREACHABLE AND INERT

The K12 placement lane's cross-lane flag (commit `cb5615fe`,
`K12_PLACEMENT.md` §3): the K10 eager oracle's latent multi-lane MERGE bug —
the `if sel["merge"].any():` block index-puts zeros at `second_index` for
EVERY (batch, head) lane whenever ANY lane merges — was copied into the k11
oracle.  This gate receipts the assessment; **the engine is NOT patched**
(it is committed; per the lane's rules an engine bug escalates with receipts
rather than being patched silently).

* **The pattern's location:** `k11_learned_gates.py:336-352`
  (`gated_oracle_scan`; the flagged `keys.index_put` at `:326` is the lane-0
  write — the multi-lane one is the MERGE-source close below it).
* **The kernel guards per lane:** `_k11_seq_fwd_kernel` computes
  `hit2 = merge & keep_r & (second == ranks)` with `merge` a
  per-(batch,head)-lane scalar, so state/keys/live are zeroed only in the
  merging lane — production is per-lane-correct by construction.
* **The corner is real in the k11 oracle, and demonstrated:** on a crafted
  heterogeneous two-lane fixture (head 0 MERGEs — second-key similarity
  0.9 ≥ tau_merge; head 1 CORRECTs with a live second row), the oracle kills
  the NON-merging head's live second row and the kernel keeps it — the
  differing-element count is exactly 1 on each of state/live/keys (and 0 on
  every recorded schedule channel): oracle-only, exactly K10's pattern.
* **Reachability measured on the kernel path:** every committed
  oracle-parity/gradcheck fixture and every receipt fixture produces ZERO
  MERGEs (the identity-regime, registered-shape flip, gradcheck, zero-flip,
  gate_a registered 8×2048, gate_b floor and gate_b gradcheck fixtures:
  `merge_lane_fires = 0`, `corner_firing_nonmerging_live_second = 0` in
  all); the committed MERGE fixture is single-lane by construction
  (`heads=1`).  A corner firing in any parity fixture would fail LOUDLY
  (oracle-vs-kernel bit-equality), never silently corrupt a receipt.
* **Production is kernel-only at the registered shape:**
  `GatedSeparatedMixer.mix → gated_scan → x45::k11_scan_fwd` (the opaque
  kernel op) whenever `separated_available` — true at rank 128 / value_dim
  128 / key_dim 16.  The eager oracle is only the fallback for
  oversized/non-CUDA stores and this lane's receipt/gradcheck instrument.
* **Verdict:** UNREACHABLE AND INERT — no committed or banked K11 receipt is
  invalidated, and the arms may proceed.  Escalated to the operator with
  the K12 flag; the engine left unpatched per the escalation rule.

## 3. Gate B — gradient parity within measured floors

* **The fp64 gradcheck of the oracle passes** (`eps=1e-6, atol=rtol=1e-4`)
  through the payload, the dense lane, the GATES, the key channel, the value
  state and the key table, with the branch histogram NONE 196 / APPEND 200 /
  CORRECT 180 (POP/MERGE 0 on this fixture — the same branches the committed
  suite's gradcheck exercises; POP/MERGE adjoints are pinned by the
  committed structural tests).
* **The eight gradients vs the fp32-vs-fp64 floor** (fp32 kernel path vs the
  oracle's fp32 autograd, each read against the floor measured on the same
  fixture, 8× noise factor): every one within the factor — `dpayload`
  3.81e-06 dev at a 3.81e-06 floor, `ddense` 2.86e-06 at 1.91e-06,
  `dwrite_keys` 2.86e-06 at 2.38e-06, `dquery_keys` 1.91e-06 at 1.91e-06,
  `ddecay` 3.05e-05 at 1.67e-05, `dgates` 3.05e-05 at 3.05e-05, `dstate`
  3.81e-06 at 5.72e-06, `dkeys` exactly 0.0 at a 0.0 floor.
* **The replay receipt:** the 3-lane recorded schedule rebuilds the
  forward's own segment checkpoints BIT-EXACTLY through K4's imported
  `persistent_scan_trajectory`, dense lane included.

## 4. Gate D — walls at the registered shape (the honest same-session read)

Block-call walls, measured against **K10's OWN launch on the same fixture
and box** (not a citation), the backward timed backward-only on BOTH sides
(K10's own gate-D pattern; the schedule is computed once outside the timed
loop):

| path | forward ms/scan | backward ms/scan | overhead vs K10 |
|---|---:|---:|---:|
| k11 (gated + dense, 3-lane) | 24.81 | 19.12 | fwd **1.38×**, bwd **1.12×** |
| k10 (same session) | 18.02 | 17.13 | — |

Forward throughput 660,356 tok/s at the registered shape.  The gate channel
adds per token: one gate-scalar load, the CORRECT blend's
`sub/mul/mul/mul/add` chain (or one `mul` at APPEND), and — for the dense
arms — one (VP,) deposit load plus one masked row-add at the best row; the
2-spill register report of gate A is the cost's mechanism, and the measured
overhead is the cost's number.

## 5. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k11_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k11_gates.json
```

The gate-A/B/C fixtures of the committed engine suite:
`test_k11_learned_gates.py` (green at the engine commit `4cb6623a`).
