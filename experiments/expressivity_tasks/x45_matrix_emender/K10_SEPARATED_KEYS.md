# K10 — the separated-key arm at 150 M: the receipted SEP-KV fix, applied to real text at scale

Stage 10 of the X45 kernel lane, and the answer to the question this stage
existed for: **does the receipted key fix transfer to real text at 150 M
scale?** The chain: the 150 M board arm's residual was diagnosed as the
identity-cache regime (`CMAES_TUNING.md`: CORRECT 60-71 % of decisions firing
mostly as exact-match identity rewriting over repeated byte-level payload
directions); SEP-KV proved the cause at probe scale
(`RESULT_KEY_SEPARATION.md`: key=value fusion is mechanically role-blind — the
fused arm pinned at the derived 0.5 set-retrieval ceiling across all 48 depth-1
cells while the separated arm reached 1.000) and receipted the fix (small
separated write-context keys, key_dim 4-16 all 1.000; the free-key collapse
basin killed by mimetic birth — `RESULT_WHEELER_CAPACITY.md` §4b). The fix had
never touched real text at scale. **It transfers.**

The registration is the DRAFT `PREREGISTRATION_SEPARATED_KEYS.md` (a new
registration of ADDRESSING, not of the algebra; the operator signs it off
before any 2700 s claim — it builds on the K6 granularity draft's own pending
sign-off). Everything here is probe-scale evidence under that draft's gates.

**Headline, as narrowly as the evidence supports.**

* **The key arm beats the banked K6 fused per-token row: 1.8418 vs 2.0278
  nats/byte** (−0.1860) at the sweep's own matched-exposure protocol —
  20,022,866 targets (byte-for-byte the K6 arm's exposure, same stream, same
  data order, same stopping rule), seed 3401, one transferred LR, the same
  document-disjoint calibration eval, 180 steps. The pre-registered K-1 band
  was "improvement in [0.02, 0.35], honestly wide" — the landing is IN BAND.
  This is the largest single architectural gain at this protocol since the
  per-token limit itself (which bought −0.151 beyond chunk 4), and it closes
  36 % of the probe-scale gap the fused per-token arm still left to gdn2's
  1.5163 anchor.
* **The refutation clause does NOT fire**: the identity-rewrite share is NOT
  within noise of the fused anchors (0.103 vs 0.519 init / 0.134 paired
  ablation), and the NLL is not worse than 2.0278.
* **Gate A lands at the K6 bar**: the persistent kernel's decisions, payload
  lanes, read WEIGHTS, state, liveness, pre-decision liveness AND the key
  table are **bit-exact against the eager per-token oracle — 0 flips in
  17,336,320 lanes** at the registered shape (bf16, key_dim 16), the read at
  0.5× the measured fp32-vs-fp64 floor.
* **Gate B**: all six gradients (`dpayload`, `dwrite_keys`, `dquery_keys`,
  `ddecay`, `dstate`, `dkeys`) land within **0.67–2.0× the measured floor**
  against the oracle's own autograd, with the fp64 oracle first validated by
  a full `gradcheck` over every adjoint branch. Both segment recomputes (the
  value side through K4's replay, the key side through the key-lane replay)
  are bit-exact against the forward's own checkpoints.
* **Gate D, honestly**: the key channel's forward is FASTER than K6's
  (19.7 vs 26.5 ms per (layer, window) scan — the 16-wide key folds replace
  the 128-wide content score/similarity folds), the backward is 12 % slower
  (17.2 vs 15.3 ms — the key walk and the key-lane replay), and the whole-arm
  price is **10.7 k tok/s against the fused arm's 11.6 k** — an 8 %
  throughput cost stated, not hidden.

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_SEPARATED_KEYS.md` | the DRAFT registration (operator sign-off pending): the four pre-stated semantics, the arithmetic spec, the parameter-neutral swap, the claim set, the pre-registered mechanism predictions and the corrected baseline provenance |
| `k10_separated_keys.py` | the eager per-token oracle (`separated_sequential_oracle_scan`), the persistent forward kernel with the key-match side, the key-lane trajectory replay, the two backward walks (value + key), the `x45::k10_scan_fwd/bwd` op pair, the refusals, the mixer conversion (`convert_model_to_separated_keys`) and the mimetic family init |
| `test_k10_separated_keys.py` | 19 tests: gates A and B, the fp64 gradcheck, the POP/MERGE/refile adjoint boundaries, the born-spread watch, the refusals, determinism |
| `k10_gate_receipt.py` | the registered-shape receipts into `receipts/k10_gates.json` |
| `k10_probe_separated_keys.py` + `run_k10_probe_leased.sh` | the matched-exposure quality probe (gate C) with the identity-rewrite instrument, its two measured anchors, and its detached leased driver |
| `K10_SEPARATED_KEYS.md` | this document |

No `k6_*`/`k7_*`/`k8_*`/`k9_*` source, `PREREGISTRATION_*` file, `lattice.py`
or `matched_arms.py` was edited; K6's spec helpers and K4's trajectory replay
are imported, and the SEP-KV lane's `causal_context` is reused verbatim.

### 1.1 The semantics (the registration's §1, pre-stated)

The K6 per-token sequential machine with its ADDRESSING moved from content to
a small learned key channel: a `rank × 16` key table per head, one row per
slot, re-filed by the same schedule's write events (APPEND/CORRECT/MERGE
re-file the target row to the write's emitted key — a learned projection of
the last 4 positions of the mixer's input; MERGE zeroes the source row's key
with its value; POP clears the address); routing similarity AND read scores
on `cos(q_key, K_j)`; values, payloads, DECAY, growth-rule arithmetic and all
five ops' arithmetic UNCHANGED (the MERGE blend still uses the registered
detached VALUE strengths); keys NOT decayed; mimetic birth at a fixed seeded
QR-orthonormal family (only the current-token context position non-zero, the
same map for the write-key and query-key projections — born mimicking the
registered fused content-addressing, born spread) with free drift. The
content-query projection is REMOVED, and the swap is parameter-NEUTRAL at the
registered shape (147,456 out, 147,456 in per mixer; the model stays at
exactly the audited 149,981,808 parameters — asserted by the probe).

### 1.2 The arithmetic spec — inherited, extended, and one real Triton finding

K6's defined-order spec applies verbatim (balanced binary tree over each
reduced axis, fp32, correctly-rounded IEEE ops, libdevice exp, `mul.rn`
everywhere), extended to the key axis (`key_dim` 16, 4 tree levels). One
engineering finding is worth recording for every future kernel on this stack:
**folding a small `(rank × key_dim)` tile with `tl.sum`-based tree reductions
beside the live whole-store value tile collapses Triton's register allocator**
— the first build of the forward kernel compiled to 40 registers with 30,608
spills and 4,546 ms per scan (170× K6's wall). Replacing the key-axis tree
with a `tl.split`-based fold (`_split_fold_axis2`: pure elementwise `add.rn`
pair-sums, the SAME pairings and the same single IEEE adds as torch's
`tree_fold` — bit-identical by construction, and gate A's 0 flips over 17.3 M
lanes is the proof) holds the kernel at 255 registers / 4 spills and 19.7 ms.
The `tl.sum` reduction-layout machinery on a small tile beside a large live
tile is the trigger; the split form has no reduction op at all.

## 2. Gates A and B (the receipts: `receipts/k10_gates.json`)

**Gate A** — the registered shape (batch 8, 2048 tokens, 2 heads, rank 128,
value_dim 128, key_dim 16, bf16 inputs): every channel `torch.equal` against
the eager oracle — schedule fields (262,144 lanes), the payload lane with the
MERGE blend (8,388,608), the read weights (4,194,304), the pre-decision
liveness (4,194,304), the exit state/liveness and **the exit key table**
(65,536) — **0 flips in 17,336,320 lanes**; the read at 0.5× the measured
fp32-vs-fp64 floor; the KEY-COLLAPSE watch on the same fixture: span_key_std
1.00 (guard floor 0.05). The suite's 19 tests add: ragged token/segment
lengths, the bf16 regime, one-token/one-rank edges, score ties (lowest index
wins in both implementations), crafted key tables that force MERGE (two rows
with keys at cos 0.9) and POP (a full store, a write key orthogonal to every
row), the keys-not-decayed semantic, the all-dead zero read, determinism, and
the born-spread mimetic receipt (the family orthonormal, the projections
equal at birth, only the current position non-zero).

**Gate B** — the fp64 `gradcheck` of the oracle validates every adjoint branch
(NONE/APPEND/CORRECT present; POP/MERGE/refile boundaries pinned by the
structural tests: a POPped row's VALUE and KEY gradients terminate at its
death token and its POP write key carries nothing; a re-filed row's earlier
write key carries nothing from later reads while the re-file token's carries
them; the MERGE blend reaches BOTH source rows). The kernel path's six
gradients vs the fp32-vs-fp64 floor, measured on the same fixture: dpayload
1.33×, dwrite_keys 1.09×, dquery_keys 2.0×, ddecay 0.67×, dstate 1.33×,
dkeys exact (0/0) — all within `NOISE_FACTOR` (8×), none bit-exact by design
(the softmax adjoint's cross-axis reductions are autograd-ordered on the
oracle side — the K6 statement, now with two more outputs). The
segment-recompute contract holds on BOTH channels.

Two fixture degeneracies found by the tests are worth recording because they
are analytic, not bugs: a CONSTANT upstream makes the read output invariant
to the weights when all rows share a coordinate sum (the gradient is exactly
zero in both implementations); and a query key EXACTLY PARALLEL to its row's
key sits at the cosine's stationary point, so the score's gradient w.r.t. the
row key vanishes to first order. Both fixtures were fixed to probe off the
stationary points; the oracle and the kernel agreed exactly in both cases.

## 3. Gate C — the probe: the fix transfers

`receipts/k10_probe.json`. The protocol is the K6 probe's verbatim: the same
`matrix_ops` build order (seed 3401, the init shared bit for bit), the
parameter-neutral key swap asserted (149,981,808 before and after), the same
registered flagship stream, the same transferred LR, the same stopping rule
(20,000,000 targets — landing at exactly the K6 arm's 20,022,866), the same
honest clock, the same document-disjoint calibration eval (14,336 targets).

| arm | addressing | eval NLL (nats/byte) | Δ vs fused per-token | train wall (s) | tok/s | CORRECT/tok | live slots | read entropy (nats) | identity share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused per-token (banked, K6) | content | 2.0278 | — | 1,720 | 11.6k | 0.884 | 73.7 | 1.085 | (not banked) |
| fused at init (anchor a) | content | — | — | — | — | 0.975 | 41.3 | — | **0.519** |
| paired ablation (anchor b) | content over the trained key arm's payloads | — | — | — | — | 0.993 | ~48-66 | — | **0.134** |
| **separated keys (this arm)** | **write-context keys, dim 16** | **1.8418** | **−0.1860** | 1,866 | 10.7k | 0.979 | **24.0** | **1.654** | **0.103** |

### 3.1 The pre-registered predictions, evaluated

* **K-1 (the NLL improves on 2.0278) — MET, in band.** −0.1860, inside the
  pre-registered [0.02, 0.35] band (which was stated honestly wide because
  SEP-KV's necessity claim is per-layer and depth substitution is real at 24
  layers).
* **K-2 (the identity-rewrite share falls below BOTH anchors) — PARTIAL.**
  The key arm's share is 0.103, below the fused-at-init anchor (0.519) and
  below the paired ablation (0.134), and NOT within the 0.02 noise window of
  either — but it misses the pre-registered materiality band (≥ 0.10 below
  the LOWER anchor: 0.134 − 0.10 = 0.034 > 0.103). Reported as measured, not
  smoothed: the share falls (0.031 below the lower anchor, outside noise),
  the band does not. The receipts decompose the fall: **training alone**
  moves the value channel a long way (0.519 at init → 0.134 under content
  addressing over the trained payloads — the trained residual stream's
  payloads stop being pure byte functions from layer ~2 on), and **key
  addressing** adds the rest at fixed payloads (0.134 → 0.103). The
  dissociation the registration asked for is therefore answered: the
  addressing change contributes a real but modest fraction of the
  identity-share fall; the NLL/occupancy/entropy changes (below) are where
  the addressing channel's effect concentrates.
* **K-3 (occupancy falls from the banked 73.7/128) — MET, far below the
  band.** 24.0 mean live slots (band: ≤ 65; the operator's original ~48-slot
  figure was the K8 smoke's init snapshot — this arm's trained store is far
  sparser than even that). The store operates as a small hot set of
  context-addressed rows; capacity is not the binding constraint — exactly
  the CMA-ES diagnosis's reading, now under the fixed addressing.
* **K-4 (read entropy rises from 1.085) — MET.** 1.654 nats (> 1.2 band).
  The sharpening alternative (a context-addressed read locking onto one row
  would push entropy DOWN) did not happen: the reads spread over the hot set.
* **K-5 (MERGE) — reported, not re-predicted.** 0.0031/token (the fused
  per-token arm's 0.0002; the dial's staleness artifact is gone but MERGE is
  still a minor operator at per-token granularity — consistent with K6's
  falsification of the recovery prediction).
* **The KEY-COLLAPSE guard — MET.** span_key_std 1.22–1.79 in every layer
  (floor 0.05; the Wheeler collapse signature appears nowhere), and training
  EARNED drift exactly as the mimetic receipt predicted: the emitted keys'
  mean pairwise cosine fell from 0.249 at birth (the 16-d sketch of the
  content geometry, above `tau_match`'s collision regime) to **0.099**
  — training pushed distinct contexts' addresses apart, below the match
  threshold.
* **Refutation clause — NOT triggered.** Neither condition fired; the fix
  transfers to real text, and the residual's ownership question (what owns
  the remaining 1.8418 − 1.5163 = 0.33-nat probe-scale gap to gdn2) is
  reopened for the next stage, not excused.

### 3.2 The layer-0 receipt — the SEP-KV per-layer scope, at 150 M

The identity share is NOT uniform: layer 0 holds **0.614** (layers 5–23 hold
0.072–0.085). At the first store layer the payload is a nearly pure function
of the byte — the collision lemma's premise — so even context-addressed keys
cannot avoid byte-identity CORRECTs there; the layers that CAN be
context-addressed are. This is the SEP-KV claim's honest per-layer scope
showing up at 150 M exactly as registered (depth substitution is available
from layer 2; the necessity claim was never "context-blind everywhere"), and
it names where the remaining 0.33-nat residual plausibly lives: the
addressing cannot fix what the VALUE channel cannot distinguish.

### 3.3 The honest walls and caveats

* The wall is 1,866 s (10.7 k tok/s) against the fused arm's 1,720 s
  (11.6 k) — the key channel's 8 % throughput price, with the forward
  actually FASTER per scan (19.7 vs 26.5 ms; the 16-wide key folds are
  cheaper than the 128-wide content folds they replace) and the backward
  12 % slower (17.2 vs 15.3 ms). Peak allocation 23.09 GB (the fused arm's
  19.97).
* The train loss is noisier than the fused arm's (per-step range ~1.9–3.7
  late in training against the fused arm's smooth ~2.45 plateau) — spikier
  reads concentrate the loss variance per window; the eval's 7-window mean
  is what the claim rests on, and the eval boundary is 14,336 targets.
* One receipt blemish, disclosed: the banked probe's per-layer
  `op_rate_per_token` fields were divided by the GLOBAL decision count
  instead of the per-layer one (the aggregate op rates, the per-layer
  occupancy/entropy/identity shares and every prediction read correctly
  normalized fields; nothing downstream consumed the misnormalized ones).
  Fixed in `k10_probe_separated_keys.py` after the run; the banked receipt
  stands as written by the code that ran.
* Single seed (3401), probe exposure, one transferred LR — the sweep's own
  honest caveat inherited verbatim. The board-scale (2700 s) run is gated on
  the operator's sign-off of BOTH pending registrations (this one and the
  K6 granularity draft it builds on).

## 4. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py -q
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k10_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k10_gates.json
setsid bash -c 'experiments/expressivity_tasks/x45_matrix_emender/run_k10_probe_leased.sh \
    > experiments/expressivity_tasks/x45_matrix_emender/k10_runs/driver.console.txt 2>&1' &
```

## 5. Recommended next step

The operator reads `PREREGISTRATION_SEPARATED_KEYS.md` (and the K6 draft) and
signs off or amends. With sign-off, the natural follow-ups, in the lane's own
order: (1) the 2700 s matched-time board arm of the separated-key machine
against the banked chunk-128 1.7576 and gdn2's 1.1976, with the tuned-LR
confirmation as its twin; (2) the K8 growth composition wired fully (the key
lane is a schedule-driven lane over the same recorded schedule — growth
reveals zero-keyed rows that join the address space at their first write);
(3) the K9 distributed-walk composition at high rank; and (4) the named
registration follow-up — ordered/Wheeler-style keys (addresses of addresses)
now that the flat small-key case is receipted on real text.
