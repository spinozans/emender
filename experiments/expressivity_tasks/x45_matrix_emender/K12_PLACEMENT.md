# K12 — the learned placement arm: the APPEND's free-slot choice made a learned policy on the dead-key residue

Stage 12 of the X45 kernel lane, and the answer to the op-set audit's named
gap: our APPEND's WRITE half is instantiated (the rank-1 lane), but its
ALLOCATION half — **WHICH free slot the new row lands in** — was the fixed
first-free-ordinal rule, a choice the model could not make. The operator
named the lever: a learned placement policy, natural form **"write where the
evicted content lived"** — score the free slots by the similarity of the new
write key to the dead row's last-key residue — generalising to learned
ADDRESS LOCALITY (related content clustering in slot space). The
registration is the DRAFT `PREREGISTRATION_PLACEMENT.md` (committed before
any probe measurement; the operator signs it off before any 2700 s claim —
it inherits the K10 draft's own pending sign-off). This document is the
semantics and the GATE receipts; `RESULT_PLACEMENT.md` is the probe.

**Headline, as narrowly as the evidence supports.**

* **The birth bar holds bit-exactly (P5).**  The placement weight `w` is one
  per-head scalar per mixer born at EXACT `+0.0`; the score spec adds one
  `+0.0` (IEEE: `-0.0 + +0.0 = +0.0`), so at birth EVERY free slot scores
  exactly `+0.0` and the argmax's lowest-index tie-break reproduces K10's
  first-free rule **bit-exactly**: 0 flips vs K10's own launch on every K10
  channel over 17,369,088 lanes (the gate receipt's birth-parity lane), and
  bit-equality FOREVER with `w` frozen at `+0.0` (the never-fires falsifier:
  a 3-step trained receipt in the tests — the losses equal the K10
  machine's step for step).
* **THE ZERO-GRADIENT FINDING (the supervisor's own ruling, receipted
  prominently).**  Under P2+P3 the placement effect is purely DISCRETE
  (which free slot receives the APPEND), and the recorded-schedule doctrine
  makes the slot a backward constant — so the STRUCTURAL gradient to `w` is
  EXACTLY ZERO: the eager oracle's autograd returns `dL/dw == 0` on every
  fixture (asserted in the tests, `oracle_dw_is_none: true` in the gate
  receipt).  **"The model can learn where to write" therefore REQUIRES an
  admitted estimator, and that requirement is itself a structural fact
  about this architecture** — the same wall the zero-learned-management
  doctrine built Paper 1 on, now surfaced from the placement side.
* **The admitted estimator (supervisor-approved option (a); option (c) —
  attaching the score inputs — REJECTED: it amends the detached-decision
  doctrine and muddies mechanism attribution).**  The kernel path's backward
  is K10's own `_k10_backward_body` imported VERBATIM (the six value/key
  gradients UNCHANGED, within the measured floors) plus ONE small reduction:
  at each APPEND token, `g_t = ⟨dpayload_t, p_t⟩` (the EXISTING value walk's
  own write-content sensitivity — no new walk), `p = softmax(w·sim over
  free slots)`, and
  `dL/dw = Σ_t g_t · p_{j*} · (sim_{j*} − Σ_j p_j·sim_j)`.
  The score inputs are DETACHED (the write key is decision-side, the residue
  a detached recording): **only the +48 per-head `w` learns**.  The
  estimator equals its DEFINED torch reference to 0.0 max diff (gate
  receipt, replayed from the recorded channels), is nonzero at the
  zero-born `w` (P5's step-0 receipt), and its bias is disclosed verbatim in
  the registration: it captures the write-content sensitivity of the chosen
  slot, NOT the full trajectory sensitivity of the slot choice.
* **The placement channel costs ~5 % forward** (20 vs 19 ms per
  (layer, window) scan at the registered shape — the residue tile beside the
  key tile; the placement math runs only under a scalar `if appendable`
  guard, the ~2–5 % of tokens that append).

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_PLACEMENT.md` | the DRAFT registration (operator sign-off pending): the semantics, the structural facts F1–F3, the estimator and its bias, the arms, the pre-stated predictions P-1..P-5 with refutation clauses, the three-way read |
| `k12_placement.py` | the eager per-token oracle (`placement_oracle_scan`), the persistent forward kernel with the residue tile and the placement scores (`_k12_seq_fwd_kernel`), the estimator's reduction (`_k12_placement_grad_kernel`), the `x45::k12_scan_fwd/bwd` op pair, the wrapper (`placement_scan`), the mixer and the conversions (`PlacementSeparatedMixer`, `convert_model_to_placement`) |
| `test_k12_placement.py` | 18 tests: gate A parity (four weights, ragged, bf16, ties, the crafted firing regime, the heterogeneous-lane pin), birth parity vs K10's launch, the frozen-`w` falsifier through training, the zero-gradient finding, the estimator vs its reference, gate B floors, the six-gradients-unchanged receipt, refusals, determinism, the conversion accounting |
| `k12_gate_receipt.py` + `receipts/k12_gates.json` | the registered-shape receipts: gates A/B/D, the birth-parity lane, the placement-regime fire lane, the finding, the estimator receipt |
| `k12_probe_placement.py` + `run_k12_placement_leased.sh` | the matched-exposure probe under the corrected scoring (cards 2/3) with the placement/address-locality/read-mask instruments, and its detached leased driver |

No `k10_*`/`k11_*` source, PREREGISTRATION, `lattice.py` or `matched_arms.py`
was edited; K10's spec helpers, its backward body, its selection chain, its
mixer/conversion and the K10 probe's own instrument functions are IMPORTED.
The K11 lane (cards 0/1) is untouched throughout.

### 1.1 The semantics (the registration's §1, pre-stated)

The K10 separated-keys machine VERBATIM (per-token, key_dim 16, mimetic key
birth, free drift, keys undecayed) with one new RECORDED channel and one new
choice:

1. **The residue channel (P1).**  `Res` (per head, `rank × key_dim`): at a
   row's DEATH token — POP's target row, MERGE's source row —
   `Res[j] := K[j]`, the dying row's last key, recorded PRE-CLEAR.  Never-
   used slots keep a zero residue.  Placement scores take state dependence
   EXACTLY through this recorded channel, the recorded liveness, and the
   current token's write key — nothing else.
2. **The placement score and the birth tie-break (P5).**  At an APPEND
   decision, the score of a FREE slot `j` is
   `s_j = w·cos(wk_t, Res_j) + (+0.0)` in the K10 defined-order arithmetic
   (the same `_split_fold_axis2` folds, EPS-clamped norms — FINITE at a zero
   residue), and the APPEND lands in `argmax over free slots of s_j`, lowest
   index wins ties — the lane's established first-max semantics, the SAME
   construction in the oracle and the kernel.  The `+0.0` add is part of the
   spec: `0·(negative cosine)` yields `-0.0`, and IEEE `-0.0 + +0.0 = +0.0`,
   so at `w = +0.0` every free slot scores EXACTLY `+0.0` and the argmax
   reproduces first-free bit-exactly.
3. **P2/P3 untouched.**  Placement selects WHICH free slot an APPEND lands
   in; it never touches the op code, thresholds, payload, key values or the
   liveness arithmetic, and never a live row (the APPEND target is free by
   definition).  `op/erase/live_after` are functions of the op chain alone.

### 1.2 The structural facts that size the lever (the registration's §0)

* **F1 — the decision window is rare.**  Placement can differ from
  first-free ONLY when ≥ 2 slots are simultaneously free; POP requires a
  FULL store, so the only window is a POP-then-MERGE (or MERGE-then-MERGE)
  sequence before an APPEND consumes a slot.  The banked op mix (CORRECT
  0.97939, APPEND 0.01748, MERGE 0.00313, POP **0.00000** per token) makes
  that window rare — the honest prior on the NLL headroom is SMALL, and the
  pre-registered band is sized accordingly.
* **F2 — the zero-gradient finding** (above).
* **F3 — the falsifier is bit-exact by construction** (above).

## 2. Gates A, B and D (the receipts: `receipts/k12_gates.json`)

**Gate A — parity at the registered shape** (batch 8, 2048 tokens, 2 heads,
rank 128, value_dim 128, key_dim 16, bf16 inputs): every channel — the
schedule fields (the PLACED slot at APPEND included), the payload lane, the
read weights, the pre-decision liveness, the exit state/liveness, the key
table AND the residue table — bit-equal against the eager oracle at `w = 0`
AND `w = 0.5`: **0 flips** in 17,369,088 lanes at each weight, the read at
the fp32-vs-fp64 floor.  **The birth-parity lane:** the k12 launch at
`w = +0.0` vs K10's OWN launch — **0 flips** on every K10 channel.  **The
placement-regime lane** (the registered shape with the store seeded FULL —
128 live rows, 8 mergeable key pairs at cos 0.9 beside unique fillers, a
token stream mixing pair keys with fresh unmatcheds): parity holds (0
flips) while **51.7 % of the 145 APPENDs fire off the first-free rule**,
the chosen slots' residue similarity 0.152 vs the free-slot mean −0.014 —
the mechanism receipt that placement fires and chooses residue-similar
slots.  The suite adds: ragged token/segment lengths, the bf16 regime, the
equal-residue tie (lowest free index in both implementations), the
heterogeneous-lane MERGE corner (below), determinism, and the refusals by
name (append-only, growth, a rank prefix, a malformed placement weight) —
each taking the eager oracle verbatim.

**Gate B — the gradients.**  The six K10 gradients through the k12 backward
(K10's own body, imported verbatim) sit within `NOISE_FACTOR` (8×) of the
fp32-vs-fp64 floors measured on the gate fixture, and are BIT-EQUAL to
K10's own body's outputs on the same recorded channels (the receipt that
the estimator changes nothing else).  The placement weight's gradient is
NOT the true adjoint — it is the admitted estimator, gated by its DEFINED
torch reference (max abs diff **0.0** on the gate fixture), with the
zero-gradient finding asserted on the same fixture (`oracle_dw_is_none`).
The fp64 `gradcheck` of the placement oracle validates the six K10 channels
through every adjoint branch (the placement weight excluded by the finding).

**Gate D — the walls.**  Forward 18–20 ms vs K10's 18–19 ms per scan at the
registered shape (~5 %, the residue tile); the backward is K10's body plus
the estimator's whole-window replay (one light kernel, one program per
(batch·head) row).  The K10 suite stays GREEN with the new module loaded
(19 passed); the K12 suite is 18 tests.

## 3. Two findings the gates surfaced, stated honestly

* **The K10 eager oracle's MERGE branch carries a LATENT MULTI-LANE BUG**
  (found by this lane's placement-regime gate fixture; fixed in THIS
  module's oracle copy only — no `k10_*`/`k11_*` file edited).  The
  `if sel["merge"].any():` block index-puts zeros at EVERY (batch, head)
  lane's second row whenever ANY lane merges, instead of only the merging
  lanes' source rows.  The KERNEL guards per lane and is correct.  The bug
  is unreachable by K10's own gate fixtures (its MERGE fixtures are
  single-lane; its multi-lane fixtures produce zero MERGEs — the banked
  receipts' merge counts verified 0 or single-lane) and inert in
  production (training takes the kernel path), so **no banked K10 result is
  invalidated**.  This module's copy masks the update per lane and the
  heterogeneous-lane test pins the corner (only the merging lane's source
  dies).  The K11 oracle was copied from the same pattern and likely
  carries the same latent bug with the same inertness argument — flagged to
  the operator, not touched here.
* **The op chain structurally throttles placement** (F1, above): the
  fire-rate instrument reads honestly as "the lever exists only in
  multi-free-slot windows", which the trained machine's op mix makes rare.
  The probe measures whether that rare window is load-bearing; the
  registration's three-way read makes every outcome a result, including
  THE HONEST NULL.

## 4. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k12_placement.py -q
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py -q
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k12_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k12_gates.json
setsid bash -c 'OMP_NUM_THREADS=4 \
    experiments/expressivity_tasks/x45_matrix_emender/run_k12_placement_leased.sh \
    baseline_seed3401 \
    > …/k12_runs/baseline_seed3401.console.txt 2>&1' &
```

## 5. Recommended next step

The operator reads `PREREGISTRATION_PLACEMENT.md` (and the K10/K11 drafts it
inherits) and signs off or amends.  The probe arms land in the ruled order —
baseline → placement-learned → the fixed-`w` bracket {0.25, 0.5, 1.0} → the
second seed on any ≥ 0.01-nat win — and `RESULT_PLACEMENT.md` reads the
three-way outcome: (i) the estimator works, (ii) the surrogate's bias is
binding (option (c) becomes the named follow-up), or (iii) the honest null:
free-slot placement is decorative on natural text.
