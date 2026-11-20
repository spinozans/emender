# K13 — THE ADDITIVE READ: GDN2's read form transplanted into the K10 winner

Stage 13 of the X45 kernel lane. **The question this stage exists for:**
the side-by-side receipted the winner's residual as C5, the write/read form,
with the read-dilution finding at its center — reads over **~5.6 effective
rows of ~27 live** (competitive softmax dilution). GDN2's read is ADDITIVE
(`out_t = S_t^T q_t` + an output gate): every matching direction contributes,
magnitude carries information, nothing competes for a probability simplex.
The read form is transplantable WITHOUT changing the store or the ops — it
is a value-channel change. This lane is that transplant, against the banked
K10 winner, at the matched protocol, both seeds.

The registration is the DRAFT `PREREGISTRATION_ADDITIVE_READ.md` (committed
BEFORE any probe measurement, the K11B lesson; the supervisor's R13-1..R13-5
rulings cited inline). The engine claims are receipted in
`receipts/k13_gates.json`; the arms' claims are in
`RESULT_ADDITIVE_READ.md`. **No board-scale (2700 s) claim is run before the
operator signs the registration off.**

**Headline, as narrowly as the evidence supports.**

* **The read form is a surgical swap.** `w_j = read_scale·cos(q_key, K_j)` —
  the banked score channel VERBATIM (R13-1; `read_scale` 2.0
  INHERITED-FROM-BANKED, disclosed as a gain now, not re-tuned) — raw and
  live-gated: **no softmax, no `||S_j||` strength factor** (R13-2: magnitude
  rides `v_j = S[j]` itself, GDN2-faithful; a POP'd row contributes exactly
  0 through `S[j] = 0`; the live mask still gates because dead rows' keys are
  zeroed). The output is gated by the committed k11b `g_proj` read gate
  (R13-3: `r_vec = 1 + m_o·sigmoid(g_proj(x))`, zero-born at EXACT 1,
  applied outside the scan). Decisions are INVOLIABLE: the decision chain is
  K10's own on the untempered key similarity.
* **Gate A: state and channels bit-exact at every scale** — 0 flips in
  17,336,320 lanes vs the eager additive oracle at the registered shape
  (bf16 autocast corners included, the K11B lesson), the read at **1.14× its
  measured fp32-vs-fp64 floor**; the kernel holds **255 registers, 2 spills**
  (K10's own numbers).
* **The machine-variant's birth receipts** (no birth parity is possible — a
  different function; the K10 machine-variant tradition): the recorded
  schedule is **bit-identical to the BANKED read's own committed kernel at
  fixed inputs** (0 flips in 315,408,384 lanes at probe scale, 13,142,016 at
  the gate fixture), while the reads and weights differ as designed; the
  **birth-magnitude diagnostic (R13-5)** carries the honest ~O(live-rows)×
  consequence (§2); the gate is born at exact 1.
* **Gate B: the fp64 gradcheck through the ADDITIVE path OK** (the
  raw-score weights are attached — no softmax jacobian anywhere; the
  checked branches include the read's direct term into the state and the
  score path into the query keys and the key table), and all six gradients
  at **≤ 1.0× the measured floor** vs the oracle's own autograd — the
  simpler adjoint is *measurably* cleaner than the softmax's (several
  deviations exactly 0.0 on the gate fixture). The segment recompute is
  bit-equal on both channels through the imported replays.
* **The honest walls**: forward **16.43 ms** per (layer, window) scan vs
  K10's banked 19.7 — the softmax's exp/max/denominator are GONE, and the
  form is *cheaper*, as it should be (backward 15.99 vs 17.2). The read
  still costs its floor-channel discipline (the native rank sum); the
  parameter delta is the k11b R4 precedent's own: **+2,562,096 (+1.70%)**
  for the gate, the read-form change itself parameter-FREE.

---

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_ADDITIVE_READ.md` | the DRAFT registration (operator sign-off pending): the read form (rulings R13-1..R13-5), the parameter accounting, the protocol, the birth receipts, the pre-stated M-R1..M-R3 with refutation clauses, NO directional headline prediction |
| `k13_additive_read.py` | the eager per-token oracle (`additive_oracle_scan`), the persistent forward kernel with the read block replaced (`_k13_seq_fwd_kernel`), the lane's backward prep (`_k13_bwd_prep_kernel`), the backward body wiring K10's imported walks, the `x45::k13_scan_fwd/bwd` op pair, the refusals, `AdditiveReadMixer` + the conversion + the parameter-delta accounting |
| `test_k13_additive_read.py` | 19 tests: gates A/A2/B, the fp64 gradcheck through the additive path, the decisions-bit-identical receipt, the gate-birth exact identity, the strength-path-gone receipt, the conversion, the refusals, determinism |
| `k13_gate_receipt.py` | the gate receipts into `receipts/k13_gates.json` (gates A, A2, B, D) |
| `k13_probe_additive_read.py` | the arm runner + the assemble (the birth receipts, the corrected scoring, the mechanism instrument with the \|w\|-mass channels, the decorative-store sweep, the M-R1..M-R3 adjudication) |
| `run_k13_additive_read_leased.sh` | the detached leased driver (cards 0/1) |
| `K13_ADDITIVE_READ.md` / `RESULT_ADDITIVE_READ.md` | this document / the arms' result document |

### 1.1 The read, stated exactly once

```
score_j = read_scale · cos(q_key(t), K_j)   # the banked score channel, VERBATIM
w_j     = score_j  if slot j is live else 0 # raw; NO softmax, NO ||S_j||
y_raw   = Σ_j w_j · S[j]                     # the K6 floor-channel rank sum
y       = r_vec ⊙ y_raw                      # the g_proj gate, outside the scan
```

Everything else — decisions, payload lanes, the MERGE blend (with its own
detached strength weights, K6 verbatim), DECAY (values only), the key lane,
POP/MERGE zeroing — is the K10 winner VERBATIM. The **K7 full-population
read law** question is dissolved by the form itself: the additive read has
NO denominator at all, so no per-block renormalisation exists to obstruct
the law — the accumulation is a plain live-gated sum over the
register-resident whole store (the K10 walk holds the full population in
registers), keeping the K6 floor-channel discipline (the read feeds no
decision, so no flip can come of it).

### 1.2 The kernels — import, never modify

`_k13_seq_fwd_kernel` is K10's persistent whole-window walk with the read
block replaced (`w = where(live, score, 0)`; the exp/max/denominator GONE).
`_k13_bwd_prep_kernel` is K10's token-parallel prep with the read adjoint's
coefficients replaced: `dscores_j = dw_j` on live rows (the identity —
`w_j = score_j`), the strength channel written as EXACT ZEROS, and the same
clamp-select key-side identities (`z`/`duu`/`dc`/`dnk`) on the new
`dscores`. K10's **key-traj replay, value walk and key walk kernels are
consumed VERBATIM** — the value walk's strength term dies by the zero
`dstrength` channel (IEEE add of 0.0 is exact, receipted in the suite on a
fixture with wildly different row norms), its direct term consumes the
score weights, and the key walk's c/nk terms consume the prep's channels
unchanged. No committed engine is touched.

### 1.3 The g_proj read gate

The committed k11b read channel's own form and birth: `g_proj =
Sequential(Linear(width, d_v, bias=False), Linear(d_v, heads·d_v, bias=True))`
zero-born, `m_o` a per-head scalar born at EXACT zero, applied
`reads ⊙ r_vec` before the output projection (K11/K11B's read-out
convention, outside the scan). At birth `r_vec == 1` **bit for bit** under
autocast (receipted per layer, per batch), so the arm is the PURE additive
read, and the gate grows by training (the K11 two-step wake-up pattern: the
scalar's gradient is live at step 0 through `sigmoid(0) = 0.5`, the
projection wakes one step later — receipted so a step-0 zero is not read as
a dead channel).

## 2. The gates (receipts/k13_gates.json)

| gate | bar | measured | verdict |
| --- | --- | --- | --- |
| A — kernel vs oracle, every channel, registered shape (bf16) | bit-equal; read ≤ 8× floor | **0 flips / 17,336,320 lanes**; read 2e-6 dev at 1.14× the 2e-6 floor; 255 regs / 2 spills | GREEN |
| A — the signed-weight channel (the form's own receipt) | negatives present; gain-bounded on live; zero on dead; NOT a simplex | 1,341,361 negative entries; max abs 1.868 ≤ read_scale 2.0; exact 0 on dead; sums ≠ 1 | GREEN |
| A2 — decisions-bit-identical vs K10's own kernel, fixed inputs | 0 flips | **0 / 13,142,016** (gate fixture), **0 / 315,408,384** (probe scale, the birth receipt); reads/weights differ as designed | GREEN |
| A2 — the birth-magnitude diagnostic (R13-5) | stated, not guessed | additive/banked per-head read RMS **2.47×/2.52×** at the gate fixture (near-empty store: ~2× at birth, growing with occupancy) | RECEIPTED |
| A2 — the \|w\|-mass dilution instrument (R13-4) | e^H of \|w\|-mass, live rows | **62.8 vs 30.2** effective rows on the same gate fixture | RECEIPTED |
| B — fp64 gradcheck THROUGH the additive path | every branch | OK (append 12 / correct 180 / none 192; POP/MERGE pinned by the suite's reachability + the committed kernels' own receipts — the prep is op-agnostic) | GREEN |
| B — the six gradients vs the oracle's floors | ≤ 8× floor | dpayload 0.00×, dwrite_keys 1.00×, dquery_keys 0.83×, ddecay 0.75×, dstate 0.00×, dkeys 0.00× | GREEN |
| B — segment recompute, both channels | bit-equal | bit-equal (K4's value replay + K10's key replay, imported) | GREEN |
| D — block-call walls, registered shape | stated vs K10's banked | forward **16.43 ms** (K10: 19.7), backward **15.99 ms** (K10: 17.2) | CHEAPER, as the form predicts |

## 3. The honest scope

* **Probe scale only** (20,022,866-target matched exposure, 180+24 steps),
  both seeds (3401/3402) — no birth parity is possible, so both seeds are
  primary evidence. No board-scale claim before the operator signs the
  registration off.
* The arms ADD parameters (the k11b R4 precedent: +2,562,096, +1.70% — the
  0.1% tolerance is dropped for a real GDN2-granular gate transplant; the
  read-form change itself is parameter-free).
* The read keeps its floor-channel discipline: the read OUTPUT is a native
  sum, so it is gated at floors (never bits) against the oracle — every
  other channel is bit-equal by construction.
* `read_scale = 2.0` is INHERITED-FROM-BANKED (R13-1): under an additive
  read it is a gain, absorbed by the downstream learned projection; it was
  NOT re-tuned, and any gain-sensitivity is what the birth-magnitude
  diagnostic makes inspectable.

## 4. Commands

```bash
# the suite (the fp64 gradcheck through the additive read path is mandatory):
/home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k13_additive_read.py -x
# the gate receipts:
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)" && \
CUDA_VISIBLE_DEVICES=$GPU_LEASE_HELD OMP_NUM_THREADS=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k13_gate_receipt
# the arms (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k13_additive_read_leased.sh additive 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k13_runs/\
additive_seed3401.console.txt 2>&1 < /dev/null &
# the assemble:
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.\
x45_matrix_emender.k13_probe_additive_read assemble
```
