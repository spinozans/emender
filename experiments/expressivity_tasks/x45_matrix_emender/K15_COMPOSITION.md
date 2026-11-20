# K15 — THE COMPOSITION: the additive read + the per-channel gate stack

Stage 15 of the X45 kernel lane. **The question this stage exists for:**
the session's two proven-positive quality levers — K13's additive read
(post-cooldown 1.3928/1.3845, −0.041/−0.049 vs the banked 1.4338, ~60% of
the GDN2 gap closed, |w|-mass effective rows 3.6×, no collapse) and K11B
p3's per-channel decay/write/read gate stack (+0.0103/+0.0104 seed-stable
on the SOFTMAX read) — touch different sub-mechanisms of the same store.
**Do they compose?** This lane composes them into one machine-variant and
answers at the matched-exposure probe protocol, both seeds, under the
corrected cooldown-primary scoring.

The registration is the DRAFT `PREREGISTRATION_COMPOSITION.md` (committed
BEFORE any probe measurement). The engine claims are receipted in
`receipts/k15_gates.json`; the arms' claims are in
`RESULT_COMPOSITION.md`. **No board-scale (2700 s) claim is run before the
operator signs the registration off.**

**Headline, as narrowly as the evidence supports.**

* **The composition is BORN AS THE ADDITIVE MACHINE, bit for bit** — the
  session's cleanest composition receipt: the K11B gates born at the
  registered scalar equivalents (bit-equal through the committed
  `channel_decay` wrapper, the falsifier-fixed carrier-dtype construction)
  make the composed forward **bit-identical to K13's own committed kernel
  on every channel**: **0 differing elements in 43,061,248 lanes** at the
  gate fixture (both gated combinations) and **0 in 516,734,976 lanes** on
  the probe's own first train batch, layer for layer — and the **step-1
  train loss is BIT-IDENTICAL to the K13 additive arm's own same-seed row
  at BOTH seeds** (5.782775391046648 / 5.720172563202926), the nearest-kin
  parity, asserted loudly. No banked birth parity exists (a different
  function) and none is asserted — a machine-variant arm like K13 was.
* **Gate A: state and channels bit-exact at every scale** — 0 flips in
  17,336,320 lanes vs the eager composed oracle per gated combination
  (34,672,640 total, bf16 autocast corners included), the read at exactly
  **1.0× its measured fp32-vs-fp64 floor**; the kernel holds **255
  registers, 2 spills** (K11B's own walk needed 6 — the softmax's removal
  freed the registers the vector loads cost).
* **Gate B: the fp64 gradcheck THROUGH THE COMPOSED PATH OK** (the raw-score
  weights attached under the per-channel DECAY vector and write gate; none
  192 / append 12 / correct 180), and all **seven** gradients at ≤ 1.5× the
  measured fp32-vs-fp64 floor — dpayload and dstate exactly 0.0 deviation.
  The strength path is gone with the softmax (K13's receipt carried over:
  the prep's EXACT-ZERO dstrength channel into K11B's committed value walk).
  The segment recompute is bit-equal on both channels (K11B's per-channel
  value replay + K10's key replay, imported).
* **The honest walls**: forward **21.25 ms** per (layer, window) scan —
  cheaper than K11B's own 22.44 (the softmax dropped) and pricier than
  K13's 15.40 (the per-channel vector loads), both measured in the SAME
  session; backward 17.13 vs K11B's 18.40 and K13's 15.92. The composition
  carries BOTH engines' costs, exactly as the form predicts. The parameter
  delta is **p3's own 8,657,040** (the read channel is K13's own committed
  g_proj construction, instantiated once, never double-added; the read-form
  change itself parameter-free).

---

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_COMPOSITION.md` | the DRAFT registration (operator sign-off pending): the composition's semantics, the parameter accounting, the protocol, the birth receipts, the pre-stated C-1/C-2/C-3 clauses with refutation language, NO directional headline prediction |
| `k15_composition.py` | the composed oracle, the ONE new forward kernel, the composed backward body wiring the COMMITTED kernels, the op pair, `ComposedReadMixer` (subclassing the committed p3 mixer) + the conversion + the parameter-delta accounting |
| `test_k15_composition.py` | 25 tests: gates A/A2/B, the fp64 gradcheck through the composed path (mandatory), the born-as-the-additive-machine receipts, the heterogeneous-MERGE corner, the carrier-corner remedy, the refusals, determinism |
| `k15_gate_receipt.py` | the gate receipts into `receipts/k15_gates.json` (gates A, A2, B, D) |
| `k15_probe_composition.py` | the arm runner + the assemble (the birth receipts, the corrected scoring, the composed mechanism instrument, the C-1/C-2/C-3 adjudication) |
| `run_k15_composition_leased.sh` | the detached leased driver (cards 0/1) |
| `K15_COMPOSITION.md` / `RESULT_COMPOSITION.md` | this document / the arms' result document |

### 1.1 The composed machine, stated exactly once

```
lam_vec = clamp(lam_inc*(1 + m_f*sigmoid(f_proj(x))), floor, 1-1e-6)  # per channel, born == lam_inc
g_vec   = 1 - m_w*sigmoid(b_proj(x))                                   # per channel, born == 1
score_j = read_scale * cos(q_key(t), K_j)   # the banked score channel, VERBATIM (R13-1)
w_j     = score_j  if slot j is live else 0 # raw; NO softmax, NO ||S_j|| (R13-2)
y_raw   = sum_j w_j * S[j]                   # the K6 floor-channel rank sum
y       = r_vec * y_raw                       # the g_proj gate, outside the scan
```

with the per-channel DECAY applied to the values only and K11B's gated
lane-0 arithmetic (APPEND admits `g_vec ⊙ p`, CORRECT blends
`g_vec ⊙ p + (1-g_vec) ⊙ (lam_vec ⊙ v_pre)`). Everything else — decisions,
the MERGE blend (its own detached strengths), the key lane, POP/MERGE
zeroing — is the K10 winner VERBATIM through both engines' commitments.
The composed oracle takes K11B's per-lane-guarded MERGE close (the
K12-flagged multi-lane corner of the K10/K11/K13 oracles is NOT copied),
receipted on the K11 gate-E heterogeneous fixture: oracle-vs-kernel
bit-identical with MERGE and CORRECT both live.

### 1.2 The kernels — import, never modify

`_k15_seq_fwd_kernel` is THE ONE NEW KERNEL: K11B's persistent
whole-window walk with the read block replaced by K13's additive block
(`w = where(live, score, 0)`; the softmax's exp/max/denominator GONE).
The backward composes the COMMITTED kernels VERBATIM: K11B's per-channel
trajectory replay, K10's key-traj replay (the K13 backward's own
construction), **K13's additive prep** (`_k13_bwd_prep_kernel`:
`dscores = dw` on live rows, the strength channel EXACT ZEROS), **K11B's
per-channel value walk** (the read adjoint's value terms consume the
additive weights; the strength path dies by the zero channel — IEEE add of
0.0 is exact, receipted in the suite on a fixture with wildly different
row norms), and K10's key walk. No committed engine is touched.

## 2. The gates (receipts/k15_gates.json)

| gate | bar | measured | verdict |
| --- | --- | --- | --- |
| A — kernel vs composed oracle, every channel, registered shape (bf16), BOTH gated combinations | bit-equal; read ≤ 8× floor | **0 flips / 17,336,320 lanes per combination** (34,672,640 total); read 1.19e-6 dev at exactly 1.0× the 1.19e-6 floor; 255 regs / 2 spills | GREEN |
| A — the signed-weight channel under the gates | negatives present; gain-bounded on live; zero on dead; NOT a simplex | 1,341,361 negative entries; max abs 1.868 ≤ read_scale 2.0; exact 0 on dead; sums ≠ 1 | GREEN |
| A2 — BORN AS THE ADDITIVE MACHINE vs K13's own kernel | 0 differing elements | **0 / 43,061,248** (gate fixture, both gated combos); **0 / 516,734,976** (the probe's first train batch, layer for layer — the probe's own receipt); step-1 bit-identical to the K13 arm's row at both seeds | GREEN |
| A2 — decisions-bit-identical under random per-channel gates/decay | 0 flips | **0 / 8,978,432** (no gate value feeds op selection) | GREEN |
| A2 — the birth-magnitude diagnostic (R13-5 carried) | stated, not guessed | composed/banked per-head read RMS **2.47×/2.52×** at the gate fixture (the additive machine's own rows — the gates cannot change the read at birth) | RECEIPTED |
| A2 — the \|w\|-mass dilution instrument (R13-4) | e^H of \|w\|-mass, live rows | **62.76 vs 30.25** effective rows on the same fixture (K13's own rows) | RECEIPTED |
| B — fp64 gradcheck THROUGH THE COMPOSED PATH | every branch | OK (none 192 / append 12 / correct 180; POP/MERGE pinned by the suite's crafted fixtures + the committed kernels' own receipts) | GREEN |
| B — the seven gradients vs the oracle's floors | ≤ 8× floor | dpayload 0.00×, dwrite_keys 1.33×, dquery_keys 1.50×, ddecay 0.50×, dgates 0.33×, dstate 0.00×, dkeys 0.00× | GREEN |
| B — segment recompute, both channels | bit-equal | bit-equal (K11B's per-channel value replay + K10's key replay, imported) | GREEN |
| D — block-call walls, registered shape | stated vs both engines' same-session launches | forward **21.25 ms** (K13: 15.40, K11B: 22.44), backward **17.13 ms** (K13: 15.92, K11B: 18.40) | BOTH ENGINES' COSTS, as the form predicts |

## 3. The honest scope

* **Probe scale only** (20,022,866-target matched exposure, 180+24 steps),
  both seeds (3401/3402) — the composition's own baseline (the K13
  additive rows) exists at both seeds, so both are primary. No board-scale
  claim before the operator signs the registration off.
* The composition ADDS p3's own parameters (the k11b R4 precedent: the 0.1%
  capacity tolerance is dropped for a real GDN2-granular gate stack;
  +8,657,040 = +5.77%, ~3× leaner than GDN2's own 18.3% gate share — the
  asymmetry is the finding).
* The committed K10/K13/K11B suites stay green **with the module loaded**
  (59/59, `k15_runs/k10_k13_k11b_suites_console.txt`); the lane's own
  suite is 25/25.

## 4. Commands

```bash
# the suite (the fp64 gradcheck through the composed path is mandatory):
/home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k15_composition.py -x
# the gate receipts:
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)" && \
CUDA_VISIBLE_DEVICES=$GPU_LEASE_HELD OMP_NUM_THREADS=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k15_gate_receipt
# the arms (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k15_composition_leased.sh composition 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k15_runs/\
composition_seed3401.console.txt 2>&1 < /dev/null &
# the assemble:
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.\
x45_matrix_emender.k15_probe_composition assemble
```
