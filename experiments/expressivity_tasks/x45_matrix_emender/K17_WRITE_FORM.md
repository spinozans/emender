# K17 — THE WRITE FORM UNDER THE ADDITIVE READ: the dense gated write (the
b_proj-delta form) and the read-gain bracket, on the committed composed
machine

Stage 17 of the X45 kernel lane. **The question this stage exists for:** the
K15 adjudication named the WRITE FORM (event writes vs dense superposition)
as the LAST structural candidate owner of the composed machine's remaining
~0.03 vs GDN2, and the K16 config bracket discharged the config alibi and
left the residual UNCHANGED (+0.0289/+0.0311, parity-within-band under the
supervisor's ruling of 2026-11-25). The one untested transplant in the whole
program is the morning's G2 arm — the dense gated write — which HURT on the
SOFTMAX machine (+0.007): the per-head dense-write channel did not compose
with the COMPETITIVE read. The machine has since changed fundamentally (the
K13 additive read — the load-bearing lever — under the K11B p3 per-channel
gate stack), and the read/write interface is different now: **the read law
and the write law may compose the way the gates and the read law did not.**
This lane is G2's form reborn on the new interface, at the composed machine's
own per-channel granularity (the K11B registration's own named follow-up
§7.2: "a per-channel dense deposit — named, not run"), plus the read-gain
bracket (the additive read's one never-re-tuned constant), plus the
mandatory frozen-write falsifier control.

The registration is the DRAFT `PREREGISTRATION_K17_WRITE_FORM.md` (committed
BEFORE any probe measurement, 365ab3a5). The engine claims are receipted in
`receipts/k17_gates.json`; the arms' claims are in `RESULT_WRITE_FORM.md`.
**No board-scale (2700 s) claim is run before the operator signs the
registration off.**

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_K17_WRITE_FORM.md` | the DRAFT registration (operator sign-off pending): the dense channel's semantics (P3 verbatim, the b_proj form, the zero-born birth), the read-gain bracket with its **factual read_scale correction** (§3), the protocol, the pre-stated W-1/W-2/W-3/G-1 clauses with the conditional combined arm and the standing 0.01 win-bar rule, NO directional headline prediction |
| `k17_dense_write.py` | the dense oracle, the ONE new forward kernel (the K15 composed walk + the dense lane 2), the K17 value-walk backward (K11B's per-channel walk + the dense lane's `ddense = dS[best]`), the op pair, `DenseWriteComposedMixer` + the conversion + the parameter accounting |
| `test_k17_dense_write.py` | 27 tests: gates A/A2/B, the MANDATORY fp64 gradcheck through the dense-write path, the born-as-the-composed-machine receipts, the dense lane's P3 guards, the −0.0 corner, the refusals, determinism |
| `k17_gate_receipt.py` | the gate receipts into `receipts/k17_gates.json` (gates A, A2, B, D) |
| `k17_probe_write_form.py` | the arm runner (dense / dense_frozen / gain{1,2,4}x / conditional combined) + the birth receipts at probe scale + the C5 write-channel instrument + the assemble |
| `run_k17_write_form_leased.sh` | the detached leased driver (cards 0/1) |
| `tests/test_k17_write_form_probe.py` | the probe's source-named checks (the cited rows, the inherited-gain correction, the parameter accounting, the routing inventory) — 6/6 |
| `K17_WRITE_FORM.md` / `RESULT_WRITE_FORM.md` | this document / the arms' result document |

### 1.1 The write-form machine, stated exactly once

```
lam_vec = clamp(lam_inc*(1 + m_f*sigmoid(f_proj(x))), floor, 1-1e-6)  # per channel, born == lam_inc  (K11B p3, verbatim)
g_vec   = 1 - m_w*sigmoid(b_proj(x))                                   # per channel, born == 1        (K11B p3, verbatim)
score_j = read_scale * cos(q_key(t), K_j)   # the banked score channel, VERBATIM (K13's additive read)
w_j     = score_j  if slot j is live else 0  # raw; NO softmax, NO ||S_j||  (K13, verbatim)
y_raw   = sum_j w_j * S[j]                   # the floor-channel rank sum   (K13, verbatim)
y       = r_vec * y_raw                       # the g_proj gate, outside the scan (K11B p3, verbatim)

dense_t = m_d[h] * sigmoid(d_proj(x)) ⊙ p_t        # the b_proj-delta form: per channel, born == 0
S[best] += dense_t   if best is LIVE at decision time AND still live after lanes 0/1   # P3, lane 2
```

with `d_proj = Linear(width, heads*d_v, bias=False)` — **the b_proj form**
(GDN2's own single bias-free write-gate Linear shape, the K11B §7.2 named
granularity) — born at exact ZERO weight and `m_d` a per-head scalar born at
EXACT zero (the committed zero-born gate conventions), so at birth
`dense ≡ 0` and **the machine is BORN AS THE COMPOSED MACHINE bit for bit**
(the K15 birth discipline). The deposit fires on the LIVE GUARDS ALONE
(never on its value — a value-dependent no-op rule would make the recorded
op code a function of the deposit and cut the birth gradient), is recorded
as schedule **lane 2** (`op=APPEND`, `slot=best`, `erase=False`,
`live_after=True`, `payload=dense_t`), lands after lanes 0/1 and the
MERGE-source close, and rides the VALUE channels only: no deposit value
feeds op selection, thresholds, routing similarity, liveness or the key
lane (P2). The parameter delta is the dense channel's own
**+3,538,992** (147,458/mixer: 576·256 + 2) on top of the composed
machine's 158,638,848 → **162,177,840** (+2.36 % over the audited winner;
GDN2's own gate share is 18.3 %).

### 1.2 The kernels — import, never modify

`_k17_seq_fwd_kernel` is THE ONE NEW KERNEL: the K15 composed walk (K11B's
persistent whole-window walk with K13's additive read block) plus the dense
lane — the deposit load, the lane-2 record, and the guarded additive
application (K11's unconditional-select form: no runtime branch reassigning
the loop-carried tile). The backward composes the committed and
lane-stride-generic kernels: **K11B's per-channel value replay**
(`channel_persistent_trajectory`, imported verbatim — LANE-GENERIC, the
3-lane schedule replays bit-exactly), **K11's stride-generalized key
replay and key walk** (`_k11_key_traj_kernel`, `_k11_bwd_key_walk_kernel`,
imported verbatim — K10's own `_k10_key_traj_kernel` and
`_k10_bwd_key_walk_kernel` HARDCODE the 2-lane channel layout and cannot
consume a 3-lane schedule; K11 generalized them for its own dense lane, and
K17 imports K11's), **K13's additive prep** (`_k13_bwd_prep_kernel`, the
strength channel EXACT ZEROS), and the K17 value walk (K11B's per-channel
walk plus the dense lane's adjoint: `ddense = dS[best]`, `ds` passes
through — the deposit is additive). No committed engine is touched.

## 2. The gates (receipts/k17_gates.json)

| gate | bar | measured | verdict |
| --- | --- | --- | --- |
| A — kernel vs dense oracle, every channel, registered shape (bf16), BOTH gated combinations, dense lane ACTIVE | bit-equal; read ≤ 8× floor | **0 flips / 43,323,392 lanes total**; read at 1.0×/1.33× the 1.19e-6 fp32-vs-fp64 floor; dense fire rate 0.9995, every fired target live at decision time, payload exactly zero where the guards refuse | GREEN |
| A2 — BORN AS THE COMPOSED MACHINE vs the K15 composed kernel's own launch | 0 differing elements | **0 / 43,061,248** (the K15 birth receipt's own lane count, both gated combos, the mixer's own per-element-signed zero deposit) | GREEN |
| A2 — decisions + the dense lane's recorded target under RANDOM deposits | 0 flips | **0 / 8,683,520** (no deposit and no gate value feeds op selection; the argmax target is invariant) | GREEN |
| A2 — the C5 write-channel + dilution instruments | stated, not guessed | dense/event mass ratio 0.097 at the G2 fixture scale; \|w\|-mass effective rows **62.76 at the zero birth — bit-identical to the composed machine's own** (invariant at birth) | RECEIPTED |
| B — fp64 gradcheck THROUGH THE DENSE-WRITE PATH | every branch | OK (append 108 / correct 80 / none 100; POP/MERGE pinned by the suite's crafted fixtures + the committed kernels' own receipts) | GREEN |
| B — the EIGHT gradients vs the oracle's floors | ≤ 8× floor | dpayload 0.00×, **ddense 0.00×**, dwrite_keys 1.33×, dquery_keys 1.00×, ddecay 0.67×, dgates 0.33×, dstate 0.00×, dkeys 0.00× | GREEN |
| B — the 3-lane trajectory replay (value + key channels) | bit-equal | bit-equal (K11B's lane-generic value replay + K11's stride-generalized key replay) | GREEN |
| D — block-call walls, registered shape, vs the K15 launch in the SAME session | stated | forward **24.10 ms** (K15: 19.55 — the dense lane costs +23 %), backward **18.03 ms** (K15: 17.17); 255 regs / 2 spills (unchanged) | THE DENSE LANE'S HONEST COST |

The lane's own suite is 27/27; the committed K10/K13/K11B/K15 suites are
**84/84 with the K17 module loaded**.

## 3. The read-gain bracket's factual correction (stated before any arm ran)

`read_scale` — the additive read's score-channel GAIN — is inherited at the
**tuned softmax-machine point 2.7870458737509636**
(`optimizer_bakeoff.TUNED_WINNER`), NOT at 2.0: every banked build passes
the tuned value to `velocity_ladder._build_k10`, and the "2.0" of the K13
registration text is the `RoutingPolicy` CLASS DEFAULT the banked build
overrides (the receipts' own `read_scale = 2.0` rows are the BASE build's
stale summary record, not the machine's policy). The bracket {1.0, 2.0,
4.0} runs as specified — three CONFIG arms — with the inherited 2.787
cell's rows cited from the K15 composition's own landed rows at both seeds
(1.3811 / 1.3832 post-cooldown).

## 4. The honest scope

* Probe scale only (20,022,866-target matched exposure, 180+24 steps), seed
  3401 first on every arm; the standing 0.01 win-bar rule owes seed 3402;
  the combined arm fires only on a lever's ≥ 0.01 signal at 3401.
* The dense arm ADDS parameters (+3,538,992 = +2.36 %): the comparison is
  config- and protocol-matched, the asymmetry vs GDN2's 18.3 % gate share
  is quoted in every receipt.
* The −0.0 IEEE corner of the zero deposit is the birth receipt's only
  legal escape; its count is asserted zero on every receipt fixture (and at
  probe scale: 0 flips in 516,734,976 lanes, the K15 receipt's own count).
* The frozen-write falsifier control carries the K11B §2.4 honest
  limitation: full-trajectory bit-identity is NOT achievable by
  construction (the dense lane's backward emits exact-zero payload-gradient
  contributions whose signs are reduction-order artifacts, compounding
  chaotically through the optimizer steps even with the channel frozen);
  the control's per-step loss divergence vs the composed machine's own
  banked training log is measured and quoted in its receipt.
* No `k10_*`/`k11_*`/`k11b_*`/`k13_*`/`k15_*`/`k16_*` source, no committed
  PREREGISTRATION, no `lattice.py` is edited.

## 5. Commands

```bash
# the suite (the MANDATORY fp64 gradcheck through the dense-write path):
/home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k17_dense_write.py -x

# the gate receipts:
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)" && \
CUDA_VISIBLE_DEVICES=$GPU_LEASE_HELD OMP_NUM_THREADS=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k17_gate_receipt

# one arm (the detached leased driver; cards 0/1):
setsid nohup bash experiments/expressivity_tasks/x45_matrix_emender/\
run_k17_write_form_leased.sh dense 3401 \
  > experiments/expressivity_tasks/x45_matrix_emender/k17_runs/\
dense_seed3401.console.txt 2>&1 < /dev/null &

# the assemble:
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.\
x45_matrix_emender.k17_probe_write_form assemble
```
