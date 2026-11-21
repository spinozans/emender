# K18 — THE SHORT-CONV TRANSPLANT: GDN2's registered short depthwise
convolution (kernel 4) on the composed machine's payload path

Stage 18 of the X45 kernel lane, run under the operator's EXPLICIT
DIRECTIVE (the K17 ruling closed the probe-scale quality program; the K18
task is that directive). **The question this stage exists for:** GDN2's
registered block carries a short depthwise convolution (`conv_size=4`,
default ON, `ShortConvolution` in the q/k/v path) that our comparison
machine has NEVER had — no convolution exists anywhere in the K10/K15
mixer lineage — and every transplant this session ran machine-without-conv
vs machine-with-conv. The delta-rule family's own ablations say the short
conv is load-bearing locally (published with/without deltas 0.05–13.5 ppl
across the lineage; V-only placement the literature-supported point in
two independent studies). Is the conv the missing component that owns part
of the composed machine's remaining ~0.03 vs GDN2?

The registration is the DRAFT `PREREGISTRATION_SHORT_CONV.md` (committed
BEFORE any probe measurement, 1c97ba9d). The engine claims are receipted in
`receipts/k18_gates.json`; the arms' claims are in `RESULT_SHORT_CONV.md`.

**Headline, as narrowly as the evidence supports.**

* **The escalated, supervisor-approved design (ruling of 2026-11-26):**
  GDN2's conv sits POST-projection per side (`v_conv1d(v_proj(x))`); the
  K18 transplant is the PAYLOAD side — `payload = v_conv(value(x))`, the
  `v_conv1d` analog, depthwise over 256 channels, kernel 4, bias-free.
  The asymmetry receipt: our key/query projections ALREADY consume a
  4-token causal context window (exactly the conv's receptive-field width),
  so the q/k side has the locality GDN2's q/k convs buy, while the payload
  path had NO temporal receptive field — **the payload conv COMPLETES the
  receptive-field parity GDN2 has on all three sides.** The q/k-conv
  variant (P2 re-registered as decisions conv-influenced) and the
  conv+silu form are NAMED FOLLOW-UPS, not this lane.
* **Two spec errors of the task caught and corrected on escalation:**
  the registered identity birth is **`[0, 0, 0, 1]`** in the STANDARD
  nn.Conv1d/fla tap layout (the LAST tap is the current token; the
  task's `[1,0,0,0]` assumed a current-token-first convention — under the
  actual kernels that is a pure 3-token delay, not identity), and the
  fused **SiLU is NOT transplanted** (ours never silus the payload; GDN2
  silus q/k/v even with the conv OFF, so the temporal filter is the
  ablated object and ours isolates it — a conv+silu arm would break the
  0-flip birth bar by construction).
* **Born as the composed machine, bit for bit:** the identity kernel is
  exactly the pass-through (`torch.equal` in fp32/bf16/fp64; the IEEE
  sign-of-zero corner counted, 0 flips outside exact-zero entries) —
  **0 differing elements in 21,530,624 lanes** at the registered scan
  shape (engine receipt) and **0 in 516,734,976 lanes** on the probe's
  own first train batch, every layer (the arm's own receipt).
* **P2 trivially held:** under RANDOM conv kernels the schedule, the read
  weights, the pre-decision liveness and the key table are bit-identical
  (0 flips in 8,683,520 lanes) while the value state and reads move as
  designed — no conv value feeds op selection.
* **The fp64 gradcheck THROUGH THE CONV PATH green** (both gated
  combinations, append/correct branches fired), and the kernel path's
  gradients through the conv chain at **exactly 0.0 deviation** from the
  eager oracle on both dp and dw. The conv's own adjoint is the analytic
  tap correlation EXACTLY in fp64; the explicit tap sum is
  `F.conv1d(pad(x), w, groups=C)` up to accumulation order (fp64 1.8e-15,
  fp32 1.9e-6).
* **The honest walls:** the conv's forward block call is **0.17 ms** per
  (layer, window) vs the composed scan's own 22.96 ms — **0.75 %**,
  negligible. No new scan kernel exists in this lane: the conv sits in
  front of the unchanged `composed_scan` opaque op pair.

---

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_SHORT_CONV.md` | the DRAFT registration (operator sign-off pending): the escalated placement + the two corrected specs, the parameter accounting, the protocol, the THREE-OUTCOME decision rule (SC-1/SC-2/SC-3), the SH-1 shift clause, the falsifier's bar, NO directional headline prediction |
| `k18_short_conv.py` | the engine: the causal depthwise tap sum (`short_conv1d_causal` — exactly `nn.Conv1d(C,C,4,groups=C,bias=False)` with left zero-padding, deterministic, fp64-clean), `IdentityBornShortConv` (the registered births: identity `[0,0,0,1]`, shift `[0,0,0.5,0.5]`), `ShortConvComposedMixer` (the K15 mixer with the conv on the value path) + the conversion + the parameter-delta accounting |
| `test_k18_short_conv.py` | 18 tests: the conv unit receipts (identity exactness per dtype, the tap layout, causal padding, the F.conv1d equivalence, the analytic tap adjoint, the fp64 gradcheck, the shift semantics, the corner), the born-as-composed mixer receipt, the decisions-invariance under random kernels, the fp64 gradcheck through conv+composed oracle, the kernel-gradient floors through the conv chain, the step-0 gradient receipt, the frozen plumbing, the conversion receipts |
| `k18_gate_receipt.py` | the engine gates into `receipts/k18_gates.json` (A-conv, A2, B, D) |
| `k18_probe_short_conv.py` | the arm runner + the assemble (the birth receipts at probe scale, the corrected scoring, the C5 instrument on the conved payload + the LEARNED KERNEL receipts, the three-outcome adjudication) |
| `run_k18_short_conv_leased.sh` | the detached leased driver (cards 0/1, commit per arm) |
| `tests/test_k18_short_conv_probe.py` | 7 CPU probe tests (the arms, the births, the baselines, the accounting, the rules verbatim, the clause logic, the AdamW routing) |

### 1.1 The machine, stated exactly once

```
payload = v_conv(value(x))     # THE TRANSPLANT: GDN2's v_conv1d position
                               # depthwise, 256 channels, kernel 4, bias-free
                               # born [0, 0, 0, 1]  ->  EXACTLY value(x)
```

Everything else is the COMMITTED K15 composed machine VERBATIM (imported,
never edited): the additive read under the p3 per-channel gate stack, the
five-op schedule, the key channel, the gates at the registered scalar
equivalents. The conv is a VALUE-CHANNEL change — the composed scan and
its committed engines are untouched, autograd carries the conv's adjoint.

### 1.2 The parameter accounting

One depthwise kernel of 256 channels × 4 taps per mixer, 24 mixers:
**+24,576** (+0.015 %) → **158,663,424** total, disclosed and asserted.
The live conv's kernel routes to the hybrid's **AdamW side** (the GDN2
convention mirrored: a depthwise kernel is not a hidden-map matrix); the
frozen controls' kernels are excluded by `requires_grad`.

## 2. The gates (receipts/k18_gates.json)

| gate | bar | measured | verdict |
| --- | --- | --- | --- |
| A-conv — identity birth exact pass-through | `torch.equal` per dtype; corner counted | exact in fp32/bf16/fp64; sign-of-zero corner 1 flip on an exact-zero entry, **0 outside exact zeros**; F.conv1d equivalence fp64 1.8e-15 / fp32 1.9e-6; fp64 tap adjoint the analytic correlation EXACTLY; fp64 gradcheck OK | GREEN |
| A2 — BORN AS THE COMPOSED MACHINE vs the K15 kernel | 0 differing elements | **0 / 21,530,624** (registered shape); **0 / 516,734,976** (the probe's first train batch, layer for layer — the arm's own receipt); step-1 bit-identical to the composed machine's own row | GREEN |
| A2 — decisions-bit-identical under RANDOM conv kernels | 0 flips | **0 / 8,683,520**; the value state and reads move as designed | GREEN |
| B — fp64 gradcheck THROUGH THE CONV PATH | both gated combos, content branches | OK (append 16 / correct 32 fired; the conv's adjoint is upstream-invariant to the scan's branch structure — the committed K15 suite receipts POP/MERGE for the scan itself) | GREEN |
| B — kernel gradients through the conv chain vs the oracle's floors | ≤ 8× floor | dp deviation **0.00×** (floor 2.9e-6), dw deviation **0.00×** (floor 9.2e-5) | GREEN |
| D — block-call walls at the registered shape | stated vs the composed scan's own launch | conv forward **0.17 ms** / backward 0.87 ms vs scan 22.96 ms — **0.75 %** | GREEN |
| the committed suites with the module loaded | all green | **K10/K13/K11B/K15/K17: 111/111** (`k18_runs/k10_k13_k11b_k15_k17_suites_console.txt`, 1:19:50); the lane's own suite 25/25 + the probe tests 7/7 | GREEN |

## 3. The honest scope

* Probe scale only (the 20,022,866-target matched exposure, 180+24
  steps); no board-scale claim before the operator signs off.
* The silu mismatch and the q/k-conv absence are standing, disclosed
  cross-machine differences this lane does not touch (§1 of the
  registration); both are named follow-ups.
* The engine's `IdentityBornShortConv` realizes the conv as the explicit
  deterministic tap sum (NOT cuDNN): bit-exact birth on every backend,
  fp64 gradcheck-able, and the falsifier not hostage to cuDNN's
  nondeterministic backward — the equivalence receipt covers the
  semantics.
