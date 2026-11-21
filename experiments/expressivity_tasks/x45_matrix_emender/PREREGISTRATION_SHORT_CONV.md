Pre-registration draft — THE SHORT-CONV TRANSPLANT (K18): GDN2's
registered short depthwise convolution (kernel 4), on the COMMITTED
composed machine's payload path

**Date:** 2026-11-26 · **Status: DRAFT, operator sign-off pending.** This is
a NEW registration (it adds a LEARNED module to the mixer's input path — a
VALUE-CHANNEL change; the algebra's discrete op selection is untouched),
written and committed BEFORE any probe measurement is claimed. Probe-scale
engineering checks (parity gates, gradchecks, memory, timing) are
pre-registration evidence and are in scope now; **no board-scale (2700 s)
claim is run before the operator signs this document off.** It does not
touch any `k10_*`/`k11*`/`k12_*`/`k13_*`/`k15_*`/`k16_*`/`k17_*` source,
any committed PREREGISTRATION, `lattice.py`, or GDN2 model code — those are
imported, never edited. **This lane runs under the operator's EXPLICIT
DIRECTIVE** (the K17 ruling closed the probe-scale quality program "no more
quality probe arms without an explicit operator directive"; the K18 task is
that directive, and this document is its registration).

## 0. The chain this stage closes — the missing-component candidate with
published backing

GDN2's registered block carries a short depthwise convolution
(`ShortConvolution(hidden_size, kernel_size=conv_size=4, bias=False,
activation='silu')`, applied to each of the q/k/v projections' outputs,
**default ON** — `/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py`) that our
comparison machine has **NEVER had**: no convolution exists anywhere in the
K10/K15 mixer lineage, and every transplant this session ran
machine-without-conv vs machine-with-conv. The delta-rule family's own
ablations say the short conv is load-bearing locally: the FLA
documentation's "do not turn off" warning, and published with/without
deltas spanning **0.05 ppl (Kimi Linear ablation) to 13.5 (DeltaNet
LAMBADA)** — heterogeneous across the lineage, never zero. Two independent
studies place **V-only above Q- or K-only** for the conv's placement. This
is the missing-component candidate for the composed machine's remaining
**+0.0289/+0.0311 vs GDN2's 1.3521734476089478** (the composed machine's
own 1.3811104467936925 / 1.383232252938407 post-cooldown, the program's
best absolute rows, parity-within-band under the supervisor's ruling of
2026-11-25), with published backing.

**No single expected effect size exists at our setting.** That is exactly
why the decision rule below is a THREE-OUTCOME rule, not a point prediction.

## 1. The architecture under test

**The COMMITTED K15 composed machine VERBATIM** (the banked winner build
`velocity_ladder._build_k10` + `convert_model_to_composition` — the gates
born at the registered scalar equivalents; imported byte for byte, never
modified), at the matched training surface (the banked winner lineage:
hybrid Muon at the inherited 4× point, every gate projection on the AdamW
side, the K15/K16 arms' own routing), at the exact probe protocol (§3), with
ONE transplant added per §1.1.

### 1.1 THE TRANSPLANT — the short depthwise convolution on the PAYLOAD
path, at GDN2's own position

**The escalated placement (dug from GDN2's source, escalated, and approved
by the supervisor, 2026-11-26):** GDN2's conv sits POST-projection, per
side — `q_conv1d(q_proj(x))`, `k_conv1d(k_proj(x))`, `v_conv1d(v_proj(x))`
— three separate depthwise convs, NOT a conv on the mixer input before the
projections. The K18 transplant is the PAYLOAD side: **`payload =
v_conv(value(x))`** — `v_conv1d`'s exact functional analog, depthwise over
`heads·value_dim = 256` channels, **kernel 4** (GDN2's registered
`conv_size`), **bias-free**, in front of the unchanged composed scan.

**The asymmetry receipt that justifies the single-side transplant (the
hypothesis's mechanism statement, verbatim per the ruling):** our key/query
projections already consume a **4-token causal context window**
(`DEFAULT_CONTEXT=4` — exactly the conv's receptive-field width), so the
q/k side **already has the locality GDN2's q/k convs buy**, while the
PAYLOAD path had **NO temporal receptive field at all** (`value(x)` sees the
current token only). The payload conv is therefore not a shortcut of
GDN2's triple; it is the faithful functional analog that **COMPLETES the
receptive-field parity GDN2 has on all three sides.** The q/k-conv variant
(with P2 re-registered as decisions conv-influenced) is a NAMED FOLLOW-UP
if the payload conv banks a win — not this lane.

**The birth (the K15 birth discipline, escalated and approved):** the conv
is born at the **IDENTITY KERNEL `[0, 0, 0, 1]`** in the **STANDARD
nn.Conv1d/fla tap layout** — the LAST tap multiplies the current token
(tap 0 is the oldest, `x[t-3]`; fla's `causal_conv1d` and `nn.Conv1d` both
use this layout, so the learned kernels stay directly comparable to
GDN2's and to the literature's). *The task's original `[1,0,0,0]` spec was
written assuming a current-token-first convention — wrong under the actual
kernels (a literal `[1,0,0,0]` is a pure 3-token delay); the correction was
escalated and approved.* At that birth the conv is EXACTLY the pass-through
(`1.0·x` is exact in every dtype; the zero taps add ±0.0), so **the machine
is BORN AS THE COMPOSED MACHINE bit for bit** (the 0-flip bar, §2), and the
conv then LEARNS to deviate — a **VALUE-CHANNEL change**: no conv value
feeds op selection, thresholds, routing similarity or liveness (P2
trivially held — the 0-flip receipt under RANDOM kernels), so the frozen-conv
falsifier must stay bit-identical forever.

**The fused SiLU is NOT transplanted (escalated, approved):** GDN2's
`ShortConvolution` carries `activation='silu'` inside the conv op, and even
`use_short_conv=False` keeps the silu (`silu(q_proj(x))`). Our machine
never silus the payload — a standing, pre-existing cross-machine mismatch.
Because GDN2's own conv ablation holds the silu constant in BOTH arms, the
conv's own ablated delta in GDN2's family is the **TEMPORAL FILTER**, and
ours isolates the same object. A conv+silu arm would break the 0-flip
birth bar by construction and confound the conv's delta with the silu's;
it is a named follow-up with its own step-1-tolerant registration.

**The engine realization, stated:** the conv is evaluated as an explicit,
deterministic causal tap sum — exactly `nn.Conv1d(C, C, 4, groups=C,
bias=False)` with left zero-padding (fla's `ShortConvolution` semantics),
receipted equivalent to `F.conv1d` up to floating-point accumulation order
(fp64 within 1e-15, fp32 within 1e-6) — so (a) the birth is bit-exact and
deterministic on every backend, (b) the fp64 gradcheck can pass through
the conv in fp64 (fla's fused kernel casts to fp32 and cannot), and (c)
the falsifier's full-trajectory bit-identity is not hostage to cuDNN's
nondeterministic backward. **No new scan kernel exists in this lane**: the
conv sits in the mixer's input path BEFORE the unchanged `composed_scan`
opaque op pair, so autograd carries the conv's own adjoint.

**The parameter delta (stated, not hidden):** one depthwise kernel of
256 channels × 4 taps per mixer, 24 mixers — **+24,576** (0.015 % of the
machine) → **158,663,424** total. Asserted by the probe, not hidden.

**The routing:** the live conv's kernel lands on the hybrid's **AdamW
side** (the GDN2 convention mirrored — a depthwise conv kernel is not a
hidden-map matrix; Muon's Newton-Schulz does not apply to it), alongside
the p3 gate tensors. The frozen controls' kernels carry
`requires_grad=False` and are excluded by the filter — the split then
equals the composed machine's own exactly.

### 1.2 The arms

* **`conv`** — the transplant arm: the conv born identity, **learns**.
* **`conv_frozen`** — the **FROZEN-CONV FALSIFIER CONTROL** (mandatory,
  the K11B R4 precedent): the kernel frozen at the identity birth,
  `requires_grad=False`. The bar: step-1 bit-identity to the composed
  machine's own row, a landing at parity with its same-seed banked row,
  and the full-trajectory receipt against the composed machine's own
  banked training log (the K17 falsifier's perfect bit-identity is the
  expectation to beat; ULP-scale divergence from exact-zero gradient
  contributions is the honest escape, the K11B §2.4 precedent).
* **`shift_fixed`** — the **FIXED NON-IDENTITY CONTROL** (the published
  mechanism decomposition's own recommendation): the conv at the FIXED
  token-shift kernel **`[0, 0, 0.5, 0.5]`** (the average of the current
  and the previous token, in the standard tap layout), frozen, **never
  learned** — the arm that separates **ANY LOCAL MIXING** from a
  **LEARNED FILTER**. **NO birth parity exists or is asserted** (a
  different function at birth — disclosed, the config arms' own
  convention). If cards/time pinch, this arm is dropped BEFORE the
  falsifier (the falsifier is non-negotiable).

## 2. The parity gates (committed before any probe measurement)

* **Gate A-conv** — the conv's own receipts at the registered channel
  count: the identity birth is exactly the pass-through in fp32/bf16/fp64
  (`torch.equal`; the IEEE sign-of-zero corner on exact-zero entries
  counted — invisible to every `==`-based parity bar and to the loss),
  the explicit tap sum equals `F.conv1d` up to accumulation order, the
  fp64 tap adjoint is the analytic correlation EXACTLY, and the fp64
  gradcheck through the conv passes.
* **Gate A2 — the birth receipts**: at the identity birth the conv'd
  payload is bit-equal to the raw projection and the composed kernel's
  every channel launched on the conv'd payload is bit-equal to its own
  launch on the raw payload (the born-as-the-composed-machine receipt,
  0 differing elements); the DECISIONS-BIT-IDENTICAL bar under RANDOM
  conv kernels (P2 trivially held); the p3 gates' birth checks re-run.
* **Gate B — MANDATORY: the fp64 gradcheck THROUGH THE CONV PATH** (the
  conv's tap adjoint attached in front of the composed walk, both gated
  combinations, every content-carrying branch fired) — then the kernel
  path's gradients through the conv chain within `NOISE_FACTOR` (8×) of
  the measured fp32-vs-fp64 floor. The conv's step-0 gradient receipt:
  the identity-born kernel is NOT a dead parameter (the payload path
  reaches it at step 0 — no two-step wake-up, unlike the zero-born
  gates).
* **Gate D** — the honest walls: the conv's own forward/backward
  block-call cost per (layer, window) at the registered shape, stated
  against the composed scan's own same-session launch.
* The committed K10/K13/K11B/K15/K17 suites re-run green with the new
  module loaded.

## 3. The protocol

Every arm: the EXACT probe protocol — the 20,022,866-target
matched-exposure landing at 180 constant-LR steps on the registered
flagship stream (identity asserted), the document-disjoint calibration
eval (recorded-read path), the honest clock, the protocol's own grad-clip
(1.0, error_if_nonfinite), the banked knobs asserted at launch (betas
(0.9, 0.9392), weight decay 0.17422, warmup 45 on both hybrid groups,
microbatch 8), the inherited 4× hybrid-Muon point (K16's own receipt:
already the composed machine's bracket optimum) — THEN the corrected
scoring: the 24-step linear cooldown PRIMARY (post-cooldown calibration
eval), EMA-0.99 at step 180 SECONDARY (ranking-only), constant-180
retained for banked comparability. Seed 3401 first on every arm.

**The anchors (all quoted, none re-run):** the composed machine's OWN
rows at BOTH seeds — post-cooldown **1.3811104467936925 (3401) /
1.383232252938407 (3402)**, constant-180 1.433674931526184 /
1.4341776371002197, EMA 1.424389226096017 / 1.4385861498968942 — are the
BASELINE; GDN2's post-cooldown **1.3521734476089478** is the gap
reference; the K13 additive rows (1.3928/1.3845) and the banked winner's
corrected anchor (1.4338, seed 3402) are context. The bands on every
margin: the 0.075-nat endpoint-variance band and the winner's 0.0119 seed
delta.

**PRE-STATE: NO DIRECTIONAL PREDICTION.** The honest prior is genuinely
two-sided: the published effects are heterogeneous (0.05–13.5 ppl) and
ours may be at the small end; and our machine's payload path is
current-token-only by construction (write what you read), so a temporal
blur could plausibly HURT the slot semantics. The decision rule is
therefore a THREE-OUTCOME rule, stated BEFORE any result.

## 3.1 Mechanism duties (reported as measured, no bar pre-stated)

* **The K15 mechanism instrument ON THE CONVED PAYLOAD** (the machine the
  arm trains on): |w|-mass effective rows, the top-row-share collapse
  detector, occupancy, the identity-rewrite share, the op mix, and ALL
  THREE p3 gate channels' born-identity drift — imported verbatim, re-run
  on the conv machine's launch. **Gate telemetry expected UNMOVED** (the
  conv touches neither the gates nor the decisions).
* **The LEARNED KERNEL receipts (the mechanism's direct read, the
  supervisor's addition):** per layer — the per-tap means and spreads
  across the channels (did the conv learn a token-shift-like mix of the
  last few payload vectors?), the kernel L2 norm, and the distance from
  the identity birth. The frozen controls' rows are the birth kernels'
  own, unchanged by construction.
* **The decorative-store sweep** (`k10_read_mask_sweep`, imported
  verbatim) at BOTH the step-180 and post-cooldown states — **the
  uniformity hypothesis's direct test: does the local temporal filter
  make more of the 24 layers load-bearing?** (The banked regime's own
  rows: 4–5 of 24 layers ≥ 0.02 dNLL.)
* The dissociation clause (K11 §4's own): if the mechanism numbers do
  not move while NLL does (or vice versa), the dissociation is reported
  honestly.

## 4. The pre-stated decision rule (the THREE-OUTCOME rule)

* **SC-1 (LOCAL-FEATURES)**: the conv arm's post-cooldown row improves on
  the composed machine's same-seed banked row by **≥ 0.01 at seed 3401**
  ⇒ local-features owns a share of the residual; **seed 3402 is owed
  BEFORE any claim** (the standing win-bar rule); if replicated at both
  seeds, the attribution chain gains the LOCAL-FEATURES term and the
  missing-component story replaces part of the uniformity story.
* **SC-2 (PARITY)**: improvement inside the 0.0119 seed band (and/or no
  replication at the second seed) ⇒ **the conv is NOT the residual's
  owner on byte-LM text at probe scale; the uniformity attribution
  stands with one more receipt** — a real possibility, stated BEFORE the
  result: the published effects are heterogeneous and ours may be at the
  small end.
* **SC-3 (HURTS)**: the conv arm lands WORSE than its same-seed baseline
  beyond the band ⇒ **the payload path's current-token-only structure
  was load-bearing for the slot semantics (write what you read, no
  temporal blur)** — receipted as a machine-family difference, not a
  failure.
* **SH-1 (the shift decomposition)**: the fixed-shift control decomposes
  the conv's effect — **ANY LOCAL MIXING** (the shift arm's row) vs a
  **LEARNED FILTER** (the conv arm's row); no directional prediction is
  pre-stated for the decomposition; both rows are reported as measured
  against the composed machine's own AND against each other.
* **The falsifier's own bar**: (a) the control's step-1 train loss MUST
  be bit-identical to the composed machine's own same-seed row; (b) the
  control's landing MUST be at parity with the composed machine's
  same-seed banked row; (c) the full-trajectory bit-identity against the
  composed machine's own banked training log is receipted (the K17
  falsifier's perfect receipt is the expectation to beat; ULP-scale
  divergence is the honest escape, quoted if it fires).
* **SECOND SEEDS (the standing win-bar rule)**: any arm whose seed-3401
  improvement is ≥ 0.01 owes seed 3402 BEFORE any claim.
* **THE BAND DISCIPLINE (the K15/K16/K17 clause, verbatim)**: any arm
  whose post-cooldown row would put the machine AT OR PAST GDN2's
  1.3521734476089478 is reported with BOTH scorings (post-cooldown
  PRIMARY, constant-180 retained) and BOTH seeds before any claim, and
  ESCALATED, not adjudicated here. Any inversion of a standing quality
  conclusion escalates likewise.
* **The final gap-quote phrasing is the supervisor's to adjudicate**
  (the K13 clause-conflict precedent): this lane's assemble states the
  measured rows and the flags.

## 5. Honest scope

* Probe scale only (the 20 M matched exposure, 180+24 steps, 7-window
  eval); no board-scale claim before the operator signs off.
* The conv arm ADDS parameters (§1.1: +24,576 = +0.015 % — negligible,
  but quoted in every receipt).
* The silu mismatch and the q/k-conv absence are STANDING, DISCLOSED
  cross-machine differences this lane does not touch (§1.1); a conv+silu
  arm and the q/k-conv variant are named follow-ups, not this
  registration.
* **The X46 disclosure (the supervisor's addition, stated here for the
  record):** the machine that beat the Welch-bound ceiling in X46
  (gdn2-fixed 0.93 vs 0.80) is the conv-HAVING machine — GDN2's
  registered block carries the short conv ON (conv_size=4, default), so
  the ceiling refutation was achieved WITH the conv present and is not
  conv-attributable-in-doubt; the linear ceiling was derived from the
  key-space geometry regardless of the conv, and the trained machine
  exceeded it with ALL its components, conv included.
* Single-protocol rows; the endpoint-variance band (0.075) and the
  winner's seed delta (0.0119) are quoted with every delta; the
  composed-machine-vs-GDN2 row is a cross-machine context row at the
  same probe protocol, not a controlled comparison.
* **Expectations discipline (the brief's own context):** published
  with/without-conv deltas range 0.05 ppl (Kimi Linear) to 13.5
  (DeltaNet LAMBADA); kernel-size-3 beat 4 in two of three published
  sweeps but 4 is the family default (and GDN2's registered value —
  this lane transplants the registered object, not the sweep winner);
  V-only placement helped more than Q- or K-only in two independent
  studies — our payload-only choice is the literature-supported point.
  No single expected effect size exists at our setting.
