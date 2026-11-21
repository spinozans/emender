# X45 K23 — THE STEP-LEVEL SPEED PROGRAM: interleaving, flows, graphs, and the floor, answered with receipts

Result document for the K23 task (K21's direct follow-up). **Zero training
ran in this lane** — profiling + micro-benchmarks only, one card (lease on
GPU 5, RTX 6000 Ada, released cleanly on exit; `receipts/k23_speed.json`,
console in `k23_runs/`). The registration-style pre-state held: no
directional prediction was made, and the decomposition IS the result. The
committed engines were imported verbatim and never modified — the receipt
hash-binds every one of them, the working tree was clean on every engine
path, and the lane's only new code is the driver (`k23_step_speed.py`), the
task's named **scratch** file (`k23_batch_brackets.py`), and K21's scratch
file (`k21_micro_brackets.py`, imported for the stack re-validation).
**Nothing validated here is integrated anywhere** — the parallel K22 hybrid
lane is hash-binding the committed engines, and integration of validated
speedups is a later, sequenced lane (the supervisor's ruling, recorded
below).

**The question.** K21 receipted the composed scan as latency-bound (16
independent rows on 142 SMs, 0.7% of DRAM peak, 17% of its own 16-SM issue
ceiling, 5.9× above its own 3.74 ms floor) and the step wall at 10.62 s
with the scan forwards at ~4.5 s. The task asked four levers, each under
the standing bar — **bit-identity per output element, or it is a new
engine version** — and each answered by measurement below.

**Headline, as narrowly as the evidence supports.**

* **The step is device-bound, and the "non-mixer ~6 seconds" are mostly
  the scan's own BACKWARD.** The clean step wall re-measured this session
  is 10.587 s (K20's 10.624, −0.4%); the torch-profiler decomposition (2
  profiled steps, device kernels only): **the scan forward kernel 4.169 s
  (40.1%), the scan backward's five-kernel set 3.905 s (37.5%), eager
  elementwise/copy glue 1.340 s (12.9%), GEMMs (projections/MLP/head,
  fwd+bwd) 0.915 s (8.8%), the optimizer 0.067 s (0.6%)**, memcpy/memset
  5 ms, the fused CE head 0.6 ms. The **host gap is 185.5 ms = 1.75% of
  the wall** — the step is NOT launch-bound, which bounds the graphs and
  compile levers from above before they are even bracketed. The launch
  census: **105,534 kernel launches per step**, of which 74,812 are under
  50 µs (1.127 s total) and 30,720 are the scan backward's segment train
  (5 kernels × 32 segments × 192 scans). Nothing outside the four levers'
  reach dominates the wall — the two largest items ARE the scan kernels,
  which is lever 1's object.
* **Lever 1, batching/interleaving: BIT-IDENTICAL, and it is the big one.**
  All 8 window-groups of a layer in ONE committed kernel launch (batch
  8→64, grid 16→128 programs), on layer 6's captured production scan
  arguments of a full probe step: **every output channel (all 13) is
  `torch.equal` per group, at every bracket**. The wall is FLAT in rows —
  22.06 / 22.20 / 22.22 / 22.27 ms for 1/2/4/8 groups (16/32/64/128 rows)
  — the pure latency-bound signature: the committed 8-call layer-step
  costs 173.3 ms, so **batching alone is 7.78× on the forward**. The
  backward is bit-identical too (all 7 gradient tensors per group,
  `torch.equal`): **159.6 → 83.3 ms per layer-step = 1.92×** — the
  adjoint's kernel set is partially throughput-bound, unlike the forward.
* **The K21 1.32× patch holds UNDER THE BATCHED GRID** — the interaction
  the task named is receipted: the stack (lazy-norms + lazy-old-row +
  split-fold at 8 warps, 1 stage) on the batch-64 grid is bit-identical
  per group and costs **16.873 ms per layer-step = 10.27×** vs the
  committed loop; the same-session single-group re-validation reproduces
  K21's receipted **16.785 ms exactly** (1.318×). The recommended
  combination is batched grid + stack at (8 warps, 1 stage).
* **The composite: a 3.32× bit-identical per-layer-step scan pair
  (fwd+bwd 332.9 → 100.2 ms)** — above the task's own 2× escalation bar,
  **escalated as instructed** (the supervisor acknowledged and ruled;
  see §5). Projected from measured terms: **pure-ours step 10.587 → 4.97
  s; per-mixer pair ratio 7.28× → 2.78× (below the 3× pole K20's
  viability example assumed); hybrid 3-of-24 exposure 73.1% → 87.8%** of
  GDN2's matched-time exposure.
* **Lever 2, flows/fusion: mostly parked, honestly.** `torch.compile` of
  the whole mixer (the scan op opaque, the glue around it fused) is
  **bit-identical on the forward — for 0.097 ms per mixer** (23.291 →
  23.194 ms; ~0.019 s/step, 0.2%): receipted, immaterial. The compiled
  **backward is BLOCKED** by the committed backward op's fake-tensor
  registration (a stride assertion inside `x45::k15_scan_bwd`'s fake
  kernel; the failure string is in the receipt) — fixing that is an
  engine edit with its own parity gates, parked. The **compiled MLP
  FLIPS BITS** (1.911 → 1.719 ms, `torch.equal` fails): parked for a
  new-engine-version decision, not smuggled in. The 1.34 s elementwise
  glue bucket therefore stays essentially unrecovered under the
  bit-identity bar at this shape.
* **Lever 3, graphs: parked, bounded by measurement.** CUDA-graph capture
  of the scan's own kernel trains WORKS (despite the expandable-segments
  allocator) and replays **bit-identically — for 1.011× (per-group
  committed train) and 1.005× (the batch-64 train)**: the ~200-launch
  backward train pipelines behind its own device work, and the step-level
  host gap is 1.75%. Host-issue receipts agree (the forward issues in
  0.31 ms against a 22.06 ms wall; the pair in 13.6 ms against 42.4 ms).
  Graphs are bounded by ~1.75% of the wall — not worth an engine lane.
* **Lever 4, the floor bracket.** K21's floor for one group's scan at the
  16-row ceiling is 3.741 ms; the batched launch's 8× work runs at the
  128-row ceiling the batch legitimately exposes (the semantics forbid
  token-parallelism, not group-parallelism), so **the floor for the whole
  8-group layer-step is the same 3.741 ms**. Against it: the committed
  layer-step forward (173.3 ms) sits 46.3× above (the 8× serialization of
  K21's 5.9×); **the composite (16.873 ms) sits 4.51× above, closing
  92.3% of the step-seconds gap** between committed and floor. The
  remaining 4.5× is the same in-kernel headroom K21 pinned to the
  receipts bar (`tl.sum`'s exact order for the reads channel, no-FMA
  fp32, no tensor cores) — a rewrite lane's to claim or a bar change the
  operator must sign off. The backward's floor was never modeled (K21's
  boundary); its 1.92× here is pure re-scheduling, and its in-kernel
  levers remain the next profiling lane's object.
* **The memory bill, measured: the naive integration does NOT fit.** One
  window-group's retained forward graph is **23.54 GiB** on the
  allocator's own books (the MLP's hidden activations dominate); the
  all-groups-forward-then-backward restructure would hold **188.3 GiB
  against 21.1 GiB free** (47.4 GiB card). The batching lever's
  integration is therefore NOT the naive loop swap — it is a layer-major
  schedule (all groups' forward at each layer, batched 128-row scans,
  backward in group order), whose bill is the integration lane's first
  measurement, and per the supervisor's ruling it must be re-measured at
  the BOARD protocol's shape before any board run depends on it.

**The answer, in one sentence:** the step's biggest recoverable cost was
never in the eager glue — it is the scan kernels' 16-row exposure, and
re-scheduling the window-groups into one launch is bit-identical, flat in
rows, and worth 3.32× on the whole scan pair today (10.27× on the forward
with the K21 stack), taking the projected pure-ours step from 10.6 s to
~5.0 s and the hybrid 3-of-24 exposure from 73% to 88% — with the
integration gated, correctly, behind an end-to-end restructure
bit-identity falsifier and a board-shape memory receipt, while flows and
graphs park under their own honest numbers.

---

## 1. Method (the boundary, stated once)

Same discipline as K20/K21: the committed machine is the K18 conv arm
verbatim (`velocity_ladder._build_k10(TUNED_WINNER)` +
`convert_model_to_composition` + `convert_model_to_short_conv(birth=
'identity')`), 158,663,424 parameters asserted, engines imported and never
modified, hash-bound in the receipt; the lane refuses to run on a dirty
engine tree. The step is the banked probe protocol's honest clock verbatim
(`K20.steps_ours`'s body: sync → stream read → zero_grad → lr → every
microbatch's fwd/bwd → clip → `HybridMuonX45.step` → sync; 64 windows, 8
window-groups, microbatch 8). Timings: CUDA-synced medians, spread stated,
warmups include kernel compilation; bit-identity is always `torch.equal`,
never allclose. The scan-argument fixture is layer 6's **exact production
arguments for all 8 window-groups of one probe step** (a recording shim on
the committed mixer's own `composed_scan` binding, removed before any
timing; capture losses 5.784–5.809, K21's regime; the same 8 groups feed
every lever). Hardware counters remain unavailable on this host
(`ERR_NVGPUCTRPERM`, K21's receipt), so the host-vs-device split is
clean-wall minus summed kernel self-CUPTI-time, honestly labeled.
`k23_speed.json` records two passes in `driver_notes`: run 1 (all stages;
its profile stage failed on a driver-side bucketing bug and its flows
forward receipt was overwritten by the backward-compile failure — both
instrument bugs, fixed and disclosed) and run 2 (`--stages
profile,flows,verdict` re-run on the same receipt). The batching, graphs,
capture, and walls receipts are run 1's, unchanged.

## 2. The full-step decomposition (levers 2 and 3's evidence base)

Per step (torch profiler, 2 steps, device kernels only; the clean wall is
the unprofiled stage's median):

| bucket | ms/step | share | launches/step |
|---|---|---|---|
| the composed scan FORWARD kernel (`_k15_seq_fwd_kernel`) | 4,169.3 | 40.1% | 192 |
| the composed scan BACKWARD kernels (5-kernel segment train × 32 segments × 192 scans) | 3,904.7 | 37.5% | 30,720 |
| elementwise/copy glue (conv taps, causal-context gather, sigmoids/gates, casts, autograd machinery) | 1,340.5 | 12.9% | 56,493 |
| GEMM (mixer projections, MLP fwd+bwd, head) | 914.7 | 8.8% | 10,772 |
| the optimizer (Muon + fused aux AdamW; isolated profile 95.5 ms incl. shared-name elementwise) | 66.6 | 0.6% | 2,550 |
| memcpy/memset | 5.2 | 0.05% | 4,799 |
| the fused cross-entropy head | 0.6 | 0.01% | 8 |
| **device busy** | **10,401.5** | **98.25% of the wall** | **105,534** |
| host gap (launch/dispatch/data pipeline) | 185.5 | 1.75% | — |

The backward train's per-kernel census at the committed grid (per launch
averages): value walk 181 µs, prep 177 µs, key walk 164 µs, value
trajectory replay 86 µs, key trajectory 28 µs — ~6,144 launches each per
step. Cross-check against the micro-receipts: 24 × 173.3 ms = 4.16 s
forward ✓; 24 × 159.6 ms = 3.83 s backward ✓ (the profiler sees the
in-vivo numbers; the micro-brackets see the op-level walls; they agree
within 2%). **The task's "~4.5 s of mixers" is the scan forward alone;
the largest single "glue" item is the scan's own backward — a different
kernel set from the forward's, and the object K21 already named as the
next lane's.** No bucket outside the four levers' reach dominates.

## 3. Lever 1 — batching/interleaving (the receipts)

The bar: `torch.equal` on all 13 output channels of the committed opaque
op pair, **per group**, on the captured production arguments; for the
backward, all 7 gradient tensors per group at fixed seeded `dreads`.
Independence is structural (each program's tile arithmetic is untouched;
the grid only chooses the SM), but the Triton lowering's warp-layout
machinery made every bracket an empirical question — and every bracket
passed.

| variant | bit-identical | ms per layer-step (8 groups) | speedup |
|---|---|---|---|
| committed: 8 sequential `composed_scan` calls (the production form) | — | 173.3 | 1.00× |
| committed kernel, batch 2 groups (32 rows) | **yes** | 22.20 | 7.81× |
| committed kernel, batch 4 groups (64 rows) | **yes** | 22.22 | 7.80× |
| committed kernel, batch 8 groups (128 rows) | **yes** | 22.27 | 7.78× |
| **K21 stack (8 warps, 1 stage) under the batch-64 grid** | **yes** | **16.87** | **10.27×** |
| K21 stack (8, 3) under the batch-64 grid | yes | 17.63 | 9.83× |
| K21 stack (8, 1), single-group grid — K21 re-validation | yes | 16.785 (K21: 16.785) | 1.318× |
| committed backward: 8 sequential adjoint ops | — | 159.6 | 1.00× |
| **committed backward, batch-64 (7 grads per group equal)** | **yes** | **83.3** | **1.92×** |
| **composite pair (fwd stack-batched + bwd batched)** | **yes** | **100.2** | **3.32×** |

Readings, narrowly:

* The forward wall is **flat from 16 to 128 rows** — the card absorbs 8×
  the work at the same wall, which is the latency-bound diagnosis
  confirmed by re-scheduling alone. Per-group amortized, the composite
  forward costs **2.11 ms/group** against the committed 21.9.
* The backward improves only 1.92×: its kernels are partially
  throughput-bound at 128 rows (their per-launch walls grow with the
  grid). Its in-kernel levers were NOT bracketed (K21's boundary; a
  different kernel set) — at the composite the backward owns 83% of the
  scan pair, and the next profiling lane's object is named.
* The single-group stack row reproduces K21's banked 16.785 ms **to the
  microsecond**, and the stack composes with the batched grid without a
  single flipped bit — the composed receipt holds for the combination
  this lane recommends.
* **The 3.32× composite cleared the task's 2× escalation bar and was
  escalated as instructed** (message receipted in the lane log; the
  supervisor's three rulings are recorded in §5 and shape the
  integration recommendation).

## 4. Levers 2 and 3 — flows and graphs (mostly parked, with receipts)

| piece | bit-identical | eager → compiled/replay | verdict |
|---|---|---|---|
| `torch.compile` of the whole mixer FORWARD (glue fused around the opaque scan op) | **yes** | 23.291 → 23.194 ms | receipted; 0.097 ms/mixer (~0.019 s/step) — immaterial alone |
| the compiled mixer BACKWARD | **blocked** | — | the committed `x45::k15_scan_bwd` fake-tensor kernel rejects the compiled graph's strided views (failure string in the receipt); fixing it is an engine edit — parked |
| `torch.compile` of the SwiGLU MLP | **NO — bits flip** | 1.911 → 1.719 ms | parked for a new-engine-version decision, per the bar |
| CUDA graph of the per-group committed fwd+bwd train | yes (replay) | 42.377 → 41.910 ms | 1.011× — launch overhead is pipelined away |
| CUDA graph of the batch-64 fwd+bwd train | yes (replay) | 105.966 → 105.440 ms | 1.005× — same |
| host-issue (no-sync) receipts | — | fwd 0.31 ms vs 22.06 wall; pair 13.6 vs 42.4; batched pair 13.8 vs 106.0 | the host always issues far inside the device wall |

With the step-level host gap at 1.75% of the wall, **graphs are bounded
below relevance** and the compile lever's bit-identical share is 0.2% of
the step: both park, with the numbers that park them. The 1.34 s
elementwise bucket is real, but its compiled forms either flip bits (MLP)
or are blocked (the backward op's fake kernel) — under the receipts bar
it is not recoverable today, and that is stated as the finding, not
hidden.

## 5. The composite, the exposure-table update, and the escalation record

Anchor rows cited from K20/K21; the composite row is computed from this
session's measured terms only (the formula and every input are in the
receipt's `verdict`): per-group composite pair = (16.873 + 8×glue_fwd)/8 +
(83.317 + 8×glue_bwd)/8 = 18.298 ms, with glue_fwd = 1.274 ms (K21's own
mixer-minus-scan) and glue_bwd = 4.598 ms (K20's mixer backward minus the
measured scan adjoint).

| scenario | per-mixer pair (ms) | ratio vs GDN2's 6.589 | 3-of-24 hybrid step (s) | 3-of-24 exposure | 4-of-24 | 6-of-24 | 8-of-24 | pure-ours step (s) |
|---|---|---|---|---|---|---|---|---|
| K20 committed anchor | 47.98 | 7.28× | 4.266 | 73.1% | 4.598 / 67.8% | 5.260 / 59.3% | 5.922 / 52.7% | 10.624 (10.587 this session) |
| K21 stack only (fwd), no restructure | 41.6 | 6.44× | 4.14 | 75.4% | — | — | — | 9.61 |
| **K23 composite (batched + stack, restructure-gated)** | **18.30** | **2.78×** | **3.554** | **87.8%** | 3.648 / 85.5% | 3.835 / 81.3% | 4.023 / 77.5% | **4.97** |
| at the forward's semantics floor (K21's analytic bound; backward untouched) | — | 4.47× | 3.82 | 81.6% | — | — | — | 7.10 |

(3-of-24 at GDN2's registered AdamW instead of the banked Muon: 3.400 s,
91.7% exposure.) The composite's pure-ours step cross-checks against the
profile buckets: 10.587 − (4.169 − 0.405) − (3.905 − 2.000) − 0.019 ≈
4.90 s, consistent with the wall-based 4.97 s within the sync/overlap
slack between loop walls and summed device time (scan savings 5.60 s +
fusion 0.02 s; the graph term is folded at its measured 0.53 ms per
layer-step).

**Escalation record.** The 3.32× bit-identical composite cleared the
task's 2× bar and was escalated to the supervisor with the receipts. The
rulings, which shape §6: (1) **integration sequencing stands** — the
integration lane waits for K22's hybrid arms (the hybrid's science is
matched-exposure and unaffected; its BOARD case is what this composite
improves); (2) **the restructure falsifier is the gate, and its failure
is a fork, not a dead end** — if end-to-end step bit-identity cannot be
made to hold, the composite banks as a NEW ENGINE VERSION decision (new
anchors, its own banked rows, old rows stay on the old engine), and that
fork goes to the operator with these numbers; (3) **the memory bill gets
a board-shape receipt** — the probe-shape bill below is the input, and
the integration lane must re-measure at the board protocol's shape before
the hybrid board run depends on it.

**The memory bill, measured at the probe shape:** one window-group's
retained forward graph is **23.54 GiB** (steady-state allocator books,
optimizer states resident; the MLP's 3352-wide hidden activations
dominate); the all-groups-at-once restructure would hold **188.3 GiB
against 21.1 GiB free** — impossible. The viable integration is a
**layer-major schedule**: forward all 8 groups at each layer (the batched
128-row scan), keep only that layer's saved tensors and the per-layer
boundary residuals, backward in group order before moving to the next
layer. Its peak (per-layer saved tensors ~0.9 GiB + boundary stash
~3.6 GiB + params/grads/optimizer ~1.8 GiB, by arithmetic on the
measured components — labeled reasoning, not a receipt) fits the card,
but the bill is the integration lane's first measurement, at board shape.

## 6. The integration recommendation (the sequenced follow-up)

Each step names its own **trajectory falsifier** — the observation that
would refute it, per the house discipline. Nothing below is integrated in
this lane.

1. **The K21+K23 forward stack** (the three dataflow-identical levers at
   (8 warps, 1 stage)) into the committed scan kernel — bit-identical at
   both receipted grids, no restructure required, pure engine change.
   *Falsifier:* the committed engine's own parity gates on the new hash —
   the K15 step-1 parity, the K18 identity-birth 0-flip control, and a
   banked row's own digits (the K19 winner-confirm reproduction) must hold
   bit for bit; any flip stops the integration and re-banks it as a new
   engine version. **Sequence: after K22's arms complete** (ruling 1).
2. **The batched launch + the layer-major step restructure** (the big
   one, 3.32× on the scan pair): all window-groups' forward at each
   layer in one 128-row launch, backward in group order. *Falsifiers,
   both required:* (a) **end-to-end step bit-identity** vs the committed
   loop at fixed seed and data — parameter updates `torch.equal` over
   the probe protocol's step-1 gates and a banked row's digits (the
   grad-accumulation order and the data-stream consumption order are
   the two things the restructure can silently change); failure = the
   ruling-2 fork, taken to the operator, never silently absorbed;
   (b) **the memory fit at the board protocol's shape** (ruling 3) —
   the probe-shape naive bill (188.3 GiB) already excludes the
   all-graphs form; the layer-major staging's peak must be measured
   before any board run depends on it.
3. **The compiled mixer forward glue** (bit-identical, 0.097 ms/mixer):
   fold into step 2's lane only if its delta survives at board shape.
   *Falsifier:* mixer-output and step-1 bit-identity on the integrated
   form. The compiled backward stays **parked** until the committed
   backward op's fake-tensor kernel is fixed (an engine edit with its
   own parity gates) and the compiled MLP stays **parked** (bits flip —
   a new-engine-version decision, not a speedup).
4. **NOT recommended:** CUDA graphs (bounded by the measured 1.75% host
   gap; 1.005–1.011× on the trains), any tensor-core/bf16
   reformulation of the scan (refused by construction, K21), and any
   change to the backward's in-kernel arithmetic (unbracketed — the
   next profiling lane's object, not an integration candidate).
5. **The next measurement lane, named:** the scan backward's in-kernel
   levers (a different kernel set; 83% of the composite scan pair;
   partially throughput-bound — its floor was never modeled), and the
   board-shape memory bill.

## 7. What this lane does NOT say

The composite step (4.97 s) and exposure (87.8%) rows are **projections
from measured terms**, gated on a restructure whose end-to-end
bit-identity is NOT yet receipted — they are the honest arithmetic of
what the receipts would buy, not a claim about any run machine. The
batching receipts are at the scan level on one layer's production
arguments (the kernel is layer-agnostic; the cross-layer timing spread
was 0.48% in K20); the hybrid rows assume the same layer-major schedule
applies to a hybrid's 3 slot layers. The backward's floor and in-kernel
levers are unbracketed; the elementwise glue's 1.34 s is stated
unrecovered under the bar; and nothing in this lane changes any
committed engine, any banked row, or any parity bar.

**Artifacts:** `receipts/k23_speed.json` (the full receipt: sessions,
engine hash bindings, walls, the step-profile decomposition with the
launch census, every batching bracket's per-group bit-identity report,
the backward's 7-gradient-per-group checks, the graph brackets, the
compile pieces with their honest failures, the composite, the exposure
table, the escalation check, and `driver_notes` for the two-pass run),
`k23_step_speed.py` (the lane driver), `k23_batch_brackets.py` (the
scratch file), `run_k23_step_speed_leased.sh` (the one-card leased
driver), `k23_runs/console.txt`,
`k23_runs/profiler_full_step.txt`.
