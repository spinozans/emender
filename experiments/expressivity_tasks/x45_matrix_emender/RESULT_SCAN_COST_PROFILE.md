# X45 K21 — THE SCAN COST PROFILE: is the 7.28× inherent to the slot semantics, or incidental to a receipts-first kernel?

Result document for the K21 task (K20's direct follow-up). **Zero training
ran in this lane** — profiling + micro-benchmarks only, one card (lease 0,
RTX 6000 Ada, `receipts/k21_scan_profile.json`, run artifacts in
`k21_runs/`). The registration-style pre-state held: no directional
prediction was made, and the decomposition IS the result. The committed
engines were imported verbatim and never modified — the receipt hash-binds
every one of them (`bindings`), the working tree was clean on every engine
path, and the lane's only new code is the profiler driver
(`k21_scan_profile.py`) and the task's named **scratch** file
(`k21_micro_brackets.py`), which nothing committed imports.

**The question.** K20 receipted our composed mixer at 7.28× GDN2's per
(layer, window-group) cost (ours 23.40 ms fwd / 24.55 bwd; GDN2's FLA
kernel 2.35 / 4.25), and left open the question that decides the hybrid
program: is that ratio the *price of content-addressable search over N
slots per token*, or the price of *a kernel engineered for bit-exact
discipline, never speed*? This lane decomposed the 23.40 ms forward into
its passes, bounded the floor for the same semantics, bracketed every
obvious lever under the receipts bar (bit-identical or it does not count),
and projected the consequences into K20's units.

**Headline, as narrowly as the evidence supports.**

* **The forward is 93% one Triton kernel, and that kernel is neither
  memory-bound nor compute-bound — it is latency/layout-bound.** The mixer
  forward (23.29 ms this session; K20's 23.40, −0.5%) is: the composed scan
  kernel 21.71 ms (93.1%), the conv taps + sigmoid + gate arithmetic +
  pad/copy elementwise 1.20 ms (5.1%), the projections' GEMMs 0.21 ms
  (0.9%), the composed_scan wrapper's eager overhead 0.22 ms (1.0%). The
  scan kernel moves a *deterministic* 129.2 MB per scan and issues a
  deterministic 6.60 G fp32 ops (+2.37 G selects), so its achieved rates
  are traffic over measured time: **5.9 GB/s DRAM = 0.7% of the card's
  measured 839 GB/s peak, and 302 Gop/s fp32 = 17% of the semantics' own
  16-SM issue ceiling** (1.9% of the card-wide 15.65 T-op/s). Nothing about
  the slot semantics' *arithmetic volume* explains 22 ms — the serial
  per-token walk plus `tl.sum`'s reduction-layout machinery own the time.
* **The semantics' own floor is ~17% of the current cost: 3.74 ms.** The
  content-addressed decisions at token *t* read the whole store, so a
  bit-identical walk is sequential over tokens with only
  batch×heads = **16 independent rows on a 142-SM card**. At the card's
  measured fp32 rate, the same arithmetic costs 3.74 ms at that 16-SM
  ceiling (memory: 0.15 ms; card-wide, if tokens were parallel — which the
  semantics forbid: 0.42 ms). **The committed kernel sits 5.9× above its
  own floor.** The floor is an idealization (it assumes each resident CTA
  sustains full-SM fp32 issue), and the fp32 instrument (cuBLAS SGEMM, TF32
  off: 31.3 TFLOPs) is itself conservative — both corrections would only
  *lower* the floor and strengthen the "incidental" side.
* **Per-pass attribution (ablation brackets on a bit-identity-checked
  faithful copy), with each pass's floor:** the read sweep (the cosine-sim
  scores + the rank-axis read gather) costs **8.77 ms = 39.4%** of the
  kernel (floor 0.69 ms = 7.9% of it); the two hit-sums (the CORRECT-path
  `old_pre` + the write-path `old_row`) **3.50 ms = 15.8%** (floor ≈1.23 ms
  = 35%); the per-token value-norm fold (the MERGE strengths) **3.15 ms =
  14.2%** (floor 0.62 ms = 20%); the segment checkpoint stores **1.40 ms =
  6.3%** (memory floor 0.045 ms); the record-channel stores **0.64 ms =
  2.9%** (0.046 ms); the decay tile multiply **0.06 ms** — below its own
  0.31 ms arithmetic floor, the clean signature of a latency-bound kernel
  (its issue slots hide under other work). The un-ablated remainder
  (decision sim, threshold chain, loads, the scatter apply) is 4.72 ms
  (21%) with a ≈0.96 ms floor. Every pass is far from both its compute and
  its memory bound.
* **Measured, bit-identical, and validated: a 1.32× forward speedup
  exists** — the stack of three dataflow-identical levers (the value-norm
  chain moved inside the `if merge:` branch it alone feeds — the K19
  op-mix receipts say merges are ≤0.34% of tokens, yet the committed
  kernel folds a 16K-element tile every token to serve them; the `old_row`
  hit-sum moved inside the `else` of `if erase0` — unused on ~98% of
  tokens; and the value-norm fold's `tl.sum` machinery replaced by the
  same-pairings `tl.split` form) at launch config (8 warps, 1 stage).
  **22.13 ms → 16.79 ms, every output channel `torch.equal` against the
  committed opaque op pair at captured real activations** (layer 6's own
  production arguments, captured with a recording shim on the committed
  mixer's own `composed_scan` binding). Spread ≤0.2%, reproducible across
  three sessions. **NOT integrated into the committed engines** — it lives
  only in the scratch file.
* **The levers that fail, stated honestly:** the defined-order axis-0 fold
  substitution for the three `tl.sum` reductions **cannot preserve
  bit-exactness** (the `reads` channel flips: `tl.sum`'s full-axis
  reduction order is *not* the adjacent-pair tree) and is 3.6× *slower*
  besides; **the launch config is load-bearing for the receipts**: 4 and 16
  warps produce *different `reads`* (~1e-6 on synthetic probes) — the read
  gather's `tl.sum` order is implementation-defined and warp-layout
  dependent, the one place Triton's lowering leaks into the committed
  receipt; bf16 accumulation and tensor-core (`tl.dot`) reformulations are
  refused by construction (different rounding everywhere); the fused
  key|query_key projection is 1.87× faster on its 0.68 ms but **flips bits**
  (cuBLAS's own accumulation at the fused width) — it does not count.
* **The verdict on K20's question.** Of the scan's 22.13 ms forward:
  **24.1% is recoverable today by validated bit-identical motion** (the
  1.32× stack), **16.9% is the semantics' own floor**, and the ~59%
  in between is implementation headroom that only a kernel-rewrite lane
  could claim — under the same bar (it must reproduce `tl.sum`'s exact
  order for the read) or under a bar the operator would have to change
  (that is a different machine, and not this lane's call to make). In
  K20's units: per-mixer ratio **7.28× → 6.44×** after the recoverable
  fraction (4.47× at the forward's semantics floor, backward untouched);
  **3-of-24 hybrid exposure 73.1% → 75.4%** (81.6% floor-bound); pure-ours
  step wall **10.62 s → 9.61 s** (7.10 s floor-bound). **No lever reached
  the 1.5× candidate-engineering bar (best: 1.32×), and no ≥2× finding
  exists to escalate.**

**The answer, in one sentence:** the 7.28× is *mostly incidental* — the
committed kernel runs at 17% of the parallelism-limited arithmetic ceiling
its own semantics impose, and even at its floor the forward would cost
only ~1.6× GDN2's committed mixer forward — but a large share of that
incidental cost is *entrenched by the receipts bar itself*, which pins the
read gather to `tl.sum`'s implementation-defined order, pins the warp
count, forbids FMA contraction and every tensor-core reformulation; the
cheap dataflow-identical motions recover a quarter of it, and the rest is
a rewrite lane's to claim or a bar change the operator must sign off.

---

## 1. Method (the boundary, stated once)

Everything at the registered probe shape — batch 8 × 2048 tokens, width
576, heads 2, rank = value_dim 128, key_dim 16, segment 64 — the same unit
K20 timed. The committed machine is K20's own per-layer build verbatim
(`velocity_ladder._build_k10(TUNED_WINNER)` + `convert_model_to_composition`
+ `convert_model_to_short_conv(birth='identity')`), engines imported and
never modified, hash-bound in the receipt; the lane refuses to run on a
dirty engine tree.

* **Real activations, real arguments:** one real microbatch group of the
  flagship stream driven through the machine (capture loss 5.7849, 64
  rows, 117,712 targets — the K20 capture's own regime), with a recording
  shim on the committed mixer's `composed_scan` binding capturing layer
  6's *exact production scan arguments* (dtypes included: payload/keys
  bf16, decay/gates fp32). The shim is removed before anything is timed.
* **The receipts bar, operationalized:** the committed opaque op pair
  (`x45::k15_scan_fwd`) was first verified deterministic across runs at
  fixed inputs (it is, bit for bit). Every lever's outputs were then
  compared to the committed outputs channel by channel — reads, state,
  live, keys, slot, op, erase, live_after, payload, live_pre, ckpt_state,
  ckpt_live, ckpt_keys — with `torch.equal`. A lever counts only if every
  channel matches. Timings: CUDA-synced medians, spread stated, ≥3 warmup
  (kernel compiles included), 10 measured for the micro-brackets, 25 for
  the headline reproductions.
* **Profiling tools, honestly:** this host's hardware performance counters
  are **not accessible** (`ERR_NVGPUCTRPERM` for both Nsight Compute and
  Nsight Systems' GPU-metrics sampling, verified same-session with a plain
  sgemm probe — an environment permission, not a property of the scan). The
  per-kernel breakdown is torch-profiler CUPTI timing (device kernels only;
  op/autograd wrapper rows that mirror their child kernel's time are
  excluded to avoid double counting). The memory-vs-compute classification
  is the honest analytic substitute: the scan kernel's byte traffic and op
  counts are **deterministic** (counted from the committed Triton source,
  every op a separate `mul.rn`/`add.rn` — no FMA contraction), so achieved
  rate = traffic / measured kernel time, against the card's **measured**
  peaks (DRAM copy: 839 GB/s; SGEMM fp32, TF32 off: 31.3 TFLOPs ⇒
  15.65 T-op/s issue rate; bf16 tensor: 114 TFLOPs; 142 SMs).
* **The scratch file:** `k21_micro_brackets.py` holds a faithful copy of
  `_k15_seq_fwd_kernel` with constexpr-gated levers, plus ablation
  brackets. The faithful configuration (all flags zero) was
  bit-identity-checked against the committed op pair **first** — the
  instrument fails closed otherwise (it passed: bit-identical, and its
  22.24 ms timing matches the committed 22.13 ms within spread, so the
  attribution deltas are in the committed kernel's own regime).

## 2. Who owns the 23.40 ms (the per-kernel and per-pass decomposition)

Mixer forward, per iteration (torch profiler, 10 iterations, device
kernels only):

| component | ms | share |
|---|---|---|
| `_k15_seq_fwd_kernel` (the composed scan) | 21.71 | 93.1% |
| elementwise/copy (conv taps, sigmoid, gate arithmetic, the causal-context pad) | 1.20 | 5.1% |
| GEMM (value/key/query_key/b_proj/g_proj/f_proj/out projections) | 0.21 | 0.9% |
| composed_scan wrapper eager overhead (zeros init, dispatch, output allocation) | 0.22 | 1.0% |

The scan kernel itself (22.13 ms direct launch; 21.86 ms pure kernel CUDA
time), attributed by ablation (base = the bit-identity-checked faithful
copy at 22.24 ms), each ablation chosen so the discrete op mix is
unchanged:

| pass | measured ms | share | floor (16-SM) | floor as % of measured |
|---|---|---|---|---|
| the read sweep: cosine-sim scores + the rank-axis read gather | 8.77 | 39.4% | 0.69 ms | 7.9% |
| the hit-sums (`old_pre` on the CORRECT path + `old_row` on the write path) | 3.50 | 15.8% | ≈1.23 ms | ≈35% |
| the per-token value-norm fold (the MERGE strengths) | 3.15 | 14.2% | 0.62 ms | 20% |
| the segment checkpoint stores | 1.40 | 6.3% | 0.045 ms (mem) | 3.2% |
| the record-channel stores (schedule lanes, live_pre, payload lanes) | 0.64 | 2.9% | 0.046 ms (mem) | 7.2% |
| the decay tile multiply | 0.06 | 0.3% | 0.31 ms | (below its own floor — see below) |
| un-ablated remainder (decision sim, threshold chain, input loads, scatter apply) | 4.72 | 21.2% | ≈0.96 ms | ≈20% |

The decay row is the tell: its *marginal* cost (0.06 ms) sits below its own
*arithmetic* floor (0.31 ms) — in a throughput-bound kernel that is
impossible; in a latency-bound one it is expected (the multiply's issue
slots hide under the wait for the serial chain). The op-count model and
the ablations agree on the shape: **the read sweep is the dominant pass
(39%), and its cost is 12.7× its arithmetic floor.**

## 3. The floor for the same semantics

The semantics' parallelism ceiling at the registered shape: the decisions
at token *t* address the *whole* store (read scores, decision similarity,
thresholds — all over all 128 slots), so the schedule at *t* depends on
the state after *t−1*; a bit-identical walk is **sequential over tokens**,
leaving batch×heads = **16 independent rows on a 142-SM card**. (This is
exactly the property GDN2's chunked delta-rule kernel does *not* have —
its chunked matmul formulation is a different semantics, not a faster
implementation of this one; making the same trade here changes the
machine, and the receipts bar forbids it.)

| bound | ms | as % of the committed 22.13 ms |
|---|---|---|
| compute at the card-wide measured fp32 peak (token-parallel — FORBIDDEN by the semantics; stated only to bound the forbidden direction) | 0.42 | 1.9% |
| **compute at the semantics' 16-SM ceiling — THE floor** | **3.74** | **16.9%** |
| memory at the measured 839 GB/s (129.2 MB deterministic traffic) | 0.15 | 0.7% |

Per the task's phrasing, pass by pass: the read sweep costs 39.4% of the
kernel today and its floor is 7.9% of that cost; the hit-sums cost 15.8%
and their floor is ~35% of that; the value-norm fold costs 14.2% and its
floor is 20% of that; the I/O passes cost 9.2% combined and their floors
are ~4% of that. The whole-kernel floor ratio is **0.169** — the direct
answer to the inherent-vs-incidental question: **~17% of the scan's
forward cost is the semantics' own; ~83% is implementation.**

Caveats, stated not hidden: the 16-SM ceiling idealizes full-SM fp32 issue
per resident CTA; the fp32 instrument is cuBLAS SGEMM (31.3 TFLOPs ≈ 34%
of the card's theoretical 91 — a hand-tuned FFMA loop could raise it,
which would *lower* the floor); the floors exclude the reduction-layout
machinery by definition (that machinery is precisely the implementation
headroom being bounded).

## 4. The micro-brackets (bit-identical or it does not count)

Every lever at the registered shape on the captured real arguments; the
receipts bar is `torch.equal` on all 13 output channels:

| lever | bit-identical | median ms | speedup |
|---|---|---|---|
| launch-config sweep of the COMMITTED kernel: (8,1) | **yes** | 21.22 | 1.04× |
| — (8,5) | yes | 22.18 | 1.00× |
| — (4,3) | **NO — the `reads` channel flips (~1e-6)** | 32.42 | 0.68× |
| — (16,3) | **NO — the `reads` channel flips** | 25.87 | 0.86× |
| `reduce_nr`: the value-norm fold's `tl.sum` machinery → the same-pairings `tl.split` form | yes | 20.22 | 1.10× |
| `lazy_norms`: the value-norm/strength chain into the `if merge:` branch it alone feeds | yes | 19.29 | 1.15× |
| `lazy_old_row`: the `old_row` hit-sum into the `else` of `if erase0` | yes | 21.19 | 1.05× |
| **stacked (the three above), at (8,3)** | **yes** | 17.53 | 1.26× |
| **stacked, at (8,1) — the lane's best** | **yes** | **16.79** | **1.32×** |
| `reduce_sum`: the three axis-0 `tl.sum`s → the defined-order fold | **NO — `reads` flips** (and 3.6× slower besides) | 80.33 | 0.28× |
| fp32 pre-cast of the scan inputs (vs the committed bf16 inputs) | yes (exact upcast, as expected) | 22.59 | 0.98× |
| fused key\|query_key projections (0.68 ms of eager GEMM) | **NO** — cuBLAS differs at the fused width | 0.37 | (1.87×, does not count) |
| fused value\|b_proj projections (0.19 ms) | yes | 0.13 | (1.42× on 0.19 ms — negligible) |

Honest negatives, per the task: **bf16 accumulation** and any
**tensor-core (`tl.dot`) read gather** change the rounding of every folded
sum — refused by construction; **incremental/precomputed key norms** —
only one slot's key changes per token so the recomputation is ~2% of the
per-token ops (bounded below relevance) and the *arithmetic* incremental
update is not the defined recompute; **batched cosine-sim over slots** —
not applicable, the committed kernel already sweeps all 128 slots per
token in one register tile (there is no per-slot loop to batch away).

Two receipts-discipline observations worth the operator's attention:

1. **The read gather's `tl.sum` order is part of the committed
   bit-exactness contract.** It is the only channel whose reduction order
   is *implementation-defined* (every other fold is a defined-order tree),
   it changes with the warp count (4 and 16 warps produce different
   `reads`; the committed 8-warp config is load-bearing), and it cannot be
   replaced by the defined-order fold without flipping bits. Any future
   kernel lane that touches the read must reproduce Triton's `tl.sum`
   lowering bit for bit — or re-register the reads channel's parity bar.
2. **The committed kernel computes a per-token two-pass 16K-element value
   norm whose sole consumer is the MERGE blend — a ≤0.34%-of-tokens event
   at the registered op mix** (the K19 receipts' own number). That is the
   single clearest "engineered receipts-first, never speed" artifact this
   lane found, and its removal is the largest validated lever.

## 5. The verdict, in K20's own units

Anchor (K20, same shape): per-mixer ratio **7.28×** (fwd+bwd 47.98 vs
6.59 ms), pure-ours step **10.62 s**, pure-GDN2 step **3.12 s**, 3-of-24
hybrid (mixer swap, banked Muon) step 4.27 s ⇒ **73.1% of GDN2's matched-time
exposure**.

| scenario | scan fwd | mixer fwd | per-mixer ratio | 3-of-24 hybrid step | 3-of-24 exposure | pure-ours step |
|---|---|---|---|---|---|---|
| K20 anchor (committed) | 22.13 ms | 23.29 ms | 7.28× | 4.27 s | 73.1% | 10.62 s |
| **after the measured bit-identical recovery (1.32×)** | 16.79 ms | 17.99 ms | **6.44×** | **4.14 s** | **75.4%** | **9.61 s** |
| at the forward's semantics floor (backward untouched) | 3.74 ms | 4.95 ms | 4.47× | 3.82 s | 81.6% | 7.10 s |

Readings, narrowly:

* **Recoverable bit-identically, measured: 24.1% of the scan forward**
  (1.32×). Applied to the forward term only — the backward's levers were
  not bracketed (its cost body is a different kernel set: segment replay
  plus two walks), and the receipt says so — the 3-of-24 hybrid concession
  shrinks from 26.9% to 24.6% of GDN2's matched-time exposure, and the
  pure machine's step wall from 10.62 s to 9.61 s.
* **The semantics' floor: 16.9% of the current forward.** Even with the
  forward at its floor and the backward untouched, the per-mixer ratio
  stays 4.47× — at that point the *backward* (24.55 ms, unbracketed)
  owns the ratio. Against the committed forward, the kernel sits 5.9×
  above its own floor; against GDN2's committed mixer *forward*, the
  floor itself is 3.74 vs 2.35 ms ≈ 1.6× — the slot semantics at this
  shape are *not* inherently ~7×; that number is dominated by
  implementation.
* **No candidate engineering follow-up is claimed:** the best validated
  speedup is 1.32× < the 1.5× bar the task set for naming one. The 1.32×
  stack lives in the scratch file with its bit-identity receipts, ready
  for a sign-off lane if the operator wants it, integrated nowhere.
* **No escalation finding:** no bit-identical speedup ≥2× exists; the
  committed scan was profiled as-committed throughout.

## 6. What this lane does NOT say

The floors bound the *forward* at this shape under an idealized 16-SM
issue ceiling; they are not a promise that a rewrite reaches them — the
read gather's `tl.sum` order is pinned by the receipts bar, and most of the
headroom lives in the reduction-layout machinery and the serial chain that
bar constrains. The backward (24.55 ms, ~52% of the per-mixer pair) was
not decomposed; if the hybrid program's arithmetic matters, the backward
is the next lane's object, and at the forward's floor it becomes the
dominant term. Nothing in this lane changes any committed engine, any
banked row, or any parity bar — the 1.32× stack is a receipted candidate,
and the decision to spend a lane on it (or on a bar change for the reads
channel) belongs to the operator, with these receipts.

**Artifacts:** `receipts/k21_scan_profile.json` (the full receipt:
session, engine hash bindings, profiler tables, peaks, the op-count model,
ablation brackets, every lever's bit-identity channel report, the
verdict), `k21_scan_profile.py` (the lane driver), `k21_micro_brackets.py`
(the scratch file), `k21_runs/console.txt`,
`k21_runs/profiler_mixer_forward.txt`, `k21_runs/profiler_scan_only.txt`.
