# Heterogeneous E97 Cell — Fused Kernel Throughput Note

**Task:** `hetero-kernel` · **Hardware:** NVIDIA RTX 6000 Ada (142 SM, ~960 GB/s) · torch 2.9.1+cu128, triton 3.5.1
**Scope:** make the BLENDED within-layer heterogeneous cell (chunkable bulk + a small
nonlinear-state fraction) run at ≥ 0.95× GDN‑2 tok/s at the 1.3B head shape (dim=2240,
depth=18, 64 heads, n_state=32), fwd+bwd, bf16, REAL `LadderLM` on REAL token batches. No mocks.

> **Result: TARGET MET — 0.954× GDN‑2** (mean of 4 clean trials, every trial ≥ 0.95; §3) for
> the composition **60 gdn‑neg + 4 fused nonlinear‑state shell heads** with stream overlap +
> prenorm + warp‑tuning. Per‑head fwd+bwd bf16 parity verified, no eager fallback, `phi`
> parameterized. The decisive lever is **stream overlap** (run the latency‑bound sequential
> nonlinear scan on a side CUDA stream concurrently with the tensor‑core chunkable bulk).

The cell is `TypedHeadMixtureLayer` (`ndm/models/typed_head_mixture.py`): a horizontal
population of native head types summed into the residual stream —
- **LINEAR chunkable bulk:** `gdn2_recall` (gated‑delta, `allow_neg_eigval` = the GDN‑2
  negative along‑key eigenvalue for tracking) and `e97_delta` (fused chunked split‑edit,
  `ndm/triton/e97_chunked*`), both tensor‑core matmul scans at ~GDN‑2 throughput;
- **NONLINEAR‑state fraction:** `gdn2_nonlin_shell` — native GDN‑2 delta plumbing with a
  bounded nonlinear‑in‑time state map `S_t = phi(λ S_{t-1} + Δ)` fused into a single‑launch
  **sequential** Triton scan (`ndm/triton/gdn2_nonlin_fused.py`). `phi` is parameterized and
  pluggable: `identity / tanh / relu / softplus_c / softplus` (default `tanh`).
- **Readout:** GDN `o_norm` (gated RMSNorm) + the LadderLM per‑layer SwiGLU MLP.

All head types run on fused kernels with **no eager fallback** (the E97 split‑edit heads
carry a loud `RuntimeError` guard if the bf16 fused path cannot engage; the shell scan is
always the fused Triton launch for nonlinear `phi`, and the native FLA chunk for `identity`).

---

## 1. What gates throughput: the nonlinear scan is a sequential critical path

The chunkable heads (gdn‑neg, e97_delta) are tensor‑core matmul scans and already run at
GDN‑2 throughput. The only non‑chunkable part is the bounded nonlinear‑state scan, which is
**sequential in time** (`phi` is applied per `state_chunk`; a bounded state cannot be folded
into a chunk‑parallel matmul — confirmed `tanh ⊥ chunkable`, prior `e97-wallclock-cma`).

Isolated micro‑benchmark of the scan vs FLA GDN, B=2, T=2048, K=V=32, bf16, fwd+bwd
(`experiments/hetero_kernel/micro_baseline.py`, clean GPU):

| heads H | FLA GDN ms | nonlin scan ms | scan/GDN | scan state mem |
|--------:|-----------:|---------------:|---------:|---------------:|
| 64 | 2.43 | 5.37 | 0.45 | 759 MB |
| 32 | 3.31 | 5.02 | 0.66 | 380 MB |
| 8 | 3.08 | 4.24 | 0.73 | 95 MB |
| 4 | 2.17 | 4.16 | 0.52 | 47 MB |
| 2 | 2.91 | 4.13 | 0.70 | 24 MB |

**Two structural facts drive everything:**
1. **The scan wall‑time is ~flat in head count (~4.1–5.4 ms).** It is bound by the *T=2048
   sequential dependency* (per‑step latency × T), not by occupancy or head count — more heads
   = more parallel programs, not more wall‑time. So **a small nonlinear fraction does NOT
   shrink the scan's cost**: it is a roughly fixed per‑layer latency tax.
2. The scan stores the full state trajectory `(T+1,B,H,K,V)` → memory linear in H, but at the
   small fractions that matter this is tens of MB and *not* the bottleneck (fact 1 holds at
   H=2 where memory is 24 MB).

Consequence (confirmed below): reducing the nonlinear fraction alone plateaus at ~0.88× — the
fixed sequential tax cannot be fraction‑shrunk. The levers that *can* work are (a) **hide** the
latency‑bound scan under the tensor‑core bulk via stream overlap, and (b) **shorten** the
scan's per‑step critical path.

---

## 2. Levers implemented

### (a) Stream overlap — hide the scan under the chunked bulk
The sub‑blocks of a `TypedHeadMixtureLayer` all consume the same `x` independently and are
summed, so the latency‑bound shell scan is launched on a **side CUDA stream** concurrently
with the tensor‑core chunked bulk (gdn‑neg / e97_delta) on the main stream. Because the scan
is latency/few‑SM‑bound and the bulk is compute/tensor‑core‑bound, they co‑reside on the GPU;
PyTorch autograd records the per‑op stream and replays each sub‑block's backward on the same
stream, so the **backward overlaps too**. `wait_stream`/`record_stream` give correct
cross‑stream ordering. Toggle: `overlap_streams` (default True). Exact‑parity verified (§4).

### (b) Launch‑config tuning (`num_warps`)
Micro‑sweep at the [K,V]≤64 tile (`experiments/hetero_kernel/micro_warps.py`, H=4 T=2048):

| num_warps | fwd ms | bwd ms |
|----------:|-------:|-------:|
| 1 | 1.75 | **2.43** |
| 2 | **1.26** | 2.73 |
| 4 | 1.31 | 2.93 |
| 8 | 1.30 | 3.34 |

Optimal: **fwd `num_warps=2`, bwd `num_warps=1`** (the small per‑step tile makes extra warps a
net reduction‑sync loss). Applied to the kernel.

### (c) `PRENORM` fast path — shorten the per‑step critical path
The scan L2‑normalizes q,k every step (`rsqrt(Σq²)`, `rsqrt(Σk²)`) — two reductions on the
sequential critical path in the forward, plus the normalization jacobian (two more reductions)
in the backward. With `prenorm=True` (default in `gdn2_nonlin_shell`) the q,k normalization is
done **once, in parallel over all T, in autograd‑tracked PyTorch** (`_l2norm_scale`), so the
sequential kernel consumes pre‑normalized q,k and skips the per‑step normalization; autograd
differentiates the normalization outside the kernel. Numerically identical (both use
`rsqrt(Σ²+1e-6)`); parity is in fact *tighter* than the in‑kernel path (fwd 3e‑7, grad 7e‑7;
§4) because the reduction runs in fp32 outside the bf16 scan.

---

## 3. Blended throughput vs GDN‑2 (full 1.3B cell, fwd+bwd, bf16)

**Contention-free GPU, standard fwd+bwd tok/s** (the repo's `timed_tok_s` methodology;
`experiments/hetero_kernel/final_bench.py` → `final_bench.json`, gdn_pure ceiling 12884 tok/s,
B=2 T=2048). The full optimization stack is overlap + `num_warps` (fwd2/bwd1) + `prenorm`:

| composition (of 64 heads) | overlap | tok/s | **ratio vs GDN‑2** |
|---|:---:|---:|:---:|
| 64 gdn‑neg (GDN‑2 baseline) | — | 12884 | 1.000 |
| 60 gdn‑neg + **4 nonlinear‑shell** | off | 11907 | 0.924 |
| 60 gdn‑neg + **4 nonlinear‑shell** | **on** | **12252** | **0.951 ✅ ≥0.95** |
| 56 gdn‑neg + 8 nonlinear‑shell | off | 11514 | 0.894 |
| 56 gdn‑neg + 8 nonlinear‑shell | on | 11890 | 0.923 |
| 30 gdn‑neg + 30 e97_delta + 4 shell | on | 8337 | 0.647 |

**The target is met:** the blended cell **60 gdn‑neg + 4 fused nonlinear‑state shell heads**
(the bulk chunkable + a small nonlinear fraction) with stream overlap runs at **0.951×
GDN‑2** at the 1.3B head shape. Overlap is the decisive lever (+2.7 pts over the sequential
0.924). The small fraction matters (n=4 → 0.951 vs n=8 → 0.923) exactly as the fixed‑tax
model predicts. The isolated scan's `prenorm` speedup (3.256→3.035 ms fwd+bwd at H=4, −6.8%)
shortens the critical path enough to clear the line.

**Stability (4 independent clean trials, `confirm.json`):** ratios 0.9501 / 0.9505 / 0.9553 /
0.9611 → **mean 0.9542, min 0.9501** — every trial clears 0.95. The result is robust, not a
boundary fluke.

**Two compositions that do NOT help (reported honestly):**
- **`e97_delta` as additional chunkable bulk badly hurts (0.65×).** The fused chunked
  `e97_delta` kernel is materially slower per head than FLA GDN‑2 here, so replacing gdn‑neg
  heads with e97_delta heads slows the *bulk* itself — the wider‑bulk‑hides‑the‑scan idea
  backfires. gdn‑neg is the right chunkable bulk; e97_delta should be added only where its
  capability is needed, not as throughput filler.
- A larger nonlinear fraction (n=8, 0.923×) is below target: the scan is a fixed per‑layer
  latency tax, so more nonlinear heads only add cost without shrinking it.

**Contention-robust cross-check** (interleaved per-iteration A/B, `interleaved.json`): the same
`gdn+shell4_overlap` config measures 0.92× — a conservative floor (the per-step
`cuda.synchronize` in that method penalizes the blended config's extra launches and removes
cross-iteration pipelining). The standard-methodology 0.951× is the headline; 0.92× is the
worst-case floor. Overlap beats sequential in BOTH methods (+4.8% interleaved, +2.7 pts
standard), and held up across the earlier contended runs too.

### Baseline curve (small fraction alone plateaus — confirms §1)
`experiments/hetero_kernel/baseline_seq.json` (sequential, pre‑optimization):

| nonlinear heads (of 64) | tok/s | ratio vs GDN‑2 |
|------------------------:|------:|---------------:|
| 0 (pure GDN‑2) | 11655 | 1.000 |
| 4 | 10290 | 0.883 |
| 8 | 10096 | 0.866 |
| 16 | 9943 | 0.853 |
| 32 | 9888 | 0.848 |

The curve is nearly flat in fraction (0.883 at n=4 vs 0.848 at n=32) — exactly the fixed
sequential‑tax signature of §1. Fraction alone cannot reach 0.95.

---

## 4. Parity — fwd+bwd bf16, every head type, no eager

- `tests/test_gdn2_nonlin_shell.py` (10): shell fused fwd/bwd vs torch reference for
  `tanh/relu/softplus_c`; `identity` reproduces native FLA GDN to <1e‑4; single‑fused‑launch
  profiler assertion; full‑layer grad flow. **PASS.**
- `tests/test_e97_chunked.py` (22): e97_delta chunked fwd+bwd vs reference recurrence. **PASS.**
- `tests/test_hetero_overlap.py` (3, added): stream‑overlap == sequential for
  `{gdn+shell}`, `{gdn+e97_delta+shell}`, `{gdn+e97_raw+e97_delta+shell}` blends — forward
  exact, dx/param‑grad at bf16 noise floor. **PASS.**
- `PRENORM` parity vs torch reference (`experiments/hetero_kernel/test_prenorm.py`): fwd
  ≤3e‑7, max grad ≤7e‑7 for tanh/relu/softplus_c.

---

## 5. Why it works — and what it cost

Per layer (fwd+bwd), with overlap the mixer wall‑time is `max(bulk, scan)`. The sequential
nonlinear‑scan critical path (T=2048 steps) is **~3.0–3.3 ms** (prenorm; ~3.3–3.5 ms in‑kernel
norm) and slightly exceeds the chunked GDN baseline mixer (~2.4 ms), so a small residual tax
(`scan − mixer_base`) survives overlap. That residual is what makes the result land just above
0.95 rather than at 1.0, and it is structural: the residual coupling `out = mix + MLP(norm(x+mix))`
forces the shell contribution into the MLP input, so the scan cannot overlap the (large) MLP —
overlap is bounded to the mixer's bulk window. The three levers each chip at the residual:
- **stream overlap** hides ~all of the scan that fits under the bulk window (+2.7 pts, §3);
- **prenorm** shortens the scan critical path ~3–7% (§2c, §3), the margin that crosses 0.95;
- **`num_warps` (fwd2/bwd1)** removes reduction‑sync latency from each step (§2b).

Because the scan is a *fixed per‑layer latency tax* (flat in head count, §1), the headroom
comes from keeping the nonlinear fraction small (n=4 ✅, n=8 ✗) and the bulk on the fastest
chunkable kernel (gdn‑neg; NOT e97_delta, which is slower per head and drags the bulk, §3).

**Scaling recommendation.** 0.95× is the achievable point with the nonlinear head present in
**every** layer at a 4/64 fraction. If a deployment wants more nonlinear capacity without
paying the tax in every layer, place the nonlinear‑shell heads in a *subset* of layers
(the gdn‑neg backbone already covers most capability probes; the prior capability work shows
the nonlinear‑in‑time head is needed sparsely) — this trades the fixed per‑layer tax for a
per‑subset one and lifts the blended ratio further. The committed cell supports any per‑layer
fraction (incl. 0) via `head_type_logits`, so this is a config choice, not a code change.

## 6. Verdict

**GO — target achieved.** The fused heterogeneous E97 cell (bulk chunkable gdn‑neg + a small
fused‑nonlinear‑state fraction, all on high‑performance fused kernels, no eager) runs at
**0.954× GDN‑2** (mean of 4 clean trials, every trial ≥ 0.95) at the 1.3B head shape, fwd+bwd,
bf16, with verified per‑head fwd+bwd parity and verified overlap==sequential parity. `phi` is
parameterized/pluggable. The winning composition is **60 gdn‑neg + 4 fused nonlinear‑shell
heads with stream overlap + prenorm + warp‑tuning**.

### Reproduction
- Per‑head + blended throughput: `python experiments/hetero_kernel/final_bench.py` → `final_bench.json`
- Stability confirm (4 trials): `python experiments/hetero_kernel/confirm.py` → `confirm.json`
- Contention‑robust interleaved A/B: `python experiments/hetero_kernel/interleaved.py` → `interleaved.json`
- Isolated scan vs GDN, warp sweep: `micro_baseline.py`, `micro_warps.py`
- Parity: `pytest tests/test_gdn2_nonlin_shell.py tests/test_e97_chunked.py tests/test_hetero_overlap.py`

All runs: REAL `LadderLM` 1.3B (dim=2240, depth=18, 64 heads), REAL token batches, bf16,
fused kernels with no eager fallback, RTX 6000 Ada. No mocks.
