# Scale proof — is the fused matrix emender fast at all three scales?

The K4 receipt flagged rank 512 as trajectory-bandwidth-bound and "still 2.2x over
eager", and the operator read that hedge as "so it is NOT fast at the scales that
matter."  This document settles it with receipts at three levels of the system —
the block call, the model training step, and whole-arm training throughput — each
with its measurement boundary stated, every comparison **same-session** unless
explicitly labelled RECORDED, and no number massaged.  The verdict up front:

* **Scale 1, the block call: FAST at ranks 64 and 128, faster-but-narrowing at
  512.**  9.4–15.4x over the same-session eager path at rank 64, 4.4–10.3x at
  rank 128, and 1.1–2.7x at rank 512, where the fused path is device-bound at
  wall/device ≈ 1.00 at every shape and the margin narrows because the eager
  baseline is launch-bound (it barely moves with shape) while the fused path
  honestly scales with the state it must store and re-read.  The rank-512
  trajectory (8–16 GiB) is the cause; recompute-over-store (K4's unbuilt option
  2) is the lever.
* **Scale 2, the model training step: NOT fast as wired — memory-bound.**  The
  fused path **cannot run at the registered microbatch at all**: its per-chunk
  trajectory store is 2.15 GB per layer per microbatch at the registered shape,
  24 layers sum to 51.5 GB, and the 48 GB card refuses (CUDA OOM at 46.8 GB
  allocated, receipted at microbatch 8 AND 4).  At microbatch 2 — the largest
  that fits — the fused step is 132.8 s vs the registered twin's 50.0 s
  (**2.65x slower by matched wall**), because the fused autograd node cannot be hosted by
  `torch.compile` (dynamo/inductor refuse it, error strings receipted in §2.2),
  so the routing host loop runs uncompiled: 6.35 million kernel launches per
  step, 89 % host-bound.
* **Scale 3, training throughput: does NOT reach GDN2's class — report the
  number.**  On the gate-D protocol (chunk 128, 64 flagship windows/step): the
  same-session registered arm re-measures **12,326 tok/s** (RECORDED matrix_ops
  13,190 validated to within 6.5 %, and the profile shows the *same*
  CompiledFxGraph hashes as the cd receipt); the fused arm **OOMs at the
  registered microbatch** and at the largest fitting microbatch delivers
  **1,773 tok/s** (64.25 s/step) against its registered twin's 3,751
  (26.16 s/step) — 2.46x slower by matched wall, and 16x
  below the RECORDED GDN2 28,124.  The fused runs reproduce the registered arm's
  loss trajectory on the same windows to three decimals (model-level parity
  holds); what fails is not the kernel, it is the store-and-dispatch structure
  the kernel lives in.

Everything below was measured on one NVIDIA RTX 6000 Ada (48 GB), fp32 at the
block boundary, the registered bf16-compute regime at the model boundary, torch
2.9.1+cu128, Triton 3.5.1.  Receipts: `receipts/scale_proof_blocks.json` (scale 1),
`receipts/scale_proof_model_step.json` (scale 2),
`receipts/scale_proof_training.json` (scale 3).  Bench:
`bench_scale_proof.py` (`--scale {1,2,3}`).  All committed incrementally
(d69df834 → 42f4b7bd).

---

## 1. Scale 1 — the block call, full rank sweep

**Boundary (the K4 boundary, unchanged):** one block call, forward and backward —
state and liveness in, the recorded schedule and read weights as inputs, the
tokens' reads and the exit state out, and (for the step columns) the four
gradients back.  Routing is outside, as in `K4_FUSED_BACKWARD.md` §5.  fp32, 2
heads, d_v 128, lanes 2, 5 reps, p50s in the receipt.  The eager baselines were
**re-measured in this session** — the K4 receipts are never used as a comparison.
`launches` is the profiler's device-kernel count per call.

| shape (rank_b_T) | traj GiB | fused step: wall / device / w÷d / launches | fused fwd wall | eager C128 step: wall / launches | eager C64 step: wall / launches | fused ÷ eager C128 | fused ÷ eager C64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r64_b8_T2048 | 1.0 | **3.96 / 3.98 / 1.00** / 6 | 1.95 ms | 52.6 ms / 3 435 | 102.1 ms / 6 139 | **13.3x** | **25.8x** |
| r64_b16_T2048 | 2.0 | **5.75 / 5.77 / 1.00** / 6 | 2.85 ms | 53.9 ms / 3 435 | 104.4 ms / 6 363 | **9.4x** | **18.2x** |
| r64_b8_T4096 | 2.0 | **7.76 / 7.79 / 1.00** / 6 | 3.63 ms | 119.6 ms / 6 875 | 235.7 ms / 12 283 | **15.4x** | **30.4x** |
| r128_b8_T2048 | 2.0 | **5.99 / 6.01 / 1.00** / 6 | 3.10 ms | 61.3 ms / 3 435 | 117.1 ms / 6 139 | **10.2x** | **19.5x** |
| r128_b16_T2048 | 4.0 | **13.90 / 13.92 / 1.00** / 6 | 8.22 ms | 60.9 ms / 3 435 | 120.4 ms / 6 363 | **4.4x** | **8.7x** |
| r128_b8_T4096 | 4.0 | **11.71 / 11.74 / 1.00** / 6 | 5.81 ms | 120.6 ms / 6 875 | 235.7 ms / 12 283 | **10.3x** | **20.1x** |
| r512_b8_T2048 | 8.0 | **24.19 / 24.21 / 1.00** / 5 | 12.97 ms | 61.3 ms / 3 434 | 117.3 ms / 6 138 | **2.5x** | **4.9x** |
| r512_b16_T2048 | 16.0 | **56.94 / 56.97 / 1.00** / 5 | 34.45 ms | 61.4 ms / 3 434 | 120.6 ms / 6 363 | **1.1x** | **2.1x** |
| r512_b8_T4096 | 16.0 | **49.26 / 49.29 / 1.00** / 5 | 27.05 ms | 135.4 ms / 6 875 | 237.8 ms / 12 282 | **2.7x** | **4.8x** |

The fused step's 5–6 device kernels are the K4 walk + 2 reductions (fwd 1 launch,
bwd 3) plus autograd glue (copies of the incoming gradient layouts); the eager
path's thousands are its per-chunk pipeline.  Host dispatch: fused 0.64–1.04 ms
per step against the eager path's 52–236 ms — the fused block call is
**device-bound at every shape** (wall/device = 1.00, occupancy within 1.03–1.16x
of wall, in the receipt).

**Verdict, scale 1: fast.**  At the registered ranks (64, 128) the fused block call
is 4.4–30x faster than the same-session eager path at both chunk lengths, at
every batch/length.  At rank 512 it is still faster everywhere (1.1–4.9x) but the
margin collapses at large batch, and one cell — rank 512 at batch 16 — is within
10 % of the eager wall.  The cause is the honest one K4 named: the trajectory
store is 16 GiB there and the backward's 22 ms is bandwidth-bound reading it; the
eager baseline, being launch-bound, never grows with the state.  Levers, in the
order K4 priced them: recompute-over-store (option 2: swap 16 GiB of store for a
second walk), or the WY-style adjoint (option 3, GDN-2's trick, not available to
a selection-based write).

### 1.1 The same-session GDN-2 block baseline (a different boundary, stated)

**Boundary (`bench_blocks`', unchanged):** ONE whole mixer block — projections +
scan + mixer arithmetic, forward + backward of `out.pow(2).sum()` — eager, no
`torch.compile`, no CUDA graph, fp32.  This is **not** the scan-block boundary
above and not comparable to it; it exists because it is the same-session form of
the gate-D block-vs-block receipt, with GDN-2's autotune pin recorded
(`NDM_PIN_TRITON_AUTOTUNE=0`; the pinned config OOMs shared memory at the
rank-128 shape).

| b × T | rank | matrix block (eager) | matrix block (fused) | GDN-2 block |
| --- | --- | --- | --- | --- |
| 8 × 2048 | 64 | 124.2 ms (device 21.1) | **77.4 ms** (device 12.3) | 8.48 ms (device 3.92) |
| 8 × 2048 | 128 | 124.9 ms (device 19.9) | **76.0 ms** (device 15.8) | 7.61 ms (device 5.61) |
| 8 × 2048 | 512 | 122.9 ms (device 24.1) | **77.5 ms** (device 34.5) | **cannot** (head_dim > 256) |
| 16 × 2048 | 64 | 122.7 ms | **78.6 ms** (device 19.3) | 8.65 ms |
| 16 × 2048 | 128 | 106.4 ms (device 26.3) | **65.0 ms** (device 25.2) | 11.97 ms |
| 16 × 2048 | 512 | 106.7 ms | **73.0 ms** (device 70.3) | **cannot** |
| 8 × 4096 | 64 | 212.7 ms | **131.4 ms** (device 29.8) | 9.03 ms |
| 8 × 4096 | 128 | 249.1 ms (device 47.3) | **130.3 ms** (device 35.5) | 12.13 ms |
| 8 × 4096 | 512 | 212.1 ms | **131.9 ms** (device 72.3) | **cannot** |

Plainly: the fused backward cuts the whole-block wall 1.5–1.9x, and the gap to
GDN-2's block narrows at every shape — e.g. at the gate-D receipt's own shape
(b8 × 2048, rank 128) the matrix block stands 16.4x behind GDN-2 eager and
**10.0x fused** (same-session; the recorded gate-D ratio was 13.2x, with the
same relation) — but the whole matrix block remains host-bound at every rank
(its wall barely moves with rank or batch because the routing host loop
dominates), and **GDN-2's kernel cannot express the rank-512 store at all** —
`GDN-2 kernels require head_dim <= 256` — while the fused matrix path runs it.
The rank-64 GDN-2 row runs at expansion 2.0 so its state is 64 × 128 per head,
the axis the registration calls capacity.

---

## 2. Scale 2 — the model training step, fused ON vs OFF

**Boundary (the gates/cd protocol):** the whole model at its registered shape —
width 576, depth 24, chunk 64, ~150 M parameters (149,981,808), embedding + head
+ clip + fused AdamW included — one step = 64 flagship-stream windows (131,072
bytes), microbatched, bf16 compute, fp32 master weights, ≥5 steady steps after
warmup, per-step wall from the gates phase protocol (prepare / fwd_bwd / opt).
Same card, same session for every row.

**What was wired (explicit opt-in, default unchanged):** `MatrixMixer(...,
fused_backward=True)` → `routed_chunk_scan(..., trainable_fused=True)` → one
`trainable_block_scan(fused=True)` autograd node per chunk (the K4 trajectory
forward + reverse walk) in place of autograd through the eager chunk scan.  The
routing loop, the recorded schedule and every gradient path are unchanged — the
read weights and payloads stay attached exactly as the registered path attaches
them — verified three ways before any timing: every one of the 243 parameters
carries a finite non-zero gradient on the fused path; whole-window state is
bit-identical to the eager path (so every routing decision is identical, liveness
equal); and at the model level in fp32 the fused and eager gradients agree to
6.6e-7 relative max (the 1 % differences under bf16 autocast are rounding-quantum
flips, median 0 in most tensors).  The default in `model.py` remains
`fused_backward=False`; the registered arm is untouched.

| config (chunk 64) | steps | wall/step p50 | tok/s steady | peak GB | verdict |
| --- | --- | --- | --- | --- | --- |
| **OFF, registered** (torch.compile, mb 8) | 8 | **13.17 s** | **9,306** | 24.3 | the baseline |
| ON, fused, mb 8 | 0 | — | — | 50.25 | **CUDA OOM** at 46.8 GB allocated |
| OFF, registered, mb 4 | 8 | 24.83 s | 4,899 | 13.5 | memory-matched twin probe |
| ON, fused, mb 4 | 0 | — | — | 50.17 | **CUDA OOM** at 46.7 GB |
| OFF, registered, mb 2 | 3 | 50.05 s | **2,111** | 8.1 | the largest fitting microbatch |
| **ON, fused, mb 2** | 6 | **132.83 s** | **881** | 33.0 | **2.65x slower than its twin by wall** |

Both fused runs pass the gates that matter (grad probe all-finite all-nonzero, no
replay/checkpoint kernels in the profile).  The tok/s fields of the mb2 rows sit
over different numbers of stream windows (counted tokens vary per window — a
padded window costs a full 2048-byte pass but counts only its valid targets —
so the matched comparison is the per-step wall: 132.83 vs 50.05 s, **2.65x**;
the receipt's per-step token counts let any other normalisation be recomputed).
The two failures are structural, not implementation noise:

### 2.1 The OOM: the trajectory store sums across layers

The K4 trajectory is `batch × heads × T × rank × d_v` fp32 = **2.15 GB per layer
per microbatch** at the registered shape.  A training forward holds all 24
layers' saved tensors until the backward frees them layer by layer, so the peak
is the **sum**: 51.5 GB at microbatch 8 — past the 48 GB card before the MLP
activations are even counted, and the allocator refuses at 46.8 GB (microbatch
4 sums to 25.8 GB but dies at 46.7 GB with the rest of the step resident; only
microbatch 2 fits, at 33.0 GB peak).  This is exactly the cost K4 §3 priced at
the block level ("2.00 GiB, 31x the block's every other tensor") — at one layer
it was 4.2 % of the card; at 24 layers it is the card.  K4 §1.1 named the lever
for precisely this objection: **option 2, chunk-boundary checkpoints +
recompute** ("stays unbuilt until a measurement asks for it" — this is the
measurement).  Recompute would trade ~17 % more block time (a second walk) for a
~4000x smaller per-chunk residency and would let the fused path run at the
registered microbatch; it is a new backward wrapper, not a kernel change, and it
was not built here.

### 2.2 The 2.65x loss at the fitting microbatch: the fused node cannot be compiled

`torch.compile` cannot host the K4 autograd node.  Two independent failures,
both reproduced on a two-layer probe: compiling `routed_chunk_scan` with the
fused calls inside dies with
`InductorError: LoweringException: TypeError: torch.bool is not supported by
torch.iinfo`, and marking the node `allow_in_graph` dies with dynamo's
"Cannot access data pointer of Tensor (FakeTensor) ... wrap the custom kernel
into an opaque custom op".  The registered arm's speed comes from
`torch.compile` over the whole chunk loop (its profile is two CompiledFxGraph
regions at 62 % of device time); the fused path must therefore run the routing
**uncompiled**, and the routing is a host loop: 24 layers × 32 chunks × 32
microbatches = 24,576 chunk-iterations per step, **6.35 million kernel launches
per step**, 89 % host-bound (device busy 14.9 s of a 132.8 s step).  The lever
is the one the error message names: wrap the fused node as an opaque
`torch.library` custom op so dynamo can compile the routing around it (and/or
capture the routing in a CUDA graph) — new machinery, deliberately not built
here, because the task was to measure, not to add it.

**Verdict, scale 2: the fused path is NOT fast at the model step as wired.**  It
is memory-bound first (it cannot run at the registered microbatch) and
host-bound second (where it can run).  Both causes are receipted with their
levers; neither is the kernel itself, whose block-level speed is §1.

---

## 3. Scale 3 — training throughput on the gate-D harness

**Boundary (the cd_* protocol):** whole-arm training on the flagship stream —
64 windows/step, chunk 128 (the cd receipts' own chunk), microbatch 8, bf16
compute, fused CE, clip + AdamW, per-step walls on the gates phase protocol,
steady-state tok/s over a 278–457 s matched window (the short-run form of the
2,700 s arm, per the task).  `registered_mb8` is the registered arm re-measured
**in this session**; the cd numbers it is checked against are labelled
**RECORDED**.

| run (chunk 128) | steps | steady window | tok/s | tok/s steady | wall/step p50 | peak GB | comparison |
| --- | --- | --- | --- | --- | --- | --- | --- |
| registered, mb 8 | 30 | 278 s | 12,326 | 12,398 | 9.27 s | 25.1 | same-session; RECORDED matrix_ops 13,190 → validated −6.5 % |
| fused, mb 8 | 0 | — | — | — | — | 50.3 | **CUDA OOM** (46.8 GB), receipted |
| registered, mb 2 | 12 | 379 s | 3,751 | 3,945 | 26.16 s | 8.3 | same-session twin |
| **fused, mb 2** | 7 | 457 s | **1,773** | **1,856** | 64.25 s | 32.8 | same-session; **2.46x slower than its twin by wall** |
| RECORDED gdn2 (cd_gdn2_c128_rerun) | 3 | — | 28,124 | — | — | 4.5 | **RECORDED, not same-session** — the class target |
| RECORDED matrix_ops (cd_matrix_ops_c128) | 3 | — | 13,190 | — | — | 25.1 | RECORDED |

The same-session anchor matters: the registered arm re-measures within 6.5 % of
its RECORDED number, and its top kernels are the *same* CompiledFxGraph hashes
as the cd receipt's (`f3u46ozu…`, `fcfasltfa…`) — this is the same computation on
the same protocol, so the fused numbers below it are not a harness artifact.
Window richness moves tok/s run-to-run (a padded window costs a full pass but
counts only its valid targets — the fused mb2 arm's 7-step window averaged
115.8k counted tokens/step against the twin's 118.5k over 12, so the tok/s
fields are close to matched here; the wall ratio 64.25/26.16 s is the exact
one).  No replay/checkpoint kernel appears in any profile (the `replay_absent`
gate holds on every run), every fused run passes the grad probe, and the fused
runs **reproduce the registered arm's loss trajectory on the same windows to
three decimals** (4.272 / 3.987 / 3.295 / 3.469 both) — the training semantics
are the registered ones.

Attribution of the fused step (profiled steady step, `--profile-mode cuda`):
device busy is 10.0 s of the 64.3 s step (16 %); the fused kernels themselves are
`_persistent_bwd_kernel` 1.74 s + `_persistent_scan_trajectory_kernel` 1.14 s
over 12,288 calls each — **the whole fused scan is ~2.9 s of the 64.3 s step** —
against 3.19 million launches of eager routing and autograd glue.  The
registered twin at the same microbatch is also host-bound (23 % device) but
launches 4x fewer kernels (0.84 M/step) because its routing is compiled.

**Verdict, scale 3: fused matrix_ops training throughput does NOT reach GDN2's
class.**  The number is 1,773 tok/s (1,856 steady) at the only microbatch whose
trajectory store fits the card — 16x below the RECORDED GDN2 28,124 and 2.46x
below its own registered twin measured in the same session (by wall; 2.1x by
the tok/s fields, whose window counts differ).  The registered arm
itself (12,326 same-session) remains 2.3x below GDN2's class for its own,
pre-existing reason (the routing host loop + eager chunk arithmetic), which the
fused backward was supposed to attack; the measurement says the attack cannot
be delivered by the K4 store-and-walk form at the model scale, for the two
receipted causes of §2: the store does not fit (microbatch ceiling) and the walk
cannot be compiled (host loop exposed).  The levers, both outside this task's
scope: recompute-over-store for the memory, an opaque-custom-op wrapper (or a
routing CUDA graph) for the host loop.

---

## 4. What is honest, what is not done

1. **No kernel was tuned or changed.**  Every fused number above is the K4 kernel
   as gated; the only new code is the wiring (`scan.py`'s
   `trainable_fused`, `model.py`'s `fused_backward`, both default-off) and the
   bench (`bench_scale_proof.py`).
2. **The rank-512 hedge is now a measured curve, not a flag.**  At the block
   boundary the fused path is faster than the eager path at every one of the
   nine shapes, but the ratio at rank 512 spans 1.1x (b16) to 4.9x (C64), and
   the b16/T2048 cell is the honest boundary: 16 GiB of trajectory store, both
   paths within 10 % of each other, cause and lever stated.
3. **The scale-2/3 negatives are structural, receipted, and levered.**  They are
   not "the fused kernel is slow" — the block tables and the 2.9 s/step of fused
   device work say otherwise.  They are: the trajectory store summed over 24
   layers (51.5 GB > 48 GB card at the registered microbatch) and the
   uncompiled routing the fused node forces (6.35 M launches/step at chunk 64,
   3.19 M at chunk 128).  K4 §1.1 pre-registered the first lever (recompute over
   store) as the response to exactly this memory objection; the second lever is
   the custom-op wrapper the dynamo error names.  Neither was built, because
   building either is a scope decision this measurement was not authorised to
   make — they are the next stage's, with the numbers to justify them now on
   file.
4. **Not measured:** a fused run at chunk 128 microbatch 4 — its trajectory
   sums to the same 25.8 GB (the store is chunk-length independent), and at
   chunk 64 the run died ~100 MB short of the card (46.7 GB allocated, a
   54 MiB request refused with 37 MiB free); chunk 128 halves the routing but
   not the binding term, and the configuration was not re-tested.  Also not
   measured: any rank other than 128 at the model scale (the registered shape
   is rank 128).  The first `fused_mb2` attempt stopped at 4 steps / 272 s
   (271.7 s of step walls + ~30 s of CUDA-profiler flush against a 300 s
   `--min-seconds` budget); the reported row is the re-run at 7 steps / 457 s,
   and the 4-step row is preserved in the git history (commit 42f4b7bd) — its
   lower tok/s (1,548) is window richness, not a code change: its four steps
   averaged 105.1k counted tokens against the re-run's 115.8k.
5. **One operational note, receipted in the commit history:** the
   microbatch-2 runs produce millions of profiler events per step, and
   `key_averages` post-processing over them did not finish in a session (the
   first `off_mb2` attempt appeared hung for this reason).  The bench therefore
   has `--profile-mode cuda`, which profiles one steady step with CUDA activity
   only and aggregates in a single pass; the mb2 step walls are the clean
   unprofiled ones.

## 5. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
# scale 1: the block call + whole-block baselines
.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.bench_scale_proof \
    --scale 1 --repeats 5 \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/scale_proof_blocks.json
# scale 2: one process per config (each writes into the same receipt)
for cfg in off on_mb8 off_mb4 on_mb4 off_mb2 on_mb2; do
  .venv/bin/python -m ...bench_scale_proof --scale 2 --config $cfg --warmup 2 --steps 8 \
      [--profile-mode cuda] --out .../receipts/scale_proof_model_step.json
done
# scale 3: the cd protocol, one process per config
for cfg in registered_mb8 fused_mb8 registered_mb2 fused_mb2; do
  .venv/bin/python -m ...bench_scale_proof --scale 3 --config $cfg --warmup 2 \
      --steps 30 [--min-seconds 300 --profile-mode cuda] \
      --out .../receipts/scale_proof_training.json
done
```

## 6. Recommended next step

Decide the two levers on this evidence, in the order the receipts price them:
(1) the **recompute-over-store backward** (K4 option 2) — ~17 % more block time
for a 24-layer-fitting memory profile, which alone would let the fused model
step run at the registered microbatch; and (2) the
**opaque custom-op wrapper** so `torch.compile` can host the fused node and the
routing stops paying 3–6 M launches per step.  With (1) alone the fused path
should be re-measured at microbatch 8; only with both does the GDN2-class
question (§3) deserve re-asking.
