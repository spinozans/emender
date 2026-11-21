"""K21 -- THE SCAN COST PROFILE: K20's direct follow-up (zero training,
profiling + micro-benchmarks only, one card).

THE QUESTION (K20 left it open, and it decides the hybrid program): our
composed mixer costs 7.28x GDN2's per (layer, window-group) at the
registered shape (ours fwd 23.40 ms / bwd 24.55 ms; GDN2's FLA kernel
2.35/4.25 -- ``receipts/k20_hybrid_velocity.json``), and the step walls are
10.62 s vs 3.12 s.  Is the 7.28x INHERENT to the slot semantics
(content-addressable search over N slots per token is fundamentally more
work than a chunked delta-rule update) or INCIDENTAL to a kernel built
receipts-first (the composed_scan opaque op pair was engineered for
bit-exact discipline, never speed)?

THE DECOMPOSITION (this lane's four parts, all at the registered probe
shape -- batch 8 x 2048 tokens, width 576, heads 2, rank 128, value_dim 128,
key_dim 16, segment 64 -- on captured REAL activations, the committed
engines imported verbatim and never modified, hash-bound in the receipt):

1. PROFILE the committed composed scan's 23.40 ms forward: per-kernel
   breakdown via torch profiler (projections / conv / the scan kernel /
   eager overhead), and per-PASS breakdown inside the one opaque kernel via
   ablation brackets (the scratch copy with one pass deleted per bracket --
   each ablation chosen so the discrete op mix is unchanged).  Memory-bound
   vs compute-bound classification per pass: the card's hardware counters
   are NOT accessible on this host (ERR_NVGPUCTRPERM, receipted), so the
   classification is the honest analytic one -- the kernel's byte traffic
   and op counts are DETERMINISTIC (counted from the committed Triton
   source), so achieved bandwidth / achieved op rate = traffic / measured
   time, against the card's MEASURED peak bandwidth and fp32 rate.

2. BOUND THE FLOOR: for each dominant pass, the theoretical minimum cost
   for the SAME arithmetic at the card's measured peaks -- stated per pass
   as "pass X costs Y ms today, its floor is Z% of that".  Two ceilings
   stated: the card-wide peak (as if the token axis were parallel, which
   the semantics forbid) and the SEMANTICS' OWN parallelism ceiling -- the
   content-addressed decisions at token t depend on the whole store, so a
   bit-identical walk is sequential over tokens with only
   batch*heads = 16 independent rows: 16 of the card's SMs.

3. MICRO-BRACKET the obvious speedup levers WITHOUT changing semantics
   (the scratch file ``k21_micro_brackets.py``; the committed engine
   untouched): launch-config sweep of the committed kernel, defined-order
   fold forms, lazy evaluation of passes whose sole consumer is a <1%
   event, fp32-vs-bf16 input regimes, fused projections.  THE RECEIPTS BAR:
   a lever counts only if it reproduces the committed scan's outputs
   BIT-IDENTICALLY (torch.equal on every output channel of the committed
   opaque op pair); anything else is reported as "cannot preserve
   bit-exactness", never shipped.

4. VERDICT: what fraction of the 7.28x is recoverable bit-identically
   (measured), what fraction is the semantics' floor (analytic), the
   projected hybrid exposure at 3-of-24 (K20's 73%) AFTER the recoverable
   fraction, and the same for the pure machine's step wall (10.62 s).  A
   bit-identical speedup >= 1.5x is stated as a candidate engineering
   follow-up for its own lane + sign-off; nothing is integrated into the
   committed engines in this lane.

PRE-STATE: no directional prediction; the decomposition IS the result.

Usage (one leased card):

    eval "$(GPU_LEASE_VISIBLE=0 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k21_scan_profile \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k21_scan_profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# The probe subprocess discipline (configure_production fail-closes on a
# missing allocator knob if these are not set before CUDA initialization).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
    configure_production)
from experiments.expressivity_tasks.x45_matrix_emender import (
    k20_hybrid_velocity as K20)
from experiments.expressivity_tasks.x45_matrix_emender import (
    k21_micro_brackets as MB)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
    import DEFAULT_KEY_DIM
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
    import (_k15_forward_launch, composed_scan,
            convert_model_to_composition)
from experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \
    import CONV_MACHINE_PARAMETERS
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
    import convert_model_to_short_conv
from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
    import causal_context
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
    import _round_f32
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import MICROBATCH, TUNED_WINNER, _require_lease
from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
    import _build_k10

# k18_short_conv's own module object -- the committed mixer resolves
# ``composed_scan`` through THIS binding, so a recording shim installed on
# it (and removed before anything is timed) sees the exact production args.
import experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
    as _k18mod

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k21_runs"
SEED = 3401                                  # the standing seed
LAYER = 6                                    # K20's median-measured layer
UNIT = ("one (layer, window-group): batch 8 x 2048 tokens at the committed "
        "150M width 576, bf16 autocast compute, fp32 params -- the scan "
        "probe at batch 8 x 2048 x heads 2 x rank/value 128 x key 16, "
        "segment 64 (the registered shape)")
PROFILING_TOOLS = dict(
    torch_profiler="CUPTI kernel timings: per-kernel self CUDA time, "
                   "per-op CPU time",
    ncu="UNAVAILABLE on this host: ERR_NVGPUCTRPERM (driver-level "
        "performance-counter permission denies non-admin profiling, "
        "verified same-session with a plain sgemm probe) -- the "
        "achieved-bandwidth/compute classification is therefore ANALYTIC: "
        "the scan kernel's byte traffic and op counts are DETERMINISTIC "
        "(counted from the committed Triton source), so achieved rate = "
        "traffic / measured kernel time, against the card's measured peaks",
    nsys_gpu_metrics="UNAVAILABLE: the same ERR_NVGPUCTRPERM")


# ------------------------------------------------------------------ plumbing


def profile_table(prof, path: Path, iters: int) -> dict:
    """The per-kernel self-CUDA-time table: full text to the run dir, the
    classified aggregates + top rows into the receipt."""
    averages = prof.key_averages()
    table = averages.table(sort_by="self_cuda_time_total", row_limit=40)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table)
    rows = []
    total_us = 0.0
    for evt in averages:
        self_us = float(getattr(evt, "self_device_time_total", 0.0)
                        or getattr(evt, "self_cuda_time_total", 0.0))
        if self_us <= 0.0:
            continue
        total_us += self_us
        rows.append(dict(name=evt.key[:120], count=int(evt.count),
                         self_cuda_ms=round(self_us / 1000.0, 3)))
    rows.sort(key=lambda r: -r["self_cuda_ms"])

    # count DEVICE KERNELS only: op/autograd wrapper rows ("aten::...",
    # "GeneratedBackwardFor_...") mirror their child kernel's device time and
    # would double-count it; kernels appear as "void at::native::<...>_kernel",
    # cuBLAS "void nvjet_...", "Memcpy/Memset ...", or a bare Triton name.
    def is_kernel(name: str) -> bool:
        lowered = name.lower()
        return (name.startswith("void ") or "_kernel" in lowered
                or "nvjet" in lowered or "gemm" in lowered
                or "cutlass" in lowered or name.startswith("Memcpy")
                or name.startswith("Memset") or "_k15_seq_fwd_kernel" in name)

    kernel_rows = [row for row in rows if is_kernel(row["name"])]
    kernel_total_us = sum(row["self_cuda_ms"] * 1000.0
                          for row in kernel_rows)

    def bucket(name: str) -> str:
        if "_k15_seq_fwd_kernel" in name:
            return "the composed scan kernel"
        lowered = name.lower()
        if ("gemm" in lowered or "cublas" in lowered or "nvjet" in lowered
                or "cutlass" in lowered):
            return "GEMM (the projections / the out projection)"
        return "elementwise/copy (conv taps, sigmoid, gate arithmetic, casts)"

    by_bucket: dict[str, float] = {}
    for row in kernel_rows:
        key = bucket(row["name"])
        by_bucket[key] = by_bucket.get(key, 0.0) + row["self_cuda_ms"]
    return dict(iterations=iters,
                kernel_self_cuda_ms_per_iter=round(kernel_total_us / 1000.0
                                                   / iters, 3),
                buckets_ms_per_iter={k: round(v / iters, 3)
                                     for k, v in sorted(
                                         by_bucket.items(), key=lambda kv:
                                         -kv[1])},
                top_kernels=kernel_rows[:15],
                full_table=str(path.relative_to(REPO)),
                counting_note="device kernels only -- op/autograd wrapper "
                             "rows that mirror their child kernel's device "
                             "time are excluded to avoid double-counting")


# ------------------------------------------------------- the machine + capture


def build_and_capture(stream):
    """The committed K18 machine (K20's per_layer build verbatim), one real
    microbatch group driven through it with hooks: layer 6's own mixer input
    AND the exact scan arguments its mixer passes (the recording shim sees
    the production call; it is removed before anything is timed)."""
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), SEED)
    model = convert_model_to_composition(model)
    model = convert_model_to_short_conv(model, birth="identity")
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != CONV_MACHINE_PARAMETERS:
        raise ValueError(f"the conv machine's parameters must be the "
                         f"registration's {CONV_MACHINE_PARAMETERS}; got "
                         f"{parameters}")
    depth = len(model.blocks)
    recorded_scans: list[tuple[list, dict]] = []
    original = _k18mod.composed_scan

    def shim(*args, **kwargs):
        if len(recorded_scans) < depth:
            recorded_scans.append(([a.detach().clone() for a in args],
                                   dict(kwargs)))
        return original(*args, **kwargs)

    _k18mod.composed_scan = shim
    try:
        captured, cap = K20.capture_inputs(model, "ours", stream, [LAYER],
                                           optimizer_config["microbatch"],
                                           model.shape.width)
    finally:
        _k18mod.composed_scan = original
    if len(recorded_scans) != depth:
        raise ValueError(f"expected {depth} composed_scan calls in the "
                         f"capture forward; saw {len(recorded_scans)}")
    x = captured[f"layer_{LAYER}"]["mixer"]
    scan_args, scan_kwargs = recorded_scans[LAYER]
    (payload, write_keys, query_keys, decay, gates) = scan_args
    policy = scan_kwargs["policy"]
    mixer = model.blocks[LAYER].mixer
    shape_note = dict(
        payload=tuple(payload.shape), payload_dtype=str(payload.dtype),
        write_keys=tuple(write_keys.shape),
        write_keys_dtype=str(write_keys.dtype),
        decay_dtype=str(decay.dtype), gates_dtype=str(gates.dtype),
        rank=int(scan_kwargs["rank"]),
        rank_max=int(scan_kwargs["rank_max"]),
        segment_tokens=int(scan_kwargs["segment_tokens"]),
        gated=bool(scan_kwargs["gated"]),
        key_dim=int(write_keys.shape[-1]),
        policy=dict(tau_match=policy.tau_match, tau_merge=policy.tau_merge,
                    tau_pop=policy.tau_pop, tie_break=policy.tie_break,
                    read_scale=policy.read_scale, ops=bool(policy.ops)),
        mixer_input=tuple(x.shape))
    del model, captured, recorded_scans
    torch.cuda.empty_cache()
    return (mixer, x, scan_args, scan_kwargs, cap, shape_note, summary,
            parameters)


# --------------------------------------------------------------- the peaks


def measure_peaks() -> dict:
    """The card's measured ceilings the floors are stated against."""
    properties = torch.cuda.get_device_properties(0)
    sm_count = int(properties.multi_processor_count)
    # DRAM bandwidth: a 2 GiB device copy moves 4 GiB per iteration.
    n = 1 << 29
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        torch.cuda.synchronize()
        started = time.monotonic()
        b.copy_(a)
        torch.cuda.synchronize()
        samples.append(time.monotonic() - started)
    dram_gbs = (2 * n * 4) / statistics.median(samples) / 1e9
    del a, b
    torch.cuda.empty_cache()
    # fp32 (no TF32) and bf16 matmul rates.  The scan's arithmetic is fp32
    # CUDA-core work (inline-asm mul.rn/add.rn, NO FMA contraction), so its
    # issue ceiling is the fp32 pipe; a matmul FMA counts 2 flops.
    m = 8192
    matmul_tflops = {}
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        aa = torch.randn(m, m, dtype=dtype, device="cuda")
        bb = torch.randn(m, m, dtype=dtype, device="cuda")
        tf32_state = torch.backends.cuda.matmul.allow_tf32
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for _ in range(3):
                aa @ bb
            torch.cuda.synchronize()
            times = []
            for _ in range(8):
                torch.cuda.synchronize()
                started = time.monotonic()
                aa @ bb
                torch.cuda.synchronize()
                times.append(time.monotonic() - started)
            matmul_tflops[label] = round(
                2 * m ** 3 / statistics.median(times) / 1e12, 1)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_state
        del aa, bb
        torch.cuda.empty_cache()
    fp32_ops_per_s = matmul_tflops["fp32"] * 1e12 / 2   # FMA = 2 flops
    return dict(
        device=torch.cuda.get_device_name(0),
        sm_count=sm_count,
        measured_dram_bandwidth_gbs=round(dram_gbs, 1),
        measured_matmul_tflops=matmul_tflops,
        measured_fp32_op_rate_gop_s=round(fp32_ops_per_s / 1e9, 1),
        scan_parallelism_note=(
            "the content-addressed decisions at token t read the whole "
            "store, so a bit-identical walk is SEQUENTIAL over tokens; the "
            "independent rows at the registered shape are batch*heads = 16 "
            f"of the card's {sm_count} SMs -- the semantics' own occupancy "
            "ceiling, an idealization that assumes each of the 16 resident "
            "CTAs sustains the full-SM fp32 issue rate"),
        semantics_ceiling_fp32_gop_s=round(
            fp32_ops_per_s * 16 / sm_count / 1e9, 1),
        cardwide_ceiling_fp32_gop_s=round(fp32_ops_per_s / 1e9, 1))


# --------------------------------------------------- the analytic cost model


def scan_cost_model(shape: dict, peaks: dict) -> dict:
    """The committed kernel's DETERMINISTIC per-pass arithmetic and byte
    traffic, counted from the Triton source at the registered shape
    (R = rank = 128, V = value_dim = 128, K = key_dim = 16, T = 2048,
    rows = batch*heads = 16, SEG = 64, L = 2 lanes, GATED = 1).  Every fp32
    op is a SEPARATE instruction (the discipline's mul.rn/add.rn inline asm
    -- no FMA contraction), so 1 op = 1 fp32-pipe issue; selects/compares
    are counted as their own class (they issue on the same pipes)."""
    R = int(shape["rank"])
    V = int(shape["payload"][-1])
    K = int(shape["key_dim"])
    T = int(shape["payload"][1])
    rows = int(shape["payload"][0]) * int(shape["payload"][2])
    SEG = int(shape["segment_tokens"])
    token_rows = T * rows
    # per token-row counts, from the committed kernel source, in execution
    # order.  m/a/d/s = fp32 mul/add/div/sqrt; sel = select/compare ops;
    # ld/st = load/store bytes (gates are fp32, the other inputs bf16).
    passes = [
        ("input loads (p/wk/qk/lam/gates)",
         dict(m=0, a=0, d=0, s=0, sel=0,
              ld=V * 2 + K * 2 + K * 2 + V * 2 + V * 4, st=0)),
        ("value norms: the nr fold + strength (the MERGE strengths)",
         dict(m=R * V + R, a=R * (V - 1), d=0, s=R, sel=R, ld=0, st=0)),
        ("key/query norms (nk/nqk/uqk)",
         dict(m=R * K + K, a=R * (K - 1) + (K - 1), d=K, s=R + 1, sel=0,
              ld=0, st=0)),
        ("read scores: the cosine-sim sweep over the slots",
         dict(m=R * K + R, a=R * (K - 1), d=R, s=0, sel=R, ld=0, st=0)),
        ("the read gather (the rank-axis fold) + the read store",
         dict(m=R * V, a=V * (R - 1), d=0, s=0, sel=0, ld=0, st=V * 4)),
        ("decision sim sweep (the threshold chain's inputs)",
         dict(m=R * K + K, a=R * (K - 1) + (K - 1), d=R + K, s=1, sel=R,
              ld=0, st=0)),
        ("the threshold-decision chain (best/second/worst + op select)",
         dict(m=R, a=6 * (R - 1), d=0, s=0, sel=14 * R, ld=0, st=0)),
        ("the MERGE strengths (fw/sw/a_mix/b_mix)",
         dict(m=0, a=2 * (R - 1) + 1, d=2, s=0, sel=2 * R, ld=0, st=0)),
        ("gates lane-0 (the old_pre hit-sum + the delta-rule blend)",
         dict(m=4 * V, a=V * (R - 1) + V, d=0, s=0, sel=R * V, ld=0, st=0)),
        ("the write scatter's old_row hit-sum",
         dict(m=0, a=V * (R - 1), d=0, s=0, sel=R * V, ld=0, st=0)),
        ("DECAY: the per-channel value tile multiply",
         dict(m=R * V, a=0, d=0, s=0, sel=0, ld=0, st=0)),
        ("the write scatter apply (new_row + where-updates + key lane)",
         dict(m=K, a=V, d=0, s=0, sel=2 * R * V + 2 * R + 2 * R * K + V,
              ld=0, st=0)),
        ("record stores (schedule lanes + live_pre + payload lanes)",
         dict(m=0, a=0, d=0, s=0, sel=0, ld=0,
              st=2 * 8 + 2 * 8 + 2 + 2 + 2 * V * 4 + R)),
        ("segment checkpoints (amortized per token)",
         dict(m=0, a=0, d=0, s=0, sel=0, ld=0,
              st=(R * V * 4 + R + R * K * 4) // SEG)),
    ]
    rate_card = peaks["cardwide_ceiling_fp32_gop_s"] * 1e9
    rate_16 = peaks["semantics_ceiling_fp32_gop_s"] * 1e9
    bw = peaks["measured_dram_bandwidth_gbs"] * 1e9
    table = []
    totals = dict(ops=0, sel=0, ld_bytes=0, st_bytes=0,
                  floor_16sm_ms=0.0, floor_card_ms=0.0, floor_memory_ms=0.0)
    for name, c in passes:
        ops = c["m"] + c["a"] + c["d"] + c["s"]
        total_ops = (ops + c["sel"]) * token_rows
        total_ld = c["ld"] * token_rows
        total_st = c["st"] * token_rows
        floor_16 = total_ops / rate_16 * 1000.0
        floor_card = total_ops / rate_card * 1000.0
        floor_mem = (total_ld + total_st) / bw * 1000.0
        table.append(dict(
            name=name,
            per_token_row=dict(fp32_mul_add_div_sqrt_ops=ops,
                               select_compare_ops=c["sel"],
                               load_bytes=c["ld"], store_bytes=c["st"]),
            per_scan=dict(fp32_ops=total_ops, load_bytes=total_ld,
                          store_bytes=total_st),
            floor_at_semantics_ceiling_ms=round(floor_16, 4),
            floor_at_card_wide_peak_ms=round(floor_card, 4),
            floor_memory_at_measured_bw_ms=round(floor_mem, 4)))
        totals["ops"] += total_ops
        totals["sel"] += c["sel"] * token_rows
        totals["ld_bytes"] += total_ld
        totals["st_bytes"] += total_st
        totals["floor_16sm_ms"] += floor_16
        totals["floor_card_ms"] += floor_card
        totals["floor_memory_ms"] += floor_mem
    return dict(
        shape=dict(R=R, V=V, K=K, T=T, rows=rows, segment=SEG,
                   token_rows=token_rows),
        counting_discipline=("1 op = 1 fp32-pipe issue (the discipline's "
                             "inline-asm mul.rn/add.rn -- no FMA "
                             "contraction); selects/compares issue on the "
                             "same pipes and are counted separately"),
        per_pass=table,
        per_scan_totals=dict(
            fp32_ops=totals["ops"], select_ops=totals["sel"],
            load_bytes=totals["ld_bytes"], store_bytes=totals["st_bytes"],
            dram_bytes_moved=totals["ld_bytes"] + totals["st_bytes"]),
        floors=dict(
            compute_at_semantics_ceiling_ms=round(totals["floor_16sm_ms"], 3),
            compute_at_card_wide_peak_ms=round(totals["floor_card_ms"], 3),
            memory_at_measured_bw_ms=round(totals["floor_memory_ms"], 3)),
        note=("the semantics' ceiling: the same arithmetic, sequential over "
              "tokens, 16 independent rows (batch*heads) of the card's SMs "
              "-- the floor ANY bit-identical implementation of these "
              "semantics at this shape must exceed; the card-wide figure is "
              "stated only to show what the FORBIDDEN token-parallel "
              "reformulation would buy"))


# ------------------------------------------------- the per-pass ablations


def ablation_breakdown(scan_args, scan_kwargs, state0, live0, keys0,
                      policy_f32, reference, warmup: int, iters: int) -> dict:
    """The committed kernel's measured milliseconds attributed to its
    constituent passes: the faithful scratch copy (bit-identity-checked
    FIRST -- the instrument fails closed) minus one ablation per pass."""
    payload, write_keys, query_keys, decay, gates = scan_args
    rank = int(scan_kwargs["rank"])
    seg = int(scan_kwargs["segment_tokens"])
    gated = bool(scan_kwargs["gated"])

    def launch(abl):
        return lambda: MB.scratch_forward_launch(
            payload, write_keys, query_keys, decay, gates, state0, live0,
            keys0, policy_f32, rank, seg, gated, 8, 3, abl=abl)

    base_fn = launch(MB.ABL_NONE)
    faithful = MB.bit_identity(reference, base_fn())
    if not faithful["bit_identical"]:
        raise RuntimeError(
            "the scratch faithful copy does NOT reproduce the committed op "
            "pair bit for bit -- the instrument fails closed; nothing the "
            "ablations say can count")
    base_samples = MB.time_launch(base_fn, warmup, iters)
    base_median = statistics.median(base_samples)
    rows = {}
    for abl in MB.ABLATIONS[1:]:
        fn = launch(abl)
        samples = MB.time_launch(fn, warmup, iters)
        median = statistics.median(samples)
        rows[f"abl_{abl}"] = dict(
            name=MB.ABLATION_NAMES[abl], median_ms=round(median, 3),
            attributed_pass_cost_ms=round(base_median - median, 3),
            attributed_share_pct=round(
                100.0 * (base_median - median) / base_median, 2),
            timing=K20.spread(samples))
        print(f"[ablation] {MB.ABLATION_NAMES[abl]}: "
              f"{median:.3f} ms (delta {base_median - median:+.3f})",
              flush=True)
    return dict(
        faithful_copy=dict(
            bit_identical=True,
            median_ms=round(base_median, 3),
            timing=K20.spread(base_samples),
            note="the scratch kernel at all flags zero reproduces the "
                 "committed op pair bit for bit -- the copy is the "
                 "instrument, and its timing is the attribution baseline"),
        brackets=rows,
        caveat=("ablation deltas are marginal costs in the committed "
                "kernel's own regime (per-branch compiler effects included); "
                "the decision sim sweep is NOT ablated because zeroing it "
                "would change the discrete op mix and corrupt the "
                "attribution -- its cost is stated from the op-count model"))


# ------------------------------------------------------------- the levers


def _launch_closure(scan_args, scan_kwargs, state0, live0, keys0, policy_f32,
                    warps, stages, **flags):
    payload, write_keys, query_keys, decay, gates = scan_args
    rank = int(scan_kwargs["rank"])
    seg = int(scan_kwargs["segment_tokens"])
    gated = bool(scan_kwargs["gated"])

    def fn():
        return MB.scratch_forward_launch(
            payload, write_keys, query_keys, decay, gates, state0, live0,
            keys0, policy_f32, rank, seg, gated, warps, stages, **flags)
    return fn


def lever_brackets(scan_args, scan_kwargs, state0, live0, keys0, policy_f32,
                   reference, committed_direct_median: float,
                   warmup: int, iters: int) -> dict:
    """Each lever alone (bit-identity first, then timing), then the stack of
    every passing lever.  Failing levers are still timed (fewer iters) for
    the record, marked not-bit-identical."""
    levers = {
        "reduce_nr__same_pairings_split_fold": dict(reduce_nr=1),
        "reduce_sum__defined_order_axis0_fold": dict(reduce_sum=1),
        "lazy_norms__merge_branch_only": dict(lazy_norms=1),
        "lazy_old_row__erase_else_only": dict(lazy_old_row=1),
    }
    results = {}
    passing = []
    for name, flags in levers.items():
        fn = _launch_closure(scan_args, scan_kwargs, state0, live0, keys0,
                             policy_f32, 8, 3, **flags)
        check = MB.bit_identity(reference, fn())
        if check["bit_identical"]:
            samples = MB.time_launch(fn, warmup, iters)
            passing.append(name)
        else:
            samples = MB.time_launch(fn, 2, 4)
        median = statistics.median(samples)
        results[name] = dict(
            flags=flags, bit_identical=check["bit_identical"],
            failed_channels=[k for k, v in check["channels"].items()
                             if not v["bit_identical"]],
            median_ms=round(median, 3),
            speedup_vs_committed=round(committed_direct_median / median, 3),
            timing=K20.spread(samples))
        print(f"[lever] {name}: bit_identical={check['bit_identical']} "
              f"{median:.3f} ms "
              f"({committed_direct_median / median:.2f}x)", flush=True)
    stacked = None
    if passing:
        stacked_flags: dict = {}
        for name in passing:
            stacked_flags.update(levers[name])
        fn = _launch_closure(scan_args, scan_kwargs, state0, live0, keys0,
                             policy_f32, 8, 3, **stacked_flags)
        check = MB.bit_identity(reference, fn())
        samples = MB.time_launch(fn, warmup, iters) \
            if check["bit_identical"] else MB.time_launch(fn, 2, 4)
        median = statistics.median(samples)
        stacked = dict(
            flags=stacked_flags, levers_stacked=passing,
            bit_identical=check["bit_identical"],
            failed_channels=[k for k, v in check["channels"].items()
                             if not v["bit_identical"]],
            median_ms=round(median, 3),
            speedup_vs_committed=round(committed_direct_median / median, 3),
            timing=K20.spread(samples))
        print(f"[lever] stacked({'+'.join(passing)}): "
              f"bit_identical={check['bit_identical']} {median:.3f} ms "
              f"({committed_direct_median / median:.2f}x)", flush=True)
    return dict(individual=results, stacked=stacked)


def launch_config_sweep(scan_args, scan_kwargs, state0, live0, keys0,
                        policy_f32, reference, committed_direct_median,
                        warmup: int, iters: int, stacked_flags: dict | None):
    """The COMMITTED kernel verbatim at other (num_warps, num_stages)
    brackets, and the stacked scratch config at a small warp bracket."""
    payload, write_keys, query_keys, decay, gates = scan_args
    rank = int(scan_kwargs["rank"])
    seg = int(scan_kwargs["segment_tokens"])
    gated = bool(scan_kwargs["gated"])

    def committed(warps, stages):
        return lambda: _k15_forward_launch(
            payload, write_keys, query_keys, decay, gates, state0, live0,
            keys0, policy_f32, rank, seg, gated, collect=False,
            warps=warps, stages=stages)

    rows = []
    for warps, stages in ((8, 3), (4, 3), (8, 1), (8, 5), (16, 3)):
        fn = committed(warps, stages)
        check = MB.bit_identity(reference, fn())
        samples = MB.time_launch(fn, warmup, iters) \
            if check["bit_identical"] else MB.time_launch(fn, 2, 4)
        median = statistics.median(samples)
        rows.append(dict(kernel="committed", warps=warps, stages=stages,
                         bit_identical=check["bit_identical"],
                         median_ms=round(median, 3),
                         speedup_vs_committed=round(
                             committed_direct_median / median, 3)))
        print(f"[launch] committed (w{warps}, s{stages}): "
              f"bit={check['bit_identical']} {median:.3f} ms "
              f"({committed_direct_median / median:.2f}x)", flush=True)
    scratch_rows = []
    if stacked_flags:
        for warps, stages in ((8, 3), (8, 1), (8, 5), (4, 3), (16, 3),
                              (16, 4)):
            fn = _launch_closure(scan_args, scan_kwargs, state0, live0,
                                 keys0, policy_f32, warps, stages,
                                 **stacked_flags)
            check = MB.bit_identity(reference, fn())
            samples = MB.time_launch(fn, warmup, iters) \
                if check["bit_identical"] else MB.time_launch(fn, 2, 4)
            median = statistics.median(samples)
            scratch_rows.append(dict(kernel="scratch_stacked", warps=warps,
                                     stages=stages,
                                     bit_identical=check["bit_identical"],
                                     median_ms=round(median, 3),
                                     speedup_vs_committed=round(
                                         committed_direct_median / median,
                                         3)))
            print(f"[launch] stacked (w{warps}, s{stages}): "
                  f"bit={check['bit_identical']} {median:.3f} ms "
                  f"({committed_direct_median / median:.2f}x)", flush=True)
    return dict(committed_kernel=rows, stacked_scratch=scratch_rows,
                note="(8, 3) is the committed configuration "
                     "(_seq_warps(128, 128) = 8 warps, stages 3)")


def wrapper_levers(mixer, x, scan_args, scan_kwargs, policy_f32, reference,
                   warmup: int, iters: int) -> dict:
    """Levers OUTSIDE the kernel: the scan's input dtype regime, and fused
    projections.  The same receipts bar: bit-identity vs the committed
    outputs at the same inputs."""
    payload, write_keys, query_keys, decay, gates = scan_args
    rank = int(scan_kwargs["rank"])
    seg = int(scan_kwargs["segment_tokens"])
    gated = bool(scan_kwargs["gated"])
    results = {}
    # ---- the fp32 pre-cast regime: the kernel loads bf16 and casts to fp32
    # (exact); feeding it fp32 tensors gives the SAME fp32 values.
    f32_args = [t.to(torch.float32) for t in scan_args]

    def fp32_call():
        return composed_scan(*f32_args, **scan_kwargs)

    check = MB.bit_identity(reference, MB.committed_reference(
        *f32_args, policy_f32, rank, seg, gated))
    samples = MB.time_launch(fp32_call, 2, 6)
    median = statistics.median(samples)
    results["fp32_precast_inputs"] = dict(
        bit_identical=check["bit_identical"],
        failed_channels=[k for k, v in check["channels"].items()
                         if not v["bit_identical"]],
        median_ms=round(median, 3), timing=K20.spread(samples),
        note="the committed kernel already loads bf16 and casts to fp32 "
             "in-register (exact), so pre-casting only moves the cast out "
             "of the kernel and QUADRUPLES the input bytes")
    print(f"[wrapper] fp32 pre-cast: bit={check['bit_identical']} "
          f"{median:.3f} ms", flush=True)
    # ---- fused projections: key | query_key (both consume the ctx-window
    # context) and value | b_proj (both consume x).  Whether cuBLAS picks
    # the same accumulation for the fused width is EMPIRICAL.
    batch, tokens, _ = x.shape
    context = causal_context(x, mixer.ctx).reshape(
        batch, tokens, mixer.ctx * mixer.width)

    def separate_kq():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mixer.key(context), mixer.query_key(context)

    k_out, q_out = separate_kq()
    w_kq = torch.cat([mixer.key.weight, mixer.query_key.weight], 0)
    k_width = mixer.key.weight.shape[0]

    def fused_kq():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = F.linear(context, w_kq)
            return fused.split([k_width, fused.shape[-1] - k_width], dim=-1)

    k_fused, q_fused = fused_kq()
    kq_identical = bool(torch.equal(k_out, k_fused)
                        and torch.equal(q_out, q_fused))
    sep_samples = MB.time_launch(lambda: separate_kq(), warmup, iters)
    fused_samples = MB.time_launch(lambda: fused_kq(), warmup, iters)
    results["fused_key_query_key_projections"] = dict(
        bit_identical=kq_identical,
        separate_ms=round(statistics.median(sep_samples), 3),
        fused_ms=round(statistics.median(fused_samples), 3),
        speedup=round(statistics.median(sep_samples)
                      / statistics.median(fused_samples), 3),
        note="the two projections share the (b, T, ctx*width) context input; "
             "the fused GEMM's accumulation order is cuBLAS's own choice at "
             "the wider output -- the check decides")
    print(f"[wrapper] fused key|query_key: bit={kq_identical} "
          f"{statistics.median(sep_samples):.3f} -> "
          f"{statistics.median(fused_samples):.3f} ms", flush=True)

    def separate_vb():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mixer.value(x), mixer.b_proj(x)

    v_out, b_out = separate_vb()
    w_vb = torch.cat([mixer.value.weight, mixer.b_proj.weight], 0)
    v_width = mixer.value.weight.shape[0]

    def fused_vb():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = F.linear(x, w_vb)
            return fused.split([v_width, fused.shape[-1] - v_width], dim=-1)

    v_fused, b_fused = fused_vb()
    vb_identical = bool(torch.equal(v_out, v_fused)
                        and torch.equal(b_out, b_fused))
    sep_samples = MB.time_launch(lambda: separate_vb(), warmup, iters)
    fused_samples = MB.time_launch(lambda: fused_vb(), warmup, iters)
    results["fused_value_bproj_projections"] = dict(
        bit_identical=vb_identical,
        separate_ms=round(statistics.median(sep_samples), 3),
        fused_ms=round(statistics.median(fused_samples), 3),
        speedup=round(statistics.median(sep_samples)
                      / statistics.median(fused_samples), 3))
    print(f"[wrapper] fused value|b_proj: bit={vb_identical} "
          f"{statistics.median(sep_samples):.3f} -> "
          f"{statistics.median(fused_samples):.3f} ms", flush=True)
    del context, k_out, q_out, k_fused, q_fused, v_out, b_out, v_fused, b_fused
    torch.cuda.empty_cache()
    return results


def out_of_scope_statements() -> list[dict]:
    """The levers that CANNOT be bracketed as speedups because they cannot
    preserve bit-exactness, or do not exist to begin with -- stated
    honestly, per the task."""
    return [
        dict(lever="bf16 accumulation in the scan's value/state arithmetic",
             verdict="cannot preserve bit-exactness",
             reason="the committed discipline is fp32 mul.rn/add.rn with no "
                    "FMA contraction; any bf16 accumulation regime changes "
                    "the rounding of every folded sum -- a different "
                    "machine, refused by the receipts bar"),
        dict(lever="precomputed / incremental key norms (maintain nk "
                   "across tokens instead of recomputing the tile)",
             verdict="not worth a bracket + the incremental form is not "
                     "bit-identical",
             reason="only one slot's key changes per token, so recomputing "
                    "only the hit slot's norm is value-identical, but the "
                    "recompute is <= 2*R*K of ~194K ops per token-row "
                    "(~2%): bounded below relevance; the arithmetic "
                    "incremental update (sum += new - old) is not the "
                    "defined recompute and is not bit-identical"),
        dict(lever="a batched cosine-sim over all slots at once vs "
                   "per-slot loops",
             verdict="not applicable -- the committed kernel is already "
                     "batched",
             reason="the committed kernel already sweeps all 128 slots per "
                    "token in ONE register tile (scores and sim are "
                    "(rank,) tiles built by one (rank, key) fold each); "
                    "there is no per-slot loop to batch away; the eager "
                    "oracle is batched the same way"),
        dict(lever="the read gather / hit-sums as tl.dot (tensor-core) "
                   "matmuls",
             verdict="cannot preserve bit-exactness",
             reason="MMA accumulation order is implementation-defined and "
                    "is not the defined-order rank-axis tree; the TF32/bf16 "
                    "tensor paths change the arithmetic outright -- the "
                    "chunked-delta-rule formulation GDN2 uses is exactly "
                    "the token-parallel reformulation these semantics "
                    "forbid at fixed schedule"),
    ]


# ---------------------------------------------------------------- the verdict


def verdict_stage(reproduce, profiler_scan, peaks, cost_model, brackets,
                  sweep, k20: dict) -> dict:
    """The inherent-vs-incidental decomposition, in K20's own units."""
    committed_scan_wall_ms = reproduce["composed_scan_forward"]["median_ms"]
    # the committed DIRECT launch at (8, 3) -- the apples-to-apples baseline
    committed_direct = None
    for row in sweep["committed_kernel"]:
        if row["warps"] == 8 and row["stages"] == 3:
            committed_direct = row["median_ms"]
    if committed_direct is None:
        raise ValueError("the committed (8, 3) launch row is missing")
    # the best BIT-IDENTICAL variant measured anywhere in this lane
    candidates = []
    for row in sweep["committed_kernel"]:
        if row["bit_identical"]:
            candidates.append((row["median_ms"], "committed launch config "
                               f"(w{row['warps']}, s{row['stages']})"))
    for name, row in brackets["individual"].items():
        if row["bit_identical"]:
            candidates.append((row["median_ms"], f"lever {name}"))
    if brackets["stacked"] and brackets["stacked"]["bit_identical"]:
        candidates.append((brackets["stacked"]["median_ms"],
                           "the stacked levers"))
    for row in sweep["stacked_scratch"]:
        if row["bit_identical"]:
            candidates.append((row["median_ms"],
                               f"stacked at (w{row['warps']}, "
                               f"s{row['stages']})"))
    best_ms, best_name = min(candidates)
    speedup = committed_direct / best_ms

    ours = k20["projection"]["per_layer_medians"]["ours"]
    gdn2 = k20["projection"]["per_layer_medians"]["gdn2"]
    steps = k20["projection"]["measured_step_costs"]
    groups = k20["projection"]["groups_per_step"]      # 8
    depth = k20["projection"]["depth"]                 # 24
    mixer_fwd_ms = reproduce["mixer_forward"]["median_ms"]
    mixer_bwd_ms = ours["mixer_backward_ms"]
    gdn2_pair_ms = gdn2["mixer_forward_ms"] + gdn2["mixer_backward_ms"]

    def project(scan_ms_now: float) -> dict:
        mixer_fwd_new = mixer_fwd_ms - committed_scan_wall_ms + scan_ms_now
        pair_new = mixer_fwd_new + mixer_bwd_ms
        ratio_new = pair_new / gdn2_pair_ms
        delta_new = pair_new - gdn2_pair_ms
        added = 3 * groups * delta_new / 1000.0
        step_hybrid = steps["pure_gdn2_s"] + added \
            + (steps["ours_optimizer_s"] - steps["gdn2_optimizer_s"])
        step_ours_new = steps["pure_ours_s"] - depth * groups * (
            committed_scan_wall_ms - scan_ms_now) / 1000.0
        return dict(
            scan_forward_ms=round(scan_ms_now, 3),
            mixer_forward_ms=round(mixer_fwd_new, 3),
            per_mixer_ratio=round(ratio_new, 2),
            hybrid_step_s_at_3_of_24=round(step_hybrid, 3),
            hybrid_exposure_pct_of_gdn2_in_2700s=round(
                100.0 * steps["pure_gdn2_s"] / step_hybrid, 1),
            pure_ours_step_s=round(step_ours_new, 3))

    floor_compute = cost_model["floors"]["compute_at_semantics_ceiling_ms"]
    floor_memory = cost_model["floors"]["memory_at_measured_bw_ms"]
    floor_bound = max(floor_compute, floor_memory)
    # ---- the honest bound classification: the scan kernel's byte traffic
    # and op counts are DETERMINISTIC, so achieved rates are traffic over
    # the measured kernel time -- against the card's measured peaks.
    kernel_ms = profiler_scan["buckets_ms_per_iter"].get(
        "the composed scan kernel") if profiler_scan else None
    if kernel_ms is None:
        kernel_ms = committed_direct
    bytes_moved = cost_model["per_scan_totals"]["dram_bytes_moved"]
    fp32_ops = cost_model["per_scan_totals"]["fp32_ops"]
    sel_ops = cost_model["per_scan_totals"]["select_ops"]
    achieved_bw_gbs = bytes_moved / (kernel_ms / 1000.0) / 1e9
    achieved_fp32_gops = fp32_ops / (kernel_ms / 1000.0) / 1e9
    classification = dict(
        kernel_cuda_ms=round(kernel_ms, 3),
        dram_bytes_moved=bytes_moved,
        achieved_dram_gbs=round(achieved_bw_gbs, 2),
        achieved_vs_measured_peak_bw_pct=round(
            100.0 * achieved_bw_gbs / peaks["measured_dram_bandwidth_gbs"], 2),
        achieved_fp32_issue_gop_s=round(achieved_fp32_gops, 1),
        achieved_vs_cardwide_fp32_peak_pct=round(
            100.0 * achieved_fp32_gops
            / peaks["cardwide_ceiling_fp32_gop_s"], 2),
        achieved_vs_semantics_ceiling_pct=round(
            100.0 * achieved_fp32_gops
            / peaks["semantics_ceiling_fp32_gop_s"], 2),
        select_ops_also_issued=sel_ops,
        classification="LATENCY/LAYOUT-BOUND, neither memory- nor "
                      "throughput-bound: the kernel moves 129 MB per scan "
                      "(a sub-millisecond memory floor) while running at "
                      f"{100.0 * achieved_fp32_gops / peaks['semantics_ceiling_fp32_gop_s']:.0f}% "
                      "of its own 16-SM fp32 issue ceiling -- the serial "
                      "per-token walk plus tl.sum's reduction-layout "
                      "machinery own the time, not the arithmetic's volume")
    return dict(
        the_question="is the 7.28x INHERENT to the slot semantics or "
                     "INCIDENTAL to a receipts-first kernel",
        measured=dict(
            committed_scan_wall_ms=committed_scan_wall_ms,
            committed_direct_launch_ms=committed_direct,
            scan_kernel_cuda_ms=profiler_scan[
                "buckets_ms_per_iter"].get("the composed scan kernel")
                if profiler_scan else None,
            best_bit_identical_ms=best_ms,
            best_bit_identical_variant=best_name,
            bit_identical_speedup=round(speedup, 2)),
        bound_classification=classification,
        decomposition=dict(
            recoverable_bit_identically_pct=round(
                100.0 * (1.0 - best_ms / committed_direct), 1),
            semantics_floor_pct_of_current=round(
                100.0 * floor_bound / committed_direct, 1),
            floor_detail=dict(
                compute_at_16_sm_ceiling_ms=floor_compute,
                memory_at_measured_bw_ms=floor_memory,
                floor_bound_ms=round(floor_bound, 3),
                floor_over_current=round(floor_bound / committed_direct, 3)),
            residual_note="the gap between the best bit-identical variant "
                          "and the semantics' floor is implementation "
                          "headroom that only a kernel rewrite lane could "
                          "claim -- stated, not taken in this lane"),
        projections=dict(
            k20_anchor=dict(
                per_mixer_ratio=round(
                    ours["mixer_fwd_bwd_ms"] / gdn2["mixer_fwd_bwd_ms"], 2),
                pure_ours_step_s=steps["pure_ours_s"],
                pure_gdn2_step_s=steps["pure_gdn2_s"],
                hybrid_exposure_pct_at_3_of_24=
                k20["projection"]["ratios"]["3"]["mixer_swap_banked_muon"][
                    "exposure_pct_of_gdn2_in_2700s"]),
            after_the_bit_identical_recovery=project(best_ms),
            at_the_semantics_floor=project(floor_bound)),
        candidate_followup=(
            "a bit-identical scan speedup of "
            f"{speedup:.2f}x was found and validated on captured real "
            "activations (every output channel torch.equal vs the committed "
            "opaque op pair) -- stated as a CANDIDATE ENGINEERING FOLLOW-UP "
            "for its own lane + sign-off; NOT integrated into the committed "
            "engines in this lane"
            if speedup >= 1.5 else
            f"no bit-identical lever reached the 1.5x candidate bar "
            f"(best: {speedup:.2f}x)"),
        backward_boundary="the levers bracketed here are the FORWARD "
                          "scan's; the backward's cost body is a different "
                          "kernel set (segment replay + walks) whose levers "
                          "were not bracketed in this lane -- the step-wall "
                          "projections apply the forward's recoverable "
                          "fraction to the forward term only, honestly")


# ------------------------------------------------------------------- driver


def run_stage(out: str, key: str, fn):
    try:
        payload = fn()
    except Exception:
        import traceback
        failure = dict(error=traceback.format_exc()[-4000:])
        print(f"[{key}] FAILED:\n{failure['error']}", flush=True)
        K20.accumulate(out, key + "_failed", failure)
        return None
    K20.accumulate(out, key, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(RECEIPTS / "k21_scan_profile.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--micro-warmup", type=int, default=3)
    parser.add_argument("--micro-iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=10)
    args = parser.parse_args()
    assert args.seed == SEED, "the standing seed is 3401"
    card = _require_lease()
    configure_production()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "K21 needs CUDA"
    RUNS.mkdir(parents=True, exist_ok=True)

    receipt = dict(
        probe="k21_scan_profile",
        question="is the composed scan's 7.28x per-mixer cost INHERENT to "
                 "the slot semantics or INCIDENTAL to a receipts-first "
                 "kernel -- decompose the 23.40 ms forward, bound the "
                 "floor, bracket the bit-identical levers",
        pre_state="no directional prediction; the decomposition IS the "
                  "result",
        unit=UNIT, seed=args.seed,
        session=K20.session_meta(card), git=K20.git_state(),
        bindings=K20.engine_bindings(), profiling_tools=PROFILING_TOOLS,
        recorded_anchors=dict(
            k20_receipt=str(RECEIPTS / "k20_hybrid_velocity.json"),
            mixer_forward_ms_per_window_k20=23.403,
            mixer_backward_ms_per_window_k20=24.554,
            gdn2_mixer_forward_ms=2.352, gdn2_mixer_backward_ms=4.252,
            per_mixer_ratio_k20=7.28,
            composed_scan_forward_ms_banked=22.964699504276116,
            composed_scan_banked_receipt=str(RECEIPTS / "k18_gates.json"),
            pure_ours_step_s_k20=10.624, pure_gdn2_step_s_k20=3.119,
            hybrid_exposure_pct_at_3_of_24_k20=73.1))
    if receipt["git"]["relevant_paths_dirty"]:
        raise ValueError("the committed engine files must be clean before "
                         "profiling them as-committed; dirty: "
                         f"{receipt['git']['relevant_paths_dirty']}")
    K20.dump(args.out, receipt)
    stream = K20.make_stream()

    # ---- the machine, the real activations, the production scan args
    # (not run_stage'd: the capture returns live modules/tensors; failures
    # fail the lane)
    (mixer, x, scan_args, scan_kwargs, cap, shape_note, summary,
     parameters) = build_and_capture(stream)
    K20.accumulate(args.out, "machine", dict(
        build="velocity_ladder._build_k10(TUNED_WINNER) + "
              "convert_model_to_composition + convert_model_to_short_conv("
              "birth='identity') -- the committed K18 machine, K20's "
              "per_layer build verbatim; engines imported, never modified",
        parameters=parameters, summary=summary, capture=cap,
        scan_shape=shape_note, layer=LAYER,
        scan_args_note="the EXACT production arguments layer 6's mixer "
                       "passed to composed_scan during the capture forward "
                       "(a recording shim on the committed mixer's own "
                       "composed_scan binding, removed before any timing)",
        microbatch=MICROBATCH))
    payload, write_keys, query_keys, decay, gates = scan_args
    rank = int(scan_kwargs["rank"])
    seg = int(scan_kwargs["segment_tokens"])
    gated = bool(scan_kwargs["gated"])
    policy = scan_kwargs["policy"]
    policy_f32 = tuple(_round_f32(v) for v in
                       (policy.tau_match, policy.tau_merge, policy.tau_pop,
                        policy.tie_break, policy.read_scale))
    state0 = torch.zeros(payload.shape[0], payload.shape[2], rank,
                         payload.shape[-1], dtype=torch.float32,
                         device=payload.device)
    live0 = torch.zeros(payload.shape[0], payload.shape[2], rank,
                        dtype=torch.bool, device=payload.device)
    keys0 = torch.zeros(payload.shape[0], payload.shape[2], rank,
                        write_keys.shape[-1], dtype=torch.float32,
                        device=payload.device)

    # ---- reproduce: the mixer forward and the composed scan alone
    def reproduce_stage() -> dict:
        def mixer_fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return mixer(x)

        def scan_call():
            return composed_scan(*scan_args, **scan_kwargs)

        mixer_samples = MB.time_launch(mixer_fwd, args.warmup_iters,
                                       args.iters)
        scan_samples = MB.time_launch(scan_call, args.warmup_iters,
                                      args.iters)
        out = dict(
            mixer_forward=K20.spread(mixer_samples),
            composed_scan_forward=K20.spread(scan_samples),
            recorded_reference=dict(
                mixer_forward_ms_k20=23.403,
                composed_scan_forward_ms_banked=22.964699504276116))
        print(f"[reproduce] mixer fwd {statistics.median(mixer_samples):.3f}"
              f" ms, composed_scan {statistics.median(scan_samples):.3f} ms",
              flush=True)
        return out

    reproduce = run_stage(args.out, "reproduce", reproduce_stage)
    if reproduce is None:
        raise SystemExit("the reproduce stage failed; the lane cannot "
                         "continue")

    # ---- the torch-profiler breakdown (per-kernel; the counters note is
    # in profiling_tools)
    def profiler_stage() -> dict:
        def mixer_fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return mixer(x)

        def scan_call():
            return composed_scan(*scan_args, **scan_kwargs)

        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_iters):
                mixer_fwd()
            torch.cuda.synchronize()
        mixer_row = profile_table(prof, RUNS / "profiler_mixer_forward.txt",
                                  args.profile_iters)
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_iters):
                scan_call()
            torch.cuda.synchronize()
        scan_row = profile_table(prof, RUNS / "profiler_scan_only.txt",
                                 args.profile_iters)
        scan_wall = reproduce["composed_scan_forward"]["median_ms"]
        scan_row["eager_gap_ms_per_iter"] = round(
            scan_wall - scan_row["kernel_self_cuda_ms_per_iter"], 3)
        scan_row["eager_gap_note"] = (
            "the composed_scan wrapper's own cost (the zeros state init, "
            "the custom-op dispatch, the output allocations) -- measured as "
            "the wall minus the kernel's self CUDA time; the outputs are "
            "allocator-cached across calls")
        return dict(mixer_forward=mixer_row, scan_only=scan_row)

    profiler = run_stage(args.out, "profiler", profiler_stage)

    # ---- the card's measured peaks, then the analytic cost model
    peaks = run_stage(args.out, "peaks", measure_peaks)
    cost_model = run_stage(args.out, "cost_model", lambda: scan_cost_model(
        shape_note, peaks))

    # ---- the committed reference outputs + the per-pass ablations
    reference = MB.committed_reference(*scan_args, policy_f32, rank, seg,
                                       gated)
    reference_twice = MB.committed_reference(*scan_args, policy_f32, rank,
                                             seg, gated)
    determinism = MB.bit_identity(reference, reference_twice)
    if not determinism["bit_identical"]:
        raise RuntimeError("the committed scan is NOT deterministic across "
                           "runs at fixed inputs -- the bit-identity bar is "
                           "void; escalate")
    K20.accumulate(args.out, "committed_determinism_selfcheck", dict(
        bit_identical=True,
        note="the committed opaque op pair reproduces its own outputs bit "
             "for bit across two runs at fixed inputs -- the receipts bar "
             "is well-posed"))
    run_stage(args.out, "pass_breakdown", lambda: (
        ablation_breakdown(scan_args, scan_kwargs, state0, live0, keys0,
                           policy_f32, reference, args.micro_warmup,
                           args.micro_iters)))

    # ---- the levers (each alone, then stacked), then the launch sweep
    def committed_direct() -> float:
        fn = lambda: _k15_forward_launch(                     # noqa: E731
            payload, write_keys, query_keys, decay, gates, state0, live0,
            keys0, policy_f32, rank, seg, gated, collect=False, warps=8,
            stages=3)
        samples = MB.time_launch(fn, args.micro_warmup, args.micro_iters)
        return statistics.median(samples)

    committed_direct_median = committed_direct()
    brackets = run_stage(args.out, "micro_brackets__levers", lambda: (
        lever_brackets(scan_args, scan_kwargs, state0, live0, keys0,
                       policy_f32, reference, committed_direct_median,
                       args.micro_warmup, args.micro_iters)))
    stacked_flags = (brackets["stacked"]["flags"]
                     if brackets and brackets["stacked"]
                     and brackets["stacked"]["bit_identical"] else None)
    sweep = run_stage(args.out, "micro_brackets__launch_sweep", lambda: (
        launch_config_sweep(scan_args, scan_kwargs, state0, live0, keys0,
                            policy_f32, reference, committed_direct_median,
                            args.micro_warmup, args.micro_iters,
                            stacked_flags)))
    wrapper = run_stage(args.out, "micro_brackets__wrapper",
                        lambda: wrapper_levers(mixer, x, scan_args,
                                               scan_kwargs, policy_f32,
                                               reference, args.micro_warmup,
                                               args.iters))
    statements = run_stage(args.out, "micro_brackets__out_of_scope",
                           out_of_scope_statements)
    K20.accumulate(args.out, "micro_brackets__bar", dict(
        committed_direct_launch_ms=round(committed_direct_median, 3),
        receipts_bar="a lever counts only if every output channel of the "
                     "committed opaque op pair (reads, state, live, keys, "
                     "slot, op, erase, live_after, payload, live_pre, "
                     "ckpt_state, ckpt_live, ckpt_keys) is torch.equal at "
                     "the captured real activations"))

    # ---- the verdict
    k20 = json.loads((RECEIPTS / "k20_hybrid_velocity.json").read_text())
    verdict = run_stage(args.out, "verdict", lambda: verdict_stage(
        reproduce, profiler["scan_only"] if profiler else None,
        peaks, cost_model, brackets if brackets else dict(individual={},
                                                        stacked=None),
        sweep, k20))
    if verdict is None:
        raise SystemExit("the verdict stage failed -- the receipt carries "
                         "the failure")
    print(json.dumps(verdict["measured"], indent=1), flush=True)
    print(json.dumps(verdict["decomposition"], indent=1), flush=True)
    print(json.dumps(verdict["projections"], indent=1), flush=True)
    print(f"receipt written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
