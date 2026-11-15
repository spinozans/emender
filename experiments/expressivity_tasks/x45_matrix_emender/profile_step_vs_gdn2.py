"""Same-session step decomposition: the x45 matrix model vs GDN-2, component by component.

The question this script answers (diagnosis only, no fixes):

    At matched parameters (~150 M) the registered x45 matrix model trains at
    ~9.3 k tok/s while GDN-2 trains at ~28 k tok/s on the same gate-D harness
    (RECORDED receipts; this run re-measures BOTH on the same card in the same
    process).  Fusing the block scan to ~1 ms did not close the gap, and the
    fused path is ~2.4x slower than the eager path at the same microbatch --
    so what is the step actually spending its time on?

Decomposition per model step (mb8, 64 windows, 2048 bytes each, the registered
regime: bf16 autocast, fp32 master weights, fused CE, AdamW):

  (a) block-scan kernels   -- x45: the compiled routed chunk scan (and, for the
                               fused arms, the K4 persistent kernels);
                               GDN-2: the chunk_gdn2 Triton kernels
  (b) routing/schedule      -- x45: route_chunk's similarity/argmax/scatter
                               arithmetic (GDN-2 has none)
  (c) embedding + head/CE  -- shared backbone cost
  (d) optimizer step        -- clip + fused AdamW
  (e) data prepare          -- host-side window grouping (gates protocol)
  (f) host gaps             -- step wall minus device-busy time, plus kernel
                               launch counts

Method, per arm: >=2 warmup steps, then steady steps timed WITHOUT any
profiler (walls + the gates.py phase protocol), then ONE additional steady
step under ``torch.profiler`` whose chrome trace is parsed kernel by kernel
(correlation -> cuda_runtime -> enclosing cpu-op chain) so that every device
microsecond is attributed to the launching operation and phase.  For the
microbatch-2 fused arms (millions of kernel events per step) the profiled
step records CUDA activity only and is aggregated in one pass, and an extra
steady step runs under cProfile for the host-side function attribution.

The routing-vs-scan split INSIDE the compiled mixer graph (routing lives in
the same CompiledFxGraph as the scan arithmetic, so kernel names cannot
separate them) comes from a same-session differential micro-benchmark at the
exact registered shapes:

    routing = compiled(routed_chunk_scan, fwd+bwd)
            - compiled(scan-only over the recorded schedule, fwd+bwd)

with the eager twins of both terms, and an eager route_chunk replay, measured
the same way.  The recorded schedule is produced by the arm's own router
against the arm's own chunk-entry states (a recording wrapper around
``route_chunk``; no model code is touched).

Suites (one process each; partial results accumulate into the receipt):

    core -- gdn2_mb8, x45_registered_mb8, routing_split_microbench
    mb2  -- x45_off_mb2 (eager twin), x45_on_mb2 (fused, the 881 tok/s case),
            x45_on_mb2_compile_route (fused + compiled routing: isolates the
            "routing uncompiled" share of the fused slowdown)

Usage (lease a card the scale-proof lane is NOT holding):

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.profile_step_vs_gdn2 \\
        --suite core --out experiments/expressivity_tasks/x45_matrix_emender/receipts/profile_step_vs_gdn2.json
    .venv/bin/python -m ... --suite mb2 --out .../receipts/profile_step_vs_gdn2.json
"""
from __future__ import annotations

import argparse
import bisect
import cProfile
import io
import json
import os
import pstats
import re
import statistics
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import configure_production
from experiments.expressivity_tasks.x45_matrix_emender import gates
from experiments.expressivity_tasks.x45_matrix_emender.algebra import MatrixSchedule
from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import (
    kernel_count, timed)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    eager_block_scan)
from experiments.expressivity_tasks.x45_matrix_emender.model import (
    COMPILE_MODE, compiled_scan, matrix_config, model_for as matrix_model_for,
    parameter_count)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunk_scan, routed_chunk_scan)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
TRACE_DIR = ROOT / "receipts" / "profile_traces"
# The registered x45 optimizer constants (the registered gdn2 nomination's
# settings; identical to gates.py / bench_scale_proof.py).
LR = 5.42e-4
BETAS = (0.9, 0.9392004369596886)
WEIGHT_DECAY = 0.20059436743465126
# RECORDED reference points this run re-measures (not same-session there).
RECORDED = {
    "gdn2_tokens_per_s": 28124.0,
    "matrix_ops_tokens_per_s": 13189.7,
    "x45_registered_mb8_tokens_per_s_steady": 9306.3,
    "x45_on_mb2_tokens_per_s_steady": 880.8,
    "x45_off_mb2_tokens_per_s_steady": 2110.7,
    "source": ("receipts/cd_gdn2_c128_rerun.json, receipts/cd_matrix_ops_c128.json, "
               "receipts/scale_proof_model_step.json -- RECORDED, not same-session"),
}
BOUNDARY = ("whole model, whole step: prepare + all microbatches' forward and "
            "backward + clip/AdamW; flagship-stream read excluded; tokens are "
            "measured targets over the same 64-window stream batches for every "
            "arm (the gates.py/cd_* protocol)")

GEMM_KERNEL = re.compile(r"ampere_|cutlass|cublas|gemm|s16816|s1688|sgemm|nvjet|sm90")
MATMUL_OP = re.compile(r"aten::(mm|bmm|addmm|matmul)$")


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2**30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__,
                ndm_pin_triton_autotune=os.environ.get("NDM_PIN_TRITON_AUTOTUNE", "1"))


def dump(out_path: str, report: dict) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def accumulate(out_path: str, key: str, payload: dict) -> dict:
    """Merge one component's result into the shared receipt as it lands."""
    path = Path(out_path)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[key] = payload
    dump(out_path, data)
    return data


def make_batches():
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    return iter(stream.batches("train"))


# --------------------------------------------------------------- step protocol


def one_step(model, optimizer, batches, microbatch: int, annotate: bool = False):
    """The gates.py one_step, with optional record_function phase annotations.

    Phase labels are gates' own: ``prepare`` is host-side window grouping,
    ``fwd_bwd`` is every microbatch's forward AND backward, ``opt`` is clip +
    AdamW.  The flagship-stream read is outside every phase.  ``annotate`` adds
    ``PHASE_*`` record_function ranges (used only inside the profiled step).
    """
    rows = next(batches)
    count = sum(len(r[3]) for r in rows)
    optimizer.zero_grad(set_to_none=True)
    phases = {"prepare": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
    rf = (lambda name: torch.autograd.profiler.record_function(name)) if annotate \
        else (lambda name: contextlib_null())
    torch.cuda.synchronize()
    mark = time.monotonic()
    with rf("PHASE_PREPARE"):
        groups = list(grouped(rows, microbatch))
    torch.cuda.synchronize()
    phases["prepare"] = time.monotonic() - mark
    mark = time.monotonic()
    loss_sum = 0.0
    for x, y, valid in groups:
        torch.compiler.cudagraph_mark_step_begin()
        with rf("PHASE_FWD"):
            loss = model.loss(x, y)
        with rf("PHASE_BWD"):
            (loss * (valid / count)).backward()
        loss_sum += float(loss.detach()) * valid / count
    torch.cuda.synchronize()
    phases["fwd_bwd"] = time.monotonic() - mark
    mark = time.monotonic()
    with rf("PHASE_OPT"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    phases["opt"] = time.monotonic() - mark
    return count, loss_sum, phases


class contextlib_null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def dynamo_snapshot() -> dict:
    try:
        from torch._dynamo.utils import counters
        return {str(k): dict(v) for k, v in dict(counters).items()}
    except Exception as error:                                  # pragma: no cover
        return {"error": repr(error)[:200]}


def cprofile_step(model, optimizer, batches, microbatch: int) -> dict:
    """One steady step under cProfile: host-side python attribution only."""
    profiler = cProfile.Profile()
    began = time.monotonic()
    profiler.enable()
    one_step(model, optimizer, batches, microbatch)
    profiler.disable()
    wall = time.monotonic() - began
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(60)
    text = buffer.getvalue()
    rows = []
    for line in text.splitlines():
        if "profile_step_vs_gdn2" in line or "x45_matrix_emender" in line \
                or "autograd" in line or "dynamo" in line or "linear.py" in line \
                or "fla/" in line:
            rows.append(line.strip()[:220])
    # The unfiltered top-15 rows too, so the dominant host frames are always kept.
    top = [line.strip()[:220] for line in text.splitlines()[4:22]]
    return dict(boundary="one steady training step under cProfile (host python only; "
                         "the wall here is inflated by the tracer and is NOT the "
                         "arm's tokens/s)",
                wall_s=round(wall, 3),
                interesting_frames=rows[:60], top_frames=top)


# ------------------------------------------------------- trace decomposition


def classify_component(chain_names: list[str], kernel_name: str) -> str:
    """Bucket one kernel by its launching op chain (innermost first) and name."""
    chain = " ".join(chain_names).lower()
    kernel = kernel_name.lower()
    innermost = chain_names[0].lower() if chain_names else ""
    if "trsm" in kernel or "linalg_solve_triangular" in innermost:
        return "a_scan_triangular_solve"
    if ("chunkgdn2function" in chain or "gdn2" in kernel or "chunk_kda" in kernel
            or "causal_conv" in kernel or "solve_tril" in kernel
            or "recompute_w_u" in kernel or "l2norm" in kernel):
        return "a_mixer_gdn2_kernels"
    if "cross_entropy" in chain or "cross_entropy" in kernel:
        return "c_head_cross_entropy"
    if "compiledfxgraph" in chain:
        if MATMUL_OP.search(innermost) or GEMM_KERNEL.search(kernel):
            return "a_mixer_graph_matmul"
        if "compiledfxgraph" in innermost:
            return "a_mixer_graph_fused_triton"
        return "a_mixer_graph_aten"
    if "persistent_scan_trajectory" in kernel:
        return "a_fused_scan_trajectory_fwd"
    if "persistent_bwd" in kernel or "persistent_scan_backward" in kernel:
        return "a_fused_scan_bwd"
    if "embedding" in chain or "embedding" in kernel:
        return "c_embedding"
    if "rms_norm" in kernel or "rmsnorm" in chain:
        return "backbone_norms"
    if "adam" in kernel or "adam" in chain:
        return "d_optimizer"
    if GEMM_KERNEL.search(kernel) or MATMUL_OP.search(innermost):
        return "c_matmul_backbone"
    if "memcpy" in kernel or "memset" in kernel:
        return "memcpy"
    return "z_other_elementwise"


def parse_trace(path: Path) -> dict:
    """Attribute every device kernel of one profiled step to launcher + phase.

    A kernel event carries ``args.correlation``; the cuda_runtime event with
    the same correlation launched it; the innermost cpu_op/user_annotation
    interval containing that runtime event (same pid/tid) is the launching op,
    and its containment chain gives the phase (PHASE_OPT / PHASE_PREPARE
    annotations; everything else is fwd_bwd).
    """
    events = json.loads(Path(path).read_text())["traceEvents"]
    cpu_intervals = {}
    for event in events:
        if (event.get("cat") in ("cpu_op", "user_annotation")
                and isinstance(event.get("ts"), (int, float))
                and isinstance(event.get("dur"), (int, float))):
            cpu_intervals.setdefault((event["pid"], event["tid"]), []).append(event)
    parents = {}
    starts = {}
    for key, intervals in cpu_intervals.items():
        intervals.sort(key=lambda e: e["ts"])
        starts[key] = [e["ts"] for e in intervals]
        stack = []
        chain_parents = {}
        for event in intervals:                      # properly nested by construction
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]:
                stack.pop()
            chain_parents[id(event)] = stack[-1] if stack else None
            stack.append(event)
        parents[key] = chain_parents

    def chain_for(pid: int, tid: int, ts: float) -> list[str]:
        key = (pid, tid)
        intervals = cpu_intervals.get(key)
        if not intervals:
            return []
        index = bisect.bisect_right(starts[key], ts) - 1
        innermost = None
        while index >= 0:                     # rightmost start <= ts that contains ts
            event = intervals[index]
            if event["ts"] <= ts < event["ts"] + event["dur"]:
                innermost = event
                break
            index -= 1
        chain = []
        node = innermost
        chain_parents = parents[key]
        while node is not None and len(chain) < 40:                # innermost first
            chain.append(node["name"])
            node = chain_parents.get(id(node))
        return chain

    correlation_runtime = {}
    for event in events:
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None and correlation not in correlation_runtime:
                correlation_runtime[correlation] = event

    kernels = [event for event in events
               if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    busy_us = 0.0
    streams = set()
    per_component = {}
    per_phase_component = {}
    per_kernel = {}
    per_launcher = {}
    unattributed = 0.0
    for kernel in kernels:
        duration = kernel.get("dur") or 0.0
        busy_us += duration
        streams.add((kernel.get("pid"), (kernel.get("args") or {}).get("stream")))
        correlation = (kernel.get("args") or {}).get("correlation")
        runtime = correlation_runtime.get(correlation)
        chain = (chain_for(runtime["pid"], runtime["tid"], runtime["ts"])
                 if runtime else [])
        phase = ("e_prepare" if any("PHASE_PREPARE" in name for name in chain)
                 else "d_opt" if any("PHASE_OPT" in name for name in chain)
                 else "b_fwd_bwd")
        component = classify_component(chain, kernel.get("name", "?"))
        per_component[component] = per_component.get(component, 0.0) + duration
        per_phase_component.setdefault(phase, {})
        per_phase_component[phase][component] = \
            per_phase_component[phase].get(component, 0.0) + duration
        if not chain:
            unattributed += duration
        row = per_kernel.setdefault(kernel.get("name", "?")[:110],
                                    [0.0, 0])
        row[0] += duration
        row[1] += 1
        innermost = chain[0] if chain else "<no launching op>"
        row = per_launcher.setdefault(innermost[:110], [0.0, 0])
        row[0] += duration
        row[1] += 1
    top_kernels = sorted((dict(name=name, device_ms=round(row[0] / 1000, 3),
                              calls=row[1])
                         for name, row in per_kernel.items()),
                        key=lambda row: -row["device_ms"])[:25]
    top_launchers = sorted((dict(name=name, device_ms=round(row[0] / 1000, 3),
                                 calls=row[1])
                            for name, row in per_launcher.items()),
                           key=lambda row: -row["device_ms"])[:25]
    return dict(
        kernels=len(kernels),
        device_busy_ms=round(busy_us / 1000, 3),
        device_streams=sorted(str(s) for s in streams),
        unattributed_device_ms=round(unattributed / 1000, 3),
        components_ms={k: round(v / 1000, 3) for k, v in sorted(per_component.items())},
        phase_components_ms={phase: {k: round(v / 1000, 3)
                                     for k, v in sorted(rows.items())}
                             for phase, rows in sorted(per_phase_component.items())},
        top_kernels=top_kernels, top_launchers=top_launchers)


def analyze_profiled_step(prof, trace_path: Path) -> dict:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    return parse_trace(trace_path)


def analyze_cuda_only(prof) -> dict:
    """One pass over the raw event list (the microbatch-2 fused event volume
    makes key_averages and the chrome trace parser infeasible in a session)."""
    busy_us = 0.0
    events = 0
    per_name = {}
    per_component = {}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        device = getattr(event, "self_device_time_total", 0) or 0
        busy_us += device
        events += 1
        name = event.name[:110]
        row = per_name.setdefault(name, [0.0, 0])
        row[0] += device
        row[1] += 1
        component = classify_component([name], name)
        per_component[component] = per_component.get(component, 0.0) + device
    top_kernels = sorted((dict(name=name, device_ms=round(row[0] / 1000, 3),
                              calls=row[1]) for name, row in per_name.items()),
                        key=lambda row: -row["device_ms"])[:25]
    return dict(
        kernels=events, device_busy_ms=round(busy_us / 1000, 3),
        note=("CUDA-activity-only profile of one steady step; component buckets "
              "are name-classified (no launching-op chains at this event volume)"),
        components_ms={k: round(v / 1000, 3)
                       for k, v in sorted(per_component.items())},
        top_kernels=top_kernels)


# ------------------------------------------------------------------ the arms


def build_gdn2(microbatch: int):
    model, lr, betas, weight_decay, summary = gates.build(
        "gdn2", microbatch, 128, None, False, 64, 3401)
    return model, lr, betas, weight_decay, summary


X45_CONFIGS = {
    "x45_registered_mb8": dict(microbatch=8, compile_scan=True, fused_backward=False,
                               description=("registered path: torch.compile(mode=default) "
                                            "on the one shared scan, autograd through the "
                                            "compiled chunk arithmetic; the 9,306 tok/s arm")),
    "x45_off_mb2": dict(microbatch=2, compile_scan=True, fused_backward=False,
                        description=("registered path at microbatch 2 (the fused path's "
                                     "eager twin, same session)")),
    "x45_on_mb2": dict(microbatch=2, compile_scan=False, fused_backward=True,
                       description=("fused K4 path at microbatch 2 (the 881 tok/s arm): "
                                    "one persistent autograd node per chunk, routing and "
                                    "everything else eager (no torch.compile anywhere)")),
    "x45_on_mb2_compile_route": dict(microbatch=2, compile_scan=True, fused_backward=True,
                                     description=("fused K4 path at microbatch 2 WITH the "
                                                  "routing compiled: the compiled scan "
                                                  "graph-breaks at the fused node and keeps "
                                                  "compiling the routing around it -- this "
                                                  "arm isolates the 'routing uncompiled' "
                                                  "share of the fused slowdown")),
}


def build_x45(label: str):
    config = X45_CONFIGS[label]
    torch.manual_seed(3401)
    torch.cuda.manual_seed_all(3401)
    shape = matrix_config()
    policy = RoutingPolicy(ops=True)
    model = matrix_model_for(shape, policy, 64, compile_scan=config["compile_scan"],
                              fused_backward=config["fused_backward"]).cuda().train()
    summary = dict(shape=shape.__dict__, parameters=parameter_count(shape),
                   routing=policy.record(), chunk_tokens=64)
    return model, LR, BETAS, WEIGHT_DECAY, summary


def measure_arm(label: str, build_fn, microbatch: int, out: str, warmup: int,
                wall_steps: int, profile_mode: str, do_cprofile: bool = True) -> dict:
    print(f"[{label}] building ...", flush=True)
    model, lr, betas, weight_decay, summary = build_fn()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, betas=betas, weight_decay=weight_decay,
                                  fused=True)
    batches = make_batches()
    for index in range(warmup):
        began = time.monotonic()
        one_step(model, optimizer, batches, microbatch)
        print(f"[{label}] warmup {index + 1}: {time.monotonic() - began:.1f} s",
              flush=True)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for _ in range(wall_steps):
        count, loss, phases = one_step(model, optimizer, batches, microbatch)
        records.append(dict(tokens=count, loss=round(loss, 6),
                            wall_s=round(sum(phases.values()), 3),
                            phases={k: round(v, 3) for k, v in phases.items()}))
        print(f"[{label}] steady {len(records)}: {records[-1]['wall_s']:.1f} s "
              f"(fwd_bwd {records[-1]['phases']['fwd_bwd']:.1f} s)", flush=True)
    total_s = sum(record["wall_s"] for record in records)
    tokens = sum(record["tokens"] for record in records)
    steady = records[1:] if len(records) > 2 else records
    steady_s = sum(record["wall_s"] for record in steady)
    steady_tokens = sum(record["tokens"] for record in steady)
    phases_total = {key: round(sum(record["phases"][key] for record in records), 3)
                    for key in ("prepare", "fwd_bwd", "opt")}

    profile = None
    if profile_mode == "full":
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU],
            record_shapes=False, profile_memory=False)
        prof.start()
        one_step(model, optimizer, batches, microbatch, annotate=True)
        prof.stop()
        profile = analyze_profiled_step(prof, TRACE_DIR / f"{label}_step.json")
        print(f"[{label}] profile: busy {profile['device_busy_ms'] / 1000:.2f} s, "
              f"{profile['kernels']} kernels", flush=True)
    elif profile_mode == "cuda":
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False)
        prof.start()
        one_step(model, optimizer, batches, microbatch)
        prof.stop()
        profile = analyze_cuda_only(prof)
        print(f"[{label}] profile: busy {profile['device_busy_ms'] / 1000:.2f} s, "
              f"{profile['kernels']} kernels", flush=True)

    cprofile = cprofile_step(model, optimizer, batches, microbatch) if do_cprofile else None

    report = dict(
        label=label,
        boundary=BOUNDARY,
        precision="bf16 compute (autocast), fp32 master weights, fused CE -- the registered regime",
        session="same-session (this card, this process)",
        parameters=parameters, summary=summary, microbatches=microbatch,
        windows_per_step=64, context=2048, warmup=warmup,
        steps=len(records), tokens=tokens,
        step_walls=[record["wall_s"] for record in records],
        phase_s=phases_total,
        ms_per_step_p50=round(1000 * statistics.median(
            [record["wall_s"] for record in records]), 1),
        total_s=round(total_s, 3),
        tokens_per_s=(round(tokens / total_s, 1) if total_s else None),
        tokens_per_s_steady=(round(steady_tokens / steady_s, 1) if steady_s else None),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        losses=[record["loss"] for record in records],
        step_profile=profile, cprofile_step=cprofile,
        dynamo_counters=dynamo_snapshot(),
        recorded_references=RECORDED)
    del model, optimizer
    torch.cuda.empty_cache()
    accumulate(out, label, report)
    return report


# ------------------------------------------- routing/scan differential microbench


def _record_window(query, payload, decay, policy: RoutingPolicy, rank: int,
                   chunk_tokens: int):
    """Run the arm's own eager routing once, recording every chunk's inputs.

    The recording router calls the real ``route_chunk`` and stores the
    chunk-entry state/liveness and its outputs, so the scan-only replay below
    consumes an identical decision channel (no measured path pays for routing
    inside it).  Returns (records, weights_full, decay_headmajor, schedule_full).
    """
    records = []

    def router(index, token_offset, state, live, qc, pc, pol, rk):
        schedule, weights = route_chunk(index, token_offset, state, live, qc, pc,
                                        pol, rk)
        records.append(dict(state=state.detach().clone(),
                            live=live.detach().clone(),
                            schedule=schedule, weights=weights))
        return schedule, weights

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=rank, rank_max=rank, chunk_tokens=chunk_tokens,
                          fused=False)
    weights_full = torch.cat([record["weights"] for record in records], dim=2)
    schedule_full = MatrixSchedule(
        slot=torch.cat([record["schedule"].slot for record in records], dim=1),
        erase=torch.cat([record["schedule"].erase for record in records], dim=1),
        payload=torch.cat([record["schedule"].payload for record in records], dim=1),
        op=torch.cat([record["schedule"].op for record in records], dim=1),
        live_after=torch.cat([record["schedule"].live_after for record in records],
                             dim=1))
    decay_headmajor = decay.permute(0, 2, 1).contiguous()
    return records, weights_full, decay_headmajor, schedule_full


def measure_routing_split(model, batches, microbatch: int, out: str,
                          chunk_tokens: int = 64) -> dict:
    """Routing vs scan arithmetic at the exact registered shapes, same session.

    Everything is measured at ONE microbatch's mixer shape (the model's own
    block-0 mixer, fed a real captured activation), then scaled by the
    per-step scan count (depth x microbatches_per_step).
    """
    mixer = model.blocks[0].mixer
    captured = {}

    def hook(module, inputs, output):
        captured.setdefault("x", inputs[0].detach())

    handle = mixer.register_forward_hook(hook)
    rows = next(batches)
    count = sum(len(r[3]) for r in rows)
    x_ids, y_ids, _ = next(iter(grouped(rows, microbatch)))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.loss(x_ids, y_ids)
    handle.remove()
    x = captured["x"]
    batch, tokens, width = x.shape
    policy = mixer.policy
    rank, rank_max = mixer.rank, mixer.rank_max
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = mixer.query(x).view(batch, tokens, mixer.heads, mixer.value_dim)
        payload = mixer.value(x).view(batch, tokens, mixer.heads, mixer.value_dim)
        decay = torch.sigmoid(mixer.decay(x)).clamp(min=1e-4, max=1.0 - 1e-6)
    q = query.detach().requires_grad_(True)
    p = payload.detach().requires_grad_(True)
    d = decay.detach().requires_grad_(True)

    records, weights_full, decay_hm, schedule_full = _record_window(
        q, p, d, policy, rank, chunk_tokens)
    state0 = torch.zeros(batch, mixer.heads, rank_max, mixer.value_dim,
                         dtype=torch.float32, device="cuda")
    live0 = torch.zeros(batch, mixer.heads, rank_max, dtype=torch.bool, device="cuda")
    chunks = len(records)
    scans_per_step = len(model.blocks) * (64 // microbatch)

    def fresh(*tensors):
        return tuple(t.detach().clone().requires_grad_(True) for t in tensors)

    def eager_full():
        cq, cp, cd = fresh(q, p, d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = routed_chunk_scan(cq, cp, cd, router=route_chunk, policy=policy,
                                    rank=rank, rank_max=rank_max,
                                    chunk_tokens=chunk_tokens, fused=False)
        (result.reads.sum() + result.state.sum()).backward()

    def eager_scan_only():
        (cw,) = fresh(weights_full)
        reads, state, live = eager_block_scan(state0, live0, cw, decay_hm,
                                             schedule_full, chunk_tokens)
        (reads.sum() + state.sum()).backward()

    # A routing-only replay is deliberately NOT measured: the recorded products
    # (read weights) are outputs, not inputs, of route_chunk, and replaying the
    # router against recorded chunk-entry states measures the same arithmetic the
    # eager_full - eager_scan_only difference already isolates.  The two full
    # terms below bound and split the routing cost without a third replay.

    def compiled_full():
        cq, cp, cd = fresh(q, p, d)
        scan = compiled_scan(COMPILE_MODE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = scan(cq, cp, cd, router=route_chunk, policy=policy, rank=rank,
                          rank_max=rank_max, chunk_tokens=chunk_tokens, fused=False,
                          trainable_fused=False)
        (result.reads.sum() + result.state.sum()).backward()

    def make_scan_only_runner():
        slot = schedule_full.slot
        erase = schedule_full.erase
        payload = schedule_full.payload
        op = schedule_full.op
        live_after = schedule_full.live_after
        tokens_count = schedule_full.slot.shape[1]

        def _scan_only(weights, decay, state, live):
            reads = torch.empty(batch, mixer.heads, tokens_count, mixer.value_dim,
                                dtype=torch.float32, device=state.device)
            for start in range(0, tokens_count, chunk_tokens):
                stop = start + chunk_tokens
                piece = MatrixSchedule(
                    slot=slot[:, start:stop], erase=erase[:, start:stop],
                    payload=payload[:, start:stop], op=op[:, start:stop],
                    live_after=live_after[:, start:stop])
                chunk_reads, state, live = chunk_scan(
                    state, live, weights[:, :, start:stop], decay[:, :, start:stop],
                    piece, fused=False)
                reads[:, :, start:stop] = chunk_reads
            return reads, state, live

        return _scan_only

    scan_only_form = "loop"
    try:
        compiled_scan_only_fn = torch.compile(make_scan_only_runner(), dynamic=False)
        # force the (lazy) compile now so a failure lands in this guarded block
        with torch.no_grad():
            compiled_scan_only_fn(weights_full, decay_hm, state0, live0)

        def compiled_scan_only():
            (cw,) = fresh(weights_full)
            reads, state, _ = compiled_scan_only_fn(cw, decay_hm, state0.clone(),
                                                   live0.clone())
            (reads.sum() + state.sum()).backward()
    except Exception as error:                                  # pragma: no cover
        scan_only_form = f"per-chunk fallback after {repr(error)[:160]}"
        compiled_chunk = torch.compile(chunk_scan, dynamic=False)

        def compiled_scan_only():
            (cw,) = fresh(weights_full)
            reads = torch.empty(batch, mixer.heads, tokens, mixer.value_dim,
                                dtype=torch.float32, device="cuda")
            state, live = state0.clone(), live0.clone()
            for start in range(0, tokens, chunk_tokens):
                stop = start + chunk_tokens
                chunk_reads, state, live = compiled_chunk(
                    state, live, cw[:, :, start:stop], decay_hm[:, :, start:stop],
                    schedule_full.chunk(start, stop), fused=False)
                reads[:, :, start:stop] = chunk_reads
            (reads.sum() + state.sum()).backward()

    def timed_row(fn, repeats, label):
        row = dict(timed(fn, repeats=repeats, warmup=2, label=label))
        row["kernels"] = kernel_count(fn, repeats=1)
        return row

    print(f"[routing_split] eager_full (b{batch}/T{tokens}/c{chunk_tokens}) ...",
          flush=True)
    eager_full_row = timed_row(eager_full, 3, "eager_full")
    eager_scan_row = timed_row(eager_scan_only, 3, "eager_scan_only")
    print("[routing_split] compiled_full ...", flush=True)
    compiled_full_row = timed_row(compiled_full, 5, "compiled_full")
    print("[routing_split] compiled_scan_only ...", flush=True)
    began = time.monotonic()
    compiled_scan_row = timed_row(compiled_scan_only, 5, "compiled_scan_only")
    print(f"[routing_split] scan-only compile+measure took "
          f"{time.monotonic() - began:.0f} s", flush=True)

    def ms(row, key="latency_ms_p50"):
        return row.get(key) or row.get("device_ms_mean") or 0.0

    routing_eager = ms(eager_full_row) - ms(eager_scan_row)
    routing_compiled = ms(compiled_full_row) - ms(compiled_scan_row)
    report = dict(
        label="routing_split_microbench",
        boundary=("ONE mixer scan call at the registered microbatch shape "
                  "(batch as stated, 2048 tokens, chunk 64, rank 128): forward AND "
                  "backward, attached read weights and payloads; routing = "
                  "routed_chunk_scan minus the eager chunk scan over the SAME "
                  "recorded schedule.  x is a real captured block-0 activation; "
                  "the schedule was recorded by the arm's own router against its "
                  "own chunk-entry states."),
        session="same-session (this card, this process)",
        shape=dict(batch=batch, tokens=tokens, heads=mixer.heads,
                   value_dim=mixer.value_dim, rank=rank, rank_max=rank_max,
                   chunk_tokens=chunk_tokens, chunks_per_scan=chunks),
        scans_per_step=scans_per_step,
        scan_only_form=scan_only_form,
        eager_full=eager_full_row, eager_scan_only=eager_scan_row,
        compiled_full=compiled_full_row, compiled_scan_only=compiled_scan_row,
        routing_ms_per_scan=dict(
            eager=round(routing_eager, 3),
            compiled=round(routing_compiled, 3),
            note=("routing + read-weight backward + scan-loop glue; the scan-only "
                  "term consumes the recorded schedule, so this is the router's "
                  "whole cost at this shape")),
        scan_ms_per_scan=dict(
            eager=round(ms(eager_scan_row), 3),
            compiled=round(ms(compiled_scan_row), 3)),
        per_step_seconds=dict(
            routing_eager=round(routing_eager * scans_per_step / 1000, 3),
            routing_compiled=round(routing_compiled * scans_per_step / 1000, 3),
            scan_eager=round(ms(eager_scan_row) * scans_per_step / 1000, 3),
            scan_compiled=round(ms(compiled_scan_row) * scans_per_step / 1000, 3),
            full_compiled=round(ms(compiled_full_row) * scans_per_step / 1000, 3)),
        recorded_references=RECORDED)
    accumulate(out, "routing_split_microbench", report)
    return report


# ------------------------------------------------------------------- the suites


def run_stage(out: str, key: str, fn):
    """Run one stage, banking either its result or its failure into the receipt."""
    try:
        return fn()
    except Exception:
        failure = dict(error=traceback.format_exc()[-4000:])
        print(f"[{key}] FAILED:\n{failure['error']}", flush=True)
        return accumulate(out, key + "_failed", failure)


def suite_core(args) -> dict:
    torch.set_num_threads(1)
    data = {}
    data["session"] = session_meta()
    dump(args.out, data)
    run_stage(args.out, "gdn2_mb8", lambda: measure_arm(
        "gdn2_mb8", lambda: build_gdn2(8), 8, args.out,
        warmup=args.warmup, wall_steps=args.steps,
        profile_mode="full", do_cprofile=True))
    run_stage(args.out, "x45_registered_mb8", lambda: measure_arm(
        "x45_registered_mb8", lambda: build_x45("x45_registered_mb8"),
        8, args.out, warmup=args.warmup, wall_steps=args.steps,
        profile_mode="full", do_cprofile=True))
    # The routing split needs a live registered model: rebuild block 0's mixer
    # against a captured activation (a fresh model is identical to the one just
    # measured -- same seed, same construction).
    def routing_split():
        model, _, _, _, _ = build_x45("x45_registered_mb8")
        batches = make_batches()
        report = measure_routing_split(model, batches, 8, args.out)
        del model
        torch.cuda.empty_cache()
        return report
    run_stage(args.out, "routing_split_microbench", routing_split)
    return json.loads(Path(args.out).read_text())


def suite_routing(args) -> dict:
    """Rerun only the routing/scan differential (the model arms are banked)."""
    torch.set_num_threads(1)
    if not Path(args.out).exists():
        dump(args.out, {})
    data = json.loads(Path(args.out).read_text())
    if "session" not in data:
        data["session"] = session_meta()
        dump(args.out, data)

    def routing_split():
        model, _, _, _, _ = build_x45("x45_registered_mb8")
        batches = make_batches()
        report = measure_routing_split(model, batches, 8, args.out)
        del model
        torch.cuda.empty_cache()
        return report
    run_stage(args.out, "routing_split_microbench", routing_split)
    return json.loads(Path(args.out).read_text())


def suite_mb2(args) -> dict:
    torch.set_num_threads(1)
    data = {}
    if Path(args.out).exists():
        data = json.loads(Path(args.out).read_text())
    if "session" not in data:
        data["session"] = session_meta()
        dump(args.out, data)
    for label in ("x45_off_mb2", "x45_on_mb2", "x45_on_mb2_compile_route"):
        config = X45_CONFIGS[label]
        run_stage(args.out, label, lambda label=label, config=config: measure_arm(
            label, lambda label=label: build_x45(label),
            config["microbatch"], args.out, warmup=args.warmup,
            wall_steps=args.steps, profile_mode="cuda", do_cprofile=True))
    return json.loads(Path(args.out).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "mb2", "routing"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3,
                        help="steady wall steps per arm (the fused mb2 arm is ~133 s/step)")
    parser.add_argument("--warmup", type=int, default=2)
    arguments = parser.parse_args()
    configure_production()
    if arguments.suite == "core":
        suite_core(arguments)
    elif arguments.suite == "routing":
        suite_routing(arguments)
    else:
        suite_mb2(arguments)
    print(f"receipt written to {arguments.out}")


if __name__ == "__main__":
    main()
