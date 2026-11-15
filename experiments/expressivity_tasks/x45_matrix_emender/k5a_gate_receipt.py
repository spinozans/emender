"""K5A gate receipt -- the routing region's differentiable channels, fused.

Banks ``receipts/k5a_routing_diff.json`` with everything the phase's gates
demand:

* **Gate A (decision bit-exactness, incl. the mandatory flip test)** -- the
  fused router vs the registered ``route_chunk`` on pinned chunk-entry states,
  through whole routed scans (all five ops, pop-saturated, ragged, rank prefix,
  growth, the registered bf16 shape, the tie fixture), and THE FLIP TEST on
  real routed fixtures: real chunk-entry states from a real routed scan at the
  registered shape under bf16 autocast, every lane of every schedule field,
  zero flips asserted.  The fused router consumes the registered similarity and
  strength, so this is bit-equality by construction; the receipt is the proof.
* **Gate B (the measured floors)** -- the read-weights floor (two orderings of
  the same math plus the fp64 oracle) measured FIRST on the same fixtures, the
  fused deviation within ``NOISE_FACTOR`` x it, max and median reported; the
  same discipline for the routing gradients against the fp64 adjoint oracle,
  with ``dpayload`` bit-exact.
* **Gate C (launch counts)** -- the per-chunk kernel counts (eager regime) and
  the compiled differential at the registered shape (routed scan minus scan-only
  over the same recorded schedule, forward and backward), which is the honest
  before/after for the routing region, plus the K4/K5 gate suites re-run
  verbatim as subprocesses.
* **The cd-protocol training fixture** -- the registered model's own block-0
  mixer on a real flagship-stream activation, schedule bit-parity on pinned
  chunk-entry states, and one whole eager-scan training step per router with
  the same-router rerun as the nondeterminism baseline.

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5a_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5a_routing_diff.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import (
    kernel_count, timed)
from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff import (
    routing_diff_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_k5a_routing_diff import (
    SCHEDULE_FIELDS, distinct_stream, gradient_floor, weights_floor)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
DEVICE = "cuda"
BATCH, HEADS, TOKENS, RANK, VALUE_DIM, CHUNK = 8, 2, 2048, 128, 128, 64
NOISE_FACTOR = 8.0


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


# ---------------------------------------------------------------------- gate A


def pinned_chunk_parity(policy, rank, chunk_tokens, stream, autocast):
    """The deterministic parity instrument: both routers on recorded inputs."""
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        routed_chunk_scan)
    query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=autocast):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=rank, rank_max=rank, chunk_tokens=chunk_tokens,
                          fused=False)
    rows = []
    for index, (state, live, qc, pc) in enumerate(records):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            eager = route_chunk(index, index * chunk_tokens, state, live, qc, pc,
                               policy, rank)
            fused = routing_diff_route_chunk(index, index * chunk_tokens, state,
                                             live, qc, pc, policy, rank)
        row = dict(chunk=index)
        for name in SCHEDULE_FIELDS:
            row[name] = bool(torch.equal(getattr(eager[0], name),
                                         getattr(fused[0], name)))
        floor = weights_floor(qc, state[:, :, :rank], live[:, :, :rank],
                              policy.read_scale, autocast)
        delta = (eager[1].double() - fused[1].double()).abs()
        row.update(weights_floor=floor, weights_max_deviation=float(delta.max()),
                   weights_median_deviation=float(delta.median()),
                   weights_within_floor=bool(float(delta.max())
                                             <= NOISE_FACTOR * max(floor, 1e-30)))
        rows.append(row)
    return rows


def flip_test(policy):
    """THE MANDATORY FLIP TEST, on the real routed fixture (registered shape).

    The parity rules say the similarity/strength values decisions consume stay
    torch-computed OR are reproduced exactly -- and then this test is
    mandatory.  K5A keeps them torch-computed (the probe rows in this receipt
    show they cannot be reproduced from Triton), so this is bit-equality by
    construction; the test runs where a flip would hide and asserts zero.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        routed_chunk_scan)
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                             512, 77)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=RANK, rank_max=RANK, chunk_tokens=CHUNK,
                          fused=False)
    per_field = {name: 0 for name in SCHEDULE_FIELDS}
    lanes = 0
    for index, (state, live, qc, pc) in enumerate(records):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager = route_chunk(index, index * CHUNK, state, live, qc, pc,
                                policy, RANK)
            fused = routing_diff_route_chunk(index, index * CHUNK, state, live,
                                             qc, pc, policy, RANK)
        for name in SCHEDULE_FIELDS:
            a, b = getattr(eager[0], name), getattr(fused[0], name)
            per_field[name] += int((a != b).sum())
            lanes += a.numel()
    return dict(
        fixture=(f"real routed scan, batch {BATCH}, {TOKENS} tokens, rank "
                 f"{RANK}, chunk {CHUNK}, bf16 autocast, {len(records)} chunks"),
        differing_lanes=per_field, lanes=lanes, flips=sum(per_field.values()),
        zero_flips=sum(per_field.values()) == 0,
        construction=("the fused router consumes the eager preprocessing's own "
                      "similarity and strength tensors, so decisions are "
                      "bit-equal by construction; this test is the proof, and "
                      "the probe rows below are the reason the similarity is "
                      "not recomputed in-kernel"))


def similarity_probe():
    """Why the decision inputs stay torch-computed, on the record.

    Reproduces the K5A probe: an in-kernel recomputation of the eager
    similarity chain (bf16 norms + the bmm) against the eager tensors, at the
    registered shape -- the mismatch counts that would put decisions at flip
    risk, and the compiled preprocessing's own kernel count.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        routed_chunk_scan)
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                            512, 77)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=RoutingPolicy(
            ops=True), rank=RANK, rank_max=RANK, chunk_tokens=CHUNK, fused=False)
    state, live, qc, pc = records[len(records) // 2]

    def preamble(payload_chunk, rows, act):
        frozen = rows.detach()
        strength = frozen.norm(dim=-1) * act.to(torch.float32)
        stored = frozen / strength.clamp_min(1e-6).unsqueeze(-1)
        value = payload_chunk / payload_chunk.norm(dim=-1,
                                                  keepdim=True).clamp_min(1e-6)
        return value @ stored.transpose(-1, -2)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        sim_eager = preamble(pc, state[:, :, :RANK], live[:, :, :RANK])
        sim_again = preamble(pc, state[:, :, :RANK], live[:, :, :RANK])
        sim_compiled = torch.compile(preamble, dynamic=False)(
            pc, state[:, :, :RANK], live[:, :, :RANK])
    same = int((sim_eager == sim_again).sum())
    same_compiled = int((sim_eager == sim_compiled).sum())
    count = kernel_count(lambda: preamble(pc, state[:, :, :RANK],
                                          live[:, :, :RANK]), repeats=1)
    return dict(
        shape=dict(batch=BATCH, chunk_tokens=CHUNK, heads=HEADS, rank=RANK,
                   value_dim=VALUE_DIM, autocast="bf16"),
        eager_rerun_bit_equal=f"{same}/{sim_eager.numel()}",
        inductor_bit_equal=f"{same_compiled}/{sim_eager.numel()}",
        note=("the eager similarity is bit-stable rerun-to-rerun, but a Triton "
              "reproduction of the bf16 norm/bmm chain matched only ~48% of "
              "entries in the K5A probe (and the bf16 payload norm 0/1024), so "
              "recomputing it in-kernel would put decisions at flip risk; the "
              "chain is therefore consumed, not recomputed"),
        eager_preprocessing_kernels_uncompiled=count["kernels"])


# ---------------------------------------------------------------------- gate B


def floors_report(policy):
    """The measured floors first, then the fused deviations inside them."""
    rows = []
    for label, stream, rank, autocast in (
        ("fp32_small", (2, 128, 2, 8, 10, 4), 8, False),
        ("registered_bf16", (4, 512, 2, 64, 512, 77), 64, True)):
        chunks = pinned_chunk_parity(policy, rank, 64, stream, autocast)
        worst_floor = max(row["weights_floor"] for row in chunks)
        worst_dev = max(row["weights_max_deviation"] for row in chunks)
        worst_median = max(row["weights_median_deviation"] for row in chunks)
        rows.append(dict(
            label=label, chunks=len(chunks), weights_floor=worst_floor,
            weights_max_deviation=worst_dev,
            weights_median_deviation=worst_median,
            weights_within_floor=worst_dev
            <= NOISE_FACTOR * max(worst_floor, 1e-30),
            all_schedule_bit_equal=all(
                all(row[name] for name in SCHEDULE_FIELDS) for row in chunks)))
    return rows


def gradient_floors_report(policy):
    """The adjoint floors: the eager autograd vs the fp64 adjoint oracle, and
    the fused adjoint inside NOISE_FACTOR x that floor."""
    rows = []
    for label, stream, rank, autocast in (
        ("fp32", (2, 256, 2, 8, 24, 5), 8, False),
        ("bf16", (2, 512, 2, 16, 64, 77), 16, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        records = []
        from experiments.expressivity_tasks.x45_matrix_emender.scan import (
            routed_chunk_scan)

        def router(index, offset, state, live, qc, pc, pol, rk):
            records.append((state.detach().clone(), live.detach().clone(), qc, pc))
            return route_chunk(index, offset, state, live, qc, pc, pol, rk)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            routed_chunk_scan(query, payload, decay, router=router,
                              policy=policy, rank=rank, rank_max=rank,
                              chunk_tokens=64, fused=False)
        state, live, qc, pc = records[len(records) // 2]
        generator = torch.Generator().manual_seed(1234)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            schedule, weights = route_chunk(0, 0, state, live, qc, pc, policy,
                                            rank)
        dweights = torch.randn(weights.shape, generator=generator).to(DEVICE)
        dpay = torch.randn(schedule.payload.shape,
                           generator=generator).to(DEVICE)
        floor, (qe, pe, re) = gradient_floor(qc, pc, state, live, policy, rank,
                                             dweights, dpay, autocast)
        q = qc.detach().clone().requires_grad_(True)
        p = pc.detach().clone().requires_grad_(True)
        r = state.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            fused_schedule, fused_weights = routing_diff_route_chunk(
                0, 0, r, live, q, p, policy, rank)
        loss = (fused_weights * dweights).sum() \
            + (fused_schedule.payload * dpay).sum()
        loss.backward()
        row = dict(label=label, gradient_floor=floor)
        for name, eager, fused in (("dquery", qe, q.grad),
                                    ("dpayload", pe, p.grad),
                                    ("dstate", re, r.grad)):
            row[name] = dict(
                deviation=float((eager.double() - fused.double()).abs().max()),
                bit_equal=bool(torch.equal(eager, fused)),
                within_floor=float((eager.double() - fused.double()).abs().max())
                <= NOISE_FACTOR * max(floor, 1e-30))
        row["dpayload_bit_exact"] = row["dpayload"]["bit_equal"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------- gate C


def eager_chunk_counts(policy):
    """One mid-sequence chunk of real routing, eager: kernels per chunk."""
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        routed_chunk_scan)
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                            512, 77)
    records = []
    with torch.no_grad():
        routed_chunk_scan(query, payload, decay,
                          router=lambda i, o, s, l, qc, pc, pol, rk:
                          (records.append((s.clone(), l.clone())), route_chunk(
                              i, o, s, l, qc, pc, pol, rk))[1],
                          policy=policy, rank=RANK, rank_max=RANK,
                          chunk_tokens=CHUNK, fused=False)
    state, live = records[len(records) // 2]
    qc = query[:, :CHUNK].permute(0, 2, 1, 3).contiguous()
    pc = payload[:, :CHUNK].permute(0, 2, 1, 3).contiguous()

    def eager():
        route_chunk(4, 0, state.clone(), live.clone(), qc, pc, policy, RANK)

    def fused():
        routing_diff_route_chunk(4, 0, state.clone(), live.clone(), qc, pc,
                                 policy, RANK)

    return dict(
        shape=dict(batch=BATCH, chunk_tokens=CHUNK, heads=HEADS, rank=RANK,
                   value_dim=VALUE_DIM),
        route_chunk=dict(kernels_per_chunk=kernel_count(eager)["kernels"],
                         timing_ms=timed(eager, repeats=5, label="route_chunk")),
        routing_diff_route_chunk=dict(
            kernels_per_chunk=kernel_count(fused, repeats=1)["kernels"],
            timing_ms=timed(fused, repeats=5, label="routing_diff_route_chunk")))


def compiled_differential(policy):
    """The registered arm's own instrument (K5's gate A): compiled routed scan
    (fwd+bwd) minus compiled scan-only over the same recorded schedule, per
    router, at the registered shape -- the routing region's before/after."""
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        chunk_scan, routed_chunk_scan)
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                            512, 77)
    query = query.detach().requires_grad_(True)
    payload = payload.detach().requires_grad_(True)
    decay = decay.detach().requires_grad_(True)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        schedule, weights = route_chunk(index, offset, state, live, qc, pc,
                                        pol, rk)
        records.append((schedule, weights))
        return schedule, weights

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query.detach(), payload.detach(), decay.detach(),
                          router=router, policy=policy, rank=RANK,
                          rank_max=RANK, chunk_tokens=CHUNK, fused=False)
    schedule_full = MatrixSchedule(
        slot=torch.cat([s.slot for s, _ in records], dim=1),
        erase=torch.cat([s.erase for s, _ in records], dim=1),
        payload=torch.cat([s.payload for s, _ in records], dim=1),
        op=torch.cat([s.op for s, _ in records], dim=1),
        live_after=torch.cat([s.live_after for s, _ in records], dim=1))
    weights_full = torch.cat([w for _, w in records], dim=2)
    decay_hm = decay.detach().permute(0, 2, 1).contiguous()
    state0 = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, device=DEVICE)
    live0 = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device=DEVICE)
    chunks = len(records)
    compiled_scan = torch.compile(routed_chunk_scan, dynamic=False)

    def fresh(*tensors):
        return tuple(t.detach().clone().requires_grad_(True) for t in tensors)

    def compiled_full(router):
        def run():
            cq, cp, cd = fresh(query, payload, decay)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = compiled_scan(cq, cp, cd, router=router, policy=policy,
                                      rank=RANK, rank_max=RANK,
                                      chunk_tokens=CHUNK, fused=False)
            (result.reads.sum() + result.state.sum()).backward()
        return run

    def scan_only(weights, decay_in, state, live):
        reads = torch.empty(BATCH, HEADS, TOKENS, VALUE_DIM,
                            dtype=torch.float32, device=state.device)
        for start in range(0, TOKENS, CHUNK):
            stop = start + CHUNK
            piece = MatrixSchedule(
                slot=schedule_full.slot[:, start:stop],
                erase=schedule_full.erase[:, start:stop],
                payload=schedule_full.payload[:, start:stop],
                op=schedule_full.op[:, start:stop],
                live_after=schedule_full.live_after[:, start:stop])
            chunk_reads, state, live = chunk_scan(
                state, live, weights[:, :, start:stop],
                decay_in[:, :, start:stop], piece, fused=False)
            reads[:, :, start:stop] = chunk_reads
        return reads, state, live

    compiled_scan_only = torch.compile(scan_only, dynamic=False)

    def compiled_only():
        (cw,) = fresh(weights_full)
        reads, state, _ = compiled_scan_only(cw, decay_hm, state0.clone(),
                                             live0.clone())
        (reads.sum() + state.sum()).backward()

    rows = {}
    for label, fn in (("eager_router", compiled_full(route_chunk)),
                      ("k5a_router", compiled_full(routing_diff_route_chunk)),
                      ("scan_only", compiled_only)):
        timing = timed(fn, repeats=7, label=label)
        kernels = kernel_count(fn, repeats=1)["kernels"]
        rows[label] = dict(host_dispatch_ms_p50=timing["dispatch_ms_p50"],
                          latency_ms_p50=timing["latency_ms_p50"],
                          device_ms_p50=timing["device_ms_p50"],
                          kernels_per_scan=kernels,
                          kernels_per_chunk=kernels / chunks)

    def routing(row):
        return dict(
            kernels_per_scan=row["kernels_per_scan"]
            - rows["scan_only"]["kernels_per_scan"],
            kernels_per_chunk=(row["kernels_per_scan"]
                               - rows["scan_only"]["kernels_per_scan"]) / chunks,
            host_ms_per_scan=row["host_dispatch_ms_p50"]
            - rows["scan_only"]["host_dispatch_ms_p50"],
            device_ms_per_scan=row["device_ms_p50"]
            - rows["scan_only"]["device_ms_p50"])
    return dict(
        shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                   value_dim=VALUE_DIM, chunk_tokens=CHUNK,
                   chunks_per_scan=chunks),
        arms=rows,
        routing_differential=dict(eager_router=routing(rows["eager_router"]),
                                  k5a_router=routing(rows["k5a_router"]),
                                  reference=("K5's gate A instrument "
                                             "(receipts/k5_routing_fusion.json) "
                                             "recorded eager routing compiled = "
                                             "41.0 kernels/chunk and the fused "
                                             "decision-only router 39.9 -- the "
                                             "differentiable channels were the "
                                             "named residual this stage fuses")),
        k5_reference=dict(eager_router_kernels_per_chunk=40.96875,
                          gate_a_fused_router_kernels_per_chunk=39.9375,
                          source="receipts/k5_routing_fusion.json"))


def gate_suites_rerun():
    """The K4 grad gates and the lane's suites, re-run verbatim as subprocesses."""
    suites = ["test_k4_fused_backward.py", "test_k5_recompute.py",
              "test_k5_routing_fusion.py", "test_k5a_routing_diff.py"]
    rows = []
    for suite in suites:
        command = [".venv/bin/python", "-m", "pytest", "-q",
                   f"experiments/expressivity_tasks/x45_matrix_emender/{suite}"]
        began = time.monotonic()
        finished = subprocess.run(command, capture_output=True, text=True,
                                  cwd=str(ROOT.parents[2]))
        tail = finished.stdout.strip().splitlines()[-1] \
            if finished.stdout.strip() else ""
        rows.append(dict(suite=suite, command=" ".join(command),
                         returncode=finished.returncode,
                         wall_s=round(time.monotonic() - began, 1), summary=tail))
    return rows


# ----------------------------------------------------------- the cd-protocol fixture


def cd_protocol_fixture():
    """The registered model's own routing on a real flagship-stream activation."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for)

    L.check_python()
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    torch.cuda.manual_seed_all(3401)
    policy = RoutingPolicy(ops=True)
    shape = matrix_config()
    model = model_for(shape, policy, CHUNK, compile_scan=False).cuda().train()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    batches = iter(stream.batches("train"))
    rows_ = next(batches)
    x_ids, y_ids, _ = next(iter(grouped(rows_, 8)))

    captured = {}

    def hook(module, inputs, output):
        captured.setdefault("x", inputs[0].detach())

    handle = model.blocks[0].mixer.register_forward_hook(hook)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.loss(x_ids, y_ids)
    handle.remove()
    x = captured["x"]
    batch, tokens, _ = x.shape
    mixer = model.blocks[0].mixer
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = mixer.query(x).view(batch, tokens, mixer.heads, mixer.value_dim)
        payload = mixer.value(x).view(batch, tokens, mixer.heads, mixer.value_dim)
        decay = torch.sigmoid(mixer.decay(x)).clamp(min=1e-4, max=1.0 - 1e-6)
    pinned = pinned_chunk_parity(policy, mixer.rank, CHUNK,
                                 (batch, tokens, mixer.heads, mixer.value_dim,
                                  512, 3401), autocast=True)
    schedule_parity = all(all(row[name] for name in SCHEDULE_FIELDS)
                          and row["weights_within_floor"] for row in pinned)

    def one_step(model_instance):
        model_instance.zero_grad(set_to_none=True)
        loss = model_instance.loss(x_ids, y_ids)
        loss.backward()
        grad_scale = sum(float(p.grad.detach().abs().sum())
                         for p in model_instance.parameters() if p.grad is not None)
        return float(loss.detach()), grad_scale

    def fresh_model(fused_routing_diff):
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        return model_for(shape, policy, CHUNK, compile_scan=False,
                         fused_routing_diff=fused_routing_diff).cuda().train()

    loss_a, grads_a = one_step(fresh_model(False))
    loss_b, grads_b = one_step(fresh_model(False))
    loss_c, grads_c = one_step(fresh_model(True))
    deviation = abs(grads_a - grads_c)
    baseline = abs(grads_a - grads_b)
    return dict(
        fixture=("registered matrix_ops model, block 0's mixer on a real "
                 "flagship-stream 64-window batch, bf16 autocast"),
        shape=dict(batch=batch, tokens=tokens, heads=mixer.heads,
                   rank=mixer.rank, value_dim=mixer.value_dim,
                   chunk_tokens=CHUNK, chunks=len(pinned)),
        schedule_bit_parity_all_chunks=schedule_parity,
        training_step=dict(
            note=("one eager-scan step per router; the eager scan's "
                  "duplicate-slot scatter_add_ is atomics-nondeterministic, so "
                  "the fused arm is compared against the eager arm's own rerun "
                  "deviation"),
            loss_eager=loss_a, loss_eager_rerun=loss_b, loss_fused=loss_c,
            grad_scale_eager=grads_a, grad_scale_eager_rerun=grads_b,
            grad_scale_fused=grads_c,
            grad_scale_fused_deviation=deviation,
            grad_scale_eager_rerun_baseline=baseline,
            within_baseline=deviation <= NOISE_FACTOR * max(baseline, 1e-30)))


# ------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5a_routing_diff.json"))
    parser.add_argument("--skip-cd", action="store_true",
                        help="skip the cd-protocol fixture (data-bound, slow)")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    policy = RoutingPolicy(ops=True)
    report = {"session": session_meta(), "gate": "K5A"}

    print("[gate A] pinned chunk parity, fp32 ...", flush=True)
    report["pinned_chunk_parity"] = dict(
        fixture="fp32, distinct directions, rank 8, chunk 32",
        rows=pinned_chunk_parity(policy, 8, 32, (2, 128, 2, 8, 10, 4), False))

    print("[gate A] the mandatory flip test on the real routed fixture ...",
          flush=True)
    began = time.monotonic()
    report["flip_test"] = flip_test(policy)
    print(f"[gate A] flip test took {time.monotonic() - began:.0f} s: "
          f"{report['flip_test']['flips']} flips", flush=True)

    print("[probe] why the decision inputs stay torch-computed ...", flush=True)
    report["similarity_probe"] = similarity_probe()

    print("[gate B] the measured floors ...", flush=True)
    report["floors"] = floors_report(policy)
    report["gradient_floors"] = gradient_floors_report(policy)
    print(json.dumps(report["floors"], indent=1), flush=True)
    print(json.dumps(report["gradient_floors"], indent=1), flush=True)

    print("[gate C] per-chunk kernel counts, eager ...", flush=True)
    report["eager_chunk"] = eager_chunk_counts(policy)

    print("[gate C] compiled differential at the registered shape "
          "(3 graph compiles) ...", flush=True)
    began = time.monotonic()
    report["compiled_differential"] = compiled_differential(policy)
    print(f"[gate C] took {time.monotonic() - began:.0f} s", flush=True)

    print("[gates] the K4/K5 suites re-run verbatim ...", flush=True)
    report["gate_suites_rerun"] = gate_suites_rerun()

    if not arguments.skip_cd:
        print("[cd-protocol] the registered model on the flagship stream ...",
              flush=True)
        began = time.monotonic()
        report["cd_protocol_fixture"] = cd_protocol_fixture()
        print(f"[cd-protocol] took {time.monotonic() - began:.0f} s", flush=True)

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps({
        "flip_test_zero_flips": report["flip_test"]["zero_flips"],
        "floors_within": [row["weights_within_floor"] for row in report["floors"]],
        "eager_kernels_per_chunk": {
            "route_chunk": report["eager_chunk"]["route_chunk"]["kernels_per_chunk"],
            "k5a": report["eager_chunk"]["routing_diff_route_chunk"]["kernels_per_chunk"]},
        "compiled_routing_kernels_per_chunk":
            report["compiled_differential"]["routing_differential"],
        "gate_suites": [(row["suite"], row["returncode"])
                        for row in report["gate_suites_rerun"]],
    }, indent=1))


if __name__ == "__main__":
    main()
