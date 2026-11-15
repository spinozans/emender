"""K5 gate A receipt -- the fused routing decision channel, parity and launch counts.

Everything the gate demands, banked into ``receipts/k5_routing_fusion.json``:

* **Decision parity** -- the fused router vs eager ``schedule.route_chunk``,
  bit-identical, on fixtures that produce all five ops, at ragged chunk lengths,
  across a rank prefix and the growth policy, under the model's bf16 autocast at
  the registered shape, on a duplicate-row tie fixture, on pinned chunk-entry
  states (the deterministic form), and on the cd-protocol training fixture (the
  registered model's own block-0 mixer fed a real flagship-stream activation).
* **Launch counts and host time** -- eager per-chunk (route_chunk vs the fused
  router) and the compiled differential at the registered shape (compiled routed
  scan minus compiled scan-only over the same recorded schedule, forward and
  backward), which is the honest before/after for the registered arm.
* **The cross-regime rows** -- the compiled arm's decisions against the eager
  router's, both routers, so the pre-existing inductor-vs-eager difference is on
  the record next to the zero K5 introduces.

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5_routing_fusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import (
    kernel_count, timed)
from experiments.expressivity_tasks.x45_matrix_emender.k5_routing_fusion import (
    fused_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunk_scan, routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
DEVICE = "cuda"
# The registered mixer shape (profile_step_vs_gdn2's differential microbench shape).
BATCH, HEADS, TOKENS, RANK, VALUE_DIM, CHUNK = 8, 2, 2048, 128, 128, 64
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after", "payload")


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    return query, payload, decay


def pinned_chunk_states(query, payload, decay, policy, rank, chunk_tokens,
                         autocast=False):
    """Record the eager router's own chunk-entry states, then compare both routers
    on the pinned inputs -- the deterministic parity instrument (no scan
    nondeterminism in the comparison)."""
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
            fused = fused_route_chunk(index, index * chunk_tokens, state, live,
                                      qc, pc, policy, rank)
        row = dict(chunk=index)
        for name in SCHEDULE_FIELDS:
            row[name] = bool(torch.equal(getattr(eager[0], name),
                                         getattr(fused[0], name)))
        row["weights"] = bool(torch.equal(eager[1], fused[1]))
        rows.append(row)
    return rows


def parity_row(label, query, payload, decay, policy, rank, rank_max, chunk_tokens,
               autocast):
    """Whole-scan parity: the schedule bit-exact, continuous channels against the
    eager arm's own run-to-run stability (the eager scan's duplicate-slot
    ``scatter_add_`` is atomics-nondeterministic)."""
    def run(router):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return routed_chunk_scan(query, payload, decay, router=router,
                                     policy=policy, rank=rank, rank_max=rank_max,
                                     chunk_tokens=chunk_tokens, collect=True)
    eager, again, fused = run(route_chunk), run(route_chunk), run(fused_route_chunk)
    row = dict(label=label, ops=op_histogram(eager.schedule),
               chunks=-(-query.shape[1] // chunk_tokens))
    for name in SCHEDULE_FIELDS:
        row[name] = bool(torch.equal(getattr(eager.schedule, name),
                                     getattr(fused.schedule, name)))
    for name in ("weights", "reads", "state"):
        first, second = getattr(eager, name), getattr(again, name)
        observed = getattr(fused, name)
        row[f"{name}_bit_equal"] = bool(torch.equal(first, observed))
        if not torch.equal(first, second):
            row[f"{name}_eager_rerun_noise"] = float(
                (first.double() - second.double()).abs().max())
            row[f"{name}_fused_deviation"] = float(
                (first.double() - observed.double()).abs().max())
    return row


def measure_eager_chunk(policy, rank=128):
    """One mid-sequence chunk of real routing, eager: kernels and host ms per chunk."""
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM, 512,
                                            seed=77)
    state = torch.zeros(BATCH, HEADS, rank, VALUE_DIM, device=DEVICE)
    live = torch.zeros(BATCH, HEADS, rank, dtype=torch.bool, device=DEVICE)
    # Populate the store the way the arm's own trajectory would, so the measured
    # chunk is a mid-sequence one (writes, merges and pops on a live store).
    records = []
    with torch.no_grad():
        routed_chunk_scan(query, payload, decay,
                          router=lambda i, o, s, l, qc, pc, pol, rk:
                          (records.append((s.clone(), l.clone())), route_chunk(
                              i, o, s, l, qc, pc, pol, rk))[1],
                          policy=policy, rank=rank, rank_max=rank,
                          chunk_tokens=CHUNK, fused=False)
    state, live = records[len(records) // 2]
    qc = query[:, :CHUNK].permute(0, 2, 1, 3).contiguous()
    pc = payload[:, :CHUNK].permute(0, 2, 1, 3).contiguous()

    def eager():
        route_chunk(4, 0, state.clone(), live.clone(), qc, pc, policy, rank)

    def fused():
        fused_route_chunk(4, 0, state.clone(), live.clone(), qc, pc, policy, rank)

    return dict(
        shape=dict(batch=BATCH, chunk_tokens=CHUNK, heads=HEADS, rank=rank,
                   value_dim=VALUE_DIM),
        route_chunk=dict(kernels_per_chunk=kernel_count(eager, repeats=1)["kernels"],
                         timing_ms=timed(eager, repeats=5, label="route_chunk")),
        fused_route_chunk=dict(
            kernels_per_chunk=kernel_count(fused, repeats=1)["kernels"],
            timing_ms=timed(fused, repeats=5, label="fused_route_chunk")))


def measure_compiled_differential(policy):
    """The registered arm's own instrument, from PROFILE_STEP_VS_GDN2 §3:
    compiled routed scan (fwd+bwd) minus compiled scan-only over the same recorded
    schedule, for both routers, at the registered shape."""
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM, 512,
                                            seed=77)
    query = query.detach().requires_grad_(True)
    payload = payload.detach().requires_grad_(True)
    decay = decay.detach().requires_grad_(True)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        schedule, weights = route_chunk(index, offset, state, live, qc, pc, pol, rk)
        records.append((schedule, weights))
        return schedule, weights

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query.detach(), payload.detach(), decay.detach(),
                          router=router, policy=policy, rank=RANK, rank_max=RANK,
                          chunk_tokens=CHUNK, fused=False)
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

    def fresh(*tensors):
        return tuple(t.detach().clone().requires_grad_(True) for t in tensors)

    compiled_scan = torch.compile(routed_chunk_scan, dynamic=False)

    def compiled_full(router):
        def run():
            cq, cp, cd = fresh(query, payload, decay)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = compiled_scan(cq, cp, cd, router=router, policy=policy,
                                      rank=RANK, rank_max=RANK, chunk_tokens=CHUNK,
                                      fused=False)
            (result.reads.sum() + result.state.sum()).backward()
        return run

    def scan_only(weights, decay_in, state, live):
        reads = torch.empty(BATCH, HEADS, TOKENS, VALUE_DIM, dtype=torch.float32,
                            device=state.device)
        for start in range(0, TOKENS, CHUNK):
            stop = start + CHUNK
            piece = MatrixSchedule(
                slot=schedule_full.slot[:, start:stop],
                erase=schedule_full.erase[:, start:stop],
                payload=schedule_full.payload[:, start:stop],
                op=schedule_full.op[:, start:stop],
                live_after=schedule_full.live_after[:, start:stop])
            chunk_reads, state, live = chunk_scan(
                state, live, weights[:, :, start:stop], decay_in[:, :, start:stop],
                piece, fused=False)
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
                      ("fused_router", compiled_full(fused_route_chunk)),
                      ("scan_only", compiled_only)):
        timing = timed(fn, repeats=7, label=label)
        kernels = kernel_count(fn, repeats=1)["kernels"]
        rows[label] = dict(host_dispatch_ms_p50=timing["dispatch_ms_p50"],
                          latency_ms_p50=timing["latency_ms_p50"],
                          device_ms_p50=timing["device_ms_p50"],
                          kernels_per_scan=kernels,
                          kernels_per_chunk=kernels / chunks)

    def routing(router_row):
        return dict(
            kernels_per_scan=router_row["kernels_per_scan"]
            - rows["scan_only"]["kernels_per_scan"],
            kernels_per_chunk=(router_row["kernels_per_scan"]
                               - rows["scan_only"]["kernels_per_scan"]) / chunks,
            host_ms_per_scan=router_row["host_dispatch_ms_p50"]
            - rows["scan_only"]["host_dispatch_ms_p50"])
    return dict(
        shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                   value_dim=VALUE_DIM, chunk_tokens=CHUNK, chunks_per_scan=chunks),
        arms=rows,
        routing_differential=dict(eager_router=routing(rows["eager_router"]),
                                  fused_router=routing(rows["fused_router"]),
                                  host_ms_note=("the dispatch p50 is stable to "
                                                "~+/-20% run to run at this shape; "
                                                "the kernel counts are the robust "
                                                "before/after, the host ms the "
                                                "honest same-session estimate")),
        reference=("PROFILE_STEP_VS_GDN2.md §3 recorded routing compiled = 28.5 ms "
                   "host / 724 kernels per scan at its own fixture; the absolute "
                   "numbers here are this session's own fixture, the before/after "
                   "is the comparison that matters"))


def measure_cross_regime(policy):
    """Compiled-vs-eager decision differences, both routers: the pre-existing
    inductor property, and the zero K5 adds to it."""
    query, payload, decay = distinct_stream(BATCH, 512, HEADS, VALUE_DIM, 512,
                                            seed=77)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)

    def run_compiled(router):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return compiled(query, payload, decay, router=router, policy=policy,
                            rank=RANK, rank_max=RANK, chunk_tokens=CHUNK,
                            fused=False, collect=True)

    def run_eager(router):
        return routed_chunk_scan(query, payload, decay, router=router,
                                 policy=policy, rank=RANK, rank_max=RANK,
                                 chunk_tokens=CHUNK, fused=False, collect=True)

    eager_routed = run_eager(route_chunk)
    compiled_eager = run_compiled(route_chunk)
    compiled_fused = run_compiled(fused_route_chunk)
    lanes = eager_routed.schedule.slot.numel()

    def differences(left, right):
        row = {}
        for name in SCHEDULE_FIELDS:
            a, b = getattr(left.schedule, name), getattr(right.schedule, name)
            row[name] = int((a != b).sum()) if a.dtype != torch.bool \
                else int((a != b).sum())
        row["lanes"] = lanes
        return row
    return dict(
        shape=dict(batch=BATCH, tokens=512, heads=HEADS, rank=RANK,
                   chunk_tokens=CHUNK, autocast="bf16"),
        compiled_eager_router_vs_eager_router=differences(compiled_eager,
                                                          eager_routed),
        compiled_fused_router_vs_compiled_eager_router=differences(
            compiled_fused, compiled_eager),
        note=("the first row is the pre-existing inductor-vs-eager difference of "
              "the registered arm (its similarity bmm is inductor-lowered at "
              "bf16); the second row is what K5 adds to the compiled regime and "
              "is asserted zero by test_gate_c"))


def measure_cd_protocol_fixture():
    """The registered model's own routing on a real flagship-stream activation.

    Builds the registered model (seed 3401, the gates.py construction), runs one
    real batch, captures block 0's mixer input, and compares both routers'
    decisions chunk by chunk on the pinned chunk-entry states -- the
    cd-protocol fixture, bit-exact.  One eager-scan training step per router is
    run as well, with a same-router rerun as the nondeterminism baseline.
    """
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for)
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
        NOISE_FACTOR)

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
    rows = next(batches)
    x_ids, y_ids, _ = next(iter(grouped(rows, 8)))

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
    pinned = pinned_chunk_states(query, payload, decay, policy, mixer.rank, CHUNK,
                                  autocast=True)
    schedule_parity = all(all(row[name] for name in SCHEDULE_FIELDS)
                          and row["weights"] for row in pinned)

    # One whole eager-scan training step per router, plus the rerun baseline.
    # Three identically-initialised models (same seed, same batch, no optimizer
    # step -- the step would change the weights and the "rerun" with them), so
    # the only differences are routing and the scan's own atomics noise.
    def one_step(router_model):
        router_model.zero_grad(set_to_none=True)
        loss = router_model.loss(x_ids, y_ids)
        loss.backward()
        grad_scale = sum(float(p.grad.detach().abs().sum())
                         for p in router_model.parameters() if p.grad is not None)
        return float(loss.detach()), grad_scale

    def fresh_model(fused_routing):
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        return model_for(shape, policy, CHUNK, compile_scan=False,
                         fused_routing=fused_routing).cuda().train()

    loss_a, grads_a = one_step(fresh_model(False))
    loss_b, grads_b = one_step(fresh_model(False))
    loss_c, grads_c = one_step(fresh_model(True))
    deviation = abs(grads_a - grads_c)
    baseline = abs(grads_a - grads_b)
    return dict(
        fixture="registered matrix_ops model, block 0's mixer on a real "
                "flagship-stream 64-window batch, bf16 autocast",
        shape=dict(batch=batch, tokens=tokens, heads=mixer.heads,
                   rank=mixer.rank, value_dim=mixer.value_dim,
                   chunk_tokens=CHUNK, chunks=len(pinned)),
        schedule_bit_parity_all_chunks=schedule_parity,
        training_step=dict(
            note=("one eager-scan step per router; the eager scan's duplicate-slot "
                  "scatter_add_ is atomics-nondeterministic, so the fused arm is "
                  "compared against the eager arm's own rerun deviation"),
            loss_eager=loss_a, loss_eager_rerun=loss_b, loss_fused=loss_c,
            grad_scale_eager=grads_a, grad_scale_eager_rerun=grads_b,
            grad_scale_fused=grads_c,
            grad_scale_fused_deviation=deviation,
            grad_scale_eager_rerun_baseline=baseline,
            within_baseline=deviation <= NOISE_FACTOR * max(baseline, 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5_routing_fusion.json"))
    arguments = parser.parse_args()
    report = {"session": session_meta(), "gate": "A"}
    policy = RoutingPolicy(ops=True)

    print("[parity] pinned chunk states, five ops ...", flush=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 4)
    pinned = pinned_chunk_states(query, payload, decay, policy, 8, 32)
    report["pinned_chunk_parity"] = dict(
        fixture="fp32, distinct directions, rank 8, chunk 32",
        rows=pinned,
        all_bit_equal=all(all(row[name] for name in SCHEDULE_FIELDS)
                          and row["weights"] for row in pinned))

    fixtures = [
        ("five_ops_small", dict(stream=(2, 64, 2, 8, 6, 23), rank=5, rank_max=5,
                                chunk=16, autocast=False)),
        ("pop_saturated", dict(stream=(2, 256, 2, 8, 24, 5), rank=8, rank_max=8,
                               chunk=64, autocast=False)),
        ("ragged_100_over_64", dict(stream=(2, 100, 2, 7, 8, 100), rank=7,
                                    rank_max=7, chunk=64, autocast=False)),
        ("rank_prefix_6_of_16", dict(stream=(2, 128, 2, 8, 10, 3), rank=6,
                                     rank_max=16, chunk=32, autocast=False)),
        ("registered_bf16", dict(stream=(8, 2048, 2, 128, 512, 77), rank=128,
                                 rank_max=128, chunk=64, autocast=True)),
    ]
    rows = []
    for label, spec in fixtures:
        print(f"[parity] whole-scan {label} ...", flush=True)
        stream = distinct_stream(*spec["stream"][:5], seed=spec["stream"][5])
        rows.append(parity_row(label, *stream, policy, spec["rank"],
                               spec["rank_max"], spec["chunk"], spec["autocast"]))
    growing = RoutingPolicy(ops=True, grow=True, grow_window=2, grow_trigger=0.9)
    stream = distinct_stream(2, 256, 2, 8, 12, 9)
    rows.append(parity_row("growth_policy", *stream, growing, 8, 32, 64, False))
    report["whole_scan_parity"] = rows

    print("[eager] per-chunk kernel counts and host ms ...", flush=True)
    report["eager_chunk"] = measure_eager_chunk(policy)

    print("[compiled] differential at the registered shape (3 graph compiles) ...",
          flush=True)
    began = time.monotonic()
    report["compiled_differential"] = measure_compiled_differential(policy)
    print(f"[compiled] took {time.monotonic() - began:.0f} s", flush=True)

    print("[cross-regime] compiled vs eager decision differences ...", flush=True)
    report["cross_regime"] = measure_cross_regime(policy)

    print("[cd-protocol] the registered model on the flagship stream ...",
          flush=True)
    report["cd_protocol_fixture"] = measure_cd_protocol_fixture()

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps({
        "pinned_all_bit_equal": report["pinned_chunk_parity"]["all_bit_equal"],
        "whole_scan_schedule_parity":
            [row["label"] for row in rows
             if not all(row[name] for name in SCHEDULE_FIELDS)],
        "eager_kernels_per_chunk": {
            "route_chunk": report["eager_chunk"]["route_chunk"]["kernels_per_chunk"],
            "fused": report["eager_chunk"]["fused_route_chunk"]["kernels_per_chunk"]},
        "compiled_routing_kernels_per_chunk":
            report["compiled_differential"]["routing_differential"],
        "cd_schedule_bit_parity":
            report["cd_protocol_fixture"]["schedule_bit_parity_all_chunks"],
    }, indent=1))


if __name__ == "__main__":
    main()
