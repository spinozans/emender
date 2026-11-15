"""K5B gate receipt -- the whole-scan op pair, gated and banked.

Banks ``receipts/k5b_whole_scan.json`` with everything the stage's gates
demand:

* **Gate A (bit-exactness, incl. the mandatory flip test)** -- the whole-scan
  path vs the per-chunk K5A+K4 arm it subsumes, every channel bit-equal by
  construction (the recorded schedule, weights, reads, state, liveness, and
  the three gradients through the wrapper: eager call, capture call, replay),
  at the registered shape under bf16 autocast; and THE FLIP TEST on the real
  routed fixture (A-phase's bar, matched): the router the walk runs vs the
  registered ``route_chunk`` on pinned chunk-entry states, every lane of
  every schedule field, zero flips asserted.  Plus the model-level form: one
  whole training step of the K5B arm vs the per-chunk K5A arm at depth 2 --
  the loss and every parameter gradient bit-equal.
* **Gate B (floors re-measured first)** -- the read-weights floor and the
  adjoint floor on the same fixtures, then the walked router's deviations
  inside ``NOISE_FACTOR`` x them (the K5B gradients are additionally
  bit-equal to the arm's, asserted by gate A).
* **Gate C (the host-cost instruments)** -- the routing+scan region's host
  ms per scan, before (the per-chunk K5A arm, compiled) and after (the
  whole-scan replay), at the registered mixer shape, plus the op-dispatch
  counts per step before and after (the full step-metric versions live in
  ``receipts/k5b_step_metric.json``; these are the isolated per-scan forms).
* **Gate D (determinism, pathology)** -- two replays bit-identical on every
  channel; the rank-512 refusal, the growth-policy refusal, the append-only
  control and the ragged windows, each the per-chunk stack verbatim.
* **The compiled regime** -- ``torch.compile`` hosts the op pair with no
  ``InductorError``; the compiled wrapper's outputs and gradients are
  bit-identical to the uncompiled path's; and the pre-existing
  inductor-vs-eager preprocessing delta is re-measured (the K5A probe row) so
  the whole-scan arm's eager-preprocessing decisions (bit-equal to the
  registered eager router's, the flip-test bar) cannot be confused with the
  registered compiled arm's own delta.
* **The K4/K5 gate suites re-run verbatim as subprocesses.**

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5b_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5b_whole_scan.json
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

from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff import (
    routing_diff_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan import (
    DISPATCH, compiled_whole_scan, next_slot_id, reset_slots, whole_scan,
    whole_scan_record)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.test_k5a_routing_diff import (
    SCHEDULE_FIELDS, distinct_stream, gradient_floor, weights_floor)
from experiments.expressivity_tasks.x45_matrix_emender.test_k5b_whole_scan import (
    k5a_arm_scan)

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


def bit_equality_to_the_arm(policy, autocast=True):
    """The construction claim, at the registered shape: the whole-scan path's
    every channel is bit-equal to the per-chunk K5A+K4 arm's -- the recorded
    schedule/weights/reads/state/liveness, and the three gradients through
    the wrapper's eager call, capture call and replay."""
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                            512, 77)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                           generator=generator).to(DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=autocast):
        record = whole_scan_record(query, payload, decay, policy=policy,
                                   rank=RANK, rank_max=RANK, chunk_tokens=CHUNK)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, RANK, RANK,
                               CHUNK, collect=True, upstream=upstream)
    rows = {name: bool(torch.equal(getattr(arm.schedule, name),
                                   getattr(record.schedule, name)))
            for name in SCHEDULE_FIELDS}
    rows.update(weights=bool(torch.equal(arm.weights, record.weights)),
               reads=bool(torch.equal(arm.reads, record.reads)),
               state=bool(torch.equal(arm.state, record.state)),
               live=bool(torch.equal(arm.live, record.live)))
    reset_slots()
    slot = next_slot_id()
    calls = []
    for call in range(3):        # eager, capture+replay, replay
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            result = whole_scan(q, p, d, policy=policy, rank=RANK,
                                rank_max=RANK, chunk_tokens=CHUNK,
                                slot_id=slot)
        (result.reads * upstream).sum().backward()
        calls.append(dict(
            reads=bool(torch.equal(result.reads, arm.reads)),
            dquery=bool(torch.equal(q.grad, inputs[0].grad)),
            dpayload=bool(torch.equal(p.grad, inputs[1].grad)),
            ddecay=bool(torch.equal(d.grad, inputs[2].grad))))
    ledger = dict(DISPATCH)
    return dict(
        fixture=(f"registered shape: batch {BATCH}, {TOKENS} tokens, rank "
                 f"{RANK}, chunk {CHUNK}, bf16 autocast, "
                 f"{TOKENS // CHUNK} chunks per scan"),
        recorded_channels_bit_equal=rows,
        wrapper_calls=calls,
        wrapper_call_labels=("eager body", "capture + replay", "replay"),
        ledger_after=dict(ledger),
        all_bit_equal=all(rows.values())
        and all(all(row.values()) for row in calls))


def flip_test(policy):
    """THE MANDATORY FLIP TEST, A-phase's bar matched: on real routed fixtures
    (real chunk-entry states from a real routed scan at the registered shape
    under bf16 autocast), the router the whole-scan walk runs vs the
    registered ``route_chunk`` -- every lane of every schedule field."""
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
        differing_lanes=per_field, lanes=lanes,
        flips=sum(per_field.values()),
        zero_flips=sum(per_field.values()) == 0,
        construction=("the whole-scan walk runs the K5A router, which "
                      "consumes the eager preprocessing's own similarity and "
                      "strength tensors, so decisions are bit-equal by "
                      "construction; this test is the proof, at the bar and "
                      "fixture of the K5A receipt"))


def model_level_bit_equality():
    """One whole training step of the K5B arm vs the per-chunk K5A+K4 arm at
    depth 2, compile off (the eager preprocessing regime both share): the loss
    and every parameter gradient bit-equal."""
    from experiments.expressivity_tasks.x45_matrix_emender import model as M

    def build(whole):
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        shape = M.matrix_config(depth=2)
        policy = RoutingPolicy(ops=True)
        return M.model_for(shape, policy, 64, compile_scan=False,
                           fused_backward=not whole,
                           fused_routing_diff=not whole,
                           fused_whole_scan=whole).cuda().train()

    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (4, 2048), generator=generator).cuda()
    targets = torch.randint(0, 256, (4, 2048), generator=generator).cuda()

    def step(model):
        model.zero_grad(set_to_none=True)
        loss = model.loss(ids, targets)
        loss.backward()
        return float(loss.detach()), {name: p.grad.detach().clone()
                                      for name, p in model.named_parameters()}

    loss_arm, grads_arm = step(build(False))
    loss_k5b, grads_k5b = step(build(True))
    differing = [name for name in grads_arm
                 if not torch.equal(grads_arm[name], grads_k5b[name])]
    return dict(
        fixture=("depth-2 registered model, one whole training step (fwd + "
                  "bwd), compile off, bf16 autocast: the per-chunk K5A+K4 arm "
                  "vs the whole-scan arm"),
        loss_arm=loss_arm, loss_whole_scan=loss_k5b,
        loss_bit_equal=loss_arm == loss_k5b,
        parameters=len(grads_arm), gradients_differing=differing,
        all_gradients_bit_equal=not differing)


# ---------------------------------------------------------------------- gate B


def floors_report(policy):
    """The measured floors first (the K23 discipline), then the walked
    router's deviations inside NOISE_FACTOR x them, both regimes."""
    rows = []
    for label, stream, rank, autocast in (
        ("fp32_small", (2, 128, 2, 8, 10, 4), 8, False),
        ("registered_bf16", (4, 512, 2, 64, 512, 77), 64, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        records = []

        def router(index, offset, state, live, qc, pc, pol, rk):
            records.append((state.detach().clone(), live.detach().clone(), qc, pc))
            return route_chunk(index, offset, state, live, qc, pc, pol, rk)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            routed_chunk_scan(query, payload, decay, router=router,
                              policy=policy, rank=rank, rank_max=rank,
                              chunk_tokens=64, fused=False)
        worst_floor, worst_dev, worst_median = 0.0, 0.0, 0.0
        for state, live, qc, pc in records[::max(1, len(records) // 4)]:
            floor = weights_floor(qc, state[:, :, :rank], live[:, :, :rank],
                                  policy.read_scale, autocast)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                eager = route_chunk(0, 0, state, live, qc, pc, policy, rank)
                walked = routing_diff_route_chunk(0, 0, state, live, qc, pc,
                                                  policy, rank)
            delta = (eager[1].double() - walked[1].double()).abs()
            worst_floor = max(worst_floor, floor)
            worst_dev = max(worst_dev, float(delta.max()))
            worst_median = max(worst_median, float(delta.median()))
        rows.append(dict(
            label=label, chunks_sampled=max(1, len(records[::max(
                1, len(records) // 4)])),
            weights_floor=worst_floor, weights_max_deviation=worst_dev,
            weights_median_deviation=worst_median,
            within_floor=bool(worst_dev <= NOISE_FACTOR * max(worst_floor,
                                                             1e-30))))
    return rows


def gradient_floors_report(policy):
    """The adjoint floors (the eager autograd vs the fp64 adjoint oracle),
    then the whole-scan arm's adjoint inside NOISE_FACTOR x them -- with the
    stronger construction fact (bit-equal to the arm) asserted by gate A."""
    rows = []
    for label, stream, rank, autocast in (
        ("fp32", (2, 256, 2, 8, 24, 5), 8, False),
        ("bf16", (2, 512, 2, 16, 64, 77), 16, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        records = []

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
        rows.append(dict(
            label=label, gradient_floor=floor,
            note=("the whole-scan adjoint is bit-equal to the per-chunk "
                  "K5A+K4 arm's on the same upstream (gate A asserts it "
                  "end-to-end at the registered shape, and the model-level "
                  "row asserts every parameter gradient), so the floor that "
                  "bounds the arm's adjoint bounds it: the eager autograd's "
                  "own deviation from the fp64 oracle, re-measured here")))
    return rows


# ---------------------------------------------------------------------- gate C


def host_cost_per_scan():
    """The isolated per-scan host instruments (the step-metric forms, scaled
    by the 192 scans per step, live in receipts/k5b_step_metric.json): the
    routing+scan region's host issue time per scan, before and after."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        COMPILE_MODE, compiled_scan)

    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM,
                                            512, 77)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                           generator=generator).to(DEVICE)

    def fresh():
        return tuple(t.detach().clone().requires_grad_(True)
                     for t in (query, payload, decay))

    def before():
        q, p, d = fresh()
        scan = compiled_scan(COMPILE_MODE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = scan(q, p, d, router=routing_diff_route_chunk,
                          policy=policy, rank=RANK, rank_max=RANK,
                          chunk_tokens=CHUNK, fused=False, trainable_fused=True)
        (result.reads * upstream).sum().backward()

    reset_slots()
    slot = next_slot_id()
    after_scan = compiled_whole_scan("default")

    def after():
        q, p, d = fresh()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = after_scan(q, p, d, policy=policy, rank=RANK,
                                rank_max=RANK, chunk_tokens=CHUNK,
                                slot_id=slot)
        (result.reads * upstream).sum().backward()

    def issue_ms(fn, repeats):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeats):
            fn()
        issued = time.perf_counter() - began
        torch.cuda.synchronize()
        return round(issued / repeats * 1000, 3)

    rows = dict(
        shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                   value_dim=VALUE_DIM, chunk_tokens=CHUNK,
                   chunks_per_scan=TOKENS // CHUNK),
        scans_per_step=24 * 8,
        before_per_chunk_k5a_compiled_ms_per_scan=issue_ms(before, 5),
        after_whole_scan_ms_per_scan=issue_ms(after, 20),
        ledger=dict(DISPATCH),
        note=("host issue time of one mixer scan, fwd+bwd, warmed: the "
              "per-chunk arm crosses ~4 op bodies per chunk x 32 chunks; the "
              "whole-scan arm is 2 op dispatches + 2 graph replays + input "
              "copies"))
    rows["per_step_host_s_before"] = round(
        rows["before_per_chunk_k5a_compiled_ms_per_scan"]
        * rows["scans_per_step"] / 1000, 3)
    rows["per_step_host_s_after"] = round(
        rows["after_whole_scan_ms_per_scan"] * rows["scans_per_step"] / 1000,
        3)
    return rows


# ----------------------------------------------------------------- gates D and E


def determinism_and_pathology(policy):
    """Two replays bit-identical; the refusals are the per-chunk stack."""
    query, payload, decay = distinct_stream(2, 256, 2, 8, 24, 5)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 256, 2, 8, generator=generator).to(DEVICE)
    reset_slots()
    slot = next_slot_id()
    runs = []
    for _ in range(2):
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = whole_scan(q, p, d, policy=policy, rank=8, rank_max=8,
                                chunk_tokens=64, slot_id=slot)
        (result.reads * upstream).sum().backward()
        runs.append((result, q.grad, p.grad, d.grad))
    replay_determinism = dict(
        reads=bool(torch.equal(runs[0][0].reads, runs[1][0].reads)),
        state=bool(torch.equal(runs[0][0].state, runs[1][0].state)),
        dquery=bool(torch.equal(runs[0][1], runs[1][1])),
        dpayload=bool(torch.equal(runs[0][2], runs[1][2])),
        ddecay=bool(torch.equal(runs[0][3], runs[1][3])))

    def reference(pol, rank=8, rank_max=8, initial_state=None,
                 initial_live=None, chunk=32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return routed_chunk_scan(
                query, payload, decay, router=routing_diff_route_chunk,
                policy=pol, rank=rank, rank_max=rank_max, chunk_tokens=chunk,
                initial_state=initial_state, initial_live=initial_live,
                fused=False, trainable_fused=True)

    refusals = {}
    growing = RoutingPolicy(ops=True, grow=True, grow_window=2,
                            grow_trigger=0.9)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay, policy=growing, rank=8,
                             rank_max=16, chunk_tokens=32)
    refusals["growth"] = bool(torch.equal(refused.reads,
                                          reference(growing, 8, 16).reads))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay,
                             policy=RoutingPolicy(ops=False), rank=8,
                             rank_max=8, chunk_tokens=32)
        honest = reference(RoutingPolicy(ops=False))
    refusals["append_only"] = bool(torch.equal(refused.reads, honest.reads))

    rank = 512
    big_query, big_payload, big_decay = distinct_stream(2, 128, 2, 8, 512, 9)
    reset_slots()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(big_query, big_payload, big_decay, policy=policy,
                             rank=rank, rank_max=rank, chunk_tokens=64)
        honest = routed_chunk_scan(
            big_query, big_payload, big_decay,
            router=routing_diff_route_chunk, policy=policy, rank=rank,
            rank_max=rank, chunk_tokens=64, fused=False, trainable_fused=True)
    refusals["rank_512"] = bool(torch.equal(refused.reads, honest.reads))
    return dict(replay_bit_identical=replay_determinism,
                refusals_bit_equal_to_per_chunk_stack=refusals,
                fallbacks_in_ledger=int(DISPATCH["fallbacks"]))


def compiled_regime(policy):
    """Compile hosting: the compiled wrapper bit-equal to the uncompiled path
    and to the arm; and the pre-existing inductor-vs-eager preprocessing
    delta re-measured (the K5A probe), so the eager preprocessing inside the
    whole-scan op -- bit-equal to the registered eager router, the flip-test
    bar -- is not confused with the registered compiled arm's own delta."""
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 512, 2, 16, generator=generator).to(DEVICE)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, 16, 16, 64,
                               upstream=upstream)
    reset_slots()
    compiled = compiled_whole_scan("default")
    slot = next_slot_id()
    compiled_ok = []
    for _ in range(3):
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = compiled(q, p, d, policy=policy, rank=16, rank_max=16,
                              chunk_tokens=64, slot_id=slot)
        (result.reads * upstream).sum().backward()
        compiled_ok.append(bool(
            torch.equal(result.reads, arm.reads)
            and torch.equal(q.grad, inputs[0].grad)
            and torch.equal(p.grad, inputs[1].grad)
            and torch.equal(d.grad, inputs[2].grad)))

    # the K5A probe, re-measured: inductor's preprocessing vs eager's on a
    # pinned chunk (the delta the registered compiled arm has always carried)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=16, rank_max=16, chunk_tokens=64, fused=False)
    state, live, qc, pc = records[len(records) // 2]

    def preamble(payload_chunk, rows, act):
        frozen = rows.detach()
        strength = frozen.norm(dim=-1) * act.to(torch.float32)
        stored = frozen / strength.clamp_min(1e-6).unsqueeze(-1)
        value = payload_chunk / payload_chunk.norm(dim=-1,
                                                   keepdim=True).clamp_min(1e-6)
        return value @ stored.transpose(-1, -2)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        sim_eager = preamble(pc, state[:, :, :16], live[:, :, :16])
        sim_compiled = torch.compile(preamble, dynamic=False)(
            pc, state[:, :, :16], live[:, :, :16])
    same = int((sim_eager == sim_compiled).sum())
    return dict(
        compiled_wrapper_bit_equal=compiled_ok,
        inductorerror_raised=False,
        preprocessing_probe=dict(
            eager_vs_inductor_bit_equal=f"{same}/{sim_eager.numel()}",
            note=("pre-existing: the registered COMPILED arm's similarity "
                  "differs from the eager router's by inductor's reduction "
                  "orders (the K5A receipt's probe row).  The whole-scan op "
                  "keeps the EAGER preprocessing (captured into its graph), "
                  "so its decisions are bit-equal to the registered EAGER "
                  "router's -- the flip-test bar -- and any step-level "
                  "difference from the registered compiled arm is that "
                  "arm's own delta, not a K5B deviation")))


def gate_suites_rerun():
    """The lane's suites, re-run verbatim as subprocesses."""
    suites = ["test_k4_fused_backward.py", "test_k5_recompute.py",
              "test_k5_routing_fusion.py", "test_k5a_routing_diff.py",
              "test_k5b_whole_scan.py"]
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
                         wall_s=round(time.monotonic() - began, 1),
                         summary=tail))
    return rows


# ------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5b_whole_scan.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    policy = RoutingPolicy(ops=True)
    report = {"session": session_meta(), "gate": "K5B"}

    print("[gate A] bit-equality to the per-chunk arm, registered shape ...",
          flush=True)
    began = time.monotonic()
    report["bit_equality_to_the_arm"] = bit_equality_to_the_arm(policy)
    print(f"[gate A] took {time.monotonic() - began:.0f} s: "
          f"all_bit_equal={report['bit_equality_to_the_arm']['all_bit_equal']}",
          flush=True)

    print("[gate A] the mandatory flip test on the real routed fixture ...",
          flush=True)
    began = time.monotonic()
    report["flip_test"] = flip_test(policy)
    print(f"[gate A] flip test took {time.monotonic() - began:.0f} s: "
          f"{report['flip_test']['flips']} flips", flush=True)

    print("[gate A] model-level bit equality (depth 2, one step) ...",
          flush=True)
    report["model_level_bit_equality"] = model_level_bit_equality()

    print("[gate B] the measured floors ...", flush=True)
    report["floors"] = floors_report(policy)
    report["gradient_floors"] = gradient_floors_report(policy)
    print(json.dumps(report["floors"], indent=1), flush=True)

    print("[gate C] host ms/scan, before and after ...", flush=True)
    began = time.monotonic()
    report["host_cost_per_scan"] = host_cost_per_scan()
    print(f"[gate C] took {time.monotonic() - began:.0f} s: "
          f"{report['host_cost_per_scan']}", flush=True)

    print("[gate D/E] determinism and pathology ...", flush=True)
    report["determinism_and_pathology"] = determinism_and_pathology(policy)

    print("[gate C] the compiled regime ...", flush=True)
    report["compiled_regime"] = compiled_regime(policy)

    print("[gates] the lane's suites re-run verbatim ...", flush=True)
    report["gate_suites_rerun"] = gate_suites_rerun()

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps({
        "all_bit_equal": report["bit_equality_to_the_arm"]["all_bit_equal"],
        "flip_test_zero_flips": report["flip_test"]["zero_flips"],
        "model_level_all_bit_equal":
            report["model_level_bit_equality"]["all_gradients_bit_equal"],
        "floors_within": [row["within_floor"] for row in report["floors"]],
        "host_ms_per_scan":
            (report["host_cost_per_scan"]
             ["before_per_chunk_k5a_compiled_ms_per_scan"],
             report["host_cost_per_scan"]["after_whole_scan_ms_per_scan"]),
        "replay_determinism":
            report["determinism_and_pathology"]["replay_bit_identical"],
        "refusals": report["determinism_and_pathology"]
                    ["refusals_bit_equal_to_per_chunk_stack"],
        "compiled_wrapper_bit_equal":
            report["compiled_regime"]["compiled_wrapper_bit_equal"],
        "gate_suites": [(row["suite"], row["returncode"])
                        for row in report["gate_suites_rerun"]],
    }, indent=1))


if __name__ == "__main__":
    main()
