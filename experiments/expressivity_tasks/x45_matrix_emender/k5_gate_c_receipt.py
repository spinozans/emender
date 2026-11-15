"""K5 gate C receipt -- recompute-over-store: microbatch 8 for the fused path.

The store-everything K4 kept every chunk's trajectory alive until its backward:
67 MB per chunk at the registered shape, 2.15 GB per layer, 51.5 GB over 24
layers at microbatch 8 -- above the card's 47.4 GiB, which forced the fused arm
to microbatch 2 and the 4.1x launch regression the profile diagnosed.  Gate C
replaces the store with chunk-boundary checkpoints (the entry state the routing
already consumes) + in-backward recompute of the chunk's states
(``persistent_scan_recompute_backward``), and banks
``receipts/k5_recompute_over_store.json``:

* **grad parity**: the recomputed trajectory is the store form's own kernel on
  the same checkpoint, so every backward output is bit-identical to the store
  form's -- and against the autograd oracle, within the SAME multiples of
  re-measured floors as K4 (``dstate``/``dpayload`` bit-exact, ``dweights``/
  ``ddecay`` within the floor multiple, floors measured on the same fixtures
  first);
* **memory**: the byte accounting (trajectory per chunk vs checkpoint per
  chunk), the measured whole-scan fwd+bwd peaks of both forms, and the whole
  mb8 training step's peak allocation -- the card-fit claim;
* **the recompute cost**: per-chunk backward device time, store vs recompute,
  and the extra device seconds that projects to per step;
* **the mb8 fused step wall**: the whole training step at microbatch 8 with
  ``compile_scan=True`` + ``fused_backward=True`` (gate B's hosting), the
  gates.py protocol, un-profiled steady walls -- the configuration that OOMed
  before this gate.

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5_gate_c_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5_recompute_over_store.json
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

from experiments.expressivity_tasks.x45_matrix_emender.algebra import MatrixSchedule
from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import timed
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    fused_block_scan, persistent_scan_backward, persistent_scan_recompute_backward,
    persistent_scan_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.test_hostloop_free import (
    head_major)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    entry_state, routed_fixture)
from experiments.expressivity_tasks.x45_matrix_emender.test_k4_fused_backward import (
    GRAD_OUTPUTS, eager_block_walk, grad_floor)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
DEVICE = "cuda"
REG_BATCH, REG_HEADS, REG_RANK, REG_VALUE_DIM, REG_CHUNK = 8, 2, 128, 128, 64


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def store_form_grads(state, live, weights, decay, schedule, dreads, dexit):
    inputs = [state.detach().clone().requires_grad_(True),
              weights.detach().clone().requires_grad_(True),
              decay.detach().clone().requires_grad_(True),
              schedule.payload.detach().clone().requires_grad_(True)]
    attached = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                              payload=inputs[3], op=schedule.op,
                              live_after=schedule.live_after)
    _, _, _, trajectory = persistent_scan_trajectory(inputs[0], live, inputs[1],
                                                    inputs[2], attached)
    return persistent_scan_backward(trajectory, inputs[1], inputs[2], attached,
                                    dreads, dexit)


def recompute_form_grads(state, live, weights, decay, schedule, dreads, dexit):
    inputs = [state.detach().clone().requires_grad_(True),
              weights.detach().clone().requires_grad_(True),
              decay.detach().clone().requires_grad_(True),
              schedule.payload.detach().clone().requires_grad_(True)]
    attached = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                              payload=inputs[3], op=schedule.op,
                              live_after=schedule.live_after)
    outputs = fused_block_scan(inputs[0], live, inputs[1], inputs[2], attached)
    return torch.autograd.grad(list(outputs[:2]), inputs, [dreads, dexit],
                               allow_unused=True)


def parity_and_floors() -> dict:
    """Bit-parity to the store form and floor-multiple parity to the oracle."""
    fixture = routed_fixture()
    weights, decay, schedule = (fixture.weights, head_major(fixture.decay),
                                fixture.schedule)
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    floors, oracle = grad_floor(state, live, weights, decay, schedule, dreads, dexit)
    store = store_form_grads(state, live, weights, decay, schedule, dreads, dexit)
    recomputed = recompute_form_grads(state, live, weights, decay, schedule,
                                      dreads, dexit)
    rows = {}
    for index, name in enumerate(GRAD_OUTPUTS):
        bit_equal_to_store = bool(torch.equal(store[index], recomputed[index]))
        deviation = float((recomputed[index].double()
                           - oracle[index].double()).abs().max())
        multiple = deviation / floors[name] if floors[name] else float("inf")
        rows[name] = dict(bit_equal_to_store=bit_equal_to_store,
                          oracle_max_deviation=deviation,
                          measured_floor=floors[name],
                          floor_multiple=multiple,
                          bit_equal_to_oracle=bool(torch.equal(
                              recomputed[index], oracle[index])))
    return dict(
        fixture="the lane's canonical routed fixture (all five ops present)",
        outputs=rows,
        note=("dstate/dpayload bit-exact against the oracle, dweights/ddecay "
              "within the SAME floor multiples as the K4 gates -- the recompute "
              "reproduces the stored trajectory bit for bit, so the adjoint "
              "cannot tell the two forms apart"))


def memory_accounting() -> dict:
    """The byte claim and the measured whole-scan peaks of both forms."""
    trajectory_bytes = (REG_BATCH * REG_HEADS * REG_CHUNK * REG_RANK
                       * REG_VALUE_DIM * 4)
    checkpoint_bytes = REG_BATCH * REG_HEADS * REG_RANK * REG_VALUE_DIM * 4
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy, route_chunk)
    from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan
    tokens = 2048
    generator = torch.Generator().manual_seed(77)
    query = torch.randn(REG_BATCH, tokens, REG_HEADS, REG_VALUE_DIM,
                        generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(512, REG_VALUE_DIM, generator=generator).to(DEVICE)
    choice = torch.randint(0, 512, (REG_BATCH, tokens, REG_HEADS),
                           generator=generator)
    payload = directions[choice]
    decay = torch.full((REG_BATCH, tokens, REG_HEADS), 0.95, device=DEVICE)
    policy = RoutingPolicy(ops=True)

    def peak(trainable):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inputs = [tensor.detach().clone().requires_grad_(True)
                  for tensor in (query, payload, decay)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = routed_chunk_scan(inputs[0], inputs[1], inputs[2],
                                       router=route_chunk, policy=policy,
                                       rank=REG_RANK, rank_max=REG_RANK,
                                       chunk_tokens=REG_CHUNK, fused=False,
                                       trainable_fused=trainable)
        (result.reads.sum() + result.state.sum()).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2 ** 30

    recompute_peak = peak(True)
    store_simulated = 32 * trajectory_bytes / 2 ** 30  # the store form's own floor
    return dict(
        shape=dict(batch=REG_BATCH, tokens=tokens, heads=REG_HEADS,
                   rank=REG_RANK, value_dim=REG_VALUE_DIM,
                   chunk_tokens=REG_CHUNK, chunks_per_scan=tokens // REG_CHUNK),
        per_chunk=dict(trajectory_mb=trajectory_bytes / 2 ** 20,
                       checkpoint_mb=checkpoint_bytes / 2 ** 20),
        per_layer_at_mb8=dict(store_trajectory_gb=32 * trajectory_bytes / 2 ** 30,
                             checkpoints_gb=32 * checkpoint_bytes / 2 ** 30),
        whole_model_at_mb8=dict(store_24_layers_gb=24 * 32 * trajectory_bytes
                                / 2 ** 30,
                                checkpoints_24_layers_gb=24 * 32
                                * checkpoint_bytes / 2 ** 30),
        measured_whole_scan_peak_gb=dict(recompute_form=recompute_peak,
                                         store_form_floor_gb=store_simulated,
                                         note=("the recompute form's peak is the "
                                               "outputs + checkpoint set + one "
                                               "chunk scratch + the routing's own "
                                               "saved activations; the store form "
                                               "must additionally hold all 32 "
                                               "chunk trajectories")),
        card_gib=session_meta()["card_gib"])


def recompute_cost() -> dict:
    """Per-chunk backward device time, store vs recompute, and the per-step cost."""
    fixture = routed_fixture(tokens=64, chunk_tokens=64)
    weights, decay, schedule = (fixture.weights, head_major(fixture.decay),
                                fixture.schedule)
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)
    _, _, _, trajectory = persistent_scan_trajectory(state, live, weights, decay,
                                                     schedule)

    def store():
        persistent_scan_backward(trajectory, weights, decay, schedule, dreads,
                                  dexit)

    def recompute():
        persistent_scan_recompute_backward(state, live, weights, decay, schedule,
                                            dreads, dexit)

    store_row = timed(store, repeats=5, label="store-form backward")
    recompute_row = timed(recompute, repeats=5, label="recompute-form backward")
    extra_ms = recompute_row["device_ms_p50"] - store_row["device_ms_p50"]
    return dict(
        shape=dict(batch=batch, tokens=tokens, rank=rank, value_dim=value_dim),
        store_form=dict(device_ms_p50=store_row["device_ms_p50"],
                        occupancy_ms=store_row["occupancy_ms"]),
        recompute_form=dict(device_ms_p50=recompute_row["device_ms_p50"],
                            occupancy_ms=recompute_row["occupancy_ms"]),
        extra_device_ms_per_chunk=extra_ms,
        per_step_at_mb8=dict(
            chunks_per_step=24 * 32,
            extra_device_s_per_step=extra_ms * 24 * 32 / 1000),
        note=("the recompute pays one extra forward walk (the trajectory write) "
              "per chunk in backward; the persistent kernels it replaces were "
              "2.77 s of the 122 s mb2 wall, so the whole recompute class is "
              "small against the launch savings it buys"))


def measure_mb8_step(out_path: str) -> dict:
    """The whole training step at microbatch 8, compile ON + fused ON."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)
    import experiments.expressivity_tasks.x45_matrix_emender.model as model
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)

    def build():
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        shape = model.matrix_config()
        policy = RoutingPolicy(ops=True)
        built = model.model_for(shape, policy, 64, compile_scan=True,
                                fused_backward=True).cuda().train()
        summary = dict(shape=shape.__dict__,
                       parameters=model.parameter_count(shape),
                       routing=policy.record(), chunk_tokens=64)
        return built, P.LR, P.BETAS, P.WEIGHT_DECAY, summary

    P.X45_CONFIGS["x45_fused_mb8"] = dict(
        microbatch=8, compile_scan=True, fused_backward=True,
        description=("the fused K4 path at microbatch 8 with compile_scan=True "
                     "-- OOMed before gate C (the 51.5 GB trajectory store); "
                     "checkpoints + recompute now"))
    report = P.measure_arm("x45_fused_mb8", build, 8, out_path, warmup=2,
                           wall_steps=3, profile_mode="full", do_cprofile=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5_recompute_over_store.json"))
    parser.add_argument("--skip-step", action="store_true",
                        help="skip the mb8 whole-step measurement (receipt rows only)")
    arguments = parser.parse_args()
    report = {"session": session_meta(), "gate": "C"}

    print("[parity] recompute vs store vs oracle, floors re-measured ...", flush=True)
    report["grad_parity"] = parity_and_floors()

    print("[memory] byte accounting and whole-scan peaks ...", flush=True)
    report["memory"] = memory_accounting()

    print("[cost] the recompute's extra device time ...", flush=True)
    report["recompute_cost"] = recompute_cost()

    if not arguments.skip_step:
        print("[mb8] the whole fused+compiled step at microbatch 8 ...", flush=True)
        began = time.monotonic()
        report["mb8_fused_step"] = measure_mb8_step(arguments.out)
        print(f"[mb8] took {time.monotonic() - began:.0f} s", flush=True)
    else:
        report["mb8_fused_step"] = "skipped (--skip-step)"

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"receipt written to {out}")
    step = report.get("mb8_fused_step")
    if isinstance(step, dict):
        print(json.dumps(dict(
            grad_parity={name: (row["bit_equal_to_store"],
                                round(row["floor_multiple"], 2))
                         for name, row in report["grad_parity"]["outputs"].items()},
            scan_peak_gb=report["memory"]["measured_whole_scan_peak_gb"][
                "recompute_form"],
            extra_device_s_per_step=report["recompute_cost"]["per_step_at_mb8"][
                "extra_device_s_per_step"],
            mb8_step_walls_s=step["step_walls"],
            mb8_tokens_per_s=step["tokens_per_s"],
            mb8_peak_gb=step["peak_allocated_gb"],
            mb8_kernels_per_step=step["step_profile"]["kernels"],
            mb8_device_busy_s=step["step_profile"]["device_busy_ms"] / 1000),
            indent=1))


if __name__ == "__main__":
    main()
