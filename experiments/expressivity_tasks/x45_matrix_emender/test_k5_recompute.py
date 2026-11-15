"""K5 gate C -- the recompute-over-store backward: same gradients, a card's worth of
memory back.

The store-everything K4 kept each chunk's trajectory alive until its backward ran:
67 MB per chunk at the registered shape, 2.15 GB per layer, 51.5 GB over 24 layers
at microbatch 8 -- more than the card, which is what forced microbatch 2 and the
4.1x launch regression the profile diagnosed.  The recompute form keeps only the
chunk-entry checkpoints (the state and liveness the routing already consumed,
~1 MB per chunk) and re-walks each chunk forward in backward, into a one-chunk
scratch, before the same reverse walk.

What this file establishes:

* **bit-identical gradients**: the recomputed trajectory is
  ``persistent_scan_trajectory``'s own kernel on the same inputs, so every
  backward output -- ``dstate``, ``dweights``, ``ddecay``, ``dpayload`` -- is
  bit-equal to the store form's, which in turn is the K4-gated one (oracle
  within measured floors, state/payload adjoints bit-exact).  A scratch that
  reused freed memory by accident would show up here as a wrong trajectory.
* **the memory claim, measured**: a whole registered-shape scan's forward AND
  backward peak allocation, recompute form against a simulated store form that
  keeps every chunk's trajectory the way K4 did.
* **determinism**: two runs bit-identical, the lane's rule for every kernel.

Every test needs a CUDA device.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import MatrixSchedule
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    fused_block_scan, persistent_scan_backward, persistent_scan_recompute_backward,
    persistent_scan_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.test_hostloop_free import (
    head_major)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    entry_state, routed_fixture)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
GRAD_OUTPUTS = ("dstate", "dweights", "ddecay", "dpayload")
# The registered mixer shape's dimensions (batch 8, 2048 tokens, 2 heads, rank 128).
REG_BATCH, REG_TOKENS, REG_HEADS, REG_RANK, REG_VALUE_DIM, REG_CHUNK = \
    8, 2048, 2, 128, 128, 64


def store_form_grads(state, live, weights, decay, schedule, dreads, dexit):
    """The pre-K5 backward, by hand: store the trajectory, then walk it in reverse."""
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
    """The K5 backward through the public op: checkpoint in, recompute, walk."""
    inputs = [state.detach().clone().requires_grad_(True),
              weights.detach().clone().requires_grad_(True),
              decay.detach().clone().requires_grad_(True),
              schedule.payload.detach().clone().requires_grad_(True)]
    attached = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                              payload=inputs[3], op=schedule.op,
                              live_after=schedule.live_after)
    outputs = fused_block_scan(inputs[0], live, inputs[1], inputs[2], attached)
    grads = torch.autograd.grad(list(outputs[:2]), inputs, [dreads, dexit],
                                allow_unused=True)
    return grads


@needs_cuda
def test_gate_c_recompute_grads_are_bit_identical_to_the_store_form():
    """Every backward output bit-equal to the store form's -- the design's premise.

    The recompute must reproduce the trajectory a storing forward would have
    saved, bit for bit, or the adjoint is somebody else's gradient.  The fixture
    is the lane's canonical routed one (all five ops present).
    """
    fixture = routed_fixture()
    assert fixture.result.stats.chunks, "the fixture must record its chunks"
    weights, decay, schedule = (fixture.weights, head_major(fixture.decay),
                                 fixture.schedule)
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    store = store_form_grads(state, live, weights, decay, schedule, dreads, dexit)
    recomputed = recompute_form_grads(state, live, weights, decay, schedule,
                                      dreads, dexit)
    for name, left, right in zip(GRAD_OUTPUTS, store, recomputed):
        assert torch.equal(left, right), name
        assert float(left.abs().max()) > 0.0, name


@needs_cuda
@pytest.mark.parametrize("rank", [64, 512])
def test_gate_c_recompute_grads_at_both_rank_ends(rank):
    """The register-tile ends, where a wrong RP mask or checkpoint shape breaks."""
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
        random_schedule)
    tokens, value_dim = 32, 16
    schedule, query, _, decay_t = random_schedule(2, tokens, 2, rank, value_dim,
                                                  lanes=2, seed=rank)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay_t)
    batch, heads = weights.shape[0], weights.shape[1]
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=torch.Generator().manual_seed(rank)).to(DEVICE) * 0.3
    live = torch.rand(batch, heads, rank, device=DEVICE) < 0.7
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)
    store = store_form_grads(state, live, weights, decay, schedule, dreads, dexit)
    recomputed = recompute_form_grads(state, live, weights, decay, schedule,
                                      dreads, dexit)
    for name, left, right in zip(GRAD_OUTPUTS, store, recomputed):
        assert torch.equal(left, right), (rank, name)


@needs_cuda
def test_gate_c_peak_memory_is_checkpoints_plus_one_scratch():
    """A whole registered-shape scan's fwd+bwd peak: recompute vs the store form.

    The store form is simulated exactly as K4 ran it: every chunk's trajectory
    kept alive until its backward (they cannot be freed earlier -- each backward
    consumes its own).  The recompute form must peak at a small fraction of it,
    and the absolute bound is stated in the assertion message, not the ether.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy, route_chunk)
    from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan

    generator = torch.Generator().manual_seed(77)
    query = torch.randn(REG_BATCH, REG_TOKENS, REG_HEADS, REG_VALUE_DIM,
                        generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(512, REG_VALUE_DIM, generator=generator).to(DEVICE)
    choice = torch.randint(0, 512, (REG_BATCH, REG_TOKENS, REG_HEADS),
                            generator=generator)
    payload = directions[choice]
    decay = torch.full((REG_BATCH, REG_TOKENS, REG_HEADS), 0.95, device=DEVICE)
    policy = RoutingPolicy(ops=True)

    def peak(scan, trainable):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inputs = [tensor.detach().clone().requires_grad_(True)
                  for tensor in (query, payload, decay)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = scan(inputs[0], inputs[1], inputs[2], router=route_chunk,
                          policy=policy, rank=REG_RANK, rank_max=REG_RANK,
                          chunk_tokens=REG_CHUNK, fused=False,
                          trainable_fused=trainable)
        (result.reads.sum() + result.state.sum()).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2 ** 30

    recompute_peak = peak(routed_chunk_scan, trainable=True)

    def store_form_scan(query_in, payload_in, decay_in, **kwargs):
        """K4's store-everything shape: per-chunk trajectory forward + hand backward."""
        chunk = kwargs["chunk_tokens"]
        batch, tokens, heads, value_dim = payload_in.shape
        reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32,
                            device=DEVICE)
        state = torch.zeros(batch, heads, REG_RANK, value_dim, device=DEVICE)
        live = torch.zeros(batch, heads, REG_RANK, dtype=torch.bool, device=DEVICE)
        saved = []
        for index, start in enumerate(range(0, tokens, chunk)):
            stop = min(start + chunk, tokens)
            qc = query_in[:, start:stop].permute(0, 2, 1, 3)
            pc = payload_in[:, start:stop].permute(0, 2, 1, 3)
            dc = decay_in[:, start:stop].permute(0, 2, 1)
            schedule, weights = route_chunk(index, start, state, live, qc, pc,
                                             kwargs["policy"], REG_RANK)
            piece_reads, state, live, trajectory = persistent_scan_trajectory(
                state, live, weights.to(torch.float32), dc.to(torch.float32),
                schedule)
            reads[:, start:stop] = piece_reads.permute(0, 2, 1, 3)
            saved.append((state, live, weights, dc, schedule, trajectory,
                          start, stop))
        dreads = torch.ones_like(reads).permute(0, 2, 1, 3)
        dstate = torch.zeros_like(state)
        for entry in reversed(saved):
            state_in, live_in, weights, dc, schedule, trajectory, start, stop \
                = entry
            dstate, _, _, _ = persistent_scan_backward(
                trajectory, weights.to(torch.float32), dc.to(torch.float32),
                schedule, dreads[:, :, start:stop, :], dstate)
        return reads

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        store_form_scan(query, payload, decay, chunk_tokens=REG_CHUNK,
                        policy=policy)
    torch.cuda.synchronize()
    store_peak = torch.cuda.max_memory_allocated() / 2 ** 30

    trajectory_gb = (REG_BATCH * REG_HEADS * REG_CHUNK * REG_RANK * REG_VALUE_DIM
                     * 4) / 2 ** 30
    print(f"store-form peak {store_peak:.2f} GB, recompute-form peak "
          f"{recompute_peak:.2f} GB, one chunk's trajectory {trajectory_gb:.3f} GB")
    assert recompute_peak < store_peak / 2, (recompute_peak, store_peak)
    # The absolute claim: the whole scan's fwd+bwd peaks at half a GB -- the
    # outputs, the checkpoint set, one chunk-scratch and the routing's own
    # saved activations -- where the store form holds the 32-chunk trajectory
    # set (2.15 GB at this shape, 51.5 GB over a model's 24 layers at mb8).
    assert recompute_peak < 0.5, recompute_peak


@needs_cuda
def test_gate_c_the_recompute_backward_is_bit_deterministic():
    """Two forward+backward runs bit-identical -- no atomics, fixed reductions."""
    fixture = routed_fixture(tokens=32, chunk_tokens=8)
    weights, decay, schedule = (fixture.weights, head_major(fixture.decay),
                                fixture.schedule)
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=2)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)
    first = recompute_form_grads(state, live, weights, decay, schedule, dreads,
                                 dexit)
    second = recompute_form_grads(state, live, weights, decay, schedule, dreads,
                                  dexit)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
