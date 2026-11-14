"""Gates for K4 -- the fused backward of the matrix store's block scan.

Gate discipline is the lane's (``test_k1_schedule_scan.py``,
``test_hostloop_free.py``): the oracle is an **eager per-token reference under
PyTorch autograd**, the floor is **measured** on the same fixture (fp32 against
a second fp32 ordering and against fp64), and the allowance is
``NOISE_FACTOR`` times that floor -- never a fitted constant.

The four things this file has to establish, one per section:

* **A -- grad parity**: the fp64 ``gradcheck`` validates the eager reference
  itself first (the brief's requirement: validate the oracle before trusting
  it); the fused backward then matches the oracle on every grad output
  (``dstate``, ``dweights``, ``ddecay``, ``dpayload``) within a stated multiple
  of the measured floor, and the forward stays **bit-identical** to
  ``persistent_scan`` -- state and liveness bit-exact, reads unchanged -- before
  and after a backward has run.
* **B -- the structural adjoint cases**: pure decay, append-only, a POP
  mid-block, a MERGE, a CORRECT.  Every case names the failure mode it guards.
* **E -- determinism and pathology**: bit-determinism across runs, rank 64 and
  512, the size gate (rank 2048 refused by name, ``trainable_block_scan``
  falling back to the eager autograd path), and the shape audit.

The oracle, stated once because it is the one place this lane's two eager
references disagree
--------------------------------------------------------------------------------
``algebra.sequential_matrix_scan`` applies a lane's write to **every** head of
the batch whenever *any* head's lane is active; an inactive head's row gets
``rows + payload`` with ``payload == 0`` and ``erase == False``, so the *forward*
is the identity there -- but its autograd would report ``dpayload = dS[j]`` at
that inactive lane.  The registered training path (``scan.chunk_scan``) instead
masks the payload by ``op != NONE``, so its ``dpayload`` at an inactive lane is
exactly zero.  The kernels (K1, K23, the persistent scan) skip inactive lanes
outright, so the zero-gradient convention is the registered semantic, and the
oracle below implements it (an inactive lane is the identity for every head).
Given the schedule producers' invariant -- an inactive lane carries a zero
payload and ``erase == False`` (``random_schedule``, ``route_chunk``,
``writing_schedule`` all maintain it) -- the two references' forward values are
bit-identical; only the inactive lanes' payload gradient differs, and one test
below pins ours to the registered convention.

Every test needs a CUDA device.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, NONE, POP, MERGE, MatrixSchedule, op_histogram,
    sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    persistent_available, persistent_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    check_backward_shapes, eager_block_scan, fused_block_scan,
    persistent_scan_backward, persistent_scan_trajectory, trainable_block_scan)
from experiments.expressivity_tasks.x45_matrix_emender.test_hostloop_free import (
    head_major)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    entry_state, routed_fixture)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, random_schedule)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

GRAD_OUTPUTS = ("dstate", "dweights", "ddecay", "dpayload")


# --------------------------------------------------------------------- the oracle

def eager_block_walk(state, weights, decay, schedule, read_form="einsum"):
    """The autograd oracle: the per-token walk, inactive lanes masked per head.

    ``state`` ``(batch, heads, rank, d_v)``, ``weights`` ``(batch, heads,
    tokens, rank)``, ``decay`` ``(batch, heads, tokens)``, ``schedule`` the
    block's recorded decision channel (its payload is the differentiated one).
    Returns ``(reads, exit_state)`` in the block paths' layout.  Values are the
    oracle of record's (the read's reduction order differs only through
    ``read_form``, which is how the fp32 floor's second ordering is taken).
    """
    batch, heads, tokens, rank = weights.shape
    value_dim = state.shape[-1]
    b_index = torch.arange(batch, device=state.device)[:, None]
    h_index = torch.arange(heads, device=state.device)[None, :]
    reads = []
    for t in range(tokens):
        if read_form == "einsum":
            reads.append(torch.einsum("bhr,bhrv->bhv", weights[:, :, t], state))
        else:
            reads.append((state.transpose(-1, -2)
                          @ weights[:, :, t].unsqueeze(-1)).squeeze(-1))
        state = state * decay[:, :, t][:, :, None, None]
        for lane in range(schedule.lanes):
            op = schedule.op[:, t, :, lane]
            if not bool(op.ne(NONE).any()):
                continue
            slot = schedule.slot[:, t, :, lane]
            erase = schedule.erase[:, t, :, lane]
            payload = schedule.payload[:, t, :, lane]
            active = op.ne(NONE)
            rows = state[b_index, h_index, slot]
            written = torch.where(erase[:, :, None], payload, rows + payload)
            # An inactive lane is the identity for its head: the registered
            # chunk path masks its payload, so its payload gradient is zero.
            written = torch.where(active[:, :, None], written, rows)
            state = state.index_put((b_index, h_index, slot), written)
    return torch.stack(reads, dim=2), state


def grads_of(walk, inputs, schedule, dreads, dexit, dtype=torch.float32):
    """Run ``walk`` on ``inputs`` and return ``(outputs, grads)`` in GRAD_OUTPUTS order."""
    state, weights, decay, payload = inputs
    attached = [tensor.detach().clone().to(dtype).requires_grad_(True)
                for tensor in (state, weights, decay, payload)]
    attached_schedule = MatrixSchedule(
        slot=schedule.slot, erase=schedule.erase, payload=attached[3],
        op=schedule.op, live_after=schedule.live_after)
    outputs = walk(attached[0], attached[1], attached[2], attached_schedule)
    grads = torch.autograd.grad(list(outputs), attached, [dreads, dexit],
                                allow_unused=True)
    return outputs, grads


def fused_grads(state, live, weights, decay, schedule, dreads, dexit):
    """The fused path's grads, by the same contract as :func:`grads_of`."""
    inputs = [state.detach().clone().requires_grad_(True),
              weights.detach().clone().requires_grad_(True),
              decay.detach().clone().requires_grad_(True),
              schedule.payload.detach().clone().requires_grad_(True)]
    attached_schedule = MatrixSchedule(
        slot=schedule.slot, erase=schedule.erase, payload=inputs[3],
        op=schedule.op, live_after=schedule.live_after)
    outputs = fused_block_scan(inputs[0], live, inputs[1], inputs[2],
                               attached_schedule)
    # The third output is the bool liveness channel: no gradient, by construction.
    grads = torch.autograd.grad(list(outputs[:2]), inputs, [dreads, dexit])
    return outputs, grads


def grad_floor(state, live, weights, decay, schedule, dreads, dexit):
    """The measured fp32 grad noise of the oracle, per output.

    Two fp32 orderings (the read as an einsum and as a matmul) against each
    other, and the einsum one against the same walk in fp64.  The floor is
    never allowed below one ulp of the output's own scale, so an
    exactly-representable gradient cannot manufacture a zero allowance.
    """
    _, order_a = grads_of(eager_block_walk, (state, weights, decay, schedule.payload),
                          schedule, dreads, dexit)
    _, order_b = grads_of(
        lambda *args: eager_block_walk(*args, read_form="matmul"),
        (state, weights, decay, schedule.payload), schedule, dreads, dexit)
    _, exact = grads_of(eager_block_walk, (state, weights, decay, schedule.payload),
                        schedule, dreads.to(torch.float64), dexit.to(torch.float64),
                        dtype=torch.float64)
    floors = {}
    for index, name in enumerate(GRAD_OUTPUTS):
        first = order_a[index].double()
        floor = max(float((first - order_b[index].double()).abs().max()),
                    float((first - exact[index].double()).abs().max()))
        scale = float(first.abs().max())
        floors[name] = max(floor, torch.finfo(torch.float32).eps * scale)
    return floors, order_a


def assert_within_floor(name, observed, expected, floor, factor=NOISE_FACTOR):
    """``observed`` within ``factor`` measured ``floor`` of ``expected``; report both."""
    deviation = (observed.double() - expected.double()).abs()
    peak, median = float(deviation.max()), float(deviation.median())
    scale = float(expected.double().abs().max())
    print(f"{name}: max {peak:.3e} ({peak / floor:.2f}x floor) median {median:.3e} "
          f"scale {scale:.3e}")
    assert peak <= factor * floor, (name, peak, factor * floor)
    assert median <= factor * floor, (name, median, factor * floor)
    return peak, median


# ------------------------------------------------------------------ scripted blocks

def scripted_block(tokens, ops, slots, heads=1, batch=1, rank=None, value_dim=4,
                   seed=3, payload_scale=1.0):
    """A one-lane scripted schedule: ``ops[t]`` applied to ``slots[t]``.

    The DECAY gate is random per token in [0.7, 1.0), so a wrong decay weight
    cannot hide behind a constant.  POP and inactive lanes carry a zero payload
    and APPEND/CORRECT carry a random one -- the producers' invariant.
    """
    generator = torch.Generator().manual_seed(seed)
    rank = len(set(slots)) if rank is None else rank
    shape = (batch, tokens, heads, 1)
    op = torch.tensor(ops, dtype=torch.int64).view(1, tokens, 1, 1).expand(shape)
    slot = torch.tensor(slots, dtype=torch.int64).view(1, tokens, 1, 1).expand(shape)
    erase = (op == CORRECT) | (op == POP) | (op == MERGE)
    live_after = op.ne(NONE) & op.ne(POP)
    payload = torch.randn(*shape, value_dim, generator=generator) * payload_scale
    carried = ((op == APPEND) | (op == CORRECT)).unsqueeze(-1)
    payload = torch.where(carried, payload, torch.zeros_like(payload))
    schedule = MatrixSchedule(slot=slot.contiguous().to(DEVICE),
                             erase=erase.contiguous().to(DEVICE),
                             payload=payload.to(DEVICE),
                             op=op.contiguous().to(DEVICE),
                             live_after=live_after.contiguous().to(DEVICE))
    weights = torch.rand(batch, heads, tokens, rank, generator=generator).to(DEVICE)
    decay = (0.7 + 0.3 * torch.rand(batch, heads, tokens,
                                    generator=generator)).to(DEVICE)
    return schedule, weights, decay


def empty_store(batch, heads, rank, value_dim):
    return (torch.zeros(batch, heads, rank, value_dim, device=DEVICE),
            torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE))


def write_payload_closed_form(weights, decay, dreads, dexit, write_token, slot):
    """The payload gradient of one write at ``write_token`` to ``slot``.

    Reads strictly after the write see the payload decayed by every gate
    between the write and the read; the exit sees it decayed by every gate
    after the write.  Nothing else touches it (this is the rank-1 adjoint of a
    single row write), so the closed form is exact up to fp32 grouping.
    """
    batch, heads, tokens, rank = weights.shape
    value_dim = dexit.shape[-1]
    cum = torch.ones(batch, heads, tokens + 1, device=DEVICE)
    cum[:, :, 1:] = torch.cumprod(decay, dim=-1)
    share = cum[:, :, write_token + 1:] / cum[:, :, write_token + 1, None]
    # share[s - write_token - 1] = prod of lam over (write_token, s)
    dp = torch.einsum("bhs,bhs,bhsv->bhv",
                      share[:, :, :tokens - write_token - 1],
                      weights[:, :, write_token + 1:, slot],
                      dreads[:, :, write_token + 1:])
    dp = dp + share[:, :, -1, None] * dexit[:, :, slot]
    return dp


# ------------------------------------------------------------------------ gate A

@needs_cuda
def test_the_fp64_gradcheck_validates_the_eager_reference():
    """The oracle is validated before it is trusted: fp64 gradcheck on all adjoint cases.

    Failure mode guarded: a reference whose autograd disagrees with its own
    forward (a ``where`` branch or an in-place write silently dropping a
    gradient) would make every later parity number meaningless.  The fixture
    covers every adjoint branch: NONE, APPEND (add), CORRECT (erase + payload)
    and POP (erase + zero payload) over two lanes.
    """
    schedule, query, _, decay_t = random_schedule(1, 6, 1, 4, 3, lanes=2, seed=5)
    state = torch.randn(1, 1, 4, 3, device=DEVICE)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = decay_t.permute(0, 2, 1).contiguous()

    def walk(state_, weights_, decay_, payload_):
        masked = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                                payload=payload_, op=schedule.op,
                                live_after=schedule.live_after)
        return eager_block_walk(state_, weights_, decay_, masked)

    inputs = [state.double().requires_grad_(True),
              weights.double().requires_grad_(True),
              decay.double().requires_grad_(True),
              schedule.payload.double().requires_grad_(True)]
    assert torch.autograd.gradcheck(walk, inputs, check_undefined_grad=False)


@needs_cuda
def test_the_trajectory_forward_is_bit_identical_and_unchanged_by_the_backward():
    """The backward must not perturb the forward path: reads, state, liveness.

    ``persistent_scan_trajectory`` is the forward's walk plus a store of the
    register tile, so all three outputs must be *equal* to ``persistent_scan``'s,
    before and after a full fused backward has run on the same inputs -- a
    kernel that wrote an input (or an output buffer it does not own) would show
    up here as a changed second answer.
    """
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)

    plain = persistent_scan(state, live, weights, decay, schedule)
    recorded = persistent_scan_trajectory(state, live, weights, decay, schedule)
    for left, right in zip(plain, recorded[:3]):
        assert torch.equal(left, right)

    dreads = torch.randn(batch, heads, schedule.tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)
    outputs, _ = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    for left, right in zip(plain, outputs):
        assert torch.equal(left, right)

    again = persistent_scan(state, live, weights, decay, schedule)
    for left, right in zip(plain, again):
        assert torch.equal(left, right)
    # And the fused Function's own outputs are the same walk as well.
    for left, right in zip(plain, outputs):
        assert torch.equal(left, right)


@needs_cuda
def test_fused_grads_match_the_autograd_oracle_on_the_routed_fixture():
    """Gate A: every grad output within ``NOISE_FACTOR`` of the measured floor.

    The fixture is the lane's canonical one (real ``route_chunk`` routing, all
    five ops present, histogram asserted below); the floor is measured on the
    same fixture first, then each fused grad is compared to the oracle's.
    """
    fixture = routed_fixture()
    assert op_histogram(fixture.schedule)["merge"] > 0
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    floors, oracle = grad_floor(state, live, weights, decay, schedule, dreads, dexit)
    assert all(floor > 0 for floor in floors.values())
    _, oracle_state_only = eager_block_walk(
        state, weights, decay, MatrixSchedule(
            slot=schedule.slot, erase=schedule.erase,
            payload=schedule.payload.detach(), op=schedule.op,
            live_after=schedule.live_after))
    _, _, oracle_live, _ = sequential_matrix_scan(
        weights.permute(0, 2, 1, 3), decay.permute(0, 2, 1), schedule,
        initial_state=state.clone(), initial_live=live, return_liveness=True)

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    # The forward stays the oracle's: state and liveness bit-exact, and the reads
    # bit-identical to the forward's own kernel (that is the "reads unchanged"
    # claim -- the fused backward must not change a forward output bit).
    plain = persistent_scan(state, live, weights, decay, schedule)
    assert torch.equal(outputs[1], oracle_state_only)
    assert torch.equal(outputs[2], oracle_live)
    for observed, reference in zip(outputs, plain):
        assert torch.equal(observed, reference)

    for index, name in enumerate(GRAD_OUTPUTS):
        assert_within_floor(name, grads[index], oracle[index], floors[name])
    # The two outputs whose adjoint arithmetic is elementwise (the state chain
    # and the payload extraction) are expected to be *bit-exact* against the
    # oracle: same rounding, different order of the same operations.  Asserting
    # it keeps a future refactor from silently changing their rounding class.
    assert torch.equal(grads[0], oracle[0]), "dstate"
    assert torch.equal(grads[3], oracle[3]), "dpayload"


# ------------------------------------------------------------------------ gate B

@needs_cuda
def test_no_op_block_adjoint_is_pure_elementwise_scaling():
    """Case (i): a no-op block's adjoint is the closed form of a pure decay chain.

    Failure mode guarded: a DECAY weight applied one token too early or too
    late (``lam^t`` vs ``lam^(t-1)``) leaves the state gradient almost right,
    and only the ``dlam`` closed form -- which divides by the token's own
    gate -- exposes the boundary.
    """
    tokens, rank, value_dim, heads = 9, 5, 4, 2
    schedule, weights, decay = scripted_block(tokens, [NONE] * tokens, [0] * tokens,
                                              heads=heads, rank=rank,
                                              value_dim=value_dim)
    batch = 1
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=torch.Generator().manual_seed(4)).to(DEVICE) * 0.5
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    # Closed forms: with cum[s] = prod_{u<s} lam_u, reads_s = w_s . (cum[s] S0)
    # and the exit is cum[T] S0.
    cum = torch.ones(batch, heads, tokens + 1, device=DEVICE)
    cum[:, :, 1:] = torch.cumprod(decay, dim=-1)
    reads = torch.stack(
        [torch.einsum("bhr,bhrv->bhv", weights[:, :, t], state) * cum[:, :, t, None]
         for t in range(tokens)], dim=2)
    exit_state = cum[:, :, tokens, None, None] * state
    dstate = torch.einsum("bhsr,bhsv->bhrv",
                          weights * cum[:, :, :tokens, None], dreads) \
        + cum[:, :, tokens, None, None] * dexit
    # dlam_t: every read strictly after t (and the exit) depends on lam_t through
    # its cumulative decay, linearly, so dlam_t is their inner product over lam_t.
    read_inner = torch.stack([(dreads[:, :, s] * reads[:, :, s]).sum(dim=-1)
                              for s in range(tokens)], dim=-1)   # (b, h, tokens)
    exit_inner = (dexit * exit_state).sum(dim=(-1, -2))           # (b, h)
    ddecay = torch.zeros(batch, heads, tokens, device=DEVICE)
    for t in range(tokens):
        after_t = read_inner[:, :, t + 1:].sum(dim=-1) + exit_inner
        ddecay[:, :, t] = after_t / decay[:, :, t]

    scale = max(1.0, float(dstate.abs().max()), float(ddecay.abs().max()))
    for name, observed, expected in (("dstate", grads[0], dstate),
                                     ("ddecay", grads[2], ddecay)):
        deviation = float((observed - expected).abs().max())
        print(f"{name}: closed-form deviation {deviation:.3e} on scale {scale:.3e}")
        assert deviation <= 1e-5 * scale, (name, deviation)
    # A no-op block's payloads are never touched: their gradient is exactly zero.
    assert float(grads[3].abs().max()) == 0.0
    # ... and the same closed form agrees with the oracle (the check on the check).
    _, oracle = grads_of(eager_block_walk, (state, weights, decay, schedule.payload),
                         schedule, dreads, dexit)
    for name, closed, seen in (("dstate", dstate, oracle[0]),
                               ("ddecay", ddecay, oracle[2])):
        assert float((closed - seen).abs().max()) <= 1e-5 * scale, name


@needs_cuda
def test_append_only_payload_grad_is_the_write_closed_form():
    """Case (ii): append-only -- every payload's grad is the single-write closed form.

    Failure mode guarded: the payload gradient missing the decay weighting
    between its write and each later read, or including the token's *own* read
    (which happens before the write and must not see the payload).
    """
    tokens, rank, value_dim = 8, 8, 4
    slots = list(range(tokens))
    schedule, weights, decay = scripted_block(tokens, [APPEND] * tokens, slots,
                                             rank=rank, value_dim=value_dim)
    batch = heads = 1
    state, live = empty_store(batch, heads, rank, value_dim)
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    scale = max(1.0, float(grads[3].abs().max()))
    for t, slot in enumerate(slots):
        expected = write_payload_closed_form(weights, decay, dreads, dexit, t, slot)
        observed = grads[3][:, t, :, 0]
        deviation = float((observed - expected).abs().max())
        assert deviation <= 1e-5 * scale, (t, deviation)
    # The entry state's adjoint is the pure-decay chain even though every token
    # writes: an APPEND passes the row's gradient through unchanged, so the
    # writes add content but never perturb the entry rows' coefficients.
    cum = torch.ones(batch, heads, tokens + 1, device=DEVICE)
    cum[:, :, 1:] = torch.cumprod(decay, dim=-1)
    entry = torch.einsum("bhsr,bhsv->bhrv", weights * cum[:, :, :tokens, None],
                         dreads) + cum[:, :, tokens, None, None] * dexit
    scale = max(scale, float(entry.abs().max()))
    assert float((grads[0] - entry).abs().max()) <= 1e-5 * scale, \
        "the entry state's adjoint is not the pure-decay chain"


@needs_cuda
def test_a_popped_slots_grad_terminates_at_its_death_token():
    """Case (iii): a slot popped mid-block -- its grad dies at the POP, exactly.

    Failure mode guarded: an adjoint that lets the payload's (or the entry
    row's) gradient leak *past* the POP.  The erase in the forward discards the
    row's content, so the adjoint must zero the row: with upstream gradient
    only at and after the death token, the payload written before it and the
    entry row itself must receive **exactly zero**, while upstream gradient
    between write and death reaches the payload normally.
    """
    tokens, rank, value_dim, death = 9, 6, 4, 5
    slots = [2] * tokens
    ops = [APPEND] * tokens
    ops[1] = APPEND            # the payload under test is written at token 1
    ops[death] = POP            # ... and slot 2 dies at token `death`
    schedule, weights, decay = scripted_block(tokens, ops, slots, rank=rank,
                                             value_dim=value_dim)
    batch = heads = 1
    state, live = empty_store(batch, heads, rank, value_dim)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    # Gradient only strictly after the death token: the read at the death
    # token itself is *pre-write* (the oracle's order: read, DECAY, lanes), so
    # it still sees the popped row -- the boundary this test exists to name.
    # From the token after the death on, the popped row's content (entry +
    # payload) is invisible, so both grads must be exactly zero -- not small,
    # zero.
    late = torch.zeros(batch, heads, tokens, value_dim, device=DEVICE)
    late[:, :, death + 1:] = torch.randn(batch, heads, tokens - death - 1, value_dim,
                                         generator=torch.Generator().manual_seed(7)).to(DEVICE)
    outputs, grads = fused_grads(state, live, weights, decay, schedule, late, dexit)
    assert float(grads[3][:, 1].abs().max()) == 0.0, "payload grad leaked past the POP"
    assert float(grads[0][:, :, 2].abs().max()) == 0.0, "entry row grad leaked past the POP"
    # Other rows are untouched by the POP: their adjoint is the pure-decay chain.
    assert float(grads[0][:, :, 3:].abs().max()) > 0.0

    # Gradient strictly between the write and the death reaches the payload --
    # and the closed form of a single write, because nothing else touches slot 2
    # before the death.
    early = torch.zeros(batch, heads, tokens, value_dim, device=DEVICE)
    early[:, :, 2:death] = torch.randn(batch, heads, death - 2, value_dim,
                                       generator=torch.Generator().manual_seed(8)).to(DEVICE)
    zero_exit = torch.zeros_like(dexit)
    outputs, grads = fused_grads(state, live, weights, decay, schedule, early, zero_exit)
    expected = write_payload_closed_form(weights, decay, early, zero_exit, 1, 2)
    scale = max(1.0, float(grads[3].abs().max()))
    deviation = float((grads[3][:, 1] - expected).abs().max())
    print(f"payload grad before the death: closed-form deviation {deviation:.3e}")
    assert deviation <= 1e-5 * scale
    assert float(grads[3][:, 1].abs().max()) > 0.0


@needs_cuda
def test_merge_grad_reaches_both_source_slots():
    """Case (iv): a MERGE -- the gradient reaches BOTH source rows.

    At the block boundary a MERGE is two recorded lanes: an erasing write of
    the *blend* to the target slot, and a POP of the source slot.  The source
    row's own content path dies at the POP, so the only route to the source row
    is the blend payload -- exactly as ``route_chunk`` builds it (content
    attached, blend weights detached).  The test builds that payload from the
    *attached entry state* and asserts both source rows receive gradient.
    Failure mode guarded: the blend payload's gradient dropping one of the two
    source rows (a one-lane adjoint applied to a two-lane op).
    """
    tokens, rank, value_dim, merge_token, target, source = 8, 6, 4, 3, 1, 4
    batch = heads = 1
    generator = torch.Generator().manual_seed(6)
    shape = (batch, tokens, heads, 2)
    slot = torch.zeros(shape, dtype=torch.int64)
    slot[0, merge_token, 0, 0] = target
    slot[0, merge_token, 0, 1] = source
    op = torch.full(shape, NONE, dtype=torch.int64)
    op[0, merge_token, 0, 0] = MERGE
    op[0, merge_token, 0, 1] = MERGE
    erase = (op == MERGE)
    live_after = torch.zeros(shape, dtype=torch.bool)
    live_after[0, merge_token, 0, 0] = True
    weights = torch.rand(batch, heads, tokens, rank, generator=generator).to(DEVICE)
    decay = (0.7 + 0.3 * torch.rand(batch, heads, tokens,
                                    generator=generator)).to(DEVICE)
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.5
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    state = state * live.unsqueeze(-1)

    # The recorded MERGE payload: a norm-weighted blend of the two source rows,
    # weights detached, content attached -- the arm's own construction.
    first, second = state.detach().clone(), state.detach().clone()
    strengths = first.norm(dim=-1) + second.norm(dim=-1)
    share = (first.norm(dim=-1) / strengths).detach()
    blend = (share[0, 0, target] * state[0, 0, target]
             + (1 - share[0, 0, target]) * state[0, 0, source]).unsqueeze(0)
    payload = torch.zeros(batch, tokens, heads, 2, value_dim, device=DEVICE)
    payload[0, merge_token, 0, 0] = blend[0]
    schedule = MatrixSchedule(slot=slot.to(DEVICE), erase=erase.to(DEVICE),
                              payload=payload, op=op.to(DEVICE),
                              live_after=live_after.to(DEVICE))

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    assert float(grads[0][:, :, target].abs().max()) > 0.0, "the target row's grad"
    assert float(grads[0][:, :, source].abs().max()) > 0.0, \
        "the source row's grad (only the blend payload can carry it)"
    # Both source rows also carry it in the oracle -- and everything else agrees.
    _, oracle = grads_of(eager_block_walk, (state, weights, decay, payload),
                         schedule, dreads, dexit)
    floors, _ = grad_floor(state, live, weights, decay, schedule, dreads, dexit)
    for index, name in enumerate(GRAD_OUTPUTS):
        assert_within_floor(name, grads[index], oracle[index], floors[name])
    assert float(oracle[0][:, :, source].abs().max()) > 0.0


@needs_cuda
def test_correct_adjoint_is_the_rank_one_closed_form():
    """Case (v): a CORRECT -- rank-1 adjoint against the closed form.

    A CORRECT at token ``t0`` replaces one row, so with upstream gradient only
    after ``t0``: the payload's grad is the single-write closed form (and
    nothing else in the block produces a payload grad), the overwritten row's
    *incoming* gradient is exactly zero (the old content is discarded), and
    every other row follows the pure-decay chain.  Failure mode guarded: the
    erase adjoint failing to terminate the discarded row's gradient, or the
    payload missing the decay weighting to the exit.
    """
    tokens, rank, value_dim, t0, fixed = 9, 6, 4, 4, 2
    ops = [NONE] * tokens
    ops[t0] = CORRECT
    slots = [fixed] * tokens
    schedule, weights, decay = scripted_block(tokens, ops, slots, rank=rank,
                                             value_dim=value_dim)
    batch = heads = 1
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=torch.Generator().manual_seed(9)).to(DEVICE) * 0.5
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    # Upstream gradient only strictly after the CORRECT: the read at ``t0`` is
    # pre-write, so it (and every earlier read) still sees the old row.
    dreads[:, :, :t0 + 1] = 0.0
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    expected = write_payload_closed_form(weights, decay, dreads, dexit, t0, fixed)
    scale = max(1.0, float(grads[3].abs().max()))
    deviation = float((grads[3][:, t0, :, 0] - expected).abs().max())
    print(f"correct payload grad: closed-form deviation {deviation:.3e}")
    assert deviation <= 1e-5 * scale, deviation
    # The discarded row's incoming gradient terminates at the CORRECT.
    assert float(grads[0][:, :, fixed].abs().max()) == 0.0, \
        "the overwritten row's gradient did not terminate"
    # Every other row is the pure-decay chain.
    cum = torch.ones(batch, heads, tokens + 1, device=DEVICE)
    cum[:, :, 1:] = torch.cumprod(decay, dim=-1)
    others = torch.einsum("bhsr,bhsv->bhrv", weights * cum[:, :, :tokens, None],
                          dreads) + cum[:, :, tokens, None, None] * dexit
    assert float((grads[0][:, :, :fixed] - others[:, :, :fixed]).abs().max()) \
        <= 1e-5 * scale
    assert float((grads[0][:, :, fixed + 1:] - others[:, :, fixed + 1:]).abs().max()) \
        <= 1e-5 * scale
    assert float(grads[3].abs().sum() - grads[3][:, t0].abs().sum()) == 0.0, \
        "a lane other than the CORRECT produced a payload grad"


@needs_cuda
def test_inactive_lane_payload_grads_are_zero_like_the_registered_chunk_path():
    """The convention this file's oracle takes, pinned against the registered path.

    ``sequential_matrix_scan`` applies a lane's write to every head whenever
    any head's lane is active, so its autograd would report a *nonzero* payload
    gradient at an inactive lane (an identity write is still a write of
    ``rows + 0``).  The registered training path (``scan.chunk_scan``, and the
    kernels) masks inactive lanes, so the payload gradient there is exactly
    zero -- and zero is what the fused backward must produce.  Failure mode
    guarded: a fused backward inventing payload gradient where the registered
    backward has none would silently train values that were dropped.
    """
    fixture = routed_fixture(tokens=32, chunk_tokens=8)
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    assert bool((schedule.op == NONE).any()), "the fixture must have inactive lanes"
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=2)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    inactive = (schedule.op == NONE)
    assert float(grads[3][inactive].abs().max()) == 0.0, \
        "an inactive lane produced a payload gradient"

    # The registered path (autograd through the eager chunk scan) agrees on
    # those lanes and on the rest of the payload gradient.
    inputs = [state.detach().clone().requires_grad_(True),
              weights.detach().clone().requires_grad_(True),
              decay.detach().clone().requires_grad_(True),
              schedule.payload.detach().clone().requires_grad_(True)]
    attached = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                              payload=inputs[3], op=schedule.op,
                              live_after=schedule.live_after)
    chunk_reads, chunk_state, _ = eager_block_scan(inputs[0], live, inputs[1],
                                                   inputs[2], attached, chunk_tokens=8)
    chunk_grads = torch.autograd.grad([chunk_reads, chunk_state], inputs,
                                      [dreads, dexit])
    assert float(chunk_grads[3][inactive].abs().max()) == 0.0
    scale = max(1.0, float(grads[3].abs().max()))
    assert float((grads[3] - chunk_grads[3]).abs().max()) <= 1e-4 * scale, \
        "the fused and registered payload grads disagree beyond the chunk grouping"


# ------------------------------------------------------------------------ gate E

@needs_cuda
def test_the_fused_backward_is_bit_deterministic_across_runs():
    """Two full forward+backward runs are bit-identical: no atomics, fixed reductions.

    The one place a reverse walk could lose determinism is the two cross-block
    reductions (``dw``, ``dlam``): they are summed over the block axis in a
    fixed order rather than accumulated atomically, so re-running must
    reproduce every gradient bit for bit.
    """
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = entry_state(batch, heads, rank, value_dim, seed=3)
    tokens = schedule.tokens
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    first = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    second = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    for left, right in zip(first[1], second[1]):
        assert torch.equal(left, right)
    for left, right in zip(first[0], second[0]):
        assert torch.equal(left, right)
    assert float(first[1][0].abs().max()) > 0.0


@needs_cuda
@pytest.mark.parametrize("rank", [64, 512])
def test_rank_at_both_ends_of_the_persistent_kernels_range(rank):
    """Rank 64 (a 1 024-float tile) and 512 (at the budget): grads within the floor.

    The two ends of the register-tile range, where a wrong ``RP`` mask or a
    value-block offset would break first.  The forward's state must stay
    bit-exact against the per-token oracle at both, as ``NO_HOST_LOOP.md``
    established for the forward alone.
    """
    tokens, value_dim = 32, 16
    schedule, query, _, decay_t = random_schedule(2, tokens, 2, rank, value_dim,
                                                  lanes=2, seed=rank)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay_t)
    batch, heads = weights.shape[0], weights.shape[1]
    state, live = empty_store(batch, heads, rank, value_dim)
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    floors, oracle = grad_floor(state, live, weights, decay, schedule, dreads, dexit)
    outputs, grads = fused_grads(state, live, weights, decay, schedule, dreads, dexit)
    _, oracle_state = sequential_matrix_scan(
        weights.permute(0, 2, 1, 3), decay.permute(0, 2, 1), schedule,
        initial_state=state.clone(), initial_live=live)
    assert torch.equal(outputs[1], oracle_state), rank
    for index, name in enumerate(GRAD_OUTPUTS):
        assert_within_floor(f"rank{rank} {name}", grads[index], oracle[index],
                            floors[name])


@needs_cuda
def test_rank_2048_is_refused_by_name_and_auto_falls_back():
    """The size gate, forward and backward: refused by name, fallback is the eager path.

    Rank 2048 needs 32 768 register-resident floats -- four times the tile
    budget -- so both the trajectory forward and the fused backward must refuse
    it with the forward's own message, and ``trainable_block_scan(fused=True)``
    must fall back to the registered eager autograd path rather than fail.
    """
    tokens, rank, value_dim = 8, 2048, 8
    schedule, query, _, decay_t = random_schedule(1, tokens, 1, rank, value_dim,
                                                  lanes=2, seed=7)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay_t)
    state, live = empty_store(1, 1, rank, value_dim)
    assert not persistent_available(rank, value_dim)
    with pytest.raises(ValueError, match="does not fit, so it must take the chunked path"):
        persistent_scan_trajectory(state, live, weights, decay, schedule)
    with pytest.raises(ValueError, match="does not fit"):
        fused_block_scan(state.requires_grad_(True), live, weights, decay, schedule)

    # ``auto`` (trainable_block_scan) falls back to the eager autograd path and
    # a gradient still flows -- that is what "fall back" has to mean here.
    inputs = [state.detach().requires_grad_(True),
              weights.detach().requires_grad_(True),
              decay.detach().requires_grad_(True),
              schedule.payload.detach().requires_grad_(True)]
    attached = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                              payload=inputs[3], op=schedule.op,
                              live_after=schedule.live_after)
    reads, exit_state, exit_live = trainable_block_scan(
        inputs[0], live, inputs[1], inputs[2], attached, fused=True, chunk_tokens=4)
    grads = torch.autograd.grad([reads, exit_state], inputs,
                                [torch.ones_like(reads), torch.ones_like(exit_state)])
    assert all(torch.isfinite(g).all() and float(g.abs().max()) > 0 for g in grads)
    expected = eager_block_scan(state, live, weights, decay, schedule, chunk_tokens=4)
    assert torch.equal(reads, expected[0]) and torch.equal(exit_state, expected[1])


@needs_cuda
def test_the_shape_audit_refuses_before_any_launch():
    """A backward whose inputs do not belong to the block is refused by name."""
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    batch, heads, rank = weights.shape[0], weights.shape[1], weights.shape[3]
    value_dim = schedule.payload.shape[-1]
    state, live = empty_store(batch, heads, rank, value_dim)
    _, _, _, trajectory = persistent_scan_trajectory(state, live, weights, decay,
                                                     schedule)
    dreads = torch.zeros(batch, heads, schedule.tokens, value_dim, device=DEVICE)
    dexit = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
    with pytest.raises(ValueError, match="dreads must be"):
        check_backward_shapes(trajectory, weights, decay, schedule,
                              dreads[:, :, :-1], dexit)
    with pytest.raises(ValueError, match="weights .* do not match"):
        check_backward_shapes(trajectory, weights[:, :, :, :-1].contiguous(), decay,
                              schedule, dreads, dexit)
    with pytest.raises(ValueError, match="must be fp32"):
        check_backward_shapes(trajectory, weights, decay, schedule,
                              dreads.double(), dexit)
    # And the audit fires through the public entry point too.
    with pytest.raises(ValueError):
        fused_block_scan(state, live, weights[:, :, :, :-1], decay, schedule)
