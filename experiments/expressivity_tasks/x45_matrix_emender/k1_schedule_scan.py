"""K1 — the schedule-decoded state scan of the matrix store as one fused kernel.

Why this file exists
--------------------
``scan.chunk_scan`` is written as ~58 eager torch calls per chunk (a gather of the
addressed rows, a ``[steps x steps]`` coupling matrix, a triangular solve, two
reads of ``delta``, a scatter-add, an ``amax`` liveness reduce).  Measured on one
block call it costs 21.7 ms of *device* time inside 113 ms of wall time: the memory
block is host/launch bound, 5.2x (``RESULT_MATRIX_EMENDER.md`` §6).  Almost all of
those ops are functions of the *fixed schedule* and the chunk-entry state, so they
belong in one kernel, exactly as ``x44_chunk_scan/triton_scan.py`` did for the tape
store (7 381 -> 543 kernels per chunk, bit-exact strength/liveness/death codes).

Parallelism: **one program per ``(row, head, slot_block)``**, where ``row`` indexes
``(batch, head)``.  Each program holds a ``[BS, V]`` tile of the state in registers
and walks the chunk's ``C`` tokens (``C * lanes`` steps) in order, because the
liveness channel is per-slot sequential: which lane touches a slot last is what
decides whether it is live at the chunk exit.  Nothing of shape
``[rows, steps, rank]`` is ever materialised -- the schedule is read as scalar data
and the addressed slot is a predicate against the lane index.

Semantics (the oracle is ``algebra.sequential_matrix_scan``)
-----------------------------------------------------------
Per token the reference applies, in this order: decay every row, then every lane in
order (``S[j] = erase ? v : S[j] + v``).  The fused walk performs the same
operations with the same rounding:

    v  <- mul_rn(v, lam)                          every step, whole tile
    v  <- erase ? p : add_rn(v, p)                at the addressed slot only
    live <- live_after                            at the addressed slot only

with ``mul.rn.f32``/``add.rn.f32``/``sub.rn.f32`` inline asm, so the compiler
cannot contract ``v*lam + p`` into an FMA and move the trajectory by 1 ulp (that
exact failure is recorded in ``x44/RESULT_CHUNK_SCAN.md`` §6.2: one ulp flipped a
death boundary and took the read off by three orders of magnitude).  The state, the
liveness channel and therefore every death boundary are **bit-exact** against the
sequential oracle on the same schedule; see ``test_k1_schedule_scan.py``.

The injection list
------------------
The in-chunk writes are emitted as an *injection list*: for every step ``u`` the
relaxation payload ``d_u = p_u - erase_u * v_u`` together with the step's cumulative
log decay ``gc_u``.  Then, exactly as in ``scan.chunk_scan``,

    S_s = exp(gc_s) H + sum_{u<=s} exp(gc_s - gc_u) k_{j_u} d_u^T
    gc_u = sum_{w<=u} log lam_w                      (lam sits on lane 0 of a token)

so ``gate_u = exp(gc_ref - gc_u)`` is the decay weight of that injection and the
read at token ``t`` is ``exp(gc_read) w_t^T H + sum_{u<read} exp(gc_read-gc_u)
w_t[j_u] d_u``.  ``_reads_from_injections`` performs exactly that reduction with
torch; turning it into a second kernel is the next stage (K2) and is deliberately
*not* done here.

Channels kept single-sourced
----------------------------
``read.py`` remains the only definition of the read *weights*; this file only
reduces them.  The matrix store has no separate strength scalar -- its strength is
``||S[j]||`` (``read.row_strength``) -- so the kernel emits ``||S[j]|| * live`` as
the exit strength channel for the *next* chunk's router and for diagnostics; it is
derived from the same registers the exit state is stored from.  The tape store's
separate ``strength``/``age`` scalars, its ``pop_factor`` and its ``free_threshold``
death rule have no analogue here: in this algebra a slot dies exactly when the
schedule says ``live_after=False`` (a POP), so the death code *is* the schedule's
and no threshold is invented (see ``K1_SCHEDULE_SCAN.md``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .algebra import NONE, MatrixSchedule

# ``algebra.NONE`` as the kernel sees it: Triton only reads module-level constants
# that are instantiated as ``tl.constexpr``.
NONE_CODE = tl.constexpr(int(NONE))

# The largest value dim the [BS, V] register tile is compiled for.  The arm runs
# V = 128; the ceiling is a register budget, not a semantic limit.
MAX_TILE_VALUE_DIM = 256


FORWARD_ONLY = ("K1 is forward only: the fused walk has no backward, so a graph through "
                "it would keep the read's gradient and silently drop the state's. "
                "Use the eager path, or call under torch.no_grad().")


def check_no_grad(*tensors: torch.Tensor) -> None:
    """Refuse to build a graph through a stage that has no backward."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError(FORWARD_ONLY)


def check_k1_shapes(state: torch.Tensor, schedule: MatrixSchedule) -> None:
    value_dim = schedule.payload.shape[-1]
    if state.shape[-1] != value_dim:
        raise ValueError(f"state value dim {state.shape[-1]} != payload {value_dim}")
    if value_dim > MAX_TILE_VALUE_DIM:
        raise ValueError(f"K1 compiles for value dims up to {MAX_TILE_VALUE_DIM}, "
                         f"got {value_dim}")
    if not (state.is_cuda and schedule.slot.is_cuda):
        raise ValueError("K1 is a device kernel; state and schedule must be on CUDA")
    if schedule.payload.stride(-1) != 1:
        raise ValueError("the payload's value axis must be contiguous")


@triton.jit
def _mul_rn(a, b):
    """``a*b`` with one round-to-nearest step and no FMA contraction."""
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    """``a+b`` with one round-to-nearest step and no FMA contraction."""
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub_rn(a, b):
    """``a-b`` with one round-to-nearest step (the injection's erase subtraction)."""
    return tl.inline_asm_elementwise('sub.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.autotune(
    configs=[triton.Config({'BS': bs}, num_warps=warps)
             for bs in (8, 16, 32)
             for warps in (4, 8)],
    key=['R'],
)
@triton.jit(do_not_specialize=['T', 'R'])
def _k1_state_scan_kernel(
    state_in, live_in, decay,
    slot, op, erase, live_after, payload,
    state_out, live_out, strength_out, delta, log_gate,
    T, R,
    sb, st, sh, sl,
    ob, ot, oh, ol,
    eb, et, eh, el,
    lb, lt, lh, ll,
    pb, pt, ph, pl,
    db, dt, dh,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    BS: tl.constexpr,
):
    """One program per ``(row, head, slot_block)``; the whole chunk in registers.

    ``row = program_id(0)`` indexes ``(batch, head)``; ``program_id(1)`` selects a
    block of ``BS`` slots.  ``T`` is the token count of the chunk and ``L`` the lane
    count, so the chunk's step axis is ``steps = T*L`` (a token with two lanes is
    two steps, and the DECAY gate sits on lane 0 as in the reference).

    The ``s*``/``o*``/``e*``/``l*``/``p*``/``d*`` arguments are the ``(batch, token,
    head, lane[, value])`` strides of slot / op / erase / live_after / payload / decay.
    Taking them as arguments is what lets the decision channel be the router's own
    permuted view: no host-side permute or copy of the schedule is needed.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    lanes = tl.arange(0, BS)
    offs = block * BS + lanes
    inside = offs < R
    # The value axis is walked as a power-of-two tile and masked down to V, so a
    # value dim that is not a power of two costs a predicate and nothing else.
    values = tl.arange(0, VP)
    value_mask = values < V
    steps = T * L

    state_base = row * R * V
    acc = tl.load(state_in + state_base + offs[:, None] * V + values[None, :],
                  mask=inside[:, None] & value_mask[None, :], other=0.0)
    live = tl.load(live_in + row * R + offs, mask=inside, other=0) != 0
    gc = tl.zeros((), dtype=tl.float32)

    for t in range(0, T):
        lam = tl.load(decay + batch_index * db + t * dt + head_index * dh)
        # The DECAY gate is a diagonal channel: every slot of the block takes it.
        acc = _mul_rn(acc, lam)
        if block == 0:
            gc += tl.log(lam)
        for lane in tl.static_range(L):
            off = batch_index * sb + t * st + head_index * sh + lane * sl
            active = tl.load(op + off) != NONE_CODE
            delta_row = delta + (row * steps + t * L + lane) * V + values
            if block == 0:
                # The step's cumulative log decay: lane 0 carries the token's log
                # decay, the other lanes are neutral steps (log 1 = 0), as in the
                # reference.
                tl.store(log_gate + row * steps + t * L + lane, gc)
            if active:
                target = tl.load(slot + off).to(tl.int32)
                erase_here = tl.load(erase + off) != 0
                live_here = tl.load(live_after + off) != 0
                payload_base = (batch_index * pb + t * pt + head_index * ph
                                + lane * pl)
                written = tl.load(payload + payload_base + values, mask=value_mask,
                                  other=0.0)
                tile = tl.broadcast_to(written[None, :], (BS, VP))
                hit = inside & (target == offs)
                # Erase (CORRECT / POP / MERGE): the row becomes the payload.  The
                # injected payload is p - v, the relaxation form the read needs.
                if erase_here:
                    delta_tile = tl.where(hit[:, None], _sub_rn(tile, acc), 0.0)
                    acc = tl.where(hit[:, None], tile, acc)
                else:  # APPEND: the row keeps its content and gains the payload.
                    delta_tile = tl.where(hit[:, None], tile, 0.0)
                    acc = tl.where(hit[:, None], _add_rn(acc, tile), acc)
                live = tl.where(hit, live_here, live)
                # Exactly one block owns the addressed slot and writes the step's
                # injection.  Every block computes the same row offset, so an
                # unowned block that stored too would race the owner with zeros.
                if (target >= block * BS) & (target < block * BS + BS):
                    tl.store(delta_row, tl.sum(delta_tile, 0), mask=value_mask)
            else:
                # An inactive lane is the identity whatever payload it carries, and
                # its injection is exactly zero (chunk_scan masks it explicitly).
                if block == 0:
                    tl.store(delta_row, tl.zeros((VP,), dtype=tl.float32),
                             mask=value_mask)

    tl.store(state_out + state_base + offs[:, None] * V + values[None, :], acc,
             mask=inside[:, None] & value_mask[None, :])
    tl.store(live_out + row * R + offs, live.to(tl.int8), mask=inside)
    strength = tl.sqrt(tl.sum(tl.where(value_mask[None, :], _mul_rn(acc, acc), 0.0), 1))
    tl.store(strength_out + row * R + offs, tl.where(live, strength, 0.0), mask=inside)


@dataclass(frozen=True)
class K1Chunk:
    """Everything the fused state scan emits for one chunk.

    ``state``/``live``/``strength`` are the exit channels ``(batch, heads, rank)``
    (``strength`` with the trailing value axis reduced away); ``delta``
    ``(batch, heads, steps, V)`` is the injection list and ``log_gate``
    ``(batch, heads, steps)`` its cumulative log decay, so the decay weight of an
    injection is ``exp(log_gate_ref - log_gate_u)``.
    """

    state: torch.Tensor
    live: torch.Tensor
    strength: torch.Tensor
    delta: torch.Tensor
    log_gate: torch.Tensor
    steps: int


def k1_state_scan(state: torch.Tensor, live: torch.Tensor, decay: torch.Tensor,
                  schedule: MatrixSchedule) -> K1Chunk:
    """One chunk of the fused state scan: exactly one kernel launch.

    ``state`` ``(batch, heads, rank, V)``, ``live`` ``(batch, heads, rank)``,
    ``decay`` ``(batch, tokens, heads)``, ``schedule`` the chunk's fixed decision
    channel.  The schedule's strides are passed through, so no host-side permute or
    copy of the decision data is needed.
    """
    check_k1_shapes(state, schedule)
    check_no_grad(state, decay, schedule.payload)
    batch, heads, rank, value_dim = state.shape
    tokens, lane_count = schedule.tokens, schedule.lanes
    if decay.shape != (batch, tokens, heads):
        raise ValueError(f"decay must be {(batch, tokens, heads)}, got {tuple(decay.shape)}")
    if not state.is_contiguous() or not live.is_contiguous():
        raise ValueError("state and live must be contiguous")
    if live.dtype != torch.bool:
        raise ValueError("live must be a bool tensor")
    steps = tokens * lane_count
    device = state.device
    out_state = torch.empty((batch, heads, rank, value_dim), dtype=state.dtype,
                            device=device)
    out_live = torch.empty((batch, heads, rank), dtype=torch.bool, device=device)
    strength = torch.empty((batch, heads, rank), dtype=state.dtype, device=device)
    delta = torch.empty((batch, heads, steps, value_dim), dtype=state.dtype,
                        device=device)
    log_gate = torch.empty((batch, heads, steps), dtype=torch.float32, device=device)

    pointer = schedule
    _k1_state_scan_kernel[(lambda meta: (batch * heads, triton.cdiv(rank, meta['BS'])))](
        state, live.view(torch.int8), decay,
        pointer.slot, pointer.op, pointer.erase.view(torch.int8),
        pointer.live_after.view(torch.int8), pointer.payload,
        out_state, out_live.view(torch.int8), strength, delta, log_gate,
        tokens, rank,
        *pointer.slot.stride(), *pointer.op.stride(), *pointer.erase.stride(),
        *pointer.live_after.stride(), *pointer.payload.stride()[:4], *decay.stride(),
        H=heads, L=lane_count, V=value_dim,
        VP=triton.next_power_of_2(value_dim))
    return K1Chunk(out_state, out_live, strength, delta, log_gate, steps)


def _reads_from_injections(weights: torch.Tensor, entry_state: torch.Tensor,
                           chunk: K1Chunk, schedule: MatrixSchedule) -> torch.Tensor:
    """The read reduction over the emitted injection list (torch, reference form).

    Identical in form to ``scan.chunk_scan``'s read: the chunk-entry read plus the
    in-chunk injections weighted by ``exp(gc_read - gc_u)``.  One fused kernel for
    this reduction is stage K2; here it is the reference expression so that K1 is a
    drop-in for ``chunk_scan``.
    """
    batch, heads, tokens, _ = weights.shape
    steps, lane_count = chunk.steps, schedule.lanes
    device = weights.device
    order = torch.arange(steps, device=device)
    read_index = torch.arange(tokens, device=device) * lane_count - 1
    inside = read_index >= 0
    device_zero = torch.zeros((), dtype=torch.float32, device=device)
    gc_read = torch.where(inside, chunk.log_gate[..., read_index.clamp(min=0)], device_zero)
    slot = schedule.slot.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    addressed = weights.gather(3, slot.unsqueeze(2).expand(batch, heads, tokens, steps))
    before = (torch.arange(tokens, device=device).unsqueeze(-1) * lane_count
              > order.unsqueeze(0))
    coefficients = torch.where(before,
                              torch.exp(gc_read.unsqueeze(-1) - chunk.log_gate.unsqueeze(-2)),
                              device_zero)
    injected = (coefficients * addressed) @ chunk.delta
    entry_read = (torch.exp(gc_read).unsqueeze(-1) * weights) @ entry_state
    return entry_read + injected


def k1_chunk(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
             decay: torch.Tensor, schedule: MatrixSchedule):
    """Drop-in for ``scan.chunk_scan``: ``(reads, state, live)`` from the fused path.

    ``chunk_scan`` takes the DECAY gate head-major ``(batch, heads, tokens)``; the
    kernel indexes the schedule's ``(batch, tokens, heads)`` layout, and that
    permutation is a view -- no copy, no launch.
    """
    check_no_grad(weights)
    chunk = k1_state_scan(state, live, decay.permute(0, 2, 1), schedule)
    reads = _reads_from_injections(weights, state, chunk, schedule)
    return reads, chunk.state, chunk.live


__all__ = ['FORWARD_ONLY', 'K1Chunk', 'MAX_TILE_VALUE_DIM', 'check_k1_shapes',
           'check_no_grad', 'k1_state_scan', 'k1_chunk']
