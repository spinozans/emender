"""K5 -- the routing decision chain of ``route_chunk`` as one fused kernel.

Why this file exists
--------------------
``PROFILE_STEP_VS_GDN2.md`` decomposed the registered x45 step (9.29 s wall,
434,357 kernels/step, 89 % host-bound against a 3.32 s device floor) and ranked
the bottleneck: the x45 step is *launch-bound*, and the routing region owns
~45 % of the host gap.  Routing is not a device problem (0.7 ms/scan of device
time) and not a Python problem in the registered arm (it is compiled away); it
is routing's **launch count** -- the compiled decision chain still issues ~23
kernels per chunk (6,144 chunks/step), and the eager arm pays 243 per chunk
(5.98 M launches at mb2, ``route_chunk`` cum 62.6 s of a 130.5 s traced step).

What is fused, and what deliberately is not
-------------------------------------------
The decision chain downstream of ``similarity`` is a function of the chunk-entry
state, the recorded thresholds and *nothing else* -- exactly the property that
let K1 fuse the scan (``k1_schedule_scan.py``).  ONE kernel per chunk, one
program per ``(batch, head)`` row, walks the chunk's tokens with the whole rank
axis in registers and emits the complete discrete decision channel of both
lanes: the ``-2.0`` dead-row fill, the tie-break penalty (``arange * tie_break``,
fp32), best/second/worst selections (torch's first-index rule on exact ties),
the free-slot ordinal scan (the one sequential dependency along the token axis:
the n-th appendable token takes the n-th free slot), and the five-op code
selection -- ``slot``, ``op``, ``erase``, ``live_after``, plus the
``best_index``/``second_index`` channels the payload blend gathers with.

**The inputs to the kernel are left as the eager router's own torch ops**
(``read_weights``, the row-strength normalisation, and the ``value @ stored``
similarity bmm), for one measured reason: a decision is an argmax against
thresholds, so a 1-ulp difference in ``similarity`` can flip it.  The bmm's
reduction order belongs to cuBLAS and cannot be reproduced from Triton, so the
only construction with *exact* decision parity is to consume the very tensor
the eager router produced.  Every value the kernel derives from it is either an
IEEE fp32 subtraction of the penalty (same rounding as eager's promoted
``similarity - penalty``), an exact comparison, or integer selection -- so the
decisions are **bit-identical to eager ``route_chunk``** on any fixture, with no
noise-floor allowance.  The read weights and the payload lanes stay the
verbatim torch expressions of ``schedule.route_chunk`` for the same reason and
one more: they carry gradient (the read weights are the read; the MERGE blend
reaches both source rows), and keeping them in torch keeps the registered
backward bit-identical too.

Thresholds are compared in the similarity's own dtype, exactly as eager's
tensor-vs-scalar promotion does it: under the model's bf16 autocast the
comparison ``best_value >= tau_match`` happens in bf16, so the wrapper rounds
each threshold to the similarity dtype before the launch (a bf16 -> fp32 cast
is exact, so the fp32 comparison of the cast values is the same predicate).

Compile hosting, and the one detail it turned on
-----------------------------------------------
Dynamo captures the user-defined Triton kernel as a single opaque
``triton_kernel_wrapper`` node, so the decision channel stays one launch inside
the compiled region -- but only because the kernel's liveness outputs are
allocated as *int8* and cast to bool on return.  A ``bool`` tensor passed through
``view(torch.int8)`` is an ``aten.view.dtype`` bitcast, and inductor refuses to
lower it (``torch.iinfo`` on ``torch.bool``) -- the very ``InductorError`` that
blocks the K4 node under ``compile_scan=True``
(``PROFILE_STEP_VS_GDN2.md`` §5, and the fix pattern Fix 2 reuses).  A
``torch.library`` custom op hosts the same kernel too, but its Python dispatch
cost ~0.16 ms per chunk under compile -- more host time than the ~2 inductor
kernels it replaces -- so the raw captured launch is the hosted form.

The fused router is a drop-in for the ``router=`` argument of
``scan.routed_chunk_scan``: same signature, same outputs, same layouts, and the
model wiring takes it behind ``MatrixMixer(fused_routing=True)``.  The
append-only control (``policy.ops=False``) is the eager branch verbatim -- its
decisions are a modulo, not a search.
"""
from __future__ import annotations

import struct
import torch
import triton
import triton.language as tl

from .algebra import (APPEND, CORRECT, MAX_LANES, MERGE, NONE, POP, MatrixSchedule)
from .read import read_weights
from .schedule import EPS, _blank, _schedule

# The first-index sentinel of the min-index reductions (torch's argmax/argmin
# return the first occurrence; the kernel reproduces that rule explicitly).
_BIG_INDEX = tl.constexpr(2**30)

# Triton constants for the op codes (module-level ints instantiated as constexpr).
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


def _round_threshold(value: float, dtype: torch.dtype) -> float:
    """Round a policy threshold to the similarity's dtype, as eager's comparison does.

    Eager compares ``best_value >= policy.tau_match`` with the threshold as a
    wrapped scalar, so the predicate is evaluated in the *tensor's* dtype: the
    scalar is cast to bf16 when the similarity is bf16 (the model's autocast
    regime) and to fp32 when it is fp32.  The kernel compares in fp32 after an
    exact bf16 -> fp32 cast, so it must compare against the dtype-rounded
    threshold -- then the two predicates agree bit for bit.  Pure bit math (no
    ``torch.tensor``), so dynamo constant-folds it at trace time.
    """
    if dtype == torch.float32:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if dtype == torch.bfloat16:
        # Round-to-nearest-even into the top 16 bits of the fp32 representation.
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
        return struct.unpack("<f", struct.pack("<I", rounded))[0]
    if dtype == torch.float16:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    raise ValueError(f"unsupported similarity dtype {dtype}")


class _ThresholdPolicy:
    """The four routing constants the kernel needs, as a policy-shaped adapter."""

    def __init__(self, tau_match, tau_merge, tau_pop, tie_break):
        self.tau_match = tau_match
        self.tau_merge = tau_merge
        self.tau_pop = tau_pop
        self.tie_break = tie_break


@triton.jit(do_not_specialize=['T', 'R'])
def _route_decisions_kernel(
    similarity, active,
    slot_out, op_out, erase_out, live_out, best_out, second_out,
    T, R,
    s_sim_b, s_sim_h, s_sim_t, s_sim_r,
    s_act_b, s_act_h, s_act_r,
    tau_match, tau_merge, tau_pop, tie_break,
    H: tl.constexpr, RP: tl.constexpr,
):
    """One program per ``(batch, head)`` row; the whole chunk's decisions in registers.

    ``T`` is the chunk's token count and ``R`` its active rank.  The rank axis
    lives in registers as ``RP`` lanes (masked down to ``R``); the token axis is
    walked in order because the free-slot assignment is a running ordinal along
    it -- the one sequential dependency in routing.  Outputs are the head-major
    ``(batch, heads, tokens, lanes)`` decision channel (the layout
    ``schedule._blank`` builds) plus the best/second index channels
    ``(batch, heads, tokens)`` the payload blend gathers with.
    """
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep = ranks < R

    act = tl.load(active + batch_index * s_act_b + head_index * s_act_h
                  + ranks * s_act_r, mask=keep, other=0) != 0
    act = act & keep                    # a padding lane is not a live row
    free = keep & (~act)
    # position[r] = (#free in [0..r]) - 1: for a free slot, its 0-based ordinal
    # in the free list -- exactly ``schedule.route_chunk``'s ``position``.
    position = tl.cumsum(free.to(tl.int32), 0) - 1
    free_count = tl.sum(free.to(tl.int32), 0)
    any_active = tl.sum(act.to(tl.int32), 0) > 0
    penalty = ranks.to(tl.float32) * tie_break

    sim_base = similarity + batch_index * s_sim_b + head_index * s_sim_h
    out_base = row.to(tl.int64) * T * 2      # the lanes axis is MAX_LANES == 2
    need_count = tl.zeros((), dtype=tl.int32)

    for t in range(0, T):
        raw = tl.load(sim_base + t * s_sim_t + ranks * s_sim_r, mask=keep,
                      other=-2.0)
        sim = tl.where(act, raw, -2.0)       # the eager masked_fill(~active, -2.0)
        score = tl.where(keep, sim - penalty, -3.0)
        # Best: the first index of the max of score (torch.argmax's rule).
        best_value_score = tl.max(score, 0)
        best = tl.min(tl.where(score == best_value_score, ranks, _BIG_INDEX), 0)
        best_value = tl.sum(tl.where(ranks == best, sim, 0.0), 0)
        # Second: the eager scatter of -3.0 over the winner, then argmax again.
        second_score = tl.where(ranks == best, -3.0, score)
        second_top = tl.max(second_score, 0)
        second = tl.min(tl.where(second_score == second_top, ranks, _BIG_INDEX), 0)
        second_value = tl.sum(tl.where(ranks == second, sim, 0.0), 0)

        correct = best_value >= tau_match
        merge = correct & (second_value >= tau_merge)
        # The append ordinal: the running count of not-correct tokens so far,
        # which is exactly ``need.cumsum(-1) - 1`` at a need token.
        need = ~correct
        ordinal = need_count
        need_count += need.to(tl.int32)
        appendable = need & (ordinal < free_count)
        # The n-th appendable token writes the n-th free slot: the first free
        # rank whose position equals the ordinal (eager's equality argmax; a
        # token with no match takes index 0, as argmax of an all-zero vector).
        match = free & (position == ordinal)
        found = tl.min(tl.where(match, ranks, _BIG_INDEX), 0)
        free_slot = tl.where(found < _BIG_INDEX, found, 0)
        # Worst: the first index of the min of score, padding lanes excluded.
        worst_score = tl.where(keep, score, float("inf"))
        worst_top = tl.min(worst_score, 0)
        worst = tl.min(tl.where(worst_score == worst_top, ranks, _BIG_INDEX), 0)
        worst_value = tl.sum(tl.where(ranks == worst, sim, 0.0), 0)

        full = free_count <= 0
        pop = (~correct) & (~appendable) & full & (worst_value <= tau_pop) \
            & any_active

        op = tl.where(correct, tl.where(merge, _MERGE_CODE, _CORRECT_CODE),
                      tl.where(appendable, _APPEND_CODE,
                               tl.where(pop, _POP_CODE, _NONE_CODE)))
        slot0 = tl.where(correct, best, tl.where(appendable, free_slot, worst))
        erase0 = (op == _CORRECT_CODE) | (op == _MERGE_CODE) | (op == _POP_CODE)
        live0 = op != _POP_CODE

        tl.store(slot_out + out_base + t * 2 + 0, slot0.to(tl.int64))
        tl.store(slot_out + out_base + t * 2 + 1, second.to(tl.int64))
        tl.store(op_out + out_base + t * 2 + 0, op.to(tl.int64))
        tl.store(op_out + out_base + t * 2 + 1,
                 tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
        tl.store(erase_out + out_base + t * 2 + 0, erase0.to(tl.int8))
        tl.store(erase_out + out_base + t * 2 + 1, merge.to(tl.int8))
        tl.store(live_out + out_base + t * 2 + 0, live0.to(tl.int8))
        tl.store(live_out + out_base + t * 2 + 1, tl.zeros((), dtype=tl.int8))
        tl.store(best_out + row.to(tl.int64) * T + t, best.to(tl.int64))
        tl.store(second_out + row.to(tl.int64) * T + t, second.to(tl.int64))


def route_decisions(similarity: torch.Tensor, active: torch.Tensor,
                    policy) -> tuple:
    """The chunk's whole discrete decision channel, in ONE kernel launch.

    ``similarity`` is the eager router's own ``(batch, heads, tokens, rank)``
    tensor (``value @ stored^T`` -- its reduction order is part of the decision
    semantics, so it is an input here, never recomputed); ``active`` is the
    chunk-entry liveness prefix ``(batch, heads, rank)``.  Returns
    ``(slot, op, erase, live_after, best_index, second_index)`` head-major:
    the four schedule fields as ``(batch, heads, tokens, 2)`` (the layout
    ``schedule._blank`` builds) and the two index channels as
    ``(batch, heads, tokens)`` int64, the exact tensors the eager payload blend
    gathers with.
    """
    batch, heads, tokens, rank = similarity.shape
    if similarity.shape[-1] != active.shape[-1]:
        raise ValueError("the similarity and the active rank disagree")
    device = similarity.device
    lanes = MAX_LANES
    slot = torch.empty((batch, heads, tokens, lanes), dtype=torch.int64,
                       device=device)
    op = torch.empty((batch, heads, tokens, lanes), dtype=torch.int64,
                     device=device)
    # The liveness channels are launched as int8 and cast to bool on return: a
    # ``bool`` tensor passed through ``view(torch.int8)`` is what inductor cannot
    # lower (``torch.iinfo`` on ``torch.bool``) once dynamo captures the launch,
    # while an int8 -> bool *value* cast is an ordinary op it fuses.
    erase_i8 = torch.empty((batch, heads, tokens, lanes), dtype=torch.int8,
                           device=device)
    live_i8 = torch.empty((batch, heads, tokens, lanes), dtype=torch.int8,
                          device=device)
    best_index = torch.empty((batch, heads, tokens), dtype=torch.int64,
                             device=device)
    second_index = torch.empty((batch, heads, tokens), dtype=torch.int64,
                               device=device)
    _route_decisions_kernel[(batch * heads,)](
        similarity, active,
        slot, op, erase_i8, live_i8,
        best_index, second_index,
        tokens, rank,
        *similarity.stride(), *active.stride(),
        tau_match=_round_threshold(policy.tau_match, similarity.dtype),
        tau_merge=_round_threshold(policy.tau_merge, similarity.dtype),
        tau_pop=_round_threshold(policy.tau_pop, similarity.dtype),
        tie_break=struct.unpack("<f", struct.pack("<f", policy.tie_break))[0],
        H=heads, RP=max(triton.next_power_of_2(rank), 16))
    return (slot, op, erase_i8.to(torch.bool), live_i8.to(torch.bool),
            best_index, second_index)


def fused_route_chunk(index, token_offset, state, live, query, payload, policy,
                      rank):
    """A drop-in for ``schedule.route_chunk`` with the decision chain fused.

    Same signature, same outputs (``(schedule, weights)``), same layouts, and
    bit-identical values on every channel: the read weights, the similarity
    preprocessing and the payload lanes are the eager router's own torch
    expressions verbatim, and the discrete fields come from one kernel that
    consumes the eager similarity.  Decisions, payloads and weights are
    therefore equal to the eager router's bit for bit -- the parity the gate
    asserts, not a tolerance.
    """
    batch, heads, tokens, value_dim = payload.shape
    device = payload.device
    dtype = torch.float32
    rank = int(rank)
    active = live[:, :, :rank]
    rows = state[:, :, :rank]
    # The read-weight channel is always as wide as the state, zero above the
    # active rank: the schedule is a prefix of the family (schedule.py's rule).
    weights = torch.zeros(batch, heads, tokens, state.shape[2], dtype=dtype,
                          device=device)
    weights[:, :, :, :rank] = read_weights(query, rows, active, policy.read_scale)

    if not policy.ops:
        # The append-only control: a modulo, not a search -- eager verbatim.
        steps = token_offset + torch.arange(tokens, device=device)
        slot = (steps % rank).view(1, 1, tokens, 1).expand(batch, heads, tokens, 1)
        fields = _blank(batch, tokens, heads, 1, value_dim, device, dtype)
        fields["slot"][...] = slot
        fields["payload"][..., 0, :] = payload.to(dtype)
        fields["op"][...] = APPEND
        fields["live_after"][...] = True
        return _schedule(fields), weights

    frozen_rows = rows.detach()
    strength = frozen_rows.norm(dim=-1) * active.to(dtype)            # (b, h, rank)
    stored = frozen_rows / strength.clamp_min(EPS).unsqueeze(-1)
    value = payload / payload.norm(dim=-1, keepdim=True).clamp_min(EPS)
    similarity = value @ stored.transpose(-1, -2)                      # (b, h, T, rank)

    slot_hm, op_hm, erase_hm, live_hm, best_index, second_index = route_decisions(
        similarity, active, policy)

    # Payloads: the eager router's expression verbatim -- the incoming value for
    # APPEND/CORRECT, zero for POP and an inactive lane, and for MERGE the
    # norm-weighted blend of the two stored rows (weights detached, content
    # attached so gradients reach what was stored).
    payload_field = torch.zeros(batch, heads, tokens, MAX_LANES, value_dim,
                                dtype=dtype, device=device)
    b_index = torch.arange(batch, device=device)[:, None, None]
    h_index = torch.arange(heads, device=device)[None, :, None]
    first_row = rows[b_index, h_index, best_index]
    second_row = rows[b_index, h_index, second_index]
    first_weight = strength.gather(-1, best_index).clamp_min(EPS)
    second_weight = strength.gather(-1, second_index).clamp_min(EPS)
    total = first_weight + second_weight
    blend = ((first_weight / total).detach().unsqueeze(-1) * first_row
             + (second_weight / total).detach().unsqueeze(-1) * second_row)
    op_lane0 = op_hm[..., 0]
    carries_value = (op_lane0 == APPEND) | (op_lane0 == CORRECT)
    payload_lane0 = torch.where(carries_value.unsqueeze(-1), payload,
                                torch.zeros_like(payload))
    payload_lane0 = torch.where((op_lane0 == MERGE).unsqueeze(-1), blend,
                               payload_lane0)
    payload_field[..., 0, :] = payload_lane0

    schedule = MatrixSchedule(
        slot=slot_hm.permute(0, 2, 1, 3),
        erase=erase_hm.permute(0, 2, 1, 3),
        payload=payload_field.permute(0, 2, 1, 3, 4),
        op=op_hm.permute(0, 2, 1, 3),
        live_after=live_hm.permute(0, 2, 1, 3))
    return schedule, weights


__all__ = ['fused_route_chunk', 'route_decisions']
