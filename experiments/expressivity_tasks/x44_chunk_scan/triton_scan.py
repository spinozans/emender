"""One Triton kernel for the intra-chunk affine scan.

``affine.scan_with_entry`` runs a Python loop over the chunk steps, which is
one elementwise launch per step per pass: at chunk length 64 that is 1536 of the
2624 CUDA-graph nodes of a chunk, and a CUDA-graph replay costs about 2.4 us per
node.  The kernel below performs the same recurrence with one thread per lane
and the chunk loop in registers, so a scan is one node.

The arithmetic is the same operation sequence as the PyTorch pass
(``s = m * s; s = s + b``, no reassociation), which keeps the strength channel
-- and therefore the derived death boundaries and routing -- identical.  Use
``same_as_torch`` to verify that on the shapes in play before relying on it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(a, b):
    """``a*b`` with one round-to-nearest step and no FMA contraction."""
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _affine_scan_kernel(multiplier, injection, entry, states, lanes,
                        steps: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < lanes
    state = tl.load(entry + offset, mask=mask, other=0.)
    for step in range(steps):
        block = step * lanes + offset
        mult = tl.load(multiplier + block, mask=mask, other=1.)
        inject = tl.load(injection + block, mask=mask, other=0.)
        # Explicit round-per-operation: Triton would otherwise contract this
        # into an FMA and move the strength trajectory (and death boundaries)
        # away from the certified sequential path.
        state = _add_rn(_mul_rn(state, mult), inject)
        tl.store(states + block, state, mask=mask)


def scan_with_entry(multiplier: torch.Tensor, injection: torch.Tensor,
                    entry: torch.Tensor, block: int = 256):
    """``[C, L]`` multiplier/injection plus ``[L]`` entry -> ``([C, L], [L])``."""
    steps, lanes = multiplier.shape
    if injection.shape != multiplier.shape or entry.shape != (lanes,):
        raise ValueError('affine scan shape mismatch')
    multiplier = multiplier.contiguous()
    injection = injection.contiguous()
    entry = entry.contiguous()
    states = torch.empty_like(multiplier)
    _affine_scan_kernel[(triton.cdiv(lanes, block),)](
        multiplier, injection, entry, states, lanes,
        steps=steps, BLOCK=block, num_warps=4)
    return states, states[-1]


def same_as_torch(multiplier: torch.Tensor, injection: torch.Tensor,
                  entry: torch.Tensor) -> tuple[bool, float]:
    """Compare against the reference implementation; returns (exact, max deviation)."""
    from . import affine

    reference, reference_final = affine.scan_with_entry(multiplier, injection, entry)
    states, final = scan_with_entry(multiplier, injection, entry)
    deviation = max((states - reference).abs().max().item(),
                    (final - reference_final).abs().max().item())
    return bool(torch.equal(states, reference)), deviation


@triton.jit
def _strength_liveness_kernel(alloc, pop_slot, pop_factor, entry_strength, entry_live,
                              observed, live_read, live_after_out, final_strength,
                              final_live, rows, slots, steps: tl.constexpr,
                              decay: tl.constexpr, threshold: tl.constexpr,
                              derive: tl.constexpr, BLOCK: tl.constexpr):
    """One walk per (row, slot) lane over the chunk: the whole strength channel.

    Reproduces the certified sequential operation order exactly (pop multiply,
    POP death check, reviving append, decay, decay death check) with
    round-per-operation arithmetic, and emits the pre-pop strength, the
    read-time and post-write liveness, and the exit strength/liveness.  This
    replaces the two affine-scan passes plus the derived death masks, so no
    ``[rows, steps, slots]`` multiplier, injection, stamp or crossover tensor
    is ever materialised.
    """
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = offs < slots
    lane = row * slots + offs
    state = tl.load(entry_strength + lane, mask=inside, other=0.)
    live = tl.load(entry_live + lane, mask=inside, other=0) != 0
    schedule_at = row * steps
    for step in range(steps):
        slot_a = tl.load(alloc + schedule_at + step)
        slot_p = tl.load(pop_slot + schedule_at + step)
        factor = tl.load(pop_factor + schedule_at + step)
        is_append = slot_a == offs
        out = (row * steps + step) * slots + offs
        tl.store(observed + out, state, mask=inside)
        tl.store(live_read + out, live.to(tl.int8), mask=inside)
        # POP: strength decrement on the popped lane, then the free threshold.
        # An append erases the lane in this same step, so its pop value is dead
        # arithmetic -- but the certified path still records the POP death
        # before the reviving append, so the check runs first.
        state = tl.where(is_append, state,
                         _mul_rn(state, tl.where(slot_p == offs, factor, 1.)))
        if derive:
            live = live & (state > threshold)
        # APPEND revives and injects 1 (then takes the birth-token decay).
        state = tl.where(is_append, 1., state)
        live = live | is_append
        tl.store(live_after_out + out, live.to(tl.int8), mask=inside)
        state = _mul_rn(state, tl.where(live, decay, 1.))
        if derive:
            live = live & (state > threshold)
    tl.store(final_strength + lane, state, mask=inside)
    tl.store(final_live + lane, live.to(tl.int8), mask=inside)


def strength_liveness(sched, entry_strength, entry_live, *, decay: float, threshold: float,
                      derive: bool = True, block: int = 256):
    """``(observed, live_read, live_after, strength_final, live_final)``.

    The first three are ``[rows, steps, slots]`` (fp32, bool, bool); the last two
    are ``[rows, slots]``.
    """
    rows, steps = sched.alloc.shape
    slots = entry_strength.shape[1]
    device = entry_strength.device
    shape = (rows, steps, slots)
    observed = torch.empty(shape, dtype=torch.float32, device=device)
    live_read = torch.empty(shape, dtype=torch.bool, device=device)
    live_after = torch.empty(shape, dtype=torch.bool, device=device)
    final_strength = torch.empty(rows, slots, dtype=torch.float32, device=device)
    final_live = torch.empty(rows, slots, dtype=torch.bool, device=device)
    _strength_liveness_kernel[(rows, triton.cdiv(slots, block))](
        sched.alloc.contiguous(), sched.pop_slot.contiguous(),
        sched.pop_factor.contiguous(), entry_strength.contiguous(),
        entry_live.to(torch.int8).contiguous(), observed, live_read, live_after,
        final_strength, final_live, rows, slots,
        steps=steps, decay=decay, threshold=threshold, derive=derive,
        BLOCK=block, num_warps=4)
    return observed, live_read, live_after, final_strength, final_live


def strength_liveness_matches_torch(sched, entry_strength, entry_live, *, decay, threshold,
                                    derive=True):
    """Bit-exactness check against the multiplicative-map formulation."""
    from . import chunked_cell

    states, live_read, live_after, final, live_final = strength_liveness(
        sched, entry_strength, entry_live, decay=decay, threshold=threshold, derive=derive)
    reference = chunked_cell._strength_reference(
        sched, entry_strength, entry_live, decay=decay, threshold=threshold, derive=derive)
    matched = all(torch.equal(a, b) if a.dtype != torch.float32
                  else torch.equal(a, b) for a, b in zip(
                      (states, live_read, live_after, final, live_final), reference))
    deviation = max((a.float() - b.float()).abs().max().item()
                    for a, b in zip((states, live_read, live_after, final, live_final),
                                    reference))
    return matched, deviation


__all__ = ['scan_with_entry', 'same_as_torch', 'strength_liveness',
           'strength_liveness_matches_torch']
