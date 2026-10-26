"""Single-owner device min-heap for cancellable exact death events.

The heap stores slot IDs; inverse positions allow a pop touch to cancel its
old death in O(log S), with O(S) bounded storage (no accumulating stale events).
Times and slot IDs jointly order events so simultaneous deaths are stable.
The sequential scan uses these primitives for import, pop touches and births.
"""
import torch
import triton
import triton.language as tl

from .lazy_decay import death_exact


@triton.jit
def _before(a, b, Due):
    ta = tl.load(Due+a)
    tb = tl.load(Due+b)
    return (ta < tb) | ((ta == tb) & (a < b))


@triton.jit
def _swap(Heap, Position, a, b):
    sa = tl.load(Heap+a)
    sb = tl.load(Heap+b)
    tl.store(Heap+a, sb)
    tl.store(Heap+b, sa)
    tl.store(Position+sa, b)
    tl.store(Position+sb, a)
    tl.debug_barrier()


@triton.jit
def _up(Heap, Position, Due, index):
    moving = index > 0
    while moving:
        parent = (index-1)//2
        child_slot = tl.load(Heap+index)
        parent_slot = tl.load(Heap+parent)
        if _before(child_slot, parent_slot, Due):
            _swap(Heap, Position, index, parent)
            index = parent
            moving = index > 0
        else:
            moving = False


@triton.jit
def _down(Heap, Position, Due, index, size):
    moving = 2*index+1 < size
    while moving:
        child = 2*index+1
        if child+1 < size:
            left = tl.load(Heap+child)
            right = tl.load(Heap+child+1)
            child = tl.where(_before(right, left, Due), child+1, child)
        child_slot = tl.load(Heap+child)
        parent_slot = tl.load(Heap+index)
        if _before(child_slot, parent_slot, Due):
            _swap(Heap, Position, index, child)
            index = child
            moving = 2*index+1 < size
        else:
            moving = False


@triton.jit
def cancel(Heap, Position, Due, size, slot):
    index = tl.load(Position+slot)
    if index >= 0:
        tl.device_assert(index < size, "invalid heap inverse position")
        size -= 1
        if index < size:
            replacement = tl.load(Heap+size)
            tl.store(Heap+index, replacement)
            tl.store(Position+replacement, index)
            tl.debug_barrier()
            # Either direction may be needed after moving the last leaf.
            _up(Heap, Position, Due, index)
            index = tl.load(Position+replacement)
            _down(Heap, Position, Due, index, size)
        tl.store(Position+slot, -1)
        tl.debug_barrier()
    return size


@triton.jit
def schedule(Heap, Position, Due, size, slot, time):
    size = cancel(Heap, Position, Due, size, slot)
    tl.store(Due+slot, time)
    tl.debug_barrier()
    if time >= 0:
        tl.store(Heap+size, slot)
        tl.store(Position+slot, size)
        tl.debug_barrier()
        _up(Heap, Position, Due, size)
        size += 1
    return size


@triton.jit
def next_due(Heap, Due, size, boundary):
    slot = tl.full((), -1, tl.int32)
    if size > 0:
        first = tl.load(Heap)
        tl.device_assert(first >= 0, "invalid heap root slot")
        if tl.load(Due+first) <= boundary:
            slot = first
    return slot


@triton.jit
def _exercise(Heap, Position, Due, Sizes, Slots, Strengths, Times, Boundaries,
              Output, Counts, S: tl.constexpr, E: tl.constexpr,
              Decay: tl.constexpr, Threshold: tl.constexpr):
    """Test driver; one program per store, no host event routing.

    Each event schedules a touched reference and drains all due deaths.
    Production callers can invoke schedule/cancel/next_due inside their scan.
    """
    n = tl.program_id(0)
    heap = Heap+n*S
    position = Position+n*S
    due = Due+n*S
    size = tl.load(Sizes+n)
    for event in range(E):
        inp = n*E+event
        slot = tl.load(Slots+inp)
        if slot >= 0:
            strength = tl.load(Strengths+inp+tl.arange(0, 1))
            horizon, _ = death_exact(strength, tl.full((1,), Threshold, tl.float32), Decay)
            dt = tl.sum(horizon, 0)
            time = tl.load(Times+inp)
            size = schedule(heap, position, due, size, slot, tl.where(dt < 0, -1, time+dt))
        boundary = tl.load(Boundaries+inp)
        dead = next_due(heap, due, size, boundary)
        count = 0
        while dead >= 0:
            tl.store(Output+inp*S+count, dead)
            count += 1
            size = cancel(heap, position, due, size, dead)
            dead = next_due(heap, due, size, boundary)
        tl.store(Counts+inp, count)
    tl.store(Sizes+n, size)


def exercise_schedule(slots, strengths, times, boundaries, capacity, decay, threshold):
    """Diagnostic driver returning padded deaths, counts and final heap state.

    Slot IDs must be -1 or in [0, capacity); references must be live. This
    diagnostic allocates dense output only to make the event order testable;
    the heap primitives produce one event at a time without dense sweeps.
    """
    if (slots.ndim != 2 or not slots.is_cuda or capacity < 1 or
        slots.dtype != torch.int64 or strengths.dtype != torch.float32 or
        times.dtype != torch.int64 or boundaries.dtype != torch.int64 or
        any(x.shape != slots.shape or x.device != slots.device or not x.is_contiguous()
            for x in (slots, strengths, times, boundaries))):
        raise ValueError('matching contiguous CUDA event tensors required')
    import struct
    decay = struct.unpack('<f', struct.pack('<f', decay))[0]
    threshold = struct.unpack('<f', struct.pack('<f', threshold))[0]
    if not 0 < decay <= 1 or not 0 <= threshold < 1:
        raise ValueError('invalid decay/threshold')
    n, e = slots.shape
    heap = torch.empty((n, capacity), device=slots.device, dtype=torch.int32)
    position = torch.full_like(heap, -1)
    due = torch.full((n, capacity), -1, device=slots.device, dtype=torch.int64)
    sizes = torch.zeros(n, device=slots.device, dtype=torch.int32)
    output = torch.full((n, e, capacity), -1, device=slots.device, dtype=torch.int32)
    counts = torch.empty_like(slots)
    _exercise[(n,)](heap, position, due, sizes, slots, strengths, times, boundaries,
                    output, counts, capacity, e, decay, threshold, enable_fp_fusion=False)
    return output, counts, heap, position, due, sizes
