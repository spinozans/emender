"""Journal-aware scan mutation helpers; diagnostics retain their original API."""
import triton
import triton.language as tl
from .journal import journal_store

@triton.jit
def _before(a, b, Due, J, Record: tl.constexpr):
    ta = tl.load(Due + a)
    tb = tl.load(Due + b)
    return (ta < tb) | (ta == tb) & (a < b)

@triton.jit
def _swap(Heap, Position, a, b, J, Record: tl.constexpr):
    sa = tl.load(Heap + a)
    sb = tl.load(Heap + b)
    journal_store(Heap + a, sb, True, J, Record)
    journal_store(Heap + b, sa, True, J, Record)
    journal_store(Position + sa, b, True, J, Record)
    journal_store(Position + sb, a, True, J, Record)
    tl.debug_barrier()

@triton.jit
def _up(Heap, Position, Due, index, J, Record: tl.constexpr):
    moving = index > 0
    while moving:
        parent = (index - 1) // 2
        child_slot = tl.load(Heap + index)
        parent_slot = tl.load(Heap + parent)
        if _before(child_slot, parent_slot, Due, J, Record):
            _swap(Heap, Position, index, parent, J, Record)
            index = parent
            moving = index > 0
        else:
            moving = False

@triton.jit
def _down(Heap, Position, Due, index, size, J, Record: tl.constexpr):
    moving = 2 * index + 1 < size
    while moving:
        child = 2 * index + 1
        if child + 1 < size:
            left = tl.load(Heap + child)
            right = tl.load(Heap + child + 1)
            child = tl.where(_before(right, left, Due, J, Record), child + 1, child)
        child_slot = tl.load(Heap + child)
        parent_slot = tl.load(Heap + index)
        if _before(child_slot, parent_slot, Due, J, Record):
            _swap(Heap, Position, index, child, J, Record)
            index = child
            moving = 2 * index + 1 < size
        else:
            moving = False

@triton.jit
def cancel(Heap, Position, Due, size, slot, J, Record: tl.constexpr):
    index = tl.load(Position + slot)
    if index >= 0:
        tl.device_assert(index < size, 'invalid heap inverse position')
        size -= 1
        if index < size:
            replacement = tl.load(Heap + size)
            journal_store(Heap + index, replacement, True, J, Record)
            journal_store(Position + replacement, index, True, J, Record)
            tl.debug_barrier()
            _up(Heap, Position, Due, index, J, Record)
            index = tl.load(Position + replacement)
            _down(Heap, Position, Due, index, size, J, Record)
        journal_store(Position + slot, -1, True, J, Record)
        tl.debug_barrier()
    return size

@triton.jit
def schedule(Heap, Position, Due, size, slot, time, J, Record: tl.constexpr):
    size = cancel(Heap, Position, Due, size, slot, J, Record)
    journal_store(Due + slot, time, True, J, Record)
    tl.debug_barrier()
    if time >= 0:
        journal_store(Heap + size, slot, True, J, Record)
        journal_store(Position + slot, size, True, J, Record)
        tl.debug_barrier()
        _up(Heap, Position, Due, size, J, Record)
        size += 1
    return size

@triton.jit
def next_due(Heap, Due, size, boundary, J, Record: tl.constexpr):
    slot = tl.full((), -1, tl.int32)
    if size > 0:
        first = tl.load(Heap)
        tl.device_assert(first >= 0, 'invalid heap root slot')
        if tl.load(Due + first) <= boundary:
            slot = first
    return slot

@triton.jit
def _first(bits, J, Record: tl.constexpr):
    return tl.inline_asm_elementwise('{ .reg .b64 reversed; brev.b64 reversed, $1; clz.b64 $0, reversed; }', constraints='=r,l', args=[bits.to(tl.uint64)], dtype=tl.int32, is_pure=True, pack=1)

@triton.jit
def set_free(L0, L1, Root, slot, available, J, Record: tl.constexpr):
    word = slot // 64
    bit = tl.full((), 1, tl.uint64) << (slot % 64).to(tl.uint64)
    bits = tl.load(L0 + word)
    bits = tl.where(available, bits | bit, bits & ~bit)
    journal_store(L0 + word, bits, True, J, Record)
    parent = word // 64
    bit = tl.full((), 1, tl.uint64) << (word % 64).to(tl.uint64)
    summary = tl.load(L1 + parent)
    summary = tl.where(bits != 0, summary | bit, summary & ~bit)
    journal_store(L1 + parent, summary, True, J, Record)
    bit = tl.full((), 1, tl.uint64) << parent.to(tl.uint64)
    root = tl.load(Root)
    journal_store(Root, tl.where(summary != 0, root | bit, root & ~bit), True, J, Record)
    tl.debug_barrier()

@triton.jit
def lowest_free(L0, L1, Root, J, Record: tl.constexpr):
    root = tl.load(Root)
    slot = tl.full((), -1, tl.int32)
    if root != 0:
        parent = _first(root, J, Record)
        word = 64 * parent + _first(tl.load(L1 + parent), J, Record)
        slot = 64 * word + _first(tl.load(L0 + word), J, Record)
    return slot

@triton.jit
def remove(Next, Prev, Head, slot, J, Record: tl.constexpr):
    before = tl.load(Prev + slot)
    after = tl.load(Next + slot)
    if before >= 0:
        journal_store(Next + before, after, True, J, Record)
    else:
        journal_store(Head, after, True, J, Record)
    if after >= 0:
        journal_store(Prev + after, before, True, J, Record)
    journal_store(Next + slot, -1, True, J, Record)
    journal_store(Prev + slot, -1, True, J, Record)
    tl.debug_barrier()

@triton.jit
def insert(Next, Prev, Head, Birth, slot, born, J, Record: tl.constexpr):
    cursor = tl.load(Head)
    before = tl.full((), -1, tl.int32)
    searching = cursor >= 0
    while searching:
        other = tl.load(Birth + cursor)
        ahead = (other > born) | (other == born) & (cursor < slot)
        if ahead:
            before = cursor
            cursor = tl.load(Next + cursor)
            searching = cursor >= 0
        else:
            searching = False
    journal_store(Prev + slot, before, True, J, Record)
    journal_store(Next + slot, cursor, True, J, Record)
    if before >= 0:
        journal_store(Next + before, slot, True, J, Record)
    else:
        journal_store(Head, slot, True, J, Record)
    if cursor >= 0:
        journal_store(Prev + cursor, slot, True, J, Record)
    tl.debug_barrier()

@triton.jit
def select(Next, Head, Birth, wanted, J, Record: tl.constexpr):
    youngest = tl.load(Head)
    cursor = youngest
    searching = cursor >= 0
    while searching:
        born = tl.load(Birth + cursor)
        if born > wanted:
            cursor = tl.load(Next + cursor)
            searching = cursor >= 0
        else:
            searching = False
    adjacent = tl.full((), False, tl.int1)
    if cursor >= 0:
        adjacent = tl.load(Birth + cursor) == wanted
    return (tl.where(adjacent, cursor, youngest), adjacent)
