"""Device linked birth order, youngest first and lowest slot first on ties.

Import sorts once. Removal is constant work; ordinary appends prepend. Future
births in arbitrary imported states use an ordered walk, without dense slot
reductions. A slot belongs to the list exactly when it is live.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _initialize(Order, Live, Next, Prev, Head, S: tl.constexpr, BS: tl.constexpr):
    n = tl.program_id(0)
    r = tl.arange(0, BS)
    slot = tl.load(Order+n*S+r, r<S, 0)
    live = (r<S) & tl.load(Live+n*S+slot, r<S, 0).to(tl.int1)
    count = tl.sum(live.to(tl.int32), 0)
    after = tl.load(Order+n*S+r+1, r+1<S, -1)
    before = tl.load(Order+n*S+r-1, (r>0) & (r<S), -1)
    tl.store(Next+n*S+slot, tl.where(live & (r+1<count), after, -1), r<S)
    tl.store(Prev+n*S+slot, tl.where(live & (r>0), before, -1), r<S)
    tl.store(Head+n, tl.where(count>0, tl.load(Order+n*S), -1))


def initialize(live, age):
    shape = live.shape
    slots = shape[-1]
    rows = live.numel() // slots
    # Stable sorts retain ascending slot IDs for age ties; dead rows go last.
    ages = age.reshape(rows, slots)
    order = torch.argsort(ages, dim=-1, stable=True)
    ordered_live = live.reshape(rows, slots).gather(1, order)
    rank = torch.argsort(ordered_live.to(torch.int8), dim=-1,
                         descending=True, stable=True)
    order = order.gather(1, rank).contiguous()
    next_slot = torch.empty((rows, slots), device=live.device, dtype=torch.int32)
    prev_slot = torch.empty_like(next_slot)
    head = torch.empty(rows, device=live.device, dtype=torch.int32)
    _initialize[(rows,)](order, live, next_slot, prev_slot, head, slots,
                         triton.next_power_of_2(slots))
    return next_slot, prev_slot, head


@triton.jit
def remove(Next, Prev, Head, slot):
    before = tl.load(Prev+slot)
    after = tl.load(Next+slot)
    if before >= 0:
        tl.store(Next+before, after)
    else:
        tl.store(Head, after)
    if after >= 0:
        tl.store(Prev+after, before)
    tl.store(Next+slot, -1)
    tl.store(Prev+slot, -1)
    tl.debug_barrier()


@triton.jit
def insert(Next, Prev, Head, Birth, slot, born):
    cursor = tl.load(Head)
    before = tl.full((), -1, tl.int32)
    searching = cursor >= 0
    while searching:
        other = tl.load(Birth+cursor)
        ahead = (other > born) | ((other == born) & (cursor < slot))
        if ahead:
            before = cursor
            cursor = tl.load(Next+cursor)
            searching = cursor >= 0
        else:
            searching = False
    tl.store(Prev+slot, before)
    tl.store(Next+slot, cursor)
    if before >= 0:
        tl.store(Next+before, slot)
    else:
        tl.store(Head, slot)
    if cursor >= 0:
        tl.store(Prev+cursor, slot)
    tl.debug_barrier()


@triton.jit
def select(Next, Head, Birth, wanted):
    youngest = tl.load(Head)
    cursor = youngest
    searching = cursor >= 0
    while searching:
        born = tl.load(Birth+cursor)
        if born > wanted:
            cursor = tl.load(Next+cursor)
            searching = cursor >= 0
        else:
            searching = False
    adjacent = tl.full((), False, tl.int1)
    if cursor >= 0:
        adjacent = tl.load(Birth+cursor) == wanted
    return tl.where(adjacent, cursor, youngest), adjacent
