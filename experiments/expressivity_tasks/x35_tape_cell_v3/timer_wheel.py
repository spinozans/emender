"""Intrusive radix timer wheel primitives for a single-owner sequential core.

Eight radix-256 levels cover nonnegative signed int64 clocks. Each live timer
owns one link pair; rescheduling cancels it rather than accumulating stale
records. Insert/cancel have bounded work, and an event cascades at most seven
times before firing. Advancing costs O(elapsed tokens + cascaded/fired events).
Every mutation uses X31's typed before-image primitive. The caller owns the
clock and must drain the current level-zero bucket before advancing time.
These components are not yet wired into the X31 scan/replay state layout.
"""
import torch
import triton
import triton.language as tl
from ..x31_tape_cell_fused.all_field_journal import journal_store

LEVELS = 8
RADIX = 256
BUCKETS = LEVELS * RADIX


@triton.jit
def unlink(Heads, Next, Prev, Bucket, slot, J, Record: tl.constexpr):
    bucket = tl.load(Bucket+slot, volatile=True)
    if bucket >= 0:
        before = tl.load(Prev+slot, volatile=True)
        after = tl.load(Next+slot, volatile=True)
        if before >= 0:
            journal_store(Next+before, after, True, J, Record)
        else:
            journal_store(Heads+bucket, after, True, J, Record)
        if after >= 0:
            journal_store(Prev+after, before, True, J, Record)
        journal_store(Bucket+slot, -1, True, J, Record)
        journal_store(Next+slot, -1, True, J, Record)
        journal_store(Prev+slot, -1, True, J, Record)
        tl.debug_barrier()


@triton.jit
def insert(Heads, Next, Prev, Bucket, Due, slot, deadline, clock, J, Record: tl.constexpr):
    unlink(Heads, Next, Prev, Bucket, slot, J, Record)
    journal_store(Due+slot, deadline, True, J, Record)
    if deadline >= 0:
        # Past-due imports are assigned to the current bucket, not a past lap.
        effective = tl.maximum(deadline, clock)
        diff = effective ^ clock
        level = tl.full((), 0, tl.int32)
        for l in tl.static_range(1,8):
            level = tl.where(diff >= (1 << (8*l)), l, level)
        bucket = level*256+((effective >> (level*8)) & 255).to(tl.int32)
        head = tl.load(Heads+bucket, volatile=True)
        journal_store(Next+slot, head, True, J, Record)
        journal_store(Prev+slot, -1, True, J, Record)
        if head >= 0:
            journal_store(Prev+head, slot, True, J, Record)
        journal_store(Bucket+slot, bucket, True, J, Record)
        journal_store(Heads+bucket, slot, True, J, Record)
        tl.debug_barrier()


@triton.jit
def _cascade_level(Heads, Next, Prev, Bucket, Due, clock, J,
                   Record: tl.constexpr, Level: tl.constexpr):
    if (clock & ((1 << (8*Level))-1)) == 0:
        bucket = Level*256+((clock >> (Level*8)) & 255).to(tl.int32)
        slot = tl.load(Heads+bucket, volatile=True)
        while slot >= 0:
            deadline = tl.load(Due+slot)
            insert(Heads,Next,Prev,Bucket,Due,slot,deadline,clock,J,Record)
            slot = tl.load(Heads+bucket, volatile=True)


@triton.jit
def cascade(Heads, Next, Prev, Bucket, Due, clock, J, Record: tl.constexpr):
    for level in tl.static_range(7,0,-1):
        _cascade_level(Heads,Next,Prev,Bucket,Due,clock,J,Record,level)


@triton.jit
def first(Heads, clock):
    return tl.load(Heads+(clock & 255), volatile=True)


@triton.jit
def _exercise(Heads, Next, Prev, Bucket, Due, Slots, Deadlines, Boundaries,
              Output, Counts, S: tl.constexpr, E: tl.constexpr, Start):
    n = tl.program_id(0)
    heads = Heads+n*2048
    nxt, prev, bucket, due = Next+n*S, Prev+n*S, Bucket+n*S, Due+n*S
    clock = Start.to(tl.int64)
    J = (Heads,Heads,Heads,Heads,n,0)
    for event in range(E):
        i = n*E+event
        slot = tl.load(Slots+i)
        deadline = tl.load(Deadlines+i)
        if slot >= 0:
            insert(heads,nxt,prev,bucket,due,slot,deadline,clock,J,False)
        boundary = tl.load(Boundaries+i)
        count = 0
        while clock <= boundary:
            cascade(heads,nxt,prev,bucket,due,clock,J,False)
            dead = first(heads,clock)
            while dead >= 0:
                tl.device_assert(tl.load(due+dead) <= clock, 'premature wheel expiry')
                tl.store(Output+i*S+count,dead)
                count += 1
                unlink(heads,nxt,prev,bucket,dead,J,False)
                dead = first(heads,clock)
            clock += 1
        tl.store(Counts+i,count)


def exercise(slots, deadlines, boundaries, capacity, start=0):
    """GPU component driver; each event schedules then drains through boundary.

    Boundaries must be nondecreasing, at least start-1 and below INT64_MAX. Scheduling -1 as
    a deadline cancels, -1 as a slot performs no scheduling. Death tie order
    is deterministic intrusive-list order, not X31 heap's sorted slot order.
    """
    if (capacity<1 or not 0<=start<2**63-1 or slots.ndim!=2 or not slots.is_cuda or
        any(x.dtype!=torch.int64 or x.shape!=slots.shape or x.device!=slots.device or not x.is_contiguous()
            for x in (slots,deadlines,boundaries))):
        raise ValueError('contiguous matching int64 CUDA events required')
    if ((slots < -1) | (slots >= capacity)).any() or (deadlines < -1).any():
        raise ValueError('invalid slot or deadline')
    if (boundaries == 2**63-1).any():
        raise ValueError('terminal clock would overflow int64')
    if (boundaries < start-1).any() or (boundaries[:,1:] < boundaries[:,:-1]).any():
        raise ValueError('boundaries must advance monotonically')
    n,e = slots.shape
    heads = torch.full((n,BUCKETS),-1,device=slots.device,dtype=torch.int32)
    nxt,prev,bucket = [torch.full((n,capacity),-1,device=slots.device,dtype=torch.int32) for _ in range(3)]
    due = torch.full((n,capacity),-1,device=slots.device,dtype=torch.int64)
    output = torch.full((n,e,capacity),-1,device=slots.device,dtype=torch.int32)
    counts = torch.empty_like(slots)
    _exercise[(n,)](heads,nxt,prev,bucket,due,slots,deadlines,boundaries,
                    output,counts,capacity,e,start,debug=True,num_warps=1)
    return output,counts,(heads,nxt,prev,bucket,due)
