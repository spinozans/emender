"""Device-only sequential correctness baseline, not performance certified.

One program owns a store. Strengths use exact lazy interval jumps on read;
cancellable heap events freeze deaths, and a three-level bitmap allocates.
Import, read scoring and birth-target selection still scale with slot count.
The read still uses dimension-wise reductions (no tensor-core claim).
"""
import triton
import triton.language as tl
from .lazy_decay import advance_exact, death_exact
from .device_schedule import schedule, cancel, next_due
from .device_free import set_free, lowest_free


@triton.jit
def scan_kernel(K, Q, V, Beta, Pop, Logit, Coef, Keys, Values, Strength,
                Live, Age, Output, Cases, Matched, Alloc, Pops, Freed,
                Pressure, History, Heap, Position, Due, Terminal, BirthStore, TouchStore,
                L0, L1, Root, Start: tl.constexpr,
                T: tl.constexpr, S: tl.constexpr, DK: tl.constexpr,
                DV: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr,
                Decay: tl.constexpr, Free: tl.constexpr,
                Enable: tl.constexpr, PopThreshold: tl.constexpr,
                Eps: tl.constexpr):
    n = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, BV)
    valid = s < S
    strength = tl.load(Strength + n*S+s, valid, 0)
    live = tl.load(Live + n*S+s, valid, 0).to(tl.int1)
    age = tl.load(Age + n*S+s, valid, 0)
    birth = -age
    touch = tl.full((BS,), 0, tl.int64)
    reference = strength
    heap = Heap+n*S
    position = Position+n*S
    due = Due+n*S
    terminal = Terminal+n*S
    free0 = L0+n*tl.cdiv(S,64)
    free1 = L1+n*tl.cdiv(tl.cdiv(S,64),64)
    root = Root+n
    import_horizon, import_terminal = death_exact(tl.where(live & (strength > Free), strength, 1.),
                            tl.full((BS,), Free, tl.float32), Decay)
    initial_due = tl.where(live & (strength > Free) & (import_horizon >= 0), import_horizon, -1)
    tl.store(due+s, initial_due, valid)
    tl.store(terminal+s, import_terminal, valid)
    tl.store(BirthStore+n*S+s, birth, valid)
    tl.store(position+s, -1, valid)
    tl.debug_barrier()
    size = tl.full((), 0, tl.int32)
    # Import is an O(S) boundary operation. Token death processing below only
    # visits the cancellable heap's due events; it never scans for deaths.
    for slot in range(S):
        if not tl.load(Live+n*S+slot):
            set_free(free0, free1, root, slot, True)
        time = tl.load(due+slot)
        if time >= 0:
            size = schedule(heap, position, due, size, slot, time)
    for t in range(T):
        tl.device_assert((size >= 0) & (size <= S), "invalid heap size")
        strength = advance_exact(reference, tl.where(live, t-touch, 0), Decay)
        age = tl.where(live, t-birth, age)
        inp = n*T+t
        scores = tl.full((BS,), 0, tl.float32)
        for j in range(DK):
            kval = tl.load(Keys+(n*S+s)*DK+j, valid, 0)
            query = tl.load(Q+inp*DK+j)
            scores = scores + kval*query
        scores = scores * (2.0 / tl.sqrt(float(DK)))
        masked = tl.where(live & valid, scores, -1.0e30)
        maximum = tl.max(masked, 0)
        numer = tl.exp(masked-maximum)
        # Padded slots must not change the denominator for an empty store.
        numer = tl.where(valid, numer, 0.)
        weights = numer / tl.sum(numer, 0) * strength * live
        for j in range(DV):
            vals = tl.load(Values+(n*S+s)*DV+j, valid, 0)
            read = tl.sum(weights*vals, 0)
            tl.store(Output+inp*DV+j, read)
        pop = tl.load(Pop+inp)
        best = tl.min(tl.where(live & (masked == maximum), s, 2147483647), 0)
        pop_id = tl.where((pop > PopThreshold) & (tl.sum(live.to(tl.int32), 0)>0),
                          best, -1)
        strength = tl.where(s == pop_id, strength*(1.0-pop), strength)
        reference = tl.where(s == pop_id, strength, reference)
        touch = tl.where(s == pop_id, t, touch)
        popped = tl.sum(tl.where(s == pop_id, strength, 0.), 0)
        if pop_id >= 0:
            pop_horizon, pop_terminal = death_exact(tl.full((1,), 1., tl.float32)*tl.where(popped > Free, popped, 1.),
                                    tl.full((1,), Free, tl.float32), Decay)
            dt = tl.sum(pop_horizon, 0)
            tl.store(terminal+pop_id, tl.sum(pop_terminal, 0))
            size = schedule(heap, position, due, size, pop_id,
                            tl.where((popped > Free) & (dt >= 0), t+dt, -1))
        freed_pop = live & (strength <= Free) & ((s == pop_id) | (t == 0))
        # Import can contain already-under-threshold live rows; native kills
        # them after the first read. Freeze those references at that boundary.
        reference = tl.where(freed_pop, strength, reference)
        if t == 0:
            # Imported under-threshold rows have no scheduled death, and
            # native freezes them after this read rather than before it.
            for slot in range(S):
                imported_live = tl.load(Live+n*S+slot)
                imported_strength = tl.load(Strength+n*S+slot)
                if imported_live & (imported_strength <= Free):
                    set_free(free0, free1, root, slot, True)
        if pop_id >= 0:
            if popped <= Free:
                set_free(free0, free1, root, pop_id, True)
        live = live & ~freed_pop
        count = tl.sum(live.to(tl.int32), 0)
        adjacent = live & (age == 1)
        has_adjacent = tl.sum(adjacent.to(tl.int32), 0)>0
        young = tl.min(tl.where(live, age, 2147483647), 0)
        target = tl.min(tl.where(tl.where(has_adjacent, adjacent,
                                         live & (age == young)), s, 2147483647), 0)
        safe_target = tl.minimum(target, S-1)
        prev = tl.load(Values+(n*S+safe_target)*DV+d, d<DV, 0)
        content = tl.load(V+inp*DV+d, d<DV, 0)
        diff = content-tl.where(has_adjacent,prev,0.)
        residual = tl.sum(tl.where(d<DV, diff*diff, 0.), 0)
        beta = tl.load(Beta+inp)
        enabled = beta > Enable
        case = tl.where(count == 0, 1,
                        tl.where(~has_adjacent, 2, tl.where(residual<Eps, 3, 4)))
        case = tl.where(enabled, case, 0)
        first_free = lowest_free(free0, free1, root)
        tl.device_assert(first_free < S, "invalid free bitmap result")
        append = (case>0) & (case<4)
        case = tl.where(append & (first_free < 0), 5, case)
        allocation = tl.where(append & (case != 5), first_free, -1)
        matched = tl.where((enabled & has_adjacent) | (case == 5), target, -1)
        error = tl.where(count>0, tl.sqrt(residual), 0.)
        gate = tl.sigmoid(tl.load(Logit+inp)+tl.load(Coef+n)*error)
        if allocation >= 0:
            set_free(free0, free1, root, allocation, False)
            tl.store(BirthStore+n*S+allocation, t)
            for j in range(DK):
                tl.store(Keys+(n*S+allocation)*DK+j, tl.load(K+inp*DK+j))
            tl.store(Values+(n*S+allocation)*DV+d, gate*content, d<DV)
        if (case == 4) | (case == 5):
            tl.store(Values+(n*S+safe_target)*DV+d,
                     prev+beta*(content-prev), d<DV)
        reference = tl.where(s == allocation, 1., reference)
        touch = tl.where(s == allocation, t, touch)
        live = live | (s == allocation)
        birth = tl.where(s == allocation, t, birth)
        if allocation >= 0:
            append_horizon, append_terminal = death_exact(tl.full((1,), 1., tl.float32),
                                    tl.full((1,), Free, tl.float32), Decay)
            dt = tl.sum(append_horizon, 0)
            tl.store(terminal+allocation, tl.sum(append_terminal, 0))
            size = schedule(heap, position, due, size, allocation,
                            tl.where(dt >= 0, t+dt, -1))
        # Spill once at the read/mutation boundary. Each due event now edits
        # scalar metadata, rather than applying O(S) vector updates per death.
        tl.store(Strength+n*S+s, reference, valid)
        tl.store(Live+n*S+s, live, valid)
        tl.store(Age+n*S+s, age, valid)
        tl.store(TouchStore+n*S+s, touch, valid)
        tl.debug_barrier()
        dead = next_due(heap, due, size, t+1)
        processed = 0
        while (dead >= 0) & (processed < S):
            processed += 1
            death_strength = tl.load(terminal+dead)
            tl.store(Strength+n*S+dead, death_strength)
            tl.store(TouchStore+n*S+dead, t+1)
            tl.store(Age+n*S+dead, t+1-tl.load(BirthStore+n*S+dead))
            tl.store(Live+n*S+dead, False)
            set_free(free0, free1, root, dead, True)
            size = cancel(heap, position, due, size, dead)
            dead = next_due(heap, due, size, t+1)
        tl.device_assert(dead < 0, "death heap did not drain within slot capacity")
        tl.debug_barrier()
        new_live = tl.load(Live+n*S+s, valid, 0).to(tl.int1)
        freed_decay = live & ~new_live
        live = new_live
        reference = tl.load(Strength+n*S+s, valid, 0)
        touch = tl.load(TouchStore+n*S+s, valid, 0)
        age = tl.load(Age+n*S+s, valid, 0)
        tl.store(Freed+(inp*S+s), freed_pop.to(tl.int8)+2*freed_decay.to(tl.int8), valid)
        tl.store(Cases+inp, case)
        tl.store(Matched+inp, matched)
        tl.store(Alloc+inp, allocation)
        tl.store(Pops+inp, pop_id)
        ring = (Start+t) % 128
        tl.store(History+n*128+ring, case)
        hist_idx = tl.arange(0, 128)
        hist = tl.load(History+n*128+hist_idx)
        requests = tl.sum(((hist>0) & (hist!=4)).to(tl.int32), 0)
        caps = tl.sum((hist == 5).to(tl.int32), 0)
        tl.store(Pressure+inp, caps / tl.maximum(1, requests))
    strength = advance_exact(reference, tl.where(live, T-touch, 0), Decay)
    age = tl.where(live, T-birth, age)
    tl.store(Strength+n*S+s, strength, valid)
    tl.store(Live+n*S+s, live, valid)
    tl.store(Age+n*S+s, age, valid)


@triton.jit
def append_phase_kernel(Content, Gates, FreeSlots, KeysIn, Keys, Values,
                        Strength, Live, Age, T: tl.constexpr, S: tl.constexpr,
                        DK: tl.constexpr, DV: tl.constexpr,
                        DecayCurve, BLOCK: tl.constexpr):
    """Certified append-only phase: free ranks supplied once, no routing.

    Each event writes a unique destination. DecayCurve contains the exact
    repeated-fp32 unit recurrence, not pow(decay, age).
    """
    event = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    slot = tl.load(FreeSlots+n*T+event)
    gate = tl.load(Gates+n*T+event)
    key = tl.load(KeysIn+(n*T+event)*DK+d, d<DK, 0)
    val = tl.load(Content+(n*T+event)*DV+d, d<DV, 0)
    tl.store(Keys+(n*S+slot)*DK+d, key, d<DK)
    tl.store(Values+(n*S+slot)*DV+d, gate*val, d<DV)
    tl.store(Strength+n*S+slot, tl.load(DecayCurve+T-event))
    tl.store(Live+n*S+slot, 1)
    tl.store(Age+n*S+slot, T-event)


@triton.jit
def free_rank_kernel(Live, Slots, Overflow, S: tl.constexpr, T: tl.constexpr,
                     BLOCK: tl.constexpr):
    n = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    free = (s<S) & ~tl.load(Live+n*S+s, s<S, 1).to(tl.int1)
    rank = tl.cumsum(free.to(tl.int32),0)-1
    tl.store(Slots+n*T+rank,s,free & (rank<T))
    tl.store(Overflow+n,tl.sum(free.to(tl.int32),0)<T)


@triton.jit
def undo_kernel(Store, Slots, Before, E: tl.constexpr, S: tl.constexpr,
                D: tl.constexpr, BLOCK: tl.constexpr):
    """Reverse ordered sparse before-images; repeated slot IDs are legal."""
    n = tl.program_id(0)
    d = tl.arange(0,BLOCK)
    for rev in range(E):
        event = E-1-rev
        slot = tl.load(Slots+n*E+event)
        prior = tl.load(Before+(n*E+event)*D+d,d<D,0)
        if slot >= 0:
            tl.store(Store+(n*S+slot)*D+d,prior,d<D)
