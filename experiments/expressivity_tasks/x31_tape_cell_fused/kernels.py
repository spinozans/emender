"""Device-only sequential correctness baseline, deliberately not performance certified.

One program owns a store for the entire chunk. Exact fp32 recurrent decay is
used instead of analytic powers. Dense metadata sweeps and dimension-wise
reductions are an explicit prototype limitation (no tensor-core claim).
"""
import triton
import triton.language as tl


@triton.jit
def scan_kernel(K, Q, V, Beta, Pop, Logit, Coef, Keys, Values, Strength,
                Live, Age, Output, Cases, Matched, Alloc, Pops, Freed,
                Pressure, History, Start: tl.constexpr,
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
    for t in range(T):
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
        freed_pop = live & (strength <= Free)
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
        first_free = tl.min(tl.where(valid & ~live, s, 2147483647), 0)
        append = (case>0) & (case<4)
        case = tl.where(append & (first_free == 2147483647), 5, case)
        allocation = tl.where(append & (case != 5), first_free, -1)
        matched = tl.where((enabled & has_adjacent) | (case == 5), target, -1)
        error = tl.where(count>0, tl.sqrt(residual), 0.)
        gate = tl.sigmoid(tl.load(Logit+inp)+tl.load(Coef+n)*error)
        if allocation >= 0:
            for j in range(DK):
                tl.store(Keys+(n*S+allocation)*DK+j, tl.load(K+inp*DK+j))
            tl.store(Values+(n*S+allocation)*DV+d, gate*content, d<DV)
        if (case == 4) | (case == 5):
            tl.store(Values+(n*S+safe_target)*DV+d,
                     prev+beta*(content-prev), d<DV)
        strength = tl.where(s == allocation, 1., strength)
        live = live | (s == allocation)
        age = tl.where(s == allocation, 0, age)
        # Preserve the native multiply's rounding; launch with enable_fp_fusion=False.
        strength = tl.where(live, strength*Decay, strength)
        age = tl.where(live, age+1, age)
        freed_decay = live & (strength <= Free)
        live = live & ~freed_decay
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
