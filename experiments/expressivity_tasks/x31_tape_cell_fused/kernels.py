"""Slot-tile-parallel scoring and exact recurrent mutation core.

Scoring grids over stores and slot tiles. A stream-ordered core handles one
token per launch, retaining exact lazy decay, heap and bitmap semantics.
Reads and final materialization are tiled; recurrent metadata is store-local.
"""
import triton
import triton.language as tl
from .lazy_decay import advance_exact, death_exact
from .journal_mutations import schedule, cancel, next_due
from .journal_mutations import set_free, lowest_free
from .journal_mutations import remove as birth_remove, insert as birth_insert, select as birth_select
from .all_field_journal import journal_store
from .write_journal import capture_payload


@triton.jit(do_not_specialize=["Token", "Start", "AllCapacity"])
def scan_kernel(K, Q, V, Beta, Pop, Logit, Coef, Keys, Values, Strength,
                Live, Age, Output, Cases, Matched, Alloc, Pops, Freed,
                Pressure, History, Heap, Position, Due, Terminal, BirthStore, TouchStore,
                L0, L1, Root, BirthNext, BirthPrev, BirthHead, Sizes, References, Scores, ReadMax, ReadDenom, ReadValues, ReadBest, CaptureScratch,
                JournalSlots, JournalKeys, JournalValues,
                RecordPayload: tl.constexpr, JournalBlock: tl.constexpr,
                AllAddresses, AllBefore, AllWidths, AllCounts, AllBoundaries,
                AllGates, AllErrors, AllCapacity, RecordAll: tl.constexpr,
                Fresh: tl.constexpr, Start, Token,
                T: tl.constexpr, S: tl.constexpr, DK: tl.constexpr,
                DV: tl.constexpr, RT: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
                Decay: tl.constexpr, Free: tl.constexpr,
                Enable: tl.constexpr, PopThreshold: tl.constexpr,
                Eps: tl.constexpr):
    n = tl.program_id(0)
    J = (AllAddresses,AllBefore,AllWidths,AllCounts,n,AllCapacity)
    d = tl.arange(0, BV)
    heap = Heap+n*S
    position = Position+n*S
    due = Due+n*S
    terminal = Terminal+n*S
    free0 = L0+n*tl.cdiv(S,64)
    free1 = L1+n*tl.cdiv(tl.cdiv(S,64),64)
    root = Root+n
    birth_next = BirthNext+n*S
    birth_prev = BirthPrev+n*S
    birth_head = BirthHead+n
    birth_store = BirthStore+n*S
    if Fresh:
        # Import is a once-per-chunk exception. Native continuation uses only
        # scalar metadata at mutation sites.
        s = tl.arange(0, BS)
        valid = s < S
        strength = tl.load(Strength+n*S+s, valid, 0.)
        live = tl.load(Live+n*S+s, valid, 0).to(tl.int1)
        age = tl.load(Age+n*S+s, valid, 0)
        import_horizon, import_terminal = death_exact(tl.where(live & (strength > Free), strength, 0.),
                                tl.full((BS,), Free, tl.float32), Decay)
        initial_due = tl.where(live & (strength > Free) & (import_horizon >= 0), Start+import_horizon, -1)
        tl.store(due+s, initial_due, valid)
        tl.store(terminal+s, import_terminal, valid)
        tl.store(BirthStore+n*S+s, Start-age, valid)
        tl.store(position+s, -1, valid)
        tl.debug_barrier()
        size = tl.full((), 0, tl.int32)
        slot = tl.load(birth_head)
        while slot >= 0:
            time = tl.load(due+slot)
            if time >= 0:
                size = schedule(heap, position, due, size, slot, time, J, False)
            slot = tl.load(birth_next+slot)
    else:
        size = tl.load(Sizes+n)
    for t in range(Token, Token+1):
        if RecordAll:
            tl.store(AllBoundaries+n*T+t, tl.load(AllCounts+n))
        clock = Start+t
        tl.device_assert((size >= 0) & (size <= S), "invalid heap size")
        inp = n*T+t
        tile_count: tl.constexpr = RT
        best = tl.load(ReadBest+n*tile_count)
        pop = tl.load(Pop+inp)
        pop_id = tl.where((pop > PopThreshold) & (best < S),
                          best, -1)
        popped=tl.full((),0.,tl.float32)
        if pop_id>=0:
            pop_reference=tl.load(References+n*S+pop_id)
            pop_touch=tl.load(TouchStore+n*S+pop_id)
            exact=advance_exact(tl.full((1,),pop_reference,tl.float32),
                                tl.full((1,),clock-pop_touch,tl.int64),Decay)
            popped=tl.sum(exact,0)*(1.0-pop)
        if pop_id >= 0:
            pop_horizon, pop_terminal = death_exact(tl.full((1,), 1., tl.float32)*tl.where(popped > Free, popped, 1.),
                                    tl.full((1,), Free, tl.float32), Decay)
            dt = tl.sum(pop_horizon, 0)
            journal_store(terminal + pop_id, tl.sum(pop_terminal, 0), True, J, RecordAll)
            size = schedule(heap, position, due, size, pop_id, tl.where((popped > Free) & (dt >= 0), clock + dt, -1), J, RecordAll)
        if Fresh and t == 0:
            slot = tl.load(birth_head)
            while slot >= 0:
                after = tl.load(birth_next+slot)
                if tl.load(Strength+n*S+slot) <= Free:
                    journal_store(Live+n*S+slot, False, True, J, RecordAll)
                    tl.store(Freed+(inp*S+slot), 1)
                    set_free(free0, free1, root, slot, True, J, RecordAll)
                    birth_remove(birth_next, birth_prev, birth_head, slot, J, RecordAll)
                slot = after
        if pop_id >= 0:
            imported_low = Fresh & (t == 0) & (tl.load(Strength+n*S+pop_id) <= Free)
            journal_store(References+n*S+pop_id, popped, True, J, RecordAll)
            journal_store(Strength+n*S+pop_id, popped, True, J, RecordAll)
            journal_store(TouchStore+n*S+pop_id, clock, True, J, RecordAll)
            if popped <= Free:
                journal_store(Age+n*S+pop_id, clock-tl.load(BirthStore+n*S+pop_id), True, J, RecordAll)
                journal_store(Live+n*S+pop_id, False, True, J, RecordAll)
                tl.store(Freed+(inp*S+pop_id), 1)
                set_free(free0, free1, root, pop_id, True, J, RecordAll)
                if not imported_low:
                    birth_remove(birth_next, birth_prev, birth_head, pop_id, J, RecordAll)
        count = (tl.load(birth_head) >= 0).to(tl.int32)
        target, has_adjacent = birth_select(birth_next, birth_head, birth_store, clock - 1, J, RecordAll)
        safe_target = tl.maximum(target, 0)
        prev = tl.load(Values+(n*S+safe_target)*DV+d, d<DV, 0)
        content = tl.load(V+inp*DV+d, d<DV, 0)
        diff = content-tl.where(has_adjacent,prev,0.)
        residual = tl.sum(tl.where(d<DV, diff*diff, 0.), 0)
        beta = tl.load(Beta+inp)
        enabled = beta > Enable
        case = tl.where(count == 0, 1,
                        tl.where(~has_adjacent, 2, tl.where(residual<Eps, 3, 4)))
        case = tl.where(enabled, case, 0)
        first_free = lowest_free(free0, free1, root, J, RecordAll)
        tl.device_assert(first_free < S, "invalid free bitmap result")
        append = (case>0) & (case<4)
        case = tl.where(append & (first_free < 0), 5, case)
        allocation = tl.where(append & (case != 5), first_free, -1)
        matched = tl.where((enabled & has_adjacent) | (case == 5), target, -1)
        error = tl.where(count>0, tl.sqrt(residual), 0.)
        gate = tl.sigmoid(tl.load(Logit+inp)+tl.load(Coef+n)*error)
        if RecordAll:
            tl.store(AllGates+inp, gate)
            tl.store(AllErrors+inp, error)
        if allocation >= 0:
            if RecordPayload:
                capture_payload(Keys,Values,JournalSlots,JournalKeys,JournalValues,
                                t,allocation,n,S,T,DK,DV,JournalBlock)
            set_free(free0, free1, root, allocation, False, J, RecordAll)
            journal_store(BirthStore + n * S + allocation, clock, True, J, RecordAll)
            tl.debug_barrier()
            birth_insert(birth_next, birth_prev, birth_head, birth_store, allocation, clock, J, RecordAll)
            key_lanes=tl.arange(0,BK)
            key_payload=tl.load(K+inp*DK+key_lanes,key_lanes<DK,0.)
            journal_store(Keys+(n*S+allocation)*DK+key_lanes,key_payload,key_lanes<DK,J,RecordAll)
            journal_store(Values + (n * S + allocation) * DV + d, gate * content, d < DV, J, RecordAll, Values + (n * S + allocation) * DV + 0, DV, CaptureScratch+n*tl.maximum(S,DV))
        if (case == 4) | (case == 5):
            if RecordPayload:
                capture_payload(Keys,Values,JournalSlots,JournalKeys,JournalValues,
                                t,safe_target,n,S,T,DK,DV,JournalBlock)
            journal_store(Values + (n * S + safe_target) * DV + d, prev + beta * (content - prev), d < DV, J, RecordAll, Values + (n * S + safe_target) * DV + 0, DV, CaptureScratch+n*tl.maximum(S,DV))
        if allocation >= 0:
            journal_store(References+n*S+allocation, 1., True, J, RecordAll)
            journal_store(Strength+n*S+allocation, 1., True, J, RecordAll)
            journal_store(TouchStore+n*S+allocation, clock, True, J, RecordAll)
            journal_store(Live+n*S+allocation, True, True, J, RecordAll)
            append_horizon, append_terminal = death_exact(tl.full((1,), 1., tl.float32),
                                    tl.full((1,), Free, tl.float32), Decay)
            dt = tl.sum(append_horizon, 0)
            journal_store(terminal + allocation, tl.sum(append_terminal, 0), True, J, RecordAll)
            size = schedule(heap, position, due, size, allocation, tl.where(dt >= 0, clock + dt, -1), J, RecordAll)
        dead = next_due(heap, due, size, clock + 1, J, RecordAll)
        processed = 0
        while (dead >= 0) & (processed < S):
            processed += 1
            death_strength = tl.load(terminal+dead)
            journal_store(Strength + n * S + dead, death_strength, True, J, RecordAll)
            journal_store(TouchStore + n * S + dead, clock + 1, True, J, RecordAll)
            journal_store(Age + n * S + dead, clock + 1 - tl.load(BirthStore + n * S + dead), True, J, RecordAll)
            journal_store(Live + n * S + dead, False, True, J, RecordAll)
            journal_store(References+n*S+dead, death_strength, True, J, RecordAll)
            tl.store(Freed+(inp*S+dead), 2)
            birth_remove(birth_next, birth_prev, birth_head, dead, J, RecordAll)
            set_free(free0, free1, root, dead, True, J, RecordAll)
            size = cancel(heap, position, due, size, dead, J, RecordAll)
            dead = next_due(heap, due, size, clock + 1, J, RecordAll)
        tl.device_assert(dead < 0, "death heap did not drain within slot capacity")
        tl.debug_barrier()
        tl.store(Cases+inp, case)
        tl.store(Matched+inp, matched)
        tl.store(Alloc+inp, allocation)
        tl.store(Pops+inp, pop_id)
        ring = (Start+t) % 128
        journal_store(History + n * 128 + ring, case, True, J, RecordAll)
        hist_idx = tl.arange(0, 128)
        hist = tl.load(History+n*128+hist_idx)
        requests = tl.sum(((hist>0) & (hist!=4)).to(tl.int32), 0)
        caps = tl.sum((hist == 5).to(tl.int32), 0)
        tl.store(Pressure+inp, caps / tl.maximum(1, requests))
    journal_store(Sizes + n, size, True, J, RecordAll)



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


@triton.jit(do_not_specialize=["Token", "Start"])
def score_heads(Q, Keys, Live, Scores, T: tl.constexpr, S: tl.constexpr,
                DK: tl.constexpr, H: tl.constexpr, HG: tl.constexpr,
                BK: tl.constexpr, BN: tl.constexpr, Start, Token):
    batch=tl.program_id(0)
    group=tl.program_id(1)
    tile=tl.program_id(2)
    row=tl.arange(0,16)
    col=tl.arange(0,BN)
    head=group*HG+col//32
    slot=tile*32+col%32
    n=batch*H+head
    live=tl.load(Live+n*S+slot,(head<H)&(slot<S),0).to(tl.int1)
    if tl.sum(live.to(tl.int32),0)>0:
        rk=tl.arange(0,32)
        qhead=group*HG+row
        cross=tl.full((16,BN),0.,tl.float32)
        # K tiles bound shared memory for the 16-head/512-column score block.
        for offset in range(tl.cdiv(DK,32)):
            d=offset*32+rk
            query=tl.load(Q+((batch*H+qhead[:,None])*T+Token)*DK+d[None,:],
                          (row[:,None]<HG)&(qhead[:,None]<H)&(d[None,:]<DK),0.)
            key=tl.load(Keys+(n[None,:]*S+slot[None,:])*DK+d[:,None],
                        live[None,:]&(d[:,None]<DK),0.)
            cross+=tl.dot(query,key,input_precision='tf32x3')
        score=tl.sum(tl.where(row[:,None]==col[None,:]//32,cross,0.),0)
        tl.store(Scores+n*S+slot,score*(2.0/tl.sqrt(float(DK))),live)


@triton.jit(do_not_specialize=["Token", "Start"])
def read_tiles(Q, Keys, Values, Live, References, Touch, Scores,
               ReadMax, ReadDenom, ReadValues, ReadBest,
               T: tl.constexpr, S: tl.constexpr, DK: tl.constexpr,
               DV: tl.constexpr, RT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
               Start, Token, Decay: tl.constexpr):
    n = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile*32 + tl.arange(0,32)
    tile_count: tl.constexpr = RT
    header=n*tile_count+tile
    live=tl.load(Live+n*S+columns,columns<S,0).to(tl.int1)
    if tl.sum(live.to(tl.int32),0)>0:
        kd=tl.arange(0,BK)
        query=tl.load(Q+(n*T+Token)*DK+kd,kd<DK,0.)
        keys=tl.load(Keys+(n*S+columns[:,None])*DK+kd[None,:],
                     live[:,None] & (kd[None,:]<DK),0.)
        scores=tl.sum(keys*query[None,:],1)*(2.0/tl.sqrt(float(DK)))
        masked=tl.where(live,scores,-1.0e30)
        maximum=tl.max(masked,0)
        numer=tl.where(live,tl.exp(masked-maximum),0.)
        reference=tl.load(References+n*S+columns,live,0.)
        touch=tl.load(Touch+n*S+columns,live,Start+Token)
        strength=advance_exact(reference,tl.where(live,Start+Token-touch,0),Decay)
        weighted=numer*strength
        d=tl.arange(0,BV)
        values=tl.load(Values+(n*S+columns[:,None])*DV+d[None,:],
                       live[:,None] & (d[None,:]<DV),0.)
        partial=tl.sum(weighted[:,None]*values,0)
        best=tl.min(tl.where(live & (masked==maximum),columns,S),0)
        tl.store(ReadMax+header,maximum)
        tl.store(ReadDenom+header,tl.sum(numer,0))
        tl.store(ReadBest+header,best)
        tl.store(ReadValues+header*DV+d,partial,d<DV)
    else:
        tl.store(ReadMax+header,-1.0e30)
        tl.store(ReadDenom+header,0.)
        tl.store(ReadBest+header,S)
        # Empty partial payloads are never loaded by the reduction pass.


@triton.jit(do_not_specialize=["Token"])
def reduce_read(ReadMax, ReadDenom, ReadValues, ReadBest, Output,
                T: tl.constexpr, S: tl.constexpr, DV: tl.constexpr, RT: tl.constexpr,
                BT: tl.constexpr, Token, BD: tl.constexpr):
    n=tl.program_id(0)
    tiles=tl.arange(0,BT)
    d=tl.program_id(1)*BD+tl.arange(0,BD)
    tile_count: tl.constexpr=RT
    valid=tiles<tile_count
    maximum=tl.load(ReadMax+n*tile_count+tiles,valid,-1.0e30)
    denom=tl.load(ReadDenom+n*tile_count+tiles,valid,0.)
    if tl.program_id(1)==0:
        candidates=tl.load(ReadBest+n*tile_count+tiles,valid,S)
        best=tl.min(tl.where(maximum==tl.max(maximum,0),candidates,S),0)
        tl.store(ReadBest+n*tile_count,best)

    scale=tl.exp(maximum-tl.max(maximum,0))
    total=tl.sum(denom*scale,0)
    payload=tl.load(ReadValues+(n*tile_count+tiles[:,None])*DV+d[None,:],
                    valid[:,None] & (denom[:,None]>0) & (d[None,:]<DV),0.)
    read=tl.sum(payload*scale[:,None],0)/tl.maximum(total,1.0e-30)
    tl.store(Output+(n*T+Token)*DV+d,read,d<DV)



@triton.jit
def initialize_free_words(Live, L0, L1, Root, S: tl.constexpr,
                          BW: tl.constexpr, BP: tl.constexpr):
    n=tl.program_id(0)
    words=tl.arange(0,BW)
    bits=tl.arange(0,64)
    slots=words[:,None]*64+bits[None,:]
    free=(slots<S) & ~tl.load(Live+n*S+slots,slots<S,0)
    packed=tl.sum(tl.where(free,tl.full((),1,tl.uint64)<<bits[None,:],0),1)
    tl.store(L0+n*tl.cdiv(S,64)+words,packed,words<tl.cdiv(S,64))
    parents=tl.arange(0,BP)
    children=parents[:,None]*64+bits[None,:]
    child_bits=tl.gather(packed,tl.minimum(children,BW-1).reshape((BP*64,)),0).reshape((BP,64))
    summary=tl.sum(tl.where((children<tl.cdiv(S,64)) & (child_bits!=0),
                           tl.full((),1,tl.uint64)<<bits[None,:],0),1)
    tl.store(L1+n*tl.cdiv(tl.cdiv(S,64),64)+parents,summary,parents<tl.cdiv(tl.cdiv(S,64),64))
    root=tl.sum(tl.where(summary!=0,tl.full((),1,tl.uint64)<<parents,0),0)
    tl.store(Root+n,root)


@triton.jit(do_not_specialize=["Clock", "Capacity"])
def materialize_state(References, Touch, Birth, Live, Strength, Age,
                      Addresses, Before, Widths, Counts, Capacity,
                      Record: tl.constexpr, S: tl.constexpr, Tiles: tl.constexpr, Clock,
                      Decay: tl.constexpr):
    n=tl.program_id(0)
    # Capture count is store-local and non-atomic: one program must own all
    # tiles of a store, including imported states with many live slot tiles.
    for tile in range(Tiles):
        slots=tile*32+tl.arange(0,32)
        valid=slots<S
        live=tl.load(Live+n*S+slots,valid,0).to(tl.int1)
        if tl.sum(live.to(tl.int32),0)>0:
            ref=tl.load(References+n*S+slots,valid,0.)
            touch=tl.load(Touch+n*S+slots,valid,Clock)
            born=tl.load(Birth+n*S+slots,valid & live,Clock)
            strength=advance_exact(ref,tl.where(live,Clock-touch,0),Decay)
            age=Clock-born
            J=(Addresses,Before,Widths,Counts,n,Capacity)
            # These independent <=32-lane groups capture directly in registers;
            # no store-local staging scratch is shared across slot-tile programs.
            journal_store(Strength+n*S+slots,strength,valid & live,J,Record)
            journal_store(Age+n*S+slots,age,valid & live,J,Record)
