"""Persistent store-owner scan with the oracle tiled reduction trees.

A device token loop retains exact lazy decay, heap, bitmap and journal
semantics. Each store owns its tile scratch throughout the chunk.
"""
import triton
import triton.language as tl
from ..x31_tape_cell_fused.lazy_decay import advance_exact, death_exact
from .mutations import schedule, cancel, next_due
from .mutations import set_free, lowest_free
from .mutations import remove as birth_remove, insert as birth_insert, select as birth_select
from .journal import journal_store
from ..x31_tape_cell_fused.write_journal import capture_payload


@triton.jit(do_not_specialize=["Token", "Start", "AllCapacity"])
def scan_kernel(K, Q, V, Beta, Pop, Logit, Coef, Keys, Values, Strength,
                Live, Age, Output, Cases, Matched, Alloc, Pops, Freed,
                Pressure, History, Heap, Position, Due, Terminal, BirthStore, TouchStore,
                L0, L1, Root, BirthNext, BirthPrev, BirthHead, Sizes, References, Scores, ReadMax, ReadDenom, ReadValues, ReadBest, ReadStrengthCache, CaptureScratch,
                JournalSlots, JournalKeys, JournalValues,
                RecordPayload: tl.constexpr, JournalBlock: tl.constexpr,
                AllAddresses, AllBefore, AllWidths, AllCounts, AllBoundaries,
                AllGates, AllErrors, AllCapacity, RecordAll: tl.constexpr,
                Fresh: tl.constexpr, InitialBound: tl.constexpr, Start, Token,
                T: tl.constexpr, S: tl.constexpr, DK: tl.constexpr,
                DV: tl.constexpr, RT: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
                Decay: tl.constexpr, Free: tl.constexpr,
                Enable: tl.constexpr, PopThreshold: tl.constexpr,
                Eps: tl.constexpr):
    # Keep store-address arithmetic wide while retaining debug assertions.
    n = tl.program_id(0).to(tl.int64)
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
    reached=tl.full((),InitialBound if InitialBound>=0 else 0,tl.int32)
    for t in range(T):
        # Track actual allocation reach on device. Keep the fixed reduction
        # tree, but do not scan future empty tiles or read stale partials.
        if InitialBound>=0:
            active_tiles=tl.minimum(RT,tl.maximum(1,tl.cdiv(reached,32)))
        else:
            active_tiles=RT
        for tile in range(active_tiles):
            r_columns = tile*32 + tl.arange(0,32)
            r_tile_count: tl.constexpr = RT
            r_header=n*r_tile_count+tile
            r_live=tl.load(Live+n*S+r_columns,r_columns<S,0).to(tl.int1)
            if tl.sum(r_live.to(tl.int32),0)>0:
                r_kd=tl.arange(0,BK)
                r_query=tl.load(Q+(n*T+t)*DK+r_kd,r_kd<DK,0.)
                r_keys=tl.load(Keys+(n*S+r_columns[:,None])*DK+r_kd[None,:],
                             r_live[:,None] & (r_kd[None,:]<DK),0.)
                r_scores=tl.sum(r_keys*r_query[None,:],1)*(2.0/tl.sqrt(float(DK)))
                r_masked=tl.where(r_live,r_scores,-1.0e30)
                r_maximum=tl.max(r_masked,0)
                r_numer=tl.where(r_live,tl.exp(r_masked-r_maximum),0.)
                r_reference=tl.load(References+n*S+r_columns,r_live,0.)
                r_touch=tl.load(TouchStore+n*S+r_columns,r_live,Start+t)
                if T>512:
                    if t==0:
                        r_strength=advance_exact(r_reference,tl.where(r_live,Start+t-r_touch,0),Decay)
                    else:
                        cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)
                        base=tl.where(r_touch>=Start+t-1,r_reference,cached)
                        if Decay==1.0:
                            next_strength=base
                        else:
                            next_strength=base*Decay
                        r_strength=tl.where(r_touch>=Start+t,r_reference,next_strength)
                    tl.store(ReadStrengthCache+n*S+r_columns,r_strength,r_live)
                else:
                    r_strength=advance_exact(r_reference,tl.where(r_live,Start+t-r_touch,0),Decay)
                r_weighted=r_numer*r_strength
                r_d=tl.arange(0,BV)
                r_values=tl.load(Values+(n*S+r_columns[:,None])*DV+r_d[None,:],
                               r_live[:,None] & (r_d[None,:]<DV),0.)
                r_partial=tl.sum(r_weighted[:,None]*r_values,0)
                r_best=tl.min(tl.where(r_live & (r_masked==r_maximum),r_columns,S),0)
                tl.store(ReadMax+r_header,r_maximum)
                tl.store(ReadDenom+r_header,tl.sum(r_numer,0))
                tl.store(ReadBest+r_header,r_best)
                tl.store(ReadValues+r_header*DV+r_d,r_partial,r_d<DV)
            else:
                tl.store(ReadMax+r_header,-1.0e30)
                tl.store(ReadDenom+r_header,0.)
                tl.store(ReadBest+r_header,S)
                # Empty r_partial payloads are never loaded by the reduction pass.
        tl.debug_barrier()
        for dim_tile in range(tl.cdiv(DV,8)):
            rr_tiles=tl.arange(0,BT)
            rr_d=dim_tile*8+tl.arange(0,8)
            rr_tile_count: tl.constexpr=RT
            rr_valid=rr_tiles<active_tiles
            rr_maximum=tl.load(ReadMax+n*rr_tile_count+rr_tiles,rr_valid,-1.0e30)
            rr_denom=tl.load(ReadDenom+n*rr_tile_count+rr_tiles,rr_valid,0.)
            if dim_tile==0:
                rr_candidates=tl.load(ReadBest+n*rr_tile_count+rr_tiles,rr_valid,S)
                rr_best=tl.min(tl.where(rr_maximum==tl.max(rr_maximum,0),rr_candidates,S),0)
                tl.store(ReadBest+n*rr_tile_count,rr_best)

            rr_scale=tl.exp(rr_maximum-tl.max(rr_maximum,0))
            rr_total=tl.sum(rr_denom*rr_scale,0)
            rr_payload=tl.load(ReadValues+(n*rr_tile_count+rr_tiles[:,None])*DV+rr_d[None,:],
                            rr_valid[:,None] & (rr_denom[:,None]>0) & (rr_d[None,:]<DV),0.)
            rr_read=tl.sum(rr_payload*rr_scale[:,None],0)/tl.maximum(rr_total,1.0e-30)
            tl.store(Output+(n*T+t)*DV+rr_d,rr_read,rr_d<DV)
        tl.debug_barrier()
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
                    tl.store(Freed+(inp.to(tl.int64)*S+slot), 1)
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
                tl.store(Freed+(inp.to(tl.int64)*S+pop_id), 1)
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
        if InitialBound>=0:
            reached=tl.maximum(reached,allocation+1)
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
            tl.store(Freed+(inp.to(tl.int64)*S+dead), 2)
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
            born=tl.load(Birth+n*S+slots,valid,Clock)
            strength=advance_exact(ref,tl.where(live,Clock-touch,0),Decay)
            age=Clock-born
            J=(Addresses,Before,Widths,Counts,n,Capacity)
            # These independent <=32-lane groups capture directly in registers;
            # no store-local staging scratch is shared across slot-tile programs.
            journal_store(Strength+n*S+slots,strength,valid & live,J,Record)
            journal_store(Age+n*S+slots,age,valid & live,J,Record)
