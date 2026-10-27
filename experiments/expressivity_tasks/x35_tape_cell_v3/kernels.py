"""Sequential recorded-decision core with immutable causal read versions.

One program owns a store for the entire chunk. Routing and payload versions
share mutation sites and pre-token boundaries with the G10 journal. Only POP
targets need sequential query scoring; output reads run after the core.
"""
import triton
import triton.language as tl
from ..x31_tape_cell_fused.lazy_decay import advance_exact, death_exact
from ..x31_tape_cell_fused.journal_mutations import schedule, cancel, next_due
from ..x31_tape_cell_fused.journal_mutations import set_free, lowest_free
from ..x31_tape_cell_fused.journal_mutations import remove as birth_remove, insert as birth_insert, select as birth_select
from .all_field_journal import journal_store
from ..x31_tape_cell_fused.write_journal import capture_payload


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
                VersionKeys, VersionValues, VersionMap, Views, ViewStrength,
                VS: tl.constexpr, VM: tl.constexpr,
                T: tl.constexpr, S: tl.constexpr, DK: tl.constexpr,
                DV: tl.constexpr, RT: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
                Decay: tl.constexpr, Free: tl.constexpr,
                Enable: tl.constexpr, PopThreshold: tl.constexpr,
                Eps: tl.constexpr, IncrementalViews: tl.constexpr = True):
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
    slots = tl.arange(0, BS)
    if IncrementalViews:
        initial_live = (slots < VS) & tl.load(Live+n*S+slots, slots<VS, 0).to(tl.int1)
        initial_refs = tl.load(References+n*S+slots, initial_live, 0.)
        initial_touch = tl.load(TouchStore+n*S+slots, initial_live, Start)
        running_strength = advance_exact(initial_refs, tl.where(initial_live,Start-initial_touch,0),Decay)
    for t in range(T):
        if RecordAll:
            tl.store(AllBoundaries+n*T+t, tl.load(AllCounts+n))
        clock = Start+t
        tl.device_assert((size >= 0) & (size <= S), "invalid heap size")
        inp = n*T+t
        tile_count: tl.constexpr = RT
        # Emit pre-token views from the same state whose decisions are journaled.
        slots = tl.arange(0, BS)
        active = (slots < VS) & tl.load(Live+n*S+slots, slots<VS, 0).to(tl.int1)
        ids = tl.load(VersionMap+n*S+slots, slots<VS, -1)
        tl.store(Views+(n*T+t)*VS+slots, tl.where(active,ids,-1), slots<VS)
        if IncrementalViews:
            view_exact = tl.where(active,running_strength,0.)
        else:
            refs = tl.load(References+n*S+slots, active, 0.)
            touched = tl.load(TouchStore+n*S+slots, active, clock)
            view_exact = advance_exact(refs,tl.where(active,clock-touched,0),Decay)
        tl.store(ViewStrength+(n*T+t)*VS+slots,view_exact,slots<VS)
        pop = tl.load(Pop+inp)
        best = tl.full((),S,tl.int32)
        if pop > PopThreshold:
            maximum = tl.full((),float("-inf"),tl.float32)
            kd = tl.arange(0,BK)
            query = tl.load(Q+inp*DK+kd,kd<DK,0.)
            for tile in range(RT):
                ss = tile*32+tl.arange(0,32)
                ll = tl.load(Live+n*S+ss,ss<VS,0).to(tl.int1)
                kk = tl.load(Keys+(n*S+ss[:,None])*DK+kd[None,:],ll[:,None] & (kd[None,:]<DK),0.)
                score = tl.sum(kk*query[None,:],1)*(2.0/tl.sqrt(float(DK)))
                score = tl.where(ll,score,-1.e30)
                mm = tl.max(score,0)
                candidate = tl.min(tl.where(ll & (score==mm),ss,S),0)
                best = tl.where(mm>maximum,candidate,tl.where(mm==maximum,tl.minimum(best,candidate),best))
                maximum = tl.maximum(maximum,mm)
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
        if IncrementalViews:
            # Exact native fp32 boundaries: pre-read -> POP -> allocation -> decay.
            # Dead lanes are masked at emission; reuse resets the lane to one.
            running_strength = tl.where(slots==pop_id,running_strength*(1.0-pop),running_strength)
            running_strength = tl.where(slots==allocation,1.,running_strength)
            running_strength = running_strength*Decay
        tl.device_assert(dead < 0, "death heap did not drain within slot capacity")
        tl.debug_barrier()
        # Each token writes at most one payload row. Shared versions are
        # immutable; deaths change visibility and reuse creates a new version.
        changed = tl.where(allocation>=0,allocation,tl.where((case==4)|(case==5),safe_target,-1))
        if changed >= 0:
            version = VS+t
            kd = tl.arange(0,BK)
            payload_key = tl.load(Keys+(n*S+changed)*DK+kd,kd<DK,0.)
            payload_value = tl.load(Values+(n*S+changed)*DV+d,d<DV,0.)
            tl.store(VersionKeys+(n*VM+version)*DK+kd,payload_key,kd<DK)
            tl.store(VersionValues+(n*VM+version)*DV+d,payload_value,d<DV)
            tl.store(VersionMap+n*S+changed,version)
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
            born=tl.load(Birth+n*S+slots,valid,Clock)
            strength=advance_exact(ref,tl.where(live,Clock-touch,0),Decay)
            age=Clock-born
            J=(Addresses,Before,Widths,Counts,n,Capacity)
            # These independent <=32-lane groups capture directly in registers;
            # no store-local staging scratch is shared across slot-tile programs.
            journal_store(Strength+n*S+slots,strength,valid & live,J,Record)
            journal_store(Age+n*S+slots,age,valid & live,J,Record)
