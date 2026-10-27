import pytest
import torch
from experiments.expressivity_tasks.x31_tape_cell_fused.tests.test_parity import device
from experiments.expressivity_tasks.x35_tape_cell_v3.timer_wheel import exercise


@pytest.mark.parametrize('start',[0,250,65530,2**32-6,2**56-6,2**63-20])
def test_radix_boundaries_cancellation_past_due_ties(start):
    dev=device()
    slots=[0,1,2,1,3,0,-1,2,-1]
    deadlines=[start+3,start+7,start+7,-1,start-1 if start else 0,start+8,-1,start+6,-1]
    boundaries=[start-1]*5+[start+2,start+5,start+7,start+9]
    tensors=[torch.tensor([x],device=dev,dtype=torch.int64) for x in (slots,deadlines,boundaries)]
    output,counts,owners=exercise(*tensors,capacity=4,start=start)
    pending={}
    expected=[]
    for s,d,b in zip(slots,deadlines,boundaries):
        if s>=0:
            pending.pop(s,None)
            if d>=0:
                pending[s]=d
        dead=sorted(s for s,d in pending.items() if d<=b and b>=start)
        for s in dead:
            del pending[s]
        expected.append(dead)
    for e,want in enumerate(expected):
        assert sorted(output[0,e,:counts[0,e]].tolist())==want, e
    assert owners[0].eq(-1).all()
    assert owners[1].eq(-1).all()
    assert owners[2].eq(-1).all()
    assert owners[3].eq(-1).all()


def test_long_future_timer_retained_without_stale_links():
    dev=device()
    slots=torch.tensor([[0,0,1,0,-1]],device=dev)
    deadlines=torch.tensor([[2**62,2**48,2**56,-1,-1]],device=dev)
    boundaries=torch.tensor([[-1,-1,-1,-1,3]],device=dev)
    out,counts,owners=exercise(slots,deadlines,boundaries,2)
    heads,nxt,prev,bucket,due=owners
    assert counts.eq(0).all()
    assert bucket[0,0]==-1 and bucket[0,1]>=0
    assert (heads>=0).sum()==1
    assert nxt.eq(-1).all() and prev.eq(-1).all()
    assert due[0,1]==2**56


def test_random_reschedule_matches_deadline_model():
    import random
    dev=device()
    rng=random.Random(3502)
    n,e,s=3,80,16
    schedules=[]; deadlines=[]; boundaries=[]; expected=[]
    for _ in range(n):
        ss=[];dd=[];bb=[];want=[];pending={};clock=0
        for _ in range(e):
            slot=rng.randrange(-1,s)
            deadline=rng.choice([-1,clock,clock+rng.randrange(1,600),clock+2**40])
            boundary=clock+rng.randrange(0,8)
            if slot>=0:
                pending.pop(slot,None)
                if deadline>=0:
                    pending[slot]=deadline
            dead=sorted(x for x,d in pending.items() if d<=boundary)
            for x in dead:
                del pending[x]
            ss.append(slot);dd.append(deadline);bb.append(boundary);want.append(dead)
            clock=boundary+1
        schedules.append(ss);deadlines.append(dd);boundaries.append(bb);expected.append(want)
    inputs=[torch.tensor(x,device=dev) for x in (schedules,deadlines,boundaries)]
    out,count,owners=exercise(*inputs,s)
    for n in range(len(expected)):
        for t in range(e):
            assert sorted(out[n,t,:count[n,t]].tolist())==expected[n][t]
    heads,nxt,prev,bucket,due=[x.cpu() for x in owners]
    # Walk each intrusive list: no cycles, duplicates, or stale back-links.
    for n in range(len(expected)):
        seen=set()
        for b,head in enumerate(heads[n].tolist()):
            before=-1
            while head>=0:
                assert head not in seen
                seen.add(head)
                assert bucket[n,head]==b and prev[n,head]==before
                before,head=head,nxt[n,head].item()
        assert seen==set(torch.nonzero(bucket[n]>=0).flatten().tolist())


def test_mutation_journal_exact_undo():
    import triton
    import triton.language as tl
    from experiments.expressivity_tasks.x35_tape_cell_v3.timer_wheel import insert, unlink, cascade
    from experiments.expressivity_tasks.x31_tape_cell_fused.all_field_journal import _undo

    @triton.jit
    def transaction(Heads,Next,Prev,Bucket,Due,A,B,W,Count,Capacity:tl.constexpr):
        J=(A,B,W,Count,0,Capacity)
        clock=tl.full((),250,tl.int64)
        insert(Heads,Next,Prev,Bucket,Due,0,258,clock,J,True)
        insert(Heads,Next,Prev,Bucket,Due,1,258,clock,J,True)
        insert(Heads,Next,Prev,Bucket,Due,0,257,clock,J,True)
        cascade(Heads,Next,Prev,Bucket,Due,tl.full((),256,tl.int64),J,True)
        unlink(Heads,Next,Prev,Bucket,1,J,True)

    dev=device()
    owners=[torch.full((size,),-1,device=dev,dtype=dtype) for size,dtype in
            [(2048,torch.int32),(3,torch.int32),(3,torch.int32),(3,torch.int32),(3,torch.int64)]]
    before=[x.clone() for x in owners]
    cap=512
    addresses=torch.empty(1,cap,device=dev,dtype=torch.uint64)
    bits=torch.empty_like(addresses)
    widths=torch.empty(1,cap,device=dev,dtype=torch.int32)
    count=torch.zeros(1,device=dev,dtype=torch.int32)
    transaction[(1,)](*owners,addresses,bits,widths,count,cap,debug=True,num_warps=1)
    assert count.item()>0
    _undo[(1,)](addresses,bits,widths,count,cap,32,debug=True,num_warps=1)
    for actual,expected in zip(owners,before):
        assert torch.equal(actual,expected)
