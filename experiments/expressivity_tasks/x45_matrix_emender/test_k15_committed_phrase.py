from collections import Counter
from itertools import product
import pytest
import torch
from .k15_committed_phrase import Factorizer, offline, scan_sources, targets
from .k15_committed_phrase_machine import conditional_loss
from .k15_committed_phrase_run import opened


def test_prefix_and_confirmed_end():
    traces={}
    for n in range(9):
        for chars in product((0,255),repeat=n):
            s=bytes(chars); f=Factorizer()
            for t,y in enumerate(s,1):
                f.observe(y); trace=list(f.rows),f.start,list(f.matches)
                assert traces.setdefault(s[:t],trace)==trace
            f.end(); assert [(r.start,r.end,r.source) for r in f.rows]==list(offline(s))
            for r in f.rows: assert f.types[s[r.start:r.end]]==r.kind
            with pytest.raises(ValueError): f.observe(30)


def test_native_prefix_seam():
    with opened(Counter()) as ix:
        for y in (b'\0\xff\x1e'*96):
            ix.observe(y); assert ix.sources()==scan_sources(ix.parse)
        ix.end(); assert ix.sources()==(0,[])


def test_prefilter_overflow():
    with pytest.raises(RuntimeError), opened(Counter()) as ix:
        ix.ids.rpc('append '+'00'*12)
        ix.ids.rpc('locate 00000000 0 12 3')['positions']


def test_set_credit_and_censor():
    lab=targets([b'abc',b'abc',b'ax'],b'ab'); assert lab==dict(start=1,best=[0,1],lengths=[2,2,1],keep=[[1],[1],[0]])
    x=torch.zeros(5,dtype=torch.double,requires_grad=True)
    def loss(x):return conditional_loss(x[0],x[1:4],x[4:],lab)
    assert torch.autograd.gradcheck(loss,(x,))
    g,=torch.autograd.grad(loss(x),x); assert g[0]<0 and g[1]==g[2]<0 and g[3]>0 and g[4]<0
    assert targets([b'abc'],b'a')['keep']==[[]]


def test_delayed_keep_first_update():
    labels = targets([b'abcx', b'abcy', b'abcy'], b'abcy')
    assert labels['keep']==[[1,1,0],[1,1,1],[1,1,1]]
    assert labels['best']==[1,2]
    labels = targets([b'abcx'], b'abcy')
    keep = torch.zeros(3,requires_grad=True)
    loss = conditional_loss(torch.tensor(0.),torch.zeros(1),keep,labels)
    gradient, = torch.autograd.grad(loss,keep)
    assert gradient[0]==gradient[1]<0 and gradient[2]>0
    start = torch.tensor(0.,requires_grad=True)
    miss = conditional_loss(start,torch.zeros(1),keep,targets([b'a'],b'z'))
    assert torch.autograd.grad(miss,start)[0]>0


def test_exhaustive_analytical_control_gradient():
    """Tiny independent probability-space reference; not a K15 surrogate run."""
    sources = (b'\x00\x01\x01', b'\x00\x01\x00', b'\x00\x01\x00')
    for n in range(4):
        for bits in product(range(2),repeat=n):
            future = bytes(bits)
            x = torch.linspace(-.2,.4,6,dtype=torch.double,requires_grad=True)
            loss = conditional_loss(x[0],x[1:4],x[4:],targets(sources,future))
            p = x[0].sigmoid(); weights=x[1:4].softmax(0); keep=x[4:].sigmoid()
            ref=x.sum()*0
            if bits:
                good=list(range(3)); decisions=[]
                for t,y in enumerate(bits):
                    surviving=[i for i in good if sources[i][t]==y]
                    if t:
                        decisions += [keep[t-1] if i in surviving else 1-keep[t-1] for i in good]
                    if not surviving:
                        if not t: good=[]
                        break
                    good=surviving
                ref=ref-(p*weights[good].sum() if good else 1-p).log()
                if decisions: ref=ref-torch.stack(decisions).log().mean()
            assert torch.allclose(loss,ref)
            assert torch.allclose(torch.autograd.grad(loss,x)[0],torch.autograd.grad(ref,x)[0],atol=1e-12)


def test_end_publishes_open_prefill_row():
    from types import SimpleNamespace
    from .k15_committed_phrase_machine import Machine
    with opened(Counter()) as ix:
        m=Machine(SimpleNamespace(write=torch.nn.Linear(4,4)),ix,b'z'*128)
        before=len(m.values); assert ix.parse.start<128
        m.end()
        assert len(m.values)==len(ix.parse.rows)==before+1 and ix.parse.start==128
        with pytest.raises(AssertionError): m.predict()


@pytest.mark.parametrize('continuing', [False, True], ids=['start','keep-and-start'])
def test_saturated_phase_preserves_literal_escape(continuing):
    from types import SimpleNamespace
    from .k15_committed_phrase_machine import Machine
    # Exercise the production transitions, not a K15 backbone measurement.
    model = SimpleNamespace(write=torch.nn.Linear(4,4), transition=torch.nn.Embedding(3,4),
                            operand=torch.nn.Linear(3,1), phase=lambda h: h.new_full((2,),20.),
                            neural=lambda raw,feedback: (feedback, feedback.new_full((256,),1/256)))
    with opened(Counter()) as ix:
        machine = Machine(model,ix,b'ABCD'*32)
        _, ids = ix.sources()
        assert ids
        if continuing:
            i = ids[0]; assert ix.parse.rows[i].end - ix.parse.rows[i].start == 4
            machine.post = {(i,1):torch.ones(())}
        p = machine.predict()
        assert p.dtype == torch.float32 and torch.all(p>0) and torch.isclose(p.sum(),torch.tensor(1.))
        machine.observe(255)  # absent from every copy edge
        assert machine.post.keys()=={None} and torch.isclose(machine.post[None],torch.tensor(1.))
        assert torch.isfinite(machine.feedback).all()


def test_index_rejects_after_end_before_rpc():
    stats = Counter()
    with opened(stats) as ix:
        ix.observe(65); ix.end(); before=stats.copy()
        with pytest.raises(ValueError,match='confirmed end'): ix.observe(66)
        assert stats==before and ix.parse.raw==b'A' and ix.sources()==(0,[])


def test_witness_observation_and_gradient_controls():
    import json
    from pathlib import Path
    from .k15_committed_phrase_run import shared_witness_byte, check_witness_gradients
    receipt = Path(__file__).parent/'receipts/k15_committed_phrase/witness2.json'
    older, alternative, cleared = json.loads(receipt.read_text())['result']['results']
    a,b = (bytes.fromhex(r['type_hex']) for r in (older,alternative))
    assert shared_witness_byte(a,b)==ord('A')
    check_witness_gradients(older['gradient_norm'],cleared['gradient_norm'])
    with pytest.raises(AssertionError): shared_witness_byte(a,b'XBCD')
    with pytest.raises(AssertionError): check_witness_gradients(older['gradient_norm'],1e-12)
