import json
import pytest
import torch
from .b1_head import XSAResidual, pack, transfer
from .b1_mount import Registry, HERE
from .bank_read import reference_read


def assert_slot_roundtrip(bank, rows):
    transferred = transfer(bank, 'cpu')
    assert isinstance(bank.slot_spans, tuple)
    assert transferred.slot_spans is bank.slot_spans
    page = transferred.checked()[2]
    assert len(bank.slot_spans) == page.records
    assert [[transferred.slot_spans[i] for i in row if i != -1] for row in page.ids.tolist()] == [
        [span for span, _ in row] for row in rows
    ]


def test_scope_before_open():
    r=Registry({'P100'})
    for name,kw in [('F6',{}),('P100',{'excluded_docs':[0]}),('P100',{'whole_bank':False})]:
        with pytest.raises(PermissionError): r.resolve(name,[[b'a']],**kw)
    assert not r.mounts


def test_session_causal_oracle():
    r=Registry({'context'}); s=r.open('context'); docs=[b'\0\x1e\xff', b'\xff\0\x1e\xff']; past=[]
    try:
        for d,doc in enumerate(docs):
            if d: s.end()
            past.append(b'')
            for x in doc:
                for p in (b'\0\x1e',b'\xff\xff',bytes([x])):
                    hits=r.resolve('context',[[p]])[0]
                    expected=[(i,j) for i,raw in enumerate(past) for j in range(len(raw)) if raw.startswith(p,j)]
                    assert [(a.doc,a.start) for a,_ in hits]==expected
                    assert all(v==(p[-1],)*8 for _,v in hits)
                s.observe(bytes([x]),[(x,)*8]); past[-1]+=bytes([x])
        rows=r.resolve('context',[[b'\xff',b'\xff'],[b'xyz']]); old=s.version; working=pack(rows,s,'context')
        assert_slot_roundtrip(working,rows)
        s.seal()
        with pytest.raises(ValueError): working.checked()
        assert old!=s.version
        with pytest.raises(ValueError): pack(rows,s,'context')
        k,v,p,b=pack(r.resolve('context',[[b'\xff',b'\xff'],[b'xyz']]),s,'context').checked()
        assert p.records==3 and p.ids[0].tolist()==[0,1,2,0,1,2] and (p.ids[1]==-1).all()
    finally: r.close()


def test_native_to_reference_head():
    r=Registry({'F6','P100','context'}); model=XSAResidual(); banks={}; means={}
    try:
        for name in ('F6','P100','context'):
            m=r.open(name)
            if name=='context':
                m.observe(b'ABA',[(1,)*8,(2,)*8,(3,)*8]); pats=[b'A',b'B']
            else: pats=m.patterns[:2]
            if name == 'P100':
                by_key = r.resolve(name, [[p] for p in pats])
                receipts = HERE.parent/'x43_fork_resolved/receipts/p100_query_verification'
                counts = json.loads((receipts/'scan.json').read_text())['counts']
                registered = json.loads((receipts/'registration.json').read_text())['patterns']
                for entry, hits in zip((2, 6), by_key, strict=True):
                    assert len(hits) == sum(counts[entry])
                    assert registered[entry]['offset'] in {m.starts[a.doc]+a.start for a, _ in hits}
                rows = [by_key[0] + by_key[1]]
            else:
                rows = r.resolve(name,[pats])
            bank=pack(rows,m,name); banks[name]=bank
            assert_slot_roundtrip(bank,rows)
            if name=='P100':
                with pytest.raises(ValueError): pack(rows,r.mounts['F6'],'F6')
            if name=='F6':
                raw=m.text.read_bytes(); expected=[]
                for p in pats:
                    hits=[]; start=0
                    while (at:=raw.find(p,start))>=0:
                        d,j=divmod(at,15626)
                        if j+len(p)<=15626: hits.append((d,j))
                        start=at+1
                    expected+=hits
                assert [(a.doc,a.start) for a,_ in rows[0]]==expected
                with pytest.raises(OverflowError): r.resolve(name,[[m.patterns[2]]])
            means[name]=torch.tensor([v for _,v in rows[0]],dtype=torch.float32).mean(0)
        for names in [('context',),('F6',),('P100',),('context','F6','P100'),('context','P100'),('P100','F6','context')]:
            h=torch.ones(1,8); reads=[(1,banks[n]) for n in names]; y=model(h,reads,read=reference_read)
            torch.testing.assert_close(y,h+sum(means[n] for n in names)); torch.testing.assert_close(model(h,reads+[(0,banks['F6'])],read=reference_read),y,rtol=0,atol=0)
        assert not torch.equal(means['F6'],means['P100'])
        # Fixed host decoder on the returned recurrent state, not a learned search compiler.
        for scale,expected in [(0,b'A'),(1,b'B')]:
            h=model(torch.zeros(1,8),[(scale,banks['context'])],read=reference_read)
            key=(b'A',b'B')[int(h[0,0]>1)]; assert key==expected
            assert r.resolve('context',[[key]])[0]
    finally: r.close()
