from collections import Counter
import pytest
import torch
from .k15_phrase_query_run import fixture, opened, scan_sources, second_law


def test_native_query_pair():
    for r in (False,True):
        records=[]
        for cue in (250,251):
            raw=fixture(cue,r); assert len(raw)==128
            with opened(Counter()) as ix:
                for y in raw:
                    ix.observe(y); assert ix.sources()==scan_sources(ix.parse)
                d,ids=ix.sources(); spans=[ix.parse.rows[i] for i in ids]
                assert sorted(raw[s.start:s.end] for s in spans)==[b'Ux12',b'Uy34']; records.append((d,ids,spans))
                ix.observe(85); assert ix.sources()==(0,[])
        assert records[0]==records[1]


def test_delayed_credit_enumeration():
    scores=torch.tensor([.3,-.4],dtype=torch.double,requires_grad=True)
    k=scores.new_tensor([.7,.9]); lit=scores.new_tensor([.2,.8]); which=torch.tensor([1,0]); a=.92
    law=lambda s: second_law(s,a,k,lit,which)
    assert torch.autograd.gradcheck(law,(scores,))
    for y in range(2):
        expect=(1-a)*lit[y]+sum(a*w*((1-p)*lit[y]+p*int(i==y)) for w,p,i in zip(scores.softmax(0),k,which))
        assert torch.allclose(law(scores)[y],expect)
    assert torch.isclose(law(scores).sum(),scores.new_tensor(1.))


def test_cursor_credit_at_offset_one():
    from types import SimpleNamespace
    from .k15_phrase_query import PhraseQuery
    from .k15_committed_phrase_machine import Machine
    torch.manual_seed(7)
    model=SimpleNamespace(write=torch.nn.Linear(8,8),phase=torch.nn.Linear(8,2),
                          operand=torch.nn.Linear(3,1),transition=torch.nn.Embedding(3,8),
                          neural=lambda raw,fb: (fb,fb.new_full((256,),1/256)))
    with opened(Counter()) as ix:
        m=Machine(model,ix,fixture(250,False),PhraseQuery(8)); _,ids=ix.sources()
        m.predict(); m.observe(85); law=m.predict(); reached=[(i,1) for i in ids]
        assert set(m.cursor_logits)==set(reached)
        logits=[m.cursor_logits[key] for key in reached]; assert logits[0]!=logits[1]
        grads=torch.autograd.grad(-law[120].log(),logits)
        for i,g in zip(ids,grads):
            supports_x=ix.parse.raw[ix.parse.rows[i].start+1]==120
            assert g<0 if supports_x else g>0


def assert_paired_trace_order(traces):
    assert [row['arm'] for row in traces] == [[250, False], [251, False], [250, True], [251, True]]


@pytest.mark.parametrize('order', [(), (0, 1, 2), (1, 0, 2, 3)], ids=['empty', 'missing-last', 'reordered'])
def test_archived_paired_address_fit_rejects_bad_trace_order(order):
    traces = [{'arm': [cue, rebound]} for rebound in (False, True) for cue in (250, 251)]
    with pytest.raises(AssertionError):
        assert_paired_trace_order([traces[i] for i in order])


def test_archived_paired_address_fit():
    import json
    from pathlib import Path
    from .k15_phrase_query import PhraseQuery
    root=Path(__file__).parent/'receipts/k15_phrase_query'
    receipt=json.loads((root/'witness1.json').read_text())
    state=torch.load(root/'controller.pt',map_location='cpu',weights_only=True)
    head=PhraseQuery(len(state['center'])); head.load_state_dict(state,strict=True)
    assert_paired_trace_order(receipt['traces'])
    for case,trace in enumerate(receipt['traces']):
        a={r['arm']:r for r in receipt['after'] if r['case']==case}; fit=a['fitted']
        assert fit['margin']>0 and fit['top']==fit['good'] and a['wrong_query']['margin']<0
        assert all(fit['nll']<a[k]['nll'] for k in ('uniform','compiled','wrong_query'))
        assert abs(trace['live_nll']-fit['nll'])<1e-5 and trace['cache_error']<1e-5
