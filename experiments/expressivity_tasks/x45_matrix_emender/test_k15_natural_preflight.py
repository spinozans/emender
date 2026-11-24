from collections import Counter
import io
import torch
from torch import nn
from .k15_natural_preflight import NaturalMachine, PhraseQuery, forks, opened


def test_exhaustion_and_censor_not_disagreement():
    assert forks([b'a',b'ab'],b'ab')==([],[])
    assert forks([b'ac',b'ab'],b'a')==([],[])
    assert forks([b'ac',b'ab'],b'ad')==([(1,1,0)],[])


class Small(nn.Module):
    def __init__(self):
        super().__init__(); self.write=nn.Linear(8,8); self.phase=nn.Linear(8,2); self.operand=nn.Linear(3,1); self.transition=nn.Embedding(3,8); self.head=nn.Linear(8,256)

    def neural(self,raw,feedback):
        h=feedback+raw[-1]/256
        return h,self.head(h).softmax(0)


def test_prefix_replay_and_masked_suffix():
    torch.manual_seed(42); m=Small(); head=PhraseQuery(8); raw=b'abcd'*34
    buf=io.BytesIO(); torch.save((m.state_dict(),head.state_dict(),raw[:131]),buf)
    def predict(prefix,suffix):
        with opened(Counter(),1000) as ix:
            state=NaturalMachine(m,ix,prefix[:128],head,[0.]*7)
            assert not any(v.any() for v in state.values)
            for y in prefix[128:]: state.predict(); state.observe(y)
            law=state.predict().detach().clone(); assert law.min()>0 and abs(law.sum()-1)<1e-6
            state.observe(suffix[0]); return law
    live=predict(raw[:131],raw[131:]); m=Small(); head=PhraseQuery(8)
    buf.seek(0); a,b,prefix=torch.load(buf,weights_only=True); m.load_state_dict(a); head.load_state_dict(b)
    assert torch.equal(live,predict(prefix,b'!!!!'))


def test_training_objective_scores_all_eight_then_requires_reset():
    import pytest
    from .k15_natural_preflight import all_byte_window_loss
    m=Small(); head=PhraseQuery(8); raw=bytes(range(136)); weights=[0.]*7
    with opened(Counter(),1000) as ix:
        actual=all_byte_window_loss(m,head,weights,ix,raw)
        with pytest.raises(ValueError): all_byte_window_loss(m,head,weights,ix,raw)
    with opened(Counter(),1000) as ix:
        state=NaturalMachine(m,ix,raw[:128],head,weights); total=0.
        for y in raw[128:]:
            law=state.predict()
            total-=law[y].log().item(); state.observe(y)
    assert abs(actual.item()-total/8)<1e-6


def test_each_original_cursor_through_adjudication():
    import pytest
    from .k15_natural_preflight import joint_credit_passed, reached_keep_logits

    for shared in (b'ab', b'abc'):
        contents = {4:shared+b'x', 5:shared+b'y', 6:b'az', 7:b'a'}
        norms = {'value:4':1.}
        for offset in range(1,len(shared)+1):
            expected = [4,5,6] if offset==1 else [4,5]
            logits = {(row,offset): torch.tensor(.2,requires_grad=True) for row in expected}
            reached = reached_keep_logits(contents, shared[:offset], logits)
            assert list(reached)==[f'keep:{row}:{offset}' for row in expected]
            assert all(reached[f'keep:{row}:{offset}'] is logits[row,offset] for row in expected)
            norms.update({name:1. for name in reached})
            del logits[4,offset]
            with pytest.raises(KeyError): reached_keep_logits(contents,shared[:offset],logits)
        competition = ([4,5,6,7],[-.3,.2,.1,0.],{4:True,5:False})
        assert joint_credit_passed(norms,*competition)
        for bad in (0.,None):
            assert not joint_credit_passed(dict(norms,**{'keep:4:1':bad}),*competition)


def test_post_backward_rss_stops_both_slice_types(monkeypatch):
    from . import k15_natural_preflight as prep
    from .k15_phrase_query_run import fixture
    raw = fixture(250,False)+b'Ux'; params=(Small(),PhraseQuery(8),[0.]*7)
    for control in (True,False):
        for excess in (0,1):
            peak = prep.RSS_CAP_KIB+excess
            readings = iter([1,1,peak]); monkeypatch.setattr(prep,'rss',lambda: next(readings))
            result = prep.run_slice(Counter(),params,raw,128,1,control)
            assert result['rss_kib']==peak and result['backward']==1
            if excess:
                assert result['stop']=='post-backward RSS cap' and result['passed'] is False
            else:
                assert 'stop' not in result
