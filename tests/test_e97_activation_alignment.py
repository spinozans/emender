import pytest
import torch
from scripts.diagnose_e97_activation_alignment import Taps,compare_banks,PROFILES,PAIRS


class Mixer(torch.nn.Module):
    def forward(self,x):return x*.5,None


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mixer=Mixer();self.mlp=torch.nn.Linear(3,3,bias=False)
        with torch.no_grad():self.mlp.weight.copy_(torch.eye(3)*.25)
    def forward(self,x):
        mixed,_=self.mixer(x);return mixed+self.mlp(x),None


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(16,3);self.layers=torch.nn.ModuleList([Block(),Block()])
        self.lm_head=torch.nn.Linear(3,16,bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(48).reshape(16,3)*.125)
            self.lm_head.weight.fill_(.25)
    def forward(self,tokens,chunk=100):
        h=self.embedding(tokens)
        for layer in self.layers:h,_=layer(h)
        return torch.cat([self.lm_head(h[:,i:i+chunk]) for i in range(0,h.shape[1],chunk)],1)


def test_prediction_alignment_across_prefill_tokens_and_chunked_head():
    model=Toy();tokens=torch.tensor([[1,2,3,4]])
    with torch.no_grad():expected=model(tokens)
    actor=Taps(model)
    with torch.no_grad():
        actor.begin([2],[4]);model(tokens[:,:3]);actor.begin([0],[5]);model(tokens[:,3:])
        actor.begin([],[]);model(torch.tensor([[5]]))
    a=actor.finish(2);actor.close()
    full=Taps(model);full.begin([2,3],[4,5])
    with torch.no_grad():actual=model(tokens,chunk=1)
    b=full.finish(2);full.close()
    assert torch.equal(actual,expected)
    assert len(a)==12 and all(torch.equal(a[k],b[k]) for k in a)
    assert actor.logprobs==full.logprobs
    assert all(row['differing_elements']==0 for row in compare_banks(a,b))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_trace_comparison_locates_changed_stage_and_records_dtype():
    a={'first':torch.ones(2,3),'second':torch.ones(2,3)}
    b={'first':a['first'].to(torch.bfloat16),'second':a['second']+.125}
    rows=compare_banks(a,b)
    assert rows[0]['dtype_changed'] and rows[0]['absolute_max']==0
    assert rows[1]['absolute_max']==.125 and rows[1]['differing_elements']==6
    with pytest.raises(ValueError):compare_banks(a,{'first':torch.ones(2,3)})
    with pytest.raises(ValueError):compare_banks(a,{**b,'second':torch.ones(3,3)})
    with pytest.raises(ValueError):compare_banks(a,{**b,'second':torch.full((2,3),float('nan'))})


@pytest.mark.parametrize('positions,targets',[([0,0],[1,2]),([1,0],[1,2]),([-1],[1]),([0],[])])
def test_bad_row_mapping_fails_closed(positions,targets):
    tap=Taps(Toy())
    try:
        with pytest.raises(ValueError):tap.begin(positions,targets)
    finally:tap.close()


def test_missing_rows_and_memory_bound_fail_closed():
    tap=Taps(Toy())
    try:
        with pytest.raises(ValueError):tap.finish(1)
        with pytest.raises(ValueError):tap.store('x',torch.ones(1,1,3841),[0])
        tap.bytes=2*1024**3//4
        with pytest.raises(ValueError,match='storage'):tap.store('x',torch.ones(1,1,3),[0])
    finally:tap.close()


def test_predeclared_pairs_isolate_named_factors():
    profiles={p['name']:p for p in PROFILES}
    expected=[{'amp'},{'segmented'},{'train'},{'amp'},{'loss'},{'padded'},{'checkpoint'},{'mlp'}]
    assert len(PROFILES)==9
    for (left,right),keys in zip(PAIRS,expected):
        assert {k for k in profiles[left] if k!='name' and profiles[left][k]!=profiles[right][k]}==keys
