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
    expected=[{'amp'},{'segmented'},{'amp'},{'loss'},{'padded'},{'train'},{'checkpoint'},{'mlp'}]
    assert len(PROFILES)==9
    for (left,right),keys in zip(PAIRS,expected):
        assert {k for k in profiles[left] if k!='name' and profiles[left][k]!=profiles[right][k]}==keys


def test_original_invalid_training_profile_is_rejected_on_cpu():
    from scripts.diagnose_e97_activation_alignment import execution_shapes
    item=dict(prefix=[1]*73,generated=[2]*19)
    old={**next(p for p in PROFILES if p['name']=='full-eval'),'train':True}
    with pytest.raises(ValueError,match='unaligned training profile'):execution_shapes(item,old)
    for p in PROFILES:
        lengths=execution_shapes(item,p)
        if p['train']:assert all(n%16==0 for n in lengths)


@pytest.mark.parametrize('prefix,generated',[(1,1),(15,1),(16,1),(127,7),(128,1),(129,512)])
def test_preflight_matches_real_loss_input_lengths(prefix,generated):
    from scripts.diagnose_e97_activation_alignment import execution_shapes
    from scripts.qualify_e97_native_rl_logprobs import turn_layout
    item=dict(prefix=[1]*prefix,generated=[2]*generated)
    for p in PROFILES:
        lengths=execution_shapes(item,p)
        if p['loss']:
            tokens,*_=turn_layout(item['prefix'],item['generated'],'cpu',128 if p['padded'] else 1)
            assert lengths==[tokens.shape[1]-1]
        if p['train']:assert all(n%16==0 for n in lengths)


def test_completed_profile_receipt_survives_later_failure(tmp_path):
    import json
    from scripts.diagnose_e97_activation_alignment import publish_profile,sha
    item=dict(id='case',turn=0);row=dict(profile=dict(name='actor-cache'),endpoint_reference_max_delta=0.)
    name,digest=publish_profile(tmp_path,item,row,'a'*64)
    with pytest.raises(RuntimeError):raise RuntimeError('later profile failed')
    path=tmp_path/name
    assert sha(path)==digest and json.loads(path.read_text())['measurement']==row
    assert path.stat().st_mode & 0o777==0o400
    assert publish_profile(tmp_path,item,row,'a'*64)==(name,digest)
    with pytest.raises(ValueError,match='conflicts'):
        publish_profile(tmp_path,item,{**row,'endpoint_reference_max_delta':1.},'a'*64)
    assert sha(path)==digest


def test_production_unaligned_training_guard_is_not_bypassed():
    from ndm.triton.e88_triton_optimized import e88_triton_optimized_apply
    k=torch.zeros((1,17,1,2),dtype=torch.bfloat16);decay=torch.zeros((1,17,1),dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match='unaligned recurrence padding is forward-only'):
        e88_triton_optimized_apply(True,k,k,k,decay,n_heads=1)
