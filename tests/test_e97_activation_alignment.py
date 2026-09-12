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


def test_observer_control_restores_absent_and_present_attributes():
    from scripts.diagnose_e97_observer_effect import attributes,configure,restore
    model=Toy();model.eval();model.loss_chunk_size=7
    initial=attributes(model);configure(model)
    assert model.loss_chunk_size==128 and model.gradient_checkpoint_group_size==3
    restore(model,initial)
    assert attributes(model)==initial and not hasattr(model,'gradient_checkpoint_group_size')


def _toy_cache_primitives():
    from types import SimpleNamespace
    def segment(loaded,ids):return SimpleNamespace(next_logits=loaded.model(torch.tensor([ids]))[0,-1])
    def step(loaded,ids,cache):return segment(loaded,ids)
    return segment,step


def test_native_cache_measurement_agrees_with_nonmutating_toy_observers():
    from types import SimpleNamespace
    from scripts.diagnose_e97_observer_effect import attributes,configure,restore,collect,MODES
    model=Toy();model.eval()
    with torch.no_grad():model.lm_head.weight.copy_(torch.arange(48).reshape(16,3)*.03125)
    loaded=SimpleNamespace(model=model);initial=attributes(model);item=dict(prefix=[1,2,3],generated=[4,5])
    segment,step=_toy_cache_primitives();reference=None
    for mode in MODES:
        restore(model,initial)
        if mode!='native':configure(model)
        result=collect(loaded,item,mode,segment=segment,step=step)
        if reference is None:reference=result['native_logprobs']
        assert result['native_logprobs']==pytest.approx(reference,abs=1e-6)
        assert result['observer_vs_native_max'] in (None,0.)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_corrupt_observer_measurement_does_not_replace_native_probability(monkeypatch):
    from types import SimpleNamespace
    from scripts.diagnose_e97_observer_effect import collect
    model=Toy();loaded=SimpleNamespace(model=model);item=dict(prefix=[1,2,3],generated=[4,5])
    segment,step=_toy_cache_primitives()
    baseline=collect(loaded,item,'native',segment=segment,step=step)
    original=Taps.head
    def corrupted(self,module,inputs,output):
        count=len(self.logprobs);original(self,module,inputs,output)
        for i in range(count,len(self.logprobs)):self.logprobs[i]+=1
    monkeypatch.setattr(Taps,'head',corrupted)
    measured=collect(loaded,item,'head-observer',segment=segment,step=step)
    assert measured['native_logprobs']==baseline['native_logprobs']
    assert measured['observer_vs_native_max']==pytest.approx(1.)


@pytest.mark.parametrize('mode',['read','allocate','copy-allocate','synchronize'])
def test_step_probes_do_not_modify_bf16_logits(mode):
    from scripts.diagnose_e97_readback_effect import StepProbe
    logits=torch.tensor([1.,2.,3.],dtype=torch.bfloat16);saved=logits.clone();probe=StepProbe(mode)
    probe(logits,1)
    assert torch.equal(logits,saved) and probe.calls==1
    if mode=='read':assert probe.values==[torch.log_softmax(logits.float(),-1)[1].item()]
    else:assert probe.values==[]
    with pytest.raises(ValueError):probe(logits,3)


def test_optional_readback_preserves_toy_trace_and_matches_native(monkeypatch):
    from types import SimpleNamespace
    import ndm.e97
    from scripts.diagnose_e97_activation_alignment import evaluate
    from scripts.diagnose_e97_readback_effect import StepProbe
    segment,step=_toy_cache_primitives()
    monkeypatch.setattr(ndm.e97,'advance_e97_cache_segment',segment);monkeypatch.setattr(ndm.e97,'advance_e97_cache',step)
    model=Toy().to(dtype=torch.bfloat16);loaded=SimpleNamespace(model=model);item=dict(prefix=[1,2,3],generated=[4,5])
    first,lp=evaluate(loaded,item,PROFILES[0])
    probe=StepProbe('read');second,observed=evaluate(loaded,item,PROFILES[0],step_probe=probe)
    assert all(torch.equal(first[k],second[k]) for k in first)
    assert lp==observed==probe.values and probe.calls==2
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
