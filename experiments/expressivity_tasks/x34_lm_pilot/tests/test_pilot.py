import torch
from experiments.expressivity_tasks.x34_lm_pilot.model import LM, parameter_receipt, rope
from experiments.expressivity_tasks.x34_lm_pilot.probes import bin_name, context_probe, depth_probe


def test_parameters_and_frozen_contract():
    receipt=parameter_receipt()
    assert 15_000_000 <= receipt['tape_lm']['total'] <= 40_000_000
    assert receipt['relative_total_difference']<.1
    tape=LM('tape_lm',259)
    cell=tape.blocks[2].cell
    assert cell.cfg.budget==2944 and cell.cfg.n_heads==4
    assert not cell.W_k.weight.requires_grad
    assert not cell.W_write.weight.requires_grad


def test_baseline_causal_and_trainable():
    torch.manual_seed(12)
    model=LM('base_lm',259)
    a=torch.randint(1,259,(1,12)); b=a.clone(); b[:,8:]=42
    x,_=model(a); y,_=model(b)
    torch.testing.assert_close(x[:,:8],y[:,:8])
    loss=torch.nn.functional.cross_entropy(x[:,:-1].flatten(0,1),a[:,1:].flatten())
    loss.backward()
    assert model.blocks[2].extra[0].weight.grad.abs().sum()>0


def test_rope_norm_and_ladder():
    x=torch.randn(2,4,20,64)
    torch.testing.assert_close(rope(x).square().sum(-1),x.square().sum(-1))
    assert [bin_name(k) for k in (128,256,512,1152)]==['ID','2x','4x','9x']


def test_probe_lengths_and_target_normalization():
    class Tokenizer:
        def encode(self,s):
            return type('Encoding',(),{'ids':[1 if c=='a' else 2 for c in s]})()
        def token_to_id(self,s):
            return 3
    class Uniform(torch.nn.Module):
        def forward(self,ids):
            return torch.zeros(*ids.shape,4),[]
    rows=depth_probe(Uniform(),Tokenizer(),'cpu',depths=[16,256])
    assert all(row['accuracy']==.5 for row in rows)
    assert rows[0]['scores'][16]['target_tokens']==32
    import numpy as np
    ctx=context_probe(Uniform(),np.ones(129,dtype=np.int64),'cpu',lengths=(32,64),windows=2)
    assert all(abs(row['perplexity']-4)<1e-5 for row in ctx)
    assert [len(row['per_position_nll']) for row in ctx]==[32,64]


def test_common_initialization_matches():
    torch.manual_seed(3401)
    a=LM('tape_lm',259)
    torch.manual_seed(3401)
    b=LM('base_lm',259)
    common=set(a.state_dict()) & set(b.state_dict())
    for name in common:
        torch.testing.assert_close(a.state_dict()[name],b.state_dict()[name],rtol=0,atol=0)


def test_data_order_and_next_token_targets(tmp_path,monkeypatch):
    import numpy as np
    from experiments.expressivity_tasks.x34_lm_pilot import data
    monkeypatch.setattr(data,'RUNS',tmp_path)
    np.arange(4097,dtype=np.uint16).tofile(tmp_path/'train.bin')
    x,y=data.batch('train',3,batch_size=4,context=16)
    xx,yy=data.batch('train',3,batch_size=4,context=16)
    assert torch.equal(x,xx) and torch.equal(y,yy)
    assert torch.equal(x+1,y)
    assert len(torch.unique(x[:,0]))==4


def test_peak_liveness_respects_prewrite_pop():
    from types import SimpleNamespace
    from experiments.expressivity_tasks.x34_lm_pilot.model import trace_metrics
    # Allocate slot zero, then pop/free/reuse it; peak must remain ONE.
    trace=SimpleNamespace(alloc_target=torch.tensor([[[0],[0]]]),
        freed=torch.tensor([[[[0,0]],[[1,0]]]]),counts=lambda: {})
    state=SimpleNamespace(native=SimpleNamespace(live=torch.tensor([[[True,False]]])))
    result=trace_metrics(state,trace,torch.zeros(1,2,1))
    assert result['peak_live_per_head']==[1]
    assert result['final_live_per_head']==[1]


def test_story_probe_resets_and_receipts_truncation(tmp_path):
    from experiments.expressivity_tasks.x34_lm_pilot.probes import story_probe
    class Tok:
        def encode(self,s):
            return type('Encoding',(),{'ids':[1]*len(s)})()
        def token_to_id(self,s): return 2
    class Uniform(torch.nn.Module):
        def forward(self,ids): return torch.zeros(*ids.shape,3),[]
    path=tmp_path/'stories.txt'
    path.write_text('abcdef\n<|endoftext|>\nab\n<|endoftext|>\n')
    rows=story_probe(Uniform(),Tok(),path,'cpu',stories=2,cap=4)
    assert len(rows)==2 and rows[0]['truncated'] and not rows[1]['truncated']
    assert rows[0]['target_tokens']==4


def test_depth_bin_aggregation():
    from experiments.expressivity_tasks.x34_lm_pilot.probes import aggregate_depth
    rows=[dict(k=16,bin='ID',accuracy=.5,scores={16:{'mean_logprob':-2.},
          14:{'mean_logprob':-3.},18:{'mean_logprob':-1.}})]
    result=aggregate_depth(rows)['ID']
    assert result['pairs']==2 and result['accuracy']==.5
    assert result['valid_mean_logprob']==result['corrupted_mean_logprob']==-2.
