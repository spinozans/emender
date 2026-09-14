import copy
import pytest
import torch
from scripts.e97_packed_forward_probe import assemble,probes,variants,HeadRows,digest
from scripts.qualify_e97_packed_forward import compare,exact,row_map


def record(n,offset=0):return dict(tokens=torch.arange(n,dtype=torch.long)+offset,mask=torch.ones(n,dtype=torch.bool))


def test_materialization_whole_records_boundaries_and_tail():
    records={0:record(4),1:record(3,10),2:record(4,20)}
    b=assemble(records,[0,1,2],16)
    assert b['starts']=={0:0,1:4,2:7} and b['length']==11
    assert int(b['valid'].sum())==11 and b['reset'].nonzero().flatten().tolist()==[0,4,7]
    assert int(b['mask'].sum())==8 and not b['mask'][3] and not b['mask'][6] and not b['mask'][10:].any()
    c=assemble(records,[2,0,1],16)
    assert int(c['mask'].sum())==8 and torch.equal(c['tokens'][:4],records[2]['tokens'])
    with pytest.raises(ValueError):assemble(records,[0,0],16)
    with pytest.raises(ValueError):assemble(records,[0,1,2],8)


def test_real_pack_geometry_selects_middle_subject_to_frozen_residue_constraints():
    lengths=[2657,4522,1801,341,4678,2787,4662,338,245,4846,2973,2963,4836,2650,409,1612,359,194,191,4674,2521,1272,4526,4836]
    records={i:record(n) for i,n in enumerate(lengths)}
    layouts=variants(records,list(records),[0,12,23])
    assert [layouts[k]['starts'][0] for k in ('original','middle','late')]==[0,31043,58236]


def test_interventions_change_only_declared_prefix_reset_or_invalid_tail():
    records={0:record(70),1:record(80,100),2:record(60,200)}
    v=variants(records,[0,1,2],[0,1,2],256,512);late=v['late'];changed=v['predecessor_changed'];start=late['starts'][0]
    assert torch.equal(late['tokens'][start:],changed['tokens'][start:])
    assert not torch.equal(late['tokens'][:start],changed['tokens'][:start])
    assert all(torch.equal(late[k],changed[k]) for k in ('mask','valid','reset'))
    missing=v['reset_removed'];assert torch.equal(missing['tokens'],changed['tokens'])
    assert (missing['reset']!=changed['reset']).nonzero().flatten().tolist()==[start]
    assert not missing['mask'][start-1]  # no cross-record target even in negative control
    base=v['original'];padding=v['padding_changed'];n=base['length']
    assert torch.equal(base['tokens'][:n],padding['tokens'][:n]) and not torch.equal(base['tokens'][n:],padding['tokens'][n:])
    assert all(torch.equal(base[k],padding[k]) for k in ('mask','valid','reset'))
    assert all(len(x['order'])==3 and x['length']==210 for x in v.values())


def test_probe_positions_are_next_token_aligned_and_supervised_span_intact():
    r=record(70);r['mask'][:20]=False;p=probes(r)
    assert p['boundary']==list(range(8)) and p['assistant']==list(range(19,51))
    assert r['mask'][torch.tensor(p['assistant'])+1].all()
    with pytest.raises(ValueError):probes(record(20))
    b=assemble({4:r},[4],80);assert row_map(b,{4:p},[4])['4.assistant']==p['assistant']


def test_head_observer_preserves_outputs_and_reconstructs_native_ce():
    torch.manual_seed(2);r=record(10);b=assemble({0:r},[0],16)
    hidden=torch.randn(1,16,4).bfloat16();logits=torch.randn(1,16,17);before=logits.clone()
    observer=HeadRows(b,{'span':[0,1,6,7]})
    for lo in range(0,16,4):assert observer.hook(None,(hidden[:,lo:lo+4],),logits[:,lo:lo+4]) is None
    labels=b['tokens'][1:].clone();labels[~b['mask']]=-100
    loss=float(torch.nn.functional.cross_entropy(logits[0],labels,reduction='sum').item())
    result=observer.finish(loss)
    assert torch.equal(logits,before) and result['targets']==9 and result['ce_delta']<1e-5
    assert result['probes']['span']['logits_sha256']==digest(logits[0,[0,1,6,7]])
    assert result['probes']['span']['hidden_sha256']==digest(hidden[0,[0,1,6,7]])
    assert len(result['probes']['span']['logprobs'])==4
    with pytest.raises(ValueError):HeadRows(b,{'missing':[99]}).finish(loss)


def test_comparison_coverage_nonfinite_thresholds_and_exact_signed_zero():
    plan=dict(abs_max=.05,abs_p99=.02)
    assert compare({'x':[1.,2.]},{'x':[1.,2.]},plan)['passed']
    assert not compare({'x':[1.,2.]},{'x':[1.,2.2]},plan)['passed']
    with pytest.raises(ValueError):compare({'x':[1.]},{'y':[1.]},plan)
    with pytest.raises(ValueError):compare({'x':[float('nan')]},{'x':[1.]},plan)
    assert not exact([0.],[-0.]) and exact(dict(a=[1.]),dict(a=[1.]))
