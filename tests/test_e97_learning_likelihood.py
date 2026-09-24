import hashlib
import json
import pytest
import torch
from scripts.probe_e97_learning_likelihood import target_logprob,sequence_logprobs
from scripts.aggregate_e97_learning_likelihood import aggregate


def test_logprob_and_teacher_forced_sequence():
    from types import SimpleNamespace
    class Engine:
        def advance(self,ids,cache):
            assert ids==[1]
            return SimpleNamespace(next_logits=torch.tensor([2.,1.]))
    first=SimpleNamespace(next_logits=torch.tensor([1.,2.]))
    observed=sequence_logprobs(Engine(),first,[1,0])
    assert observed==pytest.approx([-0.31326166,-0.31326166])


@pytest.mark.parametrize('logits',[torch.tensor([float('nan'),0.]),torch.zeros(1,2)])
def test_invalid_logits_rejected(logits):
    with pytest.raises(RuntimeError): target_logprob(logits,0)


def fixture(tmp_path):
    probes={'examples':[{'id':'a'},{'id':'b'}],'marker_candidates':['Analysis: ']}
    p=tmp_path/'probes.json'; p.write_text(json.dumps(probes))
    for rank,id in enumerate(['a','b']):
        r={'id':id,'split':'train','source':'agent','target_tokens':2,
           'full_prefix_serving_top1_equal':True,'markers':{'Analysis: ':{'logprob_sum':-1.}},
           'serving_target_nll_mean':0.5,'boundary_forward_target_nll_mean':0.6,'serving_first_target_rank':1}
        report={'schema':'emender-e97-learning-likelihood-shard-v1','checkpoint_sha256':'a'*64,
                'weight_mode':'train','probes_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'probe_source_sha256':'b'*64,'world_size':2,'rank':rank,'device':f'cuda:{rank}','results':[r]}
        (tmp_path/f'rank-{rank:02d}.json').write_text(json.dumps(report))
    return p


def test_complete_aggregate(tmp_path):
    p=fixture(tmp_path); result=aggregate(tmp_path,p,2)
    assert result['examples']==2
    assert result['by_source']['train/agent']['serving_target_token_weighted_nll']==0.5


@pytest.mark.parametrize('failure',['missing','device','duplicate'])
def test_aggregate_rejects_incomplete_or_misrouted_evidence(tmp_path,failure):
    p=fixture(tmp_path); shard=tmp_path/'rank-01.json'; d=json.loads(shard.read_text())
    if failure=='missing': shard.unlink()
    else:
        if failure=='device': d['device']='cuda:0'
        else: d['results'][0]['id']='a'
        shard.write_text(json.dumps(d))
    with pytest.raises(ValueError): aggregate(tmp_path,p,2)
