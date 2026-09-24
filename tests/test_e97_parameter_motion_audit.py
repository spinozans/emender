import hashlib
import torch
from scripts.audit_e97_schedulefree_parameter_motion import load, stats, update, finish


def test_alias_mapping_and_dtype_are_validated(tmp_path):
    x=torch.ones(4,dtype=torch.bfloat16)
    c={'model_state_dict':{'weight':x,'tied':x},
       'optimizer_state_dict':{'param_groups':[{'params':[0],'betas':(0.9,0.95),'train_mode':False}],
                               'state':{0:{'z':x.clone()}}}}
    path=tmp_path/'fixture.pt'; torch.save(c,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    _, mapped=load(path,digest)
    assert list(mapped)==['weight']


def test_exact_counts_and_norm():
    baseline=torch.tensor([1,2,3,4],dtype=torch.bfloat16)
    value=torch.tensor([1,2,4,3],dtype=torch.bfloat16)
    result=stats(); update(result,value,baseline)
    result=finish(result)
    assert result['elements']==4 and result['unchanged']==2
    assert result['unchanged_fraction']==0.5 and result['max_abs_delta']==1
    assert result['delta_squared_sum']==2
