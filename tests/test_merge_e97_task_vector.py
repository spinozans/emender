import json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.merge_e97_task_vector import (MERGE_SCHEMA,aggregate_stats,build_merged_checkpoint,
 merged_state_dicts,sha)


def fake_checkpoint(tying=True):
 embed=torch.randn(8,4).to(torch.bfloat16)
 sd={'embedding.weight':embed}
 if tying:sd['lm_head.weight']=embed
 else:sd['lm_head.weight']=torch.randn(8,4).to(torch.bfloat16)
 sd['block.w']=torch.randn(6).to(torch.bfloat16)
 sd['norm.weight']=torch.ones(4).to(torch.bfloat16)
 opt={'param_groups':[{'train_mode':False,'k':3,'state_schema':'emender-schedulefree-bf16-sr-candidate-v1','params':[0,1,2]}],
  'state':{0:{'z':sd['embedding.weight'].clone(),'exp_avg_sq':torch.ones(8,4).to(torch.bfloat16)},
           1:{'z':sd['block.w'].clone(),'exp_avg_sq':torch.ones(6).to(torch.bfloat16)},
           2:{'z':sd['norm.weight'].clone(),'exp_avg_sq':torch.ones(4).to(torch.bfloat16)}},
  'eval_live_y':{0:sd['embedding.weight'].clone(),1:sd['block.w'].clone(),2:sd['norm.weight'].clone()},
  'precision_identity':{'schema':'emender-schedulefree-bf16-sr-candidate-v1','seed':7},
  'rounding_counters':{'merge':3}}
 ck={'model_state_dict':sd,'optimizer_state_dict':opt,'loss':0.5,'step':3}
 return ck


def test_merge_tying_and_stats(tmp_path):
 parent=fake_checkpoint();candidate=fake_checkpoint()
 candidate['model_state_dict']['block.w']=torch.full((6,),2.0,dtype=torch.bfloat16)
 etas=[0.0,0.25,1.0]
 merged,stats=merged_state_dicts(parent['model_state_dict'],candidate['model_state_dict'],etas)
 assert merged[0.0]['block.w'].equal(parent['model_state_dict']['block.w'])
 assert merged[1.0]['block.w'].equal(candidate['model_state_dict']['block.w'])
 assert merged[0.25]['embedding.weight'].data_ptr()==merged[0.25]['lm_head.weight'].data_ptr()
 assert merged[0.25]['embedding.weight'].equal(merged[0.25]['lm_head.weight'])
 row=aggregate_stats(stats[0.25])
 assert row['tensor_groups']==3 and row['coordinates']==8*4+6+4
 assert row['unchanged_vs_parent_fraction']<1.0
 # fp32 arithmetic, single rounding: expected block value
 expected=(parent['model_state_dict']['block.w'].float()+0.25*(candidate['model_state_dict']['block.w'].float()-parent['model_state_dict']['block.w'].float())).to(torch.bfloat16)
 assert merged[0.25]['block.w'].equal(expected)


def test_built_checkpoint_roundtrip(tmp_path):
 parent=fake_checkpoint();candidate=fake_checkpoint()
 candidate['model_state_dict']['block.w']=torch.full((6,),2.0,dtype=torch.bfloat16)
 merged,stats=merged_state_dicts(parent['model_state_dict'],candidate['model_state_dict'],[0.25])
 mapping={0:'embedding.weight',1:'block.w',2:'norm.weight'}
 ck=build_merged_checkpoint(candidate,merged[0.25],0.25,parent_path=Path('/p.pt'),parent_sha='p'*64,
  candidate_path=Path('/c.pt'),candidate_sha='c'*64,pid_names=mapping)
 assert ck['loss'] is None
 assert ck['sft_task_merge']['schema']==MERGE_SCHEMA
 assert ck['sft_task_merge']['training_authority_inapplicable'] is True
 opt=ck['optimizer_state_dict']
 for key,name in mapping.items():
  assert torch.equal(opt['state'][key]['z'],ck['model_state_dict'][name])
  assert torch.equal(opt['eval_live_y'][key],ck['model_state_dict'][name])
  assert opt['state'][key]['z'].data_ptr()!=ck['model_state_dict'][name].data_ptr()
 path=tmp_path/'merged.pt';torch.save(ck,path)
 reloaded=torch.load(path,map_location='cpu',weights_only=False)
 rsd=reloaded['model_state_dict']
 assert rsd['embedding.weight'].data_ptr()==rsd['lm_head.weight'].data_ptr()
 for key,name in mapping.items():
  assert torch.equal(reloaded['optimizer_state_dict']['state'][key]['z'],rsd[name])
  assert torch.equal(reloaded['optimizer_state_dict']['eval_live_y'][key],rsd[name])
 assert json.loads(json.dumps(reloaded['sft_task_merge']))['eta']==0.25


def test_rejects_tying_mismatch():
 parent=fake_checkpoint();candidate=fake_checkpoint()
 candidate['model_state_dict']['lm_head.weight']=torch.randn(8,4).to(torch.bfloat16)
 try:
  merged_state_dicts(parent['model_state_dict'],candidate['model_state_dict'],[0.5])
 except ValueError:pass
 else:raise AssertionError('tying mismatch not detected')
