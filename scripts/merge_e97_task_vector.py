#!/usr/bin/env python3
"""Deterministic FP32 task-vector merges of existing E97 checkpoints.

Evaluation-only artifact builder: merged = parent + eta*(candidate-parent),
computed in FP32 per tied tensor group and rounded once to BF16. No optimizer
updates and no training. The merged checkpoint sets x, z and the SR live-y
backup to the same merged weights so saved/train modes load identical weights.

The optimizer parameter-id to model-tensor mapping is recovered exactly by
constructing the checkpoint's model on the meta device and reading
named_parameters() registration order, which is the order train.py used when
constructing the optimizer.
"""
import argparse,hashlib,json
from pathlib import Path
import torch

MERGE_SCHEMA='emender-e97-task-vector-merge-v1'
MANIFEST_SCHEMA='emender-e97-task-vector-merge-manifest-v1'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_checkpoint(path):
 ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 if not isinstance(ck,dict) or 'model_state_dict' not in ck:raise ValueError(f'{path} is not a train.py checkpoint')
 return ck

def pid_names_from_groups(opt,sd,args_json,checkpoint_path):
 from ndm.e97 import build_e97_model,_vocab_size,e97_checkpoint_config
 config=e97_checkpoint_config(checkpoint_path,{},args_json)
 with torch.device('meta'):
  model=build_e97_model(config,vocab_size=_vocab_size(config,sd),use_triton=False)
 names=[n for n,_ in model.named_parameters()]
 ids=opt['param_groups'][0]['params']
 if len(ids)!=len(names):raise ValueError('optimizer pid count does not match named parameters')
 mapping=dict(zip(ids,names))
 by_name={n:t for n,t in sd.items()}
 for pid,name in mapping.items():
  if name not in by_name:raise ValueError(f'optimizer pid {pid} maps to unknown tensor {name}')
  if tuple(opt['state'][pid]['z'].shape)!=tuple(by_name[name].shape):
   raise ValueError(f'optimizer pid {pid} shape does not match {name}')
 return mapping

def merged_state_dicts(parent_sd,candidate_sd,etas):
 """For each eta return (merged_sd, stats). Tying is preserved by identity."""
 if set(parent_sd)!=set(candidate_sd):raise ValueError('model state keys differ')
 index={}
 for name,tensor in parent_sd.items():index.setdefault(tensor.data_ptr(),[]).append(name)
 groups=list(index.values())
 merged={eta:{} for eta in etas};stats={eta:[] for eta in etas}
 for names in groups:
  base=parent_sd[names[0]];cand=candidate_sd[names[0]]
  if base.shape!=cand.shape or base.dtype!=torch.bfloat16 or cand.dtype!=torch.bfloat16:
   raise ValueError(f'unsupported tensor group {names}')
  if len(names)>1:
   tied=[candidate_sd[n] for n in names[1:]]
   if any(t.data_ptr()!=cand.data_ptr() for t in tied) or any(not torch.equal(t,cand) for t in tied):
    raise ValueError(f'candidate tying differs from parent for {names}')
  bf=base.detach().float();delta=cand.detach().float()-bf
  for eta in etas:
   out=(bf+float(eta)*delta).to(torch.bfloat16)
   if not bool(torch.isfinite(out.float()).all()):raise ValueError(f'nonfinite merge {names}')
   for n in names:merged[eta][n]=out
   stats[eta].append({'names':sorted(names),'numel':int(out.numel()),
    'unchanged_vs_parent':int((out==base).sum().item()),
    'unchanged_vs_candidate':int((out==cand).sum().item()),
    'rms_delta_vs_parent':float(torch.linalg.vector_norm(out.float()-bf)/max(1.0,float(out.numel())**0.5))})
 return merged,stats

def build_merged_checkpoint(template,merged_sd,eta,*,parent_path,parent_sha,candidate_path,candidate_sha,pid_names):
 ck=dict(template)
 ck['model_state_dict']=dict(merged_sd)
 opt=template['optimizer_state_dict']
 state={};backups={}
 for pid in opt['state']:
  name=pid_names[pid]
  if name not in merged_sd:raise ValueError(f'optimizer pid {pid} maps outside model state')
  state[pid]={'z':merged_sd[name].clone(),'exp_avg_sq':opt['state'][pid]['exp_avg_sq']}
  backups[pid]=merged_sd[name].clone()
 new_opt=dict(opt);new_opt['state']=state;new_opt['eval_live_y']=backups
 ck['optimizer_state_dict']=new_opt
 ck['loss']=None
 ck['sft_task_merge']={'schema':MERGE_SCHEMA,'purpose':'evaluation-only interpolation candidate; not a training product',
  'parent_checkpoint':str(parent_path),'parent_checkpoint_sha256':parent_sha,
  'candidate_checkpoint':str(candidate_path),'candidate_checkpoint_sha256':candidate_sha,
  'eta':float(eta),'arithmetic':'fp32 base + fp32 eta*(candidate-base), single bf16 round-to-nearest-even write per tied group',
  'optimizer_state_policy':'z and SR live-y backups set to merged x; exp_avg_sq, param_groups, precision identity and counters retained from candidate template; inert for evaluation',
  'training_authority_inapplicable':True}
 return ck

def verify_merged(merged_path,parent_path,candidate_path,eta,args_json,*,sample_groups=48):
 """Reload from disk and verify exact construction and mode consistency."""
 m=torch.load(merged_path,map_location='cpu',mmap=True,weights_only=False)
 p=torch.load(parent_path,map_location='cpu',mmap=True,weights_only=False)
 c=torch.load(candidate_path,map_location='cpu',mmap=True,weights_only=False)
 msd=m['model_state_dict'];psd=p['model_state_dict'];csd=c['model_state_dict']
 if set(msd)!=set(psd):raise ValueError('merged model keys differ')
 pid_names=pid_names_from_groups(m['optimizer_state_dict'],msd,args_json,Path(merged_path))
 for pid,entry in m['optimizer_state_dict']['state'].items():
  name=pid_names[pid]
  if not torch.equal(entry['z'],msd[name]):raise ValueError(f'z != merged x for pid {pid}')
  if not torch.equal(m['optimizer_state_dict']['eval_live_y'][pid],msd[name]):raise ValueError(f'live-y != merged x for pid {pid}')
 for name,tensor in msd.items():
  if not bool(torch.isfinite(tensor.float()).all()):raise ValueError(f'nonfinite merged tensor {name}')
 index={}
 for name,tensor in psd.items():index.setdefault(tensor.data_ptr(),[]).append(name)
 groups=[sorted(n) for n in index.values()]
 digest=hashlib.sha256(json.dumps([float(eta),[g[0] for g in groups]],sort_keys=True).encode()).hexdigest()
 rng=torch.Generator().manual_seed(int(digest[:16],16))
 picked=set()
 while len(picked)<min(sample_groups,len(groups)):picked.add(int(torch.randint(0,len(groups),(1,),generator=rng)))
 for gi in sorted(picked):
  names=groups[gi];base=psd[names[0]];cand=csd[names[0]]
  expected=(base.detach().float()+float(eta)*(cand.detach().float()-base.detach().float())).to(torch.bfloat16)
  for n in names:
   if not torch.equal(msd[n],expected):raise ValueError(f'bit-exact merge mismatch {n}')
 return {'tensor_groups':len(groups),'sampled_groups':len(picked),
  'tied_groups':sum(1 for n in groups if len(n)>1)}

def aggregate_stats(stats):
 total=sum(s['numel'] for s in stats)
 return {'tensor_groups':len(stats),'coordinates':total,
  'unchanged_vs_parent_fraction':sum(s['unchanged_vs_parent'] for s in stats)/total,
  'unchanged_vs_candidate_fraction':sum(s['unchanged_vs_candidate'] for s in stats)/total,
  'rms_delta_vs_parent':float(sum(s['rms_delta_vs_parent']**2*s['numel'] for s in stats)/total)**0.5}

def merge_command(a):
 etas=[float(x) for x in a.etas.split(',')]
 parent_path=Path(a.parent).resolve();candidate_path=Path(a.candidate).resolve()
 args_json=Path(a.args_json).resolve()
 parent_sha=sha(parent_path);candidate_sha=sha(candidate_path)
 parent=load_checkpoint(parent_path);candidate=load_checkpoint(candidate_path)
 pid_names=pid_names_from_groups(candidate['optimizer_state_dict'],candidate['model_state_dict'],args_json,candidate_path)
 group=candidate['optimizer_state_dict']['param_groups'][0]
 if group.get('train_mode') is not False:raise ValueError('candidate optimizer must be saved in eval mode')
 if group.get('state_schema')!='emender-schedulefree-bf16-sr-candidate-v1':raise ValueError('candidate optimizer schema mismatch')
 if set(candidate['optimizer_state_dict']['eval_live_y'])!=set(candidate['optimizer_state_dict']['state']):
  raise ValueError('candidate live-y backup coverage mismatch')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 manifest={'schema':MANIFEST_SCHEMA,'status':'merged','parent_checkpoint':str(parent_path),'parent_checkpoint_sha256':parent_sha,
  'candidate_checkpoint':str(candidate_path),'candidate_checkpoint_sha256':candidate_sha,
  'args_json':str(args_json),'etas':etas,'optimizer_updates':0,'training_performed':False,'merges':{}}
 merged,stats=merged_state_dicts(parent['model_state_dict'],candidate['model_state_dict'],etas)
 for eta in etas:
  ck=build_merged_checkpoint(candidate,merged[eta],eta,parent_path=parent_path,parent_sha=parent_sha,
   candidate_path=candidate_path,candidate_sha=candidate_sha,pid_names=pid_names)
  target=out/f'checkpoint_task_vector_eta{eta:g}.pt'
  torch.save(ck,target);del ck
  manifest['merges'][repr(eta)]={'checkpoint':str(target),'checkpoint_sha256':sha(target),
   'stats':aggregate_stats(stats[eta]),'verification':verify_merged(target,parent_path,candidate_path,eta,str(args_json))}
  del merged[eta]
 (out/'merge-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print('TASK_VECTOR_MERGE_COMPLETE',sha(out/'merge-manifest.json'),flush=True)

def verify_command(a):
 m=json.loads(Path(a.manifest).read_text())
 for key,row in m['merges'].items():
  if sha(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('merged checkpoint changed on disk')
  verify_merged(row['checkpoint'],m['parent_checkpoint'],m['candidate_checkpoint'],float(key),m['args_json'])
 print('TASK_VECTOR_MERGE_VERIFY_OK')

def main():
 p=argparse.ArgumentParser()
 sub=p.add_subparsers(required=True)
 m=sub.add_parser('merge');m.add_argument('--parent',required=True);m.add_argument('--candidate',required=True)
 m.add_argument('--args-json',required=True);m.add_argument('--etas',required=True);m.add_argument('--output',required=True)
 m.set_defaults(command=merge_command)
 v=sub.add_parser('verify');v.add_argument('--manifest',required=True);v.set_defaults(command=verify_command)
 a=p.parse_args();a.command(a)

if __name__=='__main__':main()
