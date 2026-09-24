#!/usr/bin/env python3
"""Deterministic FP32 soup merge of two sibling checkpoints onto a shared parent.

merged = parent + etaA*(candidateA-parent) + etaB*(candidateB-parent), computed
per tied tensor group in FP32 with a single BF16 rounding. Evaluation-only.
"""
import argparse,hashlib,json,os,shutil,struct
from pathlib import Path
import torch
from scripts.merge_e97_task_vector import (MANIFEST_SCHEMA,MERGE_SCHEMA,load_checkpoint,
 pid_names_from_groups,sha,verify_merged)

def merged_two(parent_sd,a_sd,b_sd,eta_a,eta_b):
 index={}
 for name,tensor in parent_sd.items():index.setdefault(tensor.data_ptr(),[]).append(name)
 groups=list(index.values())
 merged={};stats=[]
 for names in groups:
  base=parent_sd[names[0]];ca=a_sd[names[0]];cb=b_sd[names[0]]
  if base.shape!=ca.shape or ca.shape!=cb.shape or base.dtype!=torch.bfloat16:raise ValueError(f'bad group {names}')
  if len(names)>1:
   for sd,nm in ((a_sd,names),(b_sd,names)):
    others=[sd[n] for n in names[1:]]
    if any(t.data_ptr()!=sd[names[0]].data_ptr() or not torch.equal(t,sd[names[0]]) for t in others):raise ValueError(f'candidate tying differs {names}')
  bf=base.detach().float();da=ca.detach().float()-bf;db=cb.detach().float()-bf
  out=(bf+float(eta_a)*da+float(eta_b)*db).to(torch.bfloat16)
  if not bool(torch.isfinite(out.float()).all()):raise ValueError(f'nonfinite {names}')
  for n in names:merged[n]=out
  stats.append({'names':sorted(names),'numel':int(out.numel()),'unchanged_vs_parent':int((out==base).sum().item())})
 return merged,stats

def build_ck(template,merged_sd,*,parent_path,parent_sha,a_path,a_sha,b_path,b_sha,eta_a,eta_b,pid_names):
 ck=dict(template);ck['model_state_dict']=dict(merged_sd)
 opt=template['optimizer_state_dict'];state={};backups={}
 for pid in opt['state']:
  name=pid_names[pid]
  state[pid]={'z':merged_sd[name].clone(),'exp_avg_sq':opt['state'][pid]['exp_avg_sq']}
  backups[pid]=merged_sd[name].clone()
 new_opt=dict(opt);new_opt['state']=state;new_opt['eval_live_y']=backups
 ck['optimizer_state_dict']=new_opt;ck['loss']=None
 ck['sft_task_merge']={'schema':MERGE_SCHEMA,'purpose':'evaluation-only two-sibling soup; not a training product',
  'parent_checkpoint':str(parent_path),'parent_checkpoint_sha256':parent_sha,
  'candidate_a':str(a_path),'candidate_a_sha256':a_sha,'candidate_b':str(b_path),'candidate_b_sha256':b_sha,
  'eta_a':float(eta_a),'eta_b':float(eta_b),
  'arithmetic':'fp32 parent + etaA*deltaA + etaB*deltaB, single bf16 round-to-nearest-even per tied group',
  'optimizer_state_policy':'z and SR live-y backups set to merged x; exp_avg_sq from candidate A template; inert for evaluation',
  'training_authority_inapplicable':True}
 return ck

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--parent',required=True);p.add_argument('--candidate-a',required=True);p.add_argument('--candidate-b',required=True)
 p.add_argument('--args-json',required=True);p.add_argument('--eta-a',type=float,default=0.5);p.add_argument('--eta-b',type=float,default=0.5)
 p.add_argument('--output',required=True)
 a=p.parse_args()
 parent_path=Path(a.parent).resolve();a_path=Path(a.candidate_a).resolve();b_path=Path(a.candidate_b).resolve()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 parent=load_checkpoint(parent_path);ca=load_checkpoint(a_path);cb=load_checkpoint(b_path)
 pid_names=pid_names_from_groups(ca['optimizer_state_dict'],ca['model_state_dict'],Path(a.args_json).resolve(),a_path)
 merged,stats=merged_two(parent['model_state_dict'],ca['model_state_dict'],cb['model_state_dict'],a.eta_a,a.eta_b)
 ck=build_ck(ca,merged,eta_a=a.eta_a,eta_b=a.eta_b,parent_path=parent_path,parent_sha=sha(parent_path),
  a_path=a_path,a_sha=sha(a_path),b_path=b_path,b_sha=sha(b_path),pid_names=pid_names)
 target=out/'checkpoint_soup.pt';torch.save(ck,target);del ck
 manifest={'schema':MANIFEST_SCHEMA,'status':'merged','parent_checkpoint':str(parent_path),'parent_checkpoint_sha256':sha(parent_path),
  'candidate_a':str(a_path),'candidate_a_sha256':sha(a_path),'candidate_b':str(b_path),'candidate_b_sha256':sha(b_path),
  'etas':[a.eta_a,a.eta_b],'optimizer_updates':0,'training_performed':False,
  'checkpoint':str(target),'checkpoint_sha256':sha(target),
  'stats':{'coordinates':sum(s['numel'] for s in stats),'unchanged_vs_parent_fraction':sum(s['unchanged_vs_parent'] for s in stats)/sum(s['numel'] for s in stats)}}
 (out/'merge-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print('SOUP_MERGE_COMPLETE',sha(target),sha(out/'merge-manifest.json'),flush=True)

if __name__=='__main__':main()
