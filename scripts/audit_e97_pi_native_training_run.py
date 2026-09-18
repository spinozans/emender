#!/usr/bin/env python3
"""Independently audit the authorized 32-update Pi-native SFT run."""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def audit(args):
 import torch
 run=args.run;recipe=read(run/'recipe.json');terminal=read(run/'controller-terminal.json');schedule=read(recipe['schedule']['path']);admission=read(args.admission)
 if terminal['process_exit_code'] or terminal['automatic_retry'] or terminal['in_run_terminal_present'] or terminal['missing_in_run_terminal_reason']!='gpu lease EXIT trap replaced launcher EXIT trap; independently audited after clean process exit':raise ValueError('terminal status')
 updates=int(recipe['steps'])
 if updates<=0 or updates%32:raise ValueError('segment granularity')
 if recipe['operator_authorization_statement']!=f'I authorize the exact {updates} update proposal.' or recipe['lr']!=1e-5 or recipe['automatic_retry'] or recipe['automatic_expansion'] or recipe['checkpoint_promotion'] or recipe['new_rl_updates']:raise ValueError('authorization scope')
 if sha(args.admission)!=recipe['admission_sha256'] or admission['status']!='authorized-exact-proposal' or admission['authorized_updates']!=updates or sha(recipe['schedule']['path'])!=recipe['schedule']['sha256'] or schedule['status']!='authorized-exact-proposal' or len(schedule['steps'])!=updates:raise ValueError('admission/schedule identity')
 lines=(run/'console.log').read_text().splitlines();events=[json.loads(x) for x in lines if x.startswith('{') and '"event"' in x];starts=[x for x in events if x['event']=='start'];steps=[x for x in events if x['event']=='step'];checkpoints=[x for x in events if x['event']=='checkpoint'];completes=[x for x in events if x['event']=='complete']
 if len(starts)!=1 or len(steps)!=updates or len(checkpoints)!=1 or len(completes)!=1:raise ValueError('event coverage')
 start=starts[0]
 expected_start={'parent_sha256':recipe['parent_sha256'],'source_commit':recipe['source_commit'],'world_size':8,'context_size':65536,'start_update':0,'sampler_mode':'epoch-permutation','source_weight_mode':'new-stage-train','lr':1e-5,'diloco_merge_enabled':False,'boundary_aware_packs':True}
 if any(start.get(k)!=v for k,v in expected_start.items()) or start['total_parameters']!=4045972080:raise ValueError('start identity')
 total_tokens=total_targets=0;losses=[]
 for event,planned in zip(steps,schedule['steps']):
  if event['update']!=planned['update'] or event['rank_sample_ids']!=planned['rank_sample_ids'] or event['global_tokens']!=planned['global_tokens'] or event['global_targets']!=planned['global_targets']:raise ValueError('runtime schedule mismatch')
  total_tokens+=event['global_tokens'];total_targets+=event['global_targets'];losses.append(event['loss'])
  if event['total_tokens']!=total_tokens or event['total_targets']!=total_targets or not all(math.isfinite(event[k]) for k in ('loss','grad_norm','step_seconds','max_hbm_allocated')):raise ValueError('runtime clocks/numerics')
 checkpoint_event=checkpoints[0];checkpoint_path=Path(checkpoint_event['checkpoint']).resolve(strict=True)
 if checkpoint_path.parent!=(run/'checkpoints').resolve() or checkpoint_event['update']!=updates or sha(checkpoint_path)!=checkpoint_event['checkpoint_sha256'] or checkpoint_path.stat().st_size!=checkpoint_event['checkpoint_bytes']:raise ValueError('checkpoint publication')
 complete=completes[0]
 if complete['updates']!=updates or complete['requested_steps']!=updates or complete['reason'] is not None or complete['total_tokens']!=total_tokens or complete['assistant_target_tokens']!=total_targets:raise ValueError('completion receipt')
 checkpoint=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
 expected={'sft_updates':updates,'sft_total_tokens':total_tokens,'assistant_target_tokens':total_targets,'sampler_cursor':updates,'sampler_key':recipe['schedule']['sampler_key'],'learning_rate':1e-5,'data_world_size':8,'diloco_k':4,'diloco_merge_enabled':False,'parent_checkpoint_sha256':recipe['parent_sha256'],'authority_manifest_sha256':recipe['authority_manifest_sha256'],'pack_manifest_sha256':recipe['pack_manifest_sha256'],'source_commit':recipe['source_commit']}
 if any(checkpoint.get(k)!=v for k,v in expected.items()):raise ValueError('checkpoint clocks/identity')
 precision=checkpoint['sft_precision']
 if precision['optimizer']!='bf16-sr-candidate' or precision['learning_rate']!=1e-5 or precision['bf16_reduced_precision_reduction'] is not False:raise ValueError('precision policy')
 optimizer=checkpoint['optimizer_state_dict'];group=optimizer['param_groups'][0];ids=set(group['params'])
 if group['k']!=updates or group['train_mode'] is not False or ids!=set(optimizer['state']) or ids!=set(optimizer['eval_live_y']):raise ValueError('optimizer state coverage')
 if sum(state['z'].numel() for state in optimizer['state'].values())!=4045972080:raise ValueError('optimizer coordinates')
 tensors=list(checkpoint['model_state_dict'].values())+list(optimizer['eval_live_y'].values())
 for state in optimizer['state'].values():
  if set(state)!={'z','exp_avg_sq'}:raise ValueError('optimizer slots')
  tensors.extend(state.values())
 for tensor in tensors:
  if tensor.is_floating_point() and tensor.dtype!=torch.bfloat16:raise ValueError('persistent dtype')
  flat=tensor.reshape(-1)
  for begin in range(0,flat.numel(),1048576):
   if not bool(torch.isfinite(flat[begin:begin+1048576]).all()):raise ValueError('nonfinite checkpoint')
 if (run/'checkpoints/latest.pt').resolve()!=checkpoint_path:raise ValueError('atomic latest pointer')
 source=Counter()
 for row in schedule['steps']:source.update(row['source_targets'])
 report={'schema':'emender-e97-pi-native-curriculum-training-audit-v1','status':'passed-training-not-promoted','actual_training':True,'proposal_sha256':recipe['proposal_sha256'],'admission_sha256':recipe['admission_sha256'],'recipe_sha256':sha(run/'recipe.json'),'schedule_sha256':recipe['schedule']['sha256'],'updates':updates,'input_tokens':total_tokens,'assistant_target_tokens':total_targets,'source_target_totals':dict(source),'runtime_sample_ids_and_clocks_exact':True,'step_losses':losses,'final_loss':checkpoint_event['loss'],'peak_rank0_hbm_allocated':max(x['max_hbm_allocated'] for x in steps),'checkpoint':{'path':str(checkpoint_path),'sha256':checkpoint_event['checkpoint_sha256'],'bytes':checkpoint_event['checkpoint_bytes'],'finite_complete_bf16_state':True},'numerical_fresh_continuation':'not measured; exact restoration required; no tolerance fallback','automatic_retry':False,'automatic_expansion':False,'checkpoint_promotion':False,'stage_b_executed':False,'new_rl_updates':0,'checker_sha256':sha(__file__)}
 args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');checkpoint_path.chmod(0o400);print('PI_NATIVE_TRAINING_AUDIT_PASS',checkpoint_event['checkpoint_sha256'],sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
