#!/usr/bin/env python3
"""Fail-closed audit for the non-authorizing repository-correction proposal."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def inventory(root):
 root=Path(root); summary=json.loads((root/'summary.json').read_text()); rows={r['name']:r for r in summary['results']}
 candidates={}; families=Counter(); turns=supervised=masked=calls=errors=targets=clean=recovery=0
 for path in sorted(root.glob('training-*/episode-private.json')):
  name=path.parent.name; row=rows.pop(name); episode=json.loads(path.read_text()); generations=episode['generations']; start=row['supervise_assistant_from']
  if episode['training_eligible'] is not False or row['training_eligible'] is not False or row['evaluation_trajectory_reused'] is not False:raise ValueError('non-training-only seed')
  if start not in (0,1) or start!=row['authored_failure_prefix_turns']:raise ValueError('mask intent')
  counts=[len(g['token_ids']) for g in generations]
  candidates[name]=sha(path); turns+=len(generations);supervised+=len(generations)-start;masked+=start;targets+=sum(counts[start:])
  calls+=row['native_calls'];errors+=row['native_errors'];families[name.split('-')[2]]+=1
  clean+=start==0;recovery+=start==1
 if rows or len(candidates)!=8:raise ValueError('seed coverage')
 return dict(records=len(candidates),families=dict(sorted(families.items())),clean_discovery_records=clean,
  authored_failure_prefix_records=recovery,assistant_turns=turns,supervised_assistant_turns=supervised,
  masked_authored_assistant_turns=masked,native_calls=calls,authentic_errors=errors,supervised_target_tokens=targets,
  positive_verifiers=8,negative_verifiers=8,distinct_cleanups=32),candidates


def audit(config):
 c=json.loads(Path(config).read_text());root=Path(c['seed_preparation_root'])
 for name,file in [('seed_plan_sha256','plan-private.json'),('seed_summary_sha256','summary.json'),('seed_audit_sha256','independent-audit.json')]:
  if sha(root/file)!=c[name]:raise ValueError('seed authority changed')
 inv,files=inventory(root)
 if inv!=c['seed_inventory'] or files!=c['candidate_files_sha256']:raise ValueError('seed inventory changed')
 recipe=c['prospective_training_recipe']; admission=c['admission_prerequisites']
 if recipe['replay_total_target_quota']!=sum(recipe['replay_target_quotas'].values()):raise ValueError('replay quota arithmetic')
 if admission['minimum_verified_records']<=inv['records'] or admission['minimum_families']<=len(inv['families']):raise ValueError('proposal would admit seed')
 if any((c['training_authorized'],c['dataset_admission_authorized'],c['packing_authorized'],c['checkpoint_promotion'],c['training_eligible'])):raise ValueError('proposal authorizes action')
 if c['optimizer_updates_authorized']!=0 or c['outcome_rl_updates_authorized']!=0 or recipe['authorization_required'] is not True:raise ValueError('closed budget changed')
 return dict(status='proposal-audit-passed',seed_inventory=inv,prospective_replay_target_quota=recipe['replay_total_target_quota'],
  prospective_new_target_cap=recipe['new_discovery_target_exposure_cap'],optimizer_updates_authorized=0,training_eligible=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();result=audit(a.config)
 if a.output:a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print('REPOSITORY_DISCOVERY_PROPOSAL_AUDIT_PASSED',result['seed_inventory']['records'],result['seed_inventory']['supervised_target_tokens'])
if __name__=='__main__':main()
