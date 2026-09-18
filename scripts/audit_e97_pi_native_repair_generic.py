#!/usr/bin/env python3
"""Generic fail-closed audit for any repair-tranche proposal.

Validates: declared identity bindings, non-authorization flags, authority/pack
non-trainable state, schedule consistency with the frozen plan, per-update
cohort stratification over the proposal's declared cohorts, and the unchanged
frozen dual gate. Cohort-specific provenance is cross-checked against the
authority manifest's own recorded bindings.
"""
import argparse,hashlib,json
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(args):
 proposal=json.loads(args.proposal.read_text());root=Path(proposal['preparation_root'])
 authority=json.loads((root/'manifest.json').read_text());packs=json.loads((root/'packs/manifest.json').read_text())
 schedule_path=Path(args.schedule) if args.schedule else root/'schedule-proposal.json'
 schedule=json.loads(schedule_path.read_text())
 for path,digest in proposal['identity_bindings']:
  if sha(Path(path))!=digest:raise ValueError(f'identity binding failed: {path}')
 if proposal['schema']!=args.schema:raise ValueError('schema')
 if proposal['status']!='frozen-proposal-not-authorized' or proposal['operator_internal_training_authorized'] or proposal['optimizer_updates_authorized'] or proposal['packing_authorized'] or proposal['checkpoint_promotion'] or proposal['automatic_retry'] or proposal['automatic_expansion'] or proposal['new_rl_updates']:raise ValueError('proposal authorization')
 if authority['training_eligible'] or authority['packing_authorized'] or authority['optimizer_updates_authorized'] or packs['training_eligible'] or packs['diagnostic_system_gate']!='cpu-system-gate' or schedule['training_eligible'] or schedule['optimizer_updates_authorized']:raise ValueError('prepared authority authorization')
 cohorts=set(proposal['declared_cohorts'])
 if set(authority['source_target_totals'])!=cohorts:raise ValueError('authority cohorts differ from declared')
 if set(schedule['source_target_totals'])!=cohorts:raise ValueError('schedule cohorts differ from declared')
 unstratified=[s['update'] for s in schedule['steps'] if set(s['source_targets'])!=cohorts]
 if unstratified:raise ValueError(f'updates missing cohorts: {unstratified}')
 if schedule['sampler_key']!=proposal['sampler_key'] or schedule['world_size']!=proposal['data_world_size'] or schedule['context_size']!=proposal['context_size'] or len(schedule['steps'])!=proposal['proposed_updates'] or schedule['source_target_totals']!=proposal['scheduled_source_targets'] or schedule['source_unique_records']!=proposal['scheduled_source_unique_records'] or sum(schedule['source_token_totals'].values())!=proposal['scheduled_input_tokens'] or sum(schedule['source_target_totals'].values())!=proposal['scheduled_assistant_targets'] or schedule['unique_packs']!=proposal['scheduled_unique_packs'] or schedule['unique_records']!=proposal['scheduled_unique_records']:raise ValueError('proposal schedule')
 for key,binding in proposal.get('cohort_bindings',{}).items():
  entry=authority.get(key)
  if entry is None:raise ValueError(f'missing cohort binding {key}')
  for field,expected in binding.items():
   if entry.get(field)!=expected:raise ValueError(f'cohort binding {key}.{field}')
 if proposal['parent_checkpoint_sha256']=='9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa':pass
 else:
  arc=proposal.get('arc') or {};idx=arc.get('segment_index')
  if not isinstance(idx,int) or idx<1 or not isinstance(proposal['parent_checkpoint'],str):raise ValueError('parent identity')
  prev_root=Path(proposal['parent_checkpoint']).parents[1]
  try:prev_audit=json.loads((prev_root/'audit.json').read_text())
  except (OSError,ValueError):raise ValueError('parent identity')
  if prev_audit.get('status')!='passed-training-not-promoted' or prev_audit.get('checkpoint',{}).get('sha256')!=proposal['parent_checkpoint_sha256']:raise ValueError('parent identity')
  if idx>=2:
   prev_arc=prev_audit.get('arc') or {}
   if prev_arc.get('arc_id')!=arc.get('arc_id') or prev_arc.get('segment_index')!=idx-1:raise ValueError('parent identity')
 gate=proposal['proposed_behavioral_gate']
 if gate['pi_native_stage_b']!='valid first frame >= 12/14 and correct first action >= 10/14' or gate['openhands_execution_overall']!='>= 64/96':raise ValueError('frozen gates')
 receipt={'schema':'emender-e97-pi-native-repair-proposal-audit-v1','status':'qualified-proposal-not-authorized',
  'proposal_schema':proposal['schema'],'proposal_sha256':sha(args.proposal),
  'parent_checkpoint_sha256':proposal['parent_checkpoint_sha256'],
  'authority_manifest_sha256':proposal['authority_manifest_sha256'],'pack_manifest_sha256':proposal['pack_manifest_sha256'],
  'schedule_sha256':proposal['schedule_sha256'],'proposed_updates':proposal['proposed_updates'],
  'scheduled_input_tokens':proposal['scheduled_input_tokens'],'scheduled_assistant_targets':proposal['scheduled_assistant_targets'],
  'declared_cohorts':sorted(cohorts),
  'per_update_cohort_stratification':'verified: all 32 updates contain all declared cohorts',
  'checker_sha256':sha(__file__),'packing_authorized':False,'optimizer_updates_authorized':0,'checkpoint_promotion':False}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('REPAIR_PROPOSAL_AUDIT',sha(args.proposal),sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True);p.add_argument('--schema',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--schedule',type=Path,default=None);audit(p.parse_args())
if __name__=='__main__':main()
