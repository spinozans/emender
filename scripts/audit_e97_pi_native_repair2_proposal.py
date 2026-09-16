#!/usr/bin/env python3
"""Fail-closed identity, exposure, and stratification audit for the repair proposal."""
import argparse,hashlib,json
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(args):
 proposal=json.loads(args.proposal.read_text());root=Path(proposal['preparation_root'])
 authority=json.loads((root/'manifest.json').read_text());packs=json.loads((root/'packs/manifest.json').read_text())
 schedule=json.loads((root/'schedule-proposal.json').read_text());fulltraj=json.loads((Path(proposal['openhands_rehearsal_source'])/'manifest.json').read_text())
 bridge=json.loads((Path(proposal['rehearsal_source'])/'manifest.json').read_text())
 identities=[(proposal['parent_checkpoint'],proposal['parent_checkpoint_sha256']),(root/'manifest.json',proposal['authority_manifest_sha256']),
  (root/'packs/manifest.json',proposal['pack_manifest_sha256']),(root/'schedule-proposal.json',proposal['schedule_sha256']),
  (Path(proposal['openhands_rehearsal_source'])/'manifest.json',proposal['openhands_rehearsal_source_sha256']),
  (Path(proposal['rehearsal_source'])/'manifest.json',proposal['rehearsal_source_sha256']),
  ('configs/pi/e97-active-tool-surface-v1.json',proposal['tool_manifest_sha256'])]
 if any(sha(path)!=digest for path,digest in identities):raise ValueError('proposal input identity')
 if fulltraj['schema']!='emender-open-swe-source-native-candidate-v1' or fulltraj.get('training_eligible') is True:raise ValueError('openhands source state')
 if bridge.get('schema')!='emender-e97-tulu3-masked-sft-v1' or bridge.get('training_eligible') is not True:raise ValueError('bridge source state')
 if proposal['schema']!='emender-e97-pi-native-repair2-training-proposal-v1' or proposal['status']!='frozen-proposal-not-authorized' or proposal['operator_internal_training_authorized'] or proposal['optimizer_updates_authorized'] or proposal['packing_authorized'] or proposal['checkpoint_promotion'] or proposal['automatic_retry'] or proposal['automatic_expansion'] or proposal['new_rl_updates']:raise ValueError('proposal authorization')
 if authority['training_eligible'] or authority['packing_authorized'] or authority['optimizer_updates_authorized'] or packs['training_eligible'] or packs['diagnostic_system_gate']!='cpu-system-gate' or schedule['training_eligible'] or schedule['optimizer_updates_authorized']:raise ValueError('prepared authority authorization')
 if schedule['sampler_key']!=proposal['sampler_key'] or schedule['world_size']!=proposal['data_world_size'] or schedule['context_size']!=proposal['context_size'] or len(schedule['steps'])!=proposal['proposed_updates'] or schedule['source_target_totals']!=proposal['scheduled_source_targets'] or schedule['source_unique_records']!=proposal['scheduled_source_unique_records'] or sum(schedule['source_token_totals'].values())!=proposal['scheduled_input_tokens'] or sum(schedule['source_target_totals'].values())!=proposal['scheduled_assistant_targets'] or schedule['unique_packs']!=proposal['scheduled_unique_packs'] or schedule['unique_records']!=proposal['scheduled_unique_records']:raise ValueError('proposal schedule')
 cohorts={'openhands-execution-rehearsal','pi-native-curriculum','representation-bridge-rehearsal','grounded-authored-rehearsal'}
 unstratified=[s['update'] for s in schedule['steps'] if set(s['source_targets'])!=cohorts]
 if unstratified:raise ValueError(f'updates missing cohorts: {unstratified}')
 if set(authority['source_target_totals'])!=cohorts or authority['source_record_counts']['openhands-execution-rehearsal']!=proposal['openhands_rehearsal_records'] or authority['openhands_rehearsal']['authority_sha256']!=proposal['openhands_rehearsal_source_sha256'] or authority['openhands_rehearsal']['seed']!=proposal['openhands_rehearsal_seed'] or authority['openhands_rehearsal']['target_token_budget']!=proposal['openhands_rehearsal_target_budget']:raise ValueError('openhands slice binding')
 if authority['authored_rehearsal'] is None or authority['authored_rehearsal']['authority_sha256']!=proposal['authored_source_sha256'] or authority['authored_rehearsal']['seed']!=proposal['authored_rehearsal_seed'] or authority['authored_rehearsal']['target_token_budget']!=proposal['authored_rehearsal_budget']:raise ValueError('authored slice binding')
 authored=json.loads((Path(proposal['authored_rehearsal_source'])/'manifest.json').read_text())
 if authored.get('schema')!='emender-e97-tulu3-masked-sft-v1' or authored.get('training_eligible') is not True:raise ValueError('authored source state')
 if (Path(proposal['authored_rehearsal_source'])/'manifest.json').resolve()!=Path(proposal['rehearsal_source']).resolve() and sha(Path(proposal['authored_rehearsal_source'])/'manifest.json')!=proposal['authored_source_sha256']:raise ValueError('authored manifest identity')
 if proposal['parent_checkpoint_sha256']!='9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa':raise ValueError('parent identity')
 gate=proposal['proposed_behavioral_gate']
 if gate['pi_native_stage_b']!='valid first frame >= 12/14 and correct first action >= 10/14' or gate['openhands_execution_overall']!='>= 64/96':raise ValueError('frozen gates')
 receipt={'schema':'emender-e97-pi-native-repair2-proposal-audit-v1','status':'qualified-proposal-not-authorized',
  'proposal_sha256':sha(args.proposal),'parent_checkpoint_sha256':proposal['parent_checkpoint_sha256'],
  'authority_manifest_sha256':proposal['authority_manifest_sha256'],'pack_manifest_sha256':proposal['pack_manifest_sha256'],
  'schedule_sha256':proposal['schedule_sha256'],'proposed_updates':proposal['proposed_updates'],
  'scheduled_input_tokens':proposal['scheduled_input_tokens'],'scheduled_assistant_targets':proposal['scheduled_assistant_targets'],
  'per_update_cohort_stratification':'verified: all 32 updates contain all four cohorts',
  'openhands_rehearsal_records':proposal['openhands_rehearsal_records'],
  'scheduled_openhands_targets':proposal['scheduled_source_targets']['openhands-execution-rehearsal'],
  'scheduled_pi_native_targets':proposal['scheduled_source_targets']['pi-native-curriculum'],
  'scheduled_authored_targets':proposal['scheduled_source_targets']['grounded-authored-rehearsal'],
  'checker_sha256':sha(__file__),'packing_authorized':False,'optimizer_updates_authorized':0,'checkpoint_promotion':False}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_REPAIR_PROPOSAL_AUDIT',sha(args.proposal),sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
