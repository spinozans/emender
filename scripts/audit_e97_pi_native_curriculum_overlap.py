#!/usr/bin/env python3
"""Fixed-panel and Stage-A entity-overlap audit for selected Pi-native candidates."""
import argparse,hashlib,json
from pathlib import Path
from urllib.parse import urlparse
from ndm.e97_protected_overlap import _domains,extract_exact_scalars,load_protected_panel,normalize_content
FIXED=(
 ('/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/manifest.json','/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/records.jsonl'),
 ('/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/manifest.json','/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/records.jsonl'),
 ('/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/manifest.json','/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/tasks.jsonl'))
STAGE_SHA='07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(cases,ids,repository):
 result=[]
 for case in cases:
  if ids is not None and case['id'] not in ids:continue
  result.append({'task_id':case['id'],'family_id':case['family'],'repository':repository,'prompt_template':case['prompt'],'fixture_files':[{'path':p,'content':c} for p,c in case['files'].items()],'exact_scalars':sorted(extract_exact_scalars(case['files'].values()))})
 return result
def counts(left,right):return {field:len(left[field]&right[field]) for field in sorted(left)}
def significant_contents(items):return {x for x in items if len(x.encode())>=16}
def significant_scalars(items):return {x for x in items if len(x.encode())>=8}
def finish_values(cases,ids):
 values=set()
 for case in cases:
  if ids is not None and case['id'] not in ids:continue
  for step in case.get('steps',[]):
   if step.get('name')=='finish' and 'arguments' in step:values.add(step['arguments']['message'])
 return significant_scalars(values)
def audit(args):
 plan=json.loads(args.plan.read_text());selected_manifest=json.loads((args.selected/'candidate-authority/manifest.json').read_text());summary=json.loads((args.selected/'summary.json').read_text());source_audit=json.loads((args.source/'audit.json').read_text());stage=json.loads(args.stage_panel.read_text())
 if sha(args.plan)!=args.plan_sha or sha(args.source/'audit.json')!=args.source_audit_sha or sha(args.selected/'candidate-authority/manifest.json')!=args.selected_authority_sha or sha(args.stage_panel)!=STAGE_SHA:raise ValueError('input identity')
 if source_audit['status']!='qualified-candidates-not-admitted' or selected_manifest['status']!='verified-selection-not-admitted' or summary['records']!=selected_manifest['records'] or selected_manifest['records']<2000:raise ValueError('candidate state')
 if selected_manifest['training_eligible'] or selected_manifest['packing_authorized'] or selected_manifest['optimizer_updates_authorized']:raise ValueError('candidate authorization')
 metadata=[json.loads(x) for x in (args.selected/'candidate-authority/records.jsonl').read_text().splitlines()];ids={x['id'] for x in metadata}
 if len(ids)!=len(metadata) or any(x['family']=='preservation' for x in metadata):raise ValueError('selection coverage')
 cases={x['id']:x for x in plan['cases']}
 if not ids<=set(cases):raise ValueError('selected task identity')
 candidate_records=records(plan['cases'],ids,'synthetic/e97-pi-native-v1');candidate_domains=_domains(candidate_records)
 protected=[];manifest_shas=[];record_shas=[]
 for manifest_path,record_path in FIXED:
  manifest_sha,record_sha,items=load_protected_panel(Path(manifest_path),Path(record_path));manifest_shas.append(manifest_sha);record_shas.append(record_sha);protected.extend(items)
 fixed_domains=_domains(protected);fixed_collisions=counts(candidate_domains,fixed_domains)
 stage_records=records(stage['cases'],None,'synthetic/e97-stage-a-v1');stage_domains=_domains(stage_records);stage_collisions=counts(candidate_domains,stage_domains)
 candidate_contents={f['content'] for r in candidate_records for f in r['fixture_files']};stage_contents={f['content'] for r in stage_records for f in r['fixture_files']}
 significant_content_full=len(significant_contents(candidate_contents)&significant_contents(stage_contents));significant_content_normalized=len({normalize_content(x) for x in significant_contents(candidate_contents)}&{normalize_content(x) for x in significant_contents(stage_contents)})
 significant_scalar_collisions=len(significant_scalars(candidate_domains['exact_scalars'])&significant_scalars(stage_domains['exact_scalars']));finish_collisions=len(finish_values(plan['cases'],ids)&significant_scalars({x['expected_final'] for x in stage['cases'] if isinstance(x.get('expected_final'),str)}))
 stage_entity_core={'task_ids':stage_collisions['task_ids'],'repositories':stage_collisions['repositories'],'prompt_templates_full':stage_collisions['prompt_templates_full'],'prompt_templates_normalized':stage_collisions['prompt_templates_normalized'],'fixture_paths_full':stage_collisions['fixture_paths_full'],'significant_fixture_contents_full':significant_content_full,'significant_fixture_contents_normalized':significant_content_normalized,'significant_exact_scalars':significant_scalar_collisions,'significant_finish_values':finish_collisions}
 web_cases=[cases[x] for x in ids if cases[x]['category']=='web'];web_hosts=set();web_calls=0
 for case in web_cases:
  for step in case['steps']:
   if step.get('name')=='fetch_content' and 'arguments' in step:
    web_hosts.add(urlparse(step['arguments']['url']).hostname);web_calls+=1
 if any(fixed_collisions.values()) or any(stage_entity_core.values()) or web_hosts-{'www.rfc-editor.org','example.invalid'}:raise ValueError('protected overlap')
 receipt={'schema':'emender-e97-pi-native-curriculum-overlap-audit-v1','status':'pass','records':len(ids),'selected_authority_sha256':args.selected_authority_sha,'source_audit_sha256':args.source_audit_sha,'plan_sha256':args.plan_sha,'checker_sha256':sha(__file__),'fixed_protected_manifest_sha256s':sorted(manifest_shas),'fixed_protected_record_sha256s':sorted(record_shas),'fixed_collision_counts':fixed_collisions,'stage_a_panel_sha256':STAGE_SHA,'stage_a_entity_policy':'exact-and-significant-entity-v1; structural family/path-template reuse is reported but is not an entity collision','stage_a_entity_collision_counts':stage_entity_core,'stage_a_all_domain_collision_counts':stage_collisions,'web_records':len(web_cases),'planned_fetch_calls':web_calls,'planned_web_hosts':sorted(web_hosts),'repository_discovery_records':sum(x['repository_discovery'] for x in metadata),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_OVERLAP_AUDIT',len(ids),receipt['repository_discovery_records'],sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--source-audit-sha',required=True);p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-authority-sha',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--stage-panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
