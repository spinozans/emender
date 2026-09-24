#!/usr/bin/env python3
"""Protected-panel entity-overlap audit for the multi-episode session collection.

Same policy and same fixed protected panels + Stage-A panel as the
hybrid-conversation overlap auditor (exact-and-significant-entity-v1),
adapted to the session plan shape: every episode prompt and every episode
step of a session contributes to the candidate domains, session fixture
files contribute contents and exact scalars, and finish values are collected
across all episodes. Fail closed on any significant entity collision.
"""
import argparse,hashlib,json
from pathlib import Path
from ndm.e97_protected_overlap import _domains,extract_exact_scalars,load_protected_panel,normalize_content
FIXED=(
 ('/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/manifest.json','/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/records.jsonl'),
 ('/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/manifest.json','/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/records.jsonl'),
 ('/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/manifest.json','/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/tasks.jsonl'))
STAGE_SHA='07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef'
PLAN_SCHEMA='emender-e97-multi-episode-session-plan-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(cases):
 rows=[]
 for c in cases:
  for k,ep in enumerate(c['episodes']):
   rows.append({'task_id':f"{c['id']}#ep{k}",'family_id':c['family'],'repository':'synthetic/e97-pi-native-v1',
    'prompt_template':ep['prompt'],
    'fixture_files':[{'path':p,'content':v} for p,v in c['files'].items()],
    'exact_scalars':sorted(extract_exact_scalars(c['files'].values()))})
 return rows
def counts(left,right):return {f:len(left[f]&right[f]) for f in sorted(left)}
def significant(items,n):return {x for x in items if len(x.encode())>=n}
def audit(args):
 plan=json.loads(args.plan.read_text());summary=json.loads((args.selected/'summary.json').read_text())
 if sha(args.plan)!=args.plan_sha or summary['records']!=len(plan['cases']):raise ValueError('input identity')
 if plan.get('schema')!=PLAN_SCHEMA:raise ValueError('session plan required')
 metadata=[json.loads(x) for x in (args.selected/'candidate-authority/records.jsonl').read_text().splitlines()]
 ids={x['id'] for x in metadata}
 if not ids<=set(c['id'] for c in plan['cases']):raise ValueError('collected identity outside plan')
 cases=[c for c in plan['cases'] if c['id'] in ids]
 candidate=_domains(records(cases))
 protected=[];manifest_shas=[];record_shas=[]
 for mp,rp in FIXED:
  ms,rs,items=load_protected_panel(Path(mp),Path(rp));manifest_shas.append(ms);record_shas.append(rs);protected.extend(items)
 fixed=counts(candidate,_domains(protected))
 stage=json.loads(args.stage_panel.read_text())
 if sha(args.stage_panel)!=STAGE_SHA:raise ValueError('stage panel identity')
 stage_records=[{'task_id':r.get('task_id',r.get('id')),'family_id':r.get('family_id',r.get('family')),'repository':'synthetic/e97-stage-a-v1','prompt_template':r['prompt'],'fixture_files':[{'path':p,'content':v} for p,v in (r.get('files') or {}).items()],'exact_scalars':sorted(extract_exact_scalars((r.get('files') or {}).values()))} for r in stage['cases']]
 stage_domains=counts(candidate,_domains(stage_records))
 cand_contents={f['content'] for r in records(cases) for f in r['fixture_files']}
 stage_contents={f['content'] for r in stage_records for f in r['fixture_files']}
 sig_content_full=len(significant(cand_contents,16)&significant(stage_contents,16))
 sig_content_norm=len({normalize_content(x) for x in significant(cand_contents,16)}&{normalize_content(x) for x in significant(stage_contents,16)})
 sig_scalars=len(significant(candidate['exact_scalars'],8)&significant(_domains(stage_records)['exact_scalars'],8))
 finish_values={s['arguments']['message'] for c in cases for ep in c['episodes'] for s in ep['steps'] if s.get('name')=='finish' and 'arguments' in s}
 stage_finish=significant({x['expected_final'] for x in stage['cases'] if isinstance(x.get('expected_final'),str)},8)
 entity={'fixed_panel_all_domains':fixed,'stage_a_all_domains':stage_domains,
  'stage_a_significant_contents_full':sig_content_full,'stage_a_significant_contents_normalized':sig_content_norm,
  'stage_a_significant_exact_scalars':sig_scalars,
  'stage_a_significant_finish_values':len(significant(finish_values,8)&stage_finish)}
 trivial_numeric_fixed=sorted(x for x in set(candidate['exact_scalars'])&set(_domains(protected)['exact_scalars']) if len(x.encode())<8)
 if (any(v for k,v in fixed.items() if k!='exact_scalars')
  or significant(candidate['exact_scalars'],8)&significant(_domains(protected)['exact_scalars'],8)
  or any(stage_domains.values()) or sig_content_full or sig_content_norm or sig_scalars
  or (significant(finish_values,8)&stage_finish)):raise ValueError('protected overlap')
 receipt={'schema':'emender-e97-multi-episode-session-overlap-audit-v1','status':'pass','records':len(ids),
  'plan_sha256':args.plan_sha,'authority_sha256':sha(args.selected/'candidate-authority/manifest.json'),
  'checker_sha256':sha(__file__),'fixed_protected_manifest_sha256s':sorted(manifest_shas),
  'fixed_protected_record_sha256s':sorted(record_shas),'entity_collision_counts':entity,
  'stage_a_panel_sha256':STAGE_SHA,'entity_policy':'exact-and-significant-entity-v1; structural family/path-template reuse and trivial sub-8-byte numeric fixture scalars are reported but are not entity collisions (ledger precedent: trivial alpha/1 content)',
  'trivial_numeric_fixed_panel_scalars':trivial_numeric_fixed,
  'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('MULTI_EPISODE_SESSION_OVERLAP_PASS',len(ids),sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
 p.add_argument('--selected',type=Path,required=True);p.add_argument('--stage-panel',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
