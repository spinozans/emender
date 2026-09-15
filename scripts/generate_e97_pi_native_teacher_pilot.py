#!/usr/bin/env python3
"""Render and validate a bounded DeepSeek/Pi-subagent task-specification pilot."""
import argparse,hashlib,json,os,re
from collections import Counter
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha
ALLOWED={'read','bash','edit','write','process','ffgrep','fffind','web_search','source_check','fetch_content','get_search_content','think','finish'}

def render(config):
 lanes=[]
 schema={k:('array of strings' if k in ('required_actions','diversity_tags') else 'object mapping relative paths to complete UTF-8 contents' if k=='workspace_files' else 'boolean' if k=='failure_prefix' else 'string') for k in config['required_spec_fields']}
 for lane in config['lanes']:
  lanes.append(f"LANE {lane['name']}: create exactly {config['candidates_per_lane']} specifications spanning {', '.join(lane['families'])}.")
 return f'''You are the parent curriculum designer. Use the subagent tool exactly once with one workflowScript that launches four parallel delegate children, one for each lane below. Every child inherits the current DeepSeek model. Wait for all children, validate their counts, and return one JSON object with exactly two keys: schema and tasks. schema must be "emender-e97-pi-native-teacher-task-pool-v1". tasks must contain exactly {config['candidate_total']} objects, ordered by the lane order below and then by id. Return raw JSON only, with no markdown or explanation.

{chr(10).join(lanes)}

Each task object has exactly this field schema:
{json.dumps(schema,sort_keys=True)}

Allowed action names: {json.dumps(sorted(ALLOWED))}.
IDs must be unique and begin teacher-<lane>-. lane must exactly match its assignment. family must come from that lane's family list. user_goal must be self-contained and executable in a fresh bounded workspace. workspace_files contains only relative paths and complete initial contents; use an empty object when unnecessary. expected_first_action is one allowed action. required_actions is an ordered nonempty list ending in finish. oracle states an independently checkable final answer, file snapshot, command/test result, or sourced-evidence condition. failure_prefix is true only when an intentional failing action must be observed and masked before recovery. diversity_tags has at least three concrete axes.

Generate task specifications only. Never invent tool observations, claim a task passed, include credentials, spend money, mutate a network service, use production systems, or depend on private repositories. Avoid these held-out evaluation entities and exact values: {json.dumps(config['constraints']['evaluation_entities_forbidden'])}. Use synthetic unique names/values. Web tasks should specify a research question or safe public/local URL behavior, not an answer claimed without execution. Local-vs-web contrasts must distinguish already-supplied facts from current/external facts. Keep each workspace below 16 files and 32KB. Do not call any tool other than subagent.'''

def validate(config,raw):
 text=raw.read_text().strip()
 try:data=json.loads(text)
 except json.JSONDecodeError as e:raise ValueError(f'non-JSON teacher output: {e}')
 if set(data)!={'schema','tasks'} or data['schema']!='emender-e97-pi-native-teacher-task-pool-v1' or not isinstance(data['tasks'],list):raise ValueError('teacher envelope')
 tasks=data['tasks'];fields=set(config['required_spec_fields'])
 if len(tasks)!=config['candidate_total']:raise ValueError('teacher task count')
 lane_specs={x['name']:x for x in config['lanes']};counts=Counter();ids=set();goals=set();forbidden=[x.lower() for x in config['constraints']['evaluation_entities_forbidden']]
 for t in tasks:
  if set(t)!=fields:raise ValueError('teacher task fields')
  if not all(isinstance(t[k],str) for k in ('id','lane','family','user_goal','expected_first_action','oracle')):raise ValueError('teacher strings')
  lane=lane_specs.get(t['lane'])
  if lane is None or t['family'] not in lane['families'] or not t['id'].startswith('teacher-'+t['lane']+'-'):raise ValueError('lane/family/id')
  if t['id'] in ids or t['user_goal'] in goals:raise ValueError('duplicate identity/goal')
  ids.add(t['id']);goals.add(t['user_goal']);counts[t['lane']]+=1
  if t['expected_first_action'] not in ALLOWED or not isinstance(t['required_actions'],list) or not t['required_actions'] or t['required_actions'][-1]!='finish' or any(x not in ALLOWED for x in t['required_actions']):raise ValueError('action contract')
  if type(t['failure_prefix']) is not bool or not isinstance(t['diversity_tags'],list) or len(t['diversity_tags'])<3 or any(not isinstance(x,str) or not x for x in t['diversity_tags']):raise ValueError('tags/failure')
  files=t['workspace_files']
  if not isinstance(files,dict) or len(files)>16 or sum(len(str(v).encode()) for v in files.values())>32768:raise ValueError('workspace bound')
  for path,content in files.items():
   p=Path(path)
   if not isinstance(path,str) or not isinstance(content,str) or p.is_absolute() or '..' in p.parts or path in ('','.'):raise ValueError('unsafe workspace path')
  flat=json.dumps(t,sort_keys=True).lower()
  if any(x in flat for x in forbidden) or re.search(r'(?i)\b(api[_ -]?key|password|secret|credentials?|purchase|production deploy)\b',flat):raise ValueError('forbidden teacher content')
 if counts!={x['name']:config['candidates_per_lane'] for x in config['lanes']}:raise ValueError('lane counts')
 canonical={'schema':data['schema'],'tasks':sorted(tasks,key=lambda t:([x['name'] for x in config['lanes']].index(t['lane']),t['id']))}
 return canonical,counts

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True);r=sp.add_parser('render');r.add_argument('--config',type=Path,required=True);r.add_argument('--output',type=Path,required=True);v=sp.add_parser('validate');v.add_argument('--config',type=Path,required=True);v.add_argument('--raw',type=Path,required=True);v.add_argument('--output',type=Path,required=True);a=p.parse_args();os.umask(0o077);config=json.loads(a.config.read_text())
 if config['schema']!='emender-e97-pi-native-teacher-pilot-v1' or config['candidate_total']!=80 or config['pi_subagents']!=4 or config['provider']!='lunaroute' or config['model']!='deepseek-4.1-flash' or config['constraints']['training_eligible'] or config['constraints']['optimizer_updates']:raise ValueError('teacher pilot authority')
 if a.command=='render':a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(render(config));print('TEACHER_PILOT_PROMPT',sha(a.output))
 else:
  canonical,counts=validate(config,a.raw);a.output.mkdir(parents=True,mode=0o700,exist_ok=False);publish(a.output/'tasks.json',canonical);publish(a.output/'summary.json',{'status':'teacher-task-specifications-validated','tasks':len(canonical['tasks']),'lane_counts':dict(counts),'config_sha256':sha(a.config),'raw_sha256':sha(a.raw),'tasks_sha256':sha(a.output/'tasks.json'),'real_pi_executions':0,'model_generations':0,'optimizer_updates':0,'training_eligible':False});print('TEACHER_PILOT_VALIDATED',len(canonical['tasks']),sha(a.output/'tasks.json'))
if __name__=='__main__':main()
