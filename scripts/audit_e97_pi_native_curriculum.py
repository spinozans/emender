#!/usr/bin/env python3
"""Independent reconstruction audit for non-admitted Pi-native curriculum candidates."""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import PiNativeEpisode,native_turn,parse_turn,semantic_turn
INDEX=struct.Struct('<QQQB7x')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_json(path):return json.loads(Path(path).read_text())
def verify_authority_files(plan):
 repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()
 for raw,digest in plan['authority_files'].items():
  path=Path(raw).resolve()
  try:relative=path.relative_to(repo)
  except ValueError:
   if sha(path)!=digest:raise ValueError('external authority source changed')
  else:
   payload=subprocess.check_output(['git','show',plan['source_commit']+':'+relative.as_posix()])
   if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('committed authority source mismatch')
def message_ends(path):return [x['message'] for x in map(json.loads,Path(path).read_text().splitlines()) if x.get('type')=='message_end']
def bare(message):
 role=message['role']
 if role=='user':return {'role':'user','content':message['content']}
 if role=='assistant':return {'role':'assistant','content':message['content']}
 if role=='toolResult':return {k:message[k] for k in ('role','toolCallId','toolName','content','isError')}
 raise ValueError('raw Pi message role')
def failed(result):
 text=result['content'][0]['text'].lower()
 return result['isError'] or text.startswith(('error:','tool error:')) or any(x in text for x in ('exit code: 1','command exited with code 1','no matches found','no files found matching pattern'))
def observation_final(text):return ('Returned web evidence: '+' '.join(text.split())[:500]).strip()
def action_sequence(source,tools):return [semantic_turn(x,tools) for x in source if x['role']=='assistant']
# ------------------------------------------------------- v2 dynamic finish
# Independent re-implementation of the v2 date-observation parsing and the
# dynamic finish resolution (scripts/build_e97_hybrid_conversation_collection_v2.py
# resolve_dynamic_v2): the audit must re-derive the emitted frames from the
# recorded ToolResult without importing the builder.
_V2_WEEKDAYS=('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
_V2_WEEKDAY_FULL={'mon':'Monday','tue':'Tuesday','wed':'Wednesday','thu':'Thursday','fri':'Friday','sat':'Saturday','sun':'Sunday'}
_V2_MONTH_FULL={'jan':'January','feb':'February','mar':'March','apr':'April','may':'May','jun':'June','jul':'July','aug':'August','sep':'September','oct':'October','nov':'November','dec':'December'}
_V2_BARE=re.compile(r'^([A-Za-z]{3}) ([A-Za-z]{3}) ([ 0-9][0-9]) (\d{2}:\d{2}:\d{2}) (\S+) (\d{4})$')
_V2_YMD=re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
_V2_ABD=re.compile(r'^([A-Za-z]+), ([A-Za-z]+) (\d{1,2}), (\d{4})$')
_V2_HM=re.compile(r'^(\d{2}):(\d{2})$')
def audit_date_values(text,fmt):
 s=text.strip()
 if fmt=='bare':
  m=_V2_BARE.match(s)
  if not m:raise ValueError(f'unparsed bare date observation: {s!r}')
  abbr,mon,day,clock,zone,year=m.groups()
  weekday=_V2_WEEKDAY_FULL[abbr.lower()];month=_V2_MONTH_FULL[mon.lower()];day=int(day)
  return {'weekday':weekday,'month':month,'day':str(day),'year':year,'time':clock,'zone':zone,'date':f'{month} {day}, {year}','ymd':f'{year}-{mon.title()}-{day:02d}'}
 if fmt=='A':
  if s not in _V2_WEEKDAYS:raise ValueError(f'unparsed weekday observation: {s!r}')
  return {'weekday':s}
 if fmt=='A-ymd':
  parts=s.split(' ',1)
  if len(parts)!=2 or parts[0] not in _V2_WEEKDAYS or not _V2_YMD.match(parts[1]):raise ValueError(f'unparsed weekday-date observation: {s!r}')
  return {'weekday':parts[0],'ymd':parts[1]}
 if fmt=='u-ymd':
  if not _V2_YMD.match(s):raise ValueError(f'unparsed utc date observation: {s!r}')
  return {'ymd':s}
 if fmt=='u-HM':
  if not _V2_HM.match(s):raise ValueError(f'unparsed utc time observation: {s!r}')
  return {'time':s}
 if fmt=='A-BdY':
  m=_V2_ABD.match(s)
  if not m:raise ValueError(f'unparsed long-date observation: {s!r}')
  weekday,month,day,year=m.groups()
  if weekday not in _V2_WEEKDAYS:raise ValueError(f'bad weekday: {s!r}')
  return {'weekday':weekday,'month':month,'day':str(int(day)),'year':year,'date':f'{month} {int(day)}, {year}'}
 raise ValueError(f'unknown date format {fmt}')
def _v2_check_analysis(step,message,enc):
 analysis=message['reasoning_content']
 if step.get('analysis_dynamic'):
  if not isinstance(analysis,str) or not analysis.strip():raise ValueError('v2 dynamic analysis missing')
  return
 if not isinstance(analysis,str) or not analysis.strip():raise ValueError('v2 nonempty analysis')
 if len(enc.encode_ordinary(analysis))>2048:raise ValueError('analysis cap')
 for needle in step.get('analysis_requires') or []:
  if needle not in analysis:raise ValueError('v2 ungrounded analysis')
def verify_hybrid_v2_case(case,turns,results,enc):
 """Re-derive every v2 authored assistant frame from the plan spec plus the
 recorded observations: arguments, commentary, analysis and dynamic finish
 text must all reconstruct exactly."""
 if len(turns)!=len(case['steps']):raise ValueError('v2 turn coverage')
 ri=0
 for step,turn in zip(case['steps'],turns):
  message=parse_turn(turn)
  if message['name']!=step['name']:raise ValueError('v2 step order')
  if step['name']=='finish':
   if step.get('dynamic')=='observation-date':
    observation=results[ri-1]['content'][0]['text']
    values=audit_date_values(observation,step['fmt'])
    if message['arguments']!={'message':step['template'].format(**values)}:raise ValueError('v2 date finish message')
    if message['reasoning_content']!=step['analysis_template'].format(**values):raise ValueError('v2 date finish analysis')
    if message['content']!=step['commentary']:raise ValueError('v2 date finish commentary')
   elif step.get('dynamic')=='observation-exact':
    observation=results[ri-1]['content'][0]['text'].strip()
    if message['arguments']!={'message':observation}:raise ValueError('v2 exact finish message')
    if message['reasoning_content']!=f'The tool observed {observation}; the user-facing answer is exactly that observed value.':raise ValueError('v2 exact finish analysis')
    if message['content']!=f'The observed answer is {observation}.':raise ValueError('v2 exact finish commentary')
   else:
    if 'arguments' in step and message['arguments']!=step['arguments']:raise ValueError('v2 static finish arguments')
    if 'commentary' in step and message['content']!=step['commentary']:raise ValueError('v2 static finish commentary')
  else:
   if message['arguments']!=step['arguments']:raise ValueError('v2 tool arguments')
   if 'commentary' in step and message['content']!=step['commentary']:raise ValueError('v2 tool commentary')
   ri+=1
  _v2_check_analysis(step,message,enc)
def verify_case(case,private,tools):
 source=private['source_messages'];actions=action_sequence(source,tools)
 if [x['name'] for x in actions]!=[x['name'] for x in case['steps']]:raise ValueError('planned action sequence')
 results=[x for x in source if x['role']=='toolResult'];ri=0
 for index,(planned,actual) in enumerate(zip(case['steps'],actions)):
  if 'dynamic' not in planned:
   if actual['arguments']!=planned['arguments']:raise ValueError('static action arguments')
  elif planned['dynamic']=='observation':
   if actual['arguments']!={'message':observation_final(results[ri-1]['content'][0]['text'])}:raise ValueError('observation final grounding')
  elif planned['dynamic']=='observation-exact':
   if actual['arguments']!={'message':results[ri-1]['content'][0]['text'].strip()}:raise ValueError('exact observation final grounding')
  elif planned['dynamic']=='observation-date':
   values=audit_date_values(results[ri-1]['content'][0]['text'],planned['fmt'])
   if actual['arguments']!={'message':planned['template'].format(**values)}:raise ValueError('date observation final grounding')
  elif planned['dynamic']=='response-id':
   prior=results[ri-1]['content'][0]['text'];rid=actual['arguments'].get('responseId')
   if not isinstance(rid,str) or rid not in prior:raise ValueError('response identity grounding')
  elif planned['dynamic'] in ('process-output','process-stop'):
   prior=results[ri-1]['content'][0]['text'];pid=actual['arguments'].get('id')
   if not isinstance(pid,str) or pid not in prior:raise ValueError('process identity grounding')
  else:raise ValueError('unknown dynamic action')
  if actual['name']!='finish':ri+=1
 if private['snapshot']!=case['expected_files']:raise ValueError('snapshot oracle')
 if case['category']=='web' and case['family'] in ('fetch-content','fetch-retrieve','fetch-error-recovery'):
  if not any(case['expected_fact'] in x['content'][0]['text'] for x in results):raise ValueError('web observation oracle')
  if actions[-1]['arguments']['message']!=case['expected_fact']:raise ValueError('web final oracle')
 if case.get('answer_must_be_observed'):
  if case.get('expected_answer') is None or not any(str(case['expected_answer']) in x['content'][0]['text'] for x in results):raise ValueError('hybrid answer observation oracle')
 if case.get('answer_is_observation'):
  if not results or actions[-1]['arguments']['message']!=results[-1]['content'][0]['text'].strip():raise ValueError('hybrid observation answer oracle')
 if case.get('pure_chat') and any(x['name']!='finish' for x in actions):raise ValueError('pure-chat tool use oracle')
 prefix=sum(x['name']!='finish' for x in actions[:case['supervise_from']])
 if case['requires_error'] and (prefix<1 or not any(failed(x) for x in results[:prefix])):raise ValueError('failure prefix oracle')
 if any(failed(x) for x in results[prefix:]):raise ValueError('post-prefix success oracle')
 return results,actions

def reconstruct(private,tools,enc):
 episode=PiNativeEpisode(tools,enc)
 turns=[]
 for message in private['source_messages']:
  if message['role']=='assistant':
   text=native_turn(message,tools);episode.accept_generated_turn(text);turns.append(text)
  else:episode.append_context(message)
 if not episode.finished or episode.text()!=private['native_record'] or episode.source_messages()!=private['source_messages']:raise ValueError('episode reconstruction')
 if len(turns)!=len(private['generations']):raise ValueError('generation coverage')
 for text,generation in zip(turns,private['generations']):
  if enc.encode_ordinary(text)!=generation['token_ids']:raise ValueError('generation token reconstruction')
 return turns

def expected_mask(text,turns,supervise_from,enc):
 ids=enc.encode_ordinary(text);raw=text.encode();bounds={0:0};position=0
 for i,token in enumerate(ids,1):position+=len(enc.decode_single_token_bytes(token));bounds[position]=i
 mask=bytearray(len(ids));cursor=0
 for i,turn in enumerate(turns):
  needle=('\n\nAssistant:\n'+turn).encode();start=raw.find(needle,cursor)
  if start<0:raise ValueError('assistant turn location')
  left=start+len(b'\n\nAssistant:\n');right=start+len(needle);cursor=right
  if left not in bounds or right not in bounds:raise ValueError('mask token boundary')
  if i>=supervise_from:mask[bounds[left]:bounds[right]]=b'\1'*(bounds[right]-bounds[left])
 return ids,bytes(mask)
def verify_public(private,root):
 raw=[bare(x) for x in message_ends(root/'pi/pi-events-private.jsonl')]
 if raw!=private['public_history']:raise ValueError('raw Pi transcript mismatch')
 public=private['public_history'];source=[x for x in private['source_messages'] if x['role']!='system']
 if len(public)!=len(source):raise ValueError('public/source coverage')
 pending=None
 for left,right in zip(public,source):
  if right['role'] in ('user','toolResult'):
   expected=right
   if right['role']=='user' and isinstance(right['content'],str):expected={'role':'user','content':[{'type':'text','text':right['content']}]}
   if left!=expected:raise ValueError('context message mismatch')
   if right['role']=='toolResult':
    if pending!=(right['toolCallId'],right['toolName']):raise ValueError('call result identity')
    pending=None
  else:
   semantic=semantic_turn(right,private['_tools']);blocks=left['content'];texts=[b['text'] for b in blocks if b['type']=='text'];calls=[b for b in blocks if b['type']=='toolCall']
   if semantic['name']=='finish':
    expected=([semantic['content']] if semantic['content'] is not None else [])+[semantic['arguments']['message']]
    if calls or texts!=expected:raise ValueError('finish projection')
   else:
    if len(calls)!=1 or calls[0]['name']!=semantic['name'] or calls[0]['arguments']!=semantic['arguments']:raise ValueError('tool projection')
    if texts!=([semantic['content']] if semantic['content'] is not None else []):raise ValueError('commentary projection')
    pending=(calls[0]['id'],calls[0]['name'])
 if pending is not None:raise ValueError('unresolved public call')

def verify_hybrid_case(case,turns,results,enc):
 for step,turn in zip(case['steps'][case['supervise_from']:],turns[case['supervise_from']:]):
  message=parse_turn(turn);analysis=message['reasoning_content']
  if step.get('analysis_dynamic'):
   if not isinstance(analysis,str) or not analysis.strip():raise ValueError('hybrid dynamic analysis missing')
   continue
  requires=step.get('analysis_requires') or []
  if not requires:continue
  if not isinstance(analysis,str) or not analysis.strip():raise ValueError('hybrid cohort requires non-empty analysis')
  if len(enc.encode_ordinary(analysis))>2048:raise ValueError('analysis cap')
  if any(x not in analysis for x in requires):raise ValueError('ungrounded analysis')

def verify_reasoning_case(case,turns,enc):
 for step,turn in zip(case['steps'][case['supervise_from']:],turns[case['supervise_from']:]):
  message=parse_turn(turn);analysis=message['reasoning_content'];requires=step.get('analysis_requires') or []
  if not requires:continue
  if not isinstance(analysis,str) or not analysis.strip():raise ValueError('reasoning cohort requires non-empty analysis')
  if len(enc.encode_ordinary(analysis))>2048:raise ValueError('analysis cap')
  if any(x not in analysis for x in requires):raise ValueError('ungrounded analysis')

def _v2_plan(plan):return plan.get('schema')=='emender-e97-hybrid-conversation-plan-v2'
def _audit_episodes(plan,cases,root,tools,enc,ids):
 """Verify a set of collected episode directories end-to-end; returns
 (accepted, rejections, targets, sequences)."""
 v2=_v2_plan(plan)
 accepted=[];rejections=[];targets=0;sequences=set()
 for name in ids:
  directory=root/name
  if (directory/'episode-private.json').exists():
   if name not in cases:raise ValueError('unknown episode directory')
   private=load_json(directory/'episode-private.json');private['_tools']=tools
   terminal=private['terminal']
   if not terminal['close_verified'] or not terminal['closed'] or terminal['pi_exit'] or terminal['bridge_failed'] or terminal['reason']!='finished':raise ValueError('Pi terminal')
   verify_public(private,directory)
   turns=reconstruct(private,tools,enc)
   results,actions=verify_case(cases[name],private,tools)
   if plan.get('mix')=='hybrid':verify_hybrid_case(cases[name],turns,results,enc)
   if v2:
    verify_hybrid_v2_case(cases[name],turns,results,enc)
    if 'clock_check' in private and audit_date_values(results[-1]['content'][0]['text'],private['clock_check']['fmt'])!=private['clock_check']['observed']:raise ValueError('clock observation record')
   want_ids,want_mask=expected_mask(private['native_record'],turns,private['supervise_from'],enc)
   if private['targets']!=sum(want_mask) or private['assistant_units']!=len(turns) or private['supervised_assistant_units']!=len(turns)-private['supervise_from']:raise ValueError('episode metadata')
   if hashlib.sha256(private['native_record'].encode()).hexdigest()!=private['record_sha256']:raise ValueError('episode record identity')
   sequence=hashlib.sha256(struct.pack('<%dI'%len(want_ids),*want_ids)+want_mask).hexdigest();sequences.add(sequence)
   targets+=private['targets'];private.pop('_tools');accepted.append(name)
  elif (directory/'rejection.json').exists():
   rejection=load_json(directory/'rejection.json')
   if set(rejection)!={'id','category','family','type','message','retried'} or rejection['retried'] or rejection['type']!='ValueError' or rejection['id']!=name:raise ValueError('rejection receipt')
   rejections.append(rejection)
  else:raise ValueError(f'neither verified nor rejected: {name}')
 if set(accepted)&{x['id'] for x in rejections} or len(sequences)!=len(accepted):raise ValueError('episode dedup')
 return accepted,rejections,targets,sequences
def audit_partial(args):
 """Audit a resumable (possibly still in-progress) collect directory: every
 episode present on disk is verified to the full v1/v2 bar; the receipt
 records verified/rejected/pending against the frozen plan."""
 root=args.root;plan=load_json(args.plan)
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 if sha(plan['tool_manifest'])!=plan['tool_manifest_sha256']:raise ValueError('tool manifest binding')
 verify_authority_files(plan)
 if plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates'] or plan['training_eligible'] or plan['packing_authorized']:raise ValueError('plan authorization')
 if (root/'summary.json').exists() and (root/'candidate-authority'/'manifest.json').exists():raise ValueError('completed collection: use the full audit')
 tools=load_json(plan['tool_manifest'])['model_visible_tools'];enc=tiktoken.get_encoding('p50k_base')
 cases={x['id']:x for x in plan['cases']}
 resume_path=root/'resume-state.json'
 if resume_path.exists():
  resume=load_json(resume_path)
  if resume.get('schema')!='emender-e97-hybrid-conversation-resume-v1' or resume.get('plan_sha256')!=args.plan_sha:raise ValueError('resume state plan binding')
 present={x.name for x in root.iterdir() if x.is_dir() and ((x/'episode-private.json').exists() or (x/'rejection.json').exists())}
 if not present<=set(cases):raise ValueError('episode directory outside plan')
 accepted,rejections,targets,sequences=_audit_episodes(plan,cases,root,tools,enc,sorted(present))
 pending=[x['id'] for x in plan['cases'] if x['id'] not in present]
 receipt={'schema':'emender-e97-pi-native-curriculum-audit-v1','partial':True,'status':'partial-qualified-candidates-not-admitted','attempted_records':plan['records'],'records':len(accepted),'rejected_records':len(rejections),'pending_records':len(pending),'automatic_retries':0,'assistant_targets':targets,'deduplicated_sequences':len(sequences),'plan_sha256':args.plan_sha,'checker_sha256':sha(__file__),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,'rejected':[x['id'] for x in rejections]}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_CURRICULUM_PARTIAL_AUDIT',len(accepted),len(rejections),len(pending),targets,sha(args.output))
def audit(args):
 root=args.root;plan=load_json(args.plan);manifest=load_json(root/'candidate-authority/manifest.json');summary=load_json(root/'summary.json');tools=load_json(plan['tool_manifest'])['model_visible_tools'];enc=tiktoken.get_encoding('p50k_base')
 if sha(args.plan)!=args.plan_sha or manifest['plan_sha256']!=args.plan_sha:raise ValueError('plan binding')
 if sha(plan['tool_manifest'])!=plan['tool_manifest_sha256']:raise ValueError('tool manifest binding')
 verify_authority_files(plan)
 if plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates'] or plan['training_eligible'] or plan['packing_authorized']:raise ValueError('plan authorization')
 if manifest['training_eligible'] or manifest['packing_authorized'] or manifest['optimizer_updates_authorized']:raise ValueError('candidate authorization')
 authority=root/'candidate-authority'
 for name,descriptor in manifest['outputs'].items():
  path=authority/descriptor['path']
  if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError('authority output identity')
 metadata=[json.loads(x) for x in (authority/'records.jsonl').read_text().splitlines()];index=(authority/'records.idx').read_bytes();tokens=(authority/'tokens.uint32.bin').read_bytes();mask=(authority/'assistant_mask.uint8.bin').read_bytes()
 if len(index)!=INDEX.size*len(metadata) or len(tokens)!=4*len(mask):raise ValueError('authority shape')
 rejection_path=root/'rejections.jsonl';rejections=[json.loads(x) for x in rejection_path.read_text().splitlines()] if rejection_path.exists() and rejection_path.stat().st_size else []
 rejected_ids=set()
 for rejection in rejections:
  if set(rejection)!={'id','category','family','type','message','retried'} or rejection['retried'] or rejection['type']!='ValueError' or rejection['id'] in rejected_ids or load_json(root/rejection['id']/'rejection.json')!=rejection:raise ValueError('rejection receipt')
  rejected_ids.add(rejection['id'])
 cases={x['id']:x for x in plan['cases']};accepted_ids={x['id'] for x in metadata}
 if accepted_ids&rejected_ids or accepted_ids|rejected_ids!=set(cases) or len(metadata)<plan.get('minimum_verified_records',plan['records']):raise ValueError('attempt coverage')
 offset=targets=errors=calls=repo=0;sequences=set()
 for i,row in enumerate(metadata):
  if row['id'] not in cases or sha(root/row['id']/'episode-private.json')!=row['episode_sha256']:raise ValueError('episode identity')
  private=load_json(root/row['id']/'episode-private.json');private['_tools']=tools
  terminal=private['terminal']
  if not terminal['close_verified'] or not terminal['closed'] or terminal['pi_exit'] or terminal['bridge_failed'] or terminal['reason']!='finished':raise ValueError('Pi terminal')
  verify_public(private,root/row['id']);turns=reconstruct(private,tools,enc);results,actions=verify_case(cases[row['id']],private,tools)
  if plan.get('mix')=='reasoning':verify_reasoning_case(cases[row['id']],turns,enc)
  if plan.get('mix')=='hybrid':verify_hybrid_case(cases[row['id']],turns,results,enc)
  if _v2_plan(plan):
   verify_hybrid_v2_case(cases[row['id']],turns,results,enc)
   if 'clock_check' in private and audit_date_values(results[-1]['content'][0]['text'],private['clock_check']['fmt'])!=private['clock_check']['observed']:raise ValueError('clock observation record')
  ids,want_mask=expected_mask(private['native_record'],turns,private['supervise_from'],enc);record=INDEX.unpack_from(index,i*INDEX.size);start,n,want_targets,split=record
  if start!=offset or n!=len(ids) or split!=0:raise ValueError('record index')
  token_bytes=struct.pack('<%dI'%len(ids),*ids);actual_tokens=tokens[4*start:4*(start+n)];actual_mask=mask[start:start+n]
  if actual_tokens!=token_bytes or actual_mask!=want_mask or want_targets!=sum(want_mask):raise ValueError('token/mask reconstruction')
  sequence=hashlib.sha256(token_bytes+want_mask).hexdigest();sequences.add(sequence)
  if sequence!=row['sequence_sha256'] or row['targets']!=want_targets or row['assistant_units']!=len(turns) or row['supervised_units']!=len(turns)-private['supervise_from'] or row['calls']!=len(results) or row['errors']!=sum(x['isError'] for x in results):raise ValueError('record metadata')
  offset+=n;targets+=want_targets;errors+=row['errors'];calls+=row['calls'];repo+=bool(row['repository_discovery']);private.pop('_tools')
 if len(sequences)!=len(metadata) or offset!=manifest['counts']['tokens'] or targets!=manifest['counts']['assistant_target_tokens']:raise ValueError('aggregate authority')
 if summary['records']!=len(metadata) or summary.get('attempted_records',plan['records'])!=plan['records'] or summary.get('rejected_records',0)!=len(rejections) or summary.get('automatic_retries',0)!=0 or summary['deduplicated_sequences']!=len(sequences) or summary['assistant_targets']!=targets or summary['native_calls']!=calls or summary['authentic_tool_errors']!=errors or summary['repository_discovery_records']!=repo or summary['authority_sha256']!=sha(authority/'manifest.json'):raise ValueError('summary reconstruction')
 receipt={'schema':'emender-e97-pi-native-curriculum-audit-v1','status':'qualified-candidates-not-admitted','attempted_records':plan['records'],'records':len(metadata),'rejected_records':len(rejections),'automatic_retries':0,'tokens':offset,'assistant_targets':targets,'native_calls':calls,'authentic_tool_errors':errors,'repository_discovery_records':repo,'deduplicated_sequences':len(sequences),'plan_sha256':args.plan_sha,'authority_sha256':sha(authority/'manifest.json'),'checker_sha256':sha(__file__),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_CURRICULUM_AUDIT',len(metadata),targets,sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true',help='audit a resumable/in-progress collect directory (no candidate-authority required)')
 a=p.parse_args();audit_partial(a) if a.partial else audit(a)
if __name__=='__main__':main()
