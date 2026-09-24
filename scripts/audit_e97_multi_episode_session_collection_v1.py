#!/usr/bin/env python3
"""Independent reconstruction audit for the multi-episode session collection v1.

Same bar as the hybrid-conversation v2 reconstruction audit, extended to the
session record semantics: every session is machine-replayed from the recorded
per-episode Pi transcripts (each episode's native text re-derived from its
source messages via the codec, byte-compared), every authored frame is
re-derived from the frozen plan spec plus the recorded observations (including
the session-only prior-observation dynamic finishes, re-implemented here
without importing the builder), the stitched record text is independently
re-stitched (protocol header + System stated once at the record start; every
later episode entered from its first user message), and the token/mask payloads
of the candidate authority are re-encoded from scratch and byte-compared.
The session oracles are re-checked: per-episode cumulative workspace state,
pure-chat episodes never call tools, and every recall episode's value must be
present in the recorded tool results of the episode it references.

Nothing here admits anything to training (training_eligible stays false).
"""
import argparse,hashlib,json,struct
from pathlib import Path
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import PiNativeEpisode,native_turn,parse_turn,semantic_turn
from scripts.audit_e97_pi_native_curriculum import verify_authority_files,audit_date_values,sha

INDEX=struct.Struct('<QQQB7x')
PLAN_SCHEMA='emender-e97-multi-episode-session-plan-v1'

def load_json(path):return json.loads(Path(path).read_text())

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
def action_sequence(source,tools):return [semantic_turn(x,tools) for x in source if x['role']=='assistant']

def verify_episode_public(private,episode_dir,tools):
 raw=[bare(x) for x in message_ends(episode_dir/'pi-events-private.jsonl')]
 if raw!=private['public_history']:raise ValueError('raw Pi transcript mismatch')
 public=private['public_history'];source=private['source_messages']
 if source and source[0].get('role')=='system':source=source[1:]
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
   semantic=semantic_turn(right,tools);blocks=left['content'];texts=[b['text'] for b in blocks if b['type']=='text'];calls=[b for b in blocks if b['type']=='toolCall']
   if semantic['name']=='finish':
    expected_blocks=([semantic['content']] if semantic['content'] is not None else [])+[semantic['arguments']['message']]
    if calls or texts!=expected_blocks:raise ValueError('finish projection')
   else:
    if len(calls)!=1 or calls[0]['name']!=semantic['name'] or calls[0]['arguments']!=semantic['arguments']:raise ValueError('tool projection')
    if texts!=([semantic['content']] if semantic['content'] is not None else []):raise ValueError('commentary projection')
    pending=(calls[0]['id'],calls[0]['name'])
 if pending is not None:raise ValueError('unresolved public call')

def reconstruct_episode(private,tools,enc):
 episode=PiNativeEpisode(tools,enc);turns=[]
 for message in private['source_messages']:
  if message['role']=='assistant':
   text=native_turn(message,tools);episode.accept_generated_turn(text);turns.append(text)
  else:episode.append_context(message)
 if not episode.finished or episode.text()!=private['native_record'] or episode.source_messages()!=private['source_messages']:raise ValueError('episode reconstruction')
 if len(turns)!=len(private['generations']):raise ValueError('generation coverage')
 for text,generation in zip(turns,private['generations']):
  if enc.encode_ordinary(text)!=generation['token_ids']:raise ValueError('generation token reconstruction')
 return turns

def _summary(values):return ' and '.join(f'{k} = {v}' for k,v in sorted(values.items()))

def verify_session_steps(spec_ep,turns,results_this,results_by_episode,enc):
 """Re-derive every authored frame of one episode from the plan spec plus the
 recorded observations — including the session-only prior-observation dynamic
 finish (independent re-implementation; never imports the builder)."""
 if len(turns)!=len(spec_ep['steps']):raise ValueError('session turn coverage')
 ri=0
 for step,turn in zip(spec_ep['steps'],turns):
  message=parse_turn(turn)
  if message['name']!=step['name']:raise ValueError('session step order')
  if step.get('dynamic')=='prior-observation':
   prior=results_by_episode[step['episode']]
   index=step.get('result_index',-1)
   if not prior or index>=len(prior):raise ValueError('prior observation missing')
   values=audit_date_values(prior[index],step['fmt'])
   if message['arguments']!={'message':step['template'].format(**values)}:raise ValueError('prior-observation finish message')
   if message['reasoning_content']!=step['analysis_template'].format(summary=_summary(values)):raise ValueError('prior-observation finish analysis')
   if message['content']!=step.get('commentary'):raise ValueError('prior-observation finish commentary')
  elif step.get('dynamic')=='observation-date':
   if not results_this:raise ValueError('session date finish without observation')
   values=audit_date_values(results_this[-1],step['fmt'])
   if message['arguments']!={'message':step['template'].format(**values)}:raise ValueError('session date finish message')
   if message['reasoning_content']!=step['analysis_template'].format(**values):raise ValueError('session date finish analysis')
   if message['content']!=step['commentary']:raise ValueError('session date finish commentary')
  elif step.get('dynamic')=='observation-exact':
   if not results_this:raise ValueError('session exact finish without observation')
   observation=results_this[-1].strip()
   if message['arguments']!={'message':observation}:raise ValueError('session exact finish message')
   if message['reasoning_content']!=f'The tool observed {observation}; the user-facing answer is exactly that observed value.':raise ValueError('session exact finish analysis')
   if message['content']!=f'The observed answer is {observation}.':raise ValueError('session exact finish commentary')
  else:
   if 'arguments' in step and message['arguments']!=step['arguments']:raise ValueError('session static arguments')
   if 'commentary' in step and message['content']!=step['commentary']:raise ValueError('session static commentary')
  analysis=message['reasoning_content']
  if step.get('analysis_dynamic') or step.get('dynamic') in ('observation-date','observation-exact','prior-observation'):
   if not isinstance(analysis,str) or not analysis.strip():raise ValueError('session dynamic analysis missing')
  elif isinstance(step.get('analysis'),str):
   if not analysis.strip():raise ValueError('session nonempty analysis')
   if len(enc.encode_ordinary(analysis))>2048:raise ValueError('analysis cap')
   for needle in step.get('analysis_requires') or []:
    if needle not in analysis:raise ValueError('session ungrounded analysis')
  if message['name']!='finish':ri+=1
 return ri

def stitch(episode_texts):
 record=episode_texts[0]
 for text in episode_texts[1:]:
  marker='\n\nUser:\n';idx=text.find(marker)
  if idx<0:raise ValueError('session episode without user message')
  record+=text[idx:]
 return record

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

def verify_session(case,private,tools,enc):
 """Full per-session verification: public transcripts, episode reconstruction,
 step re-derivation, session oracles; returns (turns, results_by_episode,
 stitched record text, calls, errors)."""
 if len(private['episodes'])!=len(case['episodes']) or private['episodes_count']!=len(case['episodes']):raise ValueError('episode coverage')
 results_by_episode=[];all_turns=[];episode_texts=[];calls=0;errors=0
 root=Path(private['_root'])
 for k,(spec_ep,ep_private) in enumerate(zip(case['episodes'],private['episodes'])):
  if ep_private['episode']!=k or ep_private['prompt']!=spec_ep['prompt']:raise ValueError('episode order/prompt')
  terminal=ep_private['terminal']
  if not terminal['close_verified'] or not terminal['closed'] or terminal['pi_exit'] or terminal['bridge_failed'] or terminal['reason']!='finished':raise ValueError('Pi terminal')
  verify_episode_public(ep_private,root/f'pi-ep{k:02d}',tools)
  turns=reconstruct_episode(ep_private,tools,enc)
  source=[m for m in ep_private['source_messages'] if m['role']!='system']
  results=[m for m in source if m['role']=='toolResult']
  actions=action_sequence(source,tools)
  if any(failed(r) for r in results):raise ValueError('unexpected tool error')
  if spec_ep.get('pure_chat') and any(a['name']!='finish' for a in actions):raise ValueError('pure-chat episode used tools')
  if ep_private['snapshot']!=spec_ep.get('expected_files',{}):raise ValueError('workspace oracle')
  verify_session_steps(spec_ep,turns,[r['content'][0]['text'] for r in results],results_by_episode,enc)
  if spec_ep.get('answer_must_be_observed'):
   if not any(str(spec_ep['expected_answer']) in r['content'][0]['text'] for r in results):raise ValueError('answer observation oracle')
  if spec_ep.get('recall') is not None:
   prior=results_by_episode[spec_ep['recall']['episode']]
   if not any(spec_ep['recall']['value'] in x for x in prior):raise ValueError('recall value not observed in the referenced episode')
   final=actions[-1]['arguments']['message']
   if spec_ep['recall']['value'] not in final:raise ValueError('recall finish does not restate the observed value')
  check=ep_private.get('clock_check')
  if check:
   if audit_date_values(results[-1]['content'][0]['text'],check['fmt'])!=check['observed']:raise ValueError('clock observation record')
  results_by_episode.append([r['content'][0]['text'] for r in results])
  all_turns.extend(turns);episode_texts.append(ep_private['native_record'])
  calls+=ep_private['calls'];errors+=ep_private['errors']
 record=stitch(episode_texts)
 if record!=private['record_text']:raise ValueError('stitched record reconstruction')
 if hashlib.sha256(record.encode()).hexdigest()!=private['record_sha256']:raise ValueError('record identity')
 return all_turns,record,calls,errors

def audit(args):
 root=args.root;plan=load_json(args.plan);manifest=load_json(root/'candidate-authority/manifest.json');summary=load_json(root/'summary.json')
 tools=load_json(plan['tool_manifest'])['model_visible_tools'];enc=tiktoken.get_encoding('p50k_base')
 if sha(args.plan)!=args.plan_sha or manifest['plan_sha256']!=args.plan_sha:raise ValueError('plan binding')
 if sha(plan['tool_manifest'])!=plan['tool_manifest_sha256']:raise ValueError('tool manifest binding')
 if plan['schema']!=PLAN_SCHEMA:raise ValueError('plan schema')
 verify_authority_files(plan)
 if plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates'] or plan['training_eligible'] or plan['packing_authorized']:raise ValueError('plan authorization')
 if manifest['training_eligible'] or manifest['packing_authorized'] or manifest['optimizer_updates_authorized']:raise ValueError('candidate authorization')
 authority=root/'candidate-authority'
 for name,descriptor in manifest['outputs'].items():
  path=authority/descriptor['path']
  if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError('authority output identity')
 metadata=[json.loads(x) for x in (authority/'records.jsonl').read_text().splitlines()]
 index=(authority/'records.idx').read_bytes();tokens=(authority/'tokens.uint32.bin').read_bytes();mask=(authority/'assistant_mask.uint8.bin').read_bytes()
 if len(index)!=INDEX.size*len(metadata) or len(tokens)!=4*len(mask):raise ValueError('authority shape')
 rejection_path=root/'rejections.jsonl'
 rejections=[json.loads(x) for x in rejection_path.read_text().splitlines()] if rejection_path.exists() and rejection_path.stat().st_size else []
 rejected_ids=set()
 for rejection in rejections:
  if set(rejection)!={'id','category','family','type','message','retried'} or rejection['retried'] or rejection['id'] in rejected_ids or load_json(root/rejection['id']/'rejection.json')!=rejection:raise ValueError('rejection receipt')
  rejected_ids.add(rejection['id'])
 cases={x['id']:x for x in plan['cases']};accepted_ids={x['id'] for x in metadata}
 if accepted_ids&rejected_ids or accepted_ids|rejected_ids!=set(cases) or len(metadata)<plan.get('minimum_verified_records',plan['records']):raise ValueError('attempt coverage')
 offset=targets=errors=calls=episodes=0;sequences=set()
 for i,row in enumerate(metadata):
  if row['id'] not in cases or sha(root/row['id']/'session-private.json')!=row['episode_sha256']:raise ValueError('session identity')
  private=load_json(root/row['id']/'session-private.json');private['_tools']=tools;private['_root']=str(root/row['id'])
  turns,record,ep_calls,ep_errors=verify_session(cases[row['id']],private,tools,enc)
  ids,want_mask=expected_mask(record,turns,private['supervise_from'],enc)
  packed=INDEX.unpack_from(index,i*INDEX.size);start,n,want_targets,split=packed
  if start!=offset or n!=len(ids) or split!=0:raise ValueError('record index')
  token_bytes=struct.pack('<%dI'%len(ids),*ids)
  if tokens[4*start:4*(start+n)]!=token_bytes or mask[start:start+n]!=want_mask or want_targets!=sum(want_mask):raise ValueError('token/mask reconstruction')
  sequence=hashlib.sha256(token_bytes+want_mask).hexdigest();sequences.add(sequence)
  if (sequence!=row['sequence_sha256'] or row['targets']!=want_targets or row['assistant_units']!=len(turns)
      or row['supervised_units']!=len(turns)-private['supervise_from'] or row['calls']!=ep_calls
      or row['errors']!=ep_errors or row['episodes']!=private['episodes_count']):raise ValueError('record metadata')
  offset+=n;targets+=want_targets;errors+=ep_errors;calls+=ep_calls;episodes+=private['episodes_count']
  private.pop('_tools');private.pop('_root')
 if len(sequences)!=len(metadata) or offset!=manifest['counts']['tokens'] or targets!=manifest['counts']['assistant_target_tokens']:raise ValueError('aggregate authority')
 if (summary['records']!=len(metadata) or summary.get('attempted_records',plan['records'])!=plan['records']
  or summary.get('rejected_records',0)!=len(rejections) or summary.get('automatic_retries',0)!=0
  or summary['deduplicated_sequences']!=len(sequences) or summary['assistant_targets']!=targets
  or summary['native_calls']!=calls or summary['authentic_tool_errors']!=errors
  or summary['episodes']!=episodes or summary['repository_discovery_records']!=0
  or summary['authority_sha256']!=sha(authority/'manifest.json')):raise ValueError('summary reconstruction')
 receipt={'schema':'emender-e97-multi-episode-session-audit-v1','status':'qualified-candidates-not-admitted','attempted_records':plan['records'],'records':len(metadata),'episodes':episodes,'rejected_records':len(rejections),'automatic_retries':0,'tokens':offset,'assistant_targets':targets,'native_calls':calls,'authentic_tool_errors':errors,'repository_discovery_records':0,'deduplicated_sequences':len(sequences),'plan_sha256':args.plan_sha,'authority_sha256':sha(authority/'manifest.json'),'checker_sha256':sha(__file__),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print('MULTI_EPISODE_SESSION_AUDIT',len(metadata),episodes,targets,sha(args.output))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();audit(a)
if __name__=='__main__':main()
