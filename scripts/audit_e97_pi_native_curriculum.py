#!/usr/bin/env python3
"""Independent reconstruction audit for non-admitted Pi-native curriculum candidates."""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import PiNativeEpisode,native_turn,semantic_turn
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
 return result['isError'] or text.startswith(('error:','tool error:')) or any(x in text for x in ('exit code: 1','command exited with code 1','no matches found'))
def observation_final(text):return ('Returned web evidence: '+' '.join(text.split())[:500]).strip()
def action_sequence(source,tools):return [semantic_turn(x,tools) for x in source if x['role']=='assistant']
def verify_case(case,private,tools):
 source=private['source_messages'];actions=action_sequence(source,tools)
 if [x['name'] for x in actions]!=[x['name'] for x in case['steps']]:raise ValueError('planned action sequence')
 results=[x for x in source if x['role']=='toolResult'];ri=0
 for index,(planned,actual) in enumerate(zip(case['steps'],actions)):
  if 'dynamic' not in planned:
   if actual['arguments']!=planned['arguments']:raise ValueError('static action arguments')
  elif planned['dynamic']=='observation':
   if actual['arguments']!={'message':observation_final(results[ri-1]['content'][0]['text'])}:raise ValueError('observation final grounding')
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

def audit(args):
 root=args.root;plan=load_json(args.plan);manifest=load_json(root/'candidate-authority/manifest.json');summary=load_json(root/'summary.json');tools=load_json(plan['tool_manifest'])['model_visible_tools'];enc=tiktoken.get_encoding('p50k_base')
 if sha(args.plan)!=args.plan_sha or manifest['plan_sha256']!=args.plan_sha:raise ValueError('plan binding')
 if sha(plan['tool_manifest'])!=plan['tool_manifest_sha256']:raise ValueError('tool manifest binding')
 verify_authority_files(plan)
 if plan['model_generations'] or plan['optimizer_updates'] or plan['training_eligible'] or plan['packing_authorized']:raise ValueError('plan authorization')
 if manifest['training_eligible'] or manifest['packing_authorized'] or manifest['optimizer_updates_authorized']:raise ValueError('candidate authorization')
 authority=root/'candidate-authority'
 for name,descriptor in manifest['outputs'].items():
  path=authority/descriptor['path']
  if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError('authority output identity')
 metadata=[json.loads(x) for x in (authority/'records.jsonl').read_text().splitlines()];index=(authority/'records.idx').read_bytes();tokens=(authority/'tokens.uint32.bin').read_bytes();mask=(authority/'assistant_mask.uint8.bin').read_bytes()
 if len(index)!=INDEX.size*len(metadata) or len(tokens)!=4*len(mask):raise ValueError('authority shape')
 cases={x['id']:x for x in plan['cases']};offset=targets=errors=calls=repo=0;sequences=set()
 for i,row in enumerate(metadata):
  if row['id'] not in cases or sha(root/row['id']/'episode-private.json')!=row['episode_sha256']:raise ValueError('episode identity')
  private=load_json(root/row['id']/'episode-private.json');private['_tools']=tools
  terminal=private['terminal']
  if not terminal['close_verified'] or not terminal['closed'] or terminal['pi_exit'] or terminal['bridge_failed'] or terminal['reason']!='finished':raise ValueError('Pi terminal')
  verify_public(private,root/row['id']);turns=reconstruct(private,tools,enc);results,actions=verify_case(cases[row['id']],private,tools)
  ids,want_mask=expected_mask(private['native_record'],turns,private['supervise_from'],enc);record=INDEX.unpack_from(index,i*INDEX.size);start,n,want_targets,split=record
  if start!=offset or n!=len(ids) or split!=0:raise ValueError('record index')
  token_bytes=struct.pack('<%dI'%len(ids),*ids);actual_tokens=tokens[4*start:4*(start+n)];actual_mask=mask[start:start+n]
  if actual_tokens!=token_bytes or actual_mask!=want_mask or want_targets!=sum(want_mask):raise ValueError('token/mask reconstruction')
  sequence=hashlib.sha256(token_bytes+want_mask).hexdigest();sequences.add(sequence)
  if sequence!=row['sequence_sha256'] or row['targets']!=want_targets or row['assistant_units']!=len(turns) or row['supervised_units']!=len(turns)-private['supervise_from'] or row['calls']!=len(results) or row['errors']!=sum(x['isError'] for x in results):raise ValueError('record metadata')
  offset+=n;targets+=want_targets;errors+=row['errors'];calls+=row['calls'];repo+=bool(row['repository_discovery']);private.pop('_tools')
 if len(sequences)!=len(metadata) or offset!=manifest['counts']['tokens'] or targets!=manifest['counts']['assistant_target_tokens']:raise ValueError('aggregate authority')
 if summary['records']!=len(metadata) or summary['deduplicated_sequences']!=len(sequences) or summary['assistant_targets']!=targets or summary['native_calls']!=calls or summary['authentic_tool_errors']!=errors or summary['repository_discovery_records']!=repo or summary['authority_sha256']!=sha(authority/'manifest.json'):raise ValueError('summary reconstruction')
 receipt={'schema':'emender-e97-pi-native-curriculum-audit-v1','status':'qualified-candidates-not-admitted','records':len(metadata),'tokens':offset,'assistant_targets':targets,'native_calls':calls,'authentic_tool_errors':errors,'repository_discovery_records':repo,'deduplicated_sequences':len(sequences),'plan_sha256':args.plan_sha,'authority_sha256':sha(authority/'manifest.json'),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_CURRICULUM_AUDIT',len(metadata),targets,sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
