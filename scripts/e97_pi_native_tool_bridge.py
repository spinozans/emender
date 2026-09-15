"""Owner-side bridge from E97 Pi-native frames to real Pi tool execution."""
from copy import deepcopy
import hashlib,secrets,time
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_bridge import BridgeStopped
from scripts.e97_pi_native_codec import PiNativeEpisode,semantic_turn

PROVIDER='e97-pi-native';MODEL='e97-4b-pi-native';API='e97-pi-native-unix-v1'
PUBLIC_ERROR='Pi-native transport stopped; inspect private owner receipt.'
MODEL_STOP_REASONS=frozenset({'turn_budget','episode_generation_budget','episode_deadline','context_budget',
 'generation_budget','invalid_opening','invalid_frame','empty','separator_before_valid_turn','invalid_native_turn'})

def project_messages(messages):
 if not isinstance(messages,list):raise BridgeStopped('invalid_pi_history')
 result=[]
 for raw in messages:
  m=deepcopy(raw);role=m.get('role')
  if role=='user':
   c=m.get('content');c=[{'type':'text','text':c}] if isinstance(c,str) else c
   if not isinstance(c,list) or any(set(b)!={'type','text'} or b['type']!='text' or not isinstance(b['text'],str) for b in c):raise BridgeStopped('unsupported_user')
   result.append({'role':'user','content':c})
  elif role=='assistant':
   c=m.get('content')
   if not isinstance(c,list):raise BridgeStopped('unsupported_assistant')
   for b in c:
    keys={'type','text'} if b.get('type')=='text' else {'type','id','name','arguments'}
    if b.get('type') not in ('text','toolCall') or set(b)!=keys:raise BridgeStopped('unsupported_assistant_block')
   result.append({'role':'assistant','content':c})
  elif role=='toolResult':
   if type(m.get('isError')) is not bool:raise BridgeStopped('invalid_tool_error')
   c=m.get('content')
   if not isinstance(c,list) or any(set(b)!={'type','text'} or b.get('type')!='text' or not isinstance(b.get('text'),str) for b in c):raise BridgeStopped('unsupported_tool_content')
   result.append({k:m[k] for k in ('role','toolCallId','toolName','content','isError')})
  else:raise BridgeStopped('unsupported_history_role')
 return result

class NativePiToolBridge:
 def __init__(self,panel,prompt,encoding,generate):
  self.panel=deepcopy(panel);self.tools=deepcopy(panel['tools']);self.encoding=encoding;self.generate=generate
  self.episode=PiNativeEpisode(self.tools,encoding);self.episode.append_context(panel.get('system_message',{'role':'system','content':panel['system']}));self.episode.append_context({'role':'user','content':prompt})
  self.history=[{'role':'user','content':[{'type':'text','text':prompt}]}];self.pending=None;self.generations=[];self.tokens=0;self.final=None;self.reason=None;self.failed=False;self.closed=False;self.close_verified=False;self.deadline=time.monotonic()+panel['episode_seconds'];self.nonce=secrets.token_hex(16)
 def stop(self,reason):self.failed=True;self.reason=reason;raise BridgeStopped(reason)
 def _ingest(self,messages):
  projected=project_messages(messages)
  if self.pending is None:
   if compact(projected)!=compact(self.history):self.stop('pi_history_mismatch')
   return
  if len(projected)!=len(self.history)+1 or compact(projected[:-1])!=compact(self.history):self.stop('pi_history_mismatch')
  result=projected[-1];expected=self.pending
  if result['role']!='toolResult' or result['toolCallId']!=expected['id'] or result['toolName']!=expected['name']:self.stop('pi_tool_result_mismatch')
  self.episode.append_context(result);self.history.append(result);self.pending=None
 def next(self,request):
  if self.failed or self.closed or self.final is not None:self.stop('bridge_not_active')
  if request.get('systemPrompt')!=self.panel['system'] or request.get('model')!=MODEL or request.get('provider')!=PROVIDER or compact(request.get('tools'))!=compact(self.tools):self.stop('pi_contract_mismatch')
  self._ingest(request.get('messages'));internal=0
  while True:
   if len(self.generations)>=self.panel['max_turns']:self.stop('turn_budget')
   budget=min(self.panel['generation_budget'],self.panel['episode_generation_budget']-self.tokens)
   if budget<=0:self.stop('episode_generation_budget')
   if time.monotonic()>=self.deadline:self.stop('episode_deadline')
   prompt=self.episode.prompt();text,ids,reason=self.generate(prompt,budget,self.deadline);self.generations.append({'turn':len(self.generations),'token_ids':ids,'reason':reason,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()});self.tokens+=len(ids)
   if text is None:self.stop(reason)
   turn=self.episode.accept_generated_turn(text);native=semantic_turn(turn.source_message(),self.tools)
   if native['name']=='think':
    self.episode.resolve_think();internal+=1
    if internal>4:self.stop('internal_think_bound')
    continue
   blocks=[]
   if native['content'] is not None:blocks.append({'type':'text','text':native['content']})
   if native['name']=='finish':
    self.final=native['arguments']['message'];self.reason='finished';blocks.append({'type':'text','text':self.final});message={'role':'assistant','content':blocks};self.history.append(deepcopy(message));return {'message':message,'stop_reason':'stop','input_tokens':len(self.encoding.encode_ordinary(prompt)),'output_tokens':len(ids)}
   ref=f'{self.nonce}-{len(self.generations)}';call={'type':'toolCall','id':ref,'name':native['name'],'arguments':deepcopy(native['arguments'])};blocks.append(call);message={'role':'assistant','content':blocks};self.history.append(deepcopy(message));self.pending=deepcopy(call)
   return {'message':message,'stop_reason':'toolUse','input_tokens':len(self.encoding.encode_ordinary(prompt)),'output_tokens':len(ids)}
 def close(self,messages):
  if self.closed:raise BridgeStopped('duplicate_close')
  try:
   if not self.failed:
    self._ingest(messages)
    if self.pending is not None or self.reason!='finished':self.stop('pi_ended_before_finish')
    self.close_verified=True
  finally:self.closed=True
  return {'closed':True,'verified':self.close_verified}

def verify_model_failure(bridge,messages):
 if not bridge.failed or bridge.reason not in MODEL_STOP_REASONS or bridge.final is not None or bridge.pending is not None or not bridge.closed:
  raise ValueError('not_verified_model_failure')
 if not messages:raise ValueError('missing_pi_error')
 error=messages[-1]
 if (error.get('role')!='assistant' or error.get('content')!=[] or error.get('stopReason')!='error' or
  error.get('provider')!=PROVIDER or error.get('model')!=MODEL or error.get('api')!=API or error.get('errorMessage')!=PUBLIC_ERROR):
  raise ValueError('unexpected_pi_error')
 if compact(project_messages(messages[:-1]))!=compact(bridge.history):raise ValueError('failed_history_changed')
 return True
