#!/usr/bin/env python3
"""Multi-episode session collection v1 (freeze + collect through real Pi).

Mission: the operator's REPL finding — deployment (REPL sessions, and the
planned state-forking demos) carries the recurrent state ACROSS episode
boundaries (finish -> new episode in the same state), while every record the
model has trained on resets state at every record. The model degrades when
continuing from accumulated state. This collection authors the missing
competence: MULTI-EPISODE SESSION RECORDS — one record = several short
episodes in sequence, the protocol header + system message rendered ONCE at
the record start, every subsequent episode entered mid-record with the
recurrent state carrying (finish -> new user message -> ...), the model
supervised on staying coherent when continuing from accumulated state:
re-observing accumulated workspace state when freshness matters, and
answering from the session's already-observed facts when it does not.

Machinery: the proven hybrid-conversation v2 pipeline, unchanged in its
verification bar. Every episode of every session is executed end-to-end
through real Pi (scripts/e97_pi_native_tool_bridge.py +
scripts/e97_pi_native_tool_transport.py), every tool result is a real
observation in the session's persistent workspace (the workspace carries
across the record's episodes; that accumulated state is the point), every
workspace outcome is oracle-checked per episode, and nothing is admitted to
training. The v2 pipeline's authoring guards apply verbatim
(scripts.build_e97_hybrid_conversation_collection_v2.check_text: zero
protected-panel or panel-literal content, forbidden numbers/words) — imported,
not re-implemented, so the guard cannot drift.

Record semantics (the training unit): the record text is the first episode's
native text (Protocol header + System + User + turns ... finish) followed by
each later episode's native text FROM ITS FIRST USER MESSAGE ON — the header
and System blocks of episodes 2..N are dropped, so the protocol preamble is
stated once per record and every episode boundary is crossed INSIDE the
record. The assistant mask supervises every episode's authored assistant
turns. The record is reset at its start only (reset_before at the record
boundary via the boundary-aware packer) — the state carries across every
in-record episode boundary, which is exactly the deployment distribution
(the REPL keeps one prefix state for the protocol preamble and then carries
state across turns).

Commands:
  freeze   author the full session plan (deterministic phrasing banks; zero
           protected-panel or panel-literal content) and publish plan-private.json
  collect  execute every session through real Pi with checkpoint/resume:
           already verified sessions are re-encoded from their
           session-private.json, rejections are kept, interrupted runs resume
           where they left off. Writes candidate-authority + summary only when
           the full plan is attempted and the minimum verified count is reached.
"""
import argparse,hashlib,json,os,shutil,signal,subprocess,threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import tiktoken

from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.eval_e97_native_execution import publish,sha
from scripts.build_e97_pi_native_curriculum import (SYSTEM,MANIFEST_SHA,EXTENSIONS,grounded,action,finish,frame,encode_candidate,write_authority,last_result)
from scripts.build_e97_hybrid_conversation_collection_v2 import (
 PROVIDER_V2,check_text,opaque,pick,register_of,seam_prompt,
 GREETINGS,GREETING_CONTEXT,GREETING_REPLIES,GREETING_ANALYSES,
 SMALLTALK_USER,SMALLTALK_REPLIES,SMALLTALK_ANALYSES,
 PRE_TOOL_COMMENTARY,MID_CONVERSATIONAL_COMMENTARY,FINISH_CONVERSATIONAL_COMMENTARY,
 DATE_COMMANDS,DATE_USER,DATE_DECOR,DATE_TEMPLATES,DATE_ANALYSES,DATE_COMMENTARIES,DATE_FINISH_COMMENTARIES,
 READ_FACT_ANALYSES,COUNT_ANALYSES,DERIVE_ANALYSES,WRITE_ANALYSES,EDIT_ANALYSES,
 parse_date_observation,check_clock,utc_family,weekday_now,resolve_dynamic_v2,date_command_step,
 step_finish_static)

SEED='e97-multi-episode-session-v1-20260923'
PLAN_SCHEMA='emender-e97-multi-episode-session-plan-v1'
SUMMARY_SCHEMA='emender-e97-multi-episode-session-summary-v1'
RESUME_SCHEMA='emender-e97-multi-episode-session-resume-v1'
DATE_FMTS=('bare','A','A-ymd','u-ymd','u-HM','A-BdY')

def session_id(family,i):return f'pi-native-session1-{family}-{i:05d}-{opaque(f"{SEED}:{family}:{i}",8)}'

# ------------------------------------------------------- session-only banks
# Cross-episode follow-ups: the user continues the SAME session, so the
# follow-up must read as a continuation ("earlier", "again", "still") — the
# conversational signal that the answer lives in the accumulated state.
RECALL_OPENERS=(
 "Quick follow-up — no need to run anything: ",
 "Recall check, straight from memory: ",
 "Without running anything again, ",
 "Staying in this session — from what we just did: ",
 "You still have that in context, right? ",
 "No tools this time — ",
 "From our session so far: ",
 "From memory, without checking again: ")
RECALL_ANALYSES=(
 "The {what} was already observed earlier in this session — the tool result is part of the conversation state the session carries — so a new observation would be redundant. The earlier observation recorded {value}, and the plain-text answer restates exactly that observed value.",
 "This is a continuation of the same session: the {what} was observed a moment ago and that observation is still in context. No tool is needed; the answer is the earlier observed {value}, restated in plain text.",
 "The session already holds the observation this question asks about — the {what} was grounded when it was first checked. Re-running the tool would add nothing, so the correct continuation is the plain-text restatement of the observed {value}.")
RECALL_FINISHES=(
 "It was {value}.","From earlier: {value}.","That was {value}.","Still {value}, same as we saw.","The {what} was {value}.")
FOLLOWUP_ANALYSES=(
 "The user is continuing the same session with a new request. The workspace state accumulated across the earlier episodes is still there, so the observation this episode needs is available — but the correct move is still to observe it for real rather than assume, because the question is about current file content.",
 "This is a new episode in the same session: state accumulated earlier is present, and the right behavior is to check the live workspace again — the file may have changed since it was last seen, and the observation grounds the answer.",
 "A follow-up task in the same session. The accumulated workspace makes the check cheap, but the answer must still come from a real observation of the current state, not from memory of the earlier episodes.")
SESSION_CONTINUATIONS=(
 "Great — sticking with you in this session.","Same session, next thing:","Still here — what's next?",
 "Continuing on then.","Happy to keep going.","Right, back at it.","Onward in this session.","Next up then:")

def session_base(family,files,episodes):
 for path,content in files.items():
  if path.startswith('/') or '..' in path.split('/'):raise ValueError('fixture path')
  check_text(path);check_text(content)
 for ep in episodes:
  check_text(ep['prompt'])
  for step in ep['steps']:
   if isinstance(step.get('arguments'),dict):check_text(json.dumps(step['arguments']))
  for path in ep.get('expected_files',{}):
   if path.startswith('/') or '..' in path.split('/'):raise ValueError('expected path')
 return dict(id=None,category='session',family=family,files=files,episodes=episodes,
  supervise_from=0,requires_error=False,repository_discovery=False,session_reset_policy='record-start-only')

def recall_episode(i,what,value,prior_episode,expected_files,extra=''):
 """A pure-chat episode whose answer is a fact OBSERVED in an earlier episode
 of the same session. The oracle: the finish must contain the value AND the
 value must appear in the recorded tool results of the earlier episode."""
 prompt=pick(RECALL_OPENERS,i)+f'what was the {what} again?'+extra
 analysis=pick(RECALL_ANALYSES,i).format(what=what,value=value)
 message=pick(RECALL_FINISHES,i).format(what=what,value=value)
 check_text(message)
 return dict(prompt=prompt,steps=[step_finish_static(message,analysis,None,[str(value)])],
  pure_chat=True,expected_files=expected_files,date_fmt=None,
  recall={'value':str(value),'episode':prior_episode,'value_kind':'observed'})

def task_episode(prompt,steps,expected_files,*,answer=None,date_fmt=None,pure_chat=False):
 check_text(answer) if answer is not None else None
 return dict(prompt=prompt,steps=steps,pure_chat=pure_chat,expected_files=expected_files,
  answer_must_be_observed=answer is not None,expected_answer=answer,date_fmt=date_fmt,
  answer_is_observation=False,recall=None)

def chat_episode(i,bank,expected_files):
 """A pure conversational episode reusing the v2 greeting/smalltalk banks."""
 text,kind=bank[i%len(bank)]
 prompt=text
 if bank is GREETINGS:
  if i>=len(bank):
   prompt=prompt+' '+GREETING_CONTEXT[(i-len(bank))//len(GREETING_CONTEXT)]
  reply=pick(GREETING_REPLIES[register_of(i)],i//3);analysis=pick(GREETING_ANALYSES,i)
 else:
  reply=pick(SMALLTALK_REPLIES[kind],i);analysis=pick(SMALLTALK_ANALYSES,i)
 check_text(prompt);check_text(reply)
 return dict(prompt=prompt,steps=[step_finish_static(reply,analysis,None,[])],pure_chat=True,
  expected_files=expected_files,date_fmt=None,recall=None)

def greeting_ep(i,files):return chat_episode(i,GREETINGS,files)
def smalltalk_ep(i,files):return chat_episode(i,SMALLTALK_USER,files)

# -------------------------------------------------------------- family F1
def lifecycle_session(i):
 """Write-then-reference across episodes: read a fixture fact, write a note
 carrying it forward, recall it from session state, then re-observe the note."""
 cid=opaque(f'{SEED}:lifecycle:{i}',16);tag=opaque(cid,10)
 kinds=('port','release','version','owner','token')
 kind=kinds[i%len(kinds)]
 path={'port':f'config/service_{tag}.ini','release':f'docs/notes_{tag}.txt','version':f'state/build_{tag}.txt','owner':f'data/owner_{tag}.txt','token':f'data/token_{tag}.txt'}[kind]
 value={'port':str(10000+(i*757)%48000),'release':'REL_'+opaque(cid,14),'version':'v'+opaque(cid,6),'owner':opaque(cid,8),'token':'TKN_'+opaque(cid,10)}[kind]
 key={'port':'port','release':'release','version':'version','owner':'owner','token':'token'}[kind]
 note=f'out/note_{tag}.txt';note_content=f'{key} = {value}\n'
 files={path:f'{key} = {value}\n'}
 conversational=i%2==0
 ep1_prompt=(pick((f'Hey, can you check the config at {path} and tell me what {key} it declares?',
  f'Could you look at {path} and tell me the {key}?',
  f'What {key} is recorded in {path}? Just the value, please.',
  f'Look at {path} and report the {key} it records.'),i))
 ep1=task_episode(ep1_prompt,[grounded(action('read',{'path':path}),
  pick(READ_FACT_ANALYSES,i).format(path=path,key=key,value=value),pick(PRE_TOOL_COMMENTARY,i),[value]),
  step_finish_static((f'There you go — the {key} is {value}.' if conversational else value),
  f'The read observed {key} = {value}; the plain-text answer is exactly that observed value.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else f'The {key} is {value}.',[value])],
  files,answer=value)
 ep2_prompt=(pick((f'Now write a one-line note at {note} recording that {key} = {value}, read it back, and confirm with done.',
  f'Please create {note} containing exactly {note_content.strip()}, read it back, and finish with done.',
  f'Next: save that into {note} as a single line, verify with a read, and confirm with done.'),i))
 ep2_files=dict(files);ep2_files[note]=note_content
 ep2=task_episode(ep2_prompt,[grounded(action('write',{'path':note,'content':note_content}),
  pick(WRITE_ANALYSES,i).format(path=note,content=note_content.strip()),pick(PRE_TOOL_COMMENTARY,i),[]),
  grounded(action('read',{'path':note}),'Reading the note back to verify the write landed exactly as requested.','Verifying the note.',[]),
  step_finish_static(('done — written and verified.' if conversational else 'done'),
  f'The read-back observed {note_content.strip()}, which confirms the write; the outcome is verified, so the plain-text answer is done.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else 'Verified.',[value])],
  ep2_files,answer=value)
 ep3=recall_episode(i,f'{key} recorded in {path}',value,0,ep2_files)
 episodes=[ep1,ep2,ep3]
 if i%3!=2:  # 2/3 of records close by re-observing the accumulated note
  ep4_prompt=pick((f'Last one in this session: read {note} back and tell me what it records.',
   f'Fresh check — what does {note} say now?',
   f'Verify the session state for me: read {note} and give me the {key}.',
   f'Read {note} once more and tell me the {key} it holds.'),i)
  ep4=task_episode(ep4_prompt,[grounded(action('read',{'path':note}),
   pick(FOLLOWUP_ANALYSES,i),'Checking the note we wrote earlier in this session.',[]),
   step_finish_static((f'The note still records {key} = {value}.' if conversational else value),
   f'The read observed {note_content.strip()}; the plain-text answer is exactly that observed value.',
   pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else f'The note records {value}.',[value])],
   ep2_files,answer=value)
  episodes.append(ep4)
 case=session_base('session-file-lifecycle',files,episodes);case['id']=session_id('file-lifecycle',i);return case

# -------------------------------------------------------------- family F2
def chat_then_task_session(i):
 """A plain conversational episode, then a chat-opener tool task, then a
 recall from session state, then conversational close."""
 cid=opaque(f'{SEED}:chatthen:{i}',16);tag=opaque(cid,10)
 count=3+(i*7)%40
 path=f'notes/list_{tag}.txt'
 files={path:''.join(f'entry {j}: {opaque(cid+str(j),6)}\n' for j in range(1,count+1))}
 ep1=(greeting_ep(i,files) if i%2==0 else smalltalk_ep(i,files))
 ep2_prompt=seam_prompt(i,f'check how many lines are in {path}')+pick(('',' Take your time.',' Thanks!',' No rush.'),i)
 ep2=task_episode(ep2_prompt,[grounded(action('bash',{'command':f'wc -l < {path}'}),
  pick(COUNT_ANALYSES,i).format(path=path,count=count),pick(PRE_TOOL_COMMENTARY,i),[str(count)]),
  step_finish_static((f'There are {count} lines in {path}.' if i%2 else str(count)),
  f'The command observed {count} lines; the plain-text answer is exactly that observed count.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if i%2 else f'The file has {count} lines.',[str(count)])],
  files,answer=str(count))
 ep3=recall_episode(i,f'line count of {path}',count,1,files,
  extra=pick((' Just from the session.',' Same session, no re-run.',' You checked it a moment ago.',''),i))
 ep4=smalltalk_ep(i+1,files)
 ep4['prompt']=pick(('Thanks for the help in this session!','Appreciate it — good session so far.','Thanks! That was quick.','Great, thanks for sticking with it.'),i)
 check_text(ep4['prompt'])
 case=session_base('session-chat-then-task',files,[ep1,ep2,ep3,ep4]);case['id']=session_id('chat-then-task',i);return case

# -------------------------------------------------------------- family F3
def derive_accumulate_session(i):
 """Derive across episodes: observe operands + sum, recall the sum from
 session state, compute the double from the observed operands, recall it."""
 cid=opaque(f'{SEED}:derive:{i}',16);tag=opaque(cid,10);a=101+(i*17)%850;b=42+(i*23)%850
 while b in (157,284,441,731,992) or a in (157,284,441,731,992) or a+b in (157,284,441,731,992) or 2*(a+b) in (157,284,441,731,992):
  a=(a+13)%900+100;b=(b+29)%900+40
 total=a+b;doubled=2*total
 path=f'data/ledger_{tag}.txt';files={path:f'a = {a}\nb = {b}\n'}
 ep1_prompt=pick((f'Open {path}, add its two operands, and answer with only the sum in plain text.',
  f'Can you grab the two numbers in {path} and add them up? Answer with just the sum.',
  f'Add the operands recorded in {path} and tell me the sum.'),i)
 ep1=task_episode(ep1_prompt,[grounded(action('read',{'path':path}),
  f'The user asks for a derived fact; reading {path} observes the two operands before any arithmetic is trusted.',pick(PRE_TOOL_COMMENTARY,i),[f'ledger_{tag}']),
  grounded(action('bash',{'command':f"printf '%s\\n' $(({a}+{b}))"}),
  pick(DERIVE_ANALYSES,i).format(path=path,a=a,b=b,sum=total),'Adding the observed operands now.',[str(a),str(b)]),
  step_finish_static((f'The sum is {total}.' if i%2 else str(total)),
  f'The command observed {total}; the plain-text answer is exactly that observed sum.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if i%2 else f'The sum is {total}.',[str(total)])],
  files,answer=str(total))
 ep2=recall_episode(i,f'sum of the operands in {path}',total,0,files,
  extra=pick((' Just from this session.',' No re-run needed.',' Same session.',''),i))
 ep3_prompt=pick((f'Now double that sum with the shell and tell me the result.',
  f'Next in this session: compute twice the sum, using the shell, and give me the number.',
  f'Sticking with the same numbers — what is the sum times two? Compute it for real.'),i)
 ep3=task_episode(ep3_prompt,[grounded(action('bash',{'command':f"printf '%s\\n' $((2*{total}))"}),
  f'The operands and their sum were observed earlier in this session, but the doubled figure is a new derived fact, so the shell computes it from the observed operands and the result is observed directly rather than assumed. The earlier observation established the sum {total}; the command observed {doubled}.',
  pick(MID_CONVERSATIONAL_COMMENTARY,i),[str(total)]),
  step_finish_static((f'Doubled, it is {doubled}.' if i%2 else str(doubled)),
  f'The command observed {doubled}; the plain-text answer is exactly that observed value.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if i%2 else f'Twice the sum is {doubled}.',[str(doubled)])],
  files,answer=str(doubled))
 ep4=recall_episode(i,'doubled figure',doubled,2,files,
  extra=pick((' From memory, this session.',' No tools.',' Same session, just recall.',''),i))
 case=session_base('session-derive-accumulate',files,[ep1,ep2,ep3,ep4]);case['id']=session_id('derive-accumulate',i);return case

# -------------------------------------------------------------- family F4
def edit_chain_session(i):
 """Read, edit + verify, recall the edited value from session state, then
 re-observe the edited file."""
 cid=opaque(f'{SEED}:editchain:{i}',16);tag=opaque(cid,10);old='alpha';new='beta'
 path=f'config/mode_{tag}.txt';zone=opaque(cid,4)
 files={path:f'mode = {old}\nzone = {zone}\n'}
 edited=f'mode = {new}\nzone = {zone}\n'
 ep1_prompt=pick((f'What mode is {path} set to? Answer with the mode line.',
  f'Check {path} and tell me the current mode.',
  f'Read {path} and report the mode it declares.'),i)
 ep1=task_episode(ep1_prompt,[grounded(action('read',{'path':path}),
  pick(READ_FACT_ANALYSES,i).format(path=path,key='mode',value=f'mode = {old}'),pick(PRE_TOOL_COMMENTARY,i),[f'mode = {old}']),
  step_finish_static(f'The mode is {old}.',f'The read observed mode = {old}; the plain-text answer states that observed mode.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i),[old])],files,answer=f'mode = {old}')
 ep2_prompt=pick((f'Now change mode = {old} to mode = {new} in {path}, read it back, and confirm with done.',
  f'Edit {path}: mode = {old} becomes mode = {new}. Verify with a read and finish with done.',
  f'Flip the mode from {old} to {new} in {path}, verify the edit, and confirm.'),i)
 ep2_files=dict(files);ep2_files[path]=edited
 ep2=task_episode(ep2_prompt,[grounded(action('edit',{'path':path,'edits':[{'oldText':f'mode = {old}','newText':f'mode = {new}'}]}),
  pick(EDIT_ANALYSES,i).format(path=path,old=f'mode = {old}',new=f'mode = {new}'),pick(PRE_TOOL_COMMENTARY,i),[]),
  grounded(action('read',{'path':path}),'Reading the file back to verify the edit landed exactly as requested.','Verifying the edit.',[]),
  step_finish_static(('done — mode is '+new+' now.' if i%2 else 'done'),
  f'The read-back observed mode = {new}, which confirms the edit; the outcome is verified, so the plain-text answer is done.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if i%2 else 'Verified.',[f'mode = {new}'])],
  ep2_files,answer=f'mode = {new}')
 ep3=recall_episode(i,'mode now set in the config',new,1,ep2_files,
  extra=pick((' You just changed it.',' From the session.',' No re-read needed.',''),i))
 ep4_prompt=pick((f'One more check in this session: read {path} and confirm the mode.',
  f'Verify the final state — what mode does {path} hold now?',
  f'Last thing: fresh read of {path}; what mode is set?'),i)
 ep4=task_episode(ep4_prompt,[grounded(action('read',{'path':path}),
  pick(FOLLOWUP_ANALYSES,i),'Reading the config we edited earlier in this session.',[]),
  step_finish_static(f'The mode is {new} — confirmed.',f'The read observed mode = {new}; the plain-text answer states that observed mode.',
  pick(FINISH_CONVERSATIONAL_COMMENTARY,i),[new])],ep2_files,answer=f'mode = {new}')
 case=session_base('session-edit-chain',files,[ep1,ep2,ep3,ep4]);case['id']=session_id('edit-chain',i);return case

# -------------------------------------------------------------- family F5
DATE_RECALL_TEMPLATES={'bare':('It said {weekday}, {date}.','The clock said {weekday} — {date}.','Earlier it read {weekday}, {date}.'),
 'A':('It said {weekday}.','The clock said {weekday}.','Earlier it read {weekday}.'),
 'A-ymd':('It said {weekday}, {ymd}.','The clock said {weekday} — {ymd}.','Earlier it read {weekday}, {ymd}.'),
 'u-ymd':('It said {ymd} in UTC.','The clock said {ymd} UTC.','Earlier it read {ymd} in UTC.'),
 'u-HM':('It said {time} UTC.','The clock said {time} UTC.','Earlier it read {time} UTC.'),
 'A-BdY':('It said {weekday}, {date}.','The clock said {weekday} — {date}.','Earlier it read {weekday}, {date}.')}
DATE_RECALL_ANALYSES=(
 'The clock was observed earlier in this session — the date command output is part of the conversation state the session carries — so a new tool call would be redundant. The earlier observation showed {summary}, and the plain-text answer restates it.',
 'This continues the same session: the date was grounded by a real observation a moment ago and that observation is still in context. No tool is needed; the answer restates the observed values in plain text.',
 'The session already holds the clock observation this question asks about. Re-running the command would add nothing, so the correct continuation is the plain-text restatement of what was actually observed.')

def date_followup_session(i):
 """Observe the real clock, chat, then RESTATE the observed date from session
 state (dynamic finish resolved from the recorded prior observation), and
 for half the records observe the clock again in a second format."""
 fmt=DATE_FMTS[i%len(DATE_FMTS)];k=i//len(DATE_FMTS)
 phrases=DATE_USER[fmt]
 ep1_prompt=phrases[k] if k<len(phrases) else phrases[k%len(phrases)]+' '+DATE_DECOR[(k//len(phrases)-1)%len(DATE_DECOR)]
 template=pick(DATE_TEMPLATES[fmt],i);analysis=pick(DATE_ANALYSES[fmt],i);commentary=pick(DATE_FINISH_COMMENTARIES[fmt],i)
 ep1=task_episode(ep1_prompt,[date_command_step(fmt,i),
  {'name':'finish','dynamic':'observation-date','fmt':fmt,'template':template,'analysis_template':analysis,'commentary':commentary}],
  {},date_fmt=fmt)
 ep1['answer_is_observation']=False
 ep2=smalltalk_ep(i,{});ep2['prompt']=pick(('Neat. By the way, I might have more questions in a bit.',
  'Thanks! I may follow up on that.','Got it. Staying in this session for a moment.','Makes sense — sticking around for a follow-up.'),i)
 check_text(ep2['prompt'])
 recall_template=pick(DATE_RECALL_TEMPLATES[fmt],i)
 ep3=dict(prompt=pick(('Quick follow-up — no need to run anything: what did the clock say earlier, in this session?',
  'Recall check, straight from memory: that day/date question from a moment ago — what was the answer again?',
  'Without running anything again: restate what the clock told you earlier in this session.',
  'From our session so far: what did the date command show? No tools this time.'),i),
  steps=[{'name':'finish','dynamic':'prior-observation','episode':0,'fmt':fmt,'template':recall_template,'analysis_template':pick(DATE_RECALL_ANALYSES,i),'commentary':None}],
  pure_chat=True,expected_files={},date_fmt=None,recall=None)
 episodes=[ep1,ep2,ep3]
 if i%2==0:
  fmt2=DATE_FMTS[(i+3)%len(DATE_FMTS)]
  ep4_prompt=pick((f'Fresh check to close the session: {pick(DATE_USER[fmt2],i).lower()}',
   f'One more for the road — {pick(DATE_USER[fmt2],i).lower()}',
   f'Before we wrap: {pick(DATE_USER[fmt2],i).lower()}'),i)
  ep4=task_episode(ep4_prompt,[date_command_step(fmt2,i+1),
   {'name':'finish','dynamic':'observation-date','fmt':fmt2,'template':pick(DATE_TEMPLATES[fmt2],i),'analysis_template':pick(DATE_ANALYSES[fmt2],i),'commentary':pick(DATE_FINISH_COMMENTARIES[fmt2],i)}],
   {},date_fmt=fmt2)
  episodes.append(ep4)
 case=session_base('session-date-followup',{},episodes);case['id']=session_id('date-followup',i);return case

# ---------------------------------------------------------------- distribution
DISTRIBUTION=(
 ('session-file-lifecycle',300,lifecycle_session),
 ('session-chat-then-task',250,chat_then_task_session),
 ('session-derive-accumulate',250,derive_accumulate_session),
 ('session-edit-chain',150,edit_chain_session),
 ('session-date-followup',150,date_followup_session))

def episode_kind(ep):
 dyn=tuple(sorted({(s.get('dynamic'),s.get('fmt')) for s in ep['steps'] if isinstance(s,dict) and s.get('dynamic')},key=str))
 return (bool(ep.get('pure_chat')),ep.get('date_fmt'),dyn,tuple(s['name'] for s in ep['steps']))
def case_kind(case):
 return (case['family'],tuple(episode_kind(ep) for ep in case['episodes']))
def pilot_sample(cases,stride):
 """The v2 pilot-coverage rule (regression-tested there): the first case of
 every case kind, then the stride sample minus already-covered kinds."""
 extras=[];seen=set()
 for c in cases:
  k=case_kind(c)
  if k not in seen:extras.append(c);seen.add(k)
 if stride<=1:return list(cases)
 return extras+[c for c in cases[::stride] if case_kind(c) not in seen]

def authority_files():
 paths=[Path('scripts/build_e97_pi_native_curriculum.py'),
  Path('scripts/build_e97_hybrid_conversation_collection_v2.py'),Path(__file__),PROVIDER_V2,
  Path('configs/pi/e97-pi-native.ts'),Path('scripts/e97_pi_native_codec.py'),
  Path('scripts/e97_pi_native_tool_bridge.py'),Path('scripts/e97_pi_native_tool_transport.py'),*EXTENSIONS]
 return {str(p.resolve()):sha(p) for p in paths}

def make_all_cases():
 cases=[];distribution={}
 for family,count,fn in DISTRIBUTION:
  for j in range(count):
   case=fn(j)
   if case['family']!=family:raise ValueError(f'family mismatch {case["family"]}!={family}')
   cases.append(case)
  distribution[family]=count
 if len({c['id'] for c in cases})!=len(cases):raise ValueError('case identity')
 if len({json.dumps([ep['prompt'] for ep in c['episodes']],sort_keys=True) for c in cases})!=len(cases):raise ValueError('case prompt uniqueness')
 for c in cases:
  for ep in c['episodes']:
   for step in ep['steps']:
    if not isinstance(step,dict):raise ValueError('step shape')
    for needle in step.get('analysis_requires') or []:check_text(needle)
    if 'analysis_requires' in step and 'dynamic' not in step:
     if not step.get('analysis'):raise ValueError('missing analysis')
     for needle in step['analysis_requires']:
      if needle not in step['analysis']:raise ValueError(f'ungrounded authored analysis: {needle!r} in {c["id"]}')
    if step.get('name')=='finish' and 'dynamic' not in step and 'arguments' not in step:raise ValueError('static finish without arguments')
    check_text(step['arguments']['message']) if step.get('name')=='finish' and 'arguments' in step else None
  for ep in c['episodes']:
   if ep.get('recall') is not None:
    if not ep.get('pure_chat') or any(s['name']!='finish' for s in ep['steps']):raise ValueError('recall episode must be pure chat')
    if not 0<=ep['recall']['episode']<len(c['episodes']):raise ValueError('recall episode index')
    if ep['recall']['episode']==c['episodes'].index(ep):raise ValueError('recall references itself')
    prior=c['episodes'][ep['recall']['episode']]
    if prior.get('pure_chat'):raise ValueError('recall source must be an observed (tool) episode')
 return cases,distribution

def freeze(args):
 if sha(args.manifest)!=MANIFEST_SHA:raise ValueError('tool authority')
 if args.records!=sum(count for _,count,_ in DISTRIBUTION):raise ValueError('records must equal the frozen session distribution total')
 cases,distribution=make_all_cases()
 minimum=args.minimum_verified_records if args.minimum_verified_records is not None else args.records
 if not 1<=minimum<=args.records:raise ValueError('minimum verified records')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 plan={'schema':PLAN_SCHEMA,'seed':SEED,'records':args.records,'minimum_verified_records':minimum,'automatic_retry':False,
  'distribution':distribution,'system':SYSTEM,'tool_manifest':str(args.manifest.resolve()),'tool_manifest_sha256':sha(args.manifest),
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'authority_files':authority_files(),'pi_bin':str(args.pi_bin.resolve()),
  'pi_version':subprocess.check_output([args.pi_bin,'--version'],text=True).strip(),
  'cases':cases,'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False}
 publish(args.output/'plan-private.json',plan);print('SESSION_V1_PLAN',args.records,minimum,sha(args.output/'plan-private.json'),flush=True)

# ------------------------------------------------------------------ execution
def resolve_dynamic_session(spec,prior_observations):
 """Session-only dynamic action: a finish restating a fact OBSERVED in an
 earlier episode of the same record, resolved deterministically from the
 recorded prior observation so the independent audit can re-derive it."""
 if spec.get('dynamic')!='prior-observation':return resolve_dynamic_v2(spec,prior_observations)
 results=prior_observations[spec['episode']]
 index=spec.get('result_index',-1)
 if not results or index>=len(results):raise ValueError('prior-observation dynamic without a recorded earlier observation')
 values=parse_date_observation(results[index],spec['fmt'])
 message=spec['template'].format(**values)
 summary=' and '.join(f'{k} = {v}' for k,v in sorted(values.items()))
 analysis=spec['analysis_template'].format(summary=summary)
 if not message.strip() or not analysis.strip():raise ValueError('degenerate prior-observation finish')
 resolved=finish(message);resolved['analysis']=analysis;resolved['commentary']=spec.get('commentary')
 return resolved

def stitch_record(episode_texts):
 """One record, many episodes: the first episode's native text whole
 (Protocol header + System + User + turns), then every later episode's text
 from its FIRST user message on — header and System stated once per record,
 every episode boundary crossed inside the record."""
 if not episode_texts:raise ValueError('empty session')
 record=episode_texts[0]
 for k,text in enumerate(episode_texts[1:],1):
  marker='\n\nUser:\n'
  idx=text.find(marker)
  if idx<0:raise ValueError(f'episode {k} has no user message')
  record+=text[idx:]
 return record

def execute_session(case,panel,enc,root,pi_bin,manifest_path):
 workspace=root/'workspace';workspace.mkdir(parents=True)
 for name,text in case['files'].items():
  p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 prior_observations=[];episode_privates=[];all_generations=[];episode_texts=[]
 total_calls=0;total_errors=0
 for k,ep in enumerate(case['episodes']):
  fmt=ep.get('date_fmt');utc=utc_family(fmt) if fmt else False
  clock_before=weekday_now(utc) if fmt else None
  index=0;emitted=[];emitted_specs=[]
  def generate(prompt,budget,deadline):
   nonlocal index
   spec=ep['steps'][index]
   if 'dynamic' in spec:
    if spec.get('dynamic')=='prior-observation':
     spec=resolve_dynamic_session(spec,prior_observations)
    else:
     if last_result(bridge) is None:raise ValueError('dynamic action without observation')
     spec=resolve_dynamic_v2(spec,last_result(bridge)['content'][0]['text'])
   observed=last_result(bridge)['content'][0]['text'].strip() if last_result(bridge) else None
   comment=spec.get('commentary') if 'commentary' in spec else ('Using the verified observation.' if index and last_result(bridge) else None)
   analysis=spec.get('analysis')
   if analysis is not None:
    required=spec.get('analysis_requires') or []
    if not analysis.strip() or any(x not in analysis for x in required):raise ValueError('ungrounded analysis')
   text=frame(panel['tools'],spec['name'],spec['arguments'],comment,analysis);ids=enc.encode_ordinary(text)
   if len(ids)>budget:raise ValueError('authored turn budget')
   emitted.append(text);emitted_specs.append(spec);index+=1;return text,ids,'valid'
  bridge=NativePiToolBridge(panel,ep['prompt'],enc,generate)
  terminal=serve_pi_native_tools(bridge,root/f'pi-ep{k:02d}',pi_bin=pi_bin,provider_extension=PROVIDER_V2,pi_extensions=EXTENSIONS,cwd=workspace,seconds=180,extra_env={'E97_PI_TOOL_MANIFEST':str(manifest_path.resolve())})
  if not terminal['close_verified'] or index!=len(ep['steps']) or bridge.final is None:raise ValueError('incomplete authored trajectory')
  results=[m for m in bridge.history if m['role']=='toolResult']
  calls=[b for m in bridge.history if m['role']=='assistant' for b in m['content'] if b['type']=='toolCall']
  expected_calls=[s for s in emitted_specs if s['name']!='finish']
  if len(results)!=len(calls) or any(r['toolCallId']!=c['id'] or r['toolName']!=c['name'] for c,r in zip(calls,results)):raise ValueError('call/result pairing')
  if len(calls)!=len(expected_calls) or any(c['name']!=s['name'] or c['arguments']!=s['arguments'] for c,s in zip(calls,expected_calls)):raise ValueError('Pi executed different action')
  if bridge.final!=emitted_specs[-1]['arguments']['message']:raise ValueError('terminal final mismatch')
  failed=lambda r:r['isError'] or r['content'][0]['text'].lower().startswith(('error:','tool error:')) or any(w in r['content'][0]['text'].lower() for w in ('exit code: 1','command exited with code 1','no matches found','no files found matching pattern'))
  if any(failed(r) for r in results):raise ValueError('unexpected tool error')
  if ep.get('pure_chat') and calls:raise ValueError('pure-chat episode used tools')
  snapshot={}
  for name in ep.get('expected_files',{}):
   p=workspace/name
   snapshot[name]=p.read_text() if p.exists() else None
  if snapshot!=ep.get('expected_files',{}):raise ValueError('workspace oracle')
  if ep.get('answer_must_be_observed'):
   if ep.get('expected_answer') is None or not any(str(ep['expected_answer']) in r['content'][0]['text'] for r in results):raise ValueError('answer not observed')
  if ep.get('recall') is not None:
   prior=prior_observations[ep['recall']['episode']] if ep['recall']['episode']<len(prior_observations) else None
   if not prior or not any(ep['recall']['value'] in x for x in prior):raise ValueError('recall value not observed in the referenced episode')
   if ep['recall']['value'] not in bridge.final:raise ValueError('recall finish does not restate the observed value')
  clock_check=None
  if fmt:
   if not results:raise ValueError('date episode without observation')
   observation=results[-1]['content'][0]['text']
   values=parse_date_observation(observation,fmt)
   clock_after=weekday_now(utc)
   check_clock(fmt,values,clock_before,clock_after)
   clock_check={'fmt':fmt,'weekday_before':clock_before,'weekday_after':clock_after,'observed':values,'final':bridge.final}
  episode_privates.append({'episode':k,'prompt':ep['prompt'],'source_messages':bridge.episode.source_messages(),
   'native_record':bridge.episode.text(),'generations':bridge.generations,'public_history':bridge.history,
   'terminal':terminal,'snapshot':snapshot,'supervise_from':0,'assistant_units':len(bridge.generations),
   'clock_check':clock_check,'calls':len(calls),'errors':sum(r['isError'] for r in results)})
  prior_observations.append([r['content'][0]['text'] for r in results])
  all_generations.extend(bridge.generations)
  episode_texts.append(bridge.episode.text())
  total_calls+=len(calls);total_errors+=sum(r['isError'] for r in results)
 record_text=stitch_record(episode_texts)
 ids,mask,units=encode_candidate(record_text,all_generations,case['supervise_from'],enc)
 record_sha=hashlib.sha256(record_text.encode()).hexdigest()
 private={'id':case['id'],'category':case['category'],'family':case['family'],
  'episodes':episode_privates,'record_text':record_text,'generations':all_generations,
  'snapshot':episode_privates[-1]['snapshot'],'supervise_from':case['supervise_from'],
  'assistant_units':len(all_generations),'supervised_assistant_units':units,
  'targets':sum(mask),'record_sha256':record_sha,
  'session_reset_policy':case['session_reset_policy'],'episodes_count':len(case['episodes']),
  'calls':total_calls,'errors':total_errors}
 publish(root/'session-private.json',private);shutil.rmtree(workspace)
 return dict(id=case['id'],category=case['category'],family=case['family'],tokens=np.asarray(ids,dtype='<u4').tobytes(),mask=mask,targets=sum(mask),assistant_units=len(all_generations),supervised_units=units,errors=total_errors,calls=total_calls,episodes=len(case['episodes']),episode_sha256=sha(root/'session-private.json'),sequence_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()+mask).hexdigest(),repository_discovery=case['repository_discovery'])

def row_from_private(case,private_path,enc):
 private=json.loads(private_path.read_text())
 record_text=private['record_text']
 ids,mask,units=encode_candidate(record_text,private['generations'],private['supervise_from'],enc)
 if private['targets']!=sum(mask) or private['supervised_assistant_units']!=units:raise ValueError('resume re-encode mismatch')
 return dict(id=private['id'],category=private['category'],family=private['family'],tokens=np.asarray(ids,dtype='<u4').tobytes(),mask=mask,targets=sum(mask),assistant_units=private['assistant_units'],supervised_units=units,errors=private['errors'],calls=private['calls'],episodes=private['episodes_count'],episode_sha256=sha(private_path),sequence_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()+mask).hexdigest(),repository_discovery=False)

def collect(args):
 from scripts.e97_pi_native_bridge import BridgeStopped
 plan=json.loads(args.plan.read_text())
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 manifest_path=Path(plan['tool_manifest'])
 if sha(manifest_path)!=plan['tool_manifest_sha256']!=MANIFEST_SHA or plan['authority_files']!=authority_files() or str(args.pi_bin.resolve())!=plan['pi_bin'] or subprocess.check_output([args.pi_bin,'--version'],text=True).strip()!=plan['pi_version'] or plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates']:raise ValueError('authority')
 if plan['schema']!=PLAN_SCHEMA:raise ValueError('plan schema')
 manifest=json.loads(manifest_path.read_text())
 panel={'system':plan['system'],'tools':manifest['model_visible_tools'],'max_turns':10,'generation_budget':2048,'episode_generation_budget':8192,'episode_seconds':150}
 args.output.mkdir(parents=True,mode=0o700,exist_ok=True)
 if (args.output/'candidate-authority/manifest.json').exists() and (args.output/'summary.json').exists():
  print('SESSION_V1_ALREADY_COLLECTED',sha(args.output/'summary.json'));return
 enc=tiktoken.get_encoding('p50k_base')
 selected=pilot_sample(plan['cases'],args.stride)
 attempts=selected[:args.max_cases] if args.max_cases is not None else selected;total=len(attempts)
 resume_path=args.output/'resume-state.json'
 transient_counts={}
 if resume_path.exists():transient_counts=json.loads(resume_path.read_text()).get('transient_failures',{})
 stop=threading.Event()
 def attempt(case):
  root=args.output/case['id']
  if (root/'session-private.json').exists():return row_from_private(case,root/'session-private.json',enc),None
  if (root/'rejection.json').exists():return None,json.loads((root/'rejection.json').read_text())
  if stop.is_set():raise KeyboardInterrupt
  if root.exists():shutil.rmtree(root)
  try:
   return execute_session(case,panel,enc,root,args.pi_bin,manifest_path),None
  except ValueError as exc:
   rejection={'id':case['id'],'category':case['category'],'family':case['family'],'type':'ValueError','message':str(exc),'retried':False}
   publish(root/'rejection.json',rejection);return None,rejection
  except (BridgeStopped,OSError,TimeoutError) as exc:
   count=transient_counts.get(case['id'],0)+1
   transient_counts[case['id']]=count
   if root.exists():shutil.rmtree(root)
   if count>=2:
    rejection={'id':case['id'],'category':case['category'],'family':case['family'],'type':'BridgeStopped','message':f'transient transport failure x{count}: {exc}','retried':False}
    publish(root/'rejection.json',rejection);return None,rejection
   return None,{'id':case['id'],'transient':str(exc)}
 rows=[];rejections=[];done=0
 interrupted=None
 pool=ThreadPoolExecutor(max_workers=args.workers)
 futures=[pool.submit(attempt,case) for case in attempts]
 try:
  for future in futures:
   row,rejection=future.result();done+=1
   if row is not None:rows.append(row)
   elif rejection is not None and 'transient' not in rejection:rejections.append(rejection)
   if done%20==0 or done==total:
    print('SESSION_V1_ATTEMPTS',done,'VERIFIED',len(rows),'REJECTED',len(rejections),'TRANSIENT',len(transient_counts),flush=True)
 except (KeyboardInterrupt,TimeoutError) as exc:
  interrupted=repr(exc);stop.set();print('SESSION_V1_INTERRUPTED checkpointing',flush=True)
 finally:
  pool.shutdown(wait=True,cancel_futures=True)
 pending=[c['id'] for c in plan['cases'] if not ((args.output/c['id']/'session-private.json').exists() or (args.output/c['id']/'rejection.json').exists())]
 resume_state={'schema':RESUME_SCHEMA,'plan_sha256':args.plan_sha,'verified':len(rows),'rejected':len(rejections),'pending':pending,'transient_failures':transient_counts,'interrupted':interrupted}
 resume_path.write_text(json.dumps(resume_state,sort_keys=True,indent=1)+'\n')
 if interrupted is not None or pending:raise SystemExit(f'SESSION_V1_INCOMPLETE verified={len(rows)} rejected={len(rejections)} pending={len(pending)}')
 (args.output/'rejections.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rejections))
 if len(rows)<plan['minimum_verified_records']:raise ValueError('minimum verified records not reached')
 if len({r['sequence_sha256'] for r in rows})!=len(rows):raise ValueError('dedup')
 write_authority(rows,{**plan,'plan_sha256':args.plan_sha},args.output)
 families=Counter(r['family'] for r in rows)
 summary={'schema':SUMMARY_SCHEMA,'status':'qualified-candidates-not-admitted','attempted_records':plan['records'],'records':len(rows),'rejected_records':len(rejections),'automatic_retries':0,'minimum_verified_records':plan['minimum_verified_records'],'family_counts':dict(families),'episodes':sum(r['episodes'] for r in rows),'repository_discovery_records':sum(r['repository_discovery'] for r in rows),'native_calls':sum(r['calls'] for r in rows),'authentic_tool_errors':sum(r['errors'] for r in rows),'assistant_targets':sum(r['targets'] for r in rows),'deduplicated_sequences':len(rows),'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False,'checkpoint_promotion':False,'authority_sha256':sha(args.output/'candidate-authority/manifest.json')}
 publish(args.output/'summary.json',summary)
 print('SESSION_V1_COLLECTED',len(rows),len(rejections),summary['episodes'],summary['native_calls'],summary['assistant_targets'],flush=True)

def main():
 import os
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')))
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
 f=sub.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--records',type=int,required=True);f.add_argument('--minimum-verified-records',type=int);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
 c=sub.add_parser('collect');c.add_argument('--plan',type=Path,required=True);c.add_argument('--plan-sha',required=True);c.add_argument('--pi-bin',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--workers',type=int,default=1);c.add_argument('--max-cases',type=int,default=None,help='pilot bound: attempt only the first N selected sessions (resume-friendly checkpoint; the run stays incomplete)');c.add_argument('--stride',type=int,default=1,help='pilot sampling: attempt every Nth plan session (1 = full plan)')
 a=parser.parse_args();freeze(a) if a.command=='freeze' else collect(a)
if __name__=='__main__':main()
