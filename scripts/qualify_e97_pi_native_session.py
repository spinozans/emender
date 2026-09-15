#!/usr/bin/env python3
"""Two explicit native records through one real Pi/OpenHands session; zero model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tiktoken

from scripts.e97_native_execution_cases import context
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import compact,native_turn
from scripts.e97_pi_native_session import NativeTaskSession
from scripts.e97_pi_native_session_transport import serve_pi_session
from scripts.eval_e97_native_execution import publish,sha


def scripted():
    tasks=[dict(task_id='workspace-create',prompt='Create the requested shared workspace state, verify it, and finish.'),
           dict(task_id='workspace-followup',prompt='Inspect the existing shared workspace state, update it as requested, verify it, and finish.')]
    turns=[
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_1',think=None,tool_calls=[dict(type='function',function=dict(name='execute_bash',arguments=compact(dict(command="export E97_SESSION_MARK='persisted'; pwd"))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_2',think=None,tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=compact(dict(command='create',path='/testbed/shared.txt',file_text='alpha\n'))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_3',think=None,tool_calls=[dict(type='function',function=dict(name='finish',arguments=compact(dict(message='Created shared workspace state.'))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_4',think=None,tool_calls=[dict(type='function',function=dict(name='execute_bash',arguments=compact(dict(command="printf '%s|%s\\n' \"$E97_SESSION_MARK\" \"$PWD\""))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_5',think=None,tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=compact(dict(command='view',path='/testbed/shared.txt'))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_6',think=None,tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=compact(dict(command='str_replace',path='/testbed/shared.txt',old_str='alpha',new_str='beta'))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_7',think=None,tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=compact(dict(command='view',path='/testbed/shared.txt'))))])),
      native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_ANALYSIS_8',think=None,tool_calls=[dict(type='function',function=dict(name='finish',arguments=compact(dict(message='Updated and verified shared workspace state.'))))])),
    ]
    return tasks,turns


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--panel-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pi',default=shutil.which('pi'))
    a=p.parse_args();os.umask(0o077)
    def interrupted(signum,frame):raise TimeoutError('session qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or sha(a.panel)!=a.panel_sha:raise ValueError('CPU/panel identity')
    panel=json.loads(a.panel.read_text());tasks,turns=scripted();a.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(a.output/'plan-private.json',dict(tasks=tasks,scripted_turn_sha256=[__import__('hashlib').sha256(t.encode()).hexdigest() for t in turns],
        model_generations=0,optimizer_updates=0,max_tasks=2,session_tokens=16384,session_seconds=300,
        conversational_memory_claim=False,training_eligible=False))
    encoding=tiktoken.get_encoding('p50k_base');queue=iter(turns);generated=0;calls=[]
    try:
      with NativeSandbox(panel,a.output/'sandbox') as sandbox:
        sandbox.request('setup',files={'seed.txt':'unchanged\n'})
        def generate(prompt,budget,deadline):
            nonlocal generated
            text=next(queue);generated+=1;ids=encoding.encode_ordinary(text)
            if len(ids)>budget:raise ValueError('scripted generation budget')
            return text,ids,'valid'
        def execute(call):
            reply=sandbox.request('execute',call=call);calls.append(dict(request=call,reply=reply));return reply
        session=NativeTaskSession(panel,encoding,generate,execute,max_tasks=2,session_tokens=16384,session_seconds=300)
        terminal=serve_pi_session(session,tasks,a.output/'pi',pi_bin=a.pi,
            extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts',seconds=300)
        snapshot=sandbox.snapshot(['seed.txt','shared.txt'])
      if generated!=8 or snapshot!={'seed.txt':'unchanged\n','shared.txt':'beta\n'}:
        raise ValueError('session scripted coverage/oracle')
      if not any('persisted|/testbed' in c['reply']['result']['message']['content'] for c in calls):
        raise ValueError('persistent shell did not cross explicit task boundary')
      public=(a.output/'pi/pi-events-private.jsonl').read_text()
      if 'PRIVATE_SESSION_ANALYSIS_' in public:raise ValueError('private analysis crossed Pi boundary')
      publish(a.output/'session-private.json',dict(boundaries=session.boundaries,completed_public_history=session.completed_history,
          native_records=[dict(text=t.episode.text(),source_messages=t.episode.source_messages(),generations=t.generations,calls=t.calls,
                               final=t.final,reason=t.reason,closed=t.closed,close_verified=t.close_verified) for t in session.tasks],
          snapshot=snapshot,terminal=terminal,model_generations=0,optimizer_updates=0))
      publish(a.output/'summary.json',dict(status='two-task-real-pi-openhands-session-passed',tasks=2,native_turns=8,
          native_external_calls=6,workspace_continuity=True,persistent_shell_continuity=True,private_records_retained=2,
          prior_task_in_next_model_context=False,public_history_retained=True,conversational_memory_claim=False,
          model_generations=0,optimizer_updates=0,training_eligible=False))
      print('PI_NATIVE_TWO_TASK_SESSION_PASSED',generated,len(calls),flush=True)
    except BaseException as exc:
      publish(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),model_generations=0,optimizer_updates=0))
      raise

if __name__=='__main__':main()
