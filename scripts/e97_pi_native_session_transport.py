"""Bounded real-Pi RPC lifecycle for explicitly separate native task records."""
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import tempfile
import time

from scripts.e97_open_swe_native_codec import compact, strict_json
from scripts.e97_pi_native_bridge import API, MODEL, PROVIDER, BridgeStopped
from scripts.e97_pi_native_session import history_sha
from scripts.e97_pi_native_transport import MAX_FRAME


def serve_pi_session(session, tasks, output, *, pi_bin, extension, seconds=1200, require_successful_tasks=True):
    output=Path(output); output.mkdir(parents=True,mode=0o700,exist_ok=False)
    if (not isinstance(tasks,list) or not 1<=len(tasks)<=session.max_tasks or
        any(set(t)!={'task_id','prompt'} for t in tasks)):
        raise ValueError('bounded session tasks')
    home=output/'home'; agent=home/'.pi/agent'; agent.mkdir(parents=True,mode=0o700)
    cwd=output/'empty-cwd'; cwd.mkdir(mode=0o700)
    settings=dict(compaction=dict(enabled=False),retry=dict(enabled=False,provider=dict(maxRetries=0)),defaultTools=[],
                  packages=[],extensions=[],enableInstallTelemetry=False,enableAnalytics=False,
                  defaultProjectTrust='never',quietStartup=True)
    (agent/'settings.json').write_text(compact(settings)+'\n')
    deadline=min(session.deadline,time.monotonic()+seconds); proc=None; pidfd=None; events=[]; requests=[]; settled=0
    terminal=dict(schema='emender-e97-pi-native-session-terminal-v1',api=API,tasks=len(tasks),requests=requests,
                  settled_events=0,pi_exit=None,session_close_verified=False)
    try:
      with tempfile.TemporaryDirectory(prefix='e97-pi-native-session-') as tmp:
        address=str(Path(tmp)/'bridge.sock')
        config=dict(schema='emender-e97-pi-native-session-transport-v1',socket=address,system=session.panel['system'],
                    tasks=tasks,tools=session.tasks[0].tools if session.tasks else None,timeout_ms=int(seconds*1000))
        # Tools do not depend on a begun task.
        if config['tools'] is None:
            from scripts.e97_pi_native_bridge import transport_tools
            config['tools']=transport_tools(session.panel['tools'])
        cp=output/'transport-config.json'; cp.write_text(compact(config)+'\n')
        env=dict(PATH=os.environ['PATH'],HOME=str(home),PI_CODING_AGENT_DIR=str(agent),PI_OFFLINE='1',
                 PI_SKIP_VERSION_CHECK='1',NO_COLOR='1',TERM='dumb',E97_PI_NATIVE_CONFIG=str(cp),
                 XDG_CACHE_HOME=str(output/'cache'))
        command=[str(pi_bin),'--offline','--no-approve','--no-extensions','--no-skills','--no-prompt-templates',
                 '--no-themes','--no-context-files','--no-builtin-tools','-e',str(Path(extension).resolve()),
                 '--provider',PROVIDER,'--model',MODEL,'--system-prompt',config['system'],
                 '--session-dir',str(output/'sessions'),'--mode','rpc']
        (output/'command-private.json').write_text(compact(command)+'\n')
        with socket.socket(socket.AF_UNIX) as listener:
          listener.bind(address); os.chmod(address,0o600); listener.listen(1)
          with (output/'pi-events-private.jsonl').open('x') as journal,(output/'pi-stderr-private.log').open('x') as stderr:
            proc=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr)
            terminal['pi_pid']=proc.pid; pidfd=os.pidfd_open(proc.pid); buffer=b''
            def prompt(index):
                wire=(compact(dict(id='task-'+str(index),type='prompt',message=tasks[index]['prompt']))+'\n').encode()
                proc.stdin.write(wire); proc.stdin.flush()
            prompt(0)
            while True:
              if len(requests)>2*sum(t.panel['max_turns'] for t in session.tasks)+3*len(tasks)+4:
                  raise BridgeStopped('session_transport_request_bound')
              readable,_,_=select.select([listener,proc.stdout,pidfd],[],[],max(0,deadline-time.monotonic()))
              if not readable: raise BridgeStopped('session_transport_deadline')
              if listener in readable:
                conn,_=listener.accept()
                with conn:
                  pid,uid,_=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                  if pid!=proc.pid or uid!=os.getuid(): raise BridgeStopped('unexpected_session_peer')
                  conn.settimeout(max(.001,deadline-time.monotonic()))
                  with conn.makefile('rb') as reader: line=reader.readline(MAX_FRAME+1)
                  if len(line)>MAX_FRAME or not line.endswith(b'\n'): raise BridgeStopped('session_frame_bound')
                  requests.append(dict(op='invalid',peer_pid=pid,peer_uid=uid))
                  try:
                    req=strict_json(line.decode()); op=req['op']; requests[-1]['op']=op
                    with (output/'requests-private.jsonl').open('a') as reqs: reqs.write(compact(req)+'\n')
                    if op=='begin_task':
                      wanted=tasks[len(session.tasks)]
                      if req!={**wanted,'op':'begin_task','acknowledge_fresh_record':True}: raise BridgeStopped('session_task_request_mismatch')
                      result=session.begin_task(req['task_id'],req['prompt'],expected_history_sha=history_sha(session.completed_history),acknowledge_fresh_record=True)
                    elif op=='next': result=session.next(req)
                    elif op=='execute': result=session.execute(req)
                    elif op=='settle_task': result=session.settle_task(req['messages'])
                    elif op=='close_session': result=session.close(expected_history_sha=history_sha(session.completed_history))
                    else: raise BridgeStopped('unsupported_session_operation')
                    response=dict(ok=True,result=result)
                  except Exception as exc:
                    requests[-1]['error_type']=type(exc).__name__
                    with (output/'owner-errors-private.jsonl').open('a') as errors: errors.write(compact(dict(type=type(exc).__name__,message=str(exc)))+'\n')
                    response=dict(ok=False)
                  wire=(compact(response)+'\n').encode()
                  if len(wire)>MAX_FRAME: raise BridgeStopped('session_response_bound')
                  conn.sendall(wire)
              if proc.stdout in readable:
                chunk=os.read(proc.stdout.fileno(),65536)
                if not chunk:
                    terminal['pi_exit']=proc.wait(timeout=5)
                    if settled==len(tasks) and session.closed: break
                    raise BridgeStopped('pi_stdout_closed_before_session')
                buffer+=chunk
                if len(buffer)>MAX_FRAME: raise BridgeStopped('pi_event_frame_bound')
                while b'\n' in buffer:
                  line,buffer=buffer.split(b'\n',1); journal.write(line.decode()+'\n'); journal.flush()
                  event=strict_json(line.decode()); events.append(event)
                  if event.get('type')=='agent_settled':
                    settled+=1; terminal['settled_events']=settled
                    if settled<len(tasks): prompt(settled)
              if pidfd in readable:
                terminal['pi_exit']=proc.wait(timeout=5)
                if settled!=len(tasks) or not session.closed: raise BridgeStopped('pi_exited_before_session_close')
                break
    finally:
      if pidfd is not None: os.close(pidfd)
      if proc is not None and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
      if proc is not None: terminal['pi_exit']=proc.returncode
      terminal.update(session_close_verified=session.closed,settled_events=settled,
                      tasks_begun=len(session.tasks),successful_task_closures=sum(t.close_verified for t in session.tasks),
                      failed_task_settlements=sum(t.failed and t.closed for t in session.tasks))
      (output/'transport-terminal.json').write_text(compact(terminal)+'\n')
    settled_tasks = all(t.closed and (t.close_verified or t.failed) for t in session.tasks)
    if (terminal['pi_exit']!=0 or not session.closed or settled!=len(tasks) or not settled_tasks or
            (require_successful_tasks and any(not t.close_verified for t in session.tasks))):
        raise BridgeStopped('pi_session_not_qualified')
    return terminal
