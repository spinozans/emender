"""Bounded real-Pi transport; Pi executes captured tools, owner serves E97 turns."""
import json,os,select,socket,struct,subprocess,tempfile,time
from pathlib import Path
from scripts.e97_open_swe_native_codec import compact,strict_json
from scripts.e97_pi_native_bridge import BridgeStopped
from scripts.e97_pi_native_tool_bridge import API,MODEL,PROVIDER
MAX_FRAME=16*1024*1024

def append_jsonl(path,value):
 with Path(path).open('a') as f:f.write(compact(value)+'\n')

def serve_pi_native_tools(bridge,output,*,pi_bin,provider_extension,pi_extensions=(),cwd,seconds=120):
 output=Path(output);output.mkdir(parents=True,mode=0o700,exist_ok=False);home=output/'home';agent=home/'.pi/agent';agent.mkdir(parents=True,mode=0o700);sessions=output/'sessions';sessions.mkdir(mode=0o700)
 settings={'compaction':{'enabled':False},'retry':{'enabled':False,'provider':{'maxRetries':0}},'defaultTools':[],'packages':[],'extensions':[],'enableInstallTelemetry':False,'enableAnalytics':False,'defaultProjectTrust':'never','quietStartup':True};(agent/'settings.json').write_text(compact(settings)+'\n')
 deadline=time.monotonic()+seconds;receipts=[];proc=None;pidfd=None;terminal={'schema':'emender-e97-pi-native-tool-terminal-v1','close_verified':False,'pi_exit':None,'requests':receipts}
 try:
  with tempfile.TemporaryDirectory(prefix='e97-pi-tools-') as tmp:
   address=str(Path(tmp)/'owner.sock');config={'schema':'emender-e97-pi-native-tool-transport-v1','socket':address,'system':bridge.panel['system'],'prompt':bridge.history[0]['content'][0]['text'],'tools':bridge.tools,'timeout_ms':int(seconds*1000)};config_path=output/'transport-config.json';config_path.write_text(compact(config)+'\n')
   env={'PATH':os.environ['PATH'],'HOME':str(home),'PI_CODING_AGENT_DIR':str(agent),'PI_OFFLINE':'1','PI_SKIP_VERSION_CHECK':'1','NO_COLOR':'1','TERM':'dumb','E97_PI_NATIVE_TOOL_CONFIG':str(config_path),'XDG_CACHE_HOME':str(output/'cache'),'PI_FFF_MODE':'tools-only','FFF_ENABLE_HOME_SCAN':'0','FFF_ENABLE_ROOT_SCAN':'0'}
   command=[str(pi_bin),'--offline','--no-approve','--no-extensions','--no-skills','--no-prompt-templates','--no-themes','--no-context-files','-e',str(Path(provider_extension).resolve())]
   for ext in pi_extensions:command+=['-e',str(Path(ext).resolve())]
   command+=['--provider',PROVIDER,'--model',MODEL,'--tools',','.join(t['name'] for t in bridge.tools),'--system-prompt',config['system'],'--session-dir',str(sessions),'--mode','json','-p','--',config['prompt']]
   (output/'command-private.json').write_text(compact(command)+'\n')
   with socket.socket(socket.AF_UNIX) as listener:
    listener.bind(address);os.chmod(address,0o600);listener.listen(1)
    with (output/'pi-events-private.jsonl').open('x') as stdout,(output/'pi-stderr-private.log').open('x') as stderr:
     proc=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr);terminal['pi_pid']=proc.pid;pidfd=os.pidfd_open(proc.pid)
     while not bridge.closed:
      if len(receipts)>=2*bridge.panel['max_turns']+3:raise BridgeStopped('transport_request_bound')
      readable,_,_=select.select([listener,pidfd],[],[],max(0,deadline-time.monotonic()))
      if not readable:raise BridgeStopped('transport_deadline')
      if listener not in readable:raise BridgeStopped('pi_exited_before_close')
      conn,_=listener.accept()
      with conn:
       pid,uid,_=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
       if pid!=proc.pid or uid!=os.getuid():raise BridgeStopped('unexpected_transport_peer')
       conn.settimeout(max(.001,deadline-time.monotonic()))
       with conn.makefile('rb') as reader:line=reader.readline(MAX_FRAME+1)
       if len(line)>MAX_FRAME or not line.endswith(b'\n'):raise BridgeStopped('transport_frame_bound')
       receipts.append({'op':'invalid','peer_pid':pid,'peer_uid':uid})
       try:
        req=strict_json(line.decode());append_jsonl(output/'requests-private.jsonl',req);op=req['op'];receipts[-1]['op']=op
        if op=='next':result=bridge.next(req)
        elif op=='close':result=bridge.close(req['messages'])
        else:raise BridgeStopped('unsupported_transport_operation')
        response={'ok':True,'result':result}
       except Exception as exc:
        bridge.failed=True
        if bridge.reason in (None,'finished'):bridge.reason='transport_contract_failure'
        append_jsonl(output/'owner-errors-private.jsonl',{'type':type(exc).__name__,'message':str(exc)});response={'ok':False}
       wire=(compact(response)+'\n').encode()
       if len(wire)>MAX_FRAME:raise BridgeStopped('transport_response_bound')
       conn.sendall(wire)
     terminal['pi_exit']=proc.wait(timeout=max(.001,deadline-time.monotonic()))
 finally:
  if pidfd is not None:os.close(pidfd)
  if proc is not None and proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
  if proc is not None:terminal['pi_exit']=proc.returncode
  terminal.update(close_verified=bridge.close_verified,closed=bridge.closed,bridge_failed=bridge.failed,reason=bridge.reason);(output/'transport-terminal.json').write_text(compact(terminal)+'\n')
 if terminal['pi_exit']!=0 or not bridge.close_verified or bridge.failed:raise BridgeStopped('pi_native_tool_transport_not_qualified')
 return terminal
