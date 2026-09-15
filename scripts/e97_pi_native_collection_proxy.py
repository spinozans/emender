"""Unix owner for credential-free Pi-native collection tools in NativeSandbox."""
import base64,hashlib,json,os,select,shlex,socket,struct,time,uuid
from pathlib import Path
from scripts.e97_open_swe_native_codec import compact,strict_json
MAX_FRAME=16*1024*1024;LOCAL_TOOLS=('read','bash','edit','write','process','ffgrep','fffind')

def confined(path):
 p=Path(path)
 if not isinstance(path,str) or not path or p.is_absolute() or '..' in p.parts:raise ValueError('path outside collection workspace')
 return '/testbed/'+p.as_posix()
def content(reply):
 if 'dispatch_error' in reply:raise ValueError('sandbox dispatch')
 result=reply['result'];message=result['message'];return message['content'],bool(result.get('exit_code',0)) or message['content'].startswith('ERROR:')
def backend(sandbox,name,args):
 if name=='read':
  call={'name':'str_replace_editor','arguments':{'command':'view','path':confined(args['path'])}}
  if 'offset' in args or 'limit' in args:
   start=int(args.get('offset',1));limit=int(args.get('limit',2000));call['arguments']['view_range']=[start,start+limit-1]
  return content(sandbox.request('execute',call=call))
 if name=='bash':return content(sandbox.request('execute',call={'name':'execute_bash','arguments':{'command':args['command']}}))
 if name=='edit':
  texts=[];failed=False
  for e in args['edits']:
   text,error=content(sandbox.request('execute',call={'name':'str_replace_editor','arguments':{'command':'str_replace','path':confined(args['path']),'old_str':e['oldText'],'new_str':e['newText']}}));texts.append(text);failed|=error
   if error:break
  return '\n'.join(texts),failed
 if name=='write':
  encoded=base64.b64encode(args['content'].encode()).decode();path=confined(args['path']);script=f"import base64,pathlib;p=pathlib.Path({path!r});p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode({encoded!r}))"
  return content(sandbox.request('execute',call={'name':'execute_bash','arguments':{'command':'python -c '+shlex.quote(script)}}))
 if name in ('fffind','ffgrep'):
  payload=base64.b64encode(compact(args).encode()).decode();script=_FFF_SCRIPT;command='python -c '+shlex.quote(script)+' '+shlex.quote(name)+' '+shlex.quote(payload)
  return content(sandbox.request('execute',call={'name':'execute_bash','arguments':{'command':command}}))
 if name=='process':return 'Managed process execution is not enabled in the first credential-free collection pilot.',True
 raise ValueError('unsupported collection tool')

_FFF_SCRIPT=r'''import base64,json,pathlib,re,sys
name=sys.argv[1];a=json.loads(base64.b64decode(sys.argv[2]));root=pathlib.Path('/testbed');scope=a.get('path');base=root/(scope or '')
if not base.resolve().is_relative_to(root):raise SystemExit(2)
files=sorted(p for p in base.rglob('*') if p.is_file());limit=int(a.get('limit',20 if name=='ffgrep' else 30));pattern=a['pattern'];rows=[]
if name=='fffind':
 words=pattern.lower().split()
 for p in files:
  rel=p.relative_to(root).as_posix()
  if all(w in rel.lower() for w in words):rows.append(rel)
else:
 flags=0 if a.get('caseSensitive') else re.I
 try:rx=re.compile(pattern,flags)
 except re.error:rx=re.compile(re.escape(pattern),flags)
 for p in files:
  try:lines=p.read_text().splitlines()
  except UnicodeDecodeError:continue
  for i,line in enumerate(lines,1):
   if rx.search(line):rows.append(f'{p.relative_to(root).as_posix()}:{i}:{line}')
print('\n'.join(rows[:limit]))
'''

class ProxyOwner:
 def __init__(self,sandbox,output,timeout=300):self.sandbox=sandbox;self.output=Path(output);self.timeout=timeout;self.receipts=[]
 def serve(self,proc,address):
  deadline=time.monotonic()+self.timeout;pidfd=os.pidfd_open(proc.pid)
  try:
   with socket.socket(socket.AF_UNIX) as listener:
    listener.bind(str(address));os.chmod(address,0o600);listener.listen(8)
    while proc.poll() is None:
     readable,_,_=select.select([listener,pidfd],[],[],max(0,deadline-time.monotonic()))
     if not readable:raise TimeoutError('collection proxy deadline')
     if listener not in readable:break
     conn,_=listener.accept()
     with conn:
      pid,uid,_=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
      if pid!=proc.pid or uid!=os.getuid():raise ValueError('unexpected collection peer')
      with conn.makefile('rb') as f:line=f.readline(MAX_FRAME+1)
      if len(line)>MAX_FRAME or not line.endswith(b'\n'):raise ValueError('collection frame bound')
      request=strict_json(line.decode());response={'ok':False,'error':'rejected'}
      try:
       if set(request)!={'op','id','name','arguments'} or request['op']!='execute' or request['name'] not in LOCAL_TOOLS or not isinstance(request['arguments'],dict):raise ValueError('collection request')
       text,is_error=backend(self.sandbox,request['name'],request['arguments']);receipt=hashlib.sha256((request['name']+'\0'+compact(request['arguments'])+'\0'+text).encode()).hexdigest();result={'text':text,'is_error':is_error,'receipt':receipt};self.receipts.append({'id':request['id'],'name':request['name'],'arguments':request['arguments'],'text':text,'is_error':is_error,'receipt':receipt});response={'ok':True,'result':result}
      except Exception as exc:self.receipts.append({'id':request.get('id'),'name':request.get('name'),'rejected':type(exc).__name__,'message':str(exc)});response={'ok':False,'error':'collection owner rejected request'}
      wire=(compact(response)+'\n').encode()
      if len(wire)>MAX_FRAME:raise ValueError('collection response bound')
      conn.sendall(wire)
  finally:os.close(pidfd);self.output.mkdir(parents=True,exist_ok=True);(self.output/'proxy-receipts-private.jsonl').write_text(''.join(compact(x)+'\n' for x in self.receipts))
