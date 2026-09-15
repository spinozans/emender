import base64
import json
import pytest
from scripts.e97_pi_native_collection_proxy import backend,confined

class FakeSandbox:
 def __init__(self):self.calls=[]
 def request(self,op,**kwargs):
  self.calls.append((op,kwargs))
  return {'result':{'exit_code':0,'message':{'content':'ok'}}}

def test_confined_rejects_absolute_and_parent():
 assert confined('a/b.txt')=='/testbed/a/b.txt'
 for value in ('/etc/passwd','../x','a/../../x'):
  with pytest.raises(ValueError):confined(value)

def test_read_maps_to_bounded_native_editor_view():
 s=FakeSandbox();assert backend(s,'read',{'path':'docs/a.txt','offset':3,'limit':4})==('ok',False)
 assert s.calls==[('execute',{'call':{'name':'str_replace_editor','arguments':{'command':'view','path':'/testbed/docs/a.txt','view_range':[3,6]}}})]

def test_write_encodes_content_instead_of_interpolating_shell():
 s=FakeSandbox();backend(s,'write',{'path':'out/x.txt','content':'$(touch /tmp/no)\n'})
 command=s.calls[0][1]['call']['arguments']['command'];assert '$(touch' not in command
 assert base64.b64encode(b'$(touch /tmp/no)\n').decode() in command

def test_process_is_fail_closed():
 s=FakeSandbox();text,error=backend(s,'process',{'action':'start','name':'x','command':'sleep 1'})
 assert error and 'not enabled' in text and not s.calls

def test_backend_propagates_native_error():
 class ErrorSandbox:
  def request(self,*args,**kwargs):return {'result':{'exit_code':2,'message':{'content':'bad'}}}
 assert backend(ErrorSandbox(),'bash',{'command':'false'})==('bad',True)
