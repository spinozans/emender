import tiktoken
import pytest
from scripts.audit_e97_pi_native_curriculum import expected_mask,verify_case
from scripts.e97_open_swe_native_codec import compact

def assistant(name,args):
 return {'role':'assistant','content':None,'reasoning_content':None,'think':None,'tool_calls':[{'type':'function','function':{'name':name,'arguments':compact(args)}}]}
def test_expected_mask_excludes_failure_prefix():
 enc=tiktoken.get_encoding('p50k_base');a='Analysis: null\nCommentary: null\nThink: null\nAction: read\nArguments: {"path":"missing"}';b='Analysis: null\nCommentary: null\nThink: null\nAction: finish\nArguments: {"message":"done"}';text='header\n\nAssistant:\n'+a+'\n\nToolResult:\n{}\n\nAssistant:\n'+b
 ids,mask=expected_mask(text,[a,b],1,enc)
 assert sum(mask)==len(enc.encode_ordinary(b)) and not any(mask[:len(enc.encode_ordinary('header\n\nAssistant:\n'+a))])
def test_verify_case_requires_observation_grounded_dynamic_final():
 tools=[{'name':'web_search','label':'x','description':'x','parameters':{}}]
 result={'role':'toolResult','toolCallId':'c','toolName':'web_search','content':[{'type':'text','text':'  grounded   evidence  '}],'isError':False}
 case={'steps':[{'name':'web_search','arguments':{'query':'q'}},{'name':'finish','dynamic':'observation'}],'expected_files':{},'category':'web','family':'web-search-synthesis','supervise_from':0,'requires_error':False}
 private={'source_messages':[assistant('web_search',{'query':'q'}),result,assistant('finish',{'message':'Returned web evidence: grounded evidence'})],'snapshot':{}}
 results,actions=verify_case(case,private,tools)
 assert len(results)==1 and actions[-1]['name']=='finish'
 private['source_messages'][-1]=assistant('finish',{'message':'invented'})
 with pytest.raises(ValueError,match='grounding'):verify_case(case,private,tools)
