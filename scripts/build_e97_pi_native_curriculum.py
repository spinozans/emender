#!/usr/bin/env python3
"""Freeze and collect verified authored Pi-native curriculum through real Pi tools.

This creates a non-admitted candidate authority. It never samples E97 or trains.
"""
import argparse,hashlib,json,os,random,re,shutil,signal,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import tiktoken
from ndm.data.masked_sft_dataset import RECORD_INDEX
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import native_turn
from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.eval_e97_native_execution import publish,sha
MANIFEST_SHA='55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908';SEED='e97-pi-native-curriculum-20260915-v1'
SYSTEM='Complete the task using the declared Pi-native tools. Use actual observations, recover from errors, and finish only when complete.'
WORDS=(' archive',' datum',' vector',' amber',' quiet',' ledger',' orbit',' cedar',' mosaic',' kernel',' violet',' signal',' parcel',' meadow',' copper',' tensor')
EXTENSIONS=(Path('/home/erikg/.pi/agent/npm/node_modules/@aliou/pi-processes/extensions/processes/index.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/@ff-labs/pi-fff/src/index.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/pi-web-access/index.ts'))

def opaque(seed,n=16):return hashlib.sha256(seed.encode()).hexdigest()[:n]
def filler(n,seed):
 rng=random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16],16));return ''.join(rng.choice(WORDS) for _ in range(n))
def case_id(category,i):return f'pi-native-{category}-{i:05d}-{opaque(f"{SEED}:{category}:{i}",8)}'
def finish(message):return {'name':'finish','arguments':{'message':message}}
def action(name,arguments):return {'name':name,'arguments':arguments}

def local_case(i):
 tag=opaque(f'local:{i}',10);kind=i%8;cid=case_id('local',i);files={};expected={};prefix=0
 if kind==0:
  path=f'data/fact_{tag}.txt';value='VALUE_'+opaque(cid,14);files[path]=value+'\n';steps=[action('read',{'path':path}),finish(value)];prompt=f'Read {path} and finish with exactly its value without the trailing newline.';family='repo-explicit-read'
 elif kind==1:
  path=f'src/weather_adapter_{tag}.py';files[path]='# local adapter\n';steps=[action('fffind',{'pattern':'weather_adapter_'+tag}),finish(path)];prompt=f'Locate the repository file whose name contains weather_adapter_{tag} using fuzzy file search, then finish with exactly its relative path.';family='repo-fuzzy-path'
 elif kind==2:
  marker='LOCAL_SYMBOL_'+tag.upper();path=f'lib/component_{tag}.py';files[path]=marker+' = 1\n';steps=[action('ffgrep',{'pattern':marker}),finish(path)];prompt=f'Find the repository file containing exact symbol {marker} using content search, then finish with exactly its relative path.';family='repo-content-search'
 elif kind==3:
  a=100+i*3;b=71+i;value=str(a+b);steps=[action('bash',{'command':f"printf '%s\\n' $(({a}+{b}))"}),finish(value)];prompt=f'Use bash to compute {a} plus {b}, then finish with exactly the decimal result.';family='shell-observe'
 elif kind==4:
  path=f'config/state_{tag}.txt';marker='STATE_TARGET_'+tag.upper();files[path]=marker+'\nmode=alpha\n';expected[path]=marker+'\nmode=beta\n';steps=[action('ffgrep',{'pattern':marker,'path':'config/'}),action('edit',{'path':path,'edits':[{'oldText':'mode=alpha','newText':'mode=beta'}]}),action('read',{'path':path}),finish('done')];prompt=f'Use content search to locate the config containing exact marker {marker}, change its mode from alpha to beta, read it back, then finish with exactly done.';family='repo-symbol-edit-discovery'
 elif kind==5:
  path=f'out/result_{tag}.txt';expected[path]='written-'+tag+'\n';steps=[action('write',{'path':path,'content':expected[path]}),action('read',{'path':path}),finish('done')];prompt=f'Create {path} with exactly written-{tag} followed by a newline, read it back, then finish with exactly done.';family='repo-write-verify'
 elif kind==6:
  path=f'pkg/calc_{tag}.py';files[path]='def total(xs):\n    return sum(xs[1:])\n';expected[path]='def total(xs):\n    return sum(xs)\n';test=f'tests/test_{tag}.py';files[test]=f'import unittest\nfrom pkg.calc_{tag} import total\nclass T(unittest.TestCase):\n def test_total(self): self.assertEqual(total([3,5,7]),15)\n';steps=[action('fffind',{'pattern':'test_'+tag+'.py','path':'tests/'}),action('bash',{'command':f'python3 -m unittest {test}'}),action('edit',{'path':path,'edits':[{'oldText':'return sum(xs[1:])','newText':'return sum(xs)'}]}),action('bash',{'command':f'python3 -m unittest {test}'}),finish('done')];prompt=f'Locate the checker named test_{tag}.py under tests/, run it to observe the defect, repair the implementation while leaving tests unchanged, rerun the checker, and finish with exactly done after it passes.';family='repo-test-discovery-repair';prefix=2
 else:
  path=f'nested/{tag}/settings.json';value='BIND_'+opaque(cid,12);files[path]=json.dumps({'active':'right','values':{'left':'decoy','right':value}})+'\n';steps=[action('fffind',{'pattern':'settings.json','path':'nested/'}),action('read',{'path':path}),finish(value)];prompt=f'Task label {tag}. Find settings.json under nested/, read it, follow active into values, and finish with exactly the selected value.';family='repo-nested-binding'
 expected={**files,**expected};return dict(id=cid,category='local',family=family,prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=prefix,requires_error=prefix>0,repository_discovery=kind in (1,2,4,6,7))

def exact_case(i):
 cid=case_id('exact',i);tag=opaque(cid,12);kind=i%6;distance=(128,1024,8192,32768)[(i//6)%4];pad=filler(distance,cid);files={};expected={}
 if kind==0:
  value='COPY_'+opaque(cid,20);prompt=f'Remember exactly {value}.{pad}\nFinish with exactly the remembered value and no other text.';steps=[finish(value)];family='copy'
 elif kind==1:
  value='SELECT_'+opaque(cid,18);table=compact({'active':'violet','values':{'amber':'decoy-a','violet':value,'silver':'decoy-b'}});prompt=f'Remember this binding: {table}{pad}\nFinish with exactly the active value.';steps=[finish(value)];family='binding'
 elif kind==2:
  path=f'needles/{tag}.txt';value='PATH_'+opaque(cid,16);files[path]=value+'\n';prompt=f'The exact relative path to read is {path}.{pad}\nRead that exact path, then finish with exactly its value.';steps=[action('read',{'path':path}),finish(value)];family='path-copy'
 elif kind==3:
  a=1000+i;b=37+(i%19);value=str(a+b);prompt=f'Remember operands {a} and {b}.{pad}\nUse bash to add them and finish with exactly the observed decimal result.';steps=[action('bash',{'command':f"printf '%s\\n' $(({a}+{b}))"}),finish(value)];family='delayed-calculation'
 elif kind==4:
  path=f'blocks/{tag}.txt';block='BEGIN\nkeep-'+opaque(cid,24)+'\nEND\n';files[path]='status=old\n'+block;expected[path]='status=new\n'+block;prompt=f'Preserve this exact block when editing later:\n{block}{pad}\nIn {path}, replace status=old with status=new, read it back, then finish with exactly done.';steps=[action('edit',{'path':path,'edits':[{'oldText':'status=old','newText':'status=new'}]}),action('read',{'path':path}),finish('done')];family='preservation'
 else:
  first=f'facts/{tag}.txt';value='EARLY_'+opaque(cid,18);files[first]=value+'\n';decoy=f'facts/decoy_{tag}.txt';files[decoy]='irrelevant\n';prompt=f'Read {first}, then read {decoy}. After both observations, finish with exactly the value from the first result.';steps=[action('read',{'path':first}),action('read',{'path':decoy}),finish(value)];family='early-result-recall';distance=0
 expected={**files,**expected};return dict(id=cid,category='exact',family=family,prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,delay_tokens=distance)

def recovery_case(i):
 cid=case_id('recovery',i);tag=opaque(cid,10);kind=i%6;files={};expected={};prefix=1
 if kind==0:
  path=f'src/recovered_{tag}.txt';value='RECOVER_'+opaque(cid,12);files[path]=value+'\n';steps=[action('read',{'path':f'missing/{tag}.txt'}),action('fffind',{'pattern':'recovered_'+tag}),action('read',{'path':path}),finish(value)];prompt='Try the stated missing path, then recover by discovering the repository file and finish with its value.';family='repo-missing-read-recovery'
 elif kind==1:
  path=f'src/state_{tag}.txt';files[path]='alpha\n';expected[path]='beta\n';steps=[action('edit',{'path':path,'edits':[{'oldText':'absent','newText':'beta'}]}),action('edit',{'path':path,'edits':[{'oldText':'alpha','newText':'beta'}]}),action('read',{'path':path}),finish('done')];prompt=f'Correct {path} from alpha to beta. If an exact replacement fails, inspect/recover and finish with exactly done.';family='repo-edit-error-recovery'
 elif kind==2:
  steps=[action('bash',{'command':"printf 'expected failure\\n'; false"}),action('bash',{'command':"printf 'RECOVERED\\n'"}),finish('RECOVERED')];prompt='Run the supplied diagnostic failure, observe it, then run the recovery command and finish with exactly RECOVERED.';family='shell-error-recovery'
 elif kind==3:
  prefix=0;notify={'onSuccess':'ignore','onFailure':'context','onKilled':'ignore'};steps=[action('process',{'action':'start','name':'ready-'+tag,'command':"printf 'READY\\n'; sleep 30",'notify':notify}),{'name':'process','dynamic':'process-output'},{'name':'process','dynamic':'process-stop'},finish('done')];prompt='Start the managed READY process, inspect its output using the returned process id, stop it, then finish with exactly done.';family='process-lifecycle'
 elif kind==4:
  path=f'pkg/value_{tag}.py';files[path]='VALUE = 3\n';expected[path]='VALUE = 9\n';steps=[action('fffind',{'pattern':'does_not_exist_'+tag}),action('ffgrep',{'pattern':'VALUE = 3','path':'pkg/'}),action('edit',{'path':path,'edits':[{'oldText':'VALUE = 3','newText':'VALUE = 9'}]}),action('read',{'path':path}),finish('done')];prompt='The initial filename hint may be wrong. Search, recover through content discovery, update VALUE from 3 to 9, verify it, and finish done.';family='repo-search-recovery'
 else:
  path=f'docs/{tag}.txt';files[path]='authoritative-'+tag+'\n';steps=[action('read',{'path':'README.missing'}),action('fffind',{'pattern':tag,'path':'docs/'}),action('read',{'path':path}),finish('authoritative-'+tag)];prompt='Recover from the unavailable README by locating the tagged document under docs and finish with exactly its content.';family='repo-doc-recovery'
 prompt=f'Task label {tag}. '+prompt
 expected={**files,**expected};return dict(id=cid,category='recovery',family=family,prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=prefix,requires_error=prefix>0,repository_discovery=kind in (0,4,5))

def web_case(i,port):
 cid=case_id('web',i);tag=opaque(cid,10);kind=i%16;files={};prefix=0
 sources=(('https://www.rfc-editor.org/rfc/rfc9110.txt','HTTP Semantics'),('https://www.rfc-editor.org/rfc/rfc9111.txt','HTTP Caching'),('https://www.rfc-editor.org/rfc/rfc9112.txt','HTTP/1.1'))
 url,fact=sources[i%len(sources)]
 if kind==0:
  query=f'official Python documentation release history {2020+i%5}';steps=[action('web_search',{'query':query,'numResults':3,'workflow':'none'}),{'name':'finish','dynamic':'observation'}];prompt=f'Request label {tag}. Research this externally verifiable question on the web and summarize the returned evidence: {query}';family='web-search-synthesis'
 elif kind==1:
  claim='Python was first released in 1991';steps=[action('source_check',{'claim':claim,'queries':['Python first release 1991 official history'],'numResults':3,'fetchContent':True}),{'name':'finish','dynamic':'observation'}];prompt=f'Request label {tag}. Check the supplied historical claim against web sources, then summarize only the returned evidence.';family='source-check';
 elif kind in (2,3,4,5,6,7):
  steps=[action('fetch_content',{'url':url,'mode':'readable'}),finish(fact)];prompt=f'Request label {tag}. Fetch {url}, find the exact page-local phrase {fact!r} in the returned content, and finish with exactly that phrase.';family='fetch-content'
 elif kind in (8,9):
  steps=[action('fetch_content',{'url':url,'mode':'readable'}),{'name':'get_search_content','dynamic':'response-id'},finish(fact)];prompt=f'Request label {tag}. Fetch {url}, retrieve its stored content using the returned responseId, verify the phrase {fact!r}, and finish with exactly that phrase.';family='fetch-retrieve'
 elif kind in (10,11,12):
  good=url;bad=f'https://example.invalid/{tag}.html';steps=[action('fetch_content',{'url':bad,'mode':'readable'}),action('fetch_content',{'url':good,'mode':'readable'}),finish(fact)];prompt=f'Request label {tag}. Observe the failure from {bad}, then fetch {good}, verify {fact!r} in its returned content, and finish with exactly that phrase.';family='fetch-error-recovery';prefix=1
 else:
  supplied='SUPPLIED_'+opaque(cid,14);fact=supplied;steps=[finish(fact)];prompt=f'The verified externally sourced fact is already supplied here: {fact}. Do not search; finish with exactly it.';family='web-direct-contrast'
 return dict(id=cid,category='web',family=family,prompt=prompt,files=files,expected_files={},steps=steps,supervise_from=prefix,requires_error=prefix>0,repository_discovery=False,web_page=None,expected_fact=fact)

def loopbreak_case(i):
 cid=case_id('loopbreak',i);tag=opaque(cid,10);kind=i%2;files={};expected={}
 if kind==0:
  wrong=f'src/entry_{tag}.json';right=f'config/actual_{tag}.json';value='ALT_'+opaque(cid,12)
  files[right]=value+'\n'
  steps=[action('read',{'path':wrong}),action('read',{'path':right}),finish(value)]
  prompt=f'Task label {tag}. The stated entry path {wrong} may be absent. If a read fails, do not repeat it; use the alternate path {right} stated here, then finish with exactly its value.'
 else:
  wrong=f'reports/summary_{tag}.txt';value='DISC_'+opaque(cid,12);right=f'archive/actual_{tag}.txt'
  files[right]=value+'\n'
  steps=[action('read',{'path':wrong}),action('fffind',{'pattern':'actual_'+tag}),action('read',{'path':right}),finish(value)]
  prompt=f'Task label {tag}. The stated report path {wrong} may be absent. If a read fails, do not repeat it; discover the actual file by fuzzy search and finish with exactly its value.'
 expected={**files,**expected};return dict(id=cid,category='loopbreak',family='tool-error-alternate-recovery',prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=1,requires_error=True,repository_discovery=False)

def make_loopbreak_cases(n):
 cases=[loopbreak_case(i) for i in range(n)]
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,{'loopbreak':n}

def pointerchase_case(i):
 cid=case_id('pointerchase',i);tag=opaque(cid,10);kind=i%3;files={};expected={}
 pointer_name=('catalog','registry','index')[kind]
 wrong=f'data/entry_{tag}.json';pointer=f'{pointer_name}_{tag}.json';real=f'src/actual_{tag}.json';value='PTR_'+opaque(cid,12)
 files[pointer]=json.dumps({'active_path':real})+'\n';files[real]=value+'\n'
 if kind==0:
  steps=[action('read',{'path':wrong}),action('read',{'path':pointer}),action('read',{'path':real}),finish(value)]
  prompt=f'Task label {tag}. The stated entry path {wrong} may be absent. If a read fails, do not repeat it; explore for a plausible pointer or catalog record that names the current location, follow it, then finish with exactly the value.'
 elif kind==1:
  steps=[action('read',{'path':wrong}),action('bash',{'command':'ls'}),action('read',{'path':pointer}),action('read',{'path':real}),finish(value)]
  prompt=f'Task label {tag}. The stated entry path {wrong} may be absent. If a read fails, do not repeat it; list the workspace to locate a pointer or catalog record, follow the path it names, then finish with exactly the value.'
 else:
  steps=[action('read',{'path':wrong}),action('ffgrep',{'pattern':'active_path'}),action('read',{'path':pointer}),action('read',{'path':real}),finish(value)]
  prompt=f'Task label {tag}. The stated entry path {wrong} may be absent. If a read fails, do not repeat it; search the workspace for a pointer record that names the current location, follow it, then finish with exactly the value.'
 expected={**files,**expected};return dict(id=cid,category='pointerchase',family='tool-error-pointer-chase',prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=1,requires_error=True,repository_discovery=False)

def make_pointerchase_cases(n):
 cases=[pointerchase_case(i) for i in range(n)]
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,{'pointerchase':n}

def make_cases(n,port):
 if n<20 or n%20:raise ValueError('records must be a multiple of20')
 counts={'local':7*n//20,'web':5*n//20,'exact':5*n//20};counts['recovery']=n-sum(counts.values());cases=[]
 for category,count,fn in (('local',counts['local'],local_case),('web',counts['web'],lambda i:web_case(i,port)),('exact',counts['exact'],exact_case),('recovery',counts['recovery'],recovery_case)):
  cases.extend(fn(i) for i in range(count))
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,counts

def frame(tools,name,args,commentary=None):
 return native_turn({'role':'assistant','content':commentary,'reasoning_content':None,'think':None,'tool_calls':[{'type':'function','function':{'name':name,'arguments':compact(args)}}]},tools)
def last_result(bridge):
 rows=[m for m in bridge.history if m['role']=='toolResult'];return rows[-1] if rows else None
def response_id(text):
 matches=re.findall(r'(?i)responseId["`\s:]+([A-Za-z0-9_-]{8,})',text)
 if not matches:
  try:
   obj=json.loads(text);value=obj.get('responseId')
   if isinstance(value,str):return value
  except Exception:pass
  raise ValueError('responseId absent from real Pi result')
 return matches[-1]
def process_id(text):
 matches=re.findall(r'proc_[A-Za-z0-9]+',text)
 if not matches:raise ValueError('process id absent')
 return matches[-1]
def observation_final(text):
 clean=' '.join(text.split());return ('Returned web evidence: '+clean[:500]).strip()

def encode_candidate(text,generations,supervise_from,enc):
 ids=enc.encode_ordinary(text);bound={0:0};position=0
 for index,token in enumerate(ids,1):position+=len(enc.decode_single_token_bytes(token));bound[position]=index
 raw=text.encode();mask=bytearray(len(ids));cursor=0;supervised=0
 for i,g in enumerate(generations):
  turn=enc.decode(g['token_ids']);needle=('\n\nAssistant:\n'+turn).encode();start=raw.find(needle,cursor)
  if start<0:raise ValueError('generated turn absent from record')
  left=start+len('\n\nAssistant:\n'.encode());right=start+len(needle);cursor=right
  if left not in bound or right not in bound:raise ValueError('assistant span token boundary')
  if i>=supervise_from:mask[bound[left]:bound[right]]=b'\1'*(bound[right]-bound[left]);supervised+=1
 if not supervised or not sum(mask) or len(ids)>65536 or mask[0]:raise ValueError('invalid candidate mask')
 return ids,bytes(mask),supervised

def dynamic_step(spec,bridge):
 if 'dynamic' not in spec:return spec
 result=last_result(bridge)
 if result is None:raise ValueError('dynamic action without observation')
 text=result['content'][0]['text']
 if spec['dynamic']=='observation':return finish(observation_final(text))
 if spec['dynamic']=='response-id':return action('get_search_content',{'responseId':response_id(text),'offset':0,'limit':2000})
 if spec['dynamic']=='process-output':return action('process',{'action':'output','id':process_id(text),'tailLines':20})
 if spec['dynamic']=='process-stop':return action('process',{'action':'stop','id':process_id(text)})
 raise ValueError('unknown dynamic action')

def execute_case(case,panel,enc,root,pi_bin,manifest_path):
 workspace=root/'workspace';workspace.mkdir(parents=True);pages=root.parent/'web-pages'
 for name,text in case['files'].items():p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 index=0;emitted=[];emitted_specs=[]
 def generate(prompt,budget,deadline):
  nonlocal index
  spec=dynamic_step(case['steps'][index],bridge);comment='Using the verified observation.' if index and last_result(bridge) else None;text=frame(panel['tools'],spec['name'],spec['arguments'],comment);ids=enc.encode_ordinary(text)
  if len(ids)>budget:raise ValueError('authored turn budget')
  emitted.append(text);emitted_specs.append(spec);index+=1;return text,ids,'valid'
 bridge=NativePiToolBridge(panel,case['prompt'],enc,generate)
 terminal=serve_pi_native_tools(bridge,root/'pi',pi_bin=pi_bin,provider_extension=Path('configs/pi/e97-pi-native.ts'),pi_extensions=EXTENSIONS,cwd=workspace,seconds=180,extra_env={'E97_PI_TOOL_MANIFEST':str(manifest_path.resolve())})
 if not terminal['close_verified'] or index!=len(case['steps']) or bridge.final is None:raise ValueError('incomplete authored trajectory')
 results=[m for m in bridge.history if m['role']=='toolResult'];calls=[b for m in bridge.history if m['role']=='assistant' for b in m['content'] if b['type']=='toolCall'];expected_calls=[s for s in emitted_specs if s['name']!='finish']
 if len(results)!=len(calls) or any(r['toolCallId']!=c['id'] or r['toolName']!=c['name'] for c,r in zip(calls,results)):raise ValueError('call/result pairing')
 if len(calls)!=len(expected_calls) or any(c['name']!=s['name'] or c['arguments']!=s['arguments'] for c,s in zip(calls,expected_calls)):raise ValueError('Pi executed different action')
 if bridge.final!=emitted_specs[-1]['arguments']['message']:raise ValueError('terminal final mismatch')
 prefix_calls=sum(s['name']!='finish' for s in emitted_specs[:case['supervise_from']])
 failed=lambda r:r['isError'] or r['content'][0]['text'].lower().startswith(('error:','tool error:')) or any(w in r['content'][0]['text'].lower() for w in ('exit code: 1','command exited with code 1','no matches found','no files found matching pattern'))
 if case['requires_error'] and (prefix_calls<1 or not any(failed(r) for r in results[:prefix_calls])):raise ValueError('authored failure prefix did not fail')
 if any(failed(r) for r in results[prefix_calls:]):raise ValueError('unexpected tool error after correction prefix')
 snapshot={}
 for name,want in case['expected_files'].items():snapshot[name]=(workspace/name).read_text() if (workspace/name).exists() else None
 if snapshot!=case['expected_files']:raise ValueError('workspace oracle')
 if case['category']=='web' and case['family'] in ('fetch-content','fetch-retrieve','fetch-error-recovery'):
  if not any(case['expected_fact'] in r['content'][0]['text'] for r in results):raise ValueError('web fact not observed')
 ids,mask,units=encode_candidate(bridge.episode.text(),bridge.generations,case['supervise_from'],enc)
 private={'id':case['id'],'category':case['category'],'family':case['family'],'prompt':case['prompt'],'source_messages':bridge.episode.source_messages(),'native_record':bridge.episode.text(),'generations':bridge.generations,'public_history':bridge.history,'terminal':terminal,'snapshot':snapshot,'supervise_from':case['supervise_from'],'assistant_units':len(bridge.generations),'supervised_assistant_units':units,'targets':sum(mask),'record_sha256':hashlib.sha256(bridge.episode.text().encode()).hexdigest()}
 publish(root/'episode-private.json',private);shutil.rmtree(workspace)
 return dict(id=case['id'],category=case['category'],family=case['family'],tokens=np.asarray(ids,dtype='<u4').tobytes(),mask=mask,targets=sum(mask),assistant_units=len(bridge.generations),supervised_units=units,errors=sum(r['isError'] for r in results),calls=len(calls),episode_sha256=sha(root/'episode-private.json'),sequence_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()+mask).hexdigest(),repository_discovery=case['repository_discovery'])

def write_authority(rows,plan,output):
 authority=output/'candidate-authority';authority.mkdir();paths={'tokens':authority/'tokens.uint32.bin','mask':authority/'assistant_mask.uint8.bin','index':authority/'records.idx','metadata':authority/'records.jsonl'};offset=0;targets=0
 with paths['tokens'].open('xb') as tf,paths['mask'].open('xb') as mf,paths['index'].open('xb') as ix,paths['metadata'].open('x') as meta:
  for row in rows:
   n=len(row['mask']);tf.write(row['tokens']);mf.write(row['mask']);ix.write(RECORD_INDEX.pack(offset,n,row['targets'],0));meta.write(json.dumps({k:v for k,v in row.items() if k not in ('tokens','mask')},sort_keys=True)+'\n');offset+=n;targets+=row['targets']
 outputs={k:{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()}
 publish(authority/'manifest.json',{'schema':'emender-e97-pi-native-candidate-authority-v1','status':'verified-candidate-not-admitted','training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,'tokenizer':'p50k_base','plan_sha256':plan['plan_sha256'],'counts':{'records':len(rows),'tokens':offset,'assistant_target_tokens':targets},'outputs':outputs})

def authority_files():
 paths=[Path(__file__),Path('configs/pi/e97-pi-native.ts'),Path('scripts/e97_pi_native_codec.py'),Path('scripts/e97_pi_native_tool_bridge.py'),Path('scripts/e97_pi_native_tool_transport.py'),*EXTENSIONS]
 return {str(p.resolve()):sha(p) for p in paths}
def freeze(args):
 if sha(args.manifest)!=MANIFEST_SHA:raise ValueError('tool authority')
 minimum=args.minimum_verified_records if args.minimum_verified_records is not None else args.records
 if not 1<=minimum<=args.records:raise ValueError('minimum verified records')
 mix=getattr(args,'mix','standard')
 if mix in ('loopbreak','pointerchase'):
  cases,counts=make_loopbreak_cases(args.records) if mix=='loopbreak' else make_pointerchase_cases(args.records)
 elif mix=='standard':cases,counts=make_cases(args.records,args.port)
 else:raise ValueError('unknown mix')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);plan={'schema':'emender-e97-pi-native-curriculum-plan-v1','seed':SEED,'records':args.records,'minimum_verified_records':minimum,'automatic_retry':False,'mix_counts':counts,'port':args.port,'system':SYSTEM,'tool_manifest':str(args.manifest.resolve()),'tool_manifest_sha256':sha(args.manifest),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'authority_files':authority_files(),'pi_bin':str(args.pi_bin.resolve()),'pi_version':subprocess.check_output([args.pi_bin,'--version'],text=True).strip(),'cases':cases,'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False};publish(args.output/'plan-private.json',plan);print('PI_NATIVE_CURRICULUM_PLAN',args.records,minimum,sha(args.output/'plan-private.json'))
def collect(args):
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 plan=json.loads(args.plan.read_text());manifest_path=Path(plan['tool_manifest'])
 if sha(manifest_path)!=plan['tool_manifest_sha256']==MANIFEST_SHA or plan['authority_files']!=authority_files() or str(args.pi_bin.resolve())!=plan['pi_bin'] or subprocess.check_output([args.pi_bin,'--version'],text=True).strip()!=plan['pi_version'] or plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates']:raise ValueError('authority')
 manifest=json.loads(manifest_path.read_text());panel={'system':plan['system'],'tools':manifest['model_visible_tools'],'max_turns':10,'generation_budget':2048,'episode_generation_budget':8192,'episode_seconds':150};args.output.mkdir(parents=True,mode=0o700,exist_ok=False);rows=[];rejections=[]
 enc=tiktoken.get_encoding('p50k_base')
 for i,case in enumerate(plan['cases']):
  try:rows.append(execute_case(case,panel,enc,args.output/case['id'],args.pi_bin,manifest_path))
  except ValueError as exc:
   rejection={'id':case['id'],'category':case['category'],'family':case['family'],'type':type(exc).__name__,'message':str(exc),'retried':False};rejections.append(rejection);publish(args.output/case['id']/'rejection.json',rejection)
  if (i+1)%20==0:print('PI_NATIVE_ATTEMPTS',i+1,'VERIFIED',len(rows),'REJECTED',len(rejections),flush=True)
 (args.output/'rejections.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rejections))
 if len(rows)<plan['minimum_verified_records']:raise ValueError('minimum verified curriculum not reached')
 if len({r['sequence_sha256'] for r in rows})!=len(rows):raise ValueError('dedup')
 write_authority(rows,{**plan,'plan_sha256':args.plan_sha},args.output);counts=Counter(r['category'] for r in rows);families=Counter(r['family'] for r in rows);summary={'schema':'emender-e97-pi-native-curriculum-summary-v1','status':'verified-candidates-not-admitted','attempted_records':plan['records'],'records':len(rows),'rejected_records':len(rejections),'automatic_retries':0,'minimum_verified_records':plan['minimum_verified_records'],'mix_counts':dict(counts),'families':dict(families),'repository_discovery_records':sum(r['repository_discovery'] for r in rows),'native_calls':sum(r['calls'] for r in rows),'authentic_tool_errors':sum(r['errors'] for r in rows),'assistant_targets':sum(r['targets'] for r in rows),'deduplicated_sequences':len(rows),'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False,'checkpoint_promotion':False,'authority_sha256':sha(args.output/'candidate-authority/manifest.json')};publish(args.output/'summary.json',summary);print('PI_NATIVE_CURRICULUM_COLLECTED',len(rows),len(rejections),summary['native_calls'],summary['assistant_targets'])
def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')));p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True);f=sp.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--records',type=int,required=True);f.add_argument('--minimum-verified-records',type=int);f.add_argument('--mix',choices=('standard','loopbreak','pointerchase'),default='standard');f.add_argument('--port',type=int,required=True);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--output',type=Path,required=True);c=sp.add_parser('collect');c.add_argument('--plan',type=Path,required=True);c.add_argument('--plan-sha',required=True);c.add_argument('--pi-bin',type=Path,required=True);c.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a) if a.command=='freeze' else collect(a)
if __name__=='__main__':main()
