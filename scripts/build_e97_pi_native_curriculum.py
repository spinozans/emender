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

def extracterror_case(i):
 cid=case_id('extracterror',i);tag=opaque(cid,10);kind=i%3;files={};expected={}
 value='XVAL_'+opaque(cid,14);decoys=['XVAL_'+opaque(cid+'x',14),('XVAL_'+opaque(cid+'y',14))]
 if kind==0:
  path=f'config/options_{tag}.json'
  files[path]=json.dumps({'chosen':'silver','options':{'amber':decoys[0],'silver':value,'violet':decoys[1]}})+'\n'
  prompt=f'Task label {tag}. Read {path} and extract the value selected by the chosen key, then finish with exactly that value. If a helper command fails, never finish with an error; recover from what you already observed.'
  steps=[action('read',{'path':path}),action('bash',{'command':f"python3 -c 'import json; d=json.load(open({path!r})); print(d['options'[d['chosen']])'"}),finish(value)]
 elif kind==1:
  path=f'data/state_{tag}.json';inner=f'src/live_{tag}.json'
  files[path]=json.dumps({'active_path':inner})+'\n';files[inner]=value+'\n'
  prompt=f'Task label {tag}. Read {path}, follow the pointer it names, and finish with exactly the value at the current location. If a helper command fails, never finish with an error; recover from what you already observed.'
  steps=[action('read',{'path':path}),action('bash',{'command':f"python3 -c 'import json; d=json.load(open({path!r})); print(json.load(open(d[\"active_path\"])))'"}),action('read',{'path':inner}),finish(value)]
 else:
  path=f'records/rows_{tag}.json'
  files[path]=json.dumps([{'id':0,'v':decoys[0]},{'id':1,'v':value},{'id':2,'v':decoys[1]}])+'\n'
  prompt=f'Task label {tag}. Read {path} and finish with exactly the value of the row whose id is 1. If a helper command fails, never finish with an error; recover from what you already observed.'
  steps=[action('read',{'path':path}),action('bash',{'command':f"python3 -c 'import json; rows=json.load(open({path!r})); print([r['v' for r in rows if r['id']==1][0])'"}),finish(value)]
 expected={**files,**expected};return dict(id=cid,category='extracterror',family='tool-error-finish-recovery',prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=2,requires_error=True,repository_discovery=False)

def make_extracterror_cases(n):
 cases=[extracterror_case(i) for i in range(n)]
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,{'extracterror':n}

def longcopy_case(i):
 cid=case_id('longcopy',i);distance=(1024,4096,8192,16384,32768)[i%5];pad=filler(distance,cid)
 value='COPY_'+opaque(cid,20)
 prompt=f'Remember exactly {value}.{pad}\nFinish with exactly the remembered value and no other text.'
 return dict(id=cid,category='exact',family='long-delay-copy',prompt=prompt,files={},expected_files={},steps=[finish(value)],supervise_from=0,requires_error=False,repository_discovery=False,delay_tokens=distance)

def make_longcopy_cases(n):
 cases=[longcopy_case(i) for i in range(n)]
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,{'longcopy':n}

def grounded(step,analysis,commentary,requires):
 d=dict(step);d['analysis']=analysis;d['commentary']=commentary;d['analysis_requires']=requires;return d

def reasoning_case(i):
 cid=case_id('reasoning',i);tag=opaque(cid,10);kind=i%5;files={};expected={};prefix=0
 if kind in (0,1):
  port=8000+i*7+kind;path=f'config/app_{tag}.ini';src=f'src/greeter_{tag}.py';test=f'tests/test_greeter_{tag}.py'
  files[path]=f'[service]\nport = {port}\n';files[src]='PORT = None\n\ndef greet():\n    return f"service-on-{PORT}"\n'
  files[test]=f'import unittest,sys\nsys.path.insert(0,"src")\nfrom greeter_{tag} import PORT,greet\nclass T(unittest.TestCase):\n def test_port(self): self.assertEqual(PORT,{port})\n def test_greet(self): self.assertEqual(greet(),"service-on-{port}")\n'
  expected[src]=files[src].replace('PORT = None',f'PORT = {port}')
  steps=[grounded(action('read',{'path':path}),
    f'The checker expects the service port, which lives in config, not in the source. Reading {path} first so the edit uses the real value instead of a guess.',
    'Reading the config to learn the port.',['config']),
   grounded(action('bash',{'command':f'python3 -m unittest {test}'}),
    f'Before editing I run the unittest checker to observe the exact defect; with PORT still None it must fail, confirming the config value {port} is the only missing piece.',
    'Running the checker to observe the defect.',['unittest']),
   grounded(action('edit',{'path':src,'edits':[{'oldText':'PORT = None','newText':f'PORT = {port}'}]}),
    f'config sets port={port}; replacing the None placeholder with the observed {port} satisfies both assertions without touching the checker.',
    f'Wiring the observed port {port} into the source.',[str(port)]),
   grounded(action('bash',{'command':f'python3 -m unittest {test}'}),
    'Rerunning the same checker verifies the edit; finishing before a green run would be unverified.',
    'Verifying the repair.',[]),
   finish('done')]
  prompt=f'Fix {src} so its checker {test} passes. The port must come from the config file, not from guessing; read the config, observe the failing checker, edit, and verify before finishing with exactly done.'
  family='reasoning-config-derive';prefix=2
 elif kind in (2,3):
  a=120+i*11+kind;b=58+(i%23);total=a+b;data=f'data/inputs_{tag}.txt';report=f'reports/sum_{tag}.txt';index_md='INDEX.md'
  files[data]=f'a = {a}\nb = {b}\n';files[index_md]='# Ledger\nReports appear here after they are written.\n'
  expected[report]=f'sum = {total}\n';expected[index_md]=f'# Ledger\nReports appear here after they are written.\n- reports/sum_{tag}.txt\n'
  steps=[grounded(action('read',{'path':data}),
    f'The report value is derived from two inputs; reading {data} gives the operands, and everything later must quote the observed numbers.',
    'Reading the operands.',['inputs_']),
   grounded(action('bash',{'command':f'printf "%s\\n" $(({a}+{b}))'}),
    f'Operands are {a} and {b}; computing the sum with bash instead of mental arithmetic makes the observation authoritative before anything is written.',
    'Computing the sum from the observed operands.',[str(a),str(b)]),
   grounded(action('write',{'path':report,'content':f'sum = {total}\n'}),
    f'The observed sum is {total}; writing exactly that line to {report} keeps the report grounded in the computation.',
    f'Writing the report with the observed sum {total}.',[str(total)]),
   grounded(action('edit',{'path':index_md,'edits':[{'oldText':'Reports appear here after they are written.','newText':f'Reports appear here after they are written.\n- reports/sum_{tag}.txt'}]}),
    f'The ledger must reference the new report; appending {report} to the index keeps it truthful.',
    'Registering the report in the ledger.',[f'reports/sum_{tag}.txt']),
   grounded(action('read',{'path':report}),
    'Reading the written report back verifies the file state before finishing.',
    'Reading the report back.',[]),
   finish(f'sum = {total}')]
  prompt=f'Read {data}, compute the sum of its two operands with bash, write a report {report} containing exactly "sum = <total>" on one line, register the report path in {index_md}, verify by reading it back, then finish with exactly the report line.'
  family='reasoning-cross-file-compose'
 else:
  va='ALPHA_'+opaque(cid+':a',8);vb='BETA_'+opaque(cid+':b',8);fa=f'alpha_{tag}.txt';fb=f'beta/{tag}/beta.txt';composed=va+'::'+vb
  files[fa]=va+'\n';files[fb]=vb+'\n'
  steps=[grounded(action('bash',{'command':f'cat composed/{tag}.txt'}),
    f'The composed value needs both tokens; the guessed path composed/{tag}.txt may not exist, and if it fails I must switch to search instead of repeating it.',
    'Trying the composed path.',[]),
   grounded(action('ffgrep',{'pattern':'ALPHA_'}),
    'The direct path failed, so repeating it is prohibited; searching for the ALPHA_ marker locates the first token file.',
    'Searching for the alpha token.',['ALPHA_']),
   grounded(action('read',{'path':fa}),
    f'The search hit names {fa}; reading that exact file yields the first token without trusting the hit alone.',
    'Reading the alpha token.',[fa]),
   grounded(action('ffgrep',{'pattern':'BETA_'}),
    f'With the first token {va} observed, the same search-and-read discipline locates the second file via its BETA_ marker.',
    'Searching for the beta token.',[va,'BETA_']),
   grounded(action('read',{'path':fb}),
    f'The second search hit names {fb}; reading it gives the second exact token, and both tokens are then observed, so the composed value can be finished verbatim.',
    'Reading the beta token.',[fb]),
   finish(composed)]
  prompt=f'The value {composed[:4]}... is split across two token files in this workspace. Discover both tokens by search and reading (the direct composed path may be absent; never repeat a failed call), then finish with exactly firsttoken::secondtoken.'
  family='reasoning-recovery-compose';prefix=1
 expected={**files,**expected}
 return dict(id=cid,category='reasoning',family=family,prompt=prompt,files=files,expected_files=expected,steps=steps,supervise_from=prefix,requires_error=prefix>0,repository_discovery=False)

def make_reasoning_cases(n):
 cases=[reasoning_case(i) for i in range(n)]
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,{'reasoning-compose':sum(1 for c in cases if c['family']=='reasoning-config-derive'),'reasoning-cross-file':sum(1 for c in cases if c['family']=='reasoning-cross-file-compose'),'reasoning-recovery':sum(1 for c in cases if c['family']=='reasoning-recovery-compose')}

def hybrid_case(i):
 cid=case_id('hybrid',i);tag=opaque(cid,10);files={};expected={}
 if i==0:
  # The date-question specimen from the operator's interactive findings: the
  # current date is environment state, so the model must reason privately,
  # observe bash date, and answer in plain text with exactly the observation.
  answer_dynamic={'name':'finish','dynamic':'observation-exact','analysis_dynamic':'observation','commentary_dynamic':'observation'}
  steps=[grounded(action('bash',{'command':"date -u '+%Y-%m-%d'"}),
   'The current date is environment state I cannot know from the prompt alone; the minimal tool to observe it is bash date.',
   'Checking the current UTC date.',[]),answer_dynamic]
  prompt="What is today's date in UTC? Answer in plain text."
  return dict(id=cid,category='hybrid',family='hybrid-date-question',prompt=prompt,files=files,expected_files=dict(files),steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,answer_is_observation=True)
 kind=(i-1)%6
 if kind==0:
  path=f'config/service_{tag}.ini';port=10000+(i*379)%50000
  files[path]=f'[service]\nname = svc_{tag}\nport = {port}\n'
  steps=[grounded(action('read',{'path':path}),
   f'The user asks for a fact stored in the workspace. Reading {path} observes the declared port directly instead of guessing it.',
   'Reading the config to observe the port.',[f'service_{tag}']),
   grounded(finish(str(port)),
   f'The read observed port = {port}; the plain-text answer is exactly that observed number.',
   f'The config declares port {port}.',[str(port)])]
  prompt=f'What port does the config at {path} declare? Answer with only the port number in plain text.'
  family='hybrid-read-fact'
 elif kind==1:
  path=f'docs/notes_{tag}.txt';count=3+((i-1)%9)
  files[path]=''.join(f'note {j}: {opaque(cid+str(j),6)}\n' for j in range(1,count+1))
  steps=[grounded(action('bash',{'command':f'wc -l < {path}'}),
   f'The line count is a fact about the workspace file {path}; observing it with wc -l keeps the answer grounded instead of counted by hand.',
   'Counting the lines with wc.',[f'notes_{tag}','wc -l']),
   grounded(finish(str(count)),
   f'The command observed {count}; the plain-text answer is exactly that observed count.',
   f'The file has {count} lines.',[str(count)])]
  prompt=f'How many lines does {path} have? Answer with only the number in plain text.'
  family='hybrid-bash-count'
 elif kind==2:
  path=f'notes/release_{tag}.txt';value='REL_'+opaque(cid,14)
  files[path]=f'release = {value}\n'
  steps=[grounded(action('read',{'path':path}),
   f'The release marker is recorded in {path}; reading it makes the answer the observed value rather than a recollection.',
   'Reading the release note.',[f'release_{tag}']),
   grounded(finish(value),
   f'The read observed release = {value}; answering in plain text with exactly that observed marker.',
   f'The release marker is {value}.',[value])]
  prompt=f'What release marker does {path} record? Answer with only the marker in plain text.'
  family='hybrid-read-fact'
 elif kind==3:
  value='LEDGER_'+opaque(cid,14)
  steps=[grounded(finish(value),
   f'The user supplied the fact directly in the conversation: the recorded value is {value}. No tool is needed; answering in plain text from what was given.',
   f'The recorded value is {value}.',[value])]
  prompt=f'The workspace ledger records the value {value}. What value does the ledger record? Answer in plain text without using any tools.'
  family='hybrid-pure-chat-supplied'
 elif kind==4:
  alpha='APPROVED_'+opaque(cid+':a',10);beta='DRAFT_'+opaque(cid+':b',10)
  steps=[grounded(finish(alpha),
   f'Both labels are supplied in the conversation; the approved one is {alpha}, so no tool is needed and the plain-text answer is exactly that label.',
   f'The approved label is {alpha}.',[alpha])]
  prompt=f'For ticket {tag}, the approved label is {alpha} and the draft label is {beta}. Return only the approved label in plain text without using any tools.'
  family='hybrid-pure-chat-selection'
 else:
  path=f'data/ledger_{tag}.txt';a=100+(i-1)*7;b=43+((i-1)%31);total=a+b
  files[path]=f'a = {a}\nb = {b}\n'
  steps=[grounded(action('read',{'path':path}),
   f'The user asks for a derived fact; reading {path} observes the two operands before any arithmetic is trusted.',
   'Reading the recorded operands.',[f'ledger_{tag}']),
   grounded(action('bash',{'command':f'printf \'%s\\n\' $(({a}+{b}))'}),
   f'The read observed a = {a} and b = {b}; computing their sum with bash makes the result an observed fact instead of mental arithmetic.',
   'Computing the sum from the observed operands.',[str(a),str(b)]),
   grounded(finish(str(total)),
   f'The command observed {total}; the plain-text answer is exactly that observed sum.',
   f'The sum is {total}.',[str(total)])]
  prompt=f'Open {path}, add its two operands, and answer with only the sum in plain text.'
  family='hybrid-bash-derive'
 return dict(id=cid,category='hybrid',family=family,prompt=prompt,files=files,expected_files=dict(files),steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,
  pure_chat=kind in (3,4),answer_must_be_observed=kind in (1,5),expected_answer=None if i==0 else (steps[-1]['arguments']['message'] if 'arguments' in steps[-1] else None))

def teacher_hybrid_case(t):
 """Convert one validated teacher task specification into a hybrid case."""
 files={p:c for p,c in t['workspace_files'].items()}
 answer=t['expected_answer'];literal=t['observed_literal']
 if t['kind']=='pure-chat':
  steps=[grounded(finish(answer),
   f"The fact is supplied in the conversation: {t['analysis_focus']} The plain-text answer is {answer} and no tool is needed.",
   t['answer_commentary'],[literal])]
  return dict(id=t['id'],category='hybrid',family=t['family'],prompt=t['user_question'],files={},expected_files={},steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,pure_chat=True,expected_answer=answer)
 path=sorted(files)[0]
 if t['expected_tool']=='read':
  first=grounded(action('read',{'path':path}),
   f"{t['analysis_focus']} Reading {path} observes the recorded fact directly instead of recalling it.",
   'Reading the file to observe the fact.',[literal])
 else:
  first=grounded(action('bash',{'command':t['bash_command']}),
   f"{t['analysis_focus']} Running {t['bash_command']} observes the answer in the workspace rather than deriving it from memory.",
   'Observing the answer with bash.',[literal])
 steps=[first,grounded(finish(answer),
  f'The observed result contains {literal}; the plain-text answer is exactly {answer}.',
  t['answer_commentary'],[answer])]
 return dict(id=t['id'],category='hybrid',family=t['family'],prompt=t['user_question'],files=files,expected_files=dict(files),steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,answer_must_be_observed=True,expected_answer=answer)

def make_hybrid_cases(n,teacher_tasks):
 authored=[hybrid_case(i) for i in range(n)]
 cases=authored+[teacher_hybrid_case(t) for t in teacher_tasks]
 if len({c['id'] for c in cases})!=len(cases):raise ValueError('case identity')
 counts={}
 for c in cases:counts[c['family']]=counts.get(c['family'],0)+1
 return cases,{'hybrid-authored':len(authored),'hybrid-teacher':len(teacher_tasks),**counts}

def make_cases(n,port):
 if n<20 or n%20:raise ValueError('records must be a multiple of20')
 counts={'local':7*n//20,'web':5*n//20,'exact':5*n//20};counts['recovery']=n-sum(counts.values());cases=[]
 for category,count,fn in (('local',counts['local'],local_case),('web',counts['web'],lambda i:web_case(i,port)),('exact',counts['exact'],exact_case),('recovery',counts['recovery'],recovery_case)):
  cases.extend(fn(i) for i in range(count))
 if len({c['id'] for c in cases})!=n:raise ValueError('case identity')
 return cases,counts

def frame(tools,name,args,commentary=None,analysis=None):
 return native_turn({'role':'assistant','content':commentary,'reasoning_content':analysis,'think':None,'tool_calls':[{'type':'function','function':{'name':name,'arguments':compact(args)}}]},tools)
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
 if spec['dynamic']=='observation-exact':return finish(text.strip())
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
  spec=dynamic_step(case['steps'][index],bridge)
  observed=last_result(bridge)['content'][0]['text'].strip() if last_result(bridge) else None
  if spec.get('analysis_dynamic')=='observation':
   if observed is None:raise ValueError('dynamic analysis without observation')
   spec=dict(spec);spec['analysis']=f'The tool observed {observed}; the user-facing answer is exactly that observed value.'
  if spec.get('commentary_dynamic')=='observation':
   if observed is None:raise ValueError('dynamic commentary without observation')
   spec=dict(spec);spec['commentary']=f'The observed answer is {observed}.'
  comment=spec.get('commentary') if 'commentary' in spec else ('Using the verified observation.' if index and last_result(bridge) else None)
  analysis=spec.get('analysis')
  if analysis is not None:
   required=spec.get('analysis_requires') or []
   if not analysis.strip() or any(x not in analysis for x in required):raise ValueError('ungrounded analysis')
  text=frame(panel['tools'],spec['name'],spec['arguments'],comment,analysis);ids=enc.encode_ordinary(text)
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
 if case.get('answer_must_be_observed'):
  if case.get('expected_answer') is None or not any(str(case['expected_answer']) in r['content'][0]['text'] for r in results):raise ValueError('answer not observed')
 if case.get('answer_is_observation'):
  if not results or bridge.final!=results[-1]['content'][0]['text'].strip():raise ValueError('answer is not the observation')
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
 if mix in ('loopbreak','pointerchase','extracterror','longcopy','reasoning','hybrid'):
  if mix=='loopbreak':cases,counts=make_loopbreak_cases(args.records)
  elif mix=='pointerchase':cases,counts=make_pointerchase_cases(args.records)
  elif mix=='extracterror':cases,counts=make_extracterror_cases(args.records)
  elif mix=='reasoning':cases,counts=make_reasoning_cases(args.records)
  elif mix=='hybrid':
   teacher_tasks=[]
   if getattr(args,'teacher_tasks',None) is not None:
    pool_path=Path(args.teacher_tasks)
    if sha(pool_path)!=args.teacher_tasks_sha:raise ValueError('teacher pool identity')
    pool=json.loads(pool_path.read_text())
    if pool.get('schema')!='emender-e97-hybrid-conversation-teacher-pool-v1':raise ValueError('teacher pool schema')
    teacher_tasks=pool['tasks']
   cases,counts=make_hybrid_cases(args.records,teacher_tasks)
  else:cases,counts=make_longcopy_cases(args.records)
 elif mix=='standard':cases,counts=make_cases(args.records,args.port)
 else:raise ValueError('unknown mix')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);plan={'schema':'emender-e97-pi-native-curriculum-plan-v1','seed':SEED,'records':args.records,'minimum_verified_records':minimum,'automatic_retry':False,'mix':mix,'mix_counts':counts,'port':args.port,'system':SYSTEM,'tool_manifest':str(args.manifest.resolve()),'tool_manifest_sha256':sha(args.manifest),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'authority_files':authority_files(),'pi_bin':str(args.pi_bin.resolve()),'pi_version':subprocess.check_output([args.pi_bin,'--version'],text=True).strip(),'cases':cases,'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False};publish(args.output/'plan-private.json',plan);print('PI_NATIVE_CURRICULUM_PLAN',args.records,minimum,sha(args.output/'plan-private.json'))
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
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')));p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True);f=sp.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--records',type=int,required=True);f.add_argument('--minimum-verified-records',type=int);f.add_argument('--mix',choices=('standard','loopbreak','pointerchase','extracterror','longcopy','reasoning','hybrid'),default='standard');f.add_argument('--port',type=int,required=True);f.add_argument('--teacher-tasks',type=Path,default=None);f.add_argument('--teacher-tasks-sha',default=None);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--output',type=Path,required=True);c=sp.add_parser('collect');c.add_argument('--plan',type=Path,required=True);c.add_argument('--plan-sha',required=True);c.add_argument('--pi-bin',type=Path,required=True);c.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a) if a.command=='freeze' else collect(a)
if __name__=='__main__':main()
