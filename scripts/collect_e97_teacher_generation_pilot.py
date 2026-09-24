#!/usr/bin/env python3
"""E97 interleaved teacher-generation pilot v1 (operator design).

Generator+guide arrangement on LunarRoute background slots:
  - GENERATOR teacher (lunaroute deepseek-4.1-flash[-background]) attempts
    terminal-bench-STYLE tasks THROUGH THE REAL PI SANDBOX (owner bridge +
    real pi child executes the eleven-tool surface; every tool result is a
    real execution, never synthetic) and writes multi-turn conversations
    seeded from real WildChat user openings.
  - GUIDE teacher (lunaroute glm-5.3-flash[-background]) reviews each trace:
    is it a good exemplar, which transitions are model-worthy, where drifts.
  - Mechanical verification is the final bar: terminal traces enter the
    corpus only if the task actually completed (driver-side authored
    assertions) AND the guide rates the trace exemplar; conversation records
    must satisfy the hybrid-family structural + anti-fixture guards AND the
    guide register/drift review.

Task authoring is terminal-bench-STYLE and NEW: deterministic authored
generators in this file (seeded synthetic names/values, phrasing banks).
No Terminal-Bench benchmark instance is consulted or reproduced
(benchmark-contamination discipline: we may later evaluate on the real
benchmark). Evaluation-entity discipline reuses the frozen forbidden
literal/number sets from the teacher pilot + hybrid v2 collections.

Terminal records render in the canonical Pi-native codec authority layout
(tokens.uint32.bin + assistant_mask.uint8.bin + records.idx +
records.jsonl, p50k_base). Conversation records render in the conversation
SFT authority layout ("User:\\n...\\n\\nAssistant:\\n..." + RS).

Nothing is admitted to training. No optimizer updates. CPU only.

Commands:
  freeze-terminal        author + freeze the terminal task plan
  collect-terminal       run the terminal arm (checkpoint/resume)
  freeze-conversations   sample real WildChat openings, freeze plan
  collect-conversations  run the conversation arm (checkpoint/resume)
  smoke                  one terminal case + one conversation end-to-end
  report                 measured rates + report authoring inputs
"""
import argparse,fcntl,hashlib,json,os,random,re,shutil,signal,struct,subprocess,threading,time
import urllib.error,urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pyarrow.parquet as pq
import tiktoken

from ndm.data.masked_sft_dataset import RECORD_INDEX
from scripts import build_e97_tulu3_sft as convcodec
from scripts.e97_open_swe_native_codec import compact,strict_json
from scripts.e97_pi_native_codec import validate_generated_turn
from scripts.e97_pi_native_tool_bridge import BridgeStopped,NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.build_e97_hybrid_conversation_collection_v2 import check_text
from scripts.build_e97_pi_native_curriculum import EXTENSIONS,MANIFEST_SHA,SYSTEM,encode_candidate,write_authority
from scripts.eval_e97_native_execution import publish,sha

PILOT_SEED='e97-teacher-generation-pilot-v1-20260924'
PROVIDER_V2=Path('configs/pi/e97-pi-native-frozen-tools-v2.ts')
TERMINAL_PLAN_SCHEMA='emender-teacher-pilot-terminal-plan-v1'
CONVERSATION_PLAN_SCHEMA='emender-teacher-pilot-conversation-plan-v1'
GENERATOR_DEFAULT='deepseek-4.1-flash-background'
GUIDE_DEFAULT='glm-5.3-flash-background'
WILDCHAT_RAW=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-real-human-chat-intake-v1/raw/wildchat/data')
WILDCHAT_ID='allenai/WildChat'
WILDCHAT_REVISION='f66566ceaaeb619dd98ffb0f3bf5ce1f86775ac4'
TERMINAL_PANEL={'system':SYSTEM,'max_turns':16,'generation_budget':3072,'episode_generation_budget':24576,'episode_seconds':480}
# Terminal-bench-style task authoring uses only synthetic seeded names/values
# plus these generic words; nothing is copied from any benchmark suite.
NAME_WORDS=('harbor','willow','cobalt','juniper','lantern','saffron','quartz','marlin','orchid','pebble','sundial','tundra','cinnamon','gazelle','basalt','coral','fennel','granite','ivy','lagoon','magnolia','nettle','opal','plume','ridge','sorrel','thistle','umbral','vellum','walnut','yarrow','zephyr')
OPENINGS=('Please','I need','Can you','Could you','Hey,','')
GROUPS=('alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel')
# Standalone three-digit evaluation entities (frozen panel + teacher-pilot
# discipline) must never appear in authored task values.
FORBIDDEN_3DIGIT=frozenset(('157','284','441','731','992'))

def safe_num(rng,lo,hi):
 while True:
  n=rng.randrange(lo,hi)
  if str(n) not in FORBIDDEN_3DIGIT:return n
GUIDE_JSON_HINT='Reply with strict JSON only, no markdown: {"exemplar": true or false, "quality": 1-5, "model_worthy_transitions": ["..."], "drift": ["..."], "rejection_category": null or "routing-drift" or "hallucination" or "inefficiency" or "format" or "register" or "other", "notes": "..."}'
# Transport steering for the generator teacher. This is NOT part of the
# canonical record: the episode prompt (protocol + history) is sent as the user
# message; this system message only states the one-frame emission contract so
# the teacher emits exactly its next turn and stops (teachers otherwise tend
# to narrate whole simulated episodes in one completion).
TEACHER_STEER=('You are the assistant speaking at the final "Assistant:" position of the transcript in the user message. '
 'Reply with EXACTLY ONE canonical five-line Pi-native frame for your single next turn, then stop. The five lines are: '
 '"Analysis: <JSON string or null>", "Commentary: <JSON string or null>", "Think: <boolean or null>", '
 '"Action: <one declared tool name, think, or finish>", "Arguments: <JSON object using that tool\'s real schema>". '
 'Never emit a second turn, never simulate or invent tool results or an "Assistant:" block, never wrap the frame in markdown. '
 'Reason privately inside the Analysis JSON string, speak publicly only inside Commentary or finish.message.')


def opaque(seed,n=8):return hashlib.sha256(seed.encode()).hexdigest()[:n]


class Metrics:
 def __init__(self):self.calls=[];self.lock=threading.Lock()
 def record(self,entry):
  with self.lock:self.calls.append(entry)
 def usage(self,model):
  rows=[c for c in self.calls if c['model']==model]
  return {'calls':len(rows),'prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) for c in rows),'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in rows),'latency_s':sorted(round(c['latency_s'],3) for c in rows)}
 def dump(self):
  with self.lock:return {'calls':list(self.calls)}


def lunaroute_call(model,messages,temperature,max_tokens,metrics):
 token=json.loads(Path(os.path.expanduser('~/.pi/agent/auth.json')).read_text())['lunaroute']['access']
 url=os.environ.get('LUNAROUTE_ROUTING_URL','https://gw.lunaroute.com/v1')+'/chat/completions'
 body=json.dumps({'model':model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}).encode()
 last=None
 for attempt in range(4):
  req=urllib.request.Request(url,data=body,headers={'Authorization':f'Bearer {token}','Content-Type':'application/json'})
  t0=time.monotonic()
  try:
   with urllib.request.urlopen(req,timeout=900) as response:payload=json.load(response)
   entry={'model':model,'latency_s':round(time.monotonic()-t0,3),'attempt':attempt,'usage':payload.get('usage')}
   metrics.record(entry)
   content=payload['choices'][0]['message'].get('content')
   if content is None:raise ValueError('empty content')
   return content.strip(),entry
  except Exception as exc:
   last=exc;metrics.record({'model':model,'latency_s':round(time.monotonic()-t0,3),'attempt':attempt,'error':f'{type(exc).__name__}: {exc}'})
   time.sleep(min(20*(attempt+1),60))
 raise RuntimeError(f'lunaroute api exhausted for {model}: {last}')


FRAME_LABELS=('Analysis: ','Commentary: ','Think: ','Action: ','Arguments: ')

def normalize_teacher_frame(content,tools):
 """Rebuild the teacher's own five labeled sections into one canonical frame:
 strip blank lines, join continuation lines into their section, reserialize
 the teacher's own JSON values canonically (sorted keys, compact separators,
 exactly like native_turn). Returns None when the reply is not a five-label
 frame of the declared surface."""
 try:
  lines=[l.rstrip() for l in content.strip().split('\n') if l.strip()]
  parts=[[],[],[],[],[]];idx=-1
  for l in lines:
   if idx<4 and l.startswith(FRAME_LABELS[idx+1]):
    idx+=1;parts[idx].append(l[len(FRAME_LABELS[idx]):])
   elif idx>=0:parts[idx].append(l)
   else:return None
  if idx!=4:return None
  values=['\n'.join(part) for part in parts]
  analysis=strict_json(values[0]);commentary=strict_json(values[1]);think=strict_json(values[2])
  name=values[3].strip();arguments=strict_json(values[4])
  if not isinstance(arguments,dict):return None
  return '\n'.join(('Analysis: '+compact(analysis),'Commentary: '+compact(commentary),'Think: '+compact(think),
   'Action: '+name,'Arguments: '+compact(arguments)))
 except ValueError:return None


def make_teacher_generate(model,tools,enc,metrics,temperature=0.3):
 """Owner-side generate(): DeepSeek teacher produces the next canonical
 five-line Pi-native frame; up to three bounded attempts, strict validation,
 never any synthetic tool result. Format-only normalization (blank-line removal
 and canonical JSON reserialization of the teacher's own parsed fields) is
 applied and counted; no content is invented."""
 state={'invalid_frames':0,'overlong':0,'multi_recovered':0,'normalized':0,'raw':[]}
 def normalized_frame(content):
  return normalize_teacher_frame(content,tools)
 def generate(prompt,budget,deadline):
  correction=None
  for attempt in range(3):
   steer=TEACHER_STEER if correction is None else TEACHER_STEER+' '+correction
   messages=[{'role':'system','content':steer},{'role':'user','content':prompt}]
   content,_=lunaroute_call(model,messages,temperature,min(8192,budget+3072),metrics)
   state['raw'].append({'attempt':attempt,'prompt_tokens':len(enc.encode_ordinary(prompt)),'reply_head':content[:2000]})
   candidates=[content]
   head='\n'.join([l for l in content.split('\n') if l.strip()][:5])
   if head!=content:candidates.append(head)
   canon=normalized_frame(content)
   if canon is not None and canon not in candidates:candidates.append(canon)
   for index,candidate in enumerate(candidates):
    ids=enc.encode_ordinary(candidate)
    valid=False
    try:validate_generated_turn(candidate,tools,enc);valid=True
    except ValueError:pass
    if valid and len(ids)<=budget:
     if index==1:state['multi_recovered']+=1
     if index==2:state['normalized']+=1
     return candidate,ids,'valid'
   if valid:state['overlong']+=1
   else:state['invalid_frames']+=1
   correction=('Your previous reply was not one canonical five-line Pi-native frame. Reply with exactly five lines and nothing else: '
    '"Analysis: <JSON string or null>" then "Commentary: <JSON string or null>" then "Think: <boolean or null>" then '
    '"Action: <one declared tool name, think, or finish>" then "Arguments: <JSON object>". Use the real schema of the chosen tool. '
    'Do not simulate tool results or write additional turns.')
  raise RuntimeError(f'teacher frame invalid after 3 attempts (invalid={state["invalid_frames"]} overlong={state["overlong"]})')
 generate.state=state
 return generate


# ------------------------------------------------------------- terminal tasks
def _file_check(path,content):return {'type':'file','path':path,'content':content}
def _contains_check(path,needle):return {'type':'file_contains','path':path,'needle':needle}
def _absent_check(path):return {'type':'file_absent','path':path}
def _command_check(argv,timeout=90):return {'type':'command','argv':argv,'timeout':timeout}
def _output_check(argv,needle,timeout=60):return {'type':'command_output_contains','argv':argv,'needle':needle,'timeout':timeout}
def _clean_check():return {'type':'command_output_empty','argv':['git','status','--porcelain'],'timeout':30}

TERMINAL_TEMPLATES=(
 'merge-shards','rename-manifest','config-migration',
 'curated-commit','revert-broken-change',
 'service-lifecycle',
 'csv-aggregate','jsonl-dedupe','log-extract',
 'make-repair','pipeline-fix')
FAMILY_OF={'merge-shards':'file-manipulation','rename-manifest':'file-manipulation','config-migration':'file-manipulation',
 'curated-commit':'git-operations','revert-broken-change':'git-operations','service-lifecycle':'process-management',
 'csv-aggregate':'data-wrangling','jsonl-dedupe':'data-wrangling','log-extract':'data-wrangling',
 'make-repair':'multi-step-build','pipeline-fix':'multi-step-build'}


def terminal_case(i,template):
 tag=opaque(f'{PILOT_SEED}:terminal:{template}:{i}')
 rng=random.Random(int(hashlib.sha256(f'{PILOT_SEED}:terminal:{template}:{i}'.encode()).hexdigest()[:16],16))
 name=rng.choice(NAME_WORDS)+tag[:5]
 case={'id':f'teacher-pilot-terminal-{FAMILY_OF[template]}-{i:04d}-{tag}','family':FAMILY_OF[template],'template':template,'workspace_files':{},'setup':[],'verify':[]}
 op=rng.choice(OPENINGS)
 if template=='merge-shards':
  lines=[f'{rng.choice(NAME_WORDS)}-{rng.randrange(10,99)}' for _ in range(rng.randrange(14,22))]
  lines=lines+[rng.choice(lines) for _ in range(3)]
  shards=[[],[],[],[]]
  for k,line in enumerate(lines):shards[k%4].append(line)
  for k in range(4):case['workspace_files'][f'shard_{k}_{tag}.txt']='\n'.join(sorted(set(shards[k])))+'\n'
  expected='\n'.join(sorted(set(lines)))+'\n'
  case['prompt']=(f'{op} the workspace holds four shard files named shard_0_{tag}.txt through shard_3_{tag}.txt. '
   f'Combine all of their lines into combined_{name}.txt so that every distinct line appears exactly once, '
   f'sorted in byte order (like LC_ALL=C sort -u). Remove nothing else.')
  case['verify']=[_file_check(f'combined_{name}.txt',expected)]
 elif template=='rename-manifest':
  sources=[f'notes_{k}_{tag}.md' for k in range(3)]
  targets=[f'{rng.choice(NAME_WORDS)}_{k}_{tag}.md' for k in range(3)]
  mapping={s:t for s,t in zip(sources,targets)}
  manifest=json.dumps({'rename':mapping},indent=2)+'\n'
  contents=[f'# draft {k}\n\ncontent-{tag}-{k}\n' for k in range(3)]
  for s,c in zip(sources,contents):case['workspace_files'][s]=c
  case['workspace_files']['manifest.json']=manifest
  case['prompt']=(f'{op} apply the renames listed in manifest.json exactly as specified, then delete manifest.json. '
   f'Keep every file\'s contents unchanged.')
  checks=[_absent_check(s) for s in sources]+[_absent_check('manifest.json')]
  checks+= [_file_check(t,c) for t,c in zip(targets,contents)]
  case['verify']=checks
 elif template=='config-migration':
  host=f'{rng.choice(NAME_WORDS)}.internal';port=rng.randrange(9100,9799);items=rng.randrange(8,64);depth=rng.randrange(1,8)
  conf=f'[server]\nhost = {host}\nport = {port}\n\n[limits]\nmax_items = {items}\nmax_depth = {depth}\n'
  expected=json.dumps({'server':{'host':host,'port':port},'limits':{'max_items':items,'max_depth':depth}},indent=2)+'\n'
  case['workspace_files'][f'legacy_{name}.conf']=conf
  case['prompt']=(f'{op} migrate the legacy INI file legacy_{name}.conf to service_{name}.json in JSON: '
   f'sections become nested objects, keys keep their names, values keep their types (numbers stay numbers). '
   f'Use two-space indentation and a trailing newline. Leave the legacy file untouched.')
  case['verify']=[_file_check(f'service_{name}.json',expected),_contains_check(f'legacy_{name}.conf',f'port = {port}')]
 elif template=='curated-commit':
  keep=f'core_{name}.py';extra=f'helper_{name}.py';untouched=f'scratch_{name}.py';keepval=safe_num(rng,11,89)
  case['workspace_files'].update({
   keep:f'def base():\n    return {safe_num(rng,11,89)}\n',
   'README.md':f'# repo {name}\n\nseeded fixture repository\n',
   'tests_placeholder.txt':'do not edit\n'})
  case['setup']=[['git','init','-q'],['git','config','user.name','Pilot Fixture'],['git','config','user.email','fixture@invalid.example'],
   ['git','add','README.md','tests_placeholder.txt'],['git','commit','-q','-m','chore: initial fixture'],
   ['sh','-c',f'printf "def extra():\\n    return {safe_num(rng,11,89)}\\n" > {extra}'],
   ['sh','-c',f'printf "local scratch\\n" > {untouched}'],
   ['sh','-c',f'printf "def base():\\n    return {keepval}\\n" > {keep}']]
  branch=f'fix/{name}'
  case['prompt']=(f'{op} this repository has a modified tracked file {keep}, a new untracked file {extra}, and an unrelated '
   f'untracked file {untouched}. Create branch {branch} from the current HEAD and make exactly one commit on it that includes '
   f'only {keep} and {extra}, with commit message exactly "fix: {name}". {untouched} must remain untracked and unstaged. '
   f'Finish when the commit exists on {branch} with exactly that message.')
  case['verify']=[_command_check(['git','rev-parse','--verify',branch]),_output_check(['git','log','--format=%s','-n','1',branch],f'fix: {name}'),
   _output_check(['git','show',f'{branch}:{keep}'],f'return {keepval}'),_output_check(['git','status','--porcelain','--untracked-files=all'],f'?? {untouched}'),
   _output_check(['git','ls-tree','--name-only','HEAD',extra],extra)]
 elif template=='revert-broken-change':
  good=safe_num(rng,101,899)
  while str(good+7) in FORBIDDEN_3DIGIT:good=safe_num(rng,101,899)
  case['workspace_files'][f'src/app_{name}.py']=f'def total(xs):\n    return sum(xs)\n'
  case['workspace_files'][f'tests/test_{name}.py']=f'import unittest,sys\nsys.path.insert(0,"src")\nfrom app_{name} import total\nclass T(unittest.TestCase):\n def test_total(self): self.assertEqual(total([2,{good},5]),{good+7})\n'
  case['setup']=[['git','init','-q'],['git','config','user.name','Pilot Fixture'],['git','config','user.email','fixture@invalid.example'],
   ['git','add','.'],['git','commit','-q','-m','feat: initial implementation'],
   ['sh','-c',f'printf "def total(xs):\\n    return sum(xs[1:])\\n" > src/app_{name}.py'],
   ['git','add','.'],['git','commit','-q','-m','feat: tuning pass']]
  case['prompt']=(f'{op} the test suite in tests/ is failing on this repository. Find the change that broke it, restore the '
   f'correct behavior, and land the fix as a new commit on the current branch with message exactly "fix: restore {name}". '
   f'Do not rewrite existing history and do not modify the tests. Finish with a clean working tree and a passing suite.')
  case['verify']=[_command_check(['python3','-m','unittest','discover','-s','tests','-q']),_output_check(['git','log','--format=%s'],f'fix: restore {name}'),_clean_check()]
 elif template=='service-lifecycle':
  script=f'service_{name}.py'
  case['workspace_files'][script]=(f'import signal,time,pathlib\n'
   f'log=pathlib.Path("events.log")\n'
   f'def note(text):log.open("a").write(text+"\\n")\n'
   f'note("started")\n'
   f'print("READY",flush=True)\n'
   f'def stop(*_):\n'
   f'    note("stopped")\n'
   f'    raise SystemExit(0)\n'
   f'signal.signal(signal.SIGTERM,stop)\n'
   f'while True:time.sleep(0.2)\n')
  case['prompt']=(f'{op} {script} is a small background service that logs lifecycle events to events.log and prints READY once it is up. '
   f'Start it as a background process, confirm from its own output that it reported READY, then stop that same process cleanly '
   f'(SIGTERM) and confirm events.log records both the start and the stop. Finish when the lifecycle is recorded.')
  case['verify']=[_contains_check('events.log','started'),_contains_check('events.log','stopped')]
 elif template=='csv-aggregate':
  group_names=[f'{g}-{tag[:3]}' for g in rng.sample(GROUPS,4)]
  rows=[]
  for _ in range(rng.randrange(12,24)):
   g=rng.choice(group_names);rows.append((g,rng.randrange(1,99)))
  case['workspace_files'][f'data_{name}.csv']='group,value\n'+''.join(f'{g},{v}\n' for g,v in rows)
  agg={}
  for g,v in rows:agg.setdefault(g,[]).append(v)
  for g,vs in agg.items():
   while str(sum(vs)) in FORBIDDEN_3DIGIT:vs[0]=max(1,vs[0]-1)
  expected=''.join(f'{g} {sum(vs)} {len(vs)}\n' for g,vs in sorted(agg.items()))
  case['prompt']=(f'{op} data_{name}.csv holds group,value rows. Produce summary_{name}.txt with one line per group, '
   f'sorted by group name, each line exactly "group total count" separated by single spaces (total = sum of values, count = number of rows).')
  case['verify']=[_file_check(f'summary_{name}.txt',expected)]
 elif template=='jsonl-dedupe':
  records=[]
  for k in range(rng.randrange(10,18)):
   records.append({'id':rng.randrange(1,9),'ts':safe_num(rng,100,999),'kind':rng.choice(('open','close','poll'))})
  dup=rng.choice(records)
  rows=[dict(r) for r in records]
  rows.insert(rng.randrange(len(rows)),dict(dup))
  rng.shuffle(rows)
  seen=set();kept=[]
  for r in sorted(rows,key=lambda r:r['ts']):
   if r['id'] in seen:continue
   seen.add(r['id']);kept.append(r)
  expected=''.join(json.dumps(r,sort_keys=True)+'\n' for r in kept)
  case['workspace_files'][f'events_{name}.jsonl']=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
  case['prompt']=(f'{op} events_{name}.jsonl contains JSON objects with id, ts and kind, and some ids appear more than once. '
   f'Write cleaned_{name}.jsonl keeping, for each id, only the first occurrence by ts, then sort all kept records by ts ascending. '
   f'One compact JSON object per line, keys in alphabetical order, exactly like the input formatting.')
  case['verify']=[_file_check(f'cleaned_{name}.jsonl',expected)]
 elif template=='log-extract':
  codes=[safe_num(rng,400,599) for _ in range(rng.randrange(3,6))]
  lines=[]
  counts={}
  for k in range(rng.randrange(14,26)):
   if rng.random()<0.4:
    c=rng.choice(codes);counts[c]=counts.get(c,0)+1
    lines.append(f'{rng.randrange(10,99)}:12 ERROR [{c}] request failed')
   else:lines.append(f'{rng.randrange(10,99)}:12 INFO ok')
  expected=''.join(f'{c} {n}\n' for c,n in sorted(counts.items()))
  case['workspace_files'][f'app_{name}.log']='\n'.join(lines)+'\n'
  case['prompt']=(f'{op} app_{name}.log mixes INFO lines with ERROR lines that look like "ERROR [<code>] request failed". '
   f'Write codes_{name}.txt with one line per distinct error code, sorted by code, each line exactly "code count" with single spaces.')
  case['verify']=[_file_check(f'codes_{name}.txt',expected)]
 elif template=='make-repair':
  total=safe_num(rng,100,999)
  while str(total+8) in FORBIDDEN_3DIGIT:total=safe_num(rng,100,999)
  case['workspace_files'].update({
   'Makefile':f'all: build test\n\nbuild:\n\tpython3 src/calc_{name}.py > artifact_{name}.txt\n\tpython3 -m py_compile src/calc_{name}.py\n\ntest:\n\tpython3 -m unittest discover -s tests -q\n',
   f'src/calc_{name}.py':f'def total(xs):\n    return sum(xs[1:])\n\nprint(total([3, {total}, 5]))\n',
   f'tests/test_{name}.py':f'import unittest,sys\nsys.path.insert(0,"src")\nfrom calc_{name} import total\nclass T(unittest.TestCase):\n def test_total(self): self.assertEqual(total([1,2,3]),6)\n'})
  case['prompt']=(f'{op} the build in this project is broken: run make all, observe the failure, then fix the defect with the '
   f'smallest correct change to the source under src/ (leave the Makefile and tests untouched) and rerun make all until it '
   f'succeeds, including the build artifact artifact_{name}.txt. Finish only when make all passes.')
  case['verify']=[_command_check(['make','all']),_file_check(f'artifact_{name}.txt',f'{total+8}\n')]
 elif template=='pipeline-fix':
  rows=[{'name':f'item-{k}','value':rng.randrange(1,99)} for k in range(rng.randrange(6,12))]
  correct=sorted(rows,key=lambda r:r['value'])
  expected=''.join(f"{r['name']} {r['value']}\n" for r in correct)
  case['workspace_files'].update({
   f'input_{name}.json':json.dumps({'items':rows},indent=2)+'\n',
   f'pipeline_{name}.py':(f'import json\n'
    f'with open("input_{name}.json") as f:data=json.load(f)\n'
    f'items=data["items"]\n'
    f'items.sort(key=lambda r: r["value"], reverse=True)\n'
    f'with open("out_{name}.txt","w") as f:\n'
    f'    for r in items:f.write(r["name"]+" "+str(r["value"])+"\\n")\n'),
   f'expected_{name}.txt':expected})
  case['prompt']=(f'{op} pipeline_{name}.py processes input_{name}.json into out_{name}.txt, but its output does not match '
   f'expected_{name}.txt. Run it, compare, fix the pipeline so out_{name}.txt matches expected_{name}.txt exactly, and rerun it. '
   f'Leave expected_{name}.txt and the input untouched.')
  case['verify']=[_command_check(['python3',f'pipeline_{name}.py']),_file_check(f'out_{name}.txt',expected)]
 else:raise ValueError(template)
 for text in [case['prompt'],*case['workspace_files'].values(),json.dumps(case['verify'],sort_keys=True)]:check_text(text)
 if len(case['workspace_files'])>16 or sum(len(v.encode()) for v in case['workspace_files'].values())>32768:raise ValueError('workspace bound')
 return case


def freeze_terminal(args):
 if args.output.exists():raise FileExistsError(args.output)
 manifest_path=args.manifest
 if sha(manifest_path)!=MANIFEST_SHA:raise ValueError('tool authority')
 manifest=json.loads(manifest_path.read_text())
 pi_bin=Path(manifest['pi_bin'])
 if sha(pi_bin)!=manifest['pi_bin_sha256']:raise ValueError('frozen pi runtime identity')
 cases=[terminal_case(i,TERMINAL_TEMPLATES[i%len(TERMINAL_TEMPLATES)]) for i in range(args.cases)]
 ids=[c['id'] for c in cases]
 if len(set(ids))!=len(ids):raise ValueError('unique ids')
 goals=[c['prompt'] for c in cases]
 if len(set(goals))!=len(goals):raise ValueError('unique prompts')
 plan={'schema':TERMINAL_PLAN_SCHEMA,'seed':PILOT_SEED,'cases':args.cases,'minimum_verified_records':args.minimum_verified,
  'generator_model':args.generator,'guide_model':args.guide,'panel':TERMINAL_PANEL,'tool_manifest':str(manifest_path.resolve()),
  'tool_manifest_sha256':sha(manifest_path),'model_visible_tools':manifest['model_visible_tools'],
  'provider_extension':str(PROVIDER_V2),'extensions':[str(p) for p in EXTENSIONS],
  'pi_bin':str(pi_bin.resolve()),'pi_version':subprocess.check_output([str(pi_bin),'--version'],text=True).strip(),
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'authority_files':authority_files(),'task_authoring':{'style':'terminal-bench-style, newly authored',
   'deterministic':True,'benchmark_instances_used':0,'benchmark_suites_consulted':[],
   'seeded_synthetic_values_only':True,'forbidden_discipline':'hybrid-v2 check_text + teacher-pilot evaluation entities'},
  'automatic_retry':False,'model_generations':args.cases,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False,
  'cases':cases}
 args.output.mkdir(parents=True,mode=0o700)
 publish(args.output/'plan-private.json',plan)
 print('TERMINAL_PLAN',args.cases,sha(args.output/'plan-private.json'),flush=True)


def authority_files():
 return {str(p):sha(p) for p in (Path(__file__),Path('configs/pi/e97-pi-native-frozen-tools-v2.ts'),Path('scripts/e97_pi_native_codec.py'),
  Path('scripts/e97_pi_native_tool_bridge.py'),Path('scripts/e97_pi_native_tool_transport.py'),Path('scripts/e97_open_swe_native_codec.py'),*EXTENSIONS)}


def load_checkpoint(path):
 done={}
 if path.exists():
  for line in path.read_text().splitlines():
   if line.strip():
    row=json.loads(line);done[row['id']]=row
 return done


def append_checkpoint(path,row):
 with path.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(compact(row)+'\n');f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)


def run_check(check,workspace):
 t=check['type'];timeout=check.get('timeout',60)
 if t=='file':
  p=Path(workspace)/check['path'];return (p.is_file() and p.read_text()==check['content']),f"file {check['path']} exact"
 if t=='file_contains':
  p=Path(workspace)/check['path'];return (p.is_file() and check['needle'] in p.read_text()),f"file {check['path']} contains"
 if t=='file_absent':
  return not (Path(workspace)/check['path']).exists(),f"absent {check['path']}"
 if t in ('command','command_output_contains','command_output_empty'):
  try:cp=subprocess.run(check['argv'],cwd=workspace,capture_output=True,text=True,timeout=timeout)
  except subprocess.TimeoutExpired:return False,f'timeout {check["argv"]}'
  detail=(cp.stdout+cp.stderr)[:400]
  if t=='command':return cp.returncode==0,detail
  if t=='command_output_contains':return check['needle'] in cp.stdout,detail
  return cp.stdout.strip()=='' and cp.returncode==0,detail
 raise ValueError(t)


def guide_review_terminal(model,metrics,goal,record):
 body=record[record.index('\n\nUser:\n')+2:]
 view=body if len(body)<=30000 else body[:30000]+'\n...<truncated>'
 prompt=(f'You are the guide teacher for a small on-device coding-agent corpus. Below is a candidate training trace: the task '
  f'goal and the complete Pi-native episode transcript (the protocol header is omitted; each assistant turn is a five-line '
  f'frame: Analysis is private reasoning, Commentary is public, Action/Arguments name the tool call; ToolResult blocks are '
  f'real executions of those calls). Judge whether this trace is a good exemplar for teaching tool use to a small agent: '
  f'correct and minimal tool choices for the goal; grounded reasoning that actually uses observations (no invented results); '
  f'brief natural public commentary; no drift, padding or hallucinated state. Be decisive: a trace that completes the task with '
  f'clean, grounded tool use IS an exemplar even if a shorter path existed; reject for real defects only.'
  f'\n\nTask goal:\n{goal}\n\nEpisode:\n{view}\n\n{GUIDE_JSON_HINT}')
 raw,_=lunaroute_call(model,[{'role':'user','content':prompt}],0.2,8000,metrics)
 verdict=parse_json_object(raw)
 if not isinstance(verdict.get('exemplar'),bool):raise ValueError('guide verdict missing exemplar flag')
 return verdict


def parse_json_object(text):
 body=text.strip()
 if body.startswith('```'):
  body=re.sub(r'^```[a-z]*\n?','',body);body=re.sub(r'\n?```$','',body)
 start=body.find('{');end=body.rfind('}')
 if start<0 or end<0:raise ValueError('no JSON object in reply')
 return json.loads(body[start:end+1])


def collect_terminal_case(case,panel,ctx):
 case_dir=ctx['out']/case['id'];result={'id':case['id'],'arm':'terminal','family':case['family'],'template':case['template']}
 t0=time.monotonic()
 if case_dir.exists():shutil.rmtree(case_dir)
 case_dir.mkdir(parents=True)
 workspace=case_dir/'workspace';workspace.mkdir()
 for name,text in case['workspace_files'].items():
  p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 for argv in case.get('setup',[]):
  cp=subprocess.run(argv,cwd=workspace,capture_output=True,text=True,timeout=120)
  if cp.returncode!=0:raise RuntimeError(f'setup failed: {argv}: {cp.stderr[:200]}')
 enc=ctx['enc'];metrics=ctx['metrics']
 generate=make_teacher_generate(ctx['generator'],panel['tools'],enc,metrics)
 bridge=NativePiToolBridge(panel,case['prompt'],enc,generate)
 try:
  terminal=serve_pi_native_tools(bridge,case_dir/'pi',pi_bin=ctx['pi_bin'],provider_extension=PROVIDER_V2,
   pi_extensions=EXTENSIONS,cwd=workspace,seconds=panel['episode_seconds']+120,
   extra_env={'E97_PI_TOOL_MANIFEST':ctx['manifest']})
 except BridgeStopped as exc:
  result.update(status='rejected',reason=f'bridge:{bridge.reason or exc}',wall_s=round(time.monotonic()-t0,1))
  publish_result(case_dir,case,result,bridge,generate,None,None,None,None);return result
 except Exception as exc:
  result.update(status='rejected',reason=f'error:{type(exc).__name__}: {exc}',wall_s=round(time.monotonic()-t0,1))
  publish_result(case_dir,case,result,bridge,generate,None,None,None,None);return result
 if not terminal['close_verified'] or bridge.final is None:
  result.update(status='rejected',reason='incomplete: episode did not finish',wall_s=round(time.monotonic()-t0,1))
  publish_result(case_dir,case,result,bridge,generate,None,None,None,None);return result
 verify_receipts=[];verify_ok=True
 for check in case['verify']:
  ok,detail=run_check(check,workspace)
  verify_receipts.append({'type':check['type'],'ok':ok,'detail':detail})
  verify_ok&=ok
 native_record=bridge.episode.text()
 guide_error=None
 if verify_ok:
  try:verdict=guide_review_terminal(ctx['guide'],metrics,case['prompt'],native_record)
  except Exception as exc:verdict=None;guide_error=f'{type(exc).__name__}: {exc}'
 else:verdict=None
 if verify_ok and verdict is not None and verdict['exemplar']:
  ids,mask,units=encode_candidate(native_record,bridge.generations,0,enc)
  result.update(status='verified',reason=None,wall_s=round(time.monotonic()-t0,1),
   tokens=len(ids),targets=sum(mask),assistant_units=len(bridge.generations),supervised_units=units,
  calls=sum(1 for m in bridge.history if m['role']=='toolResult'),
   errors=sum(m['isError'] for m in bridge.history if m['role']=='toolResult'),
   guide_quality=verdict.get('quality'),guide_transitions=verdict.get('model_worthy_transitions'),
   invalid_frames=generate.state['invalid_frames'],multi_recovered=generate.state['multi_recovered'])
  publish_result(case_dir,case,result,bridge,generate,terminal,verify_receipts,verdict,None)
  return result
 if not verify_ok:reason='mechanical-verification-failed'
 elif guide_error is not None:reason=f'guide-error:{guide_error}'
 elif verdict is not None and not verdict['exemplar']:reason=f"guide-reject:{verdict.get('rejection_category') or 'unspecified'}"
 else:reason='unknown'
 result.update(status='rejected',reason=reason,wall_s=round(time.monotonic()-t0,1),
  guide_quality=(verdict or {}).get('quality'),guide_transitions=(verdict or {}).get('model_worthy_transitions'),
  invalid_frames=generate.state['invalid_frames'],multi_recovered=generate.state['multi_recovered'])
 publish_result(case_dir,case,result,bridge,generate,terminal,verify_receipts,verdict,guide_error)
 return result


def publish_result(case_dir,case,result,bridge,generate,terminal,verify_receipts,verdict,guide_error):
 private={'id':case['id'],'family':case['family'],'template':case['template'],'prompt':case['prompt'],
  'workspace_files':case['workspace_files'],'verify':case['verify'],'result':result,'verify_receipts':verify_receipts,
  'guide_verdict':verdict,'guide_error':guide_error,'terminal':terminal,'teacher_raw_replies':list(generate.state['raw'])}
 if bridge is not None:
  private.update(native_record=bridge.episode.text(),source_messages=bridge.episode.source_messages(),
   public_history=bridge.history,generations=bridge.generations,final=bridge.final,failed=bridge.failed,reason=bridge.reason)
 publish(case_dir/'episode-private.json',private)


def collect_terminal(args):
 plan=json.loads(args.plan.read_text())
 if plan['schema']!=TERMINAL_PLAN_SCHEMA:raise ValueError('plan schema')
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 if sha(Path(plan['tool_manifest']))!=MANIFEST_SHA:raise ValueError('tool authority')
 if sha(Path(plan['pi_bin']))!=json.loads(Path(plan['tool_manifest']).read_text())['pi_bin_sha256']:raise ValueError('frozen pi runtime identity')
 if plan['authority_files']!=authority_files():raise ValueError('authority files drifted')
 if args.generator!=plan['generator_model'] or args.guide!=plan['guide_model']:raise ValueError('model authority')
 enc=tiktoken.get_encoding('p50k_base')
 panel=dict(TERMINAL_PANEL);panel['tools']=plan['model_visible_tools']
 ctx={'out':args.output,'enc':enc,'metrics':Metrics(),'generator':args.generator,'guide':args.guide,
  'pi_bin':Path(plan['pi_bin']),'manifest':plan['tool_manifest']}
 checkpoint=args.output/'checkpoint.jsonl'
 done=load_checkpoint(checkpoint)
 todo=[c for c in plan['cases'] if c['id'] not in done]
 verified=sum(1 for r in done.values() if r['status']=='verified')
 print(f'TERMINAL_COLLECT start: resumed_verified={verified} pending={len(todo)} lanes={args.lanes} target={args.target}',flush=True)
 stop=threading.Event()
 def worker(case):
  if stop.is_set():return None
  try:row=collect_terminal_case(case,panel,ctx)
  except Exception as exc:
   row={'id':case['id'],'arm':'terminal','family':case['family'],'template':case['template'],'status':'rejected',
    'reason':f'error:{type(exc).__name__}: {exc}','wall_s':None}
  append_checkpoint(checkpoint,row);print('CASE',row['id'],row['status'],row.get('reason') or '',flush=True)
  return row
 with ThreadPoolExecutor(max_workers=args.lanes) as pool:
  for row in pool.map(worker,todo):
   if row and row['status']=='verified':
    verified+=1
    if verified>=args.target:stop.set()
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
 publish(args.output/f'metrics-{stamp}.json',{'collected_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'usage':{m:ctx['metrics'].usage(m) for m in (args.generator,args.guide)},'usage_calls':ctx['metrics'].dump()})
 print('TERMINAL_COLLECT done',flush=True)


# ------------------------------------------------------------ conversations
def wildchat_seed_rows(limit,stride):
 files=sorted(WILDCHAT_RAW.glob('*.parquet'))
 if not files:raise FileNotFoundError(WILDCHAT_RAW)
 seeds=[];seen=set();scanned=0;hits=0
 for path in files:
  for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
   for row in batch.to_pylist():
    scanned+=1
    if row.get('language')!='English':continue
    if row.get('toxic'):continue
    messages=row.get('conversation') or []
    moderation=row.get('openai_moderation') or []
    if not messages or not isinstance(messages[0],dict) or messages[0].get('role')!='user':continue
    if len(moderation)!=len(messages):continue
    entry=moderation[0]
    if not isinstance(entry,dict) or entry.get('flagged'):continue
    categories=entry.get('categories')
    if isinstance(categories,dict) and any(categories.values()):continue
    content=messages[0].get('content')
    if not isinstance(content,str):continue
    normalized=content.replace(convcodec.RS,' ').replace('\r\n','\n').replace('\r','\n').strip()
    if not (40<=len(normalized)<=600):continue
    try:check_text(normalized)
    except ValueError:continue
    if normalized in seen:continue
    hits+=1
    if hits%stride:continue
    seen.add(normalized)
    seeds.append({'identity':f'wildchat:{row.get("conversation_id")}','source_file':path.name,'source_row':scanned-1,'seed':normalized})
    if len(seeds)>=limit:return seeds,scanned
 return seeds,scanned


def freeze_conversations(args):
 if args.output.exists():raise FileExistsError(args.output)
 seeds,scanned=wildchat_seed_rows(args.seeds,args.stride)
 plan={'schema':CONVERSATION_PLAN_SCHEMA,'seed':PILOT_SEED,'seeds':args.seeds,'minimum_verified_records':args.minimum_verified,
  'generator_model':args.generator,'guide_model':args.guide,'temperature':0.7,
  'seed_authority':{'dataset_id':WILDCHAT_ID,'dataset_revision':WILDCHAT_REVISION,'raw_dir':str(WILDCHAT_RAW),
   'intake_workspace':'/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-real-human-chat-intake-v1',
   'filters':['english','not toxic','first-message moderation complete and unflagged','first user turn 40-600 chars normalized',
              'hybrid-v2 forbidden-literal discipline','deduplicated']},
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'authority_files':authority_files(),'scanned_rows':scanned,'automatic_retry':False,
  'model_generations':args.seeds,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False,
  'conversations':[{'id':f'teacher-pilot-conv-{i:04d}-{opaque(f"{PILOT_SEED}:conv:{i}")}','seed_identity':s['identity'],
   'source_file':s['source_file'],'source_row':s['source_row'],'seed':s['seed']} for i,s in enumerate(seeds)]}
 args.output.mkdir(parents=True,mode=0o700)
 publish(args.output/'plan-private.json',plan)
 print('CONVERSATION_PLAN',len(seeds),scanned,sha(args.output/'plan-private.json'),flush=True)


META_MARKERS=('"turns"','{"role"','User:\n','Assistant:\n','WildChat','wildchat','LMSYS','as an AI language model','the opening message','seed message')


def conversation_guard(seed,turns):
 if not isinstance(turns,list) or not (3<=len(turns)<=5) or len(turns)%2==0:raise ValueError('turn count')
 messages=[{'role':'user','content':seed}]
 for t in turns:
  if not isinstance(t,dict) or set(t)!={'role','content'} or t['role'] not in ('user','assistant') or not isinstance(t['content'],str):raise ValueError('turn shape')
  content=t['content'].replace(convcodec.RS,' ').replace('\r\n','\n').replace('\r','\n').strip()
  if not content or len(content)>6000:raise ValueError('turn content')
  for marker in META_MARKERS:
   if marker in content:raise ValueError(f'meta marker {marker!r}')
  check_text(content)
  messages.append({'role':t['role'],'content':content})
 roles=[m['role'] for m in messages]
 if roles[0]!='user' or any(roles[k]==roles[k+1] for k in range(len(roles)-1)) or roles[-1]!='assistant':raise ValueError('alternation')
 if sum(r=='assistant' for r in roles)>6:raise ValueError('assistant bound')
 return messages


def collect_conversation_case(conv,ctx):
 case_dir=ctx['out']/conv['id'];result={'id':conv['id'],'arm':'conversation','seed_identity':conv['seed_identity']}
 t0=time.monotonic()
 if case_dir.exists():shutil.rmtree(case_dir)
 case_dir.mkdir(parents=True)
 prompt=(f'A real person opened a chat with the message below. Continue the conversation naturally: write the assistant\'s '
  f'reply to the opening, then continue with further user and assistant turns in the same human register as the opening '
  f'(the follow-up user turns should read like the same person: their interests, tone and language). Write 3 or 5 continuation '
  f'turns total, alternating roles, ending with an assistant turn. Keep replies natural, helpful and concise; never mention '
  f'that you are generating a conversation.\n\nReply with strict JSON only, no markdown: '
  f'{{"turns": [{{"role": "assistant", "content": "..."}}, ...]}}\n\nOpening:\n{conv["seed"]}')
 raw,_=lunaroute_call(ctx['generator'],[{'role':'user','content':prompt}],ctx['temperature'],8000,ctx['metrics'])
 guard_error=None;messages=None;turns=None
 try:turns=parse_json_object(raw)['turns']
 except Exception as exc:guard_error=f'generator-json:{type(exc).__name__}: {str(exc)[:200]}'
 if turns is not None:
  try:messages=conversation_guard(conv['seed'],turns)
  except ValueError as exc:guard_error=str(exc)
 verdict=None
 if messages is not None:
  rendered='\n\n'.join(f"{'User' if m['role']=='user' else 'Assistant'}:\n{m['content']}" for m in messages)
  gprompt=(f'You are the guide teacher for a small on-device chat model. Below is a candidate training conversation. '
   f'The first user turn is a real human opening; the remaining turns are teacher-written. Judge register quality: does the '
   f'continuation keep the human register of the opening, are the assistant replies natural, coherent and modern; is there '
   f'any drift (contradiction, topic derailment, unnatural stiffness, assistant-speak leaking scaffolding)? Be decisive: '
   f'a coherent, natural continuation that stays on topic IS an exemplar; reject for real defects only.\n\n{rendered}\n\n{GUIDE_JSON_HINT}')
  try:verdict=parse_json_object(lunaroute_call(ctx['guide'],[{'role':'user','content':gprompt}],0.2,8000,ctx['metrics'])[0])
  except Exception as exc:guard_error=f'guide-error:{type(exc).__name__}: {exc}'
 if messages is not None and verdict is not None and verdict.get('exemplar') is True:
  result.update(status='verified',reason=None,wall_s=round(time.monotonic()-t0,1),turns=len(messages),
   guide_quality=verdict.get('quality'),guide_drift=verdict.get('drift'))
  publish(case_dir/'record-private.json',{'id':conv['id'],'seed_identity':conv['seed_identity'],'messages':messages,
   'result':result,'guide_verdict':verdict})
  return result
 reason=guard_error or (f"guide-reject:{verdict.get('rejection_category') or 'unspecified'}" if verdict is not None else 'guide-error:unknown')
 result.update(status='rejected',reason=reason,wall_s=round(time.monotonic()-t0,1),
  guide_quality=(verdict or {}).get('quality'),guide_drift=(verdict or {}).get('drift'))
 publish(case_dir/'record-private.json',{'id':conv['id'],'seed_identity':conv['seed_identity'],'messages':messages,
  'result':result,'guide_verdict':verdict,'raw_turns':turns if messages is None else None})
 return result


def collect_conversations(args):
 plan=json.loads(args.plan.read_text())
 if plan['schema']!=CONVERSATION_PLAN_SCHEMA:raise ValueError('plan schema')
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 if args.generator!=plan['generator_model'] or args.guide!=plan['guide_model']:raise ValueError('model authority')
 ctx={'out':args.output,'metrics':Metrics(),'generator':args.generator,'guide':args.guide,'temperature':plan['temperature']}
 checkpoint=args.output/'checkpoint.jsonl'
 done=load_checkpoint(checkpoint)
 todo=[c for c in plan['conversations'] if c['id'] not in done]
 verified=sum(1 for r in done.values() if r['status']=='verified')
 print(f'CONVERSATION_COLLECT start: resumed_verified={verified} pending={len(todo)} lanes={args.lanes} target={args.target}',flush=True)
 stop=threading.Event()
 def worker(conv):
  if stop.is_set():return None
  try:row=collect_conversation_case(conv,ctx)
  except Exception as exc:
   row={'id':conv['id'],'arm':'conversation','seed_identity':conv['seed_identity'],'status':'rejected','reason':f'error:{type(exc).__name__}: {exc}','wall_s':None}
  append_checkpoint(checkpoint,row);print('CASE',row['id'],row['status'],row.get('reason') or '',flush=True)
  return row
 with ThreadPoolExecutor(max_workers=args.lanes) as pool:
  for row in pool.map(worker,todo):
   if row and row['status']=='verified':
    verified+=1
    if verified>=args.target:stop.set()
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
 publish(args.output/f'metrics-{stamp}.json',{'collected_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  'usage':{m:ctx['metrics'].usage(m) for m in (args.generator,args.guide)},'usage_calls':ctx['metrics'].dump()})
 print('CONVERSATION_COLLECT done',flush=True)


def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')))
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
 f=sp.add_parser('freeze-terminal');f.add_argument('--manifest',type=Path,default=Path('configs/pi/e97-active-tool-surface-v1.json'))
 f.add_argument('--cases',type=int,required=True);f.add_argument('--minimum-verified',type=int,required=True)
 f.add_argument('--generator',default=GENERATOR_DEFAULT);f.add_argument('--guide',default=GUIDE_DEFAULT)
 f.add_argument('--output',type=Path,required=True)
 c=sp.add_parser('collect-terminal');c.add_argument('--plan',type=Path,required=True);c.add_argument('--plan-sha',required=True)
 c.add_argument('--generator',default=GENERATOR_DEFAULT);c.add_argument('--guide',default=GUIDE_DEFAULT)
 c.add_argument('--lanes',type=int,required=True);c.add_argument('--target',type=int,required=True)
 c.add_argument('--output',type=Path,required=True)
 g=sp.add_parser('freeze-conversations');g.add_argument('--seeds',type=int,required=True);g.add_argument('--stride',type=int,default=5)
 g.add_argument('--minimum-verified',type=int,required=True);g.add_argument('--generator',default=GENERATOR_DEFAULT)
 g.add_argument('--guide',default=GUIDE_DEFAULT);g.add_argument('--output',type=Path,required=True)
 h=sp.add_parser('collect-conversations');h.add_argument('--plan',type=Path,required=True);h.add_argument('--plan-sha',required=True)
 h.add_argument('--generator',default=GENERATOR_DEFAULT);h.add_argument('--guide',default=GUIDE_DEFAULT)
 h.add_argument('--lanes',type=int,required=True);h.add_argument('--target',type=int,required=True)
 h.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.command=='freeze-terminal':freeze_terminal(a)
 elif a.command=='collect-terminal':collect_terminal(a)
 elif a.command=='freeze-conversations':freeze_conversations(a)
 elif a.command=='collect-conversations':collect_conversations(a)


if __name__=='__main__':main()
