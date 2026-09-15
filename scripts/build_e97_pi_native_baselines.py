#!/usr/bin/env python3
"""Freeze held-out Pi-native exact-copy and first-tool baselines before sampling."""
import argparse,hashlib,json,os,random
from pathlib import Path
import tiktoken
from scripts.e97_pi_native_codec import PiNativeEpisode
from scripts.eval_e97_native_execution import publish,sha
MANIFEST_SHA='55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908'
TOOL_MANIFEST_REF='configs/pi/e97-active-tool-surface-v1.json'
DELAYS=(128,1024,8192,32768,58000);STAGE_A_DELAYS={128,8192,58000}
WORDS=(' archive',' datum',' vector',' amber',' quiet',' ledger',' orbit',' cedar',' mosaic',' kernel',' violet',' signal',' parcel',' meadow',' copper',' tensor')

def opaque(seed,n=24):return hashlib.sha256(seed.encode()).hexdigest()[:n]
def filler(tokens,seed,encoding):
 choices=[w for w in WORDS if len(encoding.encode_ordinary(w))==1]
 if len(choices)<8:raise ValueError('single-token filler vocabulary')
 rng=random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8],'little'))
 return ''.join(rng.choice(choices) for _ in range(tokens))
def prompt_tokens(tools,prompt,encoding):
 e=PiNativeEpisode(tools,encoding);e.append_context({'role':'system','content':'Complete the task using the declared Pi-native tools. Finish only when complete.'});e.append_context({'role':'user','content':prompt});return len(encoding.encode_ordinary(e.prompt()))

def build(manifest):
 enc=tiktoken.get_encoding('p50k_base');tools=manifest['model_visible_tools'];cases=[]
 for delay in DELAYS:
  pad=filler(delay,f'copy:{delay}',enc);answer=opaque(f'copy:{delay}')
  prompt=f'Memorize this opaque value exactly: {answer}\nDo not transform it.{pad}\nNow finish with exactly the opaque value and no other text.'
  cases.append(dict(id=f'copy-{delay:05d}',stage='A' if delay in STAGE_A_DELAYS else 'B',family='copy',delay_tokens=delay,prompt=prompt,files={},expected_final=answer,expected_first_action='finish'))
  values={k:opaque(f'bind:{delay}:{k}',16) for k in ('amber','silver','violet')};active=('amber','silver','violet')[delay%3]
  table=json.dumps({'active':active,'values':values},sort_keys=True,separators=(',',':'));pad=filler(delay,f'bind:{delay}',enc)
  prompt=f'Remember this binding exactly: {table}{pad}\nFinish with exactly the value selected by active, with no other text.'
  cases.append(dict(id=f'bind-{delay:05d}',stage='A' if delay in STAGE_A_DELAYS else 'B',family='bind',delay_tokens=delay,prompt=prompt,files={},expected_final=values[active],expected_first_action='finish'))
  token=opaque(f'path:{delay}',20);path=f'needles/{opaque(f"p:{delay}",12)}.txt';pad=filler(delay,f'path:{delay}',enc)
  prompt=f'The exact relative path to read later is {path}.{pad}\nUse read on the exact remembered path, then finish with exactly the file value.'
  cases.append(dict(id=f'path-{delay:05d}',stage='A' if delay in STAGE_A_DELAYS else 'B',family='path',delay_tokens=delay,prompt=prompt,files={path:token+'\n'},expected_final=token,expected_first_action='read',expected_first_arguments={'path':path}))
 # Three Stage-A tool-choice contrasts.
 given=opaque('direct',16)
 cases.append(dict(id='choice-direct-supplied',stage='A',family='tool-choice',delay_tokens=0,prompt=f'The verified answer is already supplied: {given}. Do not use a tool; finish with exactly that value.',files={},expected_final=given,expected_first_action='finish'))
 target='weather_adapter_73c91.py';cases.append(dict(id='choice-fffind-local',stage='A',family='tool-choice',delay_tokens=0,prompt='A repository file name contains weather_adapter. Locate it with fuzzy file search, then finish with exactly its relative path.',files={target:'# fixture\n'},expected_final=target,expected_first_action='fffind'))
 cases.append(dict(id='choice-web-current',stage='A',family='tool-choice',delay_tokens=0,prompt='What is the current weather in Reykjavik, Iceland? Research the current information on the web, then answer concisely with the reported conditions and temperature.',files={},expected_final=None,expected_first_action='web_search',external_current_fact=True))
 # Stage-B first-action coverage; outcomes are separately verified after execution.
 specs=[
  ('read-explicit','read','Read facts.txt and finish with exactly its content.',{'facts.txt':'read-ok\n'}),
  ('grep-symbol','ffgrep','Find the file containing the exact symbol PI_NATIVE_MARKER using fuzzy content search; finish with its relative path.',{'src/marker.py':'PI_NATIVE_MARKER = 1\n'}),
  ('bash-command','bash','Use bash to print the decimal sum of 731 and 992, then finish with exactly that number.',{}),
  ('edit-file','edit','In state.txt replace alpha with beta using edit, then read it and finish with exactly done.',{'state.txt':'alpha\n'}),
  ('write-file','write','Create result.txt containing exactly written followed by a newline, then read it and finish with exactly done.',{}),
  ('process-start','process','Start a managed process that prints READY and exits, inspect its completion, then finish with exactly done.',{}),
  ('source-check','source_check','Use source checking to verify the claim that Python was first released in 1991, then answer from its evidence.',{}),
  ('fetch-url','fetch_content','Fetch https://www.python.org/about/ and state the page title.',{}),
 ]
 for suffix,action,prompt,files in specs:cases.append(dict(id='choice-'+suffix,stage='B',family='tool-choice',delay_tokens=0,prompt=prompt,files=files,expected_final=None if action in ('source_check','fetch_content') else ('1723' if action=='bash' else 'done'),expected_first_action=action))
 for c in cases:
  c['initial_prompt_tokens']=prompt_tokens(tools,c['prompt'],enc)
  if c['initial_prompt_tokens']+1024>65536:raise ValueError(f"context plus generation cap {c['id']} {c['initial_prompt_tokens']}")
 ids=[c['id'] for c in cases]
 if len(ids)!=len(set(ids)) or sum(c['stage']=='A' for c in cases)!=12:raise ValueError('panel coverage')
 return cases

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();os.umask(0o077)
 if sha(a.manifest)!=MANIFEST_SHA:raise ValueError('tool manifest identity')
 manifest=json.loads(a.manifest.read_text());cases=build(manifest);a.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 panel=dict(schema='emender-e97-pi-native-baselines-v1',tool_manifest=TOOL_MANIFEST_REF,tool_manifest_sha256=sha(a.manifest),system='Complete the task using the declared Pi-native tools. Finish only when complete.',tools=manifest['model_visible_tools'],cases=cases,stages={'A':{'model_episodes':12,'run_first':True},'B':{'model_episodes':len(cases)-12,'run_only_after_stage_a_review':True}},tokenizer='p50k_base',max_context_tokens=65536,max_turns=12,generation_budget=1024,episode_generation_budget=4096,episode_seconds=300,automatic_retry=False,training_eligible=False,optimizer_updates=0)
 publish(a.output/'panel.json',panel);publish(a.output/'summary.json',dict(status='pi-native-baselines-frozen',cases=len(cases),stage_a=12,stage_b=len(cases)-12,copy_distances=list(DELAYS),model_generations=0,optimizer_updates=0,training_eligible=False,panel_sha256=sha(a.output/'panel.json')))
 print('PI_NATIVE_BASELINES_FROZEN',len(cases),12,len(cases)-12,sha(a.output/'panel.json'))
if __name__=='__main__':main()
