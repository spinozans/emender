#!/usr/bin/env python3
"""Prepare the non-authorizing repair authority for the interference repair tranche.

Combines three cohorts into one emender-e97-tulu3-masked-sft-v1 authority:

1. openhands-execution-rehearsal: a deterministic whole-record slice of the
   qualified OpenHands source-native fulltraj authority (the representation the
   authorized Pi-native tranche catastrophically interfered with).
2. pi-native-curriculum: the previously selected 2,012-record Pi-native authority.
3. representation-bridge-rehearsal: the full 2,223-record bridge authority.

Records are emitted in a deterministic weighted-fair interleave so that greedy
64K boundary packing produces cohort-mixed packs and every epoch-permutation
update sees all three cohorts. This script never materializes training tensors
beyond the authority payload copies and never trains.
"""
import argparse,hashlib,json,os,random,struct
from collections import Counter
from pathlib import Path
import numpy as np

INDEX=struct.Struct('<QQQB7x')
AUTHORITY_SCHEMA='emender-e97-tulu3-masked-sft-v1'
NATIVE_SCHEMA='emender-open-swe-source-native-candidate-v1'
CONTEXT_SIZE=65536
COHORTS=('openhands-execution-rehearsal','pi-native-curriculum','representation-bridge-rehearsal')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def desc(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}

def verify_outputs(root,manifest):
 for spec in manifest['outputs'].values():
  p=root/spec['path']
  if not spec['path'].isascii() or '/' in spec['path'] or p.stat().st_size!=spec['bytes'] or sha(p)!=spec['sha256']:
   raise ValueError('source payload identity')
 return {k:root/v['path'] for k,v in manifest['outputs'].items()}

def read_tulu3(root,label,expected_manifest_sha,*,eligible_required):
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError(f'{label} manifest identity')
 if manifest.get('schema')!=AUTHORITY_SCHEMA or manifest.get('status')!='complete':raise ValueError(f'{label} schema')
 if eligible_required and manifest.get('training_eligible') is not True:raise ValueError(f'{label} must be a training-eligible source')
 if not eligible_required and manifest.get('training_eligible') is not False:raise ValueError(f'{label} must be an explicitly non-admitted selection authority')
 paths=verify_outputs(root,manifest)
 rows=[json.loads(x) for x in paths['metadata'].read_text().splitlines()]
 index=paths['index'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError(f'{label} index shape')
 records=[]
 with paths['tokens'].open('rb') as tf,paths['mask'].open('rb') as mf:
  for i,row in enumerate(rows):
   offset,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
   if split:raise ValueError(f'{label} unexpected validation record')
   tf.seek(4*offset);tokens=tf.read(4*n)
   mf.seek(offset);mask=mf.read(n)
   if len(tokens)!=4*n or len(mask)!=n or sum(mask)!=want:raise ValueError(f'{label} record slice')
   records.append((tokens,mask,row))
 return records

def read_native_slice(root,expected_manifest_sha,seed,budget_targets):
 """Deterministic whole-record train-split slice of the native authority."""
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError('native manifest identity')
 if manifest.get('schema')!=NATIVE_SCHEMA:raise ValueError('native schema')
 if manifest.get('training_eligible') is True:raise ValueError('native source must not be an admitted authority')
 paths=verify_outputs(root,manifest)
 rows=[json.loads(x) for x in paths['records.jsonl'].read_text().splitlines()]
 index=paths['records.idx'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError('native index shape')
 mask=np.fromfile(paths['loss_mask.bin'],dtype=np.uint8)
 tokens_mm=np.memmap(paths['tokens.bin'],dtype='<u4',mode='r')
 eligible=[]
 for i,row in enumerate(rows):
  offset,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
  if split!=0:continue
  if n>CONTEXT_SIZE+1:continue
  if int(mask[offset:offset+n].sum())!=want:raise ValueError('native mask accounting')
  eligible.append((i,offset,n,want))
 rng=random.Random(seed)
 order=list(range(len(eligible)));rng.shuffle(order)
 chosen=[];seen_keys=set();consumed=0
 for j in order:
  i,offset,n,want=eligible[j]
  key=rows[i].get('problem_key')
  if not isinstance(key,str) or not key:raise ValueError('native problem key')
  if key in seen_keys:continue
  if consumed+want>budget_targets:continue
  seen_keys.add(key);consumed+=want
  token_bytes=tokens_mm[offset:offset+n].tobytes()
  mask_bytes=mask[offset:offset+n].tobytes()
  chosen.append((token_bytes,mask_bytes,dict(rows[i])))
  if len(chosen)>=10000:break
 if not chosen or consumed==0:raise ValueError('empty native slice')
 return chosen,consumed,seen_keys

def interleave(streams):
 """Weighted-fair merge by token share; deterministic tie-break by cohort order."""
 totals=[sum(len(t) for t,_,_ in records) for records,_,_ in streams]
 grand=sum(totals)
 if grand<=0:raise ValueError('empty authority')
 weights=[t/grand for t in totals]
 consumed=[0]*len(streams);positions=[0]*len(streams);order=[]
 while any(p<len(s[0]) for p,s in zip(positions,streams)):
  best=None;best_ratio=None
  for k in range(len(streams)):
   if positions[k]>=len(streams[k][0]):continue
   ratio=consumed[k]/weights[k]
   if best_ratio is None or ratio<best_ratio-1e-12:best,best_ratio=k,ratio
  k=best
  tokens,mask,row=streams[k][0][positions[k]]
  order.append((k,tokens,mask,row));consumed[k]+=len(tokens);positions[k]+=1
 return order

def prepare(args):
 native_manifest=json.loads((args.fulltraj/'manifest.json').read_text())
 selection_audit=json.loads(args.selection_audit.read_text())
 overlap_audit=json.loads(args.overlap_audit.read_text())
 if sha(args.selection_audit)!=args.selection_audit_sha or selection_audit['status']!='qualified-selection-not-admitted':raise ValueError('selection audit identity')
 if sha(args.overlap_audit)!=args.overlap_audit_sha or overlap_audit['status']!='pass':raise ValueError('overlap audit identity')
 if sha(args.parent_checkpoint)!=args.parent_sha:raise ValueError('parent checkpoint identity')
 native,consumed,keys=read_native_slice(args.fulltraj,args.fulltraj_sha,args.fulltraj_seed,args.fulltraj_budget_targets)
 selected=read_tulu3(args.selected,'pi-native-curriculum',args.selected_sha,eligible_required=False)
 rehearsal=read_tulu3(args.rehearsal,'representation-bridge-rehearsal',args.rehearsal_sha,eligible_required=True)
 streams=[(native,COHORTS[0]),(selected,COHORTS[1]),(rehearsal,COHORTS[2])]
 order=interleave(streams)
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 paths={k:args.output/v for k,v in {'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}.items()}
 offset=records=targets=0;sources=Counter();per_cohort=Counter()
 with paths['tokens'].open('xb') as tf,paths['mask'].open('xb') as mf,paths['index'].open('xb') as ix,paths['metadata'].open('x') as meta:
  for k,tokens,mask,row in order:
   cohort=COHORTS[k];n=len(mask);want=sum(mask)
   output=dict(row);output['source']=cohort
   if cohort==COHORTS[0]:output['source_record_id']=row.get('record_index');output['repair_provenance']={'authority_sha256':args.fulltraj_sha,'problem_key':row.get('problem_key')}
   elif cohort==COHORTS[1]:output['source_record_id']=row.get('id')
   else:output['source_record_id']=row.get('id')
   tf.write(tokens);mf.write(mask);ix.write(INDEX.pack(offset,n,want,0))
   meta.write(json.dumps(output,sort_keys=True)+'\n')
   offset+=n;records+=1;targets+=want;sources[cohort]+=want;per_cohort[cohort]+=1
 manifest={'schema':AUTHORITY_SCHEMA,'status':'complete',
  'purpose':'Non-authorizing repair-tranche exposure and packing preparation: OpenHands-execution rehearsal plus Pi-native curriculum plus bridge rehearsal, cohort-interleaved',
  'tokenizer':'p50k_base','training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,
  'counts':{'records':records,'train_records':records,'validation_records':0,'tokens':offset,'assistant_target_tokens':targets},
  'source_target_totals':dict(sources),'source_record_counts':dict(per_cohort),
  'interleave':{'scheme':'weighted-fair-by-token-share','cohort_order':list(COHORTS)},
  'openhands_rehearsal':{'authority':str(args.fulltraj.resolve()),'authority_sha256':args.fulltraj_sha,
   'seed':args.fulltraj_seed,'target_token_budget':args.fulltraj_budget_targets,'consumed_target_tokens':consumed,
   'records':len(native),'distinct_problem_keys':len(keys)},
  'selected_authority_sha256':args.selected_sha,'selected_selection_audit_sha256':args.selection_audit_sha,
  'selected_overlap_audit_sha256':args.overlap_audit_sha,'rehearsal_authority_sha256':args.rehearsal_sha,
  'parent_checkpoint':str(args.parent_checkpoint.resolve()),'parent_checkpoint_sha256':args.parent_sha,
  'outputs':{k:desc(v) for k,v in paths.items()}}
 tmp=args.output/'manifest.json.partial';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,args.output/'manifest.json')
 print('PI_NATIVE_REPAIR_PREPARATION',records,offset,targets,sha(args.output/'manifest.json'))

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--fulltraj',type=Path,required=True);p.add_argument('--fulltraj-sha',required=True)
 p.add_argument('--fulltraj-budget-targets',type=int,required=True);p.add_argument('--fulltraj-seed',type=int,required=True)
 p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-sha',required=True)
 p.add_argument('--selection-audit-sha',required=True);p.add_argument('--overlap-audit-sha',required=True)
 p.add_argument('--rehearsal',type=Path,required=True);p.add_argument('--rehearsal-sha',required=True)
 p.add_argument('--parent-checkpoint',type=Path,required=True);p.add_argument('--parent-sha',required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.fulltraj_budget_targets<=0:raise ValueError('positive budget required')
 prepare(a)

if __name__=='__main__':main()
