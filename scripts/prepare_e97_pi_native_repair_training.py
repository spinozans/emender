#!/usr/bin/env python3
"""Prepare the non-authorizing repair authority for the interference repair tranche.

Combines three cohorts into one emender-e97-tulu3-masked-sft-v1 authority:

1. openhands-execution-rehearsal: a deterministic whole-record slice of the
   qualified OpenHands source-native fulltraj authority (the representation the
   authorized Pi-native tranche catastrophically interfered with).
2. pi-native-curriculum: the previously selected 2,012-record Pi-native authority.
3. representation-bridge-rehearsal: the full 2,223-record bridge authority (optional; omitted in the
   translated-OH mode where the scrub report drops it).

Alternatively, the OpenHands stream is replaced by a translated + replay-verified
OH candidate authority (--translated-oh-source): every record re-rendered in the
canonical e97-pi-native-v1 form and machine-replayed against its recorded
repository state, with instance_id dedup and budget-bounded seeded selection.

Records are emitted in a deterministic weighted-fair interleave so that greedy
64K boundary packing produces cohort-mixed packs and every epoch-permutation
update sees all cohorts. This script never materializes training tensors
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

def read_tulu3(root,label,expected_manifest_sha,*,eligible_required,schema=AUTHORITY_SCHEMA,status='complete',include_families=None):
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError(f'{label} manifest identity')
 if manifest.get('schema')!=schema or manifest.get('status')!=status:raise ValueError(f'{label} schema')
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
   if include_families is not None and row.get('family') not in include_families:continue
   records.append((tokens,mask,row))
 if include_families is not None and not records:raise ValueError(f'{label} family filter selected nothing')
 return records

def read_authored_slice(root,expected_manifest_sha,seed,budget_targets):
 """Deterministic seeded whole-record slice of a training-eligible tulu3 authority."""
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError('authored manifest identity')
 if manifest.get('schema')!=AUTHORITY_SCHEMA or manifest.get('status')!='complete':raise ValueError('authored schema')
 if manifest.get('training_eligible') is not True:raise ValueError('authored source must be training-eligible')
 paths=verify_outputs(root,manifest)
 rows=[json.loads(x) for x in paths['metadata'].read_text().splitlines()]
 index=paths['index'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError('authored index shape')
 eligible=[]
 for i,row in enumerate(rows):
  offset,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
  if split:raise ValueError('authored validation record')
  eligible.append((i,offset,n,want))
 rng=random.Random(seed);order=list(range(len(eligible)));rng.shuffle(order)
 chosen=[];consumed=0
 for j in order:
  i,offset,n,want=eligible[j]
  if consumed+want>budget_targets:continue
  with paths['tokens'].open('rb') as tf,paths['mask'].open('rb') as mf:
   tf.seek(4*offset);tokens=tf.read(4*n);mf.seek(offset);mask=mf.read(n)
  if len(tokens)!=4*n or len(mask)!=n or sum(mask)!=want:raise ValueError('authored record slice')
  consumed+=want;chosen.append((tokens,mask,dict(rows[i])))
 if not chosen:raise ValueError('empty authored slice')
 return chosen,consumed

def read_conversation_slice(root,expected_manifest_sha,seed,budget_targets):
 """Deterministic seeded whole-record slice of a production-admitted tulu3 conversation authority.
 Streams the metadata once and seek-reads only chosen records so very large authorities are tractable.
 The source manifest may mark training_eligible True or omit it (production-admitted raw source); the raw value is
 returned so the preparation manifest records exactly what was consumed."""
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError('conversation manifest identity')
 if manifest.get('schema')!=AUTHORITY_SCHEMA or manifest.get('status')!='complete':raise ValueError('conversation schema')
 if manifest.get('training_eligible') is not True and 'training_eligible' in manifest:raise ValueError('conversation source must be a production-admitted authority')
 paths={}
 for key,spec in manifest['outputs'].items():
  p=Path(spec['path'])
  if not p.is_absolute():p=root/p
  if not p.exists():p=root/Path(spec['path']).name
  if not spec['path'].isascii() or p.stat().st_size!=spec['bytes'] or sha(p)!=spec['sha256']:raise ValueError('conversation payload identity')
  paths[key]=p
 rows=[]
 for line in paths['metadata'].open():rows.append(json.loads(line))
 index=paths['index'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError('conversation index shape')
 eligible=[]
 for i,row in enumerate(rows):
  offset,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
  if split or n>65536:continue
  eligible.append((i,offset,n,want))
 rng=random.Random(seed);order=list(range(len(eligible)));rng.shuffle(order)
 chosen=[];consumed=0
 with paths['tokens'].open('rb') as tf,paths['mask'].open('rb') as mf:
  for j in order:
   i,offset,n,want=eligible[j]
   if consumed+want>budget_targets:continue
   tf.seek(4*offset);tokens=tf.read(4*n);mf.seek(offset);mask=mf.read(n)
   if len(tokens)!=4*n or len(mask)!=n or sum(mask)!=want:raise ValueError('conversation record slice')
   consumed+=want;chosen.append((tokens,mask,dict(rows[i])))
 if not chosen:raise ValueError('empty conversation slice')
 return chosen,consumed,manifest.get('training_eligible')

def read_translated_oh_slice(root,expected_manifest_sha,seed,budget_targets):
 """Deterministic seeded whole-record slice of the translated + replay-verified OH
 candidate authority (e97-oh-pi-native-translation-v1). Deduplicates by
 SWE-rebench instance_id, shortest-first with a seeded tie-break, so a
 budget-bounded slice maximizes problem diversity while keeping whole-record
 64K packing feasible. The source must be an explicitly non-admitted complete
 candidate authority; eligibility comes only from admission."""
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected_manifest_sha:raise ValueError('translated-oh manifest identity')
 if manifest.get('schema')!=AUTHORITY_SCHEMA or manifest.get('status')!='complete':raise ValueError('translated-oh schema')
 if manifest.get('training_eligible') is not False:raise ValueError('translated-oh source must be an explicitly non-admitted candidate authority')
 raw=verify_outputs(root,manifest)
 # The translated collection's manifest keys outputs by payload filename; alias to roles.
 paths={}
 for k,v in raw.items():
  if k in ('tokens','mask','index','metadata'):paths[k]=v
 for k,v in raw.items():
  name=Path(v).name
  if name.startswith('tokens.uint32'):paths.setdefault('tokens',v)
  elif name.startswith('assistant_mask'):paths.setdefault('mask',v)
  elif name.startswith('records.idx'):paths.setdefault('index',v)
  elif name.startswith('records.jsonl'):paths.setdefault('metadata',v)
 if set(paths)!={'tokens','mask','index','metadata'}:raise ValueError('translated-oh outputs incomplete')
 rows=[json.loads(x) for x in paths['metadata'].read_text().splitlines()]
 index=paths['index'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError('translated-oh index shape')
 eligible=[]
 for i,row in enumerate(rows):
  offset,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
  if split or n>CONTEXT_SIZE+1:continue
  key=row.get('instance_id')
  if not isinstance(key,str) or not key:raise ValueError('translated-oh instance id')
  eligible.append((i,offset,n,want,key))
 rng=random.Random(seed)
 jitter={i:rng.random() for i,_,_,_,_ in eligible}
 eligible_sorted=sorted(eligible,key=lambda e:(e[2],jitter[e[0]]))
 chosen=[];seen=set();consumed=0
 for i,offset,n,want,key in eligible_sorted:
  if key in seen:continue
  if consumed+want>budget_targets:continue
  with paths['tokens'].open('rb') as tf,paths['mask'].open('rb') as mf:
   tf.seek(4*offset);tokens=tf.read(4*n);mf.seek(offset);mask=mf.read(n)
  if len(tokens)!=4*n or len(mask)!=n or sum(mask)!=want:raise ValueError('translated-oh record slice')
  seen.add(key);consumed+=want;chosen.append((tokens,mask,dict(rows[i])))
 if not chosen:raise ValueError('empty translated-oh slice')
 return chosen,consumed,seen

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
 # Shortest-first with a seeded deterministic tie-break: short native records
 # share 64K packs with the other cohorts, which is what makes every update
 # see every cohort under whole-record greedy packing.
 jitter={i:rng.random() for i,_,_,_ in eligible}
 eligible_sorted=sorted(eligible,key=lambda e:(e[2],jitter[e[0]]))
 chosen=[];seen_keys=set();consumed=0
 for i,offset,n,want in eligible_sorted:
  key=rows[i].get('problem_key')
  if isinstance(key,(str,int)) and not isinstance(key,bool):key=str(key)
  elif isinstance(key,(list,tuple)):key=json.dumps(key,sort_keys=True,separators=(',',':'))
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
 totals=[sum(len(t) for t,_,_ in records) for records,_ in streams]
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
 if (args.fulltraj is None)==(args.translated_oh_source is None):raise ValueError('exactly one of --fulltraj or --translated-oh-source is required')
 if args.translated_oh_source and (not args.translated_oh_sha or args.translated_oh_budget_targets<=0):raise ValueError('translated-oh requires sha and positive budget')
 selection_audit=json.loads(args.selection_audit.read_text())
 overlap_audit=json.loads(args.overlap_audit.read_text())
 if sha(args.selection_audit)!=args.selection_audit_sha or selection_audit['status']!='qualified-selection-not-admitted':raise ValueError('selection audit identity')
 if sha(args.overlap_audit)!=args.overlap_audit_sha or overlap_audit['status']!='pass':raise ValueError('overlap audit identity')
 if sha(args.parent_checkpoint)!=args.parent_sha:raise ValueError('parent checkpoint identity')
 if args.translated_oh_source:
  native,oh_consumed,keys=read_translated_oh_slice(args.translated_oh_source,args.translated_oh_sha,args.translated_oh_seed,args.translated_oh_budget_targets)
  oh_name=args.translated_oh_cohort
 else:
  native_manifest=json.loads((args.fulltraj/'manifest.json').read_text())
  native,oh_consumed,keys=read_native_slice(args.fulltraj,args.fulltraj_sha,args.fulltraj_seed,args.fulltraj_budget_targets)
  oh_name=COHORTS[0]
 include=None
 if args.pi_native_include_families:
  include=tuple(x.strip() for x in args.pi_native_include_families.split(',') if x.strip())
 selected=read_tulu3(args.selected,'pi-native-curriculum',args.selected_sha,eligible_required=False,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted',include_families=include)
 cohort_names=[oh_name,'pi-native-curriculum']
 streams=[(native,cohort_names[0]),(selected,cohort_names[1])]
 rehearsal=None
 if args.rehearsal is not None:
  cohort_names.append('representation-bridge-rehearsal')
  rehearsal=read_tulu3(args.rehearsal,'representation-bridge-rehearsal',args.rehearsal_sha,eligible_required=True)
  streams.append((rehearsal,cohort_names[len(cohort_names)-1]))
 authored=None;authored_consumed=0
 if args.authored_source:
  cohort_names.append('grounded-authored-rehearsal')
  authored,authored_consumed=read_authored_slice(args.authored_source,args.authored_sha,args.authored_seed,args.authored_budget_targets)
  streams.append((authored,cohort_names[len(cohort_names)-1]))
 correction=None
 if args.correction_source:
  cohort_names.append('grounded-correction-rehearsal')
  correction=read_tulu3(args.correction_source,'grounded-correction-rehearsal',args.correction_sha,eligible_required=True)
  streams.append((correction,cohort_names[len(cohort_names)-1]))
 loopbreak=None
 if args.loopbreak_source:
  cohort_names.append('loopbreak-rehearsal')
  loopbreak=read_tulu3(args.loopbreak_source,'loopbreak-rehearsal',args.loopbreak_sha,eligible_required=False,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
  streams.append((loopbreak,cohort_names[len(cohort_names)-1]))
 conversation=None;conversation_consumed=0;conversation_eligibility=None
 if args.conversation_source:
  cohort_names.append('conversation-rehearsal')
  conversation,conversation_consumed,conversation_eligibility=read_conversation_slice(args.conversation_source,args.conversation_sha,args.conversation_seed,args.conversation_budget_targets)
  streams.append((conversation,cohort_names[len(cohort_names)-1]))
 extra=None
 if args.extra_source:
  cohort_names.append(args.extra_cohort)
  extra=read_tulu3(args.extra_source,args.extra_cohort,args.extra_sha,eligible_required=False,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
  streams.append((extra,cohort_names[len(cohort_names)-1]))
 extra2=None
 if args.extra2_source:
  cohort_names.append(args.extra2_cohort)
  extra2=read_tulu3(args.extra2_source,args.extra2_cohort,args.extra2_sha,eligible_required=False,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
  streams.append((extra2,cohort_names[len(cohort_names)-1]))
 extra3=None
 if args.extra3_source:
  cohort_names.append(args.extra3_cohort)
  extra3=read_tulu3(args.extra3_source,args.extra3_cohort,args.extra3_sha,eligible_required=False,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
  streams.append((extra3,cohort_names[len(cohort_names)-1]))
 spec_cohorts=[]
 for spec_path in (args.cohort_spec or []):
  spec=json.loads(Path(spec_path).read_text())
  name=spec['cohort']
  if name in cohort_names:raise ValueError(f'duplicate cohort {name}')
  cohort_names.append(name)
  records,consumed,eligibility=read_conversation_slice(Path(spec['root']),spec['sha256'],spec.get('seed',0),spec['budget_targets'])
  streams.append((records,cohort_names[len(cohort_names)-1]))
  spec_cohorts.append({'cohort':name,'authority':spec['root'],'authority_sha256':spec['sha256'],'seed':spec.get('seed',0),'target_token_budget':spec['budget_targets'],'consumed_target_tokens':consumed,'records':len(records),'source_training_eligible':eligibility})
 order=interleave(streams)
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 paths={k:args.output/v for k,v in {'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}.items()}
 offset=records=targets=0;sources=Counter();per_cohort=Counter()
 with paths['tokens'].open('xb') as tf,paths['mask'].open('xb') as mf,paths['index'].open('xb') as ix,paths['metadata'].open('x') as meta:
  for k,tokens,mask,row in order:
   cohort=cohort_names[k];n=len(mask);want=sum(mask)
   output=dict(row);output['source']=cohort
   if cohort==cohort_names[0]:
    output['source_record_id']=row.get('record_index')
    if args.translated_oh_source:output['repair_provenance']={'authority_sha256':args.translated_oh_sha,'instance_id':row.get('instance_id'),'trajectory_id':row.get('trajectory_id')}
    else:output['repair_provenance']={'authority_sha256':args.fulltraj_sha,'problem_key':row.get('problem_key')}
   else:output['source_record_id']=row.get('id')
   tf.write(tokens);mf.write(mask);ix.write(INDEX.pack(offset,n,want,0))
   meta.write(json.dumps(output,sort_keys=True)+'\n')
   offset+=n;records+=1;targets+=want;sources[cohort]+=want;per_cohort[cohort]+=1
 manifest={'schema':AUTHORITY_SCHEMA,'status':'complete',
  'purpose':('Non-authorizing repair-tranche exposure and packing preparation: translated replay-verified OpenHands rehearsal plus Pi-native curriculum, cohort-interleaved (representation-bridge and grounded-authored cohorts dropped per the OH scrub report)' if args.translated_oh_source else 'Non-authorizing repair-tranche exposure and packing preparation: OpenHands-execution rehearsal plus Pi-native curriculum plus bridge rehearsal, cohort-interleaved'),
  'tokenizer':'p50k_base','training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,
  'counts':{'records':records,'train_records':records,'validation_records':0,'tokens':offset,'assistant_target_tokens':targets},
  'source_target_totals':dict(sources),'source_record_counts':dict(per_cohort),
  'interleave':{'scheme':'weighted-fair-by-token-share','cohort_order':cohort_names},
  'pi_native_family_filter':(list(include) if include else None),
  'openhands_rehearsal': ({'authority':str(args.translated_oh_source.resolve()),'authority_sha256':args.translated_oh_sha,'cohort':oh_name,'seed':args.translated_oh_seed,'target_token_budget':args.translated_oh_budget_targets,'consumed_target_tokens':oh_consumed,'records':len(native),'distinct_instance_ids':len(keys),'source_collection':'e97-oh-pi-native-translation-v1 translated+replay-verified'} if args.translated_oh_source else {'authority':str(args.fulltraj.resolve()),'authority_sha256':args.fulltraj_sha,
   'seed':args.fulltraj_seed,'target_token_budget':args.fulltraj_budget_targets,'consumed_target_tokens':oh_consumed,
   'records':len(native),'distinct_problem_keys':len(keys)}),
  'selected_authority_sha256':args.selected_sha,'selected_selection_audit_sha256':args.selection_audit_sha,
  'selected_overlap_audit_sha256':args.overlap_audit_sha,'rehearsal_authority_sha256':args.rehearsal_sha,
  'parent_checkpoint':str(args.parent_checkpoint.resolve()),'parent_checkpoint_sha256':args.parent_sha,
  **({'loopbreak_rehearsal':{'authority':str(args.loopbreak_source.resolve()),'authority_sha256':args.loopbreak_sha,'records':len(loopbreak)}} if args.loopbreak_source else {'loopbreak_rehearsal':None}),
  **({'conversation_rehearsal':{'authority':str(args.conversation_source.resolve()),'authority_sha256':args.conversation_sha,'seed':args.conversation_seed,'target_token_budget':args.conversation_budget_targets,'consumed_target_tokens':conversation_consumed,'records':len(conversation),'source_training_eligible':conversation_eligibility}} if args.conversation_source else {'conversation_rehearsal':None}),
  **({'extra_rehearsal':{'authority':str(args.extra_source.resolve()),'authority_sha256':args.extra_sha,'cohort':args.extra_cohort,'records':len(extra)}} if args.extra_source else {'extra_rehearsal':None}),
  **({'extra2_rehearsal':{'authority':str(args.extra2_source.resolve()),'authority_sha256':args.extra2_sha,'cohort':args.extra2_cohort,'records':len(extra2)}} if args.extra2_source else {'extra2_rehearsal':None}),
  **({'extra3_rehearsal':{'authority':str(args.extra3_source.resolve()),'authority_sha256':args.extra3_sha,'cohort':args.extra3_cohort,'records':len(extra3)}} if args.extra3_source else {'extra3_rehearsal':None}),
  'spec_cohorts':spec_cohorts,
  **({'correction_rehearsal':{'authority':str(args.correction_source.resolve()),'authority_sha256':args.correction_sha,'records':len(correction)}} if args.correction_source else {'correction_rehearsal':None}),
  **({'authored_rehearsal':{'authority':str(args.authored_source.resolve()),'authority_sha256':args.authored_sha,'seed':args.authored_seed,'target_token_budget':args.authored_budget_targets,'consumed_target_tokens':authored_consumed,'records':len(authored)},'authored_source_sha256':args.authored_sha} if args.authored_source else {'authored_rehearsal':None}),
  'outputs':{k:desc(v) for k,v in paths.items()}}
 tmp=args.output/'manifest.json.partial';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,args.output/'manifest.json')
 print('PI_NATIVE_REPAIR_PREPARATION',records,offset,targets,sha(args.output/'manifest.json'))

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--fulltraj',type=Path,default=None);p.add_argument('--fulltraj-sha',default=None)
 p.add_argument('--fulltraj-budget-targets',type=int,default=0);p.add_argument('--fulltraj-seed',type=int,default=0)
 p.add_argument('--translated-oh-source',type=Path,default=None);p.add_argument('--translated-oh-sha',default=None)
 p.add_argument('--translated-oh-cohort',default='openhands-translated-rehearsal')
 p.add_argument('--translated-oh-budget-targets',type=int,default=0);p.add_argument('--translated-oh-seed',type=int,default=0)
 p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-sha',required=True)
 p.add_argument('--selection-audit',type=Path,required=True);p.add_argument('--selection-audit-sha',required=True)
 p.add_argument('--overlap-audit',type=Path,required=True);p.add_argument('--overlap-audit-sha',required=True)
 p.add_argument('--rehearsal',type=Path,default=None);p.add_argument('--rehearsal-sha',default=None)
 p.add_argument('--parent-checkpoint',type=Path,required=True);p.add_argument('--parent-sha',required=True)
 p.add_argument('--authored-source',type=Path,default=None);p.add_argument('--authored-sha',default=None)
 p.add_argument('--authored-budget-targets',type=int,default=0);p.add_argument('--authored-seed',type=int,default=0)
 p.add_argument('--pi-native-include-families',default=None)
 p.add_argument('--correction-source',type=Path,default=None);p.add_argument('--correction-sha',default=None)
 p.add_argument('--loopbreak-source',type=Path,default=None);p.add_argument('--loopbreak-sha',default=None)
 p.add_argument('--conversation-source',type=Path,default=None);p.add_argument('--conversation-sha',default=None)
 p.add_argument('--conversation-budget-targets',type=int,default=0);p.add_argument('--conversation-seed',type=int,default=0)
 p.add_argument('--extra-source',type=Path,default=None);p.add_argument('--extra-sha',default=None);p.add_argument('--extra-cohort',default=None)
 p.add_argument('--extra2-source',type=Path,default=None);p.add_argument('--extra2-sha',default=None);p.add_argument('--extra2-cohort',default=None)
 p.add_argument('--extra3-source',type=Path,default=None);p.add_argument('--extra3-sha',default=None);p.add_argument('--extra3-cohort',default=None)
 p.add_argument('--cohort-spec',action='append',default=None)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.translated_oh_source is None and a.fulltraj_budget_targets<=0:raise ValueError('positive budget required')
 prepare(a)

if __name__=='__main__':main()
