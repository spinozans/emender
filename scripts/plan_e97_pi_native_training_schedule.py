#!/usr/bin/env python3
"""Descriptor-only schedule planner; never materializes loss-bearing tensors."""
import argparse,hashlib,json,math,struct
from collections import Counter
from pathlib import Path
RECORD=struct.Struct('<QQQB7x');PACK=struct.Struct('<QQQQ')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(root,descriptor):
 path=root/Path(descriptor['path']).name
 if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError('payload identity')
 return path
def identity(args):return {'authority_manifest_sha256':args.authority_sha256,'pack_manifest_sha256':args.pack_sha256,'sampler_key':args.sampler_key,'data_world_size':args.world_size,'context_size':args.context_size,'split':'train','schema':'emender-record-pack-counter-v1'}
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
def permutation(meta,mode,epoch,modulus):
 digest=hashlib.sha256(encoded({**meta,'sampler_mode':mode,'epoch':epoch})).digest();multiplier=int.from_bytes(digest[:8],'little')%modulus
 while math.gcd(multiplier,modulus)!=1:multiplier=(multiplier+1)%modulus
 return multiplier,int.from_bytes(digest[8:16],'little')%modulus
def pack_id(meta,mode,rank,cursor,modulus):
 global_index=cursor*meta['data_world_size']+rank;epoch,position=divmod(global_index,modulus);multiplier,offset=permutation(meta,mode,epoch,modulus);return (multiplier*position+offset)%modulus
def sample_id(meta,mode,rank,cursor):
 digest=hashlib.sha256(encoded({**meta,'global_rank':rank,'absolute_rank_sample_index':cursor})).digest();return hashlib.sha256(digest+mode.encode('ascii')).hexdigest()
def plan(args):
 authority=json.loads((args.authority/'manifest.json').read_text());packs=json.loads((args.packs/'manifest.json').read_text())
 if sha(args.authority/'manifest.json')!=args.authority_sha256 or sha(args.packs/'manifest.json')!=args.pack_sha256 or authority['training_eligible'] or packs['training_eligible'] or packs['sampler_mode']!='epoch-permutation' or packs['authority_manifest_sha256']!=args.authority_sha256:raise ValueError('nontraining planning authority')
 outputs=authority['outputs'];poutputs=packs['outputs'];metadata=[json.loads(x) for x in verify(args.authority,outputs['metadata']).read_text().splitlines()];records=verify(args.authority,outputs['index']).read_bytes();mask=verify(args.authority,outputs['mask']).read_bytes();pack_rows=verify(args.packs,poutputs['train_index']).read_bytes();members=verify(args.packs,poutputs['pack_records']).read_bytes();pack_count=len(pack_rows)//PACK.size
 if not pack_count or len(records)!=len(metadata)*RECORD.size or len(members)%4:raise ValueError('descriptor shapes')
 member_ids=struct.unpack('<%dI'%(len(members)//4),members);meta=identity(args);mode='epoch-permutation';steps=[];occurrences=Counter();source_occurrences=Counter();source_unique_records={};seen_packs=set()
 for cursor in range(args.steps):
  step={'update':cursor+1,'rank_sample_ids':[],'pack_ids':[],'global_tokens':0,'global_targets':0,'source_targets':Counter(),'source_tokens':Counter()}
  for rank in range(args.world_size):
   pid=pack_id(meta,mode,rank,cursor,pack_count);seen_packs.add(pid);first,count,pack_tokens,pack_targets=PACK.unpack_from(pack_rows,pid*PACK.size);ids=member_ids[first:first+count];observed_tokens=observed_targets=0
   for rid in ids:
    offset,n,want,split=RECORD.unpack_from(records,rid*RECORD.size);effective=sum(mask[offset+1:offset+n]);row=metadata[rid];name=row['source']
    if split or effective!=want:raise ValueError('record target accounting')
    observed_tokens+=n;observed_targets+=effective;step['source_tokens'][name]+=n;step['source_targets'][name]+=effective;occurrences[rid]+=1;source_occurrences[name]+=1;source_unique_records.setdefault(name,set()).add(rid)
   if observed_tokens!=pack_tokens or observed_targets!=pack_targets:raise ValueError('pack accounting')
   step['pack_ids'].append(pid);step['rank_sample_ids'].append([sample_id(meta,mode,rank,cursor)]);step['global_tokens']+=pack_tokens;step['global_targets']+=pack_targets
  step['source_targets']=dict(step['source_targets']);step['source_tokens']=dict(step['source_tokens']);steps.append(step)
 totals=Counter();token_totals=Counter()
 for step in steps:totals.update(step['source_targets']);token_totals.update(step['source_tokens'])
 report={'schema':'emender-e97-pi-native-training-schedule-proposal-v1','status':'planning-only-not-authorized','authority_manifest_sha256':args.authority_sha256,'pack_manifest_sha256':args.pack_sha256,'sampler_key':args.sampler_key,'world_size':args.world_size,'context_size':args.context_size,'steps':steps,'source_target_totals':dict(totals),'source_token_totals':dict(token_totals),'actual_source_target_fractions':{k:v/sum(totals.values()) for k,v in totals.items()},'unique_packs':len(seen_packs),'unique_records':len(occurrences),'record_occurrences':sum(occurrences.values()),'source_unique_records':{k:len(v) for k,v in source_unique_records.items()},'source_record_occurrences':dict(source_occurrences),'training_eligible':False,'optimizer_updates_authorized':0,'scope':'descriptor-only frozen proposal; no Dataset construction and no loss-bearing tensors'}
 args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_TRAINING_SCHEDULE_PLANNED',sum(token_totals.values()),sum(totals.values()),len(seen_packs),sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--authority',type=Path,required=True);p.add_argument('--packs',type=Path,required=True);p.add_argument('--authority-sha256',required=True);p.add_argument('--pack-sha256',required=True);p.add_argument('--sampler-key',type=int,required=True);p.add_argument('--world-size',type=int,default=8);p.add_argument('--context-size',type=int,default=65536);p.add_argument('--steps',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if min(a.sampler_key,a.world_size,a.context_size,a.steps)<=0:raise ValueError('positive schedule dimensions')
 plan(a)
if __name__=='__main__':main()
