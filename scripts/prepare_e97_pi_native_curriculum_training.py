#!/usr/bin/env python3
"""Prepare a non-trainable Pi-native/rehearsal authority for exposure planning."""
import argparse,hashlib,json,os,struct
from collections import Counter
from pathlib import Path
INDEX=struct.Struct('<QQQB7x');AUTHORITY_SCHEMA='emender-e97-tulu3-masked-sft-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}
def source(root,expected):
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected:raise ValueError('source manifest identity')
 outputs=manifest['outputs'];paths={k:root/v['path'] for k,v in outputs.items()}
 for k,p in paths.items():
  if p.stat().st_size!=outputs[k]['bytes'] or sha(p)!=outputs[k]['sha256']:raise ValueError('source payload identity')
 return manifest,paths
def prepare(args):
 selection_audit=json.loads((args.selected/'selection-audit.json').read_text());overlap_audit=json.loads((args.selected/'overlap-audit.json').read_text())
 if sha(args.selected/'selection-audit.json')!=args.selection_audit_sha or selection_audit['status']!='qualified-selection-not-admitted' or sha(args.selected/'overlap-audit.json')!=args.overlap_audit_sha or overlap_audit['status']!='pass':raise ValueError('selected audits')
 selected_manifest,selected=source(args.selected/'candidate-authority',args.selected_sha);rehearsal_manifest,rehearsal=source(args.rehearsal,args.rehearsal_sha)
 if sha(args.parent_checkpoint)!=args.parent_sha:raise ValueError('parent checkpoint identity')
 if selected_manifest['training_eligible'] or selected_manifest['packing_authorized'] or selected_manifest['optimizer_updates_authorized'] or rehearsal_manifest['schema']!=AUTHORITY_SCHEMA or not rehearsal_manifest['training_eligible']:raise ValueError('source eligibility')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);paths={k:args.output/v for k,v in {'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}.items()};offset=records=targets=0;sources=Counter()
 with paths['tokens'].open('xb') as tf,paths['mask'].open('xb') as mf,paths['index'].open('xb') as ix,paths['metadata'].open('x') as meta:
  for label,root,items in (('pi-native-curriculum',args.selected,selected),('representation-bridge-rehearsal',args.rehearsal,rehearsal)):
   rows=[json.loads(x) for x in items['metadata'].read_text().splitlines()];index=items['index'].read_bytes();tokens=items['tokens'].read_bytes();mask=items['mask'].read_bytes()
   if len(index)!=INDEX.size*len(rows):raise ValueError('source index shape')
   for i,row in enumerate(rows):
    start,n,want,split=INDEX.unpack_from(index,i*INDEX.size)
    if split:raise ValueError('unexpected validation record')
    token_slice=tokens[4*start:4*(start+n)];mask_slice=mask[start:start+n]
    if len(token_slice)!=4*n or len(mask_slice)!=n or sum(mask_slice)!=want:raise ValueError('source record slice')
    output=dict(row)
    if label=='pi-native-curriculum':output['source']='pi-native-curriculum';output['source_record_id']=row['id']
    source_name=output.get('source')
    if not isinstance(source_name,str):raise ValueError('metadata source')
    tf.write(token_slice);mf.write(mask_slice);ix.write(INDEX.pack(offset,n,want,0));meta.write(json.dumps(output,sort_keys=True)+'\n');offset+=n;records+=1;targets+=want;sources[source_name]+=want
 outputs={k:desc(v) for k,v in paths.items()};manifest={'schema':AUTHORITY_SCHEMA,'status':'complete','purpose':'Non-authorizing Pi-native curriculum exposure and packing preparation','tokenizer':'p50k_base','training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,'counts':{'records':records,'train_records':records,'validation_records':0,'tokens':offset,'assistant_target_tokens':targets},'source_target_totals':dict(sources),'selected_authority_sha256':args.selected_sha,'selected_selection_audit_sha256':args.selection_audit_sha,'selected_overlap_audit_sha256':args.overlap_audit_sha,'rehearsal_authority_sha256':args.rehearsal_sha,'parent_checkpoint':str(args.parent_checkpoint.resolve()),'parent_checkpoint_sha256':args.parent_sha,'outputs':outputs};tmp=args.output/'manifest.json.partial';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,args.output/'manifest.json');print('PI_NATIVE_TRAINING_PREPARATION',records,offset,targets,sha(args.output/'manifest.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-sha',required=True);p.add_argument('--selection-audit-sha',required=True);p.add_argument('--overlap-audit-sha',required=True);p.add_argument('--rehearsal',type=Path,required=True);p.add_argument('--rehearsal-sha',required=True);p.add_argument('--parent-checkpoint',type=Path,required=True);p.add_argument('--parent-sha',required=True);p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
if __name__=='__main__':main()
