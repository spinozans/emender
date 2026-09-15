#!/usr/bin/env python3
"""Select a frozen decontaminated subset without re-execution or relabeling."""
import argparse,hashlib,json,os,struct
from collections import Counter
from pathlib import Path
INDEX=struct.Struct('<QQQB7x');EXCLUDED_FAMILIES=frozenset({'preservation'})
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def publish(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def descriptor(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}
def select(args):
 source=args.source_root;source_manifest=json.loads((source/'candidate-authority/manifest.json').read_text());audit=json.loads((source/'audit.json').read_text());plan=json.loads(args.plan.read_text())
 if sha(source/'audit.json')!=args.audit_sha or audit['status']!='qualified-candidates-not-admitted' or audit['authority_sha256']!=sha(source/'candidate-authority/manifest.json') or source_manifest['plan_sha256']!=args.plan_sha or sha(args.plan)!=args.plan_sha:raise ValueError('source authority')
 if audit['training_eligible'] or audit['packing_authorized'] or audit['optimizer_updates_authorized'] or plan['training_eligible'] or plan['packing_authorized'] or plan['optimizer_updates']:raise ValueError('source authorization')
 authority=source/'candidate-authority';rows=[json.loads(x) for x in (authority/'records.jsonl').read_text().splitlines()];raw_index=(authority/'records.idx').read_bytes();raw_tokens=(authority/'tokens.uint32.bin').read_bytes();raw_mask=(authority/'assistant_mask.uint8.bin').read_bytes();selected=[];excluded=[]
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);out=args.output/'candidate-authority';out.mkdir();offset=targets=0
 with (out/'tokens.uint32.bin').open('xb') as tf,(out/'assistant_mask.uint8.bin').open('xb') as mf,(out/'records.idx').open('xb') as ix,(out/'records.jsonl').open('x') as meta:
  for i,row in enumerate(rows):
   start,n,want,split=INDEX.unpack_from(raw_index,i*INDEX.size)
   if split or row['targets']!=want:raise ValueError('source record index')
   if row['family'] in EXCLUDED_FAMILIES:excluded.append({'id':row['id'],'family':row['family'],'reason':'frozen-family-exclusion'});continue
   token_slice=raw_tokens[4*start:4*(start+n)];mask_slice=raw_mask[start:start+n]
   if len(token_slice)!=4*n or len(mask_slice)!=n or sum(mask_slice)!=want:raise ValueError('source record slice')
   tf.write(token_slice);mf.write(mask_slice);ix.write(INDEX.pack(offset,n,want,0));meta.write(json.dumps(row,sort_keys=True)+'\n');offset+=n;targets+=want;selected.append(row)
 (args.output/'excluded.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in excluded))
 if len(selected)<args.minimum_records or len({x['sequence_sha256'] for x in selected})!=len(selected):raise ValueError('selection minimum/dedup')
 outputs={name:descriptor(out/file) for name,file in {'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}.items()}
 manifest={'schema':'emender-e97-pi-native-selected-candidate-authority-v1','status':'verified-selection-not-admitted','records':len(selected),'tokens':offset,'assistant_target_tokens':targets,'source_authority_sha256':sha(source/'candidate-authority/manifest.json'),'source_audit_sha256':args.audit_sha,'plan_sha256':args.plan_sha,'selection_checker_sha256':sha(__file__),'excluded_families':sorted(EXCLUDED_FAMILIES),'excluded_records':len(excluded),'outputs':outputs,'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0};publish(out/'manifest.json',manifest)
 summary={'schema':'emender-e97-pi-native-selection-summary-v1','status':'verified-selection-not-admitted','records':len(selected),'excluded_records':len(excluded),'families':dict(Counter(x['family'] for x in selected)),'repository_discovery_records':sum(x['repository_discovery'] for x in selected),'authority_sha256':sha(out/'manifest.json'),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0};publish(args.output/'summary.json',summary);print('PI_NATIVE_CURRICULUM_SELECTED',len(selected),len(excluded),offset,targets,sha(out/'manifest.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--audit-sha',required=True);p.add_argument('--minimum-records',type=int,required=True);p.add_argument('--output',type=Path,required=True);select(p.parse_args())
if __name__=='__main__':main()
