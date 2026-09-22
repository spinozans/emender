#!/usr/bin/env python3
"""Render the e97 hybrid-conversation v2 verified candidate collection as a
non-trainable masked-SFT authority (emender-e97-tulu3-masked-sft-v1) for 64K
pack validation and cohort-spec share planning.

Byte-exact copies of the audited candidate-authority token/mask/index
payloads; metadata rows gain source='hybrid-conversation-rehearsal-v2' (the
prep interleave/include-source convention). The render is bound fail-closed
to the frozen plan, the full reconstruction audit, and the protected-panel
overlap audit. training_eligible stays false: nothing here admits anything
to training.
"""
import argparse,hashlib,json,os,struct
from pathlib import Path
INDEX=struct.Struct('<QQQB7x')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}

def load_candidate(root,expected):
 manifest=json.loads((root/'manifest.json').read_text())
 if sha(root/'manifest.json')!=expected:raise ValueError('candidate manifest identity')
 if (manifest.get('schema')!='emender-e97-pi-native-candidate-authority-v1'
  or manifest.get('status')!='verified-candidate-not-admitted'
  or manifest.get('training_eligible') is not False
  or manifest.get('plan_sha256') is None):raise ValueError('candidate authority shape')
 outputs=manifest['outputs'];paths={}
 for key in ('tokens','mask','index','metadata'):
  raw=Path(outputs[key]['path']);path=raw if raw.is_absolute() else root/raw
  if path.stat().st_size!=outputs[key]['bytes'] or sha(path)!=outputs[key]['sha256']:raise ValueError(f'candidate payload identity: {key}')
  paths[key]=path
 rows=[json.loads(x) for x in paths['metadata'].read_text().splitlines()]
 index=paths['index'].read_bytes()
 if len(index)!=INDEX.size*len(rows):raise ValueError('candidate index shape')
 if manifest['counts']['records']!=len(rows):raise ValueError('candidate record count')
 return manifest,paths,rows,index

def check_receipt(path,expected,schema,status):
 receipt=json.loads(Path(path).read_text())
 if sha(path)!=expected or receipt.get('schema')!=schema or receipt.get('status')!=status:raise ValueError('receipt identity')
 if receipt.get('training_eligible') is not False:raise ValueError('receipt eligibility')
 return receipt

def render(args):
 candidate_manifest,paths,rows,index_bytes=load_candidate(args.candidate,args.manifest_sha)
 audit=check_receipt(args.audit,args.audit_sha,'emender-e97-pi-native-curriculum-audit-v1','qualified-candidates-not-admitted')
 overlap=check_receipt(args.overlap,args.overlap_sha,'emender-e97-hybrid-conversation-overlap-audit-v1','pass')
 if (audit['records']!=len(rows) or audit['plan_sha256']!=candidate_manifest['plan_sha256']
  or overlap['records']!=len(rows) or overlap['plan_sha256']!=candidate_manifest['plan_sha256']):raise ValueError('receipt binding')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 outputs={k:args.output/v for k,v in {'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}.items()}
 offset=records=targets=0
 with outputs['tokens'].open('xb') as tf,outputs['mask'].open('xb') as mf,outputs['index'].open('xb') as ix,outputs['metadata'].open('x') as meta,paths['tokens'].open('rb') as src_t,paths['mask'].open('rb') as src_m:
  for i,row in enumerate(rows):
   start,n,want,split=INDEX.unpack_from(index_bytes,i*INDEX.size)
   if split:raise ValueError('unexpected validation record')
   src_t.seek(4*start);tokens=src_t.read(4*n)
   src_m.seek(start);mask=src_m.read(n)
   if len(tokens)!=4*n or len(mask)!=n or sum(mask)!=want:raise ValueError('record slice')
   out_row=dict(row);out_row['source']='hybrid-conversation-rehearsal-v2'
   tf.write(tokens);mf.write(mask);ix.write(INDEX.pack(offset,n,want,0));meta.write(json.dumps(out_row,sort_keys=True)+'\n')
   offset+=n;records+=1;targets+=want
 manifest={'schema':'emender-e97-tulu3-masked-sft-v1','status':'complete',
  'purpose':'Non-trainable masked-SFT render of the e97 hybrid-conversation v2 verified candidate collection (64K pack validation + cohort-spec share planning); not admitted to training',
  'tokenizer':'p50k_base','training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0,
  'counts':{'records':records,'train_records':records,'validation_records':0,'tokens':offset,'assistant_target_tokens':targets},
  'source_target_totals':{'hybrid-conversation-rehearsal-v2':targets},
  'plan_sha256':candidate_manifest['plan_sha256'],'candidate_authority_sha256':args.manifest_sha,
  'reconstruction_audit_sha256':args.audit_sha,'overlap_audit_sha256':args.overlap_sha,
  'outputs':{k:desc(v) for k,v in outputs.items()}}
 tmp=args.output/'manifest.json.partial';tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,args.output/'manifest.json')
 print('HYBRID_V2_SFT_RENDER',records,offset,targets,sha(args.output/'manifest.json'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--manifest-sha',required=True)
 p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha',required=True)
 p.add_argument('--overlap',type=Path,required=True);p.add_argument('--overlap-sha',required=True)
 p.add_argument('--output',type=Path,required=True);render(p.parse_args())
if __name__=='__main__':main()
