#!/usr/bin/env python3
"""Independent byte-level audit of the selected Pi-native authority."""
import argparse,hashlib,json,struct
from pathlib import Path
INDEX=struct.Struct('<QQQB7x');EXCLUDED={'preservation'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(args):
 source=args.source/'candidate-authority';selected=args.selected/'candidate-authority';manifest=json.loads((selected/'manifest.json').read_text());audit=json.loads((args.source/'audit.json').read_text())
 if sha(source/'manifest.json')!=args.source_authority_sha or sha(args.source/'audit.json')!=args.source_audit_sha or sha(selected/'manifest.json')!=args.selected_authority_sha or audit['authority_sha256']!=args.source_authority_sha:raise ValueError('authority identity')
 for descriptor in manifest['outputs'].values():
  path=selected/descriptor['path']
  if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError('selected output identity')
 source_rows=[json.loads(x) for x in (source/'records.jsonl').read_text().splitlines()];selected_rows=[json.loads(x) for x in (selected/'records.jsonl').read_text().splitlines()];excluded=[json.loads(x) for x in (args.selected/'excluded.jsonl').read_text().splitlines()]
 source_index=(source/'records.idx').read_bytes();selected_index=(selected/'records.idx').read_bytes();source_tokens=(source/'tokens.uint32.bin').read_bytes();selected_tokens=(selected/'tokens.uint32.bin').read_bytes();source_mask=(source/'assistant_mask.uint8.bin').read_bytes();selected_mask=(selected/'assistant_mask.uint8.bin').read_bytes();expected_rows=[];expected_tokens=bytearray();expected_mask=bytearray();expected_index=bytearray();expected_excluded=[];offset=targets=0
 for i,row in enumerate(source_rows):
  start,n,want,split=INDEX.unpack_from(source_index,i*INDEX.size)
  if split:raise ValueError('source split')
  if row['family'] in EXCLUDED:expected_excluded.append({'id':row['id'],'family':row['family'],'reason':'frozen-family-exclusion'});continue
  expected_rows.append(row);expected_tokens.extend(source_tokens[4*start:4*(start+n)]);expected_mask.extend(source_mask[start:start+n]);expected_index.extend(INDEX.pack(offset,n,want,0));offset+=n;targets+=want
 if selected_rows!=expected_rows or excluded!=expected_excluded or selected_tokens!=expected_tokens or selected_mask!=expected_mask or selected_index!=expected_index:raise ValueError('selection reconstruction')
 if manifest['records']!=len(expected_rows) or manifest['tokens']!=offset or manifest['assistant_target_tokens']!=targets or manifest['excluded_records']!=len(expected_excluded) or manifest['training_eligible'] or manifest['packing_authorized'] or manifest['optimizer_updates_authorized']:raise ValueError('selection manifest')
 receipt={'schema':'emender-e97-pi-native-selection-audit-v1','status':'qualified-selection-not-admitted','records':len(expected_rows),'excluded_records':len(expected_excluded),'tokens':offset,'assistant_target_tokens':targets,'source_authority_sha256':args.source_authority_sha,'source_audit_sha256':args.source_audit_sha,'selected_authority_sha256':args.selected_authority_sha,'checker_sha256':sha(__file__),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0};args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_SELECTION_AUDIT',len(expected_rows),len(expected_excluded),sha(args.output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--source-authority-sha',required=True);p.add_argument('--source-audit-sha',required=True);p.add_argument('--selected',type=Path,required=True);p.add_argument('--selected-authority-sha',required=True);p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
if __name__=='__main__':main()
