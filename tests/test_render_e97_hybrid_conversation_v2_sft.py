import json
import struct
import hashlib
from pathlib import Path
import numpy as np
import pytest
from scripts.render_e97_hybrid_conversation_v2_sft import render

RECORD_INDEX=struct.Struct('<QQQB7x')

def write_candidate(root,records):
 root=Path(root);(root/'candidate-authority').mkdir(parents=True)
 authority=root/'candidate-authority'
 tokens=np.concatenate([np.asarray(t,dtype='<u4') for t in records])
 mask=np.concatenate([np.ones(len(t),dtype=np.uint8) for t in records])
 offsets=np.cumsum([0]+[len(t) for t in records])[:-1]
 index=b''.join(RECORD_INDEX.pack(o,len(t),len(t),0) for o,t in zip(offsets,records))
 metadata=''.join(json.dumps({'id':f'r{i}','family':'hybrid-date-question'},sort_keys=True)+'\n' for i in range(len(records)))
 outputs={}
 for name,data in (('tokens',tokens.tobytes()),('mask',mask.tobytes()),('index',index),('metadata',metadata.encode())):
  path=authority/{'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}[name]
  path.write_bytes(data);outputs[name]={'path':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 manifest={'schema':'emender-e97-pi-native-candidate-authority-v1','status':'verified-candidate-not-admitted',
  'training_eligible':False,'plan_sha256':'plansha','counts':{'records':len(records),'tokens':len(tokens),'assistant_target_tokens':int(sum(mask))},
  'outputs':outputs}
 (authority/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
 return authority

def write_receipt(path,schema,status,records,plan_sha):
 receipt={'schema':schema,'status':status,'training_eligible':False,'records':records,'plan_sha256':plan_sha}
 Path(path).write_text(json.dumps(receipt,sort_keys=True)+'\n')

def args_for(tmp_path,records=([11,12,13,14,15,16,17,18],[21,22,23,24,25,26,27,28])):
 candidate=write_candidate(tmp_path/'collect',records)
 audit=tmp_path/'audit.json';overlap=tmp_path/'overlap-audit.json'
 write_receipt(audit,'emender-e97-pi-native-curriculum-audit-v1','qualified-candidates-not-admitted',len(records),'plansha')
 write_receipt(overlap,'emender-e97-hybrid-conversation-overlap-audit-v1','pass',len(records),'plansha')
 import argparse
 return candidate,audit,overlap,argparse.Namespace(candidate=candidate,manifest_sha=hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest(),
  audit=audit,audit_sha=hashlib.sha256(audit.read_bytes()).hexdigest(),
  overlap=overlap,overlap_sha=hashlib.sha256(overlap.read_bytes()).hexdigest(),
  output=tmp_path/'render')

def test_render_copies_slices_byte_exact_and_tags_source(tmp_path):
 candidate,audit,overlap,args=args_for(tmp_path)
 render(args)
 manifest=json.loads((args.output/'manifest.json').read_text())
 assert manifest['schema']=='emender-e97-tulu3-masked-sft-v1' and manifest['status']=='complete'
 assert manifest['training_eligible'] is False and manifest['counts']['records']==2
 assert manifest['counts']['assistant_target_tokens']==16
 assert manifest['source_target_totals']=={'hybrid-conversation-rehearsal-v2':16}
 assert (args.output/'tokens.uint32.bin').read_bytes()==(candidate/'tokens.uint32.bin').read_bytes()
 assert (args.output/'assistant_mask.uint8.bin').read_bytes()==(candidate/'assistant_mask.uint8.bin').read_bytes()
 rows=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
 assert [r['id'] for r in rows]==['r0','r1'] and all(r['source']=='hybrid-conversation-rehearsal-v2' for r in rows)

def test_render_fails_closed_on_receipt_record_mismatch(tmp_path):
 candidate,audit,overlap,args=args_for(tmp_path)
 receipt=json.loads(audit.read_text());receipt['records']=3
 audit.write_text(json.dumps(receipt,sort_keys=True)+'\n')
 args.audit_sha=hashlib.sha256(audit.read_bytes()).hexdigest()
 with pytest.raises(ValueError,match='receipt binding'):render(args)

def test_render_fails_closed_on_wrong_candidate_sha(tmp_path):
 candidate,audit,overlap,args=args_for(tmp_path)
 args.manifest_sha='0'*64
 with pytest.raises(ValueError,match='candidate manifest identity'):render(args)
