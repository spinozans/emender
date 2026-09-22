import json
import struct
import hashlib
from pathlib import Path
import numpy as np
import pytest
from scripts.audit_e97_hybrid_conversation_smoltalk2_shingles import audit

RECORD_INDEX=struct.Struct('<QQQB7x')

def write_authority(root,records):
 """records: list of (id, token list). Emits a minimal masked-SFT authority."""
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 tokens=np.concatenate([np.asarray(t,dtype='<u4') for _,t in records])
 mask=np.zeros(len(tokens),dtype=np.uint8)
 offsets=np.cumsum([0]+[len(t) for _,t in records])[:-1]
 index=b''.join(RECORD_INDEX.pack(o,len(t),0,0) for o,(_,t) in zip(offsets,records))
 metadata=''.join(json.dumps({'id':i},sort_keys=True)+'\n' for i,_ in records)
 outputs={}
 for name,data in (('tokens',tokens.tobytes()),('mask',mask.tobytes()),('index',index),('metadata',metadata.encode())):
  path=root/{'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}[name]
  path.write_bytes(data);outputs[name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 manifest={'schema':'emender-e97-tulu3-masked-sft-v1','records':len(records),'outputs':outputs,
  'training_eligible':False,'dataset_id':'HuggingFaceTB/smoltalk2','dataset_revision':'deadbeef'}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=1,sort_keys=True)+'\n')
 return root

def write_collect(root,records):
 """Emits a minimal collect dir: candidate-authority plus summary binding."""
 collect=Path(root);(collect/'candidate-authority').mkdir(parents=True)
 write_authority(collect/'candidate-authority',records)
 (collect/'summary.json').write_text(json.dumps({'records':len(records)},sort_keys=True)+'\n')
 return collect

def run(tmp_path,theirs,ours,expect_pass):
 collect=write_collect(tmp_path/'collect',ours)
 smoltalk=write_authority(tmp_path/'smoltalk2',theirs)
 out=tmp_path/'receipt.json'
 import argparse
 args=argparse.Namespace(root=collect,smoltalk2=smoltalk,output=out)
 if expect_pass:
  audit(args);receipt=json.loads(out.read_text())
  assert receipt['status']=='pass' and receipt['content_collisions']==0
  assert receipt['our_windows']==sum(len(t)-39 for _,t in ours if len(t)>=40)
  assert receipt['their_windows']==sum(len(t)-39 for _,t in theirs if len(t)>=40)
 else:
  with pytest.raises(ValueError,match='shingle collisions'):audit(args)

def test_no_shared_span_passes(tmp_path):
 ours=[('a',list(range(100,180))),('b',list(range(200,290)))]
 theirs=[('x',list(range(1000,1100))),('y',list(range(2000,2085)))]
 run(tmp_path,theirs,ours,True)

def test_shared_span_fails_closed(tmp_path):
 shared=list(range(500,560))
 ours=[('a',list(range(100,150))+shared)]
 theirs=[('x',list(range(1000,1070))+shared+list(range(2000,2060)))]
 run(tmp_path,theirs,ours,False)

def test_short_shared_span_below_width_passes(tmp_path):
 # spans shorter than the 40-token width are not shingle collisions
 short=list(range(900,930))
 ours=[('a',list(range(100,150))+short+list(range(300,360)))]
 theirs=[('x',short+list(range(1000,1090)))]
 run(tmp_path,theirs,ours,True)

def test_training_eligible_authority_rejected(tmp_path):
 ours=[('a',list(range(100,180)))]
 collect=write_collect(tmp_path/'collect',ours)
 write_authority(collect/'candidate-authority',ours)
 manifest=json.loads((collect/'candidate-authority/manifest.json').read_text())
 manifest['training_eligible']=True
 (collect/'candidate-authority/manifest.json').write_text(json.dumps(manifest,sort_keys=True))
 smoltalk=write_authority(tmp_path/'smoltalk2',[('x',list(range(1000,1090)))])
 import argparse
 with pytest.raises(ValueError,match='training eligibility'):
  audit(argparse.Namespace(root=collect,smoltalk2=smoltalk,output=tmp_path/'r.json'))
