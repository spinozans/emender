import json,struct,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_e97_pi_native_repair_training import (AUTHORITY_SCHEMA,COHORTS,INDEX,NATIVE_SCHEMA,
 interleave,prepare)

def sha(path):import hashlib;return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_tulu3(root,rows,*,eligible,records_spec,schema=AUTHORITY_SCHEMA,status='complete'):
 root.mkdir(parents=True)
 tokens=b''.join(t for t,_,_ in records_spec);mask=b''.join(m for _,m,_ in records_spec)
 offset=0;idx=b''
 for t,m,_ in records_spec:
  idx+=INDEX.pack(offset,len(m),sum(m),0);offset+=len(m)
 (root/'tokens.uint32.bin').write_bytes(tokens);(root/'assistant_mask.uint8.bin').write_bytes(mask)
 (root/'records.idx').write_bytes(idx)
 (root/'records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 outputs={k:{'path':v.name,'bytes':v.stat().st_size,'sha256':sha(v)} for k,v in
  {'tokens':root/'tokens.uint32.bin','mask':root/'assistant_mask.uint8.bin','index':root/'records.idx','metadata':root/'records.jsonl'}.items()}
 manifest={'schema':schema,'status':status,'tokenizer':'p50k_base','training_eligible':eligible,'counts':{'records':len(rows)},'outputs':outputs}
 (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
 return sha(root/'manifest.json')

def write_native(root,rows,records_spec,keys):
 root.mkdir(parents=True)
 tokens=b''.join(t for t,_,_ in records_spec);mask=b''.join(m for _,m,_ in records_spec)
 offset=0;idx=b''
 for t,m,_ in records_spec:
  idx+=INDEX.pack(offset,len(m),sum(m),0);offset+=len(m)
 (root/'tokens.bin').write_bytes(tokens);(root/'loss_mask.bin').write_bytes(mask)
 (root/'records.idx').write_bytes(idx)
 (root/'records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 (root/'exclusions.jsonl').write_text('');(root/'source_messages.jsonl').write_text('')
 outputs={k:{'path':v.name,'bytes':v.stat().st_size,'sha256':sha(v)} for k,v in
  {'tokens.bin':root/'tokens.bin','loss_mask.bin':root/'loss_mask.bin','records.idx':root/'records.idx',
   'records.jsonl':root/'records.jsonl','exclusions.jsonl':root/'exclusions.jsonl','source_messages.jsonl':root/'source_messages.jsonl'}.items()}
 manifest={'schema':NATIVE_SCHEMA,'training_eligible':False,'outputs':outputs}
 (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
 return sha(root/'manifest.json')

def test_prepare_interleaves_and_flags(tmp_path):
 full=tmp_path/'fulltraj';sel=tmp_path/'selected';reh=tmp_path/'rehearsal';out=tmp_path/'repair'
 spec_native=[(bytes([1])*(4*40),bytes([1])*8+b'\x00'*32,{'record_index':i,'problem_key':f'pk{i}','split':0}) for i in range(6)]
 nsha=write_native(full,[{'record_index':i,'problem_key':f'pk{i}','split':0} for i in range(6)],spec_native,['pk%d'%i for i in range(6)])
 spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 spec_reh=[(bytes([3])*40,bytes([1])*10,{'id':f'reh{i}'}) for i in range(6)]
 rsha=write_tulu3(reh,[{'id':f'reh{i}'} for i in range(6)],eligible=True,records_spec=spec_reh)
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();a.fulltraj=full;a.fulltraj_sha=nsha;a.fulltraj_budget_targets=60;a.fulltraj_seed=7
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=reh;a.rehearsal_sha=rsha
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=out
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None
 prepare(a)
 manifest=json.loads((out/'manifest.json').read_text())
 assert manifest['training_eligible'] is False and manifest['packing_authorized'] is False
 assert manifest['optimizer_updates_authorized']==0
 assert manifest['source_record_counts']=={COHORTS[0]:6,COHORTS[1]:6,COHORTS[2]:6}
 rows=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
 sources=[r['source'] for r in rows]
 # with equal token shares the weighted-fair merge must cycle through cohorts
 assert sources[:3]==list(COHORTS)
 # every later window of 3 contains all cohorts (no cohort starves early)
 for i in range(0,len(sources)-2,3):
  assert set(sources[i:i+3])==set(COHORTS)
 # index accounting matches payloads
 idx=(out/'records.idx').read_bytes();assert len(idx)==INDEX.size*18
 tok=(out/'tokens.uint32.bin').read_bytes();msk=(out/'assistant_mask.uint8.bin').read_bytes()
 total=0
 for i in range(18):
  o,n,want,split=INDEX.unpack_from(idx,i*INDEX.size)
  assert sum(msk[o:o+n])==want and len(tok[4*o:4*(o+n)])==4*n
  total+=n
 assert total==manifest['counts']['tokens']
 assert manifest['openhands_rehearsal']['records']==6 and manifest['openhands_rehearsal']['distinct_problem_keys']==6

def test_interleave_rejects_empty():
 try:interleave([([], 'a'),([], 'b')])
 except ValueError:return
 raise AssertionError('empty interleave accepted')
