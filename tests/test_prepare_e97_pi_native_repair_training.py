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
 manifest={'schema':schema,'status':status,'tokenizer':'p50k_base','counts':{'records':len(rows)},'outputs':outputs}
 if eligible is not None:manifest['training_eligible']=eligible
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
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=full;a.fulltraj_sha=nsha;a.fulltraj_budget_targets=60;a.fulltraj_seed=7
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=reh;a.rehearsal_sha=rsha
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=out
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.translated_oh_source=None;a.translated_oh_sha=None;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=0;a.translated_oh_seed=0
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
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

def test_conversation_rehearsal_cohort_from_production_admitted_source(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import prepare,read_conversation_slice,COHORTS
 import importlib
 mod=importlib.import_module('scripts.prepare_e97_pi_native_repair_training')
 full=tmp_path/'full';rows=[{'record_index':i,'tokens':60,'targets':10,'problem_key':f'k{i}'} for i in range(6)]
 nsha=write_native(full,rows,[(bytes([i%256])*60,bytes([1])*10,{'record_index':i}) for i in range(6)],[f'k{i}' for i in range(6)])
 sel=tmp_path/'sel';spec_sel=[(bytes([1])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}));(sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 reh=tmp_path/'reh';spec_reh=[(bytes([3])*40,bytes([1])*10,{'id':f'reh{i}'}) for i in range(6)]
 rsha=write_tulu3(reh,[{'id':f'reh{i}'} for i in range(6)],eligible=True,records_spec=spec_reh)
 conv=tmp_path/'conv';spec_conv=[(bytes([5])*30,bytes([1])*10,{'id':f'c{i}'}) for i in range(6)]
 csha=write_tulu3(conv,[{'id':f'c{i}'} for i in range(6)],eligible=True,records_spec=spec_conv)
 # production-admitted source: drop the training_eligible key entirely
 m=json.loads((conv/'manifest.json').read_text());m.pop('training_eligible');(conv/'manifest.json').write_text(json.dumps(m,sort_keys=True)+'\n')
 csha=sha(conv/'manifest.json')
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=full;a.fulltraj_sha=nsha;a.fulltraj_budget_targets=60;a.fulltraj_seed=7
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=reh;a.rehearsal_sha=rsha
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'out-conv'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=conv;a.conversation_sha=csha;a.conversation_budget_targets=30;a.conversation_seed=11
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.translated_oh_source=None;a.translated_oh_sha=None;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=0;a.translated_oh_seed=0
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert 'conversation-rehearsal' in manifest['source_record_counts']
 assert manifest['conversation_rehearsal']['source_training_eligible'] is None
 assert manifest['conversation_rehearsal']['consumed_target_tokens']==30
 # explicitly non-admitted source must be rejected
 bad=tmp_path/'bad';spec_bad=[(bytes([5])*30,bytes([1])*10,{'id':f'b{i}'}) for i in range(6)]
 bsha=write_tulu3(bad,[{'id':f'b{i}'} for i in range(6)],eligible=False,records_spec=spec_bad)
 m=json.loads((bad/'manifest.json').read_text());m['training_eligible']=False;(bad/'manifest.json').write_text(json.dumps(m,sort_keys=True)+'\n')
 bsha=sha(bad/'manifest.json')
 try:read_conversation_slice(bad,bsha,11,30)
 except ValueError:pass
 else:raise AssertionError('non-admitted conversation source accepted')

def test_conversation_slice_admits_64k_boundary_segment_records(tmp_path):
 # The long-document anchor authorities segment >65,537-token documents at the
 # 64K pack boundary, so their segment-1 records are exactly CONTEXT_SIZE+1
 # tokens and must pass through (they fill one whole boundary-aware pack); one
 # token larger cannot be packed whole and must stay excluded.
 from scripts.prepare_e97_pi_native_repair_training import read_conversation_slice,CONTEXT_SIZE
 root=tmp_path/'anchor'
 n_seg=CONTEXT_SIZE+1;n_over=n_seg+1
 spec=[(bytes([7])*(4*n_seg),bytes([1])*n_seg,{'id':'seg1'}),
       (bytes([8])*(4*n_over),bytes([1])*n_over,{'id':'oversize'}),
       (bytes([9])*(4*40),bytes([1])*10,{'id':'small'})]
 asha=write_tulu3(root,[{'id':'seg1'},{'id':'oversize'},{'id':'small'}],eligible=True,records_spec=spec)
 chosen,consumed,eligibility,excluded=read_conversation_slice(root,asha,3,n_seg+40)
 ids=sorted(r['id'] for _,_,r in chosen)
 assert ids==['seg1','small'],ids
 assert consumed==n_seg+10
 assert excluded==0

def test_translated_oh_replaces_fulltraj_and_drops_rehearsal(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import prepare
 sel=tmp_path/'sel';spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 # translated OH candidate authority: instance dedup (two records share instance_id), budget-bounded
 toh=tmp_path/'toh'
 spec_toh=[(bytes([9])*(4*30),bytes([1])*8,{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0}),
           (bytes([9])*(4*20),bytes([1])*6,{'record_index':1,'instance_id':'inst-a','trajectory_id':'t1','split':0}),
           (bytes([9])*(4*10),bytes([1])*4,{'record_index':2,'instance_id':'inst-b','trajectory_id':'t2','split':0})]
 tsha=write_tulu3(toh,[{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0},
                        {'record_index':1,'instance_id':'inst-a','trajectory_id':'t1','split':0},
                        {'record_index':2,'instance_id':'inst-b','trajectory_id':'t2','split':0}],
                  eligible=False,records_spec=spec_toh)
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=toh;a.translated_oh_sha=tsha;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=18;a.translated_oh_seed=5
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=None;a.rehearsal_sha=None
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'out-toh'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert 'openhands-translated-rehearsal' in manifest['source_record_counts']
 assert 'representation-bridge-rehearsal' not in manifest['source_record_counts']
 oh=manifest['openhands_rehearsal']
 assert oh['cohort']=='openhands-translated-rehearsal' and oh['distinct_instance_ids']==2 and oh['consumed_target_tokens']==10
 assert oh['source_collection']=='e97-oh-pi-native-translation-v1 translated+replay-verified'
 rows=[json.loads(x) for x in (a.output/'records.jsonl').read_text().splitlines()]
 oh_rows=[r for r in rows if r['source']=='openhands-translated-rehearsal']
 # shortest-first dedup: inst-b (4 targets) always chosen; only one inst-a record fits the 18-target budget
 assert {r['repair_provenance']['instance_id'] for r in oh_rows}=={'inst-a','inst-b'}
 assert len(oh_rows)==2
 assert all('instance_id' in r['repair_provenance'] and 'trajectory_id' in r['repair_provenance'] for r in oh_rows)

def test_translated_oh_and_fulltraj_are_mutually_exclusive(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import prepare
 sel=tmp_path/'sel';spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=None;a.translated_oh_sha=None;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=0;a.translated_oh_seed=0
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=None;a.rehearsal_sha=None
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'out-x'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
 try:prepare(a)
 except ValueError as e:assert 'exactly one' in str(e)
 else:raise AssertionError('both sources absent accepted')

def test_extra3_reasoning_cohort_inclusion(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import prepare
 sel=tmp_path/'sel';spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 toh=tmp_path/'toh';spec_toh=[(bytes([9])*(4*10),bytes([1])*4,{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0})]
 tsha=write_tulu3(toh,[{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0}],eligible=False,records_spec=spec_toh)
 reas=tmp_path/'reas';spec_r=[(bytes([7])*(4*12),bytes([1])*5,{'id':f'r{i}'}) for i in range(4)]
 rsha=write_tulu3(reas,[{'id':f'r{i}'} for i in range(4)],eligible=False,records_spec=spec_r,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=toh;a.translated_oh_sha=tsha;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=8;a.translated_oh_seed=1
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=None;a.rehearsal_sha=None
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'out-r'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=reas;a.extra3_sha=rsha;a.extra3_cohort='reasoning-rehearsal'
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert manifest['source_record_counts']['reasoning-rehearsal']==4
 assert manifest['extra3_rehearsal']['cohort']=='reasoning-rehearsal' and manifest['extra3_rehearsal']['records']==4

def test_extra4_hybrid_cohort_inclusion(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import prepare
 sel=tmp_path/'sel';spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (sel/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (sel/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(sel/'selection-audit.json');oasha=sha(sel/'overlap-audit.json')
 toh=tmp_path/'toh';spec_toh=[(bytes([9])*(4*10),bytes([1])*4,{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0})]
 tsha=write_tulu3(toh,[{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0}],eligible=False,records_spec=spec_toh)
 hyb=tmp_path/'hyb';spec_h=[(bytes([6])*(4*12),bytes([1])*5,{'id':f'h{i}'}) for i in range(4)]
 hsha=write_tulu3(hyb,[{'id':f'h{i}'} for i in range(4)],eligible=False,records_spec=spec_h,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 class A:pass
 a=A();
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=toh;a.translated_oh_sha=tsha;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=8;a.translated_oh_seed=1
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=sel/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=sel/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=None;a.rehearsal_sha=None
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'out-h'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=hyb;a.extra4_sha=hsha;a.extra4_cohort='hybrid-conversation-rehearsal'
 a.cohort_spec=None
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert manifest['source_record_counts']['hybrid-conversation-rehearsal']==4
 assert manifest['extra4_rehearsal']['cohort']=='hybrid-conversation-rehearsal' and manifest['extra4_rehearsal']['records']==4
 rows=[json.loads(x) for x in (a.output/'records.jsonl').read_text().splitlines()]
 assert {r['source_record_id'] for r in rows if r['source']=='hybrid-conversation-rehearsal'}=={'h0','h1','h2','h3'}

def _base_args(tmp_path, sel_root, sel_sha, audits=True):
 class A:pass
 a=A()
 for k,v in dict(allow_no_oh_cohort=False,rehearsal_repeat_epochs=1,authored_repeat_epochs=1,
   extra3_repeat_epochs=1,extra4_repeat_epochs=1,conversation_exclusion_ids=None,purpose=None,
   rehearsal_max_records=0,authored_max_records=0,rehearsal_subsample_seed=0,authored_budget_targets=0,
   authored_seed=0,correction_source=None,correction_sha=None,loopbreak_source=None,loopbreak_sha=None,
   purpose_sizing_note=False).items(): setattr(a,k,v)
 a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=None;a.translated_oh_sha=None;a.translated_oh_cohort='openhands-translated-rehearsal'
 a.translated_oh_budget_targets=0;a.translated_oh_seed=0
 a.selected=sel_root;a.selected_sha=sel_sha
 if audits:
  (sel_root/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
  (sel_root/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
  a.selection_audit=sel_root/'selection-audit.json';a.selection_audit_sha=sha(sel_root/'selection-audit.json')
  a.overlap_audit=sel_root/'overlap-audit.json';a.overlap_audit_sha=sha(sel_root/'overlap-audit.json')
 a.rehearsal=None;a.rehearsal_sha=None
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 a.parent_checkpoint=parent;a.parent_sha=sha(parent)
 a.authored_source=None;a.pi_native_include_families=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=None;a.extra4_sha=None;a.extra4_cohort=None
 a.cohort_spec=None
 return a

def _selected_authority(tmp_path,n=4):
 sel=tmp_path/'sel'
 spec=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(n)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(n)],eligible=False,records_spec=spec,
  schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 return sel,ssha

def test_no_oh_cohort_mode_with_spec_cohorts(tmp_path):
 sel,ssha=_selected_authority(tmp_path)
 a=_base_args(tmp_path,sel,ssha)
 a.allow_no_oh_cohort=True
 raw=tmp_path/'rawoh'
 spec=[(bytes([7])*(4*20),bytes([1])*5,{'id':f'raw{i}','instance_id':f'inst{i}'}) for i in range(3)]
 rsha=write_tulu3(raw,[{'id':f'raw{i}','instance_id':f'inst{i}','split':0} for i in range(3)],
  eligible=None,records_spec=spec)
 (tmp_path/'spec.json').write_text(json.dumps({'cohort':'verified-raw-oh','root':str(raw),
  'sha256':rsha,'seed':5,'budget_targets':15}))
 a.cohort_spec=[tmp_path/'spec.json']
 a.output=tmp_path/'out-nooh'
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert manifest['openhands_rehearsal'] is None
 assert 'verified-raw-oh' in manifest['source_record_counts'] and 'pi-native-curriculum' in manifest['source_record_counts']
 assert 'openhands-execution-rehearsal' not in manifest['source_record_counts']
 assert manifest['spec_cohorts'][0]['records']==3
 rows=[json.loads(x) for x in (a.output/'records.jsonl').read_text().splitlines()]
 assert {r['source_record_id'] for r in rows if r['source']=='verified-raw-oh'}=={'raw0','raw1','raw2'}
 assert all('repair_provenance' not in r for r in rows)

def test_no_oh_cohort_mode_requires_flag(tmp_path):
 sel,ssha=_selected_authority(tmp_path)
 a=_base_args(tmp_path,sel,ssha)
 a.output=tmp_path/'out-reject'
 try:prepare(a)
 except ValueError:return
 raise AssertionError('no-OH mode accepted without --allow-no-oh-cohort')

def test_repeat_epochs_triplet_and_manifest_record(tmp_path):
 sel,ssha=_selected_authority(tmp_path)
 a=_base_args(tmp_path,sel,ssha);a.allow_no_oh_cohort=True
 reh=tmp_path/'reh'
 spec_reh=[(bytes([3])*40,bytes([1])*10,{'id':f'reh{i}'}) for i in range(4)]
 rsha=write_tulu3(reh,[{'id':f'reh{i}'} for i in range(4)],eligible=True,records_spec=spec_reh)
 a.rehearsal=reh;a.rehearsal_sha=rsha;a.rehearsal_repeat_epochs=3
 raw=tmp_path/'filler'
 spec=[(bytes([7])*(4*20),bytes([1])*5,{'id':f'f{i}'}) for i in range(2)]
 fsha=write_tulu3(raw,[{'id':f'f{i}','split':0} for i in range(2)],eligible=None,records_spec=spec)
 (tmp_path/'spec.json').write_text(json.dumps({'cohort':'filler','root':str(raw),'sha256':fsha,
  'seed':1,'budget_targets':10,'repeat_epochs':2}))
 a.cohort_spec=[tmp_path/'spec.json']
 a.output=tmp_path/'out-rep'
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert manifest['source_record_counts']['representation-bridge-rehearsal']==12
 assert manifest['rehearsal_rehearsal']['repeat_epochs']==3
 assert manifest['source_record_counts']['filler']==4
 assert manifest['spec_cohorts'][0]['repeat_epochs']==2
 assert manifest['repetition']['epochs']['representation-bridge-rehearsal']==3
 assert manifest['repetition']['epochs']['filler']==2
 assert manifest['source_target_totals']['representation-bridge-rehearsal']==120

def test_conversation_exclusion_ids(tmp_path):
 from scripts.prepare_e97_pi_native_repair_training import read_conversation_slice
 conv=tmp_path/'conv'
 spec=[(bytes([4])*(4*10),bytes([1])*4,{'identity':f'smoltalk2:f:row{i}','split':0}) for i in range(4)]
 csha=write_tulu3(conv,[{'identity':f'smoltalk2:f:row{i}','split':0} for i in range(4)],
  eligible=None,records_spec=spec)
 excl=tmp_path/'excl.txt';excl.write_text('smoltalk2:f:row1\nsmoltalk2:f:row3\n')
 chosen,consumed,elig,excluded=read_conversation_slice(conv,csha,seed=1,budget_targets=100,
  exclude_ids={'smoltalk2:f:row1','smoltalk2:f:row3'})
 assert excluded==2 and len(chosen)==2
 assert {row['identity'] for _t,_m,row in chosen}=={'smoltalk2:f:row0','smoltalk2:f:row2'}
 # and the manifest records freshness when wired through prepare()
 sel,ssha=_selected_authority(tmp_path)
 a=_base_args(tmp_path,sel,ssha);a.allow_no_oh_cohort=True
 a.conversation_source=conv;a.conversation_sha=csha;a.conversation_budget_targets=100;a.conversation_seed=1
 a.conversation_exclusion_ids=excl
 a.output=tmp_path/'out-excl'
 prepare(a)
 manifest=json.loads((a.output/'manifest.json').read_text())
 assert manifest['conversation_rehearsal']['freshness']['excluded_prior_draw_records']==2
