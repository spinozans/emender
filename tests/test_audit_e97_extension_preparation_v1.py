import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_e97_extension_preparation_v1 import audit
from tests.test_prepare_e97_pi_native_repair_training import sha,write_tulu3

class A:pass

def build_prep(tmp_path):
 toh=tmp_path/'toh';spec_toh=[(bytes([9])*(4*10),bytes([1])*4,{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0})]
 tsha=write_tulu3(toh,[{'record_index':0,'instance_id':'inst-a','trajectory_id':'t0','split':0}],eligible=False,records_spec=spec_toh)
 selroot=tmp_path/'selected-root';sel=selroot/'candidate-authority'
 spec_sel=[(bytes([2])*40,bytes([1])*10,{'id':f'sel{i}'}) for i in range(6)]
 ssha=write_tulu3(sel,[{'id':f'sel{i}'} for i in range(6)],eligible=False,records_spec=spec_sel,schema='emender-e97-pi-native-selected-candidate-authority-v1',status='verified-selection-not-admitted')
 (selroot/'selection-audit.json').write_text(json.dumps({'status':'qualified-selection-not-admitted'}))
 (selroot/'overlap-audit.json').write_text(json.dumps({'status':'pass'}))
 sasha=sha(selroot/'selection-audit.json');oasha=sha(selroot/'overlap-audit.json')
 hyb=tmp_path/'hyb';spec_h=[(bytes([6])*(4*12),bytes([1])*5,{'id':f'h{i}'}) for i in range(4)]
 hsha=write_tulu3(hyb,[{'id':f'h{i}'} for i in range(4)],eligible=False,records_spec=spec_h,schema='emender-e97-pi-native-candidate-authority-v1',status='verified-candidate-not-admitted')
 parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
 from scripts.prepare_e97_pi_native_repair_training import prepare
 a=A();a.fulltraj=None;a.fulltraj_sha=None;a.fulltraj_budget_targets=0;a.fulltraj_seed=0
 a.translated_oh_source=toh;a.translated_oh_sha=tsha;a.translated_oh_cohort='openhands-translated-rehearsal';a.translated_oh_budget_targets=8;a.translated_oh_seed=1
 a.selected=sel;a.selected_sha=ssha;a.selection_audit=selroot/'selection-audit.json';a.selection_audit_sha=sasha
 a.overlap_audit=selroot/'overlap-audit.json';a.overlap_audit_sha=oasha
 a.rehearsal=None;a.rehearsal_sha=None
 a.parent_checkpoint=parent;a.parent_sha=sha(parent);a.output=tmp_path/'prep'
 a.authored_source=None;a.pi_native_include_families=None;a.correction_source=None;a.loopbreak_source=None
 a.conversation_source=None;a.conversation_sha=None;a.conversation_budget_targets=0;a.conversation_seed=0
 a.extra_source=None;a.extra_sha=None;a.extra_cohort=None
 a.extra2_source=None;a.extra2_sha=None;a.extra2_cohort=None
 a.extra3_source=None;a.extra3_sha=None;a.extra3_cohort=None
 a.extra4_source=hyb;a.extra4_sha=hsha;a.extra4_cohort='hybrid-conversation-rehearsal'
 a.cohort_spec=None
 prepare(a)
 return sha(a.output/'manifest.json'),a.output,selroot

def test_audit_passes_and_receipt_binds_cohorts(tmp_path):
 msha,out,selroot=build_prep(tmp_path)
 class B:pass
 b=B();b.preparation=out;b.manifest_sha256=msha;b.selected_root=selroot;b.output=tmp_path/'audit.json'
 audit(b)
 receipt=json.loads(b.output.read_text())
 assert receipt['status']=='qualified-preparation-not-admitted' and receipt['training_eligible'] is False
 assert receipt['records']==11 and receipt['tokens']==84 and receipt['assistant_target_tokens']==84
 table=receipt['cohort_table']
 assert set(table)=={'openhands-translated-rehearsal','pi-native-curriculum','hybrid-conversation-rehearsal'}
 assert table['hybrid-conversation-rehearsal']=={'records':4,'tokens':20,'assistant_target_tokens':20}
 assert receipt['parent_checkpoint_sha256']==sha(tmp_path/'parent.pt')

def test_audit_rejects_tampered_mask(tmp_path):
 msha,out,selroot=build_prep(tmp_path)
 class B:pass
 b=B();b.preparation=out;b.manifest_sha256=msha;b.selected_root=selroot;b.output=tmp_path/'audit.json'
 with open(out/'assistant_mask.uint8.bin','r+b') as f:
  f.seek(0);f.write(b'\x00')
 try:audit(b)
 except ValueError:pass
 else:raise AssertionError('tampered mask accepted')
