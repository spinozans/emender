import hashlib,json,struct
from pathlib import Path
from types import SimpleNamespace
from scripts.prepare_e97_pi_native_curriculum_training import INDEX,prepare

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
def authority(root,row,training,schema):
 root.mkdir(parents=True);(root/'tokens.uint32.bin').write_bytes(struct.pack('<I',row['token']));(root/'assistant_mask.uint8.bin').write_bytes(b'\1');(root/'records.idx').write_bytes(INDEX.pack(0,1,1,0));(root/'records.jsonl').write_text(json.dumps(row['metadata'])+'\n');outputs={}
 for key,name in [('tokens','tokens.uint32.bin'),('mask','assistant_mask.uint8.bin'),('index','records.idx'),('metadata','records.jsonl')]:p=root/name;outputs[key]={'path':name,'bytes':p.stat().st_size,'sha256':sha(p)}
 manifest={'schema':schema,'status':'complete','training_eligible':training,'outputs':outputs}
 if not training:manifest.update(packing_authorized=False,optimizer_updates_authorized=0)
 dump(root/'manifest.json',manifest);return sha(root/'manifest.json')
def test_preparation_concatenates_selected_then_rehearsal_and_stays_nontraining(tmp_path):
 selected=tmp_path/'selected';selected.mkdir();sel_sha=authority(selected/'candidate-authority',{'token':1,'metadata':{'id':'new','family':'copy','targets':1}},False,'selected');dump(selected/'selection-audit.json',{'status':'qualified-selection-not-admitted'});dump(selected/'overlap-audit.json',{'status':'pass'});rehearsal=tmp_path/'rehearsal';reh_sha=authority(rehearsal,{'token':2,'metadata':{'id':'old','source':'retention','targets':1}},True,'emender-e97-tulu3-masked-sft-v1');checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'parent');out=tmp_path/'out'
 prepare(SimpleNamespace(selected=selected,selected_sha=sel_sha,selection_audit_sha=sha(selected/'selection-audit.json'),overlap_audit_sha=sha(selected/'overlap-audit.json'),rehearsal=rehearsal,rehearsal_sha=reh_sha,parent_checkpoint=checkpoint,parent_sha=sha(checkpoint),output=out))
 manifest=json.loads((out/'manifest.json').read_text());assert not manifest['training_eligible'] and manifest['counts']['records']==2 and manifest['source_target_totals']=={'pi-native-curriculum':1,'retention':1}
 assert (out/'tokens.uint32.bin').read_bytes()==struct.pack('<2I',1,2)
 assert [INDEX.unpack_from((out/'records.idx').read_bytes(),i*INDEX.size)[0] for i in range(2)]==[0,1]
