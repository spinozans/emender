import hashlib,json,struct
from pathlib import Path
from types import SimpleNamespace
from scripts.select_e97_pi_native_curriculum import INDEX,select

def dump(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def test_selection_excludes_frozen_family_and_reindexes(tmp_path):
 source=tmp_path/'source';authority=source/'candidate-authority';authority.mkdir(parents=True);plan_path=tmp_path/'plan.json';dump(plan_path,{'training_eligible':False,'packing_authorized':False,'optimizer_updates':0});plan_sha=sha(plan_path)
 rows=[{'id':'a','family':'preservation','targets':1,'sequence_sha256':'a'*64,'repository_discovery':False},{'id':'b','family':'copy','targets':2,'sequence_sha256':'b'*64,'repository_discovery':False}]
 (authority/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(authority/'tokens.uint32.bin').write_bytes(struct.pack('<3I',1,2,3));(authority/'assistant_mask.uint8.bin').write_bytes(b'\1\1\1');(authority/'records.idx').write_bytes(INDEX.pack(0,1,1,0)+INDEX.pack(1,2,2,0));dump(authority/'manifest.json',{'plan_sha256':plan_sha});authority_sha=sha(authority/'manifest.json');dump(source/'audit.json',{'status':'qualified-candidates-not-admitted','authority_sha256':authority_sha,'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0});audit_sha=sha(source/'audit.json')
 output=tmp_path/'out';select(SimpleNamespace(source_root=source,plan=plan_path,plan_sha=plan_sha,audit_sha=audit_sha,minimum_records=1,output=output))
 manifest=json.loads((output/'candidate-authority/manifest.json').read_text());assert manifest['records']==1 and manifest['excluded_records']==1
 assert (output/'candidate-authority/tokens.uint32.bin').read_bytes()==struct.pack('<2I',2,3)
 assert (output/'candidate-authority/assistant_mask.uint8.bin').read_bytes()==b'\1\1'
 assert INDEX.unpack((output/'candidate-authority/records.idx').read_bytes())==(0,2,2,0)
 assert json.loads((output/'excluded.jsonl').read_text())['id']=='a'
