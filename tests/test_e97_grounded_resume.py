import json
import pytest
from scripts.eval_e97_native_execution import sha
import scripts.e97_grounded_resume as resume
import scripts.audit_e97_grounded_expansion as audit


def fixture(tmp_path,monkeypatch):
    def save(path,value):path.write_text(json.dumps(value));return sha(path)
    train=[dict(id=str(i),variant=int(i>=1024)) for i in range(2048)]
    monkeypatch.setattr(audit,'TRAIN_SHA',save(tmp_path/'training-cases-private.json',train))
    monkeypatch.setattr(audit,'EVAL_SHA',save(tmp_path/'fresh-evaluation-cases.json',[]))
    records=[dict(id=str(i),candidate={'example':i}) for i in range(1333)]
    receipts=[dict(id=str(i)) for i in range(1333)]
    journals={}
    for name,rows in (('authored-candidates-private.jsonl',records),('authored-verification-private.jsonl',receipts)):
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in rows));journals[name]=sha(tmp_path/name)
    monkeypatch.setattr(resume,'JOURNALS',journals)
    cleanups=[]
    for world in (0,1):
        folder=tmp_path/f'authored-world-{world}';folder.mkdir()
        cleanup=dict(container_id=str(world),removed=True);cleanups.append(cleanup)
        for name in ('container-before.json','container-terminal.json','cleanup.json'):save(folder/name,cleanup)
    report=dict(verified_records_retained=True,completed=1333,remaining=715,training_eligible=False,
        failure_sha256=save(tmp_path/'failure.json',{'exit':124}),completed_ids=[str(i) for i in range(1333)],remaining_ids=[str(i) for i in range(1333,2048)],cleanups=cleanups)
    monkeypatch.setattr(resume,'AUDIT_SHA',save(tmp_path/'partial-result-audit.json',report))
    def check(case,record,receipt,panel):assert int(case['id'])==record['example']==int(receipt['id'])
    monkeypatch.setattr(audit,'check_teacher',check)
    monkeypatch.setattr(audit,'reconstruct',lambda *args:(b'',b''))
    return train


def test_only_audited_complete_prefix_reused(tmp_path,monkeypatch):
    train=fixture(tmp_path,monkeypatch)
    rows,provenance=resume.load_reuse(tmp_path,train,{'tools':[]},None)
    assert len(rows)==1333 and provenance['new_records_required']==715
    output=tmp_path/'new';output.mkdir()
    for name in resume.JOURNALS:(output/name).write_bytes((tmp_path/name).read_bytes()+b'new-record\n')
    resume.verify_reuse(provenance,output)
    (output/'authored-candidates-private.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='prefix changed'):resume.verify_reuse(provenance,output)


def test_modified_original_journal_fails_closed(tmp_path,monkeypatch):
    train=fixture(tmp_path,monkeypatch)
    (tmp_path/'authored-candidates-private.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='identity'):resume.load_reuse(tmp_path,train,{'tools':[]},None)


def test_reordered_completed_cases_are_not_reused(tmp_path,monkeypatch):
    train=fixture(tmp_path,monkeypatch);train[0],train[1]=train[1],train[0]
    with pytest.raises(ValueError,match='coverage'):resume.load_reuse(tmp_path,train,{'tools':[]},None)
