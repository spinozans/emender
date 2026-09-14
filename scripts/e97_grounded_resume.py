"""Reuse only the independently audited complete prefix of the timed-out build."""
import json
from pathlib import Path
from scripts.eval_e97_native_execution import sha

AUDIT_SHA='886cf51fc0571ef86415c19a708c6b86676ea6b3c2b61f65fadd9e94540ec71e'
JOURNALS={'authored-candidates-private.jsonl':'ba5a5bf7de11f8edee3c64f431135d1ca54448bfe9712067ee139480bc2653c1',
          'authored-verification-private.jsonl':'4e6c809562f5afa0ae05fdb32de0383c0d897caa340c28559f357dd9a2e6548d'}


def load_reuse(root,train,panel,enc):
    from scripts.audit_e97_grounded_expansion import check_teacher,reconstruct,TRAIN_SHA,EVAL_SHA
    root=Path(root)
    bindings={root/'partial-result-audit.json':AUDIT_SHA,root/'training-cases-private.json':TRAIN_SHA,root/'fresh-evaluation-cases.json':EVAL_SHA}
    bindings.update({root/name:digest for name,digest in JOURNALS.items()})
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('reused evidence identity')
    report=json.loads((root/'partial-result-audit.json').read_text())
    if report['verified_records_retained'] is not True or report['completed']!=1333 or report['remaining']!=715 or report['training_eligible'] is not False:raise ValueError('partial verdict')
    if sha(root/'failure.json')!=report['failure_sha256']:raise ValueError('original failure identity')
    planned=[c for world in (0,1) for c in train if c['variant']==world]
    records=[json.loads(line) for line in (root/'authored-candidates-private.jsonl').read_text().splitlines()]
    receipts=[json.loads(line) for line in (root/'authored-verification-private.jsonl').read_text().splitlines()]
    if len(records)!=1333 or len(receipts)!=1333 or report['completed_ids']!=[c['id'] for c in planned[:1333]] or report['remaining_ids']!=[c['id'] for c in planned[1333:]]:raise ValueError('reuse coverage')
    for case,row,receipt in zip(planned,records,receipts):
        if case['id']!=row['id'] or row['id']!=receipt['id']:raise ValueError('reused prefix order')
        check_teacher(case,row['candidate'],receipt,panel);reconstruct(row['candidate'],panel['tools'],enc)
    sandboxes=[]
    for world in (0,1):
        folder=root/f'authored-world-{world}'
        files={str(folder/name):sha(folder/name) for name in ('container-before.json','container-terminal.json','cleanup.json')}
        cleanup=json.loads((folder/'cleanup.json').read_text())
        if cleanup!=report['cleanups'][world]:raise ValueError('reused cleanup binding')
        sandboxes.append(dict(path=str(folder),files=files))
    provenance=dict(schema='emender-e97-grounded-completion-v1',source=str(root),partial_audit_sha256=AUDIT_SHA,
        reused_records=1333,new_records_required=715,additional_data_seconds=1800,learning_updates=32,
        journal_sha256=JOURNALS,reused_sandboxes=sandboxes,original_failure_sha256=report['failure_sha256'],
        authorization='Operator: do it all; one additional bounded completion pass, unchanged examples/evaluation/learning budget')
    return records,provenance


def verify_reuse(provenance,combined=None):
    root=Path(provenance['source'])
    if provenance['partial_audit_sha256']!=AUDIT_SHA or provenance['journal_sha256']!=JOURNALS or sha(root/'partial-result-audit.json')!=AUDIT_SHA:raise ValueError('reuse authority')
    for name,digest in JOURNALS.items():
        if sha(root/name)!=digest:raise ValueError('reused journals changed')
        if combined is not None:
            original=(root/name).read_bytes()
            with (Path(combined)/name).open('rb') as f:
                if f.read(len(original))!=original:raise ValueError('reused journal prefix changed')
    for sandbox in provenance['reused_sandboxes']:
        for path,digest in sandbox['files'].items():
            if sha(path)!=digest:raise ValueError('reused sandbox evidence changed')
