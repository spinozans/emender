"""Complete terminal repeat records after the original cohort's outer timeout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
root=Path(__file__).resolve().parent
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3','4')
original=root/'production_9x_100_repeats.cKJR4CSr'
assert int((original/'status.txt').read_text())==124
fixture=json.loads((root/'external_checkpoint.json').read_text())
checkpoint=Path(fixture['path']);expected=fixture['sha256']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
child=root/'nine_x_once.py';child_sha=hashlib.sha256(child.read_bytes()).hexdigest()
assert child_sha in (original/'sources.txt').read_text()
progress=root/'nine_x_resume_progress.json'
if progress.exists():rows=json.loads(progress.read_text())['rows']
else:
    rows=[]
    for line in (original/'console.txt').read_text().splitlines():
        try:record=json.loads(line)
        except ValueError:continue
        if isinstance(record,dict) and 'console_sha256' in record:rows.append(record)
assert [r['repeat'] for r in rows]==list(range(len(rows)))
for row in rows:assert hashlib.sha256((root/row['console']).read_bytes()).hexdigest()==row['console_sha256']
start=len(rows);reference=next(r['logits_sha256'] for r in rows if r['logits_sha256'])
reference_input=next(r['input_sha256'] for r in rows if r['input_sha256'])
logs=Path(tempfile.mkdtemp(prefix='nine_x_resume_logs.',dir=root))
for repeat in range(start,100):
    command=[sys.executable,str(child)]
    process=subprocess.Popen(command,env=dict(os.environ,PART22_DEPTH_CHECKPOINT=str(checkpoint)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:
        raw,_=process.communicate(timeout=60);rc=process.returncode
    except subprocess.TimeoutExpired:
        process.kill();raw,_=process.communicate();raw+=b'\nREPEAT TIMEOUT 60 seconds\n';rc=124
    path=logs/f'{repeat:03}.txt';path.write_bytes(raw)
    records=[]
    for line in raw.decode(errors='replace').splitlines():
        try:record=json.loads(line)
        except ValueError:continue
        if isinstance(record,dict):records.append(record)
    meta=next((r for r in reversed(records) if 'checkpoint_sha256' in r),{})
    digest=next((r['logits_sha256'] for r in records if 'logits_sha256' in r),None)
    passed=rc==0 and meta.get('status')=='passed' and meta.get('checkpoint_sha256')==expected and meta.get('input_sha256')==reference_input and digest==reference
    row=dict(repeat=repeat,status='passed' if passed else 'failed',exit_code=rc,logits_sha256=digest,input_sha256=meta.get('input_sha256'),console_sha256=hashlib.sha256(raw).hexdigest(),console=str(path.relative_to(root)))
    rows.append(row);print(json.dumps(row),flush=True)
    progress.write_text(json.dumps(dict(status='running',checkpoint_sha256=expected,rows=rows),indent=2)+'\n')
assert len(rows)==100
result=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',repeats=len(rows),checkpoint_sha256=expected,rows=rows,original_cohort_exit=124,original_cohort_terminal_records=95,resume_start=start,resume_timeout_seconds=60,child_sha256=child_sha,scope='100 terminal clean-process reproduction records. The initial 1200-second cohort bound interrupted an additional unrecorded attempt after 95 terminal records; it is retained separately and excluded from these 100. Resume uses the identical child source and immutable checkpoint, with a 60-second per-attempt bound; all failures and earlier records are preserved. Finite hashes compare against the first finite result, not an accuracy oracle.')
(root/'nine_x.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
