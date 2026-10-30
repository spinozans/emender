"""100 isolated reproduction attempts, each with a clean CUDA context."""
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
source=Path('experiments/expressivity_tasks/x40_emender_lm/receipts/part12_checkpoint_managed/training/runs/depth.ckpt')
blob=source.read_bytes();checkpoint_sha256=hashlib.sha256(blob).hexdigest()
with tempfile.NamedTemporaryFile(prefix='x40_part22_depth_',suffix='.ckpt',delete=False) as f:
    f.write(blob);fixture=Path(f.name)
del blob
logs=Path(tempfile.mkdtemp(prefix='nine_x_repeats.',dir=root))
rows=[];reference=None;reference_input=None
try:
    for repeat in range(100):
        command=[sys.executable,str(root/'nine_x_once.py')]
        try:
            run=subprocess.run(command,env=dict(os.environ,PART22_DEPTH_CHECKPOINT=str(fixture)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
            raw=run.stdout;rc=run.returncode
        except subprocess.TimeoutExpired as exc:
            raw=(exc.stdout or b'')+b'\nREPEAT TIMEOUT 300 seconds\n';rc=124
        path=logs/f'{repeat:03}.txt';path.write_bytes(raw)
        records=[]
        for line in raw.decode(errors='replace').splitlines():
            try:record=json.loads(line)
            except ValueError:continue
            if isinstance(record,dict):records.append(record)
        passed=rc==0 and any(r.get('checkpoint_sha256')==checkpoint_sha256 and r.get('status')=='passed' for r in records)
        input_digest=next((r['input_sha256'] for r in reversed(records) if 'input_sha256' in r),None)
        if input_digest is not None:
            if reference_input is None:reference_input=input_digest
            passed=passed and input_digest==reference_input
        digest=next((r['logits_sha256'] for r in records if 'logits_sha256' in r),None)
        if passed:
            if reference is None:reference=digest
            passed=digest==reference
        row=dict(repeat=repeat,status='passed' if passed else 'failed',exit_code=rc,logits_sha256=digest,input_sha256=input_digest,console_sha256=hashlib.sha256(raw).hexdigest(),console=str(path.relative_to(root)))
        rows.append(row);print(json.dumps(row),flush=True)
finally:
    fixture.unlink(missing_ok=True)
    result=dict(status='passed' if len(rows)==100 and all(r['status']=='passed' for r in rows) else 'failed',repeats=len(rows),checkpoint_sha256=checkpoint_sha256,rows=rows,scope='100 independent full depth.Probe 9x reproduction attempts, batch 80:88, clean process/CUDA context per attempt, immutable external checkpoint copy, successful logits compared by byte hashes')
    (root/'nine_x.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
