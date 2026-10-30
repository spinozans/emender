"""Fail-closed receipts: app success, exit status and terminal sanitizer required."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
repo=root.parents[4]
runs=[]
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or directory.name=='__pycache__':
        continue
    if not (directory/'console.txt').exists():
        continue
    assert (directory/'status.txt').exists(),f'Active or interrupted runner lacks terminal status: {directory}'
    raw=(directory/'console.txt').read_bytes()
    text=raw.decode(errors='replace')
    records=[]
    for line in text.splitlines():
        try: record=json.loads(line)
        except ValueError: continue
        if isinstance(record,dict):records.append(record)
    rc=int((directory/'status.txt').read_text())
    command=(directory/'command.txt').read_text().strip()
    sanitizer='compute-sanitizer' in command
    terminal=re.findall(r'ERROR SUMMARY: ([0-9,]+) errors',text)
    terminal_zero=bool(terminal) and terminal[-1]=='0'
    application=any(record.get('status')=='passed' and ('policies' in record or 'fixtures' in record or 'counted_clone_cases' in record) for record in records)
    pytest=bool(re.search(r'\b[0-9]+ passed\b',text))
    passed=rc==0 and (application or pytest or '\nPASSED\n' in text) and (not sanitizer or terminal_zero)
    runs.append(dict(name=directory.name,exit_code=rc,command=command,
                     result='passed' if passed else 'failed_or_incomplete',
                     terminal_sanitizer_errors=terminal,
                     application_pass=application or pytest or '\nPASSED\n' in text,
                     printed_host_errors=text.count('Host API memory access error'),
                     completed_backward_rows=sum('target' in r and r.get('status')=='passed' for r in records),
                     raw_sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw)))
paths=[f'experiments/expressivity_tasks/{family}/{name}.py'
       for family,names in {
           'x31_tape_cell_fused':['kernels','cell_fused','backward','all_field_journal'],
           'x36_tape_perf':['forward','backward','replay_setup','journal'],
           'x40_emender_lm':['kernels','forward','backward'],
       }.items() for name in names]
production={}
for name in paths:
    raw=(repo/name).read_bytes()
    baseline=subprocess.check_output(['git','show',f'4051a48f:{name}'],cwd=repo)
    assert raw==baseline,name
    production[name]=hashlib.sha256(raw).hexdigest()
assert not list(root.rglob('*.pt'))
summary=dict(status='failed_incomplete_no_production_promotion',base_revision='4051a48f',
             production_unchanged=True,production_sha256=production,runs=runs,
             race_inventory_groups_remaining=18,
             race_scope='Operator explicitly scopes the 18 race groups outside the uninit promotion blockers; no race fix or race-freedom claim.',
             residual_risks=[
                 'B64/T2048 full-harness initcheck fails with CUDA OOM in parity.py:35; terminal tool zero is not application success or a completed sweep.',
                 'B128/T2048 unchanged X31 int32 journal reservation preflight fails; no all-shape promotion.',
                 'The retained-gradient lifecycle matrix was deliberately stopped, not passed; complete X36/X40 imported/continued backward coverage remains required.',
                 'Graph/eager and bit-matched 2M curve gates remain unrun in part21.',
                 '18 race groups, 100-repeat 9x bin, depth probe, part-4 connection and final depth verdict remain pending.'],
             manual_notes='contact_supervisor was searched first and again by name/description; no callable intercom tool is available. Single-threaded attended work; only the approved .venv interpreter and broker allowlist 2/3/4, actual leases on card 2. No WG service, agents, Slurm, or .pt commits. Deliberate fixture stops and the corrected initial harness error are retained as failures, not passes.')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={}
for path in sorted(root.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.name!='manifest.json':
        manifest[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'runs':[(r['name'],r['exit_code'],r['result']) for r in runs]},indent=2))
