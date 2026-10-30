"""Terminal evidence audit; no tool-only or aborted-run pass claims."""
import hashlib
import json
from pathlib import Path
import re
root=Path(__file__).resolve().parent
runs=[]
for directory in sorted(root.iterdir()):
    if not directory.is_dir() or not (directory/'console.txt').exists():continue
    assert (directory/'status.txt').exists(),f'Nonterminal run: {directory}'
    raw=(directory/'console.txt').read_bytes();text=raw.decode(errors='replace')
    assert (directory/'console.sha256').read_text().split()[0]==hashlib.sha256(raw).hexdigest(),directory
    command=(directory/'command.txt').read_text();rc=int((directory/'status.txt').read_text())
    records=[]
    for line in text.splitlines():
        try:record=json.loads(line)
        except ValueError:continue
        if isinstance(record,dict):records.append(record)
    terminal=re.findall(r'ERROR SUMMARY: ([0-9,]+) errors',text)
    races=re.findall(r'RACECHECK SUMMARY: (.*)',text)
    application=any(r.get('status')=='passed' and (any(k in r for k in ('policies','fixtures','counted_clone_cases','gradients')) or r.get('repeats')==100) for r in records) or bool(re.search(r'\b[0-9]+ passed\b',text)) or '\nPASSED\n' in text
    if directory.name.startswith('curve_'):
        arm='base' if directory.name.startswith('curve_base.') else 'promoted'
        curve=json.loads((root/f'curve_{arm}.json').read_text())
        application=curve['status']=='passed' and curve['tokens']>=2_000_000
    passed=rc==0 and application
    if directory.name.startswith('production_oracles.'):
        passed=passed and bool(re.search(r'\b109 passed\b',text))
    if directory.name.startswith('production_x36_oracles.'):
        passed=passed and bool(re.search(r'\b70 passed\b',text))
    if directory.name.startswith('production_lifecycle.'):
        passed=passed and any(r.get('status')=='passed' and r.get('fixtures')==32 for r in records)
    if directory.name.startswith('production_backward_lifecycle.'):
        passed=passed and any(r.get('status')=='passed' and r.get('fixtures')==48 and r.get('variants')==3 for r in records)
    if directory.name.startswith('production_graph.'):
        passed=passed and any(r.get('status')=='passed' and len(r.get('samples',[]))==2 for r in records)
    if directory.name.startswith(('production_B64_T512_initcheck.','production_B32_T2048_initcheck.')):
        batch,context=(64,512) if directory.name.startswith('production_B64') else (32,2048)
        passed=passed and any(r.get('status')=='passed' and r.get('pressure_gate') is True and len(r.get('policies',[]))==4 and all(p.get('status')=='passed' and p.get('batch')==batch and p.get('context')==context for p in r['policies']) for r in records if 'policies' in r)
    if '--tool\ninitcheck' in command:passed=passed and bool(terminal) and terminal[-1]=='0'
    runs.append(dict(name=directory.name,exit_code=rc,passed=passed,terminal_errors=terminal,racecheck_summary=races,raw_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),command=command))
state=(root/'disposition.txt').read_text().strip() if (root/'disposition.txt').exists() else 'not_installed'
summary=dict(status='passed' if state=='promoted_required_battery_passed' else 'failed_incomplete',disposition=state,runs=runs,residual_risks=['B128/T2048 remains excluded behind the unchanged X31 int32 capacity guard.','Certification covers B64/T512 and the authorized unpacked B32/T2048 production envelope; no B64/T2048 certification is claimed.','18 historical race groups remain unresolved; 9x/depth acceptance remains gated on race repair.'],race_scope='18 historical race groups remain unresolved, outside uninit promotion blockers by explicit part22 work order steps 3–4; no benign-race or race-free claim.',manual_notes='Earlier failed attempts were reverted in full: a 600-second X36 timeout, missing X40 wire arguments in the old retained fixture, and shared-beta version counters in its repaired wire harness. Raw logs, original harness sources and rollback receipts are retained. Single-threaded attended session; approved venv only; sequential GPU broker allowlist 2/3/4. No callable contact_supervisor tool was exposed. No WG service, Slurm, agents, or committed .pt artifacts.')
if state=='promoted_required_battery_passed':
    required=('production_B64_T512_initcheck','production_B32_T2048_initcheck','production_oracles','production_x36_oracles','production_counted_initcheck','production_snapshot_initcheck','production_lifecycle','production_backward_lifecycle','production_graph','curve_base','curve_promoted','production_9x_initcheck')
    for prefix in required:
        assert any(r['name'].startswith(prefix+'.') and r['passed'] for r in runs),prefix
    a=json.loads((root/'curve_base.json').read_text());b=json.loads((root/'curve_promoted.json').read_text())
    assert a['initial']==b['initial'] and a['rows']==b['rows'] and len(b['rows'])==64
    summary['curve_bitwise_tokens']=b['tokens']
summary['scope']='Required uninitialized-access promotion battery; post-promotion 9x and race failures remain separately recorded and blocked.'
if state=='promoted_required_battery_passed':
    audit=json.loads((root/'source_audit.json').read_text());assert audit['status']=='passed'
    nine=json.loads((root/'nine_x_analysis.json').read_text());assert nine['attempts']==100
    summary['source_audit_status']=audit['status']
    summary['post_promotion_9x']=nine
    inventory=json.loads((root/'race_inventory.json').read_text())
    summary['post_promotion_racecheck']=[{k:v for k,v in r.items() if k!='groups'} for r in inventory['runs']]
    summary['residual_risks'].append('Fresh racecheck is partial: 31 displayed race-error groups followed by the known device assertion; no completed 18-group sweep, ID mapping or closure is claimed.')
assert not list(root.rglob('*.pt'))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(disposition=state,runs=[(r['name'],r['exit_code'],r['passed']) for r in runs]),indent=2))
