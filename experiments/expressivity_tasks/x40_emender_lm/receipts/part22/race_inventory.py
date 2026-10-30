"""Begin part23 inventory without subtracting or waiving any existing race."""
import hashlib
import json
from pathlib import Path
import re
root=Path(__file__).resolve().parent
paths=sorted(root.glob('production_9x_racecheck.*/console.txt'))
assert paths,'racecheck not run'
runs=[]
for path in paths:
    raw=path.read_bytes();text=raw.decode(errors='replace');groups=[]
    for line in text.splitlines():
        if 'Error: Race reported between' in line:
            groups.append(dict(id=f'P22-G{len(groups)+1:02}',lines=[line]))
        elif groups and line.startswith('=========     and '):groups[-1]['lines'].append(line)
    runs.append(dict(console=str(path.relative_to(root)),raw_sha256=hashlib.sha256(raw).hexdigest(),exit_code=int(path.with_name('status.txt').read_text()),groups=groups,displayed_groups=len(groups),terminal_racecheck_summary=re.findall(r'RACECHECK SUMMARY: (.*)',text),application_passed='\nPASSED\n' in text))
result=dict(historical_race_groups_remaining=18,scope='Explicit operator part22 work order steps 3–4: races remain defects and next work, outside uninitialized-access promotion blockers. No race is waived, repaired, benign, or closed by a displayed group count. P22 IDs describe this specialization only; no mapping/subtraction against historical R IDs is claimed.',runs=runs,next_work='Bind each access PC to the exact compiled specialization and shared-scratch owner/lifetime, then prove a value-neutral repair preserving all reduction trees before attempting promotion. Retain failed compiler/barrier experiments as failures.')
(root/'race_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(displayed_groups=[r['displayed_groups'] for r in runs],historical_unresolved=18)))
