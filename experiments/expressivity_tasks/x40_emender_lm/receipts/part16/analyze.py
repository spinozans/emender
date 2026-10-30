"""Summarize saved text only; does not access CUDA."""
import json,re,hashlib
from pathlib import Path
r=Path(__file__).parent
rows=[]
for p in sorted(r.glob('*.txt')):
    if p.name.startswith(('repeat_','layout_','isolated100')):
        text=p.read_text()
        scans=re.findall(r'^SCAN (\d+) ',text,re.M)
        records=[]
        for line in text.splitlines():
            if line.startswith('{'):
                try: records.append(json.loads(line))
                except ValueError: pass
        entered=[x['call'] for x in records if x.get('event')=='scan']
        passed=[x['call'] for x in records if x.get('event')=='scan_pass']
        bases=[x for x in records if x.get('event')=='kernel_inputs']
        rows.append(dict(file=p.name,entered=entered or list(map(int,scans)),passed=passed,
                         success='passed 80' in text or 'ISOLATED_100_PASSED' in text,
                         assertion='device-side assert' in text,exception=text.splitlines()[-1] if text else '',
                         address_records=len(bases),input_counts=[len(x['inputs']) for x in bases]))
(r/'run_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
# Comparisons use argument identity and alignments, not absolute ASLR addresses.
layouts={}
for p in sorted(r.glob('layout_*.txt')):
    records=[json.loads(s) for s in p.read_text().splitlines() if s.startswith('{')]
    kernels=[x for x in records if x.get('event')=='kernel_inputs']
    layouts[p.stem]={str(x['call']):{v['name']:{k:v[k] for k in ('base','mod256','mod4096','bytes')} for v in x['inputs']} for x in kernels}
(r/'layout_addresses.json').write_text(json.dumps(layouts,indent=2)+'\n')
# Compare each failing launch with the same launch in a passing run.
byfile={x['file']:x for x in rows}
comparisons=[]
for p in sorted(r.glob('layout_*.txt')):
    row=byfile[p.name]
    if row['success']: continue
    call=row['entered'][-1]
    failing=layouts[p.stem][str(call)]
    refname=next(n for n in sorted(layouts) if byfile[n+'.txt']['success'] and str(call) in layouts[n])
    passing=layouts[refname][str(call)]
    base_f=int(failing['K']['base'],16);base_p=int(passing['K']['base'],16)
    comparisons.append(dict(failing=p.name,passing=refname+'.txt',call=call,input_count=len(failing),
        all_256_aligned=all(x['mod256']==0 for x in failing.values()),
        different_mod4096=[dict(name=name,fail_base=x['base'],pass_base=passing[name]['base'],fail_mod4096=x['mod4096'],pass_mod4096=passing[name]['mod4096']) for name,x in failing.items() if x['mod4096']!=passing[name]['mod4096']],
        different_relative_to_K=[name for name,x in failing.items() if int(x['base'],16)-base_f != int(passing[name]['base'],16)-base_p],
        same_address_order=sorted(failing,key=lambda n:int(failing[n]['base'],16))==sorted(passing,key=lambda n:int(passing[n]['base'],16))))
(r/'address_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.iterdir()) if p.is_file() and p.name!='manifest.json'}
(r/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
