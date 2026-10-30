"""Validate retained partial evidence without promoting it to acceptance."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

r=Path(__file__).resolve().parent
lifecycle=json.loads((r/'lifecycle.json').read_text())
assert lifecycle['status']=='passed'
for target in ('x31','x40'):
    assert len(lifecycle[target])==16
    assert all(not row['failures'] and all(v>0 for v in row['coverage'].values())
               for row in lifecycle[target])
assert 'ERROR SUMMARY: 0 errors' in (r/'isolated_initcheck.txt').read_text()
for name,row in json.loads((r/'candidate.json').read_text()).items():
    b=Path(name).read_bytes()
    assert hashlib.sha256(b).hexdigest()==row['before_sha256']
    assert b==subprocess.check_output(['git','show','HEAD:'+name])
assert not list(r.rglob('*.pt'))
p=r/'reduced_clone.txt'
b=p.read_bytes() if p.exists() else gzip.decompress((r/'reduced_clone.txt.gz').read_bytes())
s=b.decode()
assert 'Host API memory access error' in s
assert 'backward.py:237' in s and 'backward.py:245' in s
assert 'reduced_clone 124' in (r/'status.txt').read_text()
summary=dict(status='failed_incomplete_production_restored',
    lifecycle_fixtures=32,isolated_9x_initcheck_errors=0,
    certified_B64_T512_initcheck='failed; deliberately terminated; original raw log lost to edited-shell redirection; outer shell exit 2',
    reduced_clone=dict(status='failed_incomplete',exit_code=124,timeout_seconds=180,
        terminal_sanitizer_summary=False,print_limit=100,
        printed_host_api_errors=s.count('Host API memory access error'),
        printed_python_frames=sorted(set(re.findall(r'Host Frame: .+ in .+\.py:\d+',s))),
        raw_sha256=hashlib.sha256(b).hexdigest(),raw_bytes=len(b),
        scope='B1/T8 process-local candidate after production rollback; sampled reports, not certified-shape evidence'),
    production_restored=True,race_inventory_groups_remaining=18,
    not_run=['remaining all-shape parity','109 oracle tests','graph/eager','bit-matched 2M curve',
             'remaining certified-shape initcheck sweep','racecheck at both scales','certification adapter port',
             'race repairs','9x 100-repeat bin','hash-bound depth probe','pressure connection test','final depth verdict'],
    residual_risks=['Host API clones copy undefined owner/journal lanes; no repair or waiver accepted.',
                    'Original certified-shape raw log was overwritten; no complete error inventory.',
                    'Reduced diagnostic timed out; no terminal sanitizer summary or complete parity result.',
                    'All-shape production promotion battery remains incomplete; both masks reverted.',
                    '18 race groups unresolved.'])
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if p.exists():
    (r/'reduced_clone.txt.gz').write_bytes(gzip.compress(b,mtime=0))
    p.unlink()
(r/'manifest.json').write_text(json.dumps(dict(status=summary['status'],
    scope='Final retained receipt hashes, not per-run provenance',
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(r.iterdir()) if p.is_file() and p.name!='manifest.json'}),indent=2)+'\n')
print(json.dumps(summary,indent=2))
