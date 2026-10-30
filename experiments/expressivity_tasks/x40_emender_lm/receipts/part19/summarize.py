"""Verify retained terminal evidence and hash every diagnostic artifact."""
import gzip
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
statuses = dict(line.split() for line in (OUT/'status.txt').read_text().splitlines())
expected = dict(masked='143', masked_retry='0', original='86', masked_record='0',
                x40_masked_record='0')
assert statuses == expected
results = {}
for name, code in expected.items():
    path = OUT / ('materialize_harness_failed.txt.gz' if name == 'masked'
                  else f'materialize_{name}.txt' + ('.gz' if name == 'original' else ''))
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    text = raw.decode()
    summaries = re.findall(r'ERROR SUMMARY: (\d+) errors', text)
    if code == '0':
        assert summaries == ['0'] and '"status": "passed"' in text
    elif name == 'original':
        assert summaries == ['15616']
    results[name] = dict(exit=int(code), log=path.name,
        decompressed_sha256=hashlib.sha256(raw).hexdigest(),
        error_summaries=summaries, accepted_production_fix=False)
audit = json.loads((OUT/'audit.json').read_text())
assert audit['uninit_inventory']['accesses'] == 33070
summary = dict(status='failed_incomplete_no_promotion', production_unchanged=True,
    results=results, capacity_cases=len(audit['capacity_rows']),
    isolated_masked_fixtures=45, all_shape_certification=False,
    remaining=['journal reference launch-axis repair and full B128/T2048 reference capacity solution',
               'Birth imported-state/continuation byte-differential lifecycle proof',
               'full harness sanitizer sweep at both shape families', 'all race inventory repairs',
               'bitwise parity, oracle suite, graph/eager, curve match',
               '9x bin, depth probe, part-4 connection and final verdict'],
    artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'summary.json'})
(OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(dict(status=summary['status'], isolated_masked_fixtures=45,
                      retained_runs=len(results))))
