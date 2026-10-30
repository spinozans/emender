"""Cluster retained reports by source allocation, never by individual bad lane."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
raw = gzip.decompress((root.parent/'part20/reduced_clone.txt.gz').read_bytes())
groups = {}
for block in raw.decode().split('========= Host API memory access error at host access to ')[1:]:
    m = re.match(r'(0x[0-9a-f]+) of size (\d+) bytes', block)
    assert m, block[:150]
    base, size = m.group(1), int(m.group(2))
    site = sorted(set(re.findall(r'Host Frame: .*? in backward.py:(\d+)', block)))
    key = (base, size, tuple(site))
    entry = groups.setdefault(key, dict(source_address=base, bytes=size, frames=site, reports=0, offsets=[]))
    entry['reports'] += 1
    bad = re.search(r'Uninitialized access at (0x[0-9a-f]+)', block)
    entry['offsets'].append(int(bad.group(1),16)-int(base,16))
result = dict(scope='100 retained reports only; truncated timed-out diagnostic, not exhaustive',
              raw_sha256=hashlib.sha256(raw).hexdigest(), reports=sum(x['reports'] for x in groups.values()),
              allocations=list(groups.values()))
assert result['reports'] == 100
(root/'clone_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
