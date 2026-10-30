"""Read-only capacity/path and retained sanitizer inventory; no GPU launch."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

assert sys.executable == '/home/erikg/emender/.venv/bin/python'
from experiments.expressivity_tasks.x31_tape_cell_fused.all_field_journal import AllFieldJournal as Oracle
from experiments.expressivity_tasks.x36_tape_perf.journal import AllFieldJournal as Persistent
from experiments.expressivity_tasks.x40_emender_lm.model import POLICIES

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
rows = []
for batch in (32, 64, 128):
    for tokens in (512, 2047, 2048):
        for policy, cfg in enumerate(POLICIES):
            for name, cls in (('X31', Oracle), ('X40/X36', Persistent)):
                # Reproduce the implementation formulas even for rejected rows.
                s = cfg.slots
                active = min(s, 1 << (tokens - 1).bit_length())
                cap = (((6*tokens+8)*active + tokens*(64+128+64)
                        +(active+3*tokens)*(16*s.bit_length()+32))
                       if cls is Oracle else
                       tokens*(64+128+64+16*s.bit_length())
                       +tokens*(8*s.bit_length()+32))
                try:
                    actual = cls.capacity_for(batch, tokens, s, 64, 128, 0)
                    assert actual == cap
                    outcome = 'accepted'
                except ValueError as exc:
                    assert batch*cap > 2**31 or batch*s*192 > 2**31
                    outcome = str(exc)
                rows.append(dict(path=name, batch=batch, tokens=tokens, policy=policy,
                                 slots=s, capacity=cap, entries=batch*cap,
                                 reserved_journal_bytes=20*batch*cap,
                                 rebase_grid_y=(cap+255)//256,
                                 exceeds_cuda_grid_y=(cap+255)//256 > 65535,
                                 capacity_result=outcome))

surface = []
for path in sorted((OUT.parent/'part8/surface').glob('*pure*json')):
    data = json.loads(path.read_text())
    surface.append(dict(receipt=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        status=data['status'], scale=data['scale'], batch=data['batch'],
                        context=data['context'], envelope=data['envelope'],
                        measured_updates=data['measured_updates'],
                        source_matches={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==data['sources'][name]
                                        for name in ('forward.py', 'backward.py', 'part8_surface.py',
                                                     'part6_checkpoint.py', 'part6_journal.py', 'model.py')}))

log = OUT.parent/'part18/mask_certified_initcheck.txt.gz'
sites = collections.Counter()
sizes = collections.Counter()
raw = gzip.decompress(log.read_bytes())
text = raw.decode()
for match in re.finditer(r'Uninitialized __global__ memory read of size (\d+) bytes\n=========\s+at (.+)', text):
    sizes[match[1]] += 1
    sites[match[2]] += 1
assert sum(sites.values()) == 33070
report = dict(status='diagnostic_only_not_certified', capacity_rows=rows,
              part8_pure_surface=surface, uninit_inventory=dict(
                  source=str(log.relative_to(ROOT)), decompressed_sha256=hashlib.sha256(raw).hexdigest(),
                  accesses=sum(sites.values()), sites=dict(sites), sizes=dict(sizes),
                  complete=False, termination='part18 deliberate termination, exit 143'))
report['historical_source_drift'] = sorted({name for r in surface
    for name, matches in r['source_matches'].items() if not matches})
(OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(capacity_rows=len(rows), surface_receipts=len(surface), uninit_sites=dict(sites))))
