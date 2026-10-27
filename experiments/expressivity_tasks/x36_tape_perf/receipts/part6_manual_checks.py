"""Validate source, bounds and attested artifacts; no GPU or PyTorch."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
P=R.parent
repo=P.parents[2]
load=lambda name:json.loads((R/name).read_text())
source_paths=[P/n for n in ('adjoint_pool.py','backward.py','forward.py','replay_setup.py',
                             'continued_capacity.py','run_leased.sh')]
source_paths += sorted((P/'tests').glob('*.py'))
source_paths += sorted(R.glob('part6_*.py'))
for p in source_paths:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
assert not list(P.rglob('*.pt'))
assert (repo/'AGENTS.md').read_bytes()==(repo/'CLAUDE.md').read_bytes()
s=load('summary.json')
assert s['parity']==dict(tests=179,failures=0,errors=0,skipped=0)
assert s['status']=='failed'
points=0
for label in ('live17','live256','live1024','live2800','near_budget'):
    d=load(f'part6_hot_{label}.json')
    assert d['physical_gpu'] in ('0','1','5','6')
    assert d['python']=='/home/erikg/emender/.venv/bin/python'
    for r in d['results']:
        points+=1
        pool=r['pool'];n=r['batch']*r['heads'];live=r['live'];t=r['tokens'];slots=r['slots']
        assert pool['allocated_rows']==max(1,max(pool['union_counts']))==live
        assert all(x==live for x in pool['entry_counts'])
        assert not any(pool['allocation_events_per_store'])
        assert all(x==min(slots,live+t) for x in pool['worst_case_rows_per_store'])
        assert all(x==live for x in pool['measured_union_bound_per_store'])
        height=slots.bit_length()
        expected=t*(64+128+64+16*height)+(live+t)*(8*height+32)
        assert pool['journal_capacity']==expected
        assert pool['adjoint_bytes']==n*live*129*4
        assert pool['hot_payload_bytes']==n*live*192*4
        assert len(r['raw_ms'])==5
        assert all(x['peak_reserved_bytes']>=x['peak_allocated_bytes'] for x in r['raw_ms'])
assert points==50
archive=load('part6_trace_archive.json')
for trace in archive['traces']:
    packed=(R/trace['path']).read_bytes();raw=gzip.decompress(packed)
    assert hashlib.sha256(packed).hexdigest()==trace['compressed_sha256']
    assert hashlib.sha256(raw).hexdigest()==trace['uncompressed_sha256']
assert len(archive['checks'])==21 and all(x['summary_equivalence'] for x in archive['checks'])
result=dict(status='passed',parity=s['parity'],points_validated=points,
            exact_pool_and_journal_bounds_verified=True,compressed_traces_verified=len(archive['traces']),
            no_pt_files=True,agents_claude_synchronized=True,
            source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source_paths},
            parent_head_at_check=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            manual_notes='Validated exact-union row pitch, per-store worst-case/event bounds, actual production journal capacity, allocation peaks and trace attribution. All targets are not met; failed gates remain failed in summary.json. Rejected variants and raw OOM/GC receipts are retained. No candidate performance node private cleanup; only discarded historical oracle test nodes are explicitly released.')
(R/'part6_manual_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Passed: 179 parity tests, {points} sweep bounds, {len(archive["traces"])} trace hashes, AST and no-.pt checks')
