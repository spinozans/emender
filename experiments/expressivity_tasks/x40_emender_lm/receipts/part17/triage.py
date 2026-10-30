"""Lossless group/site inventory of the part-16 reports; CPU only."""
import hashlib
import gzip
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE.parent / 'part16'
report = (BEFORE / 'racecheck.txt').read_text()
groups = []
for number, chunk in enumerate(report.split('========= Error: Race reported between ')[1:], 1):
    lines = chunk.splitlines()
    accesses = [lines[0]] + [line.removeprefix('=========     and ') for line in lines[1:] if line.startswith('=========     and ')]
    groups.append(dict(id=f'R{number:02}', accesses=accesses,
        classification='REAL (conservative; not certified benign)',
        rationale='Reported unordered shared scratch accesses include writes and reads or multiple writes. Reduction operands vary across lanes and loop iterations; neither identical writer values nor thread-private ownership is established. Treat as a real repair obligation until instruction-level ordering/ownership proves otherwise.',
        proof_status='Sanitizer evidence; exact scratch lifetime and causal link to assertion unresolved'))
assert len(groups) == 35, len(groups)
init = (BEFORE / 'initcheck_isolated.txt').read_text()
sites = re.findall(r'Uninitialized __global__ memory read of size (\d+) bytes\n=========     at (.*?)\n=========     by (.*?)\n=========     Address (\S+)', init)
assert len(sites) == 100, len(sites)
out = dict(source_hashes={name: hashlib.sha256((BEFORE/name).read_bytes()).hexdigest() for name in ('racecheck.txt', 'initcheck_isolated.txt')},
    race_groups=groups,
    init_sites=[dict(id=f'U{i:03}', bytes=int(size), instruction=pc, thread_block=thread, address=address,
        classification='REAL uninitialized read; arithmetic influence unproven',
        rationale='empty_like cache has no initialization for newly allocated live slots. tl.where selects after the load, so it cannot prevent the read.') for i,(size,pc,thread,address) in enumerate(sites,1)],
    init_coverage=dict(total_reported_errors=8224, printed_sites=len(sites), suppressed=8124,
        limitation='Suppressed accesses cannot be individually classified from the retained log. Unlimited-print rerun required; no complete 8224-site inventory claimed.'))
(HERE/'hazard_inventory.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(dict(race_groups=len(groups), init_printed=len(sites), init_suppressed=8124)))
baseline = HERE/'initcheck_baseline.txt.gz'
if baseline.exists():
    raw = gzip.decompress(baseline.read_bytes())
    baseline_text = raw.decode()
    events = re.findall(r'Uninitialized __global__ memory read of size (\d+) bytes\n=========     at (.*?)\n=========     by (.*?)\n=========     Address (\S+)', baseline_text)
    total = int(re.search(r'ERROR SUMMARY: (\d+) errors', baseline_text).group(1))
    assert len(events) == total
    assert not re.search(r'errors? (?:were )?suppressed', baseline_text)
    counts = Counter(events)
    result = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), events=len(events),
        terminal_errors=total, suppressed=0,
        source_sites=sorted(set(event[1] for event in events)),
        classification='REAL uninitialized read; arithmetic influence unproven',
        rationale='All emitted events load the empty_like cache before a guaranteed store. New allocation paths choose the reference afterward, which does not prevent the read.',
        access_counts=[dict(bytes=int(k[0]), instruction=k[1], thread_block=k[2], address=k[3], occurrences=v) for k,v in sorted(counts.items())])
    (HERE/'baseline_init_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(baseline_events=len(events), distinct_accesses=len(counts), source_sites=result['source_sites'])))
