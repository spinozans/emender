"""Bind fresh failed race reports to artifacts; never infer group closure."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    captures = []
    for path in sorted(ROOT.glob('capture_*.json')):
        record = json.loads(path.read_text())
        prefix = str(path.with_suffix(''))
        for kind, expected in record['artifacts'].items():
            assert digest(gzip.decompress(Path(prefix + '.' + kind + '.gz').read_bytes())) == expected
        with tempfile.TemporaryDirectory(prefix='x40_part23_audit_') as directory:
            cubin = Path(directory) / 'kernel.cubin'
            cubin.write_bytes(gzip.decompress(Path(prefix + '.cubin.gz').read_bytes()))
            sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(cubin)])
        Path(prefix + '.sass.gz').write_bytes(gzip.compress(sass, mtime=0))
        instructions = dict(re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*', sass.decode()))
        captures.append(dict(kernel_hash=record['kernel_hash'], artifacts=record['artifacts'],
                             sass_sha256=digest(sass), instructions=instructions))
    assert len(captures) == 1, 'isolated single-launch binding must be reconsidered'
    runs = []
    for path in sorted(ROOT.glob('baseline_*/console.txt.gz')):
        raw = gzip.decompress(path.read_bytes())
        source = raw.decode(errors='replace')
        groups = []
        current = None
        isolated = path.parent.name.startswith('baseline_9x.')
        for line in source.splitlines():
            if line.startswith('========= Error: Race reported between '):
                current = dict(id=f'P23-{path.parent.name}-G{len(groups)+1:02d}', accesses=[])
                groups.append(current)
            match = re.search(r'(Read|Write) access at scan_kernel\+0x([0-9a-f]+) in (.*)', line)
            if match:
                assert current is not None
                kind, pc, location = match.groups()
                instruction = captures[0]['instructions'].get(pc.lstrip('0') or '0') if isolated else None
                # The isolated scan is the only captured launch. The certified
                # harness aborts inside X31 before the X40 capture hook runs.
                if isolated:
                    assert instruction is not None, f'unmapped PC {pc}'
                    assert ('STS' if kind == 'Write' else 'LDS') in instruction
                current['accesses'].append(dict(access=kind, pc='0x'+pc, location=location,
                    sass=instruction, cubin_sha256=captures[0]['artifacts']['cubin'] if isolated else None,
                    ownership='unproved', lifetime='unproved'))
        terminal = re.search(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', source)
        status = int((path.parent / 'status.txt').read_text())
        assert terminal and int(terminal[1]) == len(groups)
        assert status == 86 and 'device-side assert triggered' in source
        runs.append(dict(directory=path.parent.name, raw_sha256=digest(raw), exit_code=status,
                         application_completed=False, groups=groups,
                         terminal_summary=terminal[0], accepted=False,
                         binding='exact isolated X40 cubin' if isolated else 'X31 reference; cubin not captured'))
    inventory = ROOT.parent / 'part17/hazard_inventory.json'
    historical = json.loads(inventory.read_text())
    report = dict(status='failed_incomplete', production_changes=False, repaired_groups=[],
        historical_inventory_sha256=digest(inventory.read_bytes()),
        historical_groups=[dict(id=g['id'], status='unresolved',
            reason='No verified cross-specialization mapping or ownership/lifetime repair')
            for g in historical['race_groups']],
        historical_18_scope='Displayed groups in rejected part17 diagnostic, not 18 mapped R IDs',
        runs=runs, captures=[{k:v for k,v in c.items() if k != 'instructions'} for c in captures],
        residual_risks=['Shared scratch races remain real unresolved defects',
                        'Both fresh racecheck runs abort; later paths are uncovered',
                        'No proven per-lane ownership or arithmetic-neutral repair',
                        'X31 certified-shape reference fails before X40 is exercised'],
        gated_not_run=['fix-batch parity', 'full battery', '100-repeat 9x',
                       'budget-scaled prediction/evaluation', 'post-fix pressure connection', 'final depth verdict'])
    (ROOT / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],
        groups={r['directory']:len(r['groups']) for r in runs},
        mapped_isolated_endpoints=sum(len(g['accesses']) for r in runs if r['binding'].startswith('exact') for g in r['groups']),
        historical_groups_unresolved=len(report['historical_groups']))))


if __name__ == '__main__':
    main()
