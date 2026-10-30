"""Summarize retained receipts without treating sanitizer summaries as app success."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]


def read(path):
    if path.suffix == '.gz':
        return gzip.decompress(path.read_bytes()).decode()
    return path.read_text()


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    inventory = json.loads((OUT.parent / 'part17/hazard_inventory.json').read_text())
    results = {}
    for line in (OUT / 'status.txt').read_text().splitlines():
        name, code = line.split()
        results[name] = dict(exit=int(code))
    # The runner was edited while v1 was executing. Its shell then failed to
    # parse the suffix. The raw tool report and API process exit both say 86;
    # keep this separate from the successfully recorded runner exits.
    results['scratch_ordering_racecheck'] = dict(
        exit=86, accounting='manual: raw tool failure and exec API exit; runner suffix parse failed')
    for name, row in results.items():
        path = OUT / (name + '.txt')
        if not path.exists():
            path = Path(str(path) + '.gz')
        if not path.exists():
            continue
        log = read(path)
        row.update(log=path.name, log_sha256=hashlib.sha256(log.encode()).hexdigest(),
                   app_passed='PASSED' in log or '"status": "passed"' in log,
                   assertions=sorted(set(re.findall(r'Assertion `([^`]+)` failed', log))),
                   race_groups=len(re.findall(r'Error: Race reported between', log)),
                   init_errors_printed=log.count('Uninitialized __global__ memory read'))
        summary = re.findall(r'(?:RACECHECK SUMMARY|ERROR SUMMARY):[^\n]+', log)
        row['tool_summaries'] = summary
        if (OUT / (name + '.json')).exists():
            receipt = json.loads((OUT / (name + '.json')).read_text())
            row['application_receipt_status'] = receipt['status']
            row['app_passed'] = receipt['status'] == 'passed'
        row['accepted_repair'] = False
    groups = []
    sass = read(OUT / 'mask.sass.gz').splitlines()
    by_pc = {}
    for i, line in enumerate(sass):
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(.+)', line)
        if match:
            by_pc[int(match[1],16)] = dict(instruction=match[2],
                                         context=sass[max(0,i-5):i+6])
    raw_path = OUT / 'mask_racecheck.txt'
    raw = read(raw_path if raw_path.exists() else Path(str(raw_path) + '.gz'))
    for i, block in enumerate(re.split(r'========= Error: Race reported between ', raw)[1:],1):
        lines = block.splitlines()
        access_lines = [lines[0]]
        for line in lines[1:]:
            if 'and ' in line and 'access at scan_kernel+' in line:
                access_lines.append(line.removeprefix('=========     and '))
            else:
                break
        accesses = []
        for line in access_lines:
            pc = int(re.search(r'scan_kernel\+0x([0-9a-f]+)', line)[1],16)
            accesses.append(dict(report=line, pc=hex(pc), sass=by_pc.get(pc)))
        groups.append(dict(id=f'P18MASK-G{i:02d}', accesses=accesses,
                           classification='REAL; unresolved',
                           original_R_mapping='not established; source offsets are insufficient'))
    production = ROOT / 'experiments/expressivity_tasks/x40_emender_lm/kernels.py'
    before = subprocess.check_output(['git','show','231fcd9c:experiments/expressivity_tasks/x40_emender_lm/kernels.py'],cwd=ROOT)
    after = production.read_bytes()
    assert before == after, 'Production must remain unchanged after rejected candidates'
    result = dict(status='failed_incomplete_no_promotion', production_unchanged=True,
        production_sha256=hashlib.sha256(after).hexdigest(), results=results,
        original_race_obligations=[dict(id=g['id'], status='REAL; unresolved; no accepted repair')
                                   for g in inventory['race_groups']],
        mask_race_groups=groups,
        not_completed=['production promotion', 'R01-R35 repair and exact original-PC provenance',
                       'all-shape bitwise certification',
                       'int32-add assertion causal resolution', '100 full production 9x repeats',
                       '109 oracle recertification', 'graph/eager recertification',
                       '2M-token bit-matched curve', 'post-fix part-4 connection test',
                       'production 9x bin', 'budget-scaled depth probe', 'final depth verdict'],
        coordination='contact_supervisor searched first and by name/description; unavailable. No alternate supervisor channel used.',
        scope='lambda01; approved venv only; sequential broker leases restricted to 2/3/4; no WG orchestration, Slurm, resilient training or .pt commits')
    (OUT / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], runs=len(results),
                          mapped_mask_groups=len(groups), production_unchanged=True)))


if __name__ == '__main__':
    main()
