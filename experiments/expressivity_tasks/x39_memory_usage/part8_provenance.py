"""Audit recorded disk fingerprints without relabeling them as loaded sources.

CPU-only; does not modify historical receipts, checkpoints, or upstream code.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
TASKS = ROOT.parent
REC = ROOT / 'receipts'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    rows = []
    for receipt in sorted(REC.glob('part7_*_training.json')):
        run = json.loads(receipt.read_text())
        sources = {k: v for k, v in run['fingerprint']['sources'].items()
                   if k.startswith('x36_tape_perf/')}
        comparisons = {}
        for name, registered in sources.items():
            path = TASKS / name
            relative = path.relative_to(ROOT.parents[2])
            head = subprocess.run(['git', 'show', f'HEAD:{relative}'],
                                  capture_output=True, check=False)
            comparisons[name] = dict(registered_disk_sha256=registered,
                current_disk_sha256=digest(path.read_bytes()),
                head_sha256=digest(head.stdout) if head.returncode == 0 else None,
                loaded_source_sha256=None)
        rows.append(dict(receipt=str(receipt.relative_to(ROOT)),
            receipt_sha256=digest(receipt.read_bytes()),
            status=run['status'], tokens=run['tokens'], sources=comparisons))
    poor = json.loads((REC/'part7_poverty_evaluation.json').read_text())
    rich = json.loads((REC/'part7_poverty_rich_evaluation.json').read_text())
    shared = poor['checkpoint_sha256'] == rich['checkpoint_sha256']
    assert shared, 'poor/rich evaluations do not share weights'
    result = dict(status='unresolved-loaded-source-provenance', runs=rows,
        historical_integrity_sha256=digest((REC/'part7_integrity.json').read_bytes()),
        source_attestation_repaired=False,
        interpretation=('Registration fingerprints hash disk files before parity/training. '
            'All arms ran in one process with cached imports; backward.py is imported lazily. '
            'The receipts do not hash loaded code objects or snapshot source bytes. '
            'They cannot establish which changed backward implementation executed. '
            'Current oracle parity cannot retroactively supply that missing identity.'),
        poverty_existing_receipt_check=dict(shared_checkpoint=shared,
            checkpoint_sha256=poor['checkpoint_sha256'],
            budgets=[poor['budget'],rich['budget']],
            poor_slopes=poor['occupancy_log_log_slopes_per_head'],
            rich_slopes=rich['occupancy_log_log_slopes_per_head'],
            scope='Receipt consistency check only; no fresh model evaluation.'),
        semantic_arm=dict(status='not-trained',
            reason='Optimized recurrent match wire and its oracle/replay certification are absent.'),
        supervisor_contact=dict(available=False,
            discovery='ALL_TOOLS names and descriptions searched for contact_supervisor, supervisor and intercom; no matches.'),
        residual_risks=['Historical loaded-source identity remains unproven.',
            'No semantic-arm natural-text compression or quality verdict.',
            'Current parity is not parity at the historical source hashes.'])
    out = REC/'part8_provenance.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], runs=len(rows),
        poverty=result['poverty_existing_receipt_check'])))


if __name__ == '__main__':
    main()
