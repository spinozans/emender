"""Exercise acceptance and fail-closed ledger audits without touching receipts."""
import copy
import subprocess
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from validate_program import validate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEXT = (HERE / 'PROGRAM.md').read_text()
PATTERN = r'(?ms)^```json claim-bindings\n(.*?)\n```$'
LEDGER = json.loads(re.search(PATTERN, TEXT)[1])


class LedgerAuditTests(unittest.TestCase):
    def check_ledger(self, ledger, text=TEXT):
        replacement = '```json claim-bindings\n' + json.dumps(ledger) + '\n```'
        document = re.sub(PATTERN, lambda _: replacement, text)
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / 'PROGRAM.md'
            program.write_text(document)
            return validate(program, {'emender': ROOT, 'sxgc': ROOT.parent / 'sxgc'})

    def test_valid_ledger(self):
        self.assertEqual(self.check_ledger(LEDGER)['status'], 'green')

    def test_working_receipt_drift_fails(self):
        receipt = LEDGER['claims'][0]['receipts'][0]
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                       receipt['commit'] + ':' + receipt['path']])
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / receipt['path']
            target.parent.mkdir(parents=True)
            target.write_bytes(raw + b'changed')
            with patch('validate_program.git', side_effect=[
                    receipt['commit'].encode(), raw]):
                with self.assertRaisesRegex(ValueError, 'working receipt differs'):
                    validate(HERE / 'PROGRAM.md', {'emender': repo})

    def test_mutations_fail_closed(self):
        for mutation in ('hash', 'commit', 'missing', 'registered_receipt',
                         'measured_empty', 'refuted_empty', 'duplicate_id', 'heading'):
            with self.subTest(mutation=mutation):
                ledger = copy.deepcopy(LEDGER)
                text = TEXT
                receipt = ledger['claims'][0]['receipts'][0]
                if mutation == 'hash':
                    receipt['sha256'] = '0' * 64
                elif mutation == 'commit':
                    receipt['commit'] = '0' * 40
                elif mutation == 'missing':
                    receipt['path'] = 'missing-receipt.json'
                elif mutation == 'registered_receipt':
                    ledger['claims'][5]['receipts'] = [receipt]
                elif mutation == 'measured_empty':
                    ledger['claims'][0]['receipts'] = []
                elif mutation == 'refuted_empty':
                    ledger['claims'][3]['receipts'] = []
                elif mutation == 'duplicate_id':
                    ledger['claims'][1]['id'] = 'C1'
                else:
                    text = text.replace('### C1 — measured:', '### C1 — registered:')
                with self.assertRaises((ValueError, OSError, KeyError, subprocess.CalledProcessError)):
                    self.check_ledger(ledger, text)


if __name__ == '__main__':
    unittest.main()
