"""Safety checks for the attended capacity-prune receipt writer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('capacity_prune', Path(__file__).with_name('prune.py'))
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


class CapacityPruneSafetyTest(unittest.TestCase):
    def test_best_chain_uses_accuracy_loss_and_earliest_tiebreak(self):
        rows = [{'step': 0, 'accuracy': 0, 'loss': 9},
                {'step': 250, 'accuracy': 0, 'loss': 8},
                {'step': 500, 'accuracy': 0, 'loss': 8},
                {'step': 750, 'accuracy': .1, 'loss': 10}]
        self.assertEqual(prune.best_chain(rows), {0, 250, 750})

    def test_hash_absolute_relative_and_bare_path_pins(self):
        p = Path('/repo/run/step_000250.pt')
        self.assertEqual(prune.pin_sources(p, 'hash', {'hash': ['evaluation']}, {}), ['evaluation'])
        for name in (str(p), 'run/step_000250.pt', p.name):
            self.assertEqual(prune.pin_sources(p, '', {}, {name: ['report']}), ['report'])
        self.assertEqual(prune.pin_sources(p, '', {}, {'other/step_000250.pt': ['report']}), [])

    def fixture(self, base):
        prior = base / 'results_invertible_stage2b/receipts/postcompletion_capacity_prune.json'
        prior.parent.mkdir(parents=True)
        prior.write_text(json.dumps({'broadcast_x2b_stop_steps': []}))
        root = base / 'results_invertible_stage5c_growth_8m'
        run = root / 'run'
        run.mkdir(parents=True)
        ledger = {}
        for step in (0, 250, 500):
            p = run / f'step_{step:06d}.pt'
            p.write_bytes(b'test fixture, not a checkpoint')
            ledger[str(p)] = {'path': str(p), 'sha256': prune.digest(p)}
        (run / 'checkpoints.json').write_text(json.dumps(ledger))
        (run / 'result.json').write_text(json.dumps({'status': 'complete'}))
        return root, run

    def test_unknown_or_active_run_retains_everything(self):
        with tempfile.TemporaryDirectory(dir=prune.OUT) as tmp:
            base = Path(tmp)
            root, run = self.fixture(base)
            with patch.object(prune, 'BASE', base):
                plan = prune.root_plan(root, '5c_growth_8m', {}, {}, f'123 trainer --out {root}/run')
                self.assertEqual(plan['removed'], [])
                (run / 'result.json').write_text('{}')
                plan = prune.root_plan(root, '5c_growth_8m', {}, {}, '')
                self.assertEqual(plan['removed'], [])
                self.assertEqual(len(plan['ambiguities']), 3)

    def test_stage2b_override_keeps_best_matched_stops_and_downstream_pin(self):
        with tempfile.TemporaryDirectory(dir=prune.OUT) as tmp:
            base = Path(tmp)
            root, run = self.fixture(base)
            prior = base / 'results_invertible_stage2b/receipts/postcompletion_capacity_prune.json'
            prior.write_text(json.dumps({'broadcast_x2b_stop_steps': [2000]}))
            ledger = json.loads((run / 'checkpoints.json').read_text())
            for step in (1000, 2000):
                p = run / f'step_{step:06d}.pt'
                p.write_bytes(b'fixture')
                ledger[str(p)] = {'path': str(p), 'sha256': prune.digest(p)}
            (run / 'checkpoints.json').write_text(json.dumps(ledger))
            (run / 'result.json').write_text(json.dumps({'status': 'complete', 'step': 500, 'stop_step': 500}))
            with patch.object(prune, 'BASE', base):
                plan = prune.root_plan(root, '2b', {}, {'run/step_000250.pt': ['downstream evaluation']}, '')
            self.assertEqual([Path(e['path']).name for e in plan['removed']], ['step_000000.pt'])
            self.assertEqual(len(plan['retained']), 4)

    def test_compression_classification_fails_closed_for_other_projects(self):
        import compress_project_logs as logs
        with tempfile.TemporaryDirectory(dir=prune.OUT) as tmp:
            p = Path(tmp) / 'pi-bash-fixture.log'
            for text in ('unidentified project', '/home/erikg/emender/ with 1gbwt bioinformatics'):
                p.write_text(text)
                self.assertIsInstance(logs.classify(p), str)
            p.write_text('/home/erikg/emender/experiments/expressivity_tasks/result.json')
            with patch.object(logs.subprocess, 'run') as opened:
                opened.return_value.returncode = 1
                opened.return_value.stdout = opened.return_value.stderr = ''
                self.assertIsInstance(logs.classify(p), list)
                opened.return_value.returncode = 0
                self.assertIsInstance(logs.classify(p), str)

    def test_missing_hash_retained_and_removal_entry_is_hashable(self):
        with tempfile.TemporaryDirectory(dir=prune.OUT) as tmp:
            base = Path(tmp)
            root, run = self.fixture(base)
            ledger = json.loads((run / 'checkpoints.json').read_text())
            del ledger[str(run / 'step_000250.pt')]['sha256']
            (run / 'checkpoints.json').write_text(json.dumps(ledger))
            with patch.object(prune, 'BASE', base):
                plan = prune.root_plan(root, '5c_growth_8m', {}, {}, '')
            self.assertEqual(len(plan['removed']), 1)
            self.assertEqual(len(plan['retained']), 2)
            entry = dict(plan['removed'][0])
            recorded = entry.pop('manifest_entry_sha256')
            self.assertEqual(recorded, prune.entry_hash(entry))


if __name__ == '__main__':
    unittest.main()
