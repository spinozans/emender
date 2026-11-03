import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from experiments.expressivity_tasks.x43_fork_resolved.c14_registered_cache import journal, checked_resume
from experiments.expressivity_tasks.x43_fork_resolved.c14_union_service import recover, phase_target, eval_windows
from experiments.expressivity_tasks.x43_fork_resolved.c14_feature_extract import digest


class UnionServiceTests(unittest.TestCase):
    def test_crash_replays_only_intent_bound_orphans(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            journal(p/'journal.jsonl', dict(event='write_intent', file='00000000.sparse'))
            (p/'00000000.sparse').write_bytes(b'partial payload')
            with (p/'journal.jsonl').open('ab') as out:
                out.write(b'{"event":')
            recover(p)
            self.assertEqual(checked_resume(p, {}), (0, 0))
            self.assertEqual(len(list(p.glob('*.crash_*'))), 1)
            self.assertEqual(len(list(p.glob('*.partial'))), 1)
            recover(p)
            (p/'unexpected.sparse').write_bytes(b'unknown')
            with self.assertRaisesRegex(ValueError, 'unexpected orphan'):
                recover(p)

    def test_committed_corruption_still_fails_closed(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp); f = p/'00000000.sparse'; f.write_bytes(b'valid')
            journal(p/'journal.jsonl', dict(event='shard_complete', file=f.name,
                identity={}, first_window_index=0, windows=1, positions=1,
                bytes=5, sha256=digest(f)))
            f.write_bytes(b'wrong'); recover(p)
            with self.assertRaisesRegex(ValueError, 'corruption'):
                checked_resume(p, {})

    def test_union_has_no_duplicate_training_prefix(self):
        a = dict(arms=[dict(windows=10, targets=100), dict(windows=25, targets=250)],
                 eval=dict(windows=8, targets=80))
        self.assertEqual(phase_target(a, 'transformer'), dict(windows=15, targets=150))
        self.assertEqual(sum(phase_target(a, p)['targets'] for p in ('emender','eval','transformer')), 330)

    def test_eval_is_frozen_and_document_bounded(self):
        import hashlib
        with TemporaryDirectory() as tmp:
            p = Path(tmp); receipts = p/'receipts/c14_dense'; receipts.mkdir(parents=True)
            source = p/'source'; source.write_bytes(b'abcXYZ')
            offsets = p/'offsets'
            offsets.write_text(''.join(json.dumps(dict(source_index=i, split='valid',
                byte_offset=i*3, byte_length=3, sha256=hashlib.sha256(data).hexdigest()))+'\n'
                for i, data in enumerate((b'abc', b'XYZ'))))
            windows = p/'windows'
            windows.write_text(''.join(json.dumps(dict(document=i, start=0, targets=2))+'\n' for i in range(2)))
            st = source.stat()
            (receipts/'pile_span.json').write_text(json.dumps(dict(offsets_path=str(offsets),
                offsets_sha256=digest(offsets), source=str(source), source_size=st.st_size,
                source_inode=st.st_ino, source_mtime_ns=st.st_mtime_ns)))
            (receipts/'eval_construction.json').write_text(json.dumps(dict(windows_path=str(windows))))
            audit = dict(eval=dict(windows_sha256=digest(windows), windows=2, targets=4))
            with patch('experiments.expressivity_tasks.x43_fork_resolved.c14_union_service.ROOT', p):
                self.assertEqual(list(eval_windows(audit)), [(0, 0, [b'a', b'ab']), (1, 0, [b'X', b'XY'])])
                windows.write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    list(eval_windows(audit))


if __name__ == '__main__':
    unittest.main()
