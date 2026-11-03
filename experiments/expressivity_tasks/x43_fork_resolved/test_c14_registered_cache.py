from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import select
from experiments.expressivity_tasks.x43_fork_resolved.c14_feature_extract import digest
from experiments.expressivity_tasks.x43_fork_resolved.c14_registered_cache import FLOOR, checked_resume, guard, journal, slow_check
from experiments.expressivity_tasks.x43_fork_resolved.test_c14_branch_serve import oracle


class RegisteredCacheTests(unittest.TestCase):
    def test_floor_includes_pending_write(self):
        self.assertEqual(guard('.', 10, FLOOR+10), FLOOR+10)
        for free, size in ((FLOOR-1, 0), (FLOOR, 1)):
            with self.assertRaises(OSError):
                guard('.', size, free)

    def test_resume_rejects_orphans_corruption_and_identity_drift(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp); shard = p/'00000000.sparse'
            self.assertEqual(checked_resume(p, {'a': 1}), (0, 0))
            shard.write_bytes(b'example')
            with self.assertRaisesRegex(ValueError, 'unreceipted'):
                checked_resume(p, {'a': 1})
            journal(p/'journal.jsonl', dict(event='shard_complete', identity={'a': 1},
                file=shard.name, bytes=7, sha256=digest(shard),
                first_window_index=0, windows=2, positions=256))
            self.assertEqual(checked_resume(p, {'a': 1}), (2, 256))
            with self.assertRaisesRegex(ValueError, 'identity'):
                checked_resume(p, {'a': 2})
            shard.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'corruption'):
                checked_resume(p, {'a': 1})

    def test_exhaustive_sample_check_rejects_changed_feature(self):
        query = oracle(bytes(range(256))+b'abcabc\x1eab')
        lefts = [b'abc', b'\x1eab', b'notpresent', b'']
        rows, _, _ = select(lefts, query)
        self.assertEqual(len(slow_check(lefts, rows, query)), 3)
        rows[0]['context_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'slow suffix parity'):
            slow_check(lefts, rows, query)


if __name__ == '__main__':
    unittest.main()
