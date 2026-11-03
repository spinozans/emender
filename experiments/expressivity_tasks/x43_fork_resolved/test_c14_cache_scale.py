from io import BytesIO
import unittest

from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import features, select
from experiments.expressivity_tasks.x43_fork_resolved.c14_cache_scale import MAGIC, PAIR, ROW, WINDOW, QueryReuse, read_windows, write_window
from experiments.expressivity_tasks.x43_fork_resolved.test_c14_branch_serve import oracle


class CacheScaleTests(unittest.TestCase):
    def test_reuse_eviction_and_oversized_batch(self):
        native = oracle(bytes(range(256))*2+b'abcabc\x1e')
        for capacity in (0, 1, 64):
            reuse = QueryReuse(native, capacity)
            for patterns in ([b'a', b'b', b'a'], [b'b', b'c'], [b'a', b'c', b'z']):
                self.assertEqual(reuse(patterns), native(patterns))
                self.assertLessEqual(len(reuse.cache), capacity)

    def test_selector_independent_of_reuse(self):
        native = oracle(bytes(range(256))+b'abcd'*30+b'a\x1e')
        reuse = QueryReuse(native, 32)
        for lefts in ([b'a', b'abc', b'unknown', b''], [b'abc', b'\x1eabc', b'abcd'*32]):
            self.assertEqual(select(lefts, native)[:2], select(lefts, reuse)[:2])
        self.assertGreater(reuse.hits, 0)

    def test_compact_roundtrip_all_bytes_and_terminal(self):
        corpus = bytes(range(256))+b'ab\x1e'
        lefts = [bytes([b]) for b in range(256)]
        rows, _, _ = select(lefts, oracle(corpus))
        stream = BytesIO(MAGIC)
        stream.seek(len(MAGIC))
        for at in (0, 128):
            write_window(stream, 42, at, rows[at:at+128])
        stream.seek(0)
        restored = list(read_windows(stream))
        for at, (_, _, decoded) in zip((0, 128), restored):
            self.assertEqual(decoded, rows[at:at+128])
            self.assertEqual([features(r) for r in decoded], [features(r) for r in rows[at:at+128]])

    def test_all_256_branches_and_uint64_counts(self):
        pairs = [(b, 2**63+b) for b in range(256)]
        data = MAGIC+WINDOW.pack(3, 0, 1)+ROW.pack(1, 1, 1, 256)+b''.join(PAIR.pack(*p) for p in pairs)
        _, _, rows = next(read_windows(BytesIO(data)))
        self.assertEqual(rows[0]['branches'], pairs)
        self.assertEqual(rows[0]['context_count'], sum(c for _, c in pairs)+1)

    def test_corruption_and_truncation_rejected(self):
        valid = MAGIC+WINDOW.pack(1, 0, 1)+ROW.pack(1, 1, 0, 1)+PAIR.pack(0, 5)
        for end in (0, len(MAGIC)-1, len(MAGIC)+2, len(valid)-1):
            with self.assertRaises((EOFError, ValueError)):
                list(read_windows(BytesIO(valid[:end])))
        for row in (ROW.pack(2, 0, 0, 0), ROW.pack(1, 2, 0, 0), ROW.pack(1, 0, 2, 0), ROW.pack(0, 0, 0, 1)+PAIR.pack(0, 1)):
            with self.assertRaises(ValueError):
                list(read_windows(BytesIO(MAGIC+WINDOW.pack(1, 0, 1)+row)))
        with self.assertRaises(ValueError):
            QueryReuse(lambda _: [], 1)([b'a'])


if __name__ == '__main__':
    unittest.main()
