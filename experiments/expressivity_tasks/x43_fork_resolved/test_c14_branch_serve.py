import struct
import unittest
from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import decode, exact, select, features
from io import BytesIO


def oracle(corpus):
    def query(patterns):
        rows = []
        for p in patterns:
            counts = {}; eof = width = 0
            for i in range(len(corpus)-len(p)+1):
                if corpus[i:i+len(p)] == p:
                    width += 1
                    end = i+len(p)
                    if end == len(corpus):
                        eof += 1
                    else:
                        counts[corpus[end]] = counts.get(corpus[end], 0)+1
            rows.append(dict(width=width, eof=eof, branches=list(counts.items())))
        return rows
    return query


class ServingTests(unittest.TestCase):
    def test_all_bytes_and_maximality(self):
        corpus = bytes(range(256))+b'abcabc\x1eabz'
        lefts = [bytes([i]) for i in range(256)]+[b'zzabc', b'', b'\x00\n', b'-\x1eab', b'never']
        rows, rates, _ = select(lefts, oracle(corpus))
        for s, row in zip(lefts, rows):
            s = s.rsplit(b'\x1e', 1)[-1]
            k = max((k for k in range(1, len(s)+1) if s[-k:] in corpus), default=0)
            self.assertEqual(row['context_bytes'], k)
            if k:
                expected = oracle(corpus)([s[-k:]])[0]
                self.assertEqual(row['branches'], expected['branches'])
                self.assertEqual(row['context_count'], expected['width'])
            f = features(row)
            self.assertEqual(len(f), 260)
            self.assertEqual(f[30], 0)
            self.assertAlmostEqual(sum(f[:256]), float(row['coverage']))
        self.assertEqual(rates['256']['eligible'], 0)

    def test_terminal_only_is_all_zero(self):
        rows, _, _ = select([b'ab'], oracle(b'ab\x1e'))
        self.assertEqual(rows[0]['terminal_count'], 1)
        self.assertEqual(features(rows[0]), [0.]*260)

    def test_long_context_capped_and_coverage_measured(self):
        rows, rates, _ = select([b'x'*1500, b'az'*64], oracle(b'x'*2000+b'az'*32))
        self.assertEqual(rows[0]['context_bytes'], 1024)
        self.assertEqual(rows[1]['context_bytes'], 64)
        self.assertEqual(rates['1024'], dict(eligible=1, supported=1))

    def test_response_rejections(self):
        good = struct.pack('<6QH', 0, 10, 12, 2, 0, 0, 1)+struct.pack('<BQ', 255, 2)
        self.assertEqual(decode(good, 0)['branches'], [(255, 2)])
        for p, ident in [(good, 1), (good[:-1], 0), (b'', 0),
            (struct.pack('<6QH', 0, 10, 12, 2, 0, 1, 1)+struct.pack('<BQ', 0, 1), 0),
            (struct.pack('<6QH', 0, 10, 12, 2, 0, 0, 1)+struct.pack('<BQ', 0, 3), 0)]:
            with self.assertRaises(ValueError):
                decode(p, ident)
        with self.assertRaises(EOFError):
            exact(BytesIO(b'ab'), 3)

    def test_cardinality_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            select([b'a'], lambda ps: [])

if __name__ == '__main__':
    unittest.main()
