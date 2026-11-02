import hashlib
import unittest
from c14_documents import DocumentWindows, evaluation_halves


def receipt(raw, i=0, split='train'):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), source_index=i,
                split=split, truncated=False)


class DocumentBoundaryTests(unittest.TestCase):
    def test_every_context_and_shifted_target_stays_in_one_document(self):
        for context in (1, 2, 7, 128):
            sampler = DocumentWindows(context)
            for i, raw in enumerate((b'A'*129, b'B'*257, b'', b'C', bytes(range(256)),
                                     'é\x00\x1e🙂'.encode())):
                rows = list(sampler.windows(raw, receipt(raw, i)))
                self.assertEqual(sum(len(w.targets) for w in rows), max(0, len(raw)-1))
                for w in rows:
                    self.assertEqual(w.document, i)
                    self.assertEqual(w.inputs, raw[w.start:w.start+len(w.inputs)])
                    self.assertEqual(w.targets, raw[w.start+1:w.start+len(w.inputs)+1])
                    for j in range(len(w.targets)):
                        target = w.start+j+1
                        self.assertEqual(sampler.causal_context(raw, target, w.start),
                                         raw[w.start:target])

    def test_eval_uses_same_boundary_rule_and_disjoint_documents(self):
        sampler = DocumentWindows(2)
        for split in ('train', 'valid'):
            raw = b'ABCDE'
            self.assertEqual(len(list(sampler.windows(raw, receipt(raw, split=split), split))), 2)
            with self.assertRaises(ValueError):
                list(sampler.windows(raw, receipt(raw, split=split),
                                     'valid' if split == 'train' else 'train'))

    def test_fail_closed_on_corruption_and_token_truncation(self):
        sampler = DocumentWindows()
        with self.assertRaises(ValueError):
            list(sampler.windows(b'AB', receipt(b'AC')))
        r = receipt(b'AB'); r['truncated'] = True
        with self.assertRaises(ValueError):
            list(sampler.windows(b'AB', r))
        for args in ((b'abc', 0, 0), (b'abc', 3, 0), (b'abc', 1, 2)):
            with self.assertRaises(ValueError):
                sampler.causal_context(*args)

    def test_calibration_and_final_test_are_document_disjoint(self):
        self.assertEqual(evaluation_halves([22, 85, 91, 103]),
                         {22: 'calibration', 85: 'calibration',
                          91: 'final_test', 103: 'final_test'})
        for ids in ([], [22], [22, 22], [85, 22]):
            with self.assertRaises(ValueError):
                evaluation_halves(ids)
