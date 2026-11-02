import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.expressivity_tasks.x43_fork_resolved.c14_feature_extract import extract, slice_contexts, CountBridge, digest

class FeatureTests(unittest.TestCase):
    def test_full_alphabet_and_maximal_suffix_oracle(self):
        corpus=b'abcabc\x00abc\x1eabx'
        def query(ps, name):
            return [sum(corpus[i:i+len(p)]==p for i in range(len(corpus)-len(p)+1)) for p in ps]
        rows, _=extract([b'zzabc',b'ab',b'zz',b''],query)
        self.assertEqual(rows[0]['context_bytes'],3)
        self.assertEqual(rows[0]['counts'][0],1)
        self.assertEqual(rows[0]['counts'][30],1)
        self.assertEqual(rows[0]['terminal_count'],1)
        self.assertEqual(rows[0]['features'][30],0.)
        self.assertEqual(rows[0]['support_size'],2)
        self.assertEqual(rows[1]['counts'][ord('c')],3)
        self.assertEqual(rows[2]['features'],[0.]*260)
        self.assertEqual(rows[3]['features'],[0.]*260)
        self.assertTrue(rows[1]['censored'])
        self.assertAlmostEqual(sum(rows[0]['features'][:256]),1)

    def test_invalid_branch_sum_stops(self):
        def query(ps,name): return [1 if name=='suffixes' else 2 for _ in ps]
        with self.assertRaisesRegex(ValueError,'branch sum'): extract([b'a'],query)

    def test_unencodable_pattern_is_receipted_before_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge=CountBridge(tmp,xsa='/does/not/exist')
            with self.assertRaisesRegex(ValueError,'NUL\\+30'): bridge.query([b'a\x00\x1e'],'bad')
            self.assertEqual(json.loads((Path(tmp)/'bad.transport_failure.json').read_text()),['61001e'])

    def test_span_binding_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=b'a'*129+b'b'*10
            d=dict(source_index=0,sha256=hashlib.sha256(raw).hexdigest(),split='train',truncated=False)
            (root/'documents').write_text(json.dumps(d)+'\n')
            (root/'pile').write_bytes(b'other doc'+raw+b'future doc')
            span=dict(d,byte_offset=9,byte_length=len(raw))
            receipt=dict(pile_path=str(root/'pile'),documents_index_sha256=digest(root/'documents'),documents=[span])
            (root/'receipt').write_text(json.dumps(receipt))
            contexts,positions=slice_contexts(root/'receipt',root/'documents',windows=2)
            self.assertEqual(len(contexts),138)
            self.assertEqual(len(contexts[127]),128)
            self.assertEqual(contexts[128],b'a')
            self.assertEqual(positions[128]['reset'],128)
            self.assertEqual(contexts[-1],raw[128:138])
            (root/'pile').write_bytes(b'other doc'+b'z'*len(raw))
            with self.assertRaisesRegex(ValueError,'differs'): slice_contexts(root/'receipt',root/'documents',2)

if __name__=='__main__': unittest.main()
