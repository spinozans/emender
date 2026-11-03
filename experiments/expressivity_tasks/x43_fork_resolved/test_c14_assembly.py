import json
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments.expressivity_tasks.x43_fork_resolved.c14_assembly import assemble, replay, Shards
from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import features, select
from experiments.expressivity_tasks.x43_fork_resolved.test_c14_branch_serve import oracle


class AssemblyTests(unittest.TestCase):
    def test_261_contexts_exact_bits(self):
        corpus = bytes(range(256))+b'abcabc\x1eabz'
        lefts = [bytes([i]) for i in range(256)]+[b'zzabc', b'', b'\x00\n', b'-\x1eab', b'never']
        rows, _, _ = select(lefts, oracle(corpus))
        self.assertEqual(assemble(rows).tobytes(), np.asarray([features(r) for r in rows], dtype='<f8').tobytes())

    def test_large_integer_division_and_terminal_empty(self):
        rows = [dict(support_size=2**64-1, branches=[(0, 2**63+1), (255, 2**63-2)],
            entropy=0.7, distinct_branches=2, context_bytes=1024),
            dict(support_size=0, branches=[(30, 99)])]
        self.assertEqual(assemble(rows).tobytes(), np.asarray([features(r) for r in rows], dtype='<f8').tobytes())
        self.assertEqual(assemble([]).shape, (0, 260))

    def test_shard_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(document=12, start=128, position=1, support_size=0, branches=[])
            with gzip.open(root/'old.gz', 'wt') as out:
                out.write(json.dumps(row)+'\n')
            result = replay(root/'old.gz', root/'new', batch=1)
            self.assertEqual(result['positions'], 1)
            self.assertEqual(result['oracle_feature_sha256'], result['assembled_feature_sha256'])
            self.assertEqual(np.load(root/'new/000000.ids.npy').tolist(), [[12,128,1]])
            with self.assertRaises(FileExistsError):
                Shards(root/'new')
            with self.assertRaises(ValueError):
                Shards(root/'other').write([row], [])

    def test_direct_extraction_chunk_boundary(self):
        from experiments.expressivity_tasks.x43_fork_resolved.c14_assembly import extract
        class Server:
            def __init__(self, *args, **kwargs):
                self.query = oracle(b'ababc')
            def close(self):
                pass
        windows = [(7, i*128, [b'a', b'ab']) for i in range(129)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'cache'
            with patch('experiments.expressivity_tasks.x43_fork_resolved.c14_assembly.serving.BranchServer', Server), patch('experiments.expressivity_tasks.x43_fork_resolved.c14_assembly.serving.source_windows', return_value=iter(windows)):
                summary = extract(output, 129, 1)
            self.assertEqual(summary['positions'], 258)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual([s['positions'] for s in manifest['shards']], [256, 2])
            self.assertEqual(np.load(output/'000001.ids.npy').tolist(), [[7,16384,1],[7,16384,2]])
