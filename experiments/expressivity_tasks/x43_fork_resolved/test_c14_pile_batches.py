"""Exercise the production offset sampler across gaps and capped documents."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.expressivity_tasks.x43_fork_resolved import c14_dense_train as train


class PileOffsetTests(unittest.TestCase):
    def test_gaps_caps_and_tail_windows_never_enter_training(self):
        first = bytes(range(256)) + b'AB'
        second = b'xyz'
        raw = first + b'UNREGISTERED SUFFIX\x1eOMITTED DOC\x1e' + second + b'\x1e'
        second_at = raw.index(second)
        docs = [dict(source_index=4, split='train', byte_offset=0, byte_length=len(first)),
                dict(source_index=9, split='valid', byte_offset=second_at, byte_length=3)]
        with tempfile.TemporaryDirectory() as directory:
            offsets = Path(directory)/'offsets.jsonl'
            offsets.write_text(''.join(json.dumps(d)+'\n' for d in docs))
            with patch.object(train, 'OFFSETS', offsets):
                rows = [row for batch in train.batches(np.frombuffer(raw, dtype=np.uint8), 'train') for row in batch]
                self.assertEqual([(d,s,len(b)-1) for d,s,b in rows], [(4,0,128),(4,128,128),(4,256,1)])
                self.assertEqual(b''.join(bytes(b[1:]) for _,_,b in rows), first[1:])
                valid = list(train.batches(np.frombuffer(raw, dtype=np.uint8), 'valid'))
                self.assertEqual(bytes(valid[0][0][2]), second)

    def test_padding_is_masked_and_shifted_targets_are_real(self):
        as_tensor = torch.as_tensor
        with patch.object(train.torch, 'as_tensor', side_effect=lambda a,device: as_tensor(a)):
            x,y,mask = train.tensors([(7,0,np.frombuffer(b'ABC',dtype=np.uint8)),
                                     (11,0,np.frombuffer(b'YZ',dtype=np.uint8))])
        self.assertEqual(int(mask.sum()),3)
        self.assertEqual(x[mask].tolist(),list(b'ABY'))
        self.assertEqual(y[mask].tolist(),list(b'BCZ'))
        self.assertFalse(bool(mask[:,2:].any()))


if __name__ == '__main__':
    unittest.main()
