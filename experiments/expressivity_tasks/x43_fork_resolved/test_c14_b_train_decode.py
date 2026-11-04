"""Real frozen training-cache bits and raw metadata versus independent oracle."""
import hashlib,json
import numpy as np
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_train import CACHE
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_train_decode import decode
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_oracle import oracle_shard
from experiments.expressivity_tasks.x43_fork_resolved.c14_fast_features import load_decoder


def test_training_decode_first_middle_last_complete_bit_identity():
    records=[r for s in (CACHE/'journal.jsonl').read_text().splitlines() if (r:=json.loads(s))['event']=='shard_complete']
    decoder=load_decoder()
    for index in (0,len(records)//2,len(records)-1):
        r=records[index];data=(CACHE/r['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        expected=oracle_shard(data,r['windows'],decoder)
        actual=decode(data,r['windows'],decoder)
        for a,b in zip(expected[:4],actual[:4]):
            assert a.dtype==b.dtype and a.shape==b.shape
            assert np.array_equal(a.view(np.uint8),b.view(np.uint8))
        assert actual[4] is None
