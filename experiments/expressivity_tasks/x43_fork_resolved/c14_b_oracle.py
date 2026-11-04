"""Read-only B1 oracle adapter, preserving the complete registered cache rows."""
from io import BytesIO
import numpy as np
from .c14_cache_scale import read_windows
from .c14_fast_features import decode_shard
from .c14_branch_serve import features


def oracle_shard(data, count, decoder=None):
    identities, lengths, values = decode_shard(data,count,decoder)
    rows = list(read_windows(BytesIO(data)))
    if len(rows) != count:
        raise ValueError('cache row count mismatch')
    contexts = np.zeros((count,128),dtype=np.int64)
    for j,(doc,offset,rr) in enumerate(rows):
        if (doc,offset,len(rr)) != (*identities[j],lengths[j]):
            raise ValueError('cache identity mismatch')
        expected = np.asarray([features(r) for r in rr],dtype=np.float32)
        if not np.array_equal(values[j,:len(rr)].view(np.uint32),expected.view(np.uint32)):
            raise ValueError('cache oracle bit parity failed')
        contexts[j,:len(rr)] = [r['context_bytes'] for r in rr]
    return identities,lengths,values,contexts,rows
