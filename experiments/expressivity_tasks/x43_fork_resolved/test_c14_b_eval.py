"""Evaluation decoder retains raw context even when all 260 inputs are zero."""
import io
import numpy as np
import pytest
from experiments.expressivity_tasks.x43_fork_resolved.c14_cache_scale import MAGIC,write_window,read_windows
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_eval_decode import load_context_decoder,eval_shard
from experiments.expressivity_tasks.x43_fork_resolved.c14_fast_features import load_decoder


def test_raw_context_is_not_reconstructed_from_zeroed_features():
    stream=io.BytesIO();stream.write(MAGIC)
    write_window(stream,7,128,[dict(context_bytes=1,censored=True,eof=0,branches=[(30,3)]),
        dict(context_bytes=2,censored=False,eof=1,branches=[(0,1),(255,2)])])
    data=stream.getvalue()
    ids,lengths,values,contexts=eval_shard(data,1,load_decoder(),load_context_decoder())
    rows=list(read_windows(io.BytesIO(data)))[0][2]
    assert contexts[0,:2].tolist()==[r['context_bytes'] for r in rows]==[1,2]
    assert not values[0,0].any()
    assert ids.tolist()==[[7,128]] and lengths.tolist()==[2]
    assert not contexts[0,2:].any()


@pytest.mark.parametrize('data',[b'',MAGIC+b'x',b'wrong magic'])
def test_raw_metadata_rejects_malformed_cache(data):
    out=np.zeros((1,128),dtype=np.int64)
    assert load_context_decoder()(data,len(data),out.ctypes.data,1)<0


def test_native_metadata_matches_exact_full_raw_suffixes_and_expiry():
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_eval_decode import owner_metadata,load_owner_decoder
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
    rng=np.random.default_rng(3401)
    raw=rng.integers(0,4,(3,128),dtype=np.uint8);raw[0]=0
    contexts=np.array([[rng.integers(0,i+2) for i in range(128)] for _ in range(3)],dtype=np.int64)
    valid=[128,117,128]
    ids,ends=owner_metadata(raw,contexts,valid,load_owner_decoder())
    for j in range(3):
        window=(j,128);observed=bytes(raw[j]);canonical={}
        for i in range(128):
            if i>=valid[j]:assert ids[j,i]==-1;continue
            owner=queried_suffix(window,observed[:i+1],int(contexts[j,i]))
            expected=canonical.setdefault(owner,len(canonical)) if owner.suffix else -1
            assert ids[j,i]==expected
            end=next((n for n in range(i+1,valid[j]) if not owner.is_current(window,observed[:n+1])),valid[j])
            assert ends[j,i]==end
