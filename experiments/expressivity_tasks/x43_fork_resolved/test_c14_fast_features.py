"""Native conversion retains exact float32 inputs, geometry and rejection."""
import io
import numpy as np
import pytest
from experiments.expressivity_tasks.x43_fork_resolved.c14_cache_scale import MAGIC, write_window, read_windows
from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import features
from experiments.expressivity_tasks.x43_fork_resolved.c14_fast_features import decode_shard


def test_exact_native_features():
    stream=io.BytesIO(); stream.write(MAGIC)
    rows=[dict(context_bytes=1,censored=False,eof=0,branches=[(30,7)]),
          dict(context_bytes=2,censored=True,eof=1,branches=[(255,999),(0,123),(30,2)]),
          dict(context_bytes=0,censored=False,eof=0,branches=[])]
    write_window(stream,12,128,rows)
    data=stream.getvalue()
    ids,lengths,actual=decode_shard(data,1)
    expected=np.zeros((1,128,260),np.float32)
    expected[0,:3]=[features(row) for _,_,rr in read_windows(io.BytesIO(data)) for row in rr]
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert ids.tolist()==[[12,128]] and lengths.tolist()==[3]


@pytest.mark.parametrize('data',[b'',MAGIC+b'x',b'wrong magic'])
def test_native_rejects_truncation(data):
    with pytest.raises(ValueError): decode_shard(data,1)
