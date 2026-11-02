"""Independent exhaustive checks for finite suffix intervals and OOD backoff."""
import numpy as np
from experiments.expressivity_tasks.x43_fork_resolved.c9 import Index, runs

def test_queries_and_repeated_suffixes():
    rng=np.random.default_rng(4309)
    for n in range(2,32):
        raw=bytes(rng.integers(0,4,n,dtype=np.uint8))
        index=Index(raw)
        for _ in range(10):
            history=bytes(rng.integers(0,5,12,dtype=np.uint8))
            q,length,full,_=index.distribution(history)
            def ends(context):
                return [i for i in range(len(context),n) if raw[i-len(context):i]==context]
            candidates=[l for l in range(1,13) if len(ends(history[-l:]))>=2]
            expected=max(candidates,default=0)
            assert length==expected
            occurrences=ends(history[-expected:] if expected else b'')
            counts=np.bincount([raw[i] for i in occurrences],minlength=256)
            assert np.allclose(q,counts/counts.sum())
            assert full==len(ends(history))

def test_excursions_include_terminal_run():
    assert runs([True,True,False,True,False,True,True,True])==[2,1,3]
    assert runs([False,False])==[]
