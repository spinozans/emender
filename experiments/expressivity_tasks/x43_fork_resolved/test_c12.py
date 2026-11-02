import numpy as np
from experiments.expressivity_tasks.x43_fork_resolved.c12 import interval
from experiments.expressivity_tasks.x43_fork_resolved.direct_sa import suffix_array

def test_cross_corpus_intervals_match_finite_bruteforce():
    rng=np.random.default_rng(4312)
    for n in (2,7,31):
        for _ in range(8):
            raw=bytes(rng.choice([0,1,255],size=n).astype(np.uint8)); rev=raw[::-1]
            sa=suffix_array(np.append(np.frombuffer(rev,dtype=np.uint8).astype(np.int64),-1));sa=sa[sa!=0]
            contexts=[b'',b'absent',bytes([255]),raw,raw[-2:]]
            contexts += [raw[i:j] for i in range(n) for j in range(i,min(i+4,n)+1)]
            for c in contexts:
                lo,hi=interval(rev,sa,c)
                got=sorted((n-sa[lo:hi]).tolist())
                expected=[i for i in range(n) if i>=len(c) and raw[i-len(c):i]==c]
                assert got==expected,(raw,c,got,expected)

def test_no_y_target_leaks_into_query():
    raw=b'abcabz';rev=raw[::-1]
    sa=suffix_array(np.append(np.frombuffer(rev,dtype=np.uint8).astype(np.int64),-1));sa=sa[sa!=0]
    lo,hi=interval(rev,sa,b'ab')
    assert sorted(raw[len(raw)-int(p)] for p in sa[lo:hi])==[ord('c'),ord('z')]
    lo,hi=interval(rev,sa,b'absent');assert lo==hi
