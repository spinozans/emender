"""Paired positional-binding fixture and direct-scan reference (NOT a live index).

Run as a script. A query names a group, a field and a row; only their combination
identifies a value. The observed prefix is bytes only. No record IDs, gold answer,
record offsets or candidate value positions are provided to the lookup program.
"""
from collections import Counter
import hashlib
import random

K = 32
L = b' ' * K
R = b'.' * K


def locations(data, key):
    start = 0
    while (pos := data.find(key, start)) >= 0:
        yield pos
        start = pos + 1


def fixture(seed, groups=4, rows=2, fields=3):
    rng = random.Random(seed)
    def labels(tag, n):
        return [tag + x.to_bytes(2, 'big').hex().encode() for x in rng.sample(range(65536), n)]
    gs, rs, fs = labels(b'G',groups), labels(b'R',rows), labels(b'F',fields)
    keys = [(g,r,f) for g in gs for r in rs for f in fs]
    vs = dict(zip(keys,rng.sample(range(256),len(keys))))
    chosen = rng.choice(keys)
    # Swap two values under the SAME group and field but different rows.
    # A group or a (group,field) query cannot distinguish the pair by value counts.
    g,r,f = chosen
    other = (g,rng.choice([v for v in rs if v != r]),f)
    def perm(xs): return rng.sample(xs,len(xs))
    order = [(g,r,f) for g in perm(gs) for r in perm(rs) for f in perm(fs)]
    layout = [(key,rng.randrange(0,20)) for key in order]
    question = b'?'+g+r+f+b'='
    def emit(values):
        out=bytearray(); prev_g=prev_r=None
        for (g,r,f),n in layout:
            if g!=prev_g: out+=g;prev_r=None
            if r!=prev_r: out+=r
            out+=b'~'*n+f+L+bytes([values[g,r,f]])+R
            prev_g,prev_r=g,r
        return bytes(out)+question
    a=emit(vs); va=vs[chosen];vb=vs[other];vs[chosen],vs[other]=vb,va
    return (a,va),(emit(vs),vb)


def lookup(prefix):
    # Fixed grammar for reference policy only. Marks are record syntax, NOT
    # document delimiters. Never interpret arbitrary payload bytes as anchors.
    data, query = prefix[:-17],prefix[-17:]
    assert query[:1]==b'?' and query[-1:]==b'='
    g,r,f=query[1:6],query[6:11],query[11:16]
    # All field occurrences are candidates. Their preceding structural headers
    # must match both requested anchors (bounded ordered predecessor joins).
    import re
    headers=lambda tag: [(m.start(),m.group()) for m in re.finditer(tag+b'[0-9a-f]{4}',data)]
    gs,rs=headers(b'G'),headers(b'R'); candidates=[]
    for pos in locations(data,f):
        gp,gv=max(x for x in gs if x[0]<pos);rp,rv=max(x for x in rs if x[0]<pos)
        if gp<rp and gv==g and rv==r:candidates.append(pos)
    assert len(candidates)==1
    return data[candidates[0]+len(f)+K]


def check():
    transcript=hashlib.sha256(); n=0
    for seed in range(64):
        (a,y),(b,z)=fixture(seed)
        assert y!=z and a[-17:]==b[-17:] and lookup(a)==y and lookup(b)==z
        for k in range(1,K+1):
            assert Counter(a[i:i+k] for i in range(len(a)-k+1))==Counter(b[i:i+k] for i in range(len(b)-k+1))
        for p,t in [(a,y),(b,z)]:transcript.update(p);transcript.update(bytes([t]));n+=1
    print(f'{n} oracle targets correct; 64 opposite-answer pairs; all substring spectra k=1..{K} equal within each pair; sha256={transcript.hexdigest()}')

if __name__=='__main__':check()
