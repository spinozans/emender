"""Independent receipt audit; does not certify the native index against an oracle."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    path=Path(path)
    if path.exists(): return path.read_bytes()
    return gzip.decompress(Path(str(path)+'.gz').read_bytes())


def audit(directory):
    root=Path(directory)
    summary=json.loads(read(root/'summary.json'))
    raw=read(root/'features.jsonl')
    assert hashlib.sha256(raw).hexdigest()==summary['feature_sha256']
    rows=[json.loads(line) for line in raw.splitlines()]
    assert len(rows)==summary['positions']
    query_count=0
    native={}
    for t in json.loads(read(root/'timings.json')):
        assert t['returncode']==0
        journal=read(root/(t['name']+'.stderr'))
        assert b'QUERY_COUNT_IMPOSSIBLE' not in journal
        assert b'QUERY_LOCATE mode=count positions_located=0 lf_steps=0' in journal
        requests=json.loads(read(root/(t['name']+'.requests.json')))
        lines=read(root/(t['name']+'.stdout')).split(b'\n')
        if lines[-1]==b'': lines.pop()
        assert len(lines)==len(requests)==t['patterns']
        for i,line in enumerate(lines):
            assert int(line.split(b'\t',1)[0])==i
            count=int(line.rsplit(b'\t',1)[1])
            assert 0<=count<=100093934467
            native[bytes.fromhex(requests[i])]=count
        query_count+=len(lines)
    span=json.loads(read(root.parent/'first_slice_span.json'))
    documents={}
    with Path(span['pile_path']).open('rb') as f:
        for doc in span['documents']:
            f.seek(doc['byte_offset']); text=f.read(doc['byte_length'])
            assert hashlib.sha256(text).hexdigest()==doc['sha256']
            documents[doc['source_index']]=(doc['byte_offset'],text)
    for row in rows:
        offset,text=documents[row['document']]
        target=row['target_offset']-offset
        assert 0<=row['reset']<target<len(text)
        assert target-row['reset']<=128
        left=text[max(row['reset'],target-1024):target]
        k=max((k for k in range(1,len(left)+1) if native[left[-k:]]),default=0)
        assert row['context_bytes']==k
        assert row['counts']==([native[left[-k:]+bytes([b])] for b in range(256)] if k else [0]*256)
        cs=row['counts']; total=sum(cs)-cs[30]
        assert len(cs)==256 and len(row['features'])==260
        assert total==row['support_size'] and sum(cs)<=row['context_count']
        assert row['terminal_count']==cs[30]
        assert row['coverage']==bool(total)
        if total:
            ps=[c/total if b!=30 else 0. for b,c in enumerate(cs)]
            entropy=-sum(p*math.log(p) for p in ps if p)
            assert abs(entropy-row['entropy'])<1e-12
            assert row['features']==ps+[math.log1p(total),entropy,float(sum(c>0 for b,c in enumerate(cs) if b!=30)>1),math.log1p(row['context_bytes'])]
        else: assert row['features']==[0.]*260
        assert row['context_bytes']<=128
        assert row['reset'] >= 0 and row['reset'] % 128 == 0
    return dict(status='passed',rows=len(rows),counts_checked=query_count,impossible_counts=0,
                full_256_counts_per_row=True,scope='range, shape, branch-sum, feature arithmetic, journal and artifact integrity; not independent Wikipedia count correctness')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args()
    print(json.dumps(audit(a.directory),indent=2))
