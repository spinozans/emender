"""Receipt exact old/new CPU feature paths over a source-ordered prefix."""
import hashlib
import io
import json
import time
import numpy as np
from .c14_consultation_train import CACHE, ROOT
from .c14_cache_scale import read_windows
from .c14_branch_serve import features
from .c14_fast_features import decode_shard, load_decoder


def main():
    out=ROOT/'receipts/c14_consultation_emender_v2'
    out.mkdir(exist_ok=True)
    records=[json.loads(x) for x in (CACHE/'journal.jsonl').read_text().splitlines()]
    shards=[r for r in records if r['event']=='shard_complete'][:38]
    decoder=load_decoder()
    timings=dict(read=0., sparse_decode=0., python_assembly=0., native_decode_assembly=0.)
    evidence=[]
    for r in shards:
        t=time.perf_counter(); data=(CACHE/r['file']).read_bytes(); timings['read']+=time.perf_counter()-t
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        t=time.perf_counter(); rows=list(read_windows(io.BytesIO(data))); timings['sparse_decode']+=time.perf_counter()-t
        t=time.perf_counter(); expected=np.zeros((len(rows),128,260),np.float32)
        for j,(_,_,rr) in enumerate(rows): expected[j,:len(rr)]=[features(v) for v in rr]
        timings['python_assembly']+=time.perf_counter()-t
        t=time.perf_counter(); ids,lengths,actual=decode_shard(data,len(rows),decoder); timings['native_decode_assembly']+=time.perf_counter()-t
        assert ids.tolist()==[[d,o] for d,o,_ in rows]
        assert lengths.tolist()==[len(rr) for _,_,rr in rows]
        assert np.array_equal(expected.view(np.uint32),actual.view(np.uint32)),r['file']
        evidence.append(dict(file=r['file'],sparse_sha256=r['sha256'],dense_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),windows=len(rows),dense_bytes=actual.nbytes))
        print(r['file'],flush=True)
    result=dict(status='passed',scope='CPU-only feature profile; not a training throughput measurement',
        steps=sum(x['windows'] for x in evidence)//16,timings_seconds=timings,shards=evidence,
        bit_identical=True,dense_bytes_on_disk=0,max_dense_shard_bytes=max(x['dense_bytes'] for x in evidence),
        journal_sha256=hashlib.sha256((CACHE/'journal.jsonl').read_bytes()).hexdigest())
    (out/'feature_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='shards'}),flush=True)

if __name__=='__main__': main()
