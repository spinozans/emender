"""Profile/bit-check 32 real shards: 256 registered batch-16 step equivalents."""
import cProfile,hashlib,json,pstats,time
import numpy as np
from .c14_b_train import CACHE,OUT
from .c14_b_oracle import oracle_shard
from .c14_b_train_decode import decode
from .c14_fast_features import load_decoder


def main():
    out=OUT.with_name('c14_b_v2');decoder=load_decoder()
    records=[r for s in (CACHE/'journal.jsonl').read_text().splitlines() if (r:=json.loads(s))['event']=='shard_complete'][:32]
    profile=cProfile.Profile();times=[];positions=0
    for r in records:
        data=(CACHE/r['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256']
        at=time.perf_counter();profile.enable();expected=oracle_shard(data,r['windows'],decoder);profile.disable();slow=time.perf_counter()-at
        at=time.perf_counter();actual=decode(data,r['windows'],decoder);fast=time.perf_counter()-at
        for a,b in zip(expected[:4],actual[:4]):assert a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))
        positions+=int(actual[1].sum());times.append(dict(shard=r['file'],windows=r['windows'],python_oracle_seconds=slow,native_decode_seconds=fast))
    with (out/'decode_profile.txt').open('w') as f:pstats.Stats(profile,stream=f).sort_stats('cumtime').print_stats(35)
    result=dict(shards=len(records),windows=sum(r['windows'] for r in records),targets=positions,batch16_step_equivalents=sum(r['windows'] for r in records)/16,
        all_feature_and_metadata_bits_identical=True,python_oracle_seconds=sum(r['python_oracle_seconds'] for r in times),native_decode_seconds=sum(r['native_decode_seconds'] for r in times),rows=times,
        caveat='Python oracle is cProfile-instrumented; native decoder is not. End-to-end pipeline profiling follows the full launch gates.')
    (out/'decode_profile.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
