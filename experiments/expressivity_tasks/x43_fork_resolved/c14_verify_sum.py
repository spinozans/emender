"""Audit original sequential vs CPython 3.12 compensated entropy float32 bits."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .c14_fast_features import decode_shard
from .c14_consultation_train import CACHE, ROOT


def main():
    source=ROOT/'c14_fast_features.c'
    library=Path('/home/erikg/c14_consultation_features/native_decode')/(hashlib.sha256(source.read_bytes()).hexdigest()+'.sumcheck.so')
    subprocess.run(['cc','-O3','-fno-fast-math','-ffp-contract=off','-DC14_REQUIRE_IDENTICAL','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True)
    decode=ctypes.CDLL(str(library)).c14_decode
    decode.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    decode.restype=ctypes.c_int
    shards=[r for r in map(json.loads,(CACHE/'journal.jsonl').read_text().splitlines()) if r['event']=='shard_complete']
    start=time.monotonic(); nw=n=byte_count=0
    try:
        for index,r in enumerate(shards):
            data=(CACHE/r['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            ids,lengths,values=decode_shard(data,r['windows'],decode)
            nw+=len(values); n+=sum(lengths.tolist()); byte_count+=len(data)
            if index%1000==0: print(index,nw,n,flush=True)
        result=dict(status='passed',full_cache_float32_identity=True,windows=nw,positions=n,sparse_bytes=byte_count,seconds=time.monotonic()-start,journal_sha256=hashlib.sha256((CACHE/'journal.jsonl').read_bytes()).hexdigest(),check_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),oracle_algorithm_source='https://raw.githubusercontent.com/python/cpython/v3.12.3/Python/bltinmodule.c',scope='Every entropy output in the entire immutable Emender cache: original sequential double summation vs CPython 3.12 compensated sum; all other feature operations unchanged.')
    except BaseException as error:
        result=dict(status='failed',error=repr(error),completed_windows=nw,completed_positions=n,shard_index=index,shard=r['file'],seconds=time.monotonic()-start)
        (ROOT/'receipts/c14_consultation_emender_v2/full_sum_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        raise
    (ROOT/'receipts/c14_consultation_emender_v2/full_sum_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
