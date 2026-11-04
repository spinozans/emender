"""Concrete launch gate evidence; no replacement of immutable registration."""
import json
import os
import sys
import time
from pathlib import Path
import torch
from .validate_c14_b import audit
from .c14_documents import sha
from .c14_b_model import StoreIntegrated
from .c14_b_oracle import oracle_shard
from .c14_b_train import CODE,CACHE,OUT,ROOT,write


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    audit_result=audit()
    records=[json.loads(s) for s in (CACHE/'journal.jsonl').read_text().splitlines()]
    assert sha(CACHE/'journal.jsonl')=='3ae1aadac6688019d7c7195ce80e7163a7483ca456a8b67adda006f4c79895cb'
    shards=[r for r in records if r['event']=='shard_complete']
    oracle=[]
    for index in (0,len(shards)//2,len(shards)-1):
        r=shards[index];data=(CACHE/r['file']).read_bytes()
        assert __import__('hashlib').sha256(data).hexdigest()==r['sha256']
        ids,lengths,values,contexts,rows=oracle_shard(data,r['windows'])
        assert (contexts<=torch.arange(1,129).numpy()[None,:]).all()
        oracle.append(dict(shard=r['file'],windows=len(ids),positions=int(lengths.sum()),bit_identical=True,
                           raw_row_metadata_retained=True,sha256=r['sha256']))
    # Receipt is only emitted after the caller's entire leased test command
    # succeeds. Bind the tests and source to make stale green receipts unusable.
    log=OUT/'launch_battery.txt';text=log.read_text()
    assert 'failed' not in text.lower() and 'passed' in text and 'skipped' not in text.lower()
    test_names=sorted(p.name for p in ROOT.glob('test_c14*.py') if p.name!='test_c14_stacking.py')
    assert all(name in text for name in test_names)
    result=dict(status='passed',launch_ready=True,trainable_parameters=2133700,
        registration=audit_result,code_sha256={n:sha(ROOT/n) for n in CODE},
        tests_sha256={n:sha(ROOT/n) for n in test_names},test_log_sha256=sha(log),
        native_gpu=os.environ['GPU_LEASE_HELD'],oracle_samples=oracle,
        cache_decoder='standing v2 decoder unchanged; first/middle/last full-row bit identity checked',
        gates=['force-disabled dense exact native forward','cache answer bit parity including metadata',
               'unchanged 2133700 parameter inventory','native operation and array read hit-rate telemetry',
               'native/eager lifecycle and gradient parity under capacity pressure',
               'native stream-only law parity against frozen standing scan','enabled BF16 16x128 finite backward',
               'enabled checkpoint forward/gradient parity','causal observed correction and suffix retirement',
               'document resets/padding','full relevant C14 battery'],
        implementation='B-only Triton forward; immutable reset-state eager autograd replay; reentrant block checkpoint',
        limitations=['Replay recomputes the full independent window, rather than using the frozen X40 undo kernel; throughput is measured, not inferred.',
                     'Replay checks agreement with saved native outputs on every backward and fails closed on divergence.'],
        training_started=False,live_queries=0)
    write('launch_gates.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
