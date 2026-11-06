"""Fail-closed full transformer union-cache reconciliation; no extraction/training."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .c14_documents import sha
from .c14_fast_features import decode_shard
from .c14_paired_eval import oracle

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_transformer'
CACHE = Path('/home/erikg/c14_consultation_features')

def run():
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    result = dict(status='failed', training_authorized=False)
    try:
        audit_path = ROOT/'receipts/c14_registered_scale/stream_audit.json'
        audit = json.loads(audit_path.read_text())
        old = subprocess.check_output(['git','show','f29462c3abcdc23bc974efe2d3510cf55577d7ff:'+str(audit_path.relative_to(ROOT.parents[2]))])
        historical = json.loads(old)
        for k in ('arms','eval','offsets_sha256','source','receipt_hashes'):
            assert historical[k] == audit[k], k
        for path,digest in audit['receipt_hashes'].items():
            assert sha(ROOT.parents[2]/path)==digest,path
        offsets = Path('/home/erikg/c14_dense_bytes/pile_offsets.jsonl')
        assert sha(offsets)==audit['offsets_sha256']
        def expected():
            with offsets.open() as stream:
                for line in stream:
                    d=json.loads(line)
                    if d['split']!='train':continue
                    for at in range(0,d['byte_length']-1,128):
                        yield d['source_index'],at,min(128,d['byte_length']-1-at)
        windows=expected()
        service=[json.loads(s) for s in (CACHE/'union_service.jsonl').read_text().splitlines()]
        assert service[-1]['event']=='union_complete'
        decoder=oracle()
        total_windows=total_targets=0
        phases=[]
        for phase,arm in [('emender',audit['arms'][0]),('transformer',audit['arms'][1])]:
            directory=CACHE/('registered_'+phase)
            journal=directory/'journal.jsonl'
            journal_hash=sha(journal)
            boundary=[r for r in service if r['event']=='phase_boundary' and r['phase']==phase][-1]
            assert journal_hash==boundary['journal_sha256']
            records=[json.loads(s) for s in journal.read_text().splitlines()]
            shards=[r for r in records if r['event']=='shard_complete']
            identity=shards[0]['identity']
            assert identity['stream_audit_sha256'] in (hashlib.sha256(old).hexdigest(),sha(audit_path)), 'unrecognized historical/current stream audit'
            assert identity['selector_sha256']==sha(ROOT/'c14_branch_serve.py')
            assert identity['native_sha256']==json.loads((ROOT/'receipts/c14_assembly/fixture.json').read_text())['xsa_sha256']
            assert identity['runner_sha256']==sha(ROOT/('c14_registered_cache.py' if phase=='emender' else 'c14_union_service.py'))
            assert {r['file'] for r in shards}=={p.name for p in directory.glob('*.sparse')}
            nw=nt=size=0
            first=last=None
            for i,r in enumerate(shards):
                assert r['identity']==identity and r['first_window_index']==nw
                data=(directory/r['file']).read_bytes()
                assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
                assert r['full_compact_row_parity'] is True and len(r['slow_samples'])==3
                ids,lengths,values=decode_shard(data,r['windows'],decoder)
                assert np.isfinite(values).all() and int(lengths.sum())==r['positions']
                for (doc,at),length in zip(ids.tolist(),lengths.tolist()):
                    actual=(doc,at,length)
                    assert actual==next(windows), (total_windows+nw,actual)
                    first=first or actual
                    last=actual
                assert ids[0].tolist()==r['first_identity'] and ids[-1].tolist()==r['last_identity']
                nw+=r['windows'];nt+=r['positions'];size+=len(data)
                if i%1000==0:print(json.dumps(dict(phase=phase,shards=i,windows=nw,targets=nt,seconds=time.monotonic()-started)),flush=True)
            assert records[-1]['event']=='phase_complete' and records[-1]['status']=='complete'
            assert (records[-1]['windows'],records[-1]['positions'])==(nw,nt)
            assert size==boundary['cache_bytes']
            total_windows+=nw;total_targets+=nt
            assert (total_windows,total_targets)==(arm['windows'],arm['targets'])
            phases.append(dict(phase=phase,windows=nw,targets=nt,cache_bytes=size,journal_sha256=journal_hash,shards=len(shards),first_window=first,last_window=last,boundary=boundary))
        assert last==(179298,17792,128)
        result.update(status='passed',training_authorized=True,windows=total_windows,targets=total_targets,phases=phases,
            stream_audit_sha256=sha(audit_path),historical_stream_audit_sha256=hashlib.sha256(old).hexdigest(),
            offsets_sha256=sha(offsets),union_service_sha256=sha(CACHE/'union_service.jsonl'),
            explanation='746495031 is continuation-only: 225880702 shared Emender-prefix targets + 746495031 continuation targets = 972375733. 1785536 + 5900112 = 7685648 registered windows.',
            all_shard_hashes=True,all_windows_source_order_identity=True,all_entropy_float32_identity=True,
            extraction_parity='Recorded full compact row parity and three slow samples per shard; no fresh extractor queries',
            decoder_sha256=sha(ROOT/'c14_fast_features.c'))
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['seconds']=time.monotonic()-started
        result['no_training_started']=True
        (OUT/'cache_reconciliation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':run()
