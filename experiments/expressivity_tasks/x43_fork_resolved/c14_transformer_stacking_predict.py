"""Frozen C14 paired evaluation. No training or union-service mutations.

Atomic per-shard resume receipts retain float64 sums and exact denominators.
The primary table is final_test; calibration and combined rows remain explicit.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from .c11 import Boosted
from .c14_documents import sha
from .c14_fast_features import decode_shard, load_decoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_transformer_stacking'
PRED = Path('/home/erikg/c14_transformer_stacking_predictions.npy')
CACHE = Path('/home/erikg/c14_consultation_features/registered_eval')
DATA = Path('/home/erikg/c14_dense_bytes')
EXPECTED = (3431242, 434204084, 44953751774)
JOURNAL = '00e5c17a75180a367162cef6be5046a721b3d9d58ec273ef2e4a90f68522ea3b'
WINDOWS = '2ab51ce59ad94010d78f6817f29fa06fc5ee64feea766376ff0e35347f196d4d'


def write(name, value):
    p = OUT/name
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def oracle():
    """The c14_verify_sum C14_REQUIRE_IDENTICAL oracle on unseen eval shards."""
    source = ROOT/'c14_fast_features.c'
    library = Path('/home/erikg/c14_consultation_features/native_decode')/(sha(source)+'.sumcheck.so')
    subprocess.run(['cc','-O3','-fno-fast-math','-ffp-contract=off',
                    '-DC14_REQUIRE_IDENTICAL','-shared','-fPIC',str(source),'-lm','-o',str(library)], check=True)
    decode = ctypes.CDLL(str(library)).c14_decode
    decode.argtypes = [ctypes.c_char_p,ctypes.c_size_t,ctypes.c_void_p,
                       ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    decode.restype = ctypes.c_int
    return decode


def audit():
    assert sha(CACHE/'journal.jsonl') == JOURNAL
    records = rows(CACHE/'journal.jsonl')
    shards = [r for r in records if r['event']=='shard_complete']
    identity = shards[0]['identity']
    stream = ROOT/'receipts/c14_registered_scale/stream_audit.json'
    a = json.loads(stream.read_text())
    # Extraction began before the attended stopping-rule amendment. Preserve
    # that historical identity, and prove the frozen stream fields unchanged.
    historical = subprocess.check_output(['git','show',
        'f29462c3abcdc23bc974efe2d3510cf55577d7ff:'+str(stream.relative_to(ROOT.parents[2]))])
    assert hashlib.sha256(historical).hexdigest()==identity['stream_audit_sha256']
    original = json.loads(historical)
    for key in ('eval','arms','offsets_sha256','source','receipt_hashes'):
        assert original[key]==a[key],key
    for path, digest in a['receipt_hashes'].items():
        assert sha(ROOT.parents[2]/path)==digest, path
    construction = json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    assert sha(Path(construction['windows_path']))==WINDOWS==a['eval']['windows_sha256']
    assert sha(DATA/'pile_offsets.jsonl')==construction['offsets_sha256']
    check = oracle()
    nw = positions = size = 0
    start = time.monotonic()
    for i, r in enumerate(shards):
        assert r['first_window_index']==nw and r['identity']==identity
        p = CACHE/r['file']; data = p.read_bytes()
        assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
        assert r['full_compact_row_parity'] is True and len(r['slow_samples'])==3
        _, lengths, values = decode_shard(data, r['windows'], check)
        assert int(lengths.sum())==r['positions']
        assert np.isfinite(values).all()
        nw+=r['windows']; positions+=r['positions']; size+=len(data)
        if i%2000==0: print('audit',i,nw,positions,flush=True)
    assert {r['file'] for r in shards}=={p.name for p in CACHE.glob('*.sparse')}
    assert (nw,positions,size)==EXPECTED
    assert records[-1]['event']=='phase_complete' and records[-1]['status']=='complete'
    boundary = [r for r in rows(CACHE.parent/'union_service.jsonl')
                if r['event']=='phase_boundary' and r['phase']=='eval'][-1]
    assert boundary['journal_sha256']==JOURNAL and boundary['cache_bytes']==size
    result = dict(status='passed',shards=len(shards),windows=nw,targets=positions,
                  cache_bytes=size,journal_sha256=JOURNAL,windows_sha256=WINDOWS,
                  boundary=boundary,identity=identity,seconds=time.monotonic()-start,
                  parity_attestations_green=True,all_eval_entropy_float32_identity=True,
                  oracle='c14_verify_sum C14_REQUIRE_IDENTICAL, exhaustive over every eval position',
                  decoder_sha256=sha(ROOT/'c14_fast_features.c'),
                  native_extractor_oracle_rerun=False,
                  scope='all shard hashes, contiguity, cardinality, recorded native parity, full compensated/sequential decoder entropy identity; no new extraction')
    write('cache_validation.json',result)
    return result


def strata(features, targets, lengths):
    real = np.arange(128)[None,:] < lengths[:,None]
    support = features[:,:,256]>0
    q = np.take_along_axis(features[:,:,:256],targets[:,:,None],axis=2)[:,:,0]
    masks = dict(total=real,window_reset_1_16=real & (np.arange(128)<16),
                 window_interior_17_128=real & (np.arange(128)>=16),
                 wikipedia_support=real & support,empty_support=real & ~support,
                 supported_target=real & (q>0),nonempty_zero_target=real & support & (q==0),
                 zero_target=real & (q==0),fork=real & (features[:,:,258]>0),
                 # The phase-1 registration defines branch agreement as an
                 # observed positive target count, not an argmax prediction.
                 agreement=real & (q>0),
                 argmax_agreement=real & support & (features[:,:,:256].argmax(axis=2)==targets))
    return masks


def add_sums(sums, arm, half, losses, masks, selection):
    for name, mask in masks.items():
        selected = losses[mask & selection[:,None]]
        key = '/'.join((arm,half,name))
        old = sums.setdefault(key,[0.,0])
        old[0]+=float(selected.astype(np.float64).sum()); old[1]+=int(selected.size)


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-shards',type=int,default=0)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.audit_only:
        print(json.dumps(audit()),flush=True); return
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    checked=json.loads((ROOT/'receipts/c14_paired_eval/cache_validation.json').read_text())
    assert checked['status']=='passed' and checked['journal_sha256']==sha(CACHE/'journal.jsonl')==JOURNAL
    assert checked['all_eval_entropy_float32_identity']
    assert checked['decoder_sha256']==sha(ROOT/'c14_fast_features.c')
    construction=json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    assert sha(Path(construction['windows_path']))==WINDOWS
    source_receipt=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(DATA/'pile_offsets.jsonl')==source_receipt['offsets_sha256']
    source=Path(source_receipt['source']); st=source.stat()
    assert (st.st_size,st.st_ino,st.st_mtime_ns)==(source_receipt['source_size'],source_receipt['source_inode'],source_receipt['source_mtime_ns'])
    raw=np.memmap(source,dtype=np.uint8,mode='r')
    docs={}
    with (DATA/'pile_offsets.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            if d['split']=='valid': docs[d['source_index']]=d
    dense=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models'][1]
    assert dense['arm']=='transformer'
    configs={}; checkpoints={}
    checkpoints['dense']=(DATA/'transformer_dense.pt',dense['checkpoint_sha256'])
    for label,folder in [('v2','c14_transformer')]:
        r=json.loads((ROOT/'receipts'/folder/'training.json').read_text())
        checkpoints[label]=(Path(r['checkpoint']),r['checkpoint_sha256'])
        configs[label]={k:r[k] for k in ('steps','actual_bytes_targets','active_seconds')}
    configs['dense']={k:dense[k] for k in ('steps','targets','active_seconds')}
    shapes=json.loads((ROOT/'receipts/c14_byte/model_shapes.json').read_text())
    for path,digest in shapes['sources'].items():
        assert sha(ROOT.parents[2]/path)==digest,path
    torch.set_num_threads(1)
    models={}
    for label,(p,h) in checkpoints.items():
        assert sha(p)==h
        saved=torch.load(p,map_location='cpu',weights_only=False)
        model=Boosted('transformer',2000324)
        model.load_state_dict(saved if label=='dense' else saved['model'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        models[label]=model.cuda().eval()
    config=dict(interpreter=sys.executable,code_sha256=sha(__file__),
                decoder_sha256=sha(ROOT/'c14_fast_features.c'),journal_sha256=JOURNAL,
                windows_sha256=WINDOWS,precision='BF16 autocast; losses float32; sums float64',
                batch='one immutable shard, at most 128 independent windows',
                telemetry='Transformer eval; no gradients or feature dropout; identical to frozen paired transformer runner',
                checkpoint_hashes={k:h for k,(_,h) in checkpoints.items()},training=configs,
                deterministic_algorithms=True,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                modes=['dense','v2_on'],
                dropout='disabled during eval; off zeros all 260 scalars at every position',
                reset='independent forward per 128-byte window; tails loss-masked',
                primary='v2_on; final_test half; calibration and all targets reported separately',
                strata='support=nonterminal Wikipedia outgoing count>0; supported_target=agreement=q(target)>0 per phase-1 definition; zero_target=q(target)==0 includes empty; supplemental argmax_agreement=target equals smallest argmax byte',
                gpu=os.environ['GPU_LEASE_HELD'],torch_version=torch.__version__)
    if args.resume:
        state=json.loads((OUT/'resume.json').read_text())
        assert state['config']==config,'resume config changed'
    else:
        assert not (OUT/'resume.json').exists(),'use --resume'
        state=dict(config=config,next_shard=0,windows=0,targets=0,sums={},curve=[],
                   elapsed_seconds=0.,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    shards=[r for r in rows(CACHE/'journal.jsonl') if r['event']=='shard_complete']
    pred=np.lib.format.open_memmap(PRED,mode='r+' if args.resume else 'w+',dtype=np.float32,shape=(EXPECTED[1],10))
    decoder=load_decoder(); stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    start=time.monotonic(); prior=state['elapsed_seconds']; verified=set()
    def save(status):
        pred.flush()
        state['status']=status; state['elapsed_seconds']=prior+time.monotonic()-start
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write('resume.json',state)
    with Path(construction['windows_path']).open() as frozen, torch.inference_mode():
        for _ in range(state['windows']): next(frozen)
        for index in range(state['next_shard'],len(shards)):
            r=shards[index]; data=(CACHE/r['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            ids,lengths,features=decode_shard(data,r['windows'],decoder)
            assert r['first_window_index']==state['windows']
            x=np.zeros((len(ids),128),np.int64); y=np.zeros_like(x); halves=[]
            for j,((doc,at),length) in enumerate(zip(ids.tolist(),lengths.tolist())):
                expected=json.loads(next(frozen))
                assert (doc,at,length)==(expected['document'],expected['start'],expected['targets'])
                d=docs[doc]
                if doc not in verified:
                    assert sha_bytes(raw[d['byte_offset']:d['byte_offset']+d['byte_length']])==d['sha256']
                    verified.add(doc)
                assert at+length<d['byte_length']
                offset=d['byte_offset']+at; b=raw[offset:offset+length+1]
                x[j,:length]=b[:-1]; y[j,:length]=b[1:]; halves.append(expected['half'])
            masks=strata(features,y,lengths)
            halves=np.array(halves)
            xx=torch.as_tensor(x,device='cuda'); yy=torch.as_tensor(y,device='cuda')
            ff=torch.as_tensor(features,device='cuda'); zz=torch.zeros_like(ff)
            ll=torch.zeros(xx.shape,device='cuda')
            outputs={}
            for arm in config['modes']:
                label=arm.split('_')[0]
                tf=ff if arm.endswith('_on') else zz
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=models[label](xx,tf,ll)
                losses=(-lp.gather(-1,yy[...,None])[...,0]).float().cpu().numpy()
                if not np.isfinite(losses).all():
                    write('nonfinite_failure.json',dict(status='failed',label=label,shard=index,
                        nonfinite=int((~np.isfinite(losses)).sum()),
                        first_indices=np.argwhere(~np.isfinite(losses))[:20].tolist(),
                        windows=state['windows'],code_sha256=sha(__file__)))
                    raise ValueError(f'nonfinite inference loss: {label}, shard {index}')
                outputs[arm]=(losses,lp.float().exp().amax(-1).cpu().numpy())
                for half in ('all','calibration','final_test'):
                    selection=np.ones(len(ids),bool) if half=='all' else halves==half
                    add_sums(state['sums'],arm,half,losses,masks,selection)
            real=masks['total']; count=int(real.sum()); lo=state['targets']
            q=np.take_along_axis(features[:,:,:256],y[:,:,None],axis=2)[:,:,0]
            batch=np.column_stack([outputs['dense'][0][real],outputs['v2_on'][0][real],
                outputs['dense'][1][real],outputs['v2_on'][1][real],q[real],
                features[:,:,256][real],features[:,:,259][real],features[:,:,258][real],
                np.broadcast_to((halves=='final_test')[:,None],real.shape)[real],
                np.broadcast_to(np.arange(128)[None,:],real.shape)[real]])
            assert np.isfinite(batch).all()
            pred[lo:lo+count]=batch
            state['next_shard']=index+1; state['windows']+=len(ids); state['targets']+=int(lengths.sum())
            if index%64==0 or index+1==len(shards) or stopped:
                state['curve'].append(dict(shard=index+1,windows=state['windows'],targets=state['targets'],
                    seconds=prior+time.monotonic()-start,
                    final_nll={arm:mean(state['sums'].get(arm+'/final_test/total',[0.,0])) for arm in config['modes']}))
                save('running'); print(json.dumps(state['curve'][-1]),flush=True)
            if stopped or (args.max_shards and index+1>=args.max_shards):
                save('paused'); return
        assert frozen.read()==''
    assert (state['windows'],state['targets'])==EXPECTED[:2]
    save('complete')
    write('prediction_identity.json',dict(path=str(PRED),sha256=sha(PRED),columns=['dense_loss','v2_loss','dense_confidence','v2_confidence','q_target','log1p_support','log1p_length','fork','final_test','position'],shape=list(pred.shape)))
    write('table.json',{key:dict(total_nll=s,targets=n,nll=s/n if n else None) for key,(s,n) in state['sums'].items()})
    print('COMPLETE',state['targets'],flush=True)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def mean(pair):
    return pair[0]/pair[1] if pair[1] else None


if __name__=='__main__': main()
