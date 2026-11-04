"""Registered B1 training. Requires receipted gates; no live array queries."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from .c14_b_model import StoreIntegrated
from .c14_b_oracle import oracle_shard
from .c14_documents import sha
from .c14_dense_train import OFFSETS,tensors
from .c14_fast_features import load_decoder
from .validate_c14_b import audit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_b'
CACHE=Path('/home/erikg/c14_consultation_features/registered_emender')
CHECKPOINT=Path('/home/erikg/c14_dense_bytes/emender_c14_b1_resume.pt')
CODE=('c14_b_store.py','c14_b_model.py','c14_b_native.py','c14_b_oracle.py','c14_b_train.py')


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verified_batches(raw):
    rows=[]
    for line in OFFSETS.open():
        doc=json.loads(line)
        if doc['split']!='train':continue
        offset=doc['byte_offset'];length=doc['byte_length']
        if hashlib.sha256(raw[offset:offset+length]).hexdigest()!=doc['sha256']:
            raise ValueError('raw document hash differs from corpus receipt')
        for start in range(0,length-1,128):
            n=min(128,length-1-start)
            rows.append((doc['source_index'],start,raw[offset+start:offset+start+n+1]))
            if len(rows)==16:
                yield rows;rows=[]
    if rows:yield rows


def preflight():
    audit()
    gate=json.loads((OUT/'launch_gates.json').read_text())
    assert gate['status']=='passed' and gate['launch_ready']
    for name,digest in gate['code_sha256'].items():assert sha(ROOT/name)==digest,name
    p=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(OFFSETS)==p['offsets_sha256']
    source=Path(p['source']);st=source.stat()
    assert (st.st_size,st.st_ino,st.st_mtime_ns)==(p['source_size'],p['source_inode'],p['source_mtime_ns'])
    expected='3ae1aadac6688019d7c7195ce80e7163a7483ca456a8b67adda006f4c79895cb'
    assert sha(CACHE/'journal.jsonl')==expected
    shards=[json.loads(s) for s in (CACHE/'journal.jsonl').read_text().splitlines() if json.loads(s)['event']=='shard_complete']
    windows=targets=0
    for r in shards:
        assert r['first_window_index']==windows
        path=CACHE/r['file']
        assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
        windows+=r['windows'];targets+=r['positions']
    assert (windows,targets)==(1785536,225880702)
    return source,shards,gate


def accumulate(total,layers):
    for layer,heads in enumerate(layers):
        for row in heads:
            key=f'{layer}/{row["head"]}'
            agg=total.setdefault(key,dict(counts={},peak_occupancy=0,peak_array=0,peak_stream=0,sampled_occupancy=[]))
            for name in ('APPEND','POP','DECAY','CORRECT','MERGE','stale_retirements','natural_retirements',
                         'capacity_merges','duplicate_merges','array_requests','refused_insertions',
                         'array_read_opportunities','array_slots_found','array_slots_read','reads'):
                agg['counts'][name]=agg['counts'].get(name,0)+row[name]
            for name in ('peak_occupancy','peak_array','peak_stream'):agg[name]=max(agg[name],row[name])
            agg['final_array']=row['final_array'];agg['final_stream']=row['final_stream'];agg['budget']=row['budget']
            if len(agg['sampled_occupancy'])<512:agg['sampled_occupancy'].append(row['occupancy'])
            n=agg['counts']['array_read_opportunities']
            agg['array_read_hit_rate']=agg['counts']['array_slots_read']/n if n else None
            assert agg['peak_occupancy']<=agg['budget']


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    wall=time.monotonic();source,shards,gate=preflight()
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=StoreIntegrated().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(3401)
    decoder=load_decoder();raw=np.memmap(source,dtype=np.uint8,mode='r')
    steps=targets=cursor=kept=seen=0;prior=0.;telemetry={};curve=[]
    config=dict(seed=3401,batch=16,context=128,budget_active_seconds=2700,precision='BF16',
        optimizer='AdamW lr=3e-4 decay=.01 clip=1',code_sha256={n:sha(ROOT/n) for n in CODE},
        gate_sha256=sha(OUT/'launch_gates.json'),physical_gpu=os.environ['GPU_LEASE_HELD'],
        cache_journal_sha256=sha(CACHE/'journal.jsonl'),source_receipt_sha256=sha(ROOT/'receipts/c14_dense/pile_span.json'),
        checkpoint=str(CHECKPOINT),dropout='independent keep=.5 over all 260 scalars, no rescaling',
        timer='source/cache read, decode, assembly, transfer, forward, backward and update included; checkpoint save/hash excluded')
    if args.resume:
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        assert saved['config']['code_sha256']==config['code_sha256']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['dropout_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        steps=saved['steps'];targets=saved['targets'];cursor=saved['next_window'];prior=saved['active_seconds']
        telemetry=saved['telemetry'];curve=saved['curve'];kept=saved['kept'];seen=saved['seen']
    else:
        assert not CHECKPOINT.exists() and not (OUT/'training.json').exists(),'refuse overwrite; use --resume'
    row_iter=iter(verified_batches(raw))
    for _ in range(cursor//16):next(row_iter)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write('training_config.json',config)
    initialization=time.monotonic()-wall;excluded=[];excluded_seconds=0.
    torch.cuda.synchronize();start=time.monotonic()
    def active():return prior+time.monotonic()-start-excluded_seconds
    def save(status):
        nonlocal excluded_seconds
        torch.cuda.synchronize();elapsed=active();at=time.monotonic()
        saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),dropout_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),next_window=cursor,steps=steps,
            targets=targets,active_seconds=elapsed,telemetry=telemetry,curve=curve,kept=kept,seen=seen,config=config)
        temporary=CHECKPOINT.with_suffix('.tmp.pt');torch.save(saved,temporary);temporary.replace(CHECKPOINT)
        digest=sha(CHECKPOINT);duration=time.monotonic()-at;excluded_seconds+=duration
        excluded.append(dict(after_step=steps,seconds=duration))
        result=dict(status=status,steps=steps,actual_bytes_targets=targets,next_window_index=cursor,
            active_seconds=elapsed,wall_seconds=time.monotonic()-wall,initialization_seconds=initialization,
            excluded_checkpoint_stalls=excluded,lease_wait_seconds=float(os.environ.get('C14_B_LEASE_WAIT_SECONDS','0')),
            checkpoint=str(CHECKPOINT),checkpoint_sha256=digest,telemetry=telemetry,curve=curve,
            keep_fraction=kept/seen if seen else None,config=config,eval='not yet run')
        write('training.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('telemetry','curve','config')}),flush=True)
    status='source_exhausted_protocol_limitation'
    try:
        last_save=prior
        for r in shards:
            if r['first_window_index']+r['windows']<=cursor:continue
            data=(CACHE/r['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            identities,lengths,values,contexts,_=oracle_shard(data,r['windows'],decoder)
            for begin in range(0,len(values),16):
                index=r['first_window_index']+begin
                if index<cursor:continue
                if stopped or active()>=2700:
                    status='interrupted' if stopped else 'completed_active_budget';break
                assert index==cursor
                rows=next(row_iter);n=len(rows)
                assert n==16
                for j,(doc,offset,bb) in enumerate(rows):assert (doc,offset,len(bb)-1)==(*identities[begin+j],lengths[begin+j])
                xx,yy,mask=tensors(rows)
                ff=torch.as_tensor(values[begin:begin+n],device='cuda').clone()
                keep=torch.rand((n,128),device='cuda',generator=rng)>=.5
                ff*=keep[...,None];kept+=int((keep&mask).sum());seen+=int(mask.sum())
                ll=torch.as_tensor(contexts[begin:begin+n],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(xx,ff,ll,windows=[(doc,offset) for doc,offset,_ in rows],valid_lengths=lengths[begin:begin+n].tolist())
                    loss=-lp.gather(-1,yy[...,None])[...,0][mask].mean()
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                accumulate(telemetry,model.telemetry)
                steps+=1;cursor+=n;targets+=int(mask.sum());torch.cuda.synchronize()
                curve.append(dict(step=steps,active_seconds=active(),targets=targets,nll=float(loss.detach())))
                print(json.dumps(curve[-1]),flush=True)
                if active()-last_save>=120:save('running');last_save=active()
            else:continue
            break
        save(status)
    except BaseException as error:
        write('training_failure.json',dict(error=repr(error),steps=steps,targets=targets,next_window=cursor))
        save('failed');raise


if __name__=='__main__':main()
