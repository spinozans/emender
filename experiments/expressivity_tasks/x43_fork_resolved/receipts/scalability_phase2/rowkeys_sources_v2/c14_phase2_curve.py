"""Restricted paired 2M-target certification slice; no production training launch.

Uses the production document/cache assembly and update path. Both arms start
from the immutable trained checkpoint; all updates are disposable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
import subprocess
import sys
import numpy as np
import torch
from . import c14_b_control_native as native
from .c14_b_unbudgeted_profile import configure, bits
from . import c14_b_train as train
from .c14_phase2_staged import install_gluon, install_rowkeys
from .c14_launch_shape import install as install_shape
from .c14_fast_features import load_decoder


def digest(value):
    return hashlib.sha256(json.dumps(bits(value),sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gates',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--variant',choices=('gluon','rowkeys'),default='rowkeys')
    a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    subprocess.run([sys.executable,'scripts/registration_guard.py','check',str(a.registration)],check=True)
    assert not a.output.exists()
    for gate in (f'{name}_{a.variant}' for name in ('model','gradient','battery','oracle','shape')):
        assert json.loads((a.gates/f'{gate}_terminal.json').read_text())['exit_code']==0,gate
    for name,count in (('battery',105),('oracle',109)):
        text=(a.gates/f'{name}_{a.variant}_console.txt').read_text()
        assert f'{count} passed' in text and ' skipped' not in text and ' failed' not in text
    a.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(3401)
    configure()
    source,shards,_=train.preflight(check_implementation=False)
    checkpoint_hash=train.sha(train.CHECKPOINT)
    assert checkpoint_hash==json.loads((train.OUT/'HARVEST.json').read_text())['checkpoint_sha256_verified']
    saved=torch.load(train.CHECKPOINT,map_location='cpu',weights_only=False)
    original_forward,original_kernel=native.native_forward,native.kernel
    install_shape();bounded=native.native_forward
    native.native_forward=original_forward
    (install_rowkeys if a.variant=='rowkeys' else install_gluon)();staged=native.native_forward
    staged_kernel=native.kernel
    models=[];optimizers=[]
    for _ in range(2):
        model=train.StoreIntegrated().cuda().train();model.load_state_dict(saved['model'])
        models.append(model);optimizers.append(torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01))
    del saved
    rng=torch.Generator(device='cuda').manual_seed(3401)
    raw=np.memmap(source,dtype=np.uint8,mode='r');rows=iter(train.verified_batches(raw))
    decoder=load_decoder();steps=targets=cursor=0;start=time.monotonic()
    receipt=dict(status='running',scope='paired disposable training-path certification; not production training authorization',
        checkpoint_sha256=checkpoint_hash,offsets_sha256=train.sha(train.OFFSETS),
        pipeline='production verified_batches, v2 decoder, tensors, BF16, mask, clip and AdamW',
        actual_targets=0,steps=0)
    try:
        with (a.output/'steps.jsonl').open('x') as log:
            for shard in shards:
                data=(train.CACHE/shard['file']).read_bytes()
                assert hashlib.sha256(data).hexdigest()==shard['sha256']
                identities,lengths,values,contexts,_=train.oracle_shard(data,shard['windows'],decoder)
                for begin in range(0,len(values),16):
                    batch=next(rows);n=len(batch);assert n==16
                    assert shard['first_window_index']+begin==cursor
                    for j,(doc,offset,bb) in enumerate(batch):
                        assert (doc,offset,len(bb)-1)==(*identities[begin+j],lengths[begin+j])
                    xx,yy,mask=train.tensors(batch)
                    ff=torch.as_tensor(values[begin:begin+n],device='cuda').clone()
                    ff*=(torch.rand((n,128),device='cuda',generator=rng)>=.5)[...,None]
                    ll=torch.as_tensor(contexts[begin:begin+n],device='cuda')
                    fingerprints=[];durations=[]
                    for arm,(model,opt) in enumerate(zip(models,optimizers)):
                        native.native_forward=(bounded,staged)[arm]
                        native.kernel=(original_kernel,staged_kernel)[arm]
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.synchronize();at=time.monotonic()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            lp,_=model(xx,ff,ll,windows=[(d,o) for d,o,_ in batch],valid_lengths=lengths[begin:begin+n].tolist())
                            loss=-lp.gather(-1,yy[...,None])[...,0][mask].mean()
                        assert torch.isfinite(loss)
                        loss.backward()
                        grad_hash=digest({name:x.grad for name,x in model.named_parameters()})
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                        torch.cuda.synchronize();durations.append(time.monotonic()-at)
                        fingerprints.append(dict(loss=bits(loss),gradients=grad_hash,state=digest(model.state_dict()),optimizer=digest(opt.state_dict()),telemetry=digest(model.telemetry)))
                    steps+=1;cursor+=n;targets+=int(mask.sum())
                    row=dict(step=steps,targets=targets,next_window=cursor,exact=fingerprints[0]==fingerprints[1],fingerprints=fingerprints,seconds=durations)
                    log.write(json.dumps(row)+'\n');log.flush()
                    assert row['exact'],f'bitwise curve mismatch at step {steps}'
                    if steps==2:
                        # Exercise atomic checkpoint and optimizer/RNG restore in a
                        # disposable directory; never write the trained checkpoint.
                        checks=[]
                        with tempfile.TemporaryDirectory(prefix='emender-phase2-gate-') as temp:
                            for arm,(model,opt) in enumerate(zip(models,optimizers)):
                                path=Path(temp)/f'arm{arm}.pt';partial=path.with_suffix('.tmp')
                                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),dropout_rng=rng.get_state(),steps=steps,targets=targets,next_window=cursor)
                                torch.save(payload,partial);partial.replace(path)
                                loaded=torch.load(path,map_location='cpu',weights_only=False)
                                assert digest(loaded)==digest(payload)
                                model.load_state_dict(loaded['model']);opt.load_state_dict(loaded['optimizer']);rng.set_state(loaded['dropout_rng'].cpu())
                                checks.append(dict(arm=arm,sha256=train.sha(path),exact=True))
                        receipt['atomic_checkpoint_roundtrip']=checks
                    if steps%25==0:print(json.dumps(dict(step=steps,targets=targets)),flush=True)
                    if targets>=2_000_000:break
                if targets>=2_000_000:break
        assert targets>=2_000_000
        assert train.sha(train.CHECKPOINT)==checkpoint_hash
        receipt['status']='passed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        native.native_forward=original_forward;native.kernel=original_kernel
        receipt.update(actual_targets=targets,steps=steps,next_window=cursor,elapsed_seconds=time.monotonic()-start)
        (a.output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
