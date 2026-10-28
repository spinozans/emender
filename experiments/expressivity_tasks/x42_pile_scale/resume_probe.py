"""Process-level SIGKILL/resume regression using the production checkpoint helpers."""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .checkpoint import save,restore
from .model import make_model


def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):
                h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def worker(cfg,output,device,resume,pause):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(4203)
    random.seed(4203);np.random.seed(4203)
    model=make_model(cfg).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=.1,fused=device=='cuda')
    first=cursor=0; rows=[]
    checkpoint=output/'latest.ckpt'
    if resume:
        state=restore(checkpoint,model,optimizer,cfg)
        first,cursor,rows=state['step'],state['cursor'],state['rows']
    for step in range(first,3):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for _ in range(2):
            ids=torch.randint(0,50257,(1,cfg['context']+1),device=device)
            cursor+=1
            with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                loss=F.cross_entropy(model(ids[:,:-1]).flatten(0,1).float(),ids[:,1:].flatten())
                aux=model.auxiliary_loss() if hasattr(model,'auxiliary_loss') else loss.new_zeros(())
            ((loss+aux)/2).backward(); losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group['lr']=cfg['lr']*min(1.,(step+1)/cfg.get('warmup_steps',1))
        optimizer.step()
        rows.append(dict(step=step+1,cursor=cursor,losses=losses,
                         python_sample=random.random(),numpy_sample=float(np.random.random()),
                         model=state_hash(model.state_dict()),optimizer=state_hash(optimizer.state_dict()),
                         torch_rng=state_hash(torch.get_rng_state()),cuda_rng=state_hash(torch.cuda.get_rng_state() if device=='cuda' else None)))
        if step==0 or step==2:
            save(checkpoint,model,optimizer,cfg,step+1,cursor,rows,0.)
        if step==0 and pause:
            (output/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),step=1,cursor=cursor)))
            signal.pause()
    (output/'result.json').write_text(json.dumps(rows,indent=2)+'\n')


def roundtrip(cfg,device='cpu'):
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if device=='cpu':
        env['CUDA_VISIBLE_DEVICES']=''
    with tempfile.TemporaryDirectory(prefix='x42_resume_') as directory:
        root=Path(directory); config=root/'config.json';config.write_text(json.dumps(cfg))
        reference=root/'reference'; interrupted=root/'interrupted'; reference.mkdir();interrupted.mkdir()
        base=[sys.executable,'-m','experiments.expressivity_tasks.x42_pile_scale.resume_probe','--worker','--config',str(config),'--device',device]
        subprocess.run(base+['--output',str(reference)],env=env,check=True)
        proc=subprocess.Popen(base+['--output',str(interrupted),'--pause'],env=env)
        try:
            deadline=time.monotonic()+600
            while not (interrupted/'ready.json').exists():
                if proc.poll() is not None:
                    raise RuntimeError(f'interrupted worker exited {proc.returncode}')
                if time.monotonic()>deadline:
                    raise TimeoutError('checkpoint ready timeout')
                time.sleep(.1)
            killed=json.loads((interrupted/'ready.json').read_text())
            proc.kill(); code=proc.wait(timeout=30)
            assert code==-signal.SIGKILL
        finally:
            if proc.poll() is None:
                proc.kill();proc.wait()
        subprocess.run(base+['--output',str(interrupted),'--resume'],env=env,check=True)
        expected=json.loads((reference/'result.json').read_text())
        actual=json.loads((interrupted/'result.json').read_text())
        assert actual==expected,'continuation differs after SIGKILL/resume'
        return dict(status='passed',config=cfg,device=device,kill_signal='SIGKILL',kill_exit_code=code,
                    killed_checkpoint=killed,bit_identical=True,compared='losses, model, optimizer, torch/CUDA RNG, Python/NumPy samples, cursor at every update',
                    measured_accumulation=2,updates=actual,checkpoint_files='temporary .ckpt files removed after comparison')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true');p.add_argument('--config');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--pause',action='store_true')
    a=p.parse_args()
    if a.worker:
        worker(json.loads(Path(a.config).read_text()),Path(a.output),a.device,a.resume,a.pause)
    else:
        assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
        root=Path(__file__).parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        for name in ('pilot','pilot_pure','pilot_hybrid'):
            cfg=json.loads((root/'configs'/f'{name}.json').read_text())
            try:
                receipt=roundtrip(cfg,a.device)
            except Exception as exc:
                receipt=dict(status='failed',error=repr(exc));
                (out/f'{name}.json').write_text(json.dumps(receipt,indent=2)+'\n');raise
            (out/f'{name}.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(name,'bit-identical SIGKILL/resume passed',flush=True)

if __name__=='__main__':
    main()
