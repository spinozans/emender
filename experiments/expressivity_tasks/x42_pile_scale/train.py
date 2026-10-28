"""Single leased device, fp32 master AdamW with bf16 autocast. No model sharding."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Stream, digest
from .model import make_model, memory
from .checkpoint import save as save_checkpoint, restore as restore_checkpoint


def gate(path, cfg):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('verdict') not in ('pure','hybrid') or not receipt.get('approved_for_x42'):
        raise RuntimeError('X40 verdict has not authorized pilot/search')
    if receipt['verdict'] != cfg['architecture']:
        raise RuntimeError('architecture differs from X40 verdict')
    if not receipt.get('source_receipt_sha256'):
        raise RuntimeError('X40 source attestation missing')
    source = Path(receipt['source_receipt_path'])
    if digest(source) != receipt['source_receipt_sha256']:
        raise RuntimeError('X40 source attestation mismatch')


def train(cfg, data, output, verdict, resume=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    gate(verdict,cfg)
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6'):
        raise RuntimeError('run via run_leased.sh')
    out = Path(output)
    out.mkdir(parents=True,exist_ok=resume)
    manifest = json.loads((Path(data)/'manifest.json').read_text())
    for split in ('train','valid'):
        if digest(Path(data)/(split+'.bin')) != manifest['split_sha256'][split]:
            raise RuntimeError('data hash mismatch')
    if not manifest['complete']:
        raise RuntimeError('sample/incomplete dataset cannot support scientific training')
    torch.manual_seed(cfg['seed'])
    if 'x42_pile_scale.emender' in cfg.get('model_factory',''):
        from ..x34_lm_pilot.model import parity_gate
        parity_gate()
    memory_receipt = memory(cfg)
    model = make_model(cfg).cuda()
    if cfg.get('search_constants'):
        if set(getattr(model, 'consumed_search_constants', [])) != set(cfg['search_constants']):
            raise RuntimeError('model must attest consumption of every search constant')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(.9,.95),weight_decay=.1,fused=True)
    stream = Stream(Path(data)/'train.bin',cfg['context'],cfg['seed'])
    valid = Stream(Path(data)/'valid.bin',cfg['context'],cfg['seed'])
    unit = cfg['microbatch']*cfg['context']*cfg['accumulation']
    steps = math.ceil(cfg['tokens']/unit)
    if steps*unit > stream.n*cfg['context']:
        raise RuntimeError('insufficient unique packed tokens; no silent repeats')
    registration = dict(config=cfg,memory=memory_receipt,data=manifest,
                        verdict=json.loads(Path(verdict).read_text()),lease=os.environ['GPU_LEASE_HELD'],
                        target_tokens=steps*unit,steps=steps)
    # Lease can change across resume; scientific registration cannot.
    checkpoint_registration={k:v for k,v in registration.items() if k!='lease'}
    start = started
    cursor = first_step = 0
    rows=[]
    if resume:
        previous=json.loads((out/'registration.json').read_text())
        if {k:v for k,v in previous.items() if k!='lease'} != checkpoint_registration:
            raise ValueError('resume registration mismatch')
        state=restore_checkpoint(out/'latest.ckpt',model,optimizer,checkpoint_registration)
        cursor,first_step,rows=state['cursor'],state['step'],state['rows']
        if first_step != len(rows) or cursor != first_step*cfg['accumulation']*cfg['microbatch'] or not 0 <= first_step <= steps:
            raise ValueError('checkpoint step/cursor mismatch')
        (out/'resume.json').write_text(json.dumps(dict(checkpoint_sha256=digest(out/'latest.ckpt'),
                         step=first_step,cursor=cursor,lease=os.environ['GPU_LEASE_HELD']),indent=2))
        start-=state['elapsed']
    else:
        (out/'registration.json').write_text(json.dumps(registration,indent=2))
    def batch(s, index):
        z = torch.from_numpy(np.stack([s.window(index+i) for i in range(cfg['microbatch'])])).cuda()
        return z[:,:-1],z[:,1:]
    with open(out/'curve.jsonl','w',buffering=1) as log:
        for row in rows:
            log.write(json.dumps(row)+'\n')
        for step in range(first_step,steps):
            if time.monotonic()-start >= cfg['wall_seconds']:
                raise RuntimeError('wall budget exhausted before completion')
            optimizer.zero_grad(set_to_none=True)
            model.train()
            for _ in range(cfg['accumulation']):
                x,y = batch(stream,cursor)
                cursor += cfg['microbatch']
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x).flatten(0,1).float(),y.flatten())
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                aux = model.auxiliary_loss() if hasattr(model, 'auxiliary_loss') else loss.new_zeros(())
                ((loss+aux)/cfg['accumulation']).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            lr = cfg['lr']*min(1.,(step+1)/cfg['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            row = dict(step=step+1,tokens=(step+1)*unit,loss=loss.item(),grad_norm=norm.item(),seconds=time.monotonic()-start)
            if (step+1)%cfg['eval_every'] == 0 or step+1 == steps:
                model.eval()
                losses=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i in range(cfg['eval_batches']):
                        x,y = batch(valid,i*cfg['microbatch'])
                        losses.append(F.cross_entropy(model(x).flatten(0,1).float(),y.flatten()).item())
                row['valid_nll'] = sum(losses)/len(losses)
                row['perplexity'] = math.exp(row['valid_nll'])
            log.write(json.dumps(row)+'\n')
            rows.append(row)
            if (step+1)%cfg.get('checkpoint_every',cfg['eval_every'])==0 or step+1==steps:
                save_checkpoint(out/'latest.ckpt',model,optimizer,checkpoint_registration,
                                step+1,cursor,rows,time.monotonic()-start)
    row=rows[-1]
    result = dict(status='completed',**row,peak_allocated=torch.cuda.max_memory_allocated())
    if cfg.get('search_constants'):
        if not hasattr(model, 'occupancy_profile'):
            raise RuntimeError('search model requires occupancy_profile interface')
        with torch.no_grad():
            profile = model.occupancy_profile(valid, lengths=[512,1024,2048,4096])
        if not all(k in profile for k in ('occupancy_length_slope','pressure','rich_capacity')):
            raise RuntimeError('incomplete occupancy/pressure profile')
        if not profile['rich_capacity']:
            raise RuntimeError('capacity-capped profile cannot score compression')
        result.update(profile)
    (out/'failure.json').unlink(missing_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('config','data','output','verdict'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if Path(a.output).exists() and not a.resume:
        p.error('output must be a new directory')
    try:
        train(json.loads(Path(a.config).read_text()),a.data,a.output,a.verdict,resume=a.resume)
    except Exception as exc:
        out=Path(a.output)
        if out.exists():
            (out/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(exc))))
        raise
