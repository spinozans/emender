"""Synthetic engineering checks only; never a pilot or scientific endpoint."""
import argparse
import gc
import json
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import make_model, memory


def replay_nodes(tensor):
    seen=set(); pending=[tensor.grad_fn]
    while pending:
        node=pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        pending.extend(n for n,_ in node.next_functions)
    return sum(type(n).__name__=='ReplayBackward' for n in seen)


def validate(cfg, accumulation):
    torch.manual_seed(4203)
    model=make_model(cfg).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=.1,fused=True)
    ids=torch.randint(0,50257,(1,cfg['context']+1),device='cuda')
    rows=[]
    # First update warms kernels/allocates moments; second measures steady work.
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(accumulation):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(ids[:,:-1])
                loss=F.cross_entropy(logits.flatten(0,1).float(),ids[:,1:].flatten())
                aux=model.auxiliary_loss() if hasattr(model,'auxiliary_loss') else loss.new_zeros(())
            nodes=replay_nodes(logits)
            assert nodes==4*len(cfg.get('lane_positions',[])), nodes
            assert torch.isfinite(logits).all() and torch.isfinite(loss+aux)
            ((loss+aux)/accumulation).backward()
            del logits,aux
        gradients={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
        missing=[n for n,g in gradients.items() if g is None]
        # Geometry/key banks intentionally frozen; every trainable tensor must participate.
        assert not missing, missing
        assert all(torch.isfinite(g).all() for g in gradients.values())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        rows.append(dict(update=update+1,loss=loss.item(),grad_norm=norm.item(),replay_nodes=nodes,
                         seconds=seconds,tokens_per_second=cfg['context']*accumulation/seconds,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del gradients
    return dict(status='passed',config=cfg,measured_accumulation=accumulation,
                registered_accumulation=cfg['accumulation'],analytical=memory(cfg),updates=rows,
                device=torch.cuda.get_device_name(),physical_card=os.environ['GPU_LEASE_HELD'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--output',required=True); a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
    root=Path(__file__).parent
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    failed=False
    for scale in ('pilot','final'):
        for arm in ('','_pure','_hybrid'):
            name=scale+arm
            cfg=json.loads((root/'configs'/f'{name}.json').read_text())
            try:
                receipt=validate(cfg,a.accumulation)
            except Exception as exc:
                failed=True; receipt=dict(status='failed',config=cfg,error=repr(exc))
            (out/f'{name}.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(name,receipt['status'],receipt.get('error',''),flush=True)
            gc.collect();torch.cuda.empty_cache()
    if failed:
        raise SystemExit(1)

if __name__=='__main__':
    main()
