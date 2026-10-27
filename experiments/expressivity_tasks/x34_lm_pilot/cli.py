"""Resumable runner. All GPU use must go through run_leased.sh."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from .data import ROOT, RUNS, batch, prepare, sha
from .model import LM, parameter_receipt, parity_gate
from .probes import context_probe, depth_probe, story_probe, aggregate_depth


def write(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def save_checkpoint(path, state):
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def replay_present(loss):
    seen=set()
    pending=[loss.grad_fn]
    while pending:
        node=pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == 'ReplayBackward':
            return type(node).__name__
        pending.extend(n for n,_ in node.next_functions)
    raise AssertionError('X31 custom replay backward absent from LM loss graph')


@torch.no_grad()
def validate(model, device):
    was_training=model.training
    model.eval()
    total=0.
    metrics=[]
    for i in range(2):
        x,y=batch('valid',i)
        logits,t=model(x.to(device))
        total+=F.cross_entropy(logits.flatten(0,1),y.to(device).flatten()).item()
        metrics.append(t)
    model.train(was_training)
    return dict(nll=total/2,perplexity=math.exp(total/2),telemetry=metrics,tokens=8192)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','counts','train','probe','integration','parity'])
    p.add_argument('--arm',choices=['tape_lm','base_lm'],default='tape_lm')
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--tag',default='smoke')
    p.add_argument('--wall-seconds',type=float,default=43200)
    p.add_argument('--probe-smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.command=='prepare':
        print(json.dumps(prepare())); return
    tok=Tokenizer.from_file(str(RUNS/'tokenizer.json')) if (RUNS/'tokenizer.json').exists() else None
    vocab=tok.get_vocab_size() if tok else 32768
    if args.command=='counts':
        result=parameter_receipt(vocab)
        write(ROOT/'receipts/parameters.json',result)
        from .compute import estimate
        write(ROOT/'receipts/compute_estimate.json',estimate(vocab))
        print(json.dumps(result)); return
    if not os.environ.get('GPU_LEASE_HELD') or os.environ['GPU_LEASE_HELD'] not in ('0','1','5','6'):
        raise RuntimeError('use run_leased.sh; GPU lease required')
    if args.command=='parity':
        parity_gate()
        from ..x31_tape_cell_fused.tests.test_gradients import check_parity
        check_parity(2944, 3401, dk=64, dv=128, heads=4)
        write(ROOT/'receipts/parity.json',dict(passed=True,slots=2944,heads=4,key_dim=64,value_dim=128,atol=1e-5,rtol=1e-5))
        return
    torch.manual_seed(3401); np.random.seed(3401); random.seed(3401)
    session_start=time.monotonic()
    device='cuda'
    parity_gate()
    model=LM(args.arm,vocab).to(device)
    run=RUNS/f'{args.tag}_{args.arm}'
    run.mkdir(parents=True,exist_ok=True)
    checkpoint=run/'latest.pt'
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    step=0; elapsed=0.
    fingerprint=dict(tokenizer=sha(RUNS/'tokenizer.json') if tok else None,data=sha(ROOT/'receipts/data.json') if (ROOT/'receipts/data.json').exists() else None,
                     arm=args.arm,seed=3401,context=512,batch=8,
                     protocol=sha(ROOT/'RECEIPTS.md'),model=sha(ROOT/'model.py'),
                     data_code=sha(ROOT/'data.py'))
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False,map_location=device)
        if state['fingerprint'] != fingerprint:
            raise RuntimeError('resume data/config mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        step=state['step'];elapsed=state['elapsed']
        ledger=run/'elapsed.json'
        if ledger.exists():
            elapsed=max(elapsed,json.loads(ledger.read_text())['elapsed_seconds'])
        torch.set_rng_state(state['rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    if args.command=='train' and checkpoint.exists() and (run/'curve.jsonl').exists():
        lines=(run/'curve.jsonl').read_text().splitlines()
        kept=[]; abandoned=[]
        for line in lines:
            try:
                row=json.loads(line)
                (kept if row['step']<=step else abandoned).append(line)
            except json.JSONDecodeError:
                abandoned.append(line)
        if abandoned:
            (run/f'abandoned_curve_{time.time_ns()}.jsonl').write_text('\n'.join(abandoned)+'\n')
            (run/'curve.jsonl').write_text('\n'.join(kept)+'\n')
    if args.command=='integration':
        x=torch.randint(1,vocab,(1,17)); x,y=x[:,:-1].clone(),x[:,1:].clone()
        logits,metrics=model(x.to(device))
        loss=F.cross_entropy(logits.flatten(0,1),y.to(device).flatten())
        replay=replay_present(loss) if args.arm=='tape_lm' else None
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if args.arm=='tape_lm':
            assert model.blocks[2].cell.W_v.weight.grad.abs().sum()>0
            assert model.blocks[2].cell.W_q.weight.grad is not None
        result=dict(loss=loss.item(),telemetry=metrics,replay_node=replay,finite_gradients=True)
        write(ROOT/f'receipts/integration_{args.arm}.json',result)
        print(json.dumps(result));return
    if args.command=='probe':
        if not checkpoint.exists():
            raise RuntimeError('trained checkpoint required')
        model.eval()
        depths=[16,256] if args.probe_smoke else None
        kw={} if depths is None else dict(depths=depths)
        valid=np.memmap(RUNS/'valid.bin',mode='r',dtype=np.uint16)
        def budgeted(ids):
            if elapsed+time.monotonic()-session_start>=min(args.wall_seconds,43200):
                raise TimeoutError('cumulative GPU budget reached during evaluation')
            return model(ids)
        try:
            result=dict(status='completed',step=step,tokens=step*4096,
                stories=story_probe(budgeted,tok,RUNS/'valid.txt',device,stories=2 if args.probe_smoke else 16),
                depth=depth_probe(budgeted,tok,device,**kw),
                context=context_probe(budgeted,valid,device,
                    lengths=(32,64) if args.probe_smoke else (512,1024,2048,4096),
                    windows=1 if args.probe_smoke else 2),smoke=args.probe_smoke)
        except TimeoutError as exc:
            result=dict(status='right-censored',step=step,tokens=step*4096,error=str(exc),
                        smoke=args.probe_smoke,partial_scores_retained=False)
        finally:
            write(run/'elapsed.json',dict(elapsed_seconds=elapsed+time.monotonic()-session_start))
        if 'depth' in result:
            result['depth_bins']=aggregate_depth(result['depth'])
        result['elapsed_seconds']=elapsed+time.monotonic()-session_start
        result['budget_overshoot_seconds']=max(0.,result['elapsed_seconds']-min(args.wall_seconds,43200))
        write(ROOT/f'receipts/probe_{args.tag}_{args.arm}.json',result);return
    if not tok:
        raise RuntimeError('prepare data first')
    write(run/'preregistration.json',dict(sha256=sha(ROOT/'RECEIPTS.md'),fingerprint=fingerprint))
    start=session_start
    def checkpoint_now():
        save_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),
            step=step,elapsed=elapsed+time.monotonic()-start,fingerprint=fingerprint,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        write(run/'elapsed.json',dict(elapsed_seconds=elapsed+time.monotonic()-start))
    status='completed'
    try:
        with open(run/'curve.jsonl','a',buffering=1) as log:
            while step<args.steps:
                if elapsed+time.monotonic()-start>=min(args.wall_seconds,43200):
                    status='right-censored';break
                tic=time.monotonic()
                x,y=batch('train',step)
                opt.zero_grad(set_to_none=True)
                logits,metrics=model(x.to(device))
                loss=F.cross_entropy(logits.flatten(0,1),y.to(device).flatten())
                replay=replay_present(loss) if args.arm=='tape_lm' else None
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize();step+=1
                row=dict(step=step,tokens=step*4096,loss=loss.item(),grad_norm=float(norm),
                         seconds=time.monotonic()-tic,telemetry=metrics,replay_node=replay)
                if step%64==0 or step==args.steps:
                    row['validation']=validate(model,device)
                log.write(json.dumps(row)+'\n')
                write(run/'elapsed.json',dict(elapsed_seconds=elapsed+time.monotonic()-start))
                if step%64==0:
                    checkpoint_now()
    except BaseException as exc:
        status='failed'
        write(ROOT/f'receipts/{args.tag}_{args.arm}_failure.json',dict(step=step,error=repr(exc)))
        raise
    finally:
        checkpoint_now()
        rows=[json.loads(line) for line in (run/'curve.jsonl').read_text().splitlines()]
        write(ROOT/f'receipts/{args.tag}_{args.arm}.json',dict(status=status,steps=step,tokens=step*4096,
            requested_steps=args.steps,wall_budget_seconds=min(args.wall_seconds,43200),
            budget_overshoot_seconds=max(0.,elapsed+time.monotonic()-start-min(args.wall_seconds,43200)),
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
            physical_gpu=os.environ['GPU_LEASE_HELD'],elapsed_seconds=elapsed+time.monotonic()-start,
            train_seconds_per_token=sum(r['seconds'] for r in rows)/max(1,len(rows)*4096),
            curve=rows,fingerprint=fingerprint))


if __name__=='__main__':
    main()
