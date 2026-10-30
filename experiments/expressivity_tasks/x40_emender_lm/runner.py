"""Broker-only preflight and ordered TinyStories run; fail closed on parity."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import validate
from .model import LM, policy_receipt, parameter_receipt, require_replays
from .parity import certify
from ..x42_pile_scale.checkpoint import save, restore

ROOT=Path(__file__).resolve().parent
RECEIPT_ROOT=ROOT/'receipts'


def write(name,value):
    # evaluate/depth import this module separately when runner runs as __main__.
    target=Path(os.environ.get('X40_RECEIPT_DIR',str(RECEIPT_ROOT)))/name
    target.write_text(json.dumps(value,indent=2)+'\n')


def fingerprint():
    paths=[]
    for folder in ('x30_tape_cell','x31_tape_cell_fused','x34_lm_pilot','x36_tape_perf',
                   'x37_head_differentiation','x39_memory_usage','x40_emender_lm'):
        paths.extend((ROOT.parent/folder).glob('*.py'))
    paths.append(ROOT/'RECEIPTS.md')
    return {str(p.relative_to(ROOT.parent)):data.sha(p) for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',action='store_true')
    p.add_argument('--cuda-graphs',action='store_true')
    p.add_argument('--store-dtype',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--ignition-v1',action='store_true')
    p.add_argument('--receipt-dir',type=Path)

    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-every',type=int,default=128)
    p.add_argument('--match-v1',action='store_true')
    args=p.parse_args()
    if args.checkpoint_every<1:p.error('positive checkpoint interval required')
    global RECEIPT_ROOT
    if args.ignition_v1 and args.receipt_dir is None:
        p.error('--ignition-v1 requires a fresh --receipt-dir')
    if args.receipt_dir is not None:
        RECEIPT_ROOT=args.receipt_dir.resolve()
        RECEIPT_ROOT.mkdir(parents=True,exist_ok=args.resume)
    os.environ['X40_RECEIPT_DIR']=str(RECEIPT_ROOT)
    gate_cost=0. if args.ignition_v1 else .01
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in ('0','1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('single broker lease required on 0/1/5/6')
    torch.set_num_threads(1)
    fp=fingerprint()
    registration=dict(sources=fp,physical_gpu=held,ignition_v1=args.ignition_v1,gate_cost=gate_cost,policies=policy_receipt(),
        preregistration_sha256=data.sha(ROOT/'RECEIPTS.md'),torch=torch.__version__,
        cuda_graphs=args.cuda_graphs,store_dtype=args.store_dtype)
    write('registration.json',registration)
    try:
        command=[sys.executable,"-m","pytest","-q","-p",
            "experiments.expressivity_tasks.x36_tape_perf.oracle_adapter",
            str(ROOT.parent/"x31_tape_cell_fused/tests"),
            "--junitxml="+str(RECEIPT_ROOT/"oracle.xml")]
        with (RECEIPT_ROOT/"oracle.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        parity=certify()
        write('parity.json',parity)
    except BaseException as exc:
        write('parity.json',dict(status='failed',error=repr(exc)))
        raise
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=args.ignition_v1,store_dtype=args.store_dtype)
    write('parameters.json',parameter_receipt(model,tok.get_vocab_size()))
    model=model.cuda().train()
    x,y=(a.cuda() for a in data.batch('train',0,batch_size=64))
    logits,_=model(x)
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
    loss=ce+gate_cost*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
    count=require_replays(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    write('smoke.json',dict(status='passed',replay_nodes=count,loss=float(loss.detach()),batch=64,context=512))
    del logits,loss,ce,x,y
    if not args.train:return
    # Reset initialization after smoke: the smoke is never counted as exposure.
    del model
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=args.ignition_v1,store_dtype=args.store_dtype).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    registration['data']={name:data.sha(data.RUNS/name) for name in ('train.bin','valid.bin','tokenizer.json')}
    write('registration.json',registration)
    graph=None
    if args.cuda_graphs:
        from .graphs import certify_training_graph
        x,y=(a.cuda() for a in data.batch('train',0,batch_size=64))
        graph,receipt=certify_training_graph(model,x,y,gate_cost=gate_cost)
        write('graph_parity.json',receipt)
    checkpoint_dir=RECEIPT_ROOT/'runs'
    checkpoint_dir.mkdir(exist_ok=True)
    binding=dict(sources=fp,data=registration['data'],seed=3401,ignition=args.ignition_v1,
                 store_dtype=args.store_dtype,cuda_graphs=args.cuda_graphs,match_v1=args.match_v1,checkpoint_helper=data.sha(ROOT.parent/'x42_pile_scale/checkpoint.py'),environment={k:v for k,v in os.environ.items() if k.startswith('X40_') and k!='X40_RECEIPT_DIR'})
    baseline=[json.loads(line) for line in (ROOT/'receipts/ignition_v1/curve.jsonl').read_text().splitlines()] if args.match_v1 else None
    begin=time.monotonic();step=0;elapsed=0.
    if args.resume:
        state=restore(checkpoint_dir/'lm.ckpt',model,opt,binding)
        step=state['step'];elapsed=state['elapsed']
        # Truncate uncheckpointed observations before replaying their updates.
        curve=RECEIPT_ROOT/'curve.jsonl'
        curve.write_text(''.join(curve.read_text().splitlines(keepends=True)[:step]))
    first_step=step+1
    try:
        with (RECEIPT_ROOT/'curve.jsonl').open('a' if args.resume else 'x',buffering=1) as log:
            for step in range(first_step,3053):
                opt.zero_grad(set_to_none=True)
                x,y=(a.cuda() for a in data.batch('train',step-1,batch_size=64))
                if graph is None:
                    logits,_=model(x)
                    ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
                    loss=ce+gate_cost*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
                    require_replays(loss);loss.backward()
                else:
                    loss=graph.replay(x,y);ce=graph.ce
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                value=float(loss.detach())
                if baseline is not None and value != baseline[step-1]['loss']:
                    write('v1_divergence.json',dict(status='failed',step=step,expected=baseline[step-1]['loss'],actual=value,requirement='bitwise loss equality; depth prohibited'))
                    raise AssertionError('v1 bitwise trajectory divergence')
                gates=[float(b.cell.gate_mean.detach()) for b in model.blocks]
                if args.ignition_v1 and any(g<=.5 for g in gates):
                    raise AssertionError('ignition-v1 enable invariant failed')
                if not math.isfinite(value):raise FloatingPointError('nonfinite loss')
                log.write(json.dumps(dict(step=step,tokens=step*32768,loss=value,ce=float(ce.detach()),gate_per_block=gates,grad_norm=float(norm),replay_nodes=24 if graph is None else None,chunk_graphs=48 if graph is not None else 0))+'\n')
                if step%args.checkpoint_every==0 or step==3052:
                    save(checkpoint_dir/'lm.ckpt',model,opt,binding,step,step*32768,[],elapsed+time.monotonic()-begin)
                    write('checkpoint.json',dict(step=step,path=str(checkpoint_dir/'lm.ckpt'),sha256=data.sha(checkpoint_dir/'lm.ckpt')))
                if graph is None:del logits
                del loss,ce
                if step%256==0 or step in (611,3052):
                    if fingerprint()!=fp:raise RuntimeError('source changed during training')
                    write(f'quality_{step}.json',dict(step=step,tokens=step*32768,validation=validate(model,'cuda')))
                    print(json.dumps(dict(step=step,seconds=time.monotonic()-begin)),flush=True)
        if graph is not None:
            from .graphs import disable_chunk_graphs
            disable_chunk_graphs(model)
            graph=None
        if baseline is not None:write('v1_reproduction.json',dict(status='passed',steps=3052,bitwise_loss=True,baseline_sha256=data.sha(ROOT/'receipts/ignition_v1/curve.jsonl'),checkpoint_sha256=data.sha(checkpoint_dir/'lm.ckpt')))
        if not args.match_v1:
            from .evaluate import evaluate
            evaluate(model)
        from .depth import fine_tune
        fine_tune(model,checkpoint_dir=checkpoint_dir,resume=args.resume)
        write('training.json',dict(status='completed',steps=3052,tokens=100007936,seconds=time.monotonic()-begin))
    except BaseException as exc:
        write('training.json',dict(status='failed',last_attempted_step=step,error=repr(exc),seconds=time.monotonic()-begin))
        raise

if __name__=='__main__':main()
