"""Differential metadata initialization through retained backward and freeze.

All three arrays are initialized with either zeros, finite sentinels, or a
NaN terminal sentinel. Compare defined public state and gradients as bytes.
This checks initialization neutrality, not baseline kernel recertification.
"""
import ast
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused as c31
from experiments.expressivity_tasks.x36_tape_perf import forward as f36, cell_fused as c36
from experiments.expressivity_tasks.x40_emender_lm import forward as f40
from experiments.expressivity_tasks.x40_emender_lm.model import POLICIES
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state


def initialize_variant(variant):
    for module in (c31,f36,f40):
        source=Path(module.__file__).read_text()
        for pattern in ('due = torch.zeros((b*h,cfg.slots),','terminal = torch.zeros((b*h,cfg.slots),','births = torch.zeros_like(due)'):
            assert source.count(pattern)==1,(module.__name__,pattern)
        integer=0 if variant==0 else 1234567
        floating='0.' if variant==0 else ('12345.' if variant==1 else "float('nan')")
        source=source.replace('due = torch.zeros((b*h,cfg.slots),','due = torch.full((b*h,cfg.slots),'+str(integer)+',')
        source=source.replace('terminal = torch.zeros((b*h,cfg.slots),','terminal = torch.full((b*h,cfg.slots),'+floating+',')
        source=source.replace('births = torch.zeros_like(due)','births = torch.full_like(due,'+str(integer)+')')
        tree=ast.parse(source,module.__file__)
        tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='tape_scan']
        exec(compile(tree,module.__file__,'exec'),module.__dict__)
    c36.persistent_scan=f36.tape_scan


def equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().numpy().tobytes()==b.detach().cpu().numpy().tobytes()


def run(target,cfg,imported,variant):
    initialize_variant(variant)
    torch.manual_seed(4021)
    k=torch.randn(1,70,1,cfg.key_dim,device='cuda')*.3
    q=(torch.randn_like(k)*.3).requires_grad_()
    v=torch.randn(1,70,1,cfg.value_dim,device='cuda',requires_grad=True)
    pop=torch.zeros(1,70,1,device='cuda');pop[:,2::3]=1;pop.requires_grad_()
    append=torch.zeros_like(pop,requires_grad=True)
    coef=torch.zeros(1,device='cuda',requires_grad=True)
    mask=torch.ones_like(pop)
    state=None
    if imported:
        state=init_state(1,cfg,'cuda')
        state.live[...,:2]=True;state.strength[...,:2]=.5;state.age[...,:2]=3
        state.keys[...,:2,:]=.1;state.values[...,:2,:]=.2
    snapshots=[];outputs=[];start=0
    coverage=dict(pops=0,freed=0,allocations=0,reused=0)
    seen=set()
    for length in (1,2,31,1,35):
        sl=slice(start,start+length)
        xs=(k[:,sl],q[:,sl],v[:,sl],mask[:,sl],pop[:,sl],append[:,sl])
        if target=='x40':
            # Match model.scan: the wire writes beta; retained chunks must not
            # share a base tensor/version counter with later wire launches.
            xs=(*xs[:3],xs[3].detach().clone(),*xs[4:])
            from experiments.expressivity_tasks.x40_emender_lm.backward import autograd_scan
            e=torch.empty(1,cfg.n_heads,length,device='cuda');err=torch.empty_like(e)
            zero=torch.zeros(cfg.n_heads,device='cuda')
            wired=partial(f40.tape_scan,wire_logits=torch.full_like(mask[:,sl],10.),
                          match_coef=zero,error_coef=zero,evidence=e,error_evidence=err)
            with patch.object(c36,'persistent_scan',wired):
                out,state,trace,pressure=autograd_scan(xs,state,cfg,coef)
        else:
            module=c31 if target=='x31' else c36
            out,state,trace,pressure=module.tape_scan(*xs,state,cfg,coef)
        outputs.append(out)
        values=[out,pressure,state.history]
        if target=='x40':values += [e,err]
        values += [getattr(state.native,name) for name in ('keys','values','strength','live','age')]
        values += [x.reshape_as(state.native.live)[state.native.live] if i in (2,3,4) else x for i,x in enumerate(state.metadata)]
        values += [getattr(trace,name) for name in ('case','matched','alloc_target','pop_target','pop_fired','freed','w_on')]
        snapshots += [x.detach().clone() for x in values]
        coverage['pops']+=int(trace.pop_fired.sum());coverage['freed']+=int(trace.freed.sum())
        for slot in trace.alloc_target.cpu().flatten().tolist():
            if slot>=0:
                coverage['allocations']+=1;coverage['reused']+=slot in seen;seen.add(slot)
        start+=length
    loss=sum(x.square().sum() for x in outputs)+sum(getattr(state.native,name).square().sum() for name in ('keys','values','strength'))
    gradients=torch.autograd.grad(loss,(q,v,pop,append,coef))
    snapshots += [x.detach().clone() for x in gradients]
    assert all(coverage.values()),coverage
    return snapshots,coverage


if __name__=='__main__':
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    torch.set_num_threads(1)
    print(json.dumps({'mode':'production backward lifecycle'}),flush=True)
    rows=[]
    targets=tuple(sys.argv[1:]) or ('x31','x36','x40')
    assert all(target in ('x31','x36','x40') for target in targets)
    for target in targets:
        for policy,original in enumerate(POLICIES):
            for imported in (False,True):
                for decay in (original.decay_rate,1.):
                    cfg=replace(original,decay_rate=decay,key_dim=4,value_dim=8)
                    reference,coverage=run(target,cfg,imported,0)
                    for variant in (1,2):
                        observed,_=run(target,cfg,imported,variant)
                        assert len(reference)==len(observed)
                        failures=[i for i,(a,b) in enumerate(zip(reference,observed)) if not equal(a,b)]
                        assert not failures,(target,policy,imported,decay,variant,failures)
                    row=dict(target=target,policy=policy,imported=imported,decay=decay,coverage=coverage,status='passed')
                    rows.append(row);print(json.dumps(row),flush=True)
    print(json.dumps({'status':'passed','fixtures':len(rows),'variants':3,'details':rows}),flush=True)
