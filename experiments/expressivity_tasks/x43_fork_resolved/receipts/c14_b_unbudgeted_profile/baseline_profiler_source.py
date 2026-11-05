"""Read-only trained B1 profiling and byte-exact run fingerprints.

No trained checkpoint, optimizer state, production kernel or launch gate is
written. --compare checks an earlier fingerprint, including across code edits.
Timing wrappers synchronize and therefore are diagnostic, not throughput.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from . import c14_b_control_native as native
from . import c14_b_control_graph as graph
from . import c14_b_train as train
from .c14_b_v2 import configure as base_configure

def configure():
    base_configure()
    from .c14_b_controls import EXTRA
    from .c14_b_control_model import StoreIntegrated
    train.OUT=train.ROOT/'receipts/c14_b_v2_unbudgeted'
    train.CHECKPOINT=Path('/home/erikg/c14_dense_bytes/emender_c14_b1_v2_unbudgeted_resume.pt')
    train.CODE=(*train.CODE,*EXTRA)
    train.StoreIntegrated=lambda:StoreIntegrated('unbudgeted')
from .c14_fast_features import load_decoder


def bits(value):
    if isinstance(value, torch.Tensor):
        x=value.detach().cpu().contiguous()
        return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,(tuple,list)):
        return [bits(x) for x in value]
    if isinstance(value,dict):
        return {k:bits(v) for k,v in value.items()}
    return value


def split_replay(inputs,cfg,policy,metadata,steps):
    """Diagnostic captures of the unchanged replay forward and autograd.

    Compare every returned gradient byte to the standing combined graph before
    accepting these times. These captures never replace the training graphs.
    """
    from .c14_b_control_replay import replay,encode
    static=[x.detach().clone().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
    encoded=encode(metadata,inputs[0].device)
    active=[x for x in static if x.requires_grad]
    gradient=torch.ones_like(inputs[2])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def run():
        with torch.enable_grad():
            out,deps=replay(*static,cfg,policy,None,encoded=encoded)
            grads=torch.autograd.grad(out,active,gradient,allow_unused=True)
        return out,deps,grads
    with torch.cuda.stream(stream):
        for _ in range(2):run()
    torch.cuda.current_stream().wait_stream(stream)
    forward=torch.cuda.CUDAGraph();backward=torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward,stream=stream),torch.enable_grad():
        out,deps=replay(*static,cfg,policy,None,encoded=encoded)
    with torch.cuda.graph(backward,stream=stream):
        grads=torch.autograd.grad(out,active,gradient,allow_unused=True)
    forward.replay();backward.replay();torch.cuda.synchronize()
    baseline,expected=graph.backward(inputs,gradient,cfg,policy,metadata)
    used=int(deps);it=iter(grads);actual=[]
    for i,x in enumerate(static):
        g=next(it) if x.requires_grad else None
        actual.append(g if g is not None and used & (1<<i) else None)
    exact=bits((out,actual))==bits((baseline,expected))
    rows=[]
    for _ in range(steps):
        begin=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();forward.replay();middle.record();backward.replay();end.record();end.synchronize()
        rows.append([begin.elapsed_time(middle),middle.elapsed_time(end)])
    return dict(byte_exact_against_combined=exact,forward_backward_gpu_ms=rows,
        median_forward_ms=statistics.median(x[0] for x in rows),median_backward_ms=statistics.median(x[1] for x in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-slicing',action='store_true',help='read-only ablation restoring the original slice-per-token replay');p.add_argument('--candidate',action='store_true',help='allow intentional replay source change; requires --compare');p.add_argument('--steps',type=int,default=20);p.add_argument('--compare',type=Path)
    a=p.parse_args();assert a.steps>=3
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    assert not a.output.exists(), 'refuse to replace evidence'
    a.output.mkdir(parents=True)
    if a.reference_slicing:
        from . import c14_b_control_replay as batched
        source=inspect.getsource(batched.replay)
        source=source.replace('    token_columns=tuple(x.unbind(1) for x in (k,q,v,av,pop,append,logits))\n','')
        source=source.replace('tuple(column[i] for column in token_columns)','tuple(x[:,i] for x in (k,q,v,av,pop,append,logits))')
        namespace=dict(vars(batched));exec(compile(source,'<reference slicing replay>','exec'),namespace)
        batched.replay=namespace['replay'];graph.replay=batched.replay
    torch.set_num_threads(1);torch.manual_seed(3401)
    assert not a.candidate or a.compare, 'candidate requires strict reference'
    configure();source,shards,gate=train.preflight(check_implementation=False)
    assert train.sha(train.CHECKPOINT)==json.loads((train.OUT/'HARVEST.json').read_text())['checkpoint_sha256_verified']
    r=shards[0];data=(train.CACHE/r['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==r['sha256']
    identities,lengths,values,contexts,_=train.oracle_shard(data,r['windows'],load_decoder())
    batch=next(iter(train.verified_batches(np.memmap(source,dtype=np.uint8,mode='r'))))
    for j,(d,o,bb) in enumerate(batch):assert (d,o,len(bb)-1)==(*identities[j],lengths[j])
    xx,yy,mask=train.tensors(batch)
    ff=torch.as_tensor(values[:16],device='cuda').clone()
    ff*=(torch.rand((16,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(3401))>=.5)[...,None]
    ll=torch.as_tensor(contexts[:16],device='cuda')
    # Read only; never save or update this model/checkpoint.
    saved=torch.load(train.CHECKPOINT,map_location='cpu',weights_only=False)
    model=train.StoreIntegrated().cuda().train();model.load_state_dict(saved['model'])
    del saved
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    phase='warmup';timings={};record=False;capture=False;states=[];sample=[]
    def now():torch.cuda.synchronize();return time.perf_counter()
    def add(name,elapsed):
        key=phase+'/'+name;timings.setdefault(key,[]).append(elapsed*1000)
    old_native=native.native_forward;old_backward=graph.backward;old_reports=native.native_reports
    def forward(inputs,cfg,policy,metadata):
        start=now();out=old_native(inputs,cfg,policy,metadata);elapsed=now()-start
        if record:add('native_store',elapsed)
        if capture:states.append(bits(out))
        if len(sample)<4:sample.append((tuple(x.detach().clone() for x in inputs),cfg,policy,metadata))
        return out
    def backward(*args):
        start=now();out=old_backward(*args);elapsed=now()-start
        if record:add('store_replay_plus_autograd',elapsed)
        if capture:states.append(bits(out))
        return out
    def reports(*args):
        start=now();out=old_reports(*args);elapsed=now()-start
        if record:add('telemetry',elapsed)
        return out
    native.native_forward=forward;graph.backward=backward;native.native_reports=reports
    handles=[]
    # Each module is a leaf component; repeated invocations distinguish initial
    # forward from checkpoint recomputation through the phase prefix.
    for name,module in model.named_modules():
        label=None
        if name=='device_input':label='content_260_projection'
        elif name.endswith('cell.W_v'):label='value_projection_stream_and_array'
        elif '.geometry.' in name and name.endswith(('key','query')):label='key_query_projection'
        elif name.endswith(('cell.W_pop','cell.W_append','cell.W_write','cell.W_in')):label='addressing_projections'
        if label:
            stack=[]
            def before(m,args,stack=stack):stack.append(now())
            def after(m,args,out,stack=stack,label=label):
                elapsed=now()-stack.pop()
                if record:add(label,elapsed)
            handles.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
    def run(enabled=True):
        nonlocal phase
        model.zero_grad(set_to_none=True);phase='forward';start=now()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lp,_=model(xx,ff,ll,windows=[(d,o) for d,o,_ in batch],valid_lengths=lengths[:16].tolist(),enabled=enabled)
            loss=-lp.gather(-1,yy[...,None])[...,0][mask].mean()
        f=now()-start;phase='backward';start=now();loss.backward();b=now()-start
        assert torch.isfinite(loss)
        return loss,(f*1000,b*1000)
    # Compile/capture before collecting evidence. No optimizer step here.
    run();run()
    fingerprints=[]
    for repeat in range(2):
        states=[];capture=True;loss,_=run();capture=False
        fingerprints.append(dict(loss=bits(loss),stores=states,telemetry=bits(model.telemetry),
            gradients={n:bits(x.grad) for n,x in model.named_parameters()}))
    (a.output/'fingerprint.json').write_text(json.dumps(fingerprints[0],indent=2)+'\n')
    exact=fingerprints[0]==fingerprints[1]
    comparison=None
    if a.compare:comparison=json.loads(a.compare.read_text())==fingerprints[0]
    # Persist the failure before any assertion: nondeterminism is evidence.
    (a.output/'determinism.json').write_text(json.dumps(dict(run_to_run_exact=exact,comparison_exact=comparison,
        scope='loss, all native reads/final K,V,strength,metadata,traces, replay outputs/gradients, model gradients and telemetry; fixed trained weights and first real document batch',
        input_bits=bits((xx,yy,mask,ff,ll)),seed=3401),indent=2)+'\n')
    record=True;curve=[]
    for _ in range(a.steps):
        _,row=run();curve.append(row)
    record=False
    for h in handles:h.remove()
    native.native_forward=old_native;graph.backward=old_backward;native.native_reports=old_reports
    # Uninstrumented fixed-weight forward/backward ceiling, with no optimizer
    # mutation. A separate disposable copy measures full optimizer composition.
    bare=[];updates=[]
    original={n:x.detach().clone() for n,x in model.state_dict().items()}
    # Disposable optimizer weights are restored outside each timing region. No
    # updates are accumulated and nothing is saved to the trained checkpoint.
    for _ in range(a.steps):
        _,row=run();bare.append(row)
        start=now();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        updates.append((now()-start)*1000)
        model.load_state_dict(original)
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:run()
    (a.output/'kernel_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
    passive=[];passive_channel=[];passive_path=None;interventions={};split={}
    for head,(inputs,cfg,policy,metadata) in enumerate(sample):
        out=old_native(inputs,cfg,policy,metadata)
        reports=native.native_reports(out[1][1],cfg,lengths[:16].tolist())
        interventions[str(head)]=dict(slots=cfg.slots,physical_gpu_slots_power2=2**(cfg.slots-1).bit_length(),reports=reports)
    result=dict(replay_mode='reference_slicing' if a.reference_slicing else 'current',split_replay=split,status='completed' if exact and comparison is not False else 'determinism_failed',
        gpu=os.environ['GPU_LEASE_HELD'],checkpoint_read_only=str(train.CHECKPOINT),
        checkpoint_receipt_sha256=json.loads((train.OUT/'HARVEST.json').read_text())['checkpoint_sha256_verified'],
        source_sha256={n:train.sha(train.ROOT/n) for n in train.CODE},
        profiler_sha256=train.sha(Path(__file__)),cache_shard_sha256=r['sha256'],
        instrumented_forward_backward_ms=curve,uninstrumented_forward_backward_ms=bare,
        dense_bypass_forward_backward_ms=passive,passive_channel_forward_backward_ms=passive_channel,passive_checkpoint_read_only=str(passive_path),component_calls_ms=timings,
        component_per_step_ms={k:sum(v)/a.steps for k,v in timings.items()},
        occupancy_intervention_all_heads=interventions,update_ms=updates,compute_steps_per_second=1000/statistics.mean(sum(row)+u for row,u in list(zip(bare,updates))[1:]),run_to_run_exact=exact,comparison_exact=comparison,
        limitations=['Fixed first real 16x128 batch, trained weights; no retraining or checkpoint writes.',
        'Enabled=False is dense bypass, not the historical passive consultation arm.',
        'Native reads/addressing/APPEND/CORRECT/DECAY/POP are fused in one kernel; no additive per-op timing is claimed.',
        'Replay includes forward reconstruction and its autograd; projection recomputation is measured separately.',
        'Optimizer updates are disposable and restored outside measurement; no sustained end-to-end training steps/s claim.'])
    (a.output/'composition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('component_calls_ms','instrumented_forward_backward_ms','uninstrumented_forward_backward_ms','passive_forward_backward_ms')},indent=2),flush=True)
    assert all(x['byte_exact_against_combined'] for x in split.values()),'split capture changed output/gradient bits'
    assert exact,'CURRENT implementation failed byte-exact run-to-run determinism'
    assert comparison is not False,'cross-change fingerprint differs'

if __name__=='__main__':main()
