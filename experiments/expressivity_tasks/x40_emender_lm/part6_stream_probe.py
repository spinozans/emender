"""Scoped head-stream overlap experiment; native arithmetic and warp layout unchanged."""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from . import model
from .part6_gate import bitwise_equal

ROOT=Path(__file__).resolve().parent/'receipts/part6'
_DEVICE_STREAMS={}


def streamed_forward(self,x,ids):
    if not self.training or self.use_chunk_graphs:
        raise ValueError('stream probe supports recorded training without chunk graphs only')
    b,t=ids.shape
    z=self.W_in(x);values=self.W_v(x).reshape(b,t,4,128)
    pop,append,logits=self.W_pop(x).sigmoid(),self.W_append(x),self.W_write(x)
    # Layer dependencies already serialize cells. Reuse four device streams
    # so the allocator can reuse their buffers across all six blocks.
    if x.device.index not in _DEVICE_STREAMS:
        _DEVICE_STREAMS[x.device.index]=[torch.cuda.Stream(device=x.device) for _ in range(4)]
    self._part6_streams=_DEVICE_STREAMS[x.device.index]
    origin=torch.cuda.current_stream(x.device)
    reads=[];gates=[]
    for h,cfg in enumerate(self.configs):
        # Keep dense transforms and their backward GEMMs on the original
        # stream. Only independent native scans overlap, preserving cuBLAS
        # workspace/accumulation order as well as native warp geometry.
        features=z+self.features[h](ids)
        k=self.geometry[h]['key'](features).detach().unsqueeze(2)
        q=self.geometry[h]['query'](features).unsqueeze(2)
        sl=slice(h,h+1)
        inputs=(k,q,values[:,:,sl].contiguous(),(ids!=0).float().unsqueeze(-1),
                pop[:,:,sl].contiguous(),append[:,:,sl].contiguous(),
                self.append_err_coef[sl].contiguous(),logits[:,:,sl].contiguous(),
                self.match_coef[sl],self.error_coef[sl])
        stream=self._part6_streams[h];stream.wait_stream(origin)
        for tensor in inputs:tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            (r,state,trace,pressure),e,err,sig=model.scan(*inputs[:6],cfg,*inputs[6:])
            reads.append(r);gates.append(sig.mean())
    for stream in self._part6_streams:origin.wait_stream(stream)
    for tensor in (*reads,*gates):tensor.record_stream(origin)
    self.gate_mean=torch.stack(gates).mean();self.telemetry=None
    return torch.cat(reads,2)


@contextmanager
def overlap():
    with patch.object(model.Cell,'forward',streamed_forward):yield


def gate(batch,context,suffix='',deferred_pack=False,launch_registers=0,stream_candidate=True):
    torch.set_num_threads(1);torch.manual_seed(3401)
    a=model.Cell(32768).cuda().train();b=copy.deepcopy(a)
    ids=torch.randint(1,32768,(batch,context),device='cuda')
    x=torch.randn(batch,context,256,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_()
    original_scan=model.scan;reference_inputs=[];candidate_inputs=[]
    def capture(sink):
        def run(*args,**kwargs):
            for tensor in (args[1],args[2]):tensor.retain_grad()
            sink.append((args[1],args[2]))
            return original_scan(*args,**kwargs)
        return run
    with patch.object(model,'scan',capture(reference_inputs)):
        out=a(x,ids);g=a.gate_mean
        (out.square().mean()+.01*g).backward()
    from .part6_deferred_journal import packed_cells
    from .part6_register_probe import RegisterLaunch
    from . import backward
    records=[]
    with patch.object(backward,'reverse_kernel',RegisterLaunch(backward.reverse_kernel,launch_registers)) if launch_registers else nullcontext(),overlap() if stream_candidate else nullcontext(),packed_cells(records) if deferred_pack else nullcontext(),patch.object(model,'scan',capture(candidate_inputs)):
        other=b(y,ids);h=b.gate_mean
        (other.square().mean()+.01*h).backward()
    checks={'output':bitwise_equal(out,other),'gate':bitwise_equal(g,h),'input_gradient':bitwise_equal(x.grad,y.grad)}
    differences={}
    assert len(reference_inputs)==len(candidate_inputs)==4
    native_checks={}
    for h,(left,right) in enumerate(zip(reference_inputs,candidate_inputs)):
        for label,p,q in zip(('query','value'),left,right):
            name=f'native_{label}_h{h}'
            native_checks[name]=bitwise_equal(p.grad,q.grad)
            differences[name]=float((p.grad-q.grad).abs().max())
    checks.update(native_checks)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        checks[name]=(p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and bitwise_equal(p.grad,q.grad))
        if p.grad is not None and q.grad is not None:
            differences[name]=float((p.grad-q.grad).abs().max())
    result=dict(physical_gpu=os.environ['GPU_LEASE_HELD'],python=sys.executable,native_warps=4,registers=launch_registers,head_streams=stream_candidate,native_adjoints_equal=all(native_checks.values()),flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')},deferred_pack=deferred_pack,journal_records=records,status='passed' if all(checks.values()) else 'failed',batch=batch,context=context,
                checks=checks,max_parameter_gradient_differences=differences,
                input_gradient_max_difference=float((x.grad-y.grad).abs().max()),
                scope='Full Cell, exact outputs, gate, input and parameter gradients; no relaxed native tolerance or changed warp count.')
    (ROOT/f'{"head_stream_gate" if stream_candidate else "register_gate"}_B{batch}_T{context}{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise AssertionError([k for k,v in checks.items() if not v])


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['gate','pilot','curve'],default='gate');p.add_argument('--batch',type=int,default=64);p.add_argument('--context',type=int,default=512);a,rest=p.parse_known_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    if a.action=='gate':gate(a.batch,a.context)
    else:
        assert json.loads((ROOT/f'head_stream_gate_B{a.batch}_T{a.context}.json').read_text())['status']=='passed'
        if a.action=='pilot':
            from . import part5_pilot
            sys.argv=[sys.argv[0],'--batch',str(a.batch)]+rest
            with overlap():part5_pilot.main()
        else:
            from . import train_slice
            sys.argv=[sys.argv[0]]+rest
            with overlap():train_slice.main()


if __name__=='__main__':main()
