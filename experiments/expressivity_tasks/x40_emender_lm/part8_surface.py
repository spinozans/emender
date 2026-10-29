"""Three architecture surface; subprocess-isolated, measured envelope selection.

An OOM is a lower bound, never an estimate of required peak memory. Escalation
is permitted only after an actual CUDA OOM; other exceptions fail the arm.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from functools import lru_cache
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .scale_surface import SurfaceLM, SHAPES, VOCAB, counts
from .part6_surface import features
from .part6_journal import packed_journals
from ..x36_tape_perf.loss import ChunkedLoss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part8'
ARMS = ('transformer', 'gdn2', 'pure')
FLAGS = ('X40_BATCHED_READS','X40_FRESH_POOL','X40_STABLE_MAPPING','X40_FUSED_INIT')

class GDNBlock(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        # Registered gdn2-mlp implementation: external GDN-2, never GDN-1.
        from ndm.models.external_gdn2 import GDN2ExternalMLPLayer
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.layer = GDN2ExternalMLPLayer(d, expansion=2., head_dim=128,
            num_heads=max(1,round(d/288)), gdn2_mlp_ratio=hidden/d,
            gdn2_mlp_multiple=1, use_conv=True, d_conv=4)
    def forward(self, x, ids):
        return x + self.layer(self.norm(x))[0]

class GDNModel(nn.Module):
    def __init__(self, scale, capacity):
        super().__init__()
        d,l,hidden=gdn_config(capacity)
        self.embed=nn.Embedding(VOCAB,d)
        self.norm=nn.RMSNorm(d,eps=1e-5)
        self.head=nn.Linear(d,VOCAB,bias=False)
        self.hidden=hidden
        self.width=d;self.depth=l
        self.blocks=nn.ModuleList(GDNBlock(d,hidden) for _ in range(l))
        for i,b in enumerate(self.blocks): b.layer.set_layer_idx(i)


@lru_cache(None)
def gdn_config(capacity):
    # Approximate the registered paper SwiGLU ratio rather than forcing the pure
    # backbone shape (which would require a pathological 42x MLP at 22M).
    ratio=json.loads((ROOT.parents[2]/"configs/frontier/e97_gdn2_paper/gdn2_mlp.json").read_text())["gdn2_mlp_ratio"]
    candidates=[]
    for d in range(256,2305,64):
        with torch.device('meta'): probe=GDNBlock(d,1)
        per_layer=counts(probe)['total']-3*d
        fixed=(2*VOCAB+1)*d
        for depth in range(6,33):
            hidden=round((capacity-fixed-depth*per_layer)/(depth*3*d*16))*16
            if hidden<16:continue
            total=fixed+depth*(per_layer+3*d*hidden)
            if abs(total/capacity-1)<.001:
                score=abs(hidden/d/ratio-1)
                candidates.append((score,d,depth,hidden))
    if not candidates:raise ValueError('No tensor-core-aligned capacity-matched GDN2 config')
    _,d,depth,hidden=min(candidates)
    return d,depth,hidden


def envelope_specs(batch, arm):
    """Ordered policy, not an assertion of global optimality or fastest envelope."""
    rows=[dict(name='bare',microbatch=batch,checkpoint=False,packed=False,offload=False),
          dict(name='checkpoint',microbatch=batch,checkpoint=True,packed=False,offload=False)]
    if arm=='pure':
        rows.append(dict(name='checkpoint_packed',microbatch=batch,checkpoint=True,packed=True,offload=False))
    for micro in (64,32,16,8,4,2,1):
        if micro>=batch:continue
        rows.append(dict(name=f'checkpoint_B{micro}',
            microbatch=micro,checkpoint=True,packed=False,offload=False))
        if arm=='pure':
            rows.append(dict(name=f'checkpoint_packed_B{micro}',
                microbatch=micro,checkpoint=True,packed=True,offload=False))
    rows.append(dict(name='checkpoint_offload_B1',microbatch=1,checkpoint=True,packed=arm=='pure',offload=True))
    return rows


def validate_parity(context, spec):
    gate=json.loads((OUT/f'production_T{context}.json').read_text())
    assert gate['status']=='passed' and gate['pressure_gate']
    for k in FLAGS: assert os.environ.get(k,'0')==gate['flags'][k]
    for name in ('model.py','forward.py','backward.py','kernels.py','scale_surface.py'):
        assert gate['sources'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), name
    if spec['checkpoint'] or spec['packed']:
        gate_batch=1 if spec['offload'] else 32
        tag='_packed_offload_expandable' if spec['offload'] else '_packed_expandable'
        gate=json.loads((ROOT/f'receipts/part6/checkpoint_gate_B{gate_batch}_T{context}{tag}.json').read_text())
        assert gate['status']=='passed'
        for k in FLAGS:assert gate['flags'][k]==os.environ.get(k,'0')
        for name in ('part6_checkpoint.py','part6_journal.py'):
            assert gate['sources'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scale',choices=SHAPES,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--batch',type=int,choices=(64,128),required=True)
    p.add_argument('--context',type=int,choices=(512,2048),required=True)
    p.add_argument('--envelope-index',type=int,default=0)
    p.add_argument('--steps',type=int,default=20)
    a=p.parse_args(); spec=envelope_specs(a.batch,a.arm)[a.envelope_index]
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    name=f'{a.scale}_{a.arm}_B{a.batch}_T{a.context}'
    out=OUT/'attempts'/name;out.mkdir(parents=True,exist_ok=True)
    result=dict(status='failed',scale=a.scale,arm=a.arm,batch=a.batch,context=a.context,
        envelope=spec,envelope_index=a.envelope_index,physical_gpu=os.environ['GPU_LEASE_HELD'],
        python=os.sys.executable,warmups=3,measured_updates=a.steps,
        precision='bf16 dense autocast with fp32 parameters/AdamW; emender native store and projection fp32' if a.arm=='pure' else 'bf16 dense autocast with fp32 master parameters/AdamW; bf16 fused linear cross entropy',
        precision_asymmetry='Emender native stores retain parity-certified fp32; transformer and GDN2 use bf16 compute, including vocabulary projection. Ratios are against best-practice baselines.',
        memory_semantics='OOM peak is a partial-run lower bound; successful peak covers warmups and measured updates',
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
    try:
        if a.arm=='pure':validate_parity(a.context,spec)
        with torch.device('meta'): reference=SurfaceLM(a.scale,'pure');capacity=counts(reference)
        del reference
        model=(GDNModel(a.scale,capacity['total']) if a.arm=='gdn2' else SurfaceLM(a.scale,a.arm,capacity['total'])).cuda().train()
        result.update(parameters=counts(model),pure_parameters=capacity,ff_hidden=model.hidden,
                      card_bytes=torch.cuda.get_device_properties(0).total_memory)
        assert abs(counts(model)['total']/capacity['total']-1)<.001
        if a.arm=='gdn2':
            from ndm.models.external_gdn2 import probe_gdn2_external_dependencies
            result['gdn2_config']=dict(width=model.width,depth=model.depth,
                mlp_hidden=model.hidden,mlp_ratio=model.hidden/model.width,
                num_heads=max(1,round(model.width/288)),head_dim=128,expand_v=2.,
                kernel='external native GDN2 chunk Triton; FLA helpers')
            result['gdn2_backend']=probe_gdn2_external_dependencies()
            assert result['gdn2_backend']['ok']
            assert result['gdn2_backend']['external_class']=='GatedDeltaNet2'
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        generator=torch.Generator(device='cuda').manual_seed(3402)
        ids=torch.randint(1,VOCAB,(a.batch,a.context+1),device='cuda',generator=generator)
        result['data']='identical independent RNG seed 3402; synthetic nonpadding repeated token tensor across architectures'
        if a.arm!='pure':
            from fla.modules import FusedLinearCrossEntropyLoss
            criterion=FusedLinearCrossEntropyLoss()
        samples=[];losses=[];records=[]
        for it in range(3+a.steps):
            torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
            total=0.
            for begin in range(0,a.batch,spec['microbatch']):
                tokens=ids[begin:begin+spec['microbatch']]
                with packed_journals(records) if spec['packed'] else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        hidden,gates=features(model,tokens[:,:-1],a.arm,spec['checkpoint'],False,spec['offload'])
                        if a.arm!='pure':
                            loss=criterion(hidden.to(torch.bfloat16).contiguous(),tokens[:,1:].contiguous(),model.head.weight.to(torch.bfloat16))
                    if a.arm=='pure':
                        loss=ChunkedLoss.apply(hidden.float(),model.head.weight,tokens[:,1:],1024)
                        loss=loss+.01*torch.stack(gates).mean()
                    loss=loss*(len(tokens)/a.batch)
                    loss.backward()
                total+=float(loss.detach());del hidden,gates,loss
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if it>=3:samples.append(elapsed);losses.append(total)
            print(name,spec['name'],it,elapsed,total,flush=True)
        result.update(status='passed',seconds=samples,losses=losses,
            tokens_per_second=a.batch*a.context*len(samples)/sum(samples),journal_records=records)
    except torch.OutOfMemoryError as exc:
        result.update(status='oom',error=repr(exc),traceback=traceback.format_exc())
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        (out/f'{a.envelope_index:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status']!='passed':print(result.get('traceback'),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 42 if result['status']=='oom' else 1)

if __name__=='__main__':main()
