"""Physical-batch pilot occupancy with recorded checkpoint and bounded projection."""
import argparse
from contextlib import nullcontext
import hashlib
import gzip
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x36_tape_perf.loss import ChunkedLoss
from .model import LM
from .part6_surface import features
from .part6_journal import packed_journals
from .part6_register_probe import RegisterLaunch
from .part6_stream_probe import overlap
from .part6_deferred_journal import packed_cells
from .part6_pool_fusion import fused_pool
from . import backward
from .performance import instrumentation
from ..x36_tape_perf.profile_summary import summarize

ROOT=Path(__file__).resolve().parent/'receipts/part6'


def main():
    p=argparse.ArgumentParser();p.add_argument('--loss-chunk',type=int,choices=[1024,4096,16384],default=1024);p.add_argument('--batch',type=int,default=256);p.add_argument('--registers',type=int,default=0)
    p.add_argument('--profile',action='store_true');p.add_argument('--fused-pool',action='store_true');p.add_argument('--profile-if-gate',action='store_true');p.add_argument('--no-checkpoint',action='store_true');p.add_argument('--deferred-pack',action='store_true');p.add_argument('--packed',action='store_true');p.add_argument('--head-streams',action='store_true');p.add_argument('--steps',type=int,default=20);p.add_argument('--label',required=True)
    a=p.parse_args();assert a.steps>=20
    assert not (a.deferred_pack and a.packed)
    assert not a.deferred_pack or (a.no_checkpoint and a.head_streams)
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    allocator_settings=torch.cuda.memory._snapshot().get('allocator_settings',{})
    production_name='production_T512_expandable.json' if allocator_settings.get('expandable_segments') else 'production_T512.json'
    production=json.loads((ROOT/production_name).read_text());assert production['status']=='passed' and production['pressure_gate']
    for key,value in production['flags'].items():assert os.environ.get(key,'0')==value
    if not a.no_checkpoint or a.packed:
        gate=json.loads((ROOT/'checkpoint_gate_B32_T512_packed_fp32_expandable.json').read_text())
        assert gate['status']=='passed' and gate['equality'].startswith('raw contiguous bytes')
        for name in ('model.py','forward.py','backward.py','kernels.py','part6_checkpoint.py','part6_journal.py','scale_surface.py'):
            assert gate['sources'][name]==hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest(),name
    if a.fused_pool:assert json.loads((ROOT/'fused_pool_parity.json').read_text())['status']=='passed'
    if a.registers:
        stem=f'registers{a.registers}'+(f'_head_streams_B{a.batch}' if a.head_streams else '')+('_deferred_pack' if a.deferred_pack else '')
        variant=json.loads((ROOT/f'{stem}_parity.json').read_text());assert variant['status']=='passed'
        for key,value in variant['flags'].items():assert os.environ.get(key,'0')==value
        assert variant['backward_sha256']==hashlib.sha256(Path(backward.__file__).read_bytes()).hexdigest()
    elif a.head_streams:assert json.loads((ROOT/f'head_stream_gate_B{a.batch}_T512.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'curve/summary.json').read_text())['status']=='passed'
    if a.loss_chunk!=1024:assert json.loads((ROOT/f'loss_chunk{a.loss_chunk}_parity.json').read_text())['status']=='passed'
    torch.set_num_threads(1);torch.manual_seed(3401)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    x,y=(v.cuda() for v in data.batch('train',0,batch_size=a.batch))
    records=[];samples=[];losses=[];original=backward.reverse_kernel
    def step():
        calls=0;native=backward.Replay.backward
        def replay(*args):
            nonlocal calls
            calls+=1;return native(*args)
        opt.zero_grad(set_to_none=True)
        with packed_journals(records) if a.packed else nullcontext(),patch.object(backward.Replay,'backward',staticmethod(replay)):
            with torch.profiler.record_function('x40::forward'):
                hidden,gates=features(model,x,'pure',not a.no_checkpoint)
                loss=ChunkedLoss.apply(hidden,model.head.weight,y,a.loss_chunk)+0.*torch.stack(gates).mean()
            with torch.profiler.record_function('x40::backward'):
                loss.backward();assert calls==24
        with torch.profiler.record_function('x40::optimizer'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        value=float(loss.detach());del hidden,gates,loss
        return value,calls
    result=dict(status='failed',batch=a.batch,context=512,physical_batch=a.batch,checkpoint=not a.no_checkpoint,deferred_pack=a.deferred_pack,
                native_warps=4,fused_pool=a.fused_pool,registers=a.registers,head_streams=a.head_streams,packed=a.packed,loss_chunk=a.loss_chunk,warmups=3,measured_updates=a.steps,
                scope='fp32 pilot-family ignition v1, physical batch, optional recorded per-block checkpoint and journal packing, bounded projection; throughput is unqualified until matched-token curve at this batch passes',
                physical_gpu=os.environ['GPU_LEASE_HELD'],allocator_settings=allocator_settings,production_receipt=production_name)
    try:
        with patch.object(backward,'reverse_kernel',RegisterLaunch(original,a.registers)) if a.registers else nullcontext(),overlap() if a.head_streams else nullcontext(),packed_cells(records) if a.deferred_pack else nullcontext(),fused_pool() if a.fused_pool else nullcontext():
            for it in range(a.steps+3):
                torch.cuda.synchronize();start=time.perf_counter();value,calls=step()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if it>=3:samples.append(elapsed);losses.append(value)
                print(it,elapsed,value,calls,flush=True)
            result.update(status='passed',seconds=samples,losses=losses,tokens_per_second=a.batch*512*len(samples)/sum(samples))
            if a.profile or (a.profile_if_gate and result['tokens_per_second']>=100000):
                out=ROOT/a.label;out.mkdir(exist_ok=False)
                with instrumentation() as journals:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        step()
                    path=out/'trace.json';prof.export_chrome_trace(str(path))
                detail=summarize(path);detail['journal_allocations']=journals
                (out/'profile.json').write_text(json.dumps(detail,indent=2)+'\n')
                path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes()));path.unlink()
                result.update(profile_dir=a.label,profile_updates=1,peak_scope='three warmups, 20 measured updates and one separate profile update')
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                      sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
        (ROOT/f'{a.label}.json').write_text(json.dumps(result,indent=2)+'\n')
        if result.get('profile_dir'):(ROOT/a.label/'wall.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
