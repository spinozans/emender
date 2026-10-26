"""End-to-end wall and CUDA-event cost receipts, after full first-order parity."""
import argparse
import gc
import json
import os
import statistics
import time
import torch
import triton
import platform
import socket
from datetime import datetime,timezone
from triton.runtime.errors import OutOfResources
from triton.compiler.errors import CompilationError
from .cell_fused import TapeCell,TapeHeadConfig
from .train_gates import require_parity,save
from ..x30_tape_cell.cell import TapeCell as X30Cell
from ..x28e_memory_head.eng import tape_head as x28e


def account(sample):
    accounting={}
    state=sample[1]
    if hasattr(state,'native'):
        fields=[state.native.keys,state.native.values,state.native.strength,state.native.live,
                state.native.age,state.native.usage,state.history,*state.metadata]
        unique={x.data_ptr():x for x in fields}
        accounting['base_state_bytes']=sum(x.numel()*x.element_size() for x in unique.values())
        journal=state.all_field_journal
        accounting['journal_reserved_bytes']=0 if journal is None else journal.nbytes
        accounting['journal_recorded_entries']=0 if journal is None else int(journal.counts.sum())
        node=sample[0].grad_fn
        if node is not None and hasattr(node,'log'):
            accounting['journal_reserved_bytes']=sum(x.numel()*x.element_size() for x in node.log)
            accounting['journal_recorded_entries']=int(node.log[3].sum())
            accounting['private_replay_copy_bytes']=sum(x.numel()*x.element_size() for x in node.replay)
    return accounting


def measure(fn,tokens):
    for warmup in range(5):
        sample=fn()
        if warmup==4:accounting=account(sample)
        del sample
    torch.cuda.synchronize();wall=[];device=[]
    for _ in range(20):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();begin=time.perf_counter();start.record()
        result=fn();end.record();end.synchronize()
        wall.append((time.perf_counter()-begin)*1000/tokens)
        device.append(start.elapsed_time(end)/tokens)
        del result
    return dict(wall_ms_per_token=statistics.median(wall),cuda_ms_per_token=statistics.median(device),warmups=5,repetitions=20,wall_samples_ms_per_token=wall,cuda_samples_ms_per_token=device,**accounting)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parity-receipt',required=True)
    args=p.parse_args();require_parity(args.parity_receipt)
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:raise RuntimeError('lease required')
    torch.set_num_threads(1);torch.manual_seed(0)
    receipt=dict(physical_gpu=held,host=socket.gethostname(),timestamp_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,triton=triton.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,dtype='fp32',batch=64,tokens=16,width=256,key=64,value=128,scope='forward only, fresh state per call; full cell projections plus state/scan; allocation and synchronization included',measurements=[],errors=[])
    for slots,heads in [(736,4),(736,1),(736,16),(11776,4),(11776,1),(11776,16)]:
        cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=slots)
        x=torch.randn(64,16,256,device='cuda')
        ids=torch.ones(64,16,device='cuda',dtype=torch.int64)
        cell=TapeCell(256,cfg).cuda()
        old=X30Cell(256,cfg).cuda();old.load_state_dict(cell.state_dict())
        eager=x28e.TapeHead(256,cfg).cuda();eager.load_state_dict(cell.state_dict())
        engines={'x31':lambda:cell(x,ids),
                 'x30':lambda:old(x,ids),
                 'x28e':lambda:eager(x)}
        if slots==11776 and heads==4:
            def checkpoint():
                with torch.enable_grad():return cell(x,ids)
            engines['x31_checkpoint']=checkpoint
        for name,fn in engines.items():
            if heads!=4 and name!='x31':continue
            try:
                row=dict(slots=slots,heads=heads,engine=name,
                         num_warps=4 if name.startswith('x31') else None,
                         **measure(fn,16))
                receipt['measurements'].append(row);print(json.dumps(row),flush=True)
            except (RuntimeError,MemoryError,OutOfResources,CompilationError) as exc:
                error=dict(slots=slots,heads=heads,engine=name,error_type=type(exc).__name__,error=str(exc))
                receipt['errors'].append(error);print(json.dumps(error),flush=True)
            save('cost_part5.json',receipt);gc.collect();torch.cuda.empty_cache()
        del x,ids,cell,old,eager,engines;gc.collect();torch.cuda.empty_cache()
    rows={(x['slots'],x['heads'],x['engine']):x['wall_ms_per_token'] for x in receipt['measurements']}
    def ratio(a,b):return None if a not in rows or b not in rows else rows[a]/rows[b]
    receipt['x30_speedups']={str(s):ratio((s,4,'x30'),(s,4,'x31')) for s in (736,11776)}
    receipt['heads16_to_1_by_slots']={str(s):ratio((s,16,'x31'),(s,1,'x31')) for s in (736,11776)}
    receipt['heads16_to_1']=receipt['heads16_to_1_by_slots']['11776']
    overhead=ratio((11776,4,'x31_checkpoint'),(11776,4,'x31'))
    receipt['checkpoint_overhead']=None if overhead is None else overhead-1
    receipt['gates']={
        's736_speedup_ge_1':None if receipt['x30_speedups']['736'] is None else receipt['x30_speedups']['736']>=1,
        's11776_speedup_ge_10':None if receipt['x30_speedups']['11776'] is None else receipt['x30_speedups']['11776']>=10,
        'heads_ratio_le_2':None if receipt['heads16_to_1'] is None else receipt['heads16_to_1']<=2,
        'checkpoint_overhead_le_5pct':None if overhead is None else overhead<=1.05}
    save('cost_part5.json',receipt)


if __name__=='__main__':main()
