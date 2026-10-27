"""Registered full-cell gates. No component timing substitutes for cell cost."""
import gc
import json
import os
from pathlib import Path
import platform
import socket
import torch
from .cell_fused import TapeCell, TapeHeadConfig
from ..x31_tape_cell_fused.cell_fused import TapeCell as X31Cell
from ..x31_tape_cell_fused.cost_gates import measure
from ..x30_tape_cell.naive_reference import init_state

ROOT=Path(__file__).resolve().parent


def save(result):
    (ROOT/'gates_part3.json').write_text(json.dumps(result,indent=2)+'\n')


@torch.no_grad()
def main():
    held=os.environ.get('GPU_LEASE_HELD')
    assert held in list('0123456') and os.environ.get('CUDA_VISIBLE_DEVICES')==held
    # This file is generated only by the completed unchanged-oracle run.
    import xml.etree.ElementTree as ET
    suites=ET.parse(ROOT/'validation_part3.xml').getroot()
    assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites.iter('testsuite'))==0
    assert sum(int(s.get('tests',0)) for s in suites.iter('testsuite'))>=109
    torch.set_num_threads(1);torch.manual_seed(0)
    result=dict(physical_gpu=held,host=socket.gethostname(),gpu=torch.cuda.get_device_name(),
        python=platform.python_version(),warmups=5,repetitions=20,batch=64,width=256,key=64,value=128,
        methodology='X31 part8 measure() unchanged: full cell projections, fresh-state allocation, synchronized wall and CUDA events; medians per token.',
        measurements=[],errors=[],backing='Dense X31-compatible backing; occupancy-proportional target NOT implemented.')
    for slots,heads,tokens,engines in ((2944,4,64,('x31','x35')),(11776,1,16,('x35',)),(11776,16,16,('x35',))):
        cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=slots)
        x=torch.randn(64,tokens,256,device='cuda')
        ids=torch.ones(64,tokens,device='cuda',dtype=torch.int64)
        candidate=TapeCell(256,cfg).cuda()
        baseline=X31Cell(256,cfg).cuda();baseline.load_state_dict(candidate.state_dict())
        for engine in engines:
            cell=baseline if engine=='x31' else candidate
            try:
                row=dict(engine=engine,slots=slots,heads=heads,tokens=tokens,**measure(lambda:cell(x,ids),tokens))
                # Memory separately: allocator peak after cache clear, one complete
                # fresh-state call, no retained warmup sample or dense clone.
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                sample=cell(x,ids);torch.cuda.synchronize()
                row['incremental_peak_bytes']=torch.cuda.max_memory_allocated()-before
                row['retained_bytes']=torch.cuda.memory_allocated()-before
                row['live_rows']=int(sample[1].native.live.sum())
                row['logical_rows']=64*heads*slots
                del sample
                result['measurements'].append(row)
                print(json.dumps(row),flush=True)
            except Exception as exc:
                error=dict(engine=engine,slots=slots,heads=heads,tokens=tokens,error=repr(exc))
                result['errors'].append(error);print(json.dumps(error),flush=True)
            save(result);gc.collect();torch.cuda.empty_cache()
        del x,ids,candidate,baseline,cell
        gc.collect();torch.cuda.empty_cache()
    result['native_initialization_floor']=[]
    for heads in (1,16):
        cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=11776)
        row=dict(heads=heads,slots=11776,tokens=16,**measure(lambda:(None,init_state(64,cfg,'cuda')),16))
        result['native_initialization_floor'].append(row)
        print(json.dumps(dict(component='native_initialization_floor',**row)),flush=True)
        gc.collect();torch.cuda.empty_cache()
    rows={(r['engine'],r['slots'],r['heads']):r for r in result['measurements']}
    a=rows.get(('x31',2944,4));b=rows.get(('x35',2944,4))
    c=rows.get(('x35',11776,1));d=rows.get(('x35',11776,16))
    speed=None if a is None or b is None else a['wall_ms_per_token']/b['wall_ms_per_token']
    heads=None if c is None or d is None else d['wall_ms_per_token']/c['wall_ms_per_token']
    result.update(forward_speedup=speed,heads_ratio=heads,gates={
        'c64_s2944_h4_forward_ge_2x':None if speed is None else speed>=2,
        's11776_t16_heads_ratio_le_2':None if heads is None else heads<=2,
        'occupancy_proportional_backing':False,
        'live_proportional_memory':False})
    save(result);print(json.dumps({k:result[k] for k in ('forward_speedup','heads_ratio','gates')}),flush=True)


if __name__=='__main__':main()
