"""Part-8 timing protocol with physical backing accounting outside timings.

Prefix backing is not a live-pool implementation. Structural gates fail closed
regardless of sparse-start performance or footprint improvements.
"""
import gc
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
import torch
from .cell_fused import TapeCell, TapeHeadConfig
from .backing import PrefixNative, FIELDS
from ..x31_tape_cell_fused.cell_fused import TapeCell as X31Cell
from ..x31_tape_cell_fused.cost_gates import account as dense_account
from ..x30_tape_cell.naive_reference import init_state

ROOT = Path(__file__).resolve().parent


def account(sample):
    state = sample[1]
    if not hasattr(state,"native") or not isinstance(state.native, PrefixNative):
        return dense_account(sample)
    native = state.native.compact
    fields = [*(getattr(native,n) for n in FIELDS), state.history, *state.metadata]
    unique = {x.data_ptr():x for x in fields}
    return dict(base_state_bytes=sum(x.numel()*x.element_size() for x in unique.values()),
                journal_reserved_bytes=0, journal_recorded_entries=0,
                physical_slots=native.live.shape[-1])


def measure(fn,tokens):
    # The X31 part-8 measurement loop is copied verbatim. Only account() above
    # understands deferred public tensors; it executes outside timed samples.
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


def save(result):
    (ROOT/'gates_part4.json').write_text(json.dumps(result,indent=2)+'\n')


@torch.no_grad()
def main():
    import ast
    import inspect
    from ..x31_tape_cell_fused.cost_gates import measure as registered_measure
    assert ast.dump(ast.parse(inspect.getsource(measure))) == ast.dump(ast.parse(inspect.getsource(registered_measure))), "registered timing loop changed"
    held = os.environ.get("GPU_LEASE_HELD")
    assert held in list('0123456') and os.environ.get('CUDA_VISIBLE_DEVICES') == held
    from .validate_part4 import source_hashes
    parity = json.loads((ROOT/'parity_part4.json').read_text())
    assert parity['source_sha256'] == source_hashes(), 'parity receipt does not match source'
    suites = ET.parse(ROOT/'validation_part4.xml').getroot()
    assert sum(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites.iter('testsuite')) == 0
    assert sum(int(s.get('tests',0)) for s in suites.iter('testsuite')) >= 158
    torch.set_num_threads(1);torch.manual_seed(0)
    result = dict(physical_gpu=held,host=socket.gethostname(),gpu=torch.cuda.get_device_name(),
        python=platform.python_version(),warmups=5,repetitions=20,batch=64,width=256,key=64,value=128,
        methodology='X31 part8 measurement loop unchanged: five warmups, twenty synchronized full fresh-cell samples, all wall/CUDA samples retained. Physical-state accounting outside timing; deferred public dense access is excluded just as eager state inspection was outside timing.',
        source_sha256=parity['source_sha256'],timing_loop_ast_matches_x31=True,
        measurements=[],errors=[],backing='Allocation-horizon prefix, NOT occupancy-proportional live/cold pools.')
    for slots,heads,tokens,engines in ((2944,4,64,('x31','x35')),(11776,1,16,('x35',)),(11776,16,16,('x35',))):
        cfg = TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=slots)
        x = torch.randn(64,tokens,256,device='cuda')
        ids = torch.ones(64,tokens,device='cuda',dtype=torch.int64)
        candidate = TapeCell(256,cfg).cuda()
        baseline = X31Cell(256,cfg).cuda();baseline.load_state_dict(candidate.state_dict())
        for engine in engines:
            cell = baseline if engine=='x31' else candidate
            try:
                row = dict(engine=engine,slots=slots,heads=heads,tokens=tokens,
                           **measure(lambda:cell(x,ids),tokens))
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                before = torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                sample = cell(x,ids);torch.cuda.synchronize()
                row['incremental_peak_bytes'] = torch.cuda.max_memory_allocated()-before
                row['retained_bytes'] = torch.cuda.memory_allocated()-before
                native = sample[1].native
                if isinstance(native,PrefixNative): native=native.compact
                row['live_rows'] = int(native.live.sum())
                row['logical_rows'] = 64*heads*slots
                del sample,native
                result['measurements'].append(row);print(json.dumps(row),flush=True)
            except Exception as exc:
                row = dict(engine=engine,slots=slots,heads=heads,tokens=tokens,error=repr(exc))
                result['errors'].append(row);print(json.dumps(row),flush=True)
            save(result);gc.collect();torch.cuda.empty_cache()
        del x,ids,candidate,baseline,cell
        gc.collect();torch.cuda.empty_cache()
    result['structural_cost'] = []
    for heads in (1,16):
        cfg = TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=11776)
        physical_cfg = replace(cfg,budget=16)
        # The actual scan initializes a T-row working prefix. This is a
        # component floor, never subtracted from the end-to-end gate.
        row = dict(component='prefix_native_initialization',heads=heads,slots=11776,physical_slots=16,tokens=16,
                   **measure(lambda:(None,init_state(64,physical_cfg,'cuda')),16))
        result['structural_cost'].append(row);print(json.dumps(row),flush=True)
        # Independent component floors, measured with the same T16 loop.
        # They are not added or subtracted to alter either registered gate.
        from .cell_fused import tape_scan
        cell = TapeCell(256,cfg).cuda()
        x = torch.randn(64,16,256,device='cuda')
        ids = torch.ones(64,16,device='cuda',dtype=torch.int64)
        def project():
            k = cell.W_k(x).reshape(64,16,heads,64).detach()
            q = cell.W_q(x).reshape_as(k)
            v = cell.W_v(x).reshape(64,16,heads,128)
            beta = ((cell.W_write(x).sigmoid().detach()>cfg.write_enable_threshold)
                    & (ids!=0).unsqueeze(-1)).float()
            return k,q,v,beta,cell.W_pop(x).sigmoid(),cell.W_append(x)
        row = dict(component='projections_only',heads=heads,slots=11776,tokens=16,
                   **measure(lambda:(project(),None),16))
        result['structural_cost'].append(row);print(json.dumps(row),flush=True)
        projected = project()
        row = dict(component='fresh_scan_without_projections',heads=heads,slots=11776,tokens=16,
                   **measure(lambda:tape_scan(*projected,None,cfg,cell.append_err_coef),16))
        result['structural_cost'].append(row);print(json.dumps(row),flush=True)
        del cell,x,ids,projected
        gc.collect();torch.cuda.empty_cache()
    from .cell_fused import tape_scan
    cfg = TapeHeadConfig(key_dim=64,value_dim=128,n_heads=16,budget=11776)
    k = torch.zeros(64,16,16,64,device='cuda')
    v = torch.zeros(64,16,16,128,device='cuda')
    off = torch.zeros(64,16,16,device='cuda')
    state = None
    result['zero_live_continuation_probe'] = []
    for chunk in (1,2):
        _,state,_,_ = tape_scan(k,k,v,off,off,off,state,cfg)
        native = state.native.compact
        row = dict(chunks=chunk,tokens_per_chunk=16,live_rows=int(native.live.sum()),
                   **account((None,state)))
        result['zero_live_continuation_probe'].append(row)
        print(json.dumps(dict(component='zero_live_continuation_probe',**row)),flush=True)
    del k,v,off,state,native
    gc.collect();torch.cuda.empty_cache()
    rows = {(r['engine'],r['slots'],r['heads']):r for r in result['measurements']}
    a,b,c,d = (rows.get(key) for key in (('x31',2944,4),('x35',2944,4),('x35',11776,1),('x35',11776,16)))
    speed = None if a is None or b is None else a['wall_ms_per_token']/b['wall_ms_per_token']
    heads = None if c is None or d is None else d['wall_ms_per_token']/c['wall_ms_per_token']
    result.update(forward_speedup=speed,heads_ratio=heads,gates={
        'c64_s2944_h4_forward_ge_2x':None if speed is None else speed>=2,
        's11776_t16_heads_ratio_le_2':None if heads is None else heads<=2,
        'occupancy_proportional_backing':False,
        'live_proportional_memory':False},
        structural_gate_reason='Prefix size grows with cumulative chunk lengths, including zero-live scans. No live-row pool, separate cold storage or nonidentity logical map. Sparse-start footprint alone cannot certify live-proportional memory.')
    save(result);print(json.dumps({k:result[k] for k in ('forward_speedup','heads_ratio','gates')}),flush=True)


if __name__ == '__main__':main()
