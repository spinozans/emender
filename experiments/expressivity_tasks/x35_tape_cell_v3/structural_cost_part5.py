"""Exact fresh preparation and store-core isolation; no timing subtraction.

The diagnostic preparation is generated from the production function's AST up
through its scan launch. It executes the real allocation/initialization prefix,
then captures that launch without running it. Kernel-only samples prepare a new
state BEFORE each timed interval. Neither diagnostic replaces a complete gate.
"""
import ast
import copy
import inspect
import statistics
import time
from dataclasses import replace
import torch
from . import cell_fused
from .backing import FIELDS


def preparation():
    source = ast.parse(inspect.getsource(cell_fused._tape_scan))
    function = source.body[0]
    indices = [i for i, node in enumerate(function.body)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Subscript)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == 'scan_kernel']
    assert len(indices) == 1, 'production scan boundary changed'
    index = indices[0]
    launch = copy.deepcopy(function.body[index].value)
    launch.func = ast.Name(id='_capture_launch', ctx=ast.Load())
    function.body = function.body[:index] + [ast.Return(value=ast.Tuple(elts=[
        launch, ast.Call(func=ast.Name(id='FusedState', ctx=ast.Load()), args=[
            ast.Call(func=ast.Name(id='TapeState', ctx=ast.Load()), args=[
                ast.Name(id=name, ctx=ast.Load()) for name in
                ('keys','values','strength','live','age')
            ] + [ast.Attribute(value=ast.Name(id='native',ctx=ast.Load()),
                              attr='usage',ctx=ast.Load())], keywords=[]),
            ast.Name(id='history',ctx=ast.Load())], keywords=[
                ast.keyword(arg='metadata',value=ast.Name(id='metadata',ctx=ast.Load()))])],ctx=ast.Load()))]
    function.name = '_prepare_fresh'
    namespace = dict(vars(cell_fused), _capture_launch=lambda *args, **kwargs:(args,kwargs))
    exec(compile(ast.fix_missing_locations(source), '<production-fresh-preparation>', 'exec'), namespace)
    return namespace[function.name]


def launch(captured):
    args, kwargs = captured
    stores = args[0].shape[0] * args[0].shape[1]
    cell_fused.scan_kernel[(stores,)](*args, **kwargs)


def measure_core(prepare, tokens):
    """Five/twenty isolated launches; fresh preparation excluded explicitly."""
    for _ in range(5):
        captured, state = prepare()
        launch(captured)
        del captured, state
    wall, device = [], []
    for _ in range(20):
        captured, state = prepare()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        begin = time.perf_counter(); start.record()
        launch(captured)
        end.record(); end.synchronize()
        wall.append((time.perf_counter()-begin)*1000/tokens)
        device.append(start.elapsed_time(end)/tokens)
        del captured, state
    return dict(wall_ms_per_token=statistics.median(wall),
                cuda_ms_per_token=statistics.median(device),
                wall_samples_ms_per_token=wall,cuda_samples_ms_per_token=device,
                warmups=5,repetitions=20,
                scope='store core only; new exact fresh preparation before every timed interval')


@torch.no_grad()
def collect(measure):
    prepare = preparation()
    rows = []
    for heads in (1,16):
        cfg = cell_fused.TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=16)
        k = torch.randn(64,16,heads,64,device='cuda')
        v = torch.randn(64,16,heads,128,device='cuda')
        on = torch.ones(64,16,heads,device='cuda')
        off = torch.zeros_like(on)
        def fresh():
            return prepare(k,k,v,on,off,off,None,cfg)
        def fresh_sample():
            captured,state = fresh()
            # account() expects a tensor in the output position. Retain all
            # captured working owners through the timing boundary as well.
            return k,state,captured
        # Exact production setup includes native state, metadata, traces,
        # causal-version owners, input layout copies and free-word setup.
        captured,state = fresh()
        physical = state.native.live.shape[2]
        tensors = [x for x in captured[0] if isinstance(x,torch.Tensor)]
        unique = {x.data_ptr():x for x in tensors}
        byte_count = sum(x.numel()*x.element_size() for x in unique.values())
        del captured,state
        row = dict(component='exact_fresh_preparation',heads=heads,
                   architectural_slots=11776,physical_slots=physical,
                   captured_tensor_bytes=byte_count,tokens=16,
                   **measure(fresh_sample,16))
        rows.append(row)
        row = dict(component='sequential_store_core',heads=heads,
                   architectural_slots=11776,physical_slots=physical,tokens=16,
                   **measure_core(fresh,16))
        rows.append(row)
        # No-write ablation retains the entire exact fresh path and core, but
        # disables append scheduling. It is not the registered input stream.
        row = dict(component='fresh_scan_writes_disabled',heads=heads,
                   architectural_slots=11776,tokens=16,
                   **measure(lambda:cell_fused.tape_scan(k,k,v,off,off,off,None,
                       replace(cfg,budget=11776)),16))
        rows.append(row)
    return rows
