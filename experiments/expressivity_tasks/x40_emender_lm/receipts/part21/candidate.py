"""Process-local candidate. Production source files are never changed.

Initialize only the three owner arrays copied wholesale by replay/freeze.
Their dead lanes cannot safely be excluded using final-live: undo restores
historically live rows. Counted journal tails are excluded by a masked copy.
"""
import hashlib
import ast
from pathlib import Path
import torch
import triton
import triton.language as tl


@triton.jit
def copy_counted(Source, Destination, Counts, Capacity: tl.constexpr, Block: tl.constexpr):
    n=tl.program_id(0).to(tl.int64)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    active=(x<Capacity) & (x<tl.load(Counts+n))
    value=tl.load(Source+n*Capacity+x,active,0)
    tl.store(Destination+n*Capacity+x,value,x<Capacity)


def counted_clone(source, counts):
    destination=torch.empty_like(source)
    copy_counted[(len(counts),triton.cdiv(source.shape[1],256))](
        source,destination,counts,source.shape[1],256)
    return destination


def install():
    from experiments.expressivity_tasks.x31_tape_cell_fused import kernels as x31
    from experiments.expressivity_tasks.x40_emender_lm import kernels as x40
    for module in (x31,x40):
        fn=module.materialize_state
        old='born=tl.load(Birth+n*S+slots,valid,Clock)'
        assert fn.src.count(old)==1
        fn._unsafe_update_src(fn.src.replace(old,'born=tl.load(Birth+n*S+slots,valid & live,Clock)'))
        fn.hash=None
    fn=x40.scan_kernel
    old='cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)'
    assert fn.src.count(old)==1
    fn._unsafe_update_src(fn.src.replace(old,'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live & (r_touch<Start+t-1),0.)'))
    fn.hash=None
    from experiments.expressivity_tasks.x31_tape_cell_fused import cell_fused, backward
    from experiments.expressivity_tasks.x36_tape_perf import forward as f36, cell_fused as c36
    from experiments.expressivity_tasks.x40_emender_lm import forward as f40
    hashes={}
    for module in (cell_fused,f36,f40):
        path=Path(module.__file__)
        source=path.read_text()
        original=source
        for name in ('due','terminal'):
            old=f'{name} = torch.empty('
            assert source.count(old)==1
            source=source.replace(old,f'{name} = torch.zeros(')
        assert source.count('births = torch.empty_like(due)')==1
        source=source.replace('births = torch.empty_like(due)','births = torch.zeros_like(due)')
        tree=ast.parse(source,str(path))
        # Preserve FusedState identity held by already-imported replay modules.
        tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='tape_scan']
        assert len(tree.body)==1
        exec(compile(tree,str(path),'exec'),module.__dict__)
        hashes[str(path)]={'production':hashlib.sha256(original.encode()).hexdigest(),
                           'candidate':hashlib.sha256(source.encode()).hexdigest()}
    c36.persistent_scan=f36.tape_scan
    source=Path(backward.__file__).read_text()
    old='addresses=original_addresses.clone()'
    assert source.count(old)==1
    source=source.replace(old,'addresses=counted_clone(original_addresses,counts)')
    backward.counted_clone=counted_clone
    exec(compile(source,backward.__file__,'exec'),backward.__dict__)
    return hashes


if __name__=='__main__':
    import json
    import os
    import sys
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3','4')
    torch.set_num_threads(1)
    print(json.dumps({'candidate_sources':install()}),flush=True)
    from experiments.expressivity_tasks.x40_emender_lm.parity import certify
    print(json.dumps(certify(batch=1,context=8)),flush=True)
