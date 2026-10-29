"""Generate a four-launch-per-tick diagnostic reverse, preserving FP32 operands.

This measures a synchronized split implementation, not time within the fused
production binary. Spill/reload and launch overhead must be reported explicitly.
"""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def generate():
    source = (ROOT / 'backward.py').read_text()
    helpers = source[source.index('@triton.jit\ndef slot_sum'):source.index('@triton.jit\ndef reverse_program')]
    start = source.index('@triton.jit\ndef reverse_program')
    program = source[start:source.index('@triton.jit\ndef reverse_kernel', start)]
    signature, body = program.split('    # Wide offsets', 1)
    signature = signature.replace('@triton.jit\n', '@triton.jit(do_not_specialize=["Token"])\n')
    signature = signature.replace('def reverse_program(K,', 'def reverse_kernel(Scalars, Token, Phase:tl.constexpr, K,')
    # Match the public launch ABI, including the prefix bound (unused here).
    signature = signature.replace('KeyGrad:tl.constexpr,GS:', 'KeyGrad:tl.constexpr,InitialBound:tl.constexpr,GS:')
    prelude = '    # Wide offsets' + body[:body.index('    stop=tl.load(Counts+n)')]
    prelude = prelude.replace('    if T<=512:\n        gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)\n', '')
    prelude += '''    t=Token
    inp=n*T+t
    full_lane=tl.arange(0,triton.next_power_of_2(S))
    full_live=tl.load(Live+n*S+full_lane,full_lane<S,False).to(tl.int1)
    inverse=tl.load(PoolInverse+n*S+full_lane,full_lane<S,-1) if PS else full_lane
    tl.device_assert((inverse>=0 if PS else full_lane<BS)|(~full_live),'live row outside compact replay')
    live=tl.load(Live+n*S+s,valid,False).to(tl.int1)
'''
    tick = body[body.index('    for reverse in range(T):'):body.index('    if T<=512:\n        tl.store(GStrength', body.index('    for reverse in range(T):'))]
    tick = '\n'.join(line[4:] if line.startswith('    ') else line for line in tick.splitlines()[1:]) + '\n'
    undo = '''    if Phase==0:
        start=tl.load(Boundaries+inp)
        if t==T-1:
            stop=tl.load(Counts+n)
        else:
            stop=tl.load(Boundaries+inp+1)
        undo_to((Addresses,Before,Widths,Counts,n,Capacity),start,stop,UB)
'''
    store = tick[tick.index('    if T<=512:\n        live='):tick.index('    # Read adjoints cover')]
    store = store.replace('    if T>512:\n        tl.store(GStrength+n*GS+a,gs,valid & (a<GS))', '    tl.store(GStrength+n*GS+a,gs,valid & (a<GS))')
    store = '''    gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)
    coef_grad=tl.load(DCoef+n) if t<T-1 else tl.full((),0.,tl.float32)
    if S<4096 and KeyGrad:
        kd=tl.arange(0,triton.next_power_of_2(DK))
        tl.store(DKout+inp*DK+kd,0.,kd<DK)
''' + store
    store += '''    tl.store(StrengthScratch+n*RS+a,strength,valid & (a<RS))
    tl.store(DCoef+n,coef_grad)
'''
    score = tick[tick.index('    score=tl.full'):tick.index('    weight=prob*strength*live')]
    score = score[:score.index('    if T>512:')]
    score += '''    tl.store(ProbabilityScratch+n*RS+a,prob,valid & (a<RS))
    tl.store(Scalars+n*2,maximum)
    tl.store(Scalars+n*2+1,denom)
'''
    read = tick[tick.index('    weight=prob*strength*live'):]
    begin = read.index('    if T>512:\n        tl.debug_barrier()')
    end = read.index('    gs+=prob*live*dw')
    read = read[:begin] + read[end:]
    read = read.replace('    if T>512:\n        tl.store(GStrength+n*GS+a,gs,valid & (a<GS))', '    tl.store(GStrength+n*GS+a,gs,valid & (a<GS))')
    read = '''    prob=tl.load(ProbabilityScratch+n*RS+a,valid & (a<RS),0.)
    strength=tl.load(StrengthScratch+n*RS+a,valid & (a<RS),0.)
    gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)
    maximum=tl.load(Scalars+n*2)
    denom=tl.load(Scalars+n*2+1)
    tail=tl.where(full_lane<S,tl.exp(-1.e30-maximum),0.)
    scale=2./tl.sqrt(float(DK))
''' + read
    def region(phase, text):
        return f'    if Phase=={phase}:\n' + ''.join('    ' + line + '\n' for line in text.splitlines())
    header = '''"""Generated diagnostic; no production import."""
import triton
import triton.language as tl
from ..x31_tape_cell_fused.all_field_journal import undo_to
from .exact_decay import advance_unit

'''
    path = ROOT / 'part12_split_generated.py'
    path.write_text(header + helpers + signature + prelude + undo + region(1, store) + region(2, score) + region(3, read))
    return importlib.import_module('.part12_split_generated', __package__)


if __name__ == '__main__':
    generate()


class Split:
    """Synchronized exclusive wall regions in the split diagnostic only."""
    def __init__(self):
        from . import backward
        self.names = backward.reverse_kernel.arg_names
        self.kernel = generate().reverse_kernel
        self.seconds = [0.] * 4
        self.calls = 0
        self.launches = 0
        self.records = []

    def __getitem__(self, grid):
        import time
        import torch
        def launch(*args, **kw):
            before = list(self.seconds)
            args = list(args)
            def arg(name):
                return args[self.names.index(name)]
            rows = arg('RS')
            device = arg('K').device
            args[self.names.index('StrengthScratch')] = torch.empty((grid[0], rows), device=device)
            args[self.names.index('ProbabilityScratch')] = torch.empty((grid[0], rows), device=device)
            scalars = torch.empty((grid[0], 2), device=device)
            for token in reversed(range(arg('T'))):
                for phase in range(4):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    self.kernel[grid](scalars, token, phase, *args, **kw)
                    torch.cuda.synchronize()
                    self.seconds[phase] += time.perf_counter() - start
                    self.launches += 1
            self.calls += 1
            self.records.append([v - prior for v, prior in zip(self.seconds, before)])
        return launch
