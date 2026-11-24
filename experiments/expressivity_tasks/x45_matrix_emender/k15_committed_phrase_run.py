"""CPU-only census or synthetic mechanical witness; no panel scoring."""
import hashlib
import json
import os
import resource
import sys
import time
from copy import copy
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from .k15_committed_phrase import Factorizer, Index, scan_sources, targets
from experiments.expressivity_tasks.x43_fork_resolved.xsa_binding_live import Session

EXE = Path.home()/'sxgc/xsa/target/debug/xsa'


class Meter(Session):
    def __init__(self, stats):
        super().__init__(EXE)
        self.stats = stats

    def rpc(self, cmd):
        t = time.perf_counter()
        res = super().rpc(cmd)
        self.stats['rpc_seconds'] += time.perf_counter()-t
        self.stats[cmd.split()[0]] += 1
        for k in ('sealed_rows','tail_starts'):
            self.stats[k] += res.get(k,0)
        return res


@contextmanager
def opened(stats):
    a,b = Meter(stats),Meter(stats)
    try: yield Index(a,b)
    finally: a.close(); b.close()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census(stats):
    path = Path('output/k15O/panels.json')
    assert digest(path)=='ccd8ae76c12e984d4520193d4e334dfaf72a03f8ad9606e5e7aa0af3a8e0ac88'
    panel = json.loads(path.read_bytes()); ident = panel['identity']
    assert digest(ident['manifest'])==ident['manifest_sha256']
    lengths,counts,documents = Counter(),Counter(),[]
    trace = None
    with open(ident['source'],'rb') as stream:
        for d in panel['train'][:16]:
            assert d['split']=='train'
            stream.seek(d['offset']); raw=stream.read(d['length'])
            assert hashlib.sha256(raw).hexdigest()==d['sha256']
            documents.append(d['doc']); raw=raw[:4096]
            with opened(stats) as index:
                f=index.parse
                for t,y in enumerate(raw):
                    before=time.perf_counter(); depth, ids=index.sources()
                    stats['native_sources_seconds']+=time.perf_counter()-before
                    tic=time.perf_counter(); ref=scan_sources(f)
                    stats['reference_seconds']+=time.perf_counter()-tic
                    assert ref==(depth,ids)
                    contents=[bytes(f.raw[f.rows[i].start:f.rows[i].end]) for i in ids]
                    labels=targets(contents,raw[t:t+4]); lens=labels['lengths']
                    delayed=any(a and b and a!=b for a in lens for b in lens)
                    byte_fork=any(0 < a < b and a < len(content) for content,a in zip(contents,lens) for b in lens)
                    counts.update(dict(byte_fork=int(byte_fork),bytes=1,closed=int(f.start==t),legal=int(bool(ids)),ambiguous=int(len(ids)>1),multi=int(depth==2),delayed=int(delayed)))
                    rows=index.observe(y)
                    if trace is None and delayed:
                        trace=dict(doc=d['doc'],offset=t,suffix_depth=depth,source_rows=ids,spans=[vars(f.rows[i]) for i in ids],contents=[b.hex() for b in contents],observed=y,offline_labels=labels,after=dict(commits=[vars(r) for r in rows],open_start=f.start,next_sources=index.sources()))
                if len(raw)==d['length']: index.end()
                lengths.update(r.end-r.start for r in f.rows)
                repeated=sum(n>1 for n in Counter(r.kind for r in f.rows).values())
                counts.update(dict(repeated_types=repeated,commits=len(f.rows),distinct_types=len(f.types),repeat_occurrences=len(f.rows)-len(f.types)))
    return dict(counts=counts,lengths=lengths,documents=documents,trace=trace)


def shared_witness_byte(first, second):
    assert first[0] == second[0], "forced sources disagree on observation"
    return first[0]


def check_witness_gradients(intact, cleared):
    assert intact > 0 and cleared == 0, "expected only intact read to carry gradient"


def witness(stats):
    import torch
    from .k15_prefix_machine import PrefixK15
    from .k15_phrase_qualification import backbone
    from .k15_committed_phrase_machine import Machine
    assert torch.get_num_threads()==4
    torch.manual_seed(3411); model=PrefixK15(backbone()); results=[]; laws=[]; literals=[]
    with opened(stats) as index:
        m=Machine(model,index,b'z'*128)
        for y in b'#ABCD'+b'ABCD'*8:
            _,ids=index.sources(); f=index.parse
            if len(ids)>=2 and all(f.rows[i].end-f.rows[i].start==4 for i in ids[:2]): break
            with torch.no_grad(): m.predict()
            m.observe(y)
        else: raise RuntimeError('no ambiguous newborn')
        shared=m
        observed=shared_witness_byte(*(f.raw[f.rows[k].start:f.rows[k].end] for k in ids[:2]))
    for variant in ('older','alternative','clear'):
        with opened(stats) as index:
            for y in shared.index.parse.raw: index.observe(y)
            assert index.sources()[1]==ids
            m=copy(shared); m.values=list(shared.values); m.index=index; f=index.parse
            i=ids[0 if variant=='alternative' else 1]; value=m.values[i]
            law=m.predict(force=i,clear=variant=='clear'); h,lit,edges,_=m.pending
            literals.append(lit.detach()); start=sum(e[3] for e in edges if e[0]==1)
            n=len(f.raw); rows=[vars(f.rows[k]) for k in ids]; m.observe(observed)
            m.predict(); later=m.pending[1]; laws.append(later.detach())
            grad=torch.autograd.grad(later[observed].log(),value,allow_unused=True,retain_graph=True)[0]
            results.append(dict(variant=variant,observed=observed,position=n,ids=ids,source=i,rows=rows,type_hex=bytes(f.raw[f.rows[i].start:f.rows[i].end]).hex(),activation_after=f.rows[i].end,passes=m.calls,start_mass=float(start.detach()),gradient_norm=float(grad.norm()) if grad is not None else 0))
    assert all(torch.equal(literals[0],p) for p in literals) and len(set(r['start_mass'] for r in results))==1
    delta=[float((p-laws[0]).abs().max()) for p in laws[1:]]
    assert min(delta)>0
    check_witness_gradients(results[0]["gradient_norm"],results[2]["gradient_norm"])
    return dict(results=results,next_literal_linf=delta,current_literals_identical=True)


if __name__=='__main__':
    assert __debug__ and os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='4'
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    stats=Counter(); tic=time.perf_counter(); cpu=time.process_time()
    result={'census':census,'witness':witness}[sys.argv[1]](stats)
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    print(json.dumps(dict(result=result,native=stats,wall=time.perf_counter()-tic,cpu=time.process_time()-cpu,child_cpu=child.ru_utime+child.ru_stime),indent=2))
