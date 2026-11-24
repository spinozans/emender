"""Window-local O floor + committed-phrase controller; no optimizer or evaluation I/O."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time

import torch
from .k15_committed_phrase import Index
from .k15_committed_phrase_machine import Machine
from .k15_committed_phrase_run import EXE, Meter, digest
from .k15_context_credit import OutcomeRows, evidence, mass
from .k15_phrase_query import PhraseQuery
from .k15_prefix_machine import PrefixK15

PANEL = Path('output/k15O/panels.json')
PANEL_SHA = 'ccd8ae76c12e984d4520193d4e334dfaf72a03f8ad9606e5e7aa0af3a8e0ac88'
ROOT = Path(__file__).parent/'receipts/k15_natural_preflight'


def documents():
    assert digest(PANEL) == PANEL_SHA
    panel = json.loads(PANEL.read_bytes()); ident = panel['identity']
    assert digest(ident['manifest']) == ident['manifest_sha256']
    with open(ident['source'], 'rb') as f:
        for doc in panel['train'][:16]:
            assert doc['split'] == 'train'
            f.seek(doc['offset']); raw = f.read(doc['length'])
            assert len(raw) == doc['length'] and hashlib.sha256(raw).hexdigest() == doc['sha256']
            yield doc, raw[:4096]


@contextmanager
def opened(stats, cap):
    class Limited(Meter):
        def rpc(self, cmd):
            if stats['total'] >= cap:
                raise RuntimeError('native cap')
            stats['total'] += 1
            return super().rpc(cmd)
    a, b = Limited(stats), Limited(stats)
    try:
        yield Index(a,b)
    finally:
        a.close(); b.close()


def forks(contents, future):
    """Visible conflicts vs truth-adjudicated contests before either span ends."""
    visible = []; adjudicated = []
    for i,x in enumerate(contents):
        for j,y in enumerate(contents[:i]):
            for k,(a,b) in enumerate(zip(x,y)):
                if a != b:
                    if 0 < k < len(future):
                        visible.append((k,i,j))
                        if x[:k] == future[:k] and future[k] in (a,b):
                            adjudicated.append((k,i,j))
                    break
    return visible, adjudicated


def census(stats):
    counts = Counter(); docs = []; windows = []; selected = None; rank = 2
    with gzip.open(ROOT/'sites.jsonl.gz','xt') as out:
        for doc,raw in documents():
            docs.append([doc['doc'],doc['sha256']])
            for s in range(0,len(raw)-135,64):
                windows.append((doc['doc'],s))
                with opened(stats,400000) as ix:
                    for y in raw[s:s+128]: ix.observe(y)
                    for t in range(128,136):
                        depth,ids = ix.sources(); spans = [ix.parse.rows[i] for i in ids]
                        # A mismatch can publish a prefill-ending row on the first observation.
                        live = [r.end>128 for r in spans]
                        content = [bytes(ix.raw.rpc(f'byte {p}')["byte"] for p in range(r.start,r.end)) for r in spans]
                        future = raw[s+t:s+136]; vis, adj = forks(content,future)
                        counts.update(dict(targets=1, eligible=int(bool(ids)), no_contest=int(len(ids)<2), multiple=int(len(ids)>1), visible_delayed=int(bool(vis)), adjudicated=int(bool(adj)), live_candidate_sites=int(any(live)), live_candidate_rows=sum(live), prefill_candidate_rows=len(ids)-sum(live)))
                        if ids:
                            print(json.dumps([doc['doc'],s,t,depth,ids,[x.hex() for x in content],live,vis,adj]),file=out)
                        for k,i,j in adj:
                            priority = int(not(live[i] or live[j]))
                            if priority < rank:
                                selected = [doc['doc'],s,t,k]; rank = priority
                        rows = ix.observe(future[0]); counts['live_writes'] += sum(r.end>128 for r in rows)
    return dict(docs=docs,windows=len(windows),counts=counts,selected=selected,control=windows[random.Random(3411).randrange(len(windows))],sites_sha=digest(ROOT/'sites.jsonl.gz'))


class NaturalMachine(Machine):
    def __init__(self,model,index,prefix,controller,weights):
        super().__init__(model,index,prefix,controller)
        self.reliability = OutcomeRows(); self.weights = weights; self.floor_calls = 0

    def _write(self,row,hidden):
        self.row_memory.append(self.controller.encode(self.index.parse.raw[row.start:row.end]))
        self.values.append(torch.zeros_like(self.feedback) if hidden is None else self.m.write(hidden).tanh())

    def literal(self,raw,live):
        with torch.no_grad():
            _,p = self.m.neural(raw,torch.zeros_like(self.feedback)); self.floor_calls += 1
            counts = {}
            for depth in (8,4,2):
                r = self.index.raw.rpc('branch '+raw[-depth:].hex()); assert r['end']==0
                counts = {i:v for i,v in enumerate(r['counts']) if v}
                if counts: break
            base = p.numpy(); hist,features = evidence(base,counts)
            a = mass(features,self.reliability.read(b'')[0],'unkeyed',self.weights)
            self.floor_observation = base,hist
            floor = (1-a)*p+a*p.new_tensor(hist)
        return (15*floor+live)/16

    def observe(self,y):
        self.reliability.observe(b'',*self.floor_observation,y)
        super().observe(y)


def all_byte_window_loss(model, head, weights, index, window):
    """Future uniform-window objective. Caller owns a fresh native index; no step here."""
    if len(window) != 136 or index.parse.raw:
        raise ValueError("requires a fresh index and one complete 128+8 window")
    state = NaturalMachine(model, index, window[:128], head, weights)
    losses = []
    for target in window[128:]:
        law = state.predict()
        losses.append(-law[target].log())
        state.observe(target)
    return torch.stack(losses).mean()


def birth():
    from .k15_phrase_qualification import backbone, ANCHOR_SHA
    torch.manual_seed(3411); m = PrefixK15(backbone()).eval(); h = PhraseQuery(m.write.out_features)
    with torch.no_grad():
        m.phase.bias.copy_(torch.tensor([-math.log(8),0.])); h.cursor.weight.mul_(.1)
    path = PANEL.with_name('checkpoint.json'); assert digest(path)=='6c95042e21fbbf11b5bb59ebbc656f1e54658a1896a270eeb727f6eea47f69dc'
    ck = json.loads(path.read_bytes()); assert ck['panels']==PANEL_SHA and ck['anchor']==ANCHOR_SHA
    return m,h,ck['weights']['unkeyed']


def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


RSS_CAP_KIB = 10 * 1024**2


def reached_keep_logits(contents, shared, cursor_logits):
    offset = len(shared)
    return {f'keep:{i}:{offset}': cursor_logits[i,offset]
            for i,content in contents.items()
            if len(content)>offset and content.startswith(shared)}


def joint_credit_passed(norms, ids, source, signs):
    return (set(signs.values()) == {False,True}
            and any(f'value:{i}' in norms for i in signs)
            and all(v is not None and math.isfinite(v) and v>0 for v in norms.values())
            and all((source[ids.index(i)]<0 if won else source[ids.index(i)]>0)
                    for i,won in signs.items()))


def run_slice(stats,params,raw,t,k,control):
    tic = time.monotonic(); model,head,w = params
    with opened(stats,10000) as ix:
        m = NaturalMachine(model,ix,raw[:128],head,w); inputs = {}; laws=[]; signs={}
        for pos in range(128,t+k+1):
            law = m.predict(); assert law.min()>0 and abs(law.sum().item()-1)<1e-5
            laws.append(law); y = raw[pos]
            if pos==t and not control:
                _,ids = ix.sources(); inputs.update(start=m.scores[0], source=m.scores[1], query=head.query.weight, write=model.write.weight)
                contents = {}
                for i in ids:
                    r=ix.parse.rows[i]; content=raw[r.start:r.end]; contents[i]=content
                    if content[:k]==raw[t:t+k] and len(content)>k:
                        signs[i]=content[k]==raw[t+k]
                        if m.values[i].requires_grad: inputs[f'value:{i}']=m.values[i]
            if pos>t and not control:
                inputs.update(reached_keep_logits(contents,raw[t:pos],m.cursor_logits))
            m.observe(y); assert abs(sum(m.post.values()).item()-1)<1e-5
            peak = rss()
            if peak>RSS_CAP_KIB:
                return dict(stop='forward RSS cap',passed=False,rss_kib=peak,seconds=time.monotonic()-tic,forwards=m.calls+m.floor_calls,backward=0)
        loss=-laws[-1][raw[t+k]].log(); inputs=inputs or {'phase':model.phase.weight}
        gs=torch.autograd.grad(loss,list(inputs.values()),allow_unused=True)
        norms={n:None if g is None else g.norm().item() for n,g in zip(inputs,gs)}
        result=dict(forwards=m.calls+m.floor_calls,backward=1,rss_kib=rss(),seconds=time.monotonic()-tic,norms=norms)
        if result['rss_kib']>RSS_CAP_KIB:
            return dict(result,stop='post-backward RSS cap',passed=False)
        if not control:
            src=gs[1].tolist(); result.update(source=src,contest=signs,ids=ids)
            result['passed']=joint_credit_passed(norms,ids,src,signs)
        return result


def credit(stats):
    prep=json.loads((ROOT/'prep.json').read_text())['result']; assert digest(ROOT/'sites.jsonl.gz')==prep['sites_sha']
    payload={d['doc']:r for d,r in documents()}; params=birth(); d,s=prep['control']; pilot=run_slice(stats,params,payload[d][s:s+136],128,1,True)
    print(json.dumps({'pilot':pilot}),flush=True)
    if 'stop' in pilot: return dict(pilot=pilot,stop=pilot['stop'])
    if pilot['seconds']>90: return dict(pilot=pilot,stop='pilot price')
    if prep['selected'] is None: return dict(pilot=pilot,stop='no delayed site')
    d,s,t,k=prep['selected']; target=run_slice(stats,params,payload[d][s:s+136],t,k,False)
    if 'stop' in target: return dict(pilot=pilot,target=target,stop=target['stop'])
    return dict(pilot=pilot,target=target)


def main():
    cli=argparse.ArgumentParser(); cli.add_argument('mode',choices=['prep','credit-check','train']); mode=cli.parse_args().mode
    if mode=='train': cli.error('natural training disabled')
    assert __debug__ and os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.get_num_threads()==4
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    ROOT.mkdir(exist_ok=True); tic=time.monotonic(); stats=Counter()
    result=(census if mode=='prep' else credit)(stats)
    with (ROOT/f'{mode}.json').open('x') as f: json.dump(dict(result=result,native=stats,seconds=time.monotonic()-tic,rss_kib=rss()),f,indent=2)


if __name__=='__main__': main()
