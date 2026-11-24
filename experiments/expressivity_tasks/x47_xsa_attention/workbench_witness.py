"""Independent byte oracle and the two interventions registered in W1_PROTOCOL."""
import hashlib

import numpy as np
import torch

from .workbench import Decoder, Meter, compact
from .workbench_model import Workbench
from .workbench_native import Archive, Live

PREFIX = b'retry_'
WORLDS = ((b'retry_de', b'retry_de', b'retry_da', b'retry_c'),
          (b'retry_ce', b'retry_ce', b'retry_ca', b'retry_d'))
TOL = 3e-5


def initial_model():
    torch.manual_seed(4701)
    return Workbench().eval()


def digest_state(model):
    h = hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(f'{name}|{t.dtype}|{tuple(t.shape)}\0'.encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def expand(encoded, n=256):
    x = np.full(n, encoded['default'], dtype=np.float64)
    for i,v in encoded['other']:
        x[i] = v
    return x


def scan(docs, pattern, *, closed):
    counts = [0]*257
    for doc in docs:
        for i in range(len(doc)-len(pattern)+1):
            if doc[i:i+len(pattern)] == pattern:
                j = i + len(pattern)
                if j < len(doc):
                    counts[doc[j]] += 1
                elif closed:
                    counts[256] += 1
    return counts


def oracle(prefix, docs, *, suppress_archive=False):
    p = np.full(256, 1/256); weight = 1
    for collection,closed,w in (((prefix,),False,1),(docs,True,4)):
        if not collection or not prefix or (closed and suppress_archive):
            continue
        for depth in sorted({min(8,len(prefix)), min(4,len(prefix)), min(2,len(prefix))}, reverse=True):
            counts = scan(collection,prefix[-depth:],closed=closed)
            if sum(counts):
                total = sum(counts[:256])
                if total:
                    p += w*np.asarray(counts[:256])/total; weight += w
                break
    return p/weight


def close(x, y, tol=TOL):
    a,b = np.asarray(x),np.asarray(y)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('nonfinite or mismatched shapes')
    err = float(np.max(np.abs(a-b)))
    if err > tol:
        raise AssertionError(f'maximum error {err} > {tol}')
    return err


def require(condition, why):
    if not condition:
        raise AssertionError(why)


@torch.inference_mode()
def witness(model):
    device = next(model.parameters()).device
    cases, oracle_errors = [], []
    for world,docs in enumerate(WORLDS):
        selected = b'd' if world == 0 else b'c'
        other = b'c' if world == 0 else b'd'
        with Archive(docs) as archive:
            for obs in (selected, other):
                for erase in (False, True):
                    with Live(PREFIX) as live:
                        engine = Decoder(model, live, archive)
                        steps = []
                        for t,prefix in enumerate((PREFIX, PREFIX+obs)):
                            meter = Meter(device, trace=True)
                            p,g = engine.predict(meter)
                            a = p.cpu().numpy()
                            require((a>0).all() and abs(float(a.sum())-1)<TOL, 'invalid distribution')
                            suppressed = t == 1 and erase and obs == selected
                            oracle_errors.append(close(a, oracle(prefix,docs,suppress_archive=suppressed)))
                            ops = [v['op'] for v in meter.views]
                            expect_ops = (['BASE','FOLLOW','WRITE_EMIT'] if not t else
                                          ['BASE_RECALL','EMIT'] if obs == selected else ['BASE','EMIT'])
                            require(ops == expect_ops, (ops,expect_ops))
                            expect_live = [prefix, prefix[-4:], prefix[-2:]]
                            expect_archive = ([PREFIX,PREFIX+selected] if not t else
                                              [] if obs == selected else [prefix])
                            reqs = meter.requests
                            require([(r['source'],bytes.fromhex(r['key'])) for r in reqs] ==
                                    [('live',k) for k in expect_live]+[('archive',k) for k in expect_archive],
                                    'wrong request order/operand, or duplicate recall read')
                            for r in reqs:
                                arch = r['source'] == 'archive'
                                close(expand(r['h'],257),scan(docs if arch else (prefix,),
                                      bytes.fromhex(r['key']),closed=arch),0)
                            require(all(v['recent'] == prefix.hex() and v['visible'] == len(prefix)
                                        for v in meter.views), 'speculative extension altered observation')
                            handles = [h.key.hex() for h in engine.handles]
                            require(handles == [(PREFIX+selected).hex(),(PREFIX+other).hex()], 'wrong bookmarks')
                            steps.append(dict(requests=reqs, views=meter.views, handles=handles, prob=compact(a), greedy=g))
                            if not t:
                                require(engine.record is not None, 'missing computed write')
                                expected = np.zeros(256); expected[ord('e')]=2/3; expected[ord('a')]=1/3
                                close(engine.record.value.cpu().numpy(),expected)
                                if erase:
                                    engine.erase_payload()
                                engine.observe(obs[0],meter)  # only *after* prediction and interventions
                        cases.append(dict(world=world, observed=obs.hex(), erased=erase, steps=steps))
    deltas = []
    for kept,cut in zip(cases[::2], cases[1::2],strict=True):
        a,b = kept['steps'],cut['steps']
        require(a[0] == b[0], 'erasure changed past prediction')
        require(a[1]['requests']==b[1]['requests'] and a[1]['handles']==b[1]['handles'] and
                [v['op'] for v in a[1]['views']]==[v['op'] for v in b[1]['views']], 'unmatched ablation')
        delta = float(np.abs(expand(a[1]['prob'])-expand(b[1]['prob'])).max())
        hit = kept['observed'] == (b'd' if kept['world']==0 else b'c').hex()
        require(delta > .1 if hit else delta == 0, 'wrong erasure effect')
        deltas.append(delta)
    return dict(cases=cases, max_oracle_error=max(oracle_errors), erase_max_deltas=deltas)


def compare_traces(ref, test):
    """Discrete operands/provenance/flags exact; each of 256 neural lanes compared."""
    errs = []
    for a,b in zip(ref['cases'],test['cases'],strict=True):
        require({k:v for k,v in a.items() if k!='steps'} == {k:v for k,v in b.items() if k!='steps'}, 'case identity')
        for x,y in zip(a['steps'],b['steps'],strict=True):
            require(all(x[k]==y[k] for k in ('requests','handles','greedy')), 'different executed step')
            errs.append(close(expand(x['prob']),expand(y['prob'])))
            for f,g in zip(x['views'],y['views'],strict=True):
                for k,v in f.items():
                    if k in ('recall','write') and v is not None:
                        errs.append(close(expand(v),expand(g[k])))
                    else:
                        require(v == g[k], f'different workspace field {k}')
    require(len(ref['cases']) == len(test['cases']) == 8, 'incomplete parity panel')
    return max(errs)
