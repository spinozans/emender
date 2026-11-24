"""Complete registered W2 cross, independent document oracle and trace validation."""
from contextlib import ExitStack

import numpy as np
import torch

from .workbench import Meter
from .workbench_native import Live
from .workbench_witness import close, expand, require, scan
from .w2_join import Episode, FixedJoin, Neural, ReplicatedMount, array
from .w2_model import BindingWorkbench

WORLDS=((b'key=aM',b'key=bS'),(b'key=aS',b'key=bM'))
TAGS=b'MS'
ARMS=('full','erase','blind','fixed')


def initial_model():
    torch.manual_seed(4701)  # same W1 backbone initialization
    return BindingWorkbench().eval()


def prompt(tag):
    return b'unit='+bytes([tag])+b';0123456789;key='


def analytic(docs, tag, resolved):
    """Look at source documents for verification only; not the fixed-runtime baseline."""
    p=np.zeros(256); all_codes=[doc[4] for doc in docs]; gold,=[doc[4] for doc in docs if doc[5]==tag]
    for code in ([gold] if resolved else all_codes):
        p[code]+=1
    return (1/256+4*p/p.sum())/5,gold


def prefill(engine, stream, meter):
    for byte in stream:
        engine.predict(meter); engine.observe(byte,meter)


def check_case(case):
    w,tag,arm=case['world'],case['tag'],case['arm']; docs=WORLDS[w]; stream=prompt(tag)
    views=case['views']; n=len(stream)
    require(len(stream)>14 and tag not in stream[-8:],'cue visible in recent view')
    require([v['at'] for v in views]==[*range(n),n,n,n], 'predict count/order')
    ops=[('STORE_EMIT' if i==6 else 'EMIT') for i in range(n)]+['PARENT','CHILDREN','EMIT']
    require([v['op'] for v in views]==ops,'store timing / dependent schedule')
    for v in views:
        i=v['at']; require(bytes.fromhex(v['tail'])==stream[max(0,i-8):i], 'wrong causal visibility')
        require(v['written_at']==(6 if i>6 else None),'stale/early record')
    saved=views[6]['write']
    if arm=='fixed':
        require(saved==tag,'incorrect fixed write')
    else:
        one=np.zeros(256); one[tag]=1; close(expand(saved),one)
    expected=['key=', 'key=a','key=b']
    require([(r['at'],bytes.fromhex(r['key']).decode()) for r in case['requests']]==[(n,k) for k in expected],
            'wrong operand, order, or cue re-read')
    for r in case['requests']:
        close(expand(r['h'],257),scan(docs,bytes.fromhex(r['key']),closed=True),0)
    require(views[-2]['rank']==list(b'ab'), 'parent did not issue both bound edges')
    neutral,_=analytic(docs,tag,False)
    predicted,gold=analytic(docs,tag,arm in ('full','fixed'))
    err=close(expand(views[-2]['prob']),neutral)
    p=expand(views[-1]['prob']); err=max(err,close(p,predicted))
    require(np.all(p>0) and abs(sum(p)-1)<3e-5,'invalid distribution')
    require(case['byte']==int(p.argmax()) and case['reads']==3 and case['passes']==(0 if arm=='fixed' else n+3),
            'greedy output or work budget')
    if arm in ('full','fixed'):
        require(case['byte']==gold and p[gold]-neutral[gold]>.3,'failed current-byte improvement')
    for v in views[:-2]:
        close(expand(v['prob']),np.full(256,1/256))
    return err


def verify(cases):
    require([(c['world'],c['tag'],c['arm']) for c in cases]==[(w,t,a) for w in range(2) for t in TAGS for a in ARMS],
            'panel missing/reordered')
    err=max(check_case(c) for c in cases)
    for i in range(0,len(cases),4):
        active=cases[i]
        for control in cases[i+1:i+4]:
            require(control['requests']==active['requests'] and
                    [(v['at'],v['op'],v['rank']) for v in control['views']]==
                    [(v['at'],v['op'],v['rank']) for v in active['views']], 'unmatched control')
            for a,b in zip(active['views'][:-1],control['views'][:-1],strict=True):
                close(expand(a['prob']),expand(b['prob']))
        close(expand(active['views'][-1]['prob']),expand(cases[i+3]['views'][-1]['prob']))
    return err


@torch.inference_mode()
def panel(model):
    cases=[]
    for world,docs in enumerate(WORLDS):
        with ReplicatedMount(docs) as bank:
            for tag in TAGS:
                for arm in ARMS:
                    policy=FixedJoin() if arm=='fixed' else Neural(model,arm)
                    with Live() as live:
                        e=Episode(policy,live,bank); m=Meter(policy.device,trace=True)
                        prefill(e,prompt(tag),m); p,byte=e.predict(m); e.observe(byte,m)
                        require(live.tail[-1]==byte,'not appended')
                        cases.append(dict(world=world,archive_version=bank.version,tag=tag,arm=arm,byte=byte,
                                          reads=m.reads,passes=m.passes,views=m.views,requests=m.requests))
    return dict(cases=cases,max_oracle_error=verify(cases))


def parity(left,right):
    verify(left['cases']); verify(right['cases']); errs=[]
    for a,b in zip(left['cases'],right['cases'],strict=True):
        require({k:v for k,v in a.items() if k!='views'}=={k:v for k,v in b.items() if k!='views'},'device action')
        for x,y in zip(a['views'],b['views'],strict=True):
            for k,v in x.items():
                if k=='prob' or (k=='write' and isinstance(v,dict)):
                    errs.append(close(expand(v),expand(y[k])))
                else:
                    require(v==y[k],'device state')
    return max(errs)
