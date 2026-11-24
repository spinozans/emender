"""Execute the same causal I/O contract for a neural policy and an independent fixed join."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import torch

from . import workbench_model as w
from .workbench import Handle, compact
from .workbench_native import Archive
from .w2_model import Op


@dataclass(frozen=True)
class Reply:
    locus: Handle
    h: tuple[int, ...]
    interval: tuple[int, int]


class ReplicatedMount:
    """Two same-version read-only sessions, each a single FIFO. Used by one episode at a time."""
    def __init__(self, docs, *, allowed=True, excluded_docs=()):
        self.stack = ExitStack()
        try:
            docs = tuple(docs)
            self.workers = [self.stack.enter_context(Archive(docs, allowed=allowed, excluded_docs=excluded_docs))
                            for _ in range(2)]
            self.name, self.version = self.workers[0].name, self.workers[0].version
            if self.workers[1].version != self.version:
                raise ValueError('replicas differ')
            self.pool = self.stack.enter_context(ThreadPoolExecutor(max_workers=2, thread_name_prefix='w2-branch'))
        except BaseException:
            self.stack.close(); raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def locus(self, key):
        return Handle(self.name, self.version, key)

    def _one(self, slot, locus):
        if (locus.source,locus.version) != (self.name,self.version):
            raise ValueError('foreign branch locus')
        start=perf_counter_ns(); h=self.workers[slot].branch(locus.key)
        return Reply(locus,h,(start,perf_counter_ns()))

    def parent(self, key):
        return self._one(0,self.locus(key))

    def children(self, handles):
        if len(handles)!=2 or handles[0]==handles[1]:
            raise ValueError('two distinct children required')
        futures=[self.pool.submit(self._one,i,h) for i,h in enumerate(handles)]
        return tuple(f.result() for f in futures)


def bind(handles, answers):
    table={r.locus:r for r in answers}
    if len(answers)!=2 or len(table)!=2 or set(handles)!=table.keys():
        raise ValueError('incomplete, duplicated or stale child binding')
    return tuple(table[h] for h in handles)


@dataclass
class Cell:
    session: str
    written_at: int
    value: object


class Neural:
    def __init__(self, model, mode):
        if mode not in ('full','erase','blind'):
            raise ValueError('unknown intervention')
        self.model=model
        self.device=next(model.parameters()).device
        self.empty=torch.zeros(256,device=self.device)
        self.gain=self.empty.new_tensor(0. if mode=='erase' else 1.)
        self.mix=(torch.full((2,2),.5,device=self.device) if mode=='blind'
                  else torch.eye(2,device=self.device))

    def decide(self, tail, record, parent, children, meter):
        def pack():
            keys={w.ARCH:parent.locus.key} if parent else {}
            for ans,slot,book in zip(children,(w.CHILD,w.META),(w.BOOK0,w.BOOK1)):
                keys[slot]=keys[book]=ans.locus.key
            x=w.pack(tail,None,parent.h if parent else None,children[0].h if children else None,
                     (bool(parent),bool(children),False,False,False,False),keys)
            x[w.BOOK0:w.BOOK1+1,w.BYTE_ID]=256
            if children:
                x[w.META,:257]=torch.tensor(children[1].h,dtype=torch.float32)
                x[w.BOOK0,w.BYTE_ID]=children[0].locus.key[-1]
                x[w.BOOK1,w.BYTE_ID]=children[1].locus.key[-1]
            return x
        x=meter.timed('pack',pack)
        x=meter.timed('transfer',lambda:x.to(self.device),fence=True)
        packet,p,value=meter.timed('forward',lambda:self.model(x,self.empty if record is None else record.value,
                                                    self.gain,self.mix),fence=True)
        meter.passes+=1
        return meter.timed('decision_return',packet.tolist,fence=True),p,value


class FixedJoin:
    """No torch forward, no doc inspection: recognize the markers and equijoin the native rows."""
    device=torch.device('cpu')

    def decide(self, tail, record, parent, children, meter):
        op=Op.EMIT; rank=[-1,-1]; base=np.zeros(256,dtype=np.float64); value=None
        if parent is not None:
            base=np.array(parent.h[:256],dtype=np.float64)
            active=sorted(np.flatnonzero(base),key=lambda b:(-base[b],int(b)))
            rank[:len(active)]=[int(b) for b in active]
            if len(active)==2 and not children:
                op=Op.CHILDREN
            elif children and record is not None:
                matches=np.zeros_like(base)
                for r in children:
                    matches[r.locus.key[-1]]=r.h[record.value]
                if matches.any():
                    base=matches
        elif tail.endswith(b'key='):
            op=Op.PARENT
        elif tail[:-1].endswith(b'unit='):
            op=Op.STORE_EMIT; value=tail[-1]
        p=np.full(256,1/256)
        if base.any():
            p=(p+4*base/base.sum())/5
        return [int(op),4,*rank,int(p.argmax())],p,value


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class Episode:
    def __init__(self, policy, live, bank):
        self.policy,self.live,self.bank=policy,live,bank
        self.record=None
        self.pending=False

    def predict(self,meter):
        if self.pending:
            raise ValueError('observe before next prediction')
        if self.record and (self.record.session!=self.live.session_id or self.record.written_at>self.live.visible):
            raise ValueError('stale cue record')
        parent=None; children=()
        for _ in range(3):
            pkt,p,value=self.policy.decide(self.live.tail,self.record,parent,children,meter)
            opcode,n,a,b,greedy=pkt; op=Op(opcode)
            if meter.trace:
                meter.views.append(dict(at=self.live.visible,tail=self.live.tail.hex(),op=op.name,rank=[a,b],
                    prob=compact(array(p)),written_at=self.record.written_at if self.record else None))
            if op==Op.PARENT:
                if parent or not 1<=n<=len(self.live.tail):
                    raise ValueError('invalid parent request')
                key=self.live.tail[-n:]
                parent=meter.timed('parent_rpc',lambda:self.bank.parent(key)); meter.reads+=1
                if parent.locus!=self.bank.locus(key) or sum(v>0 for v in parent.h[:256])>2:
                    raise ValueError('wrong parent or fanout exceeds exact two-edge capacity')
                answers=(parent,)
            elif op==Op.CHILDREN:
                if not parent or children or a==b or not all(0<=i<256 and parent.h[i] for i in (a,b)):
                    raise ValueError('invalid child operation')
                handles=tuple(self.bank.locus(parent.locus.key+bytes([i])) for i in (a,b))
                answers=children=bind(handles,meter.timed('child_pair_rpc',lambda:self.bank.children(handles)))
                meter.reads+=2
                if meter.profile:
                    lo=max(r.interval[0] for r in children); hi=min(r.interval[1] for r in children)
                    meter.stages.append(('child_rpc_overlap',max(0,hi-lo)))  # nested in pair wall
            else:
                if op==Op.STORE_EMIT:
                    if value is None:
                        raise ValueError('missing write')
                    self.record=Cell(self.live.session_id,self.live.visible,value)
                    if meter.trace:
                        meter.views[-1]['write']=(compact(array(value)) if torch.is_tensor(value) else int(value))
                self.pending=True
                return p,greedy
            if meter.trace:
                meter.requests.extend({'at':self.live.visible,'key':r.locus.key.hex(),'h':compact(r.h)} for r in answers)
        raise ValueError('exceeded two read rounds')

    def observe(self,byte,meter):
        if not self.pending:
            raise ValueError('predict before observe')
        meter.timed('append_rpc',lambda:self.live.append(bytes([byte])))
        self.pending=False
