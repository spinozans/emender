"""Recurrent residual seam: call once per host state update with resolved banks."""
from dataclasses import dataclass

import torch
from torch import nn
from .b1_mount import Span
from .bank_read import BoundPage
from .bank_read_triton import triton_read


@dataclass(frozen=True)
class WorkingSet:
    mount: object
    version: str
    slot_spans: tuple[Span, ...]  # Host-only inverse map; one entry per distinct K/V slot.
    tensors: tuple

    def checked(self):
        if self.mount.version != self.version:
            raise ValueError('working set outlived source generation')
        return self.tensors


def pack(rows, mount, name):
    slots={}; values=[]; ids=torch.full((len(rows),max(1,max(map(len,rows)))),-1,dtype=torch.int32)
    for i,row in enumerate(rows):
        for j,(addr,value) in enumerate(row):
            if (addr.source,addr.version)!=(name,mount.version):
                raise ValueError('stale or substituted source')
            if addr not in slots:
                slots[addr]=len(values); values.append(value)
            elif values[slots[addr]]!=value:
                raise ValueError('inconsistent value identity')
            ids[i,j]=slots[addr]
    v=torch.tensor(values,dtype=torch.float32).reshape(-1,8)
    return WorkingSet(mount,mount.version,tuple(slots),(v.flip([1])+0.25,v,BoundPage.from_cpu(ids,len(v)),torch.zeros_like(ids,dtype=torch.float32)))


def transfer(bank,device):
    k,v,p,b=bank.checked()
    return WorkingSet(bank.mount,bank.version,bank.slot_spans,(k.to(device),v.to(device),BoundPage(p.ids.to(device),p.records),b.to(device)))


class XSAResidual(nn.Module):
    def __init__(self):
        super().__init__(); self.query=nn.Linear(8,8,bias=False); self.output=nn.Linear(8,8,bias=False)
        self.init_xsa_attention()

    def init_xsa_attention(self):
        nn.init.zeros_(self.query.weight); nn.init.eye_(self.output.weight)

    def forward(self,state,banks,*,read=triton_read):
        q=self.query(state); result=state
        for scale,bank in banks:
            result=result+scale*self.output(read(q,*bank.checked()))
        return result
