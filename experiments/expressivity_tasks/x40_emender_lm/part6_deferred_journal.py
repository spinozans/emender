"""Pack all four recorded head journals at the Cell boundary in one sizing transfer.

Stop traversal at replay nodes: upstream blocks and their storage are untouched.
No decision is reconstructed, and the writer's overflow checks remain enabled.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from . import model


def pack_cell(value, records, streams=()):
    pending=[value.grad_fn];seen=set();nodes=[]
    while pending:
        node=pending.pop()
        if node is None or node in seen:continue
        seen.add(node)
        if type(node).__name__=='ReplayBackward':
            nodes.append(node)
        else:pending.extend(n for n,_ in node.next_functions)
    assert len(nodes)==4,f'expected exactly four Cell-local replays, found {len(nodes)}'
    maxima=torch.stack([node.log[3].max() for node in nodes]).cpu().tolist()
    for node,used in zip(nodes,maxima):
        old=node.capacity;pitch=max(1,int(used));assert pitch<=old
        addresses,before,widths,*tail=node.log
        # Prefix copies run on the origin stream, while the writer buffers
        # belong to head streams. Protect those reads through copy completion.
        for tensor in (addresses,before,widths):tensor.record_stream(torch.cuda.current_stream(tensor.device))
        node.log=(addresses[:,:pitch].clone(),before[:,:pitch].clone(),widths[:,:pitch].clone(),*tail)
        for tensor in node.log[:3]:
            for stream in streams:tensor.record_stream(stream)
        node.capacity=pitch
        records.append(dict(stores=len(tail[0]),reserved_events_per_lane=old,
                            measured_max_events_per_lane=used,retained_events_per_lane=pitch,
                            before_bytes=20*len(tail[0])*old,after_bytes=20*len(tail[0])*pitch))


@contextmanager
def packed_cells(records):
    original=model.Cell.forward
    def forward(self,x,ids):
        value=original(self,x,ids)
        pack_cell(value,records,getattr(self,'_part6_streams',()))
        return value
    with patch.object(model.Cell,'forward',forward):yield
