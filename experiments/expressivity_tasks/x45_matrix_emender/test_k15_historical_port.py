from pathlib import Path
import torch
from .k15_historical_port import HistoricalPort
from .k15_bookmarks import BookmarkSession
from .k15_composition import ComposedReadMixer
from .schedule import RoutingPolicy


def test_miss_no_echo_paid_extension_and_retirement():
    raw=b'abcdefghijklmnop'+b'abcAZghiABxyzabc'+b'x'*48
    s=BookmarkSession(Path.home()/'sxgc/xsa/target/debug/xsa')
    try:
        mixer=ComposedReadMixer(32,2,16,8,8,8,RoutingPolicy()); port=HistoricalPort(32,16)
        out,events,_,writes=port(mixer,torch.randn(1,80,32),torch.tensor([list(raw)]),s,[(0,0),(0,0),(0,2),(0,3)],True)
        assert events[0]['miss'] and len(events[0]['calls'])==1
        assert events[1]['saved_pos']==0 and events[1]['calls'][-1][1]['byte']==ord('d')
        assert [c for c,_ in events[2]['calls']]==['byte 3','locate 61626364 0 47 48','byte 4']
        assert out[:,:32].count_nonzero()==out[:,64:].count_nonzero()==0
        assert len(writes)==2
    finally: s.close()
