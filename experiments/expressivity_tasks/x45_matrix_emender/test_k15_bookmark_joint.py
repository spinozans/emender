import torch
from pathlib import Path
from .k15_bookmarks import BookmarkPort, BookmarkSession
from .k15_bookmark_intervention import InterventionPort
from .k15_composition import ComposedReadMixer
from .schedule import RoutingPolicy


def test_port_fork_and_saved_address_intervention():
    raw=b'abcZabcX'*10
    h=torch.randn(1,80,32); ids=torch.tensor([list(raw)])
    mixer=ComposedReadMixer(32,2,16,8,8,8,RoutingPolicy()); port=InterventionPort(32,16); base=BookmarkPort(32,16);base.load_state_dict(port.state_dict())
    def run(p):
        s=BookmarkSession(Path.home()/'sxgc/xsa/target/debug/xsa')
        try: return p(mixer,h,ids,s,[(0,0),(1,0),(0,2),(0,3)],True)[:2]
        finally: s.close()
    ref,_=run(base); true,events=run(port);assert torch.equal(true,ref)
    port.intervention='address'; _,other=run(port); assert events[0]['calls']==other[0]['calls'] and events[2]['calls'][0]!=other[2]['calls'][0]
    port.intervention='clear';assert run(port)[0].count_nonzero()==0
    port.intervention='payload';assert not torch.equal(true,run(port)[0])
