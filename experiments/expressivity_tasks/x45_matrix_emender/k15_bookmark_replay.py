"""Masked online prefix replay of the retained B checkpoint; no training."""
import json
from pathlib import Path
import time
import torch
from .k15_bookmark_probe import (BookmarkK15, Stream, _build_k10, _require_lease,
                                 convert_model_to_composition, digest, flagship_identity, incumbent)


@torch.no_grad()
def main():
    assert _require_lease() in ('0','1')
    state = Path('/mnt/nvme2n1/erikg/k15_bookmarks_B/integrated.pt')
    assert digest(state) == '0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5'
    exe = Path.home()/'sxgc/xsa/target/debug/xsa'
    assert digest(exe) == '87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    base,*_ = _build_k10(dict(incumbent._tuned_winner()),3401)
    net = BookmarkK15(convert_model_to_composition(base),exe).cuda().eval()
    net.load_state_dict(torch.load(state,map_location='cpu',weights_only=True))
    identity = flagship_identity(); stream = Stream(identity['source'],identity['manifest'])
    # Exactly the B parity window, not a fresh selection after observing errors.
    x = next(x for doc,start,x,y in stream.windows('train',128) if doc == 100000007119 and start == 0)
    ids = torch.tensor(x,device='cuda')[None]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for doors in (False,True):
            full,actions,*_ = net(ids,doors=doors,audit=True)
            torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(8): net(ids,doors=doors)
            torch.cuda.synchronize();print(json.dumps(dict(doors=doors, tokens_s=1024/(time.perf_counter()-t))),flush=True)
            for end in (17,33,49,65,81,97,113):
                errors=[]
                for pad in (None,128):
                    z,chosen,*_ = net(ids[:,:end],doors=doors,audit=True,pad_to=pad)
                    assert chosen == actions[:len(chosen)]
                    errors.append(float((full[:,:end] - z).abs().max()))
                print(json.dumps(dict(doors=doors,end=end,prefix_max=errors)),flush=True)
                assert errors[1]==0

if __name__ == '__main__': main()
