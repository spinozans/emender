"""P0: singleton doors-off K15, actual CPU eager scan, no trainer."""
import json
import os
import time
from pathlib import Path

import torch
from .k15_phrase import Controller, Filter
from .k15_bookmarks import BookmarkSession
from .k15_historical_training import ANCHOR, ANCHOR_SHA, j
from .k10_separated_keys import convert_model_to_separated_keys
from .matched_arms import matrix_config, matrix_model_for, RANK, RoutingPolicy


@torch.inference_mode()
def main():
    assert __debug__ and os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.get_num_threads()==4
    root = Path(__file__).parent/'receipts/k15_phrase'
    d = json.loads((root/'document.json').read_text())['document']
    ident = j.flagship_identity()
    assert ident['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    assert j.digest(ANCHOR)==ANCHOR_SHA and j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    from hashlib import sha256
    with open(ident['source'],'rb') as f:
        f.seek(d['byte_offset']); raw=f.read(d['byte_length'])
    assert sha256(raw).hexdigest()==d['sha256']
    config = j.incumbent._tuned_winner()
    policy = RoutingPolicy(ops=True, **{k:config[k] for k in ('tau_match','tau_merge','tau_pop','read_scale')})
    base = matrix_model_for(matrix_config(rank=RANK), policy, RANK, compile_scan=False)
    base = j.convert_model_to_composition(convert_model_to_separated_keys(base))
    base.load_state_dict(torch.load(ANCHOR,map_location='cpu',weights_only=True),strict=True)
    base.requires_grad_(False).eval()
    s = BookmarkSession(j.EXE)
    try:
        m = Filter(s,Controller(),raw[:128]); tic=time.monotonic()
        for i in range(128,131):
            ids=torch.tensor(list(m.tape.raw[-128:]))[None]; assert ids.shape==(1,128)
            logits=base.head(base.features(ids))[:,-1][0]; p=m.predict(logits)
            assert p.min()>0 and abs(float(p.sum())-1)<1e-5
            m.observe(raw[i]); print(json.dumps(dict(position=i,paths=len(m.paths),minimum=float(p.min()),mass=float(p.sum()),seconds=time.monotonic()-tic)),flush=True)
    finally: s.close()


if __name__=='__main__': main()
