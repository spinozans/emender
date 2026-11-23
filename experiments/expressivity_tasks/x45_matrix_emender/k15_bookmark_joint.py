"""D: bounded K15 joint tuning. All artifacts remain local."""
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from .k15_bookmark_probe import (BookmarkK15, Stream, _build_k10, _require_lease,
                                 convert_model_to_composition, digest, flagship_identity, incumbent)
from .k15_bookmarks import BookmarkSession
from .k15_bookmark_intervention import InterventionPort

B = Path('/mnt/nvme2n1/erikg/k15_bookmarks_B/integrated.pt')
EXE = Path.home()/'sxgc/xsa/target/debug/xsa'


def emit(**kw):
    print(json.dumps(kw), flush=True)


def rows(stream, split):
    # Filter train documents before reading their payloads, not after scanning 1.45 GB.
    source = Stream(stream.source, stream.manifest)
    source.documents = lambda s: (d for d in stream.documents(s) if s != 'train' or d['doc'] >= 101450000000)
    for doc, start, x, y in source.windows(split, 128):
        if len(y) == 128 and (split == 'train' or start == 0):
            yield doc, start, torch.tensor(x, device='cuda')[None], torch.tensor(y, device='cuda')


def fixed(net, ids):
    logits = net(ids, doors=False)[0].float().log_softmax(-1)
    raw = bytes(ids[0].tolist()); session = BookmarkSession(EXE)
    try:
        for i in range(16,129,16):
            session.rpc('append '+raw[i-16:i].hex())
            if i < 128:
                c = logits.new_tensor(session.rpc('branch '+raw[i-3:i].hex())['counts'])
                logits[0,i-1] = ((logits[0,i-1].exp()+c)/(1+c.sum())).log()
    finally:
        session.close()
    return logits


@torch.no_grad()
def evaluate(net, stream, out):
    net.eval()
    with (out/'eval.jsonl').open('x') as f, torch.autocast('cuda',dtype=torch.bfloat16):
        for i,(doc,start,ids,y) in enumerate(islice(rows(stream,'final_test'),64)):
            result = dict(doc=doc, raw=bytes(ids[0].tolist()).hex(), modes={})
            choices = None
            for mode in ('none','clear','payload','address','off','fixed'):
                net.port.intervention=mode; torch.cuda.synchronize(); t=time.perf_counter()
                if mode=='fixed':
                    z=fixed(net,ids); trace=[]
                else:
                    z,trace,*_=net(ids,forced=choices,doors=mode!='off')
                loss=F.cross_entropy(z[0].float(),y,reduction='none').tolist()
                result['modes'][mode]=dict(loss=loss,trace=trace,ms=1000*(time.perf_counter()-t))
                if mode=='none':
                    choices=[(e['slot'],e['op']) for e in trace]
                    if i==0:
                        for end in (17,33,49,65,81,97,113):
                            p,tr,*_=net(ids[:,:end],pad_to=128)
                            assert torch.equal(p,z[:,:end]) and tr==trace[:len(tr)]
                        decode(net,bytes(ids[0,:64].tolist()))
            f.write(json.dumps(result)+'\n'); f.flush()
    assert i==63


@torch.no_grad()
def decode(net, prompt):
    seq=list(prompt); t=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(32):
            z,*_=net(torch.tensor(seq,device='cuda')[None],pad_to=128)
            seq.append(int(z[0,-1].argmax()))
    emit(prompt=prompt.decode(errors='replace'), generated=bytes(seq[len(prompt):]).decode(errors='replace'), hex=bytes(seq).hex(), decode_seconds=time.perf_counter()-t)


D_SHA256 = "6dc7db5428ac0248278203b85de3eeeed55c4da610eae2ba5b66ae1b83af34ba"

def load_d_model(path):
    if digest(path) != D_SHA256:
        raise ValueError("D checkpoint digest mismatch")
    state = torch.load(path, weights_only=True, map_location="cpu")
    if state["step"] != 4096:
        raise ValueError("D checkpoint step mismatch")
    return state["model"]


def main(out, inference=False):
    assert _require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    assert digest(B)=='0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5'
    identity=flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    stream=Stream(identity['source'],identity['manifest']); emit(identity=identity)
    base,*_=_build_k10(dict(incumbent._tuned_winner()),3401)
    net=BookmarkK15(convert_model_to_composition(base),EXE)
    net.port=InterventionPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim)
    net.cuda().load_state_dict(torch.load(B,weights_only=True,map_location='cpu')); path=out/'joint.pt'
    if inference:
        net.load_state_dict(load_d_model(path));net.eval(); decode(net,sys.argv[3].encode());return
    torch.manual_seed(3407); net.train(); opt=torch.optim.AdamW([dict(params=net.backbone.parameters(),lr=1e-5),dict(params=net.port.parameters(),lr=1e-4)],fused=True)
    with (out/'train.jsonl').open('x') as f:
        for step,(doc,start,ids,y) in enumerate(islice(rows(stream,'train'),4096),1):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,tr,probs,_=net(ids); losses=F.cross_entropy(logits[0],y,reduction='none'); loss=losses.mean()
                objective=loss+sum((p+q)*(losses[e['stop']:].mean().detach()-2) for p,q,e in zip(probs[::2],probs[1::2],tr))
            opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            f.write(json.dumps(dict(step=step,doc=doc,start=start,sha=hashlib.sha256(bytes(ids[0].tolist()+[int(y[-1])])).hexdigest(),loss=float(loss.detach()),actions=[(e['slot'],e['op']) for e in tr]))+'\n')
            if step<4 or step%64==0: emit(step=step, loss=float(loss.detach()));f.flush()
            if step%512==0:
                tmp=path.with_suffix('.partial');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state()),tmp);os.replace(tmp,path);emit(snapshot=step,sha=digest(path))
    assert step==4096
    net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model']);evaluate(net,stream,out);emit(complete=True)

if __name__=='__main__': main(Path(sys.argv[1]),sys.argv[2:3]==['decode'])
