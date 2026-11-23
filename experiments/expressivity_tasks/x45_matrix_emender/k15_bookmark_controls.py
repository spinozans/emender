"""Read-only E controls with true per-boundary read-count budgets."""
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from .k15_bookmark_joint import (B, EXE, BookmarkK15, BookmarkSession, InterventionPort,
                                 Stream, _build_k10, _require_lease, convert_model_to_composition,
                                 digest, emit, flagship_identity, incumbent, rows)
from ..x42_pile_scale.lattice import evaluate


def index_logits(base, raw, grants):
    z=base.float().log_softmax(-1); s=BookmarkSession(EXE); calls=0; support=[]
    try:
        for j in range(16,129,16):
            s.rpc('append '+raw[j-16:j].hex())
            if j==128:break
            counts=None
            for depth in (3,6,12)[:grants[j//16-1]]:
                reply=s.rpc('branch '+raw[j-depth:j].hex())['counts'];calls+=1
                if sum(reply): counts=reply
            support.append(sum(counts) if counts else 0)
            if counts:
                c=z.new_tensor(counts);z[0,j-1]=((z[0,j-1].exp()+c)/(1+c.sum())).log()
    finally:s.close()
    assert calls==sum(grants)
    return z,calls,support


@torch.no_grad()
def main(out):
    assert _require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'
    d=out.with_name('k15_bookmarks_D'); state=d/'joint.pt'
    source=[json.loads(line) for line in (d/'eval.jsonl').open()]; assert len(source)==64
    records=[json.loads(l) for l in (d/'console').read_text().splitlines() if l.startswith('{')];assert records[-1]=={'complete':True}
    assert digest(state)==next(r['sha'] for r in records if r.get('snapshot')==4096)
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    identity=flagship_identity(); stream=Stream(identity['source'],identity['manifest'])
    anchor=B.parents[1]/'xsa_anchor_replays_v1/k15/final.pt'; assert digest(anchor)=='accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0'
    base,*_=_build_k10(dict(incumbent._tuned_winner()),3401);base=convert_model_to_composition(base).cuda().eval()
    base.load_state_dict(torch.load(anchor,weights_only=True,map_location='cpu'));score=evaluate(base,stream,'matrix_ops','calibration');emit(anchor=score);assert score['score']==1.3811104467936925
    net=BookmarkK15(base,EXE);net.port=InterventionPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim);net.cuda().eval()
    snapshot=torch.load(state,weights_only=True,map_location='cpu'); assert snapshot['step']==4096
    assert digest(B)=='0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5'
    before=torch.load(B,weights_only=True,map_location='cpu'); changed={}
    for name,_ in net.named_parameters():
        changed[name]=float((before[name]-snapshot['model'][name]).norm())
    unchanged=[n for n,v in changed.items() if v==0];emit(unchanged=unchanged, tensors=len(changed),optimizer_states=len(snapshot['optimizer']['state']))
    (out/'changed.json').write_text(json.dumps(changed));del before
    for label in ('original','updated'):
        if label=='updated':net.load_state_dict(snapshot['model'])
        with (out/(label+'.jsonl')).open('x') as f,torch.autocast('cuda',dtype=torch.bfloat16):
            for i,((doc,_,ids,y),r) in enumerate(zip(rows(stream,'final_test'),source)):
                raw=bytes(ids[0].tolist());assert doc==r['doc'] and raw.hex()==r['raw']
                base_z=net(ids,doors=False)[0];grants=[len(e['calls']) for e in r['modes']['none']['trace']]; item=dict(doc=doc)
                if label=='updated' and i==0:
                    z,tr,*_=net(ids); assert json.loads(json.dumps(tr))==r['modes']['none']['trace'] and F.cross_entropy(z[0].float(),y,reduction='none').tolist()==r['modes']['none']['loss']; emit(exact_chosen_reload=True)
                for mode,counts in [('off',[]),('fixed',[1]*7),('yoked',grants)]:
                    torch.cuda.synchronize();t=time.perf_counter()
                    z,c,s=index_logits(base_z,raw,counts) if counts else (base_z,0,[])
                    loss=F.cross_entropy(z[0].float(),y,reduction='none').tolist()
                    item[mode]=dict(loss=loss,calls=c,support=s,index_and_score_ms=(time.perf_counter()-t)*1000)
                f.write(json.dumps(item)+'\n')
                if i%16==0:emit(arm=label,step=i+1)
    emit(complete=True)

if __name__=='__main__':main(Path(sys.argv[1]))
