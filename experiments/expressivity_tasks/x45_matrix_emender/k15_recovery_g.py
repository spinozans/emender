"""Matched D control; preserves per-site categorical sampling RNG consumption."""
from .k15_bookmark_joint import (os, sys, Path, torch, F, B, EXE, InterventionPort,
    Stream, BookmarkK15, _build_k10, incumbent, convert_model_to_composition,
    flagship_identity, _require_lease, digest, emit, rows, islice, json, hashlib, load_d_model)
from .k15_recovery_panel import panel


def draws(device):
    distributions = [torch.distributions.Categorical(logits=torch.zeros(n, device=device)) for n in (2,4)]
    for _ in range(7):
        for dist in distributions:
            dist.sample()


@torch.no_grad()
def compare(net, stream, out):
    net.eval()
    anchor=B.parents[1]/'xsa_anchor_replays_v1/k15/final.pt'
    assert digest(anchor)=='accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0'
    with (out/'panel.jsonl').open('x') as f, torch.autocast('cuda',dtype=torch.bfloat16):
        for arm in ('G','original','B','D'):
            if arm=='original': net.backbone.load_state_dict(torch.load(anchor,weights_only=True,map_location='cpu'))
            if arm=='B': net.load_state_dict(torch.load(B,weights_only=True,map_location='cpu'))
            if arm=='D': net.load_state_dict(load_d_model(B.parents[1]/'k15_bookmarks_D/joint.pt'))
            for doc,text in panel(stream,'calibration'):
                v=torch.tensor(list(text[:129]),device='cuda'); scores={}
                for doors in ((False,True) if arm in ('B','D') else (False,)):
                    logits=net(v[None,:-1],doors=doors)[0]; scores['native' if doors else 'off']=float(F.cross_entropy(logits[0].float(),v[1:]))
                f.write(json.dumps(dict(arm=arm,doc=doc,scores=scores))+'\n')
def main(out):
    assert _require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    assert digest(B)=='0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5'
    identity=flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    stream=Stream(identity['source'],identity['manifest']); emit(identity=identity)
    base,*_=_build_k10(dict(incumbent._tuned_winner()),3401)
    net=BookmarkK15(convert_model_to_composition(base),EXE)
    net.port=InterventionPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim)
    net.cuda().load_state_dict(torch.load(B,weights_only=True,map_location='cpu')); path=out/'joint.pt'
    torch.manual_seed(3407); net.train(); opt=torch.optim.AdamW([dict(params=net.backbone.parameters(),lr=1e-5),dict(params=net.port.parameters(),lr=1e-4)],fused=True)
    ref=B.parents[1]/"k15_bookmarks_D/train.jsonl"; assert digest(ref)=="d292027f6c2316510ab5b08b316c336412bbfa59efc0e81fc81b8b0443a0d5eb"
    receipt=[json.loads(line) for line in ref.read_text().splitlines()]
    with (out/'train.jsonl').open('x') as f:
        for step,(doc,start,ids,y) in enumerate(islice(rows(stream,'train'),4096),1):
            row=receipt[step-1]; assert (step,doc,start,hashlib.sha256(bytes(ids[0].tolist()+[int(y[-1])])).hexdigest())==(row["step"],row["doc"],row["start"],row["sha"])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,tr,probs,_=net(ids,doors=False);draws(ids.device); losses=F.cross_entropy(logits[0],y,reduction='none'); loss=losses.mean()
                objective=loss+sum((p+q)*(losses[e['stop']:].mean().detach()-2) for p,q,e in zip(probs[::2],probs[1::2],tr))
            opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            f.write(json.dumps(dict(step=step,doc=doc,start=start,sha=hashlib.sha256(bytes(ids[0].tolist()+[int(y[-1])])).hexdigest(),loss=float(loss.detach()),actions=[(e['slot'],e['op']) for e in tr]))+'\n')
            if step<4 or step%64==0: emit(step=step, loss=float(loss.detach()));f.flush()
            if step%512==0:
                tmp=path.with_suffix('.partial');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state()),tmp);os.replace(tmp,path);emit(snapshot=step,sha=digest(path))
    assert step==4096
    net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model']);compare(net,stream,out);emit(complete=True)

if __name__=='__main__': main(Path(sys.argv[1]))
