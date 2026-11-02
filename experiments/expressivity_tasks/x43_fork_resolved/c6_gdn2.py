"""Correct GDN-2 continuation of C6; first four arms are immutable receipts.

Head and train loop copied from c6.py with only the mixer import, resume metadata,
and arm selection changed. Uses NVIDIA GatedDeltaNet2, not FLA GatedDeltaNet.
"""
from .c6 import *

class Head(nn.Module):
    def __init__(self, arm, target):
        super().__init__()
        self.arm = arm
        # Identical extra parameter count for every architecture and control.
        self.condition = nn.Linear(260, 256, bias=False)
        self.ood = nn.Linear(256, 1, bias=False)
        if arm == 'emender':
            self.base = Candidate(Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.),
                                  vocab=256, d=256, layers=2, target=target)
        elif arm == 'transformer':
            self.base = Control(target)
        else:
            import fla
            from ndm.models.external_gdn2 import _load_gdn2_class
            GatedDeltaNet = _load_gdn2_class()
            assert fla.__version__ == '0.5.1'
            self.embed = nn.Embedding(256, 256)
            self.layers = nn.ModuleList([GatedDeltaNet(hidden_size=256, head_dim=64,
                num_heads=4, expand_v=1, layer_idx=i) for i in range(2)])
            self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(2)])
            self.norm = nn.LayerNorm(256)
            self.output = nn.Linear(256, 256, bias=False)
            remaining = target + 260*256 + 256 - count(self)
            hidden = max(1, round((remaining-512)/1026))
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(256,hidden),nn.GELU(),
                                                   nn.Linear(hidden,256)) for _ in range(2)])

    def forward(self, ids, features, aware):
        embed = self.base.embed if self.arm != 'gdn2' else self.embed
        x = embed(ids)
        if aware:
            x = x + self.condition(features)
        if self.arm == 'emender':
            from ..x40_emender_lm.part6_checkpoint import checkpoint_block
            for block in self.base.blocks:
                if self.training and torch.is_grad_enabled():
                    x, _ = checkpoint_block(block, x, ids)
                else:
                    x = block(x, ids)
            x = self.base.norm(x)
            byte = F.linear(x.float(), self.base.head.weight.float())
        elif self.arm == 'transformer':
            x = x + self.base.position(torch.arange(ids.shape[1], device=ids.device))
            mask = torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
            for block in self.base.blocks:
                x = block(x, src_mask=mask, is_causal=True)
            x = self.base.norm(x)
            byte = self.base.head(x).float()
        else:
            for layer, norm, ff in zip(self.layers,self.norms,self.ff):
                x = x + layer(norm(x), use_cache=False)[0]
                x = x + ff(x)
            x = self.norm(x)
            byte = self.output(x).float()
        if not aware:
            return byte.log_softmax(-1)
        return branch_logp(byte, self.ood(x).float(), features[..., :256]>0)


def train():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    protocol_hash = digest(ROOT/'PREREGISTRATION_C6.md')
    assert protocol_hash in (ROOT/'PROGRAM.md').read_text()
    torch.set_num_threads(1)
    z = np.load(OUT/'index.npz')
    data = torch.tensor(list((ROOT/'input.txt').read_bytes()), device='cuda', dtype=torch.long)
    ids = torch.tensor(z['ids'], device='cuda', dtype=torch.long)
    features = torch.tensor(z['features'], device='cuda', dtype=torch.float32)
    fork = torch.tensor(z['fork'], device='cuda', dtype=torch.bool)
    split = int(.9*len(data))
    meta = json.loads((OUT/'training.json').read_text())
    meta['gdn2_code_sha256'] = digest(Path(__file__))
    meta['gdn2_physical_gpu'] = os.environ['GPU_LEASE_HELD']
    meta['gdn2_protocol_sha256'] = digest(ROOT/'PREREGISTRATION_C6_GDN2.md')
    assert meta['gdn2_protocol_sha256'] in (ROOT/'PROGRAM.md').read_text()
    write('training.json',meta)
    target = meta['models'][0]['trainable_parameters'] - 260*256 - 256
    for arm in ('gdn2',):
        for aware in (True,False):
            name = arm+('_aware' if aware else '_control')
            try:
                torch.manual_seed(4301)
                init = time.monotonic()
                model = Head(arm, target).cuda().train()
                if arm == 'emender' and aware:
                    target = count(model.base)
                parameters = count(model)
                initialization_seconds = time.monotonic()-init
                opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
                rng = torch.Generator(device='cuda').manual_seed(4301)
                steps=0; curve=[]; fork_tokens=0
                torch.cuda.synchronize(); start=time.monotonic()
                while time.monotonic()-start < 180:
                    starts = torch.randint(0,split-128,(16,),device='cuda',generator=rng)
                    at = starts[:,None]+torch.arange(128,device='cuda')
                    y = data[at+1]
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logp=model(data[at],features[ids[at+1]],aware)
                        nll=-logp.gather(-1,y[...,None])[...,0]
                        mask=fork[at+1] if aware else torch.ones_like(y,dtype=torch.bool)
                        loss=nll[mask].mean()
                    assert torch.isfinite(loss), 'nonfinite loss'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step(); steps+=1
                    fork_tokens+=int(fork[at+1].sum())
                    if steps==1 or steps%100==0:
                        curve.append(dict(step=steps,seconds=time.monotonic()-start,nll=float(loss.detach())))
                        print(name,curve[-1],flush=True)
                torch.cuda.synchronize()
                training_seconds=time.monotonic()-start
                row=dict(name=name,arm=arm,fork_aware=aware,trainable_parameters=parameters,
                    total_parameters=sum(p.numel() for p in model.parameters()), steps=steps,
                    tokens=steps*16*128,fork_tokens=fork_tokens,training_seconds=training_seconds,
                    initialization_seconds=initialization_seconds,curve=curve)
                # Model is frozen before any held-out inference. No checkpoint files needed.
                model.eval(); eval_start=time.monotonic(); pp=[]; cc=[]
                with torch.no_grad():
                    for lo in range(split,len(data)-1,16*128):
                        # Each row is an independent 128-byte reset, including across batches.
                        hi=min(lo+16*128,len(data)-1)
                        width=128 if hi-lo>=128 else hi-lo
                        full=(hi-lo)//128*128
                        ranges=[]
                        if full: ranges.append(torch.arange(lo,lo+full,device='cuda').reshape(-1,128))
                        if lo+full<hi: ranges.append(torch.arange(lo+full,hi,device='cuda')[None])
                        for at in ranges:
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                lp=model(data[at],features[ids[at+1]],aware)
                            pp.append(lp.gather(-1,data[at+1,...,None])[...,0].exp().flatten().cpu().numpy())
                            cc.append(lp.max(-1).values.exp().flatten().cpu().numpy())
                p=np.concatenate(pp).astype(np.float64); confidence=np.concatenate(cc)
                assert len(p)==len(data)-split-1 and np.isfinite(p).all() and (p>0).all()
                np.savez_compressed(OUT/(name+'_prediction.npz'),p=p,confidence=confidence)
                row.update(evaluation_seconds=time.monotonic()-eval_start,
                           prediction_sha256=digest(OUT/(name+'_prediction.npz')))
                meta['models'].append(row)
                write('training.json',meta)
                print(name,'complete',row['steps'],flush=True)
                del opt,model; torch.cuda.empty_cache()
            except Exception:
                error=traceback.format_exc(); print(error,flush=True)
                meta['failures'].append(dict(name=name,error=error)); write('training.json',meta)
                if arm!='gdn2': raise
                break


if __name__ == '__main__':
    train()
