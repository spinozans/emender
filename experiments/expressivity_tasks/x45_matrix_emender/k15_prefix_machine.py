"""Prefix-synchronous Bayes cursor machine. Rows and K15 feedback are shared."""
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint
from .k15_phrase import Tape, phrase_source
from .k15_phrase_recurrent import Row, PhraseK15


class PrefixK15(PhraseK15):
    def neural(self, raw, feedback):
        ids = torch.tensor(list(raw[-128:]), device=feedback.device)[None]
        assert ids.shape == (1, 128)
        base = self.backbone
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=feedback.is_cuda):
            h = base.embed(ids)
            h = torch.cat((h[:, :-1], h[:, -1:] + feedback), 1)
            for block in base.blocks:
                h = checkpoint(block, h, use_reentrant=False) if h.requires_grad and feedback.is_cuda else block(h)
            h = base.norm(h)
            return h[0,-1].float(), base.head(h)[0,-1].float().softmax(0)


@dataclass
class Emission:
    op: int  # literal, start, continue
    cursor: tuple  # distance and age after emission, zero means idle
    weight: torch.Tensor
    byte: int | None
    address: int


class PrefixMachine:
    def __init__(self, model, session, prefix, mode='full'):
        assert len(prefix)==128 and mode in ('full','skip','wrong','scramble','freeze')
        self.model, self.mode = model, mode
        self.tape = Tape(session); self.tape.append(prefix)
        self.feedback = model.write.bias.new_zeros(model.write.out_features)
        self.rows = [Row(i,bytes([y]),self.feedback) for i,y in enumerate(prefix)]
        self.posterior = {(0,0): self.feedback.new_ones(())}
        self.pending = None

    def predict(self):
        assert self.pending is None
        m = self.model; n = len(self.rows)
        h, base = m.neural(self.tape.raw,self.feedback)
        literal = base * (1-1e-6) + 1e-6/256
        payload = m.write(h).tanh()
        start, keep = m.phase(h).clamp(-12,12).sigmoid().unbind()
        phrase = phrase_source(self.rows)
        addresses = sorted(set(self.tape.sources()) | ({phrase.start} if phrase else set()))
        assert 1<=len(addresses)<=4
        def read(a):
            return self.rows[(a-1)%n if self.mode=='scramble' else a].payload
        features = torch.stack([torch.stack((h.new_tensor(float(phrase is not None and a==phrase.start)), h.new_tensor((a-n)/n),read(a).mean())) for a in addresses])
        scores = m.operand(features).flatten(); weights = scores.softmax(0)
        if self.mode=='skip': start = start*0
        if self.mode=='wrong': weights = torch.nn.functional.one_hot(scores.argmin(),len(addresses)).float()
        edges=[]; idle=h.new_zeros(())
        for (dist,age),p in self.posterior.items():
            if dist and age<3:
                edges.append(Emission(2,(dist,age+1),p*keep,self.tape.byte(n-dist),n-dist)); idle=idle+p*(1-keep)
            else: idle=idle+p
        edges += [Emission(1,(n-a,1),idle*start*w,self.tape.byte(a),a) for a,w in zip(addresses,weights)]
        edges.append(Emission(0,(0,0),idle*(1-start),None,n-1))
        probs = literal*edges[-1].weight
        probs = probs.index_add(0,torch.tensor([e.byte for e in edges[:-1]],device=h.device),torch.stack([e.weight for e in edges[:-1]]))
        self.pending = probs,literal,edges,payload,read
        return probs

    def observe(self, y):
        p,lit,edges,payload,read = self.pending
        post={}; feedback=torch.zeros_like(self.feedback); ops=[0.,0.,0.]
        for e in edges:
            if e.byte is not None and e.byte != y: continue
            w=e.weight*(lit[y] if e.byte is None else 1)/p[y]
            post[e.cursor]=post.get(e.cursor,0)+w
            feedback=feedback+w*(self.model.transition.weight[e.op]+read(e.address)); ops[e.op]=ops[e.op]+w
        assert len(post)<=13
        self.posterior=post; self.feedback=.125*(feedback+self.feedback*.25).tanh()
        self.rows.append(Row(len(self.rows),bytes([y]),payload if self.mode!='freeze' else payload*0)); self.tape.append(bytes([y])); self.pending=None
        return [float(torch.as_tensor(v).detach()) for v in ops]
