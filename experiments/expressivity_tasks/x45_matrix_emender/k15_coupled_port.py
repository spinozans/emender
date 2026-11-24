"""Bounded historical suffix matches; offsets are local to the current document window."""
if not __debug__:
    raise RuntimeError("K15 I/J requires Python assertions; unset PYTHONOPTIMIZE and omit -O")

import torch
from torch import nn
import torch.nn.functional as F
from .k15_historical_port import historical_hits
from .key_separation_probe import causal_context


def candidates(prefix, session, audit=False):
    n = len(prefix)
    found = {}
    for depth in (2, 4, 8):
        pattern = prefix[-depth:]
        hits = historical_hits(session.rpc, pattern, n)
        if audit:
            assert hits == [i for i in range(n-depth) if prefix[i:i+depth] == pattern]
        for pos in hits[::max(1, len(hits)//3)] + hits[-1:]:
            after = pos + depth
            length = depth
            while length < min(24, after) and prefix[after-length-1] == prefix[n-length-1]:
                length += 1
            byte = session.rpc(f'byte {after}')['byte']
            assert after < n and byte == prefix[after]
            found[after] = (length, after, byte)
    return sorted(found.values(), reverse=True)


class CoupledPort(nn.Module):
    """Single reserved K/V row with hard retain or update and a continuation cursor."""
    mode = 'learned'

    def __init__(self, width, dim):
        super().__init__()
        self.byte = nn.Embedding(256, dim)
        self.query = nn.Linear(width, dim)
        self.fuse = nn.Linear(width+dim, dim)
        self.action = nn.Linear(width+dim, 1)
        self.update = nn.Linear(width+dim, dim)

    def forward(self, mixer, hidden, ids, session, forced=None, audit=False):
        raw = bytes(ids[0].tolist()); n = len(raw)
        if not 1 <= n <= 512: raise ValueError("coupled port expects 1..512 same-document bytes")
        ctx = causal_context(hidden, mixer.ctx).flatten(2)
        q = F.normalize(mixer.query_key(ctx)[0,:,:mixer.key_dim].float(), dim=-1)
        out = hidden.new_zeros(n, mixer.value_dim).float(); events=[]; gates=[]
        key = value = handle = None
        for stop in range(64,n+1,64):
            if session.rpc('append '+raw[stop-64:stop].hex())['visible'] != stop:
                raise RuntimeError('native append did not expose the expected prefix')
            if stop==512: break
            h=hidden[0,stop-1]; cs=candidates(raw[:stop],session,audit)
            event=dict(stop=stop,candidates=cs); events.append(event)
            if cs:
                lengths, positions, bs=zip(*cs)
                emb=self.byte.weight[list(bs) if self.mode!='echo' else [raw[stop-1]]*len(bs)]
                scores=(emb @ self.query(h)).float()/emb.shape[-1]**.5 + q.new_tensor(lengths)*0.6
                evidence=scores.softmax(0) @ emb.float()
                if self.mode=='matched_echo': evidence=self.byte.weight[raw[stop-1]]
                joint=torch.cat((h,evidence))
                out[stop-1]=out[stop-1]+.05*evidence
                logit=self.action(joint).squeeze(); gates.append((stop,logit)); idx=int(scores.detach().argmax())
                write=(forced[stop] if forced and stop in forced else self.mode=='always' or (self.mode!='skip' and logit.item()>=0))
                event.update(selected=positions[idx],write=bool(write))
                if write:
                    g=self.update(joint).sigmoid(); payload=self.fuse(joint).tanh(); value=g*payload if value is None else g*payload+(1-g)*value
                    key=q[stop-1]; handle=(stop,positions[idx])
            end=min(n,stop+64)
            if handle and end>stop:
                origin,addr=handle
                weights=q[stop:end] @ key
                recalled=weights[:,None]*value
                if self.mode=='no_latent': recalled=recalled*0
                if origin==stop and self.mode!='no_cursor':
                    # Address progression is validated against only the bytes already input.
                    for t in range(stop,min(end,stop+16)):
                        k=t-stop+1
                        if addr+k>=origin or raw[addr:addr+k]!=raw[origin:t+1]: break
                        b=session.rpc(f'byte {addr+k}')['byte']; assert b==raw[addr+k]
                        recalled=recalled.clone(); recalled[t-stop]=recalled[t-stop]+self.byte.weight[b if self.mode not in ('echo','matched_echo') else raw[t]]
                out[stop:end]=out[stop:end]+.05*recalled
        return out[None],events,gates,[]
