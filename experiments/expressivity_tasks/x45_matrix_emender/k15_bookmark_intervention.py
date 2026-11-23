"""D intervention-enabled port, forked from B without editing its engine."""
import torch
import torch.nn.functional as F
from .k15_bookmarks import BookmarkPort, causal_context, EPS

class InterventionPort(BookmarkPort):
    intervention = 'none'
    def forward(self, mixer, hidden, ids, session, forced=None, audit=False, stride=16):
        ctx = causal_context(hidden, mixer.ctx).flatten(2)
        queries = mixer.query_key(ctx)[..., :mixer.key_dim]
        keys = hidden.new_zeros(2, mixer.key_dim)
        values = hidden.new_zeros(2, mixer.value_dim)
        # Immutable byte strings and occurrence starts are non-differentiable fields
        # of the same two rows as keys/values. Invalidated together on retirement.
        handles = [None, None]
        output, trace, log_probs, writes = [], [], [], []
        raw = bytes(ids[0].tolist())
        for i, start in enumerate(range(0, len(raw), stride)):
            stop = min(start + stride, len(raw))
            chunk = hidden[:, start:start+stride]
            q = F.normalize(queries[:, start:start+stride].float(), dim=-1, eps=EPS)
            # K15 additive read on reserved rows, without softmax. Empty keys and
            # values are zero. Original rows still use the incumbent fast scan.
            weight = mixer.policy.read_scale * (q @ F.normalize(keys.float(), dim=-1, eps=EPS).T)
            recalled = weight @ values
            scores = self.slot_bias(chunk) + weight
            output.append(recalled[:, :stop-start])
            assert session.rpc('append ' + raw[start:stop].hex())['visible'] == stop
            if stop == len(raw):
                break
            h = chunk[0, -1]
            def choose(logits, forced_value):
                dist = torch.distributions.Categorical(logits=logits.float())
                idx = (forced_value if forced_value is not None else
                       int(dist.sample() if self.training else logits.argmax()))
                log_probs.append(dist.log_prob(logits.new_tensor(idx).long()))
                return idx
            slot = choose(scores[0, -1], None if forced is None else forced[i][0])
            prior = handles[slot]
            logits = self.action(torch.cat([h, values[slot]]))
            if prior is None:
                logits = logits.masked_fill(torch.arange(4, device=h.device) >= 2, -torch.inf)
            op = choose(logits, None if forced is None else forced[i][1])
            event = dict(stop=stop, slot=slot, op=op, calls=[])
            def read(cmd):
                ans = session.rpc(cmd)
                event['calls'].append((cmd, ans))
                return ans
            if op == 3:
                k, v = keys[slot]*0, values[slot]*0
                handles[slot] = None
            else:
                if op == 2:
                    pattern, pos = prior
                    offset = pos + len(pattern)
                    assert offset < stop
                    b = read(f'byte {offset}')['byte']
                    pattern = (pattern + bytes([b]))[-32:]
                else:
                    pattern = raw[max(0, stop-(3 if op == 0 else 12)):stop]
                ans = read(f'locate {pattern.hex()} 0 {stop} {stop}')
                hits = ans['positions']
                if audit:
                    assert hits == [j for j in range(stop-len(pattern)+1)
                                    if raw.startswith(pattern, j, stop)]
                pos = hits[0]
                offset = min(stop-1, pos+len(pattern))
                b = read(f'byte {offset}')['byte']
                if audit:
                    assert raw[offset] == b
                response = h + self.byte.weight[b]
                write_ctx = torch.cat([ctx[0, stop-1, :-mixer.width], response])
                k = mixer.key(write_ctx)[:mixer.key_dim]
                gate = 1 - mixer.write_scalar[0] * mixer.b_proj(response)[:mixer.value_dim].sigmoid()
                payload = mixer.value(response)[:mixer.value_dim]
                v = gate * payload + (1 - gate) * values[slot]
                handles[slot] = (pattern, hits[-1] if self.intervention == 'address' else pos)
                event['saved_pos'] = handles[slot][1]
                if self.intervention == 'clear':
                    k, v = k * 0, v * 0
                elif self.intervention == 'payload':
                    v = -v
                writes.append(v)
            mask = (torch.arange(2, device=h.device) == slot)[:, None]
            keys = torch.where(mask, k, keys)
            values = torch.where(mask, v, values)
            trace.append(event)
        return torch.cat(output, dim=1), trace, log_probs, writes


