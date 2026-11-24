"""Single-prefix differentiable feedback; no neural choice of factor boundaries."""
import torch
from torch.nn import functional as F


def conditional_loss(start, source, keep, labels):
    # keep[t] is the live KEEP logit AFTER observation t of this rollout,
    # never the START-prefix logit broadcast over future decisions.
    out = start*0
    if labels['start'] is not None:
        out = F.binary_cross_entropy_with_logits(start,start.new_tensor(float(labels['start'])))
    if labels['best']:
        out = out + source.logsumexp(0)-source[labels['best']].logsumexp(0)
    observed = [(i,v) for row in labels['keep'] for i,v in enumerate(row)]
    if observed:
        offsets, values = zip(*observed)
        out = out + F.binary_cross_entropy_with_logits(keep[list(offsets)],keep.new_tensor(values))
    return out


class Machine:
    def __init__(self, model, index, prefix, controller=None):
        assert len(prefix)==128
        self.m, self.index = model,index
        self.feedback = model.write.bias.new_zeros(model.write.out_features)
        self.values = []
        self.controller = controller
        self.row_memory = []
        for y in prefix:
            for row in index.observe(y):
                self._write(row, None)
        self.last_hidden = None
        self.post = {None:self.feedback.new_ones(())}
        self.pending = None
        self.calls = 0

    def _write(self, row, hidden):
        v = self.m.write(hidden).tanh() if hidden is not None else torch.zeros_like(self.feedback)
        if self.controller is not None:
            state = self.controller.encode(self.index.parse.raw[row.start:row.end])
            self.row_memory.append(state)
            v = v + self.controller.value(state).tanh()
        self.values.append(v)

    def literal(self, raw, neural_law):
        return neural_law

    def predict(self, force=None, clear=False):
        assert self.pending is None and not self.index.parse.ended
        f = self.index.parse
        h, lit = self.m.neural(f.raw,self.feedback)
        self.calls += 1
        lit = self.literal(f.raw, lit)*(1-1e-6)+1e-6/256
        start, keep = self.m.phase(h).unbind()
        # Keep raw logits for supervision; FP32 mixture complements must stay nonzero.
        start_prob = start.float().clamp(-12,12).sigmoid()
        keep_prob = keep.float().clamp(-12,12).sigmoid()
        depth, ids = self.index.sources()
        source = torch.stack([self.m.operand(torch.stack((h.new_tensor(float(depth)),
                    h.new_tensor((f.rows[i].start-len(f.raw))/len(f.raw)), self.values[i].mean()))).squeeze(0) for i in ids]) if ids else h[:0]
        if self.controller is not None and ids:
            source = source + self.controller.relative(h, torch.stack([self.row_memory[i] for i in ids]))
        self.cursor_logits = {}
        weights = source.softmax(0)
        if force is not None:
            weights = F.one_hot(torch.tensor(ids.index(force)),len(ids)).to(h)
        edges = []
        idle = h.new_zeros(())
        def copy(op, i, offset, mass):
            row = f.rows[i]
            assert row.start+offset < row.end <= len(f.raw)
            byte = self.index.raw.rpc(f'byte {row.start+offset}')['byte']
            edges.append((op,(i,offset+1),byte,mass,self.values[i]*(0 if clear and op==1 and i==force else 1)))
        for cursor,p in self.post.items():
            if cursor is not None:
                i,k = cursor
                if k < f.rows[i].end-f.rows[i].start:
                    logit = keep if self.controller is None else self.controller.continue_logit(keep,self.row_memory[i],k)
                    self.cursor_logits[cursor] = logit
                    prob = keep_prob if self.controller is None else logit.float().clamp(-12,12).sigmoid()
                    copy(2,i,k,p*prob); idle = idle+p*(1-prob); continue
            idle = idle+p
        a = start_prob if ids else start*0
        for i,w in zip(ids,weights):
            copy(1,i,0,idle*a*w)
        literal_mass = idle*(1-a)
        law = lit*literal_mass
        if edges:
            law = law.index_add(0,torch.tensor([e[2] for e in edges]),torch.stack([e[3] for e in edges]))
        edges.append((0,None,None,literal_mass,h*0))
        self.pending = h,lit,edges,law
        self.scores = start,source,keep
        return law

    def observe(self,y):
        h,lit,edges,law = self.pending
        post = {}; feedback = torch.zeros_like(h)
        for op,key,byte,p,value in edges:
            if byte is not None and byte!=y: continue
            w = p*(lit[y] if byte is None else 1)/law[y]
            post[key] = post.get(key,0)+w
            feedback = feedback+w*(self.m.transition.weight[op]+value)
        self.feedback = .125*(feedback+.25*self.feedback).tanh()
        self.post = post
        for row in self.index.observe(y):
            # On mismatch the copy ended at the preceding observation.
            last = h if row.end == len(self.index.parse.raw) else self.last_hidden
            self._write(row,last)
        self.last_hidden = h
        self.pending = None

    def end(self):
        """External document-end event, never implicit at a scoring seam."""
        if self.pending is not None:
            raise RuntimeError('observation pending')
        for row in self.index.end():
            h = self.last_hidden
            self._write(row,h)
