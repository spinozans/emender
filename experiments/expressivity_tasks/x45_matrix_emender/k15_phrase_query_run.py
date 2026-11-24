"""Reproduce the registered paired CPU intervention. Run from repository root with -m."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from .k15_committed_phrase import scan_sources
from .k15_committed_phrase_machine import Machine
from .k15_committed_phrase_run import opened
from .k15_phrase_query import PhraseQuery


def fixture(cue, rebound):
    a, b = b'Ux12', b'Uy34'
    if rebound:
        a, b = b, a
    tail = b'ABCDUx12Uy34!' + b'ABCD'+a+b'ABCD'+b+bytes([cue])+b'ABCD'
    return bytes(range(128, 256-len(tail)))+tail


def second_law(scores, posterior_start, keep, literal, second):
    survive = posterior_start * scores.softmax(0) * keep
    return (literal*(1-survive.sum())).index_add(0,second,survive)


def replay(model, head, cue, rebound, stats, force=None, clear=False):
    prefix = fixture(cue,rebound)
    with opened(stats) as ix:
        m = Machine(model, ix, prefix, head)
        depth, ids = ix.sources()
        assert (depth, ids) == scan_sources(ix.parse)
        spans = [ix.parse.rows[i] for i in ids]
        contents = [prefix[r.start:r.end] for r in spans]
        assert sorted(contents)==[b'Ux12',b'Uy34']
        y = ord('U'); target = ord('x') + cue-250
        law = m.predict(force=None if force is None else ids[force],clear=clear)
        h,lit,edges,_ = m.pending
        state = torch.stack([m.row_memory[i] for i in ids])
        start = sum(e[3] for e in edges if e[0]==1)
        scores = m.scores[1]
        m.observe(y)
        posterior = sum(m.post[(i,1)] for i in ids)
        assert ix.sources()==(0,[])
        later = m.predict(); lit2 = m.pending[1]
        keep = torch.stack([m.cursor_logits[i,1].clamp(-12,12).sigmoid() for i in ids])
        assert torch.isclose(law.sum(),h.new_ones(())) and torch.isclose(later.sum(),h.new_ones(()))
        second = torch.tensor([s[1] for s in contents])
        selected = None if force is None else m.values[ids[force]]
        return dict(h=h, state=state, scores=scores, keep=keep, lit=lit2, second=second,
                    posterior=posterior, target=target, addresses=ids, spans=[vars(r) for r in spans],
                    depth=depth, start=start, first=law, later=later, selected=selected)


def decisions(head, cases):
    rows=[]
    for j,c in enumerate(cases):
        best = (c['second']==c['target']).nonzero().item(); prior=c['scores']
        for name,s in [('fitted',prior+head.relative(c['h'],c['state'])),('compiled',prior),
                       ('uniform',prior*0),('wrong_query',prior+head.relative(cases[j^1]['h'],c['state']))]:
            law=second_law(s,c['posterior'],c['keep'],c['lit'],c['second'])
            rows.append(dict(case=j,arm=name,margin=float(s[best]-s[1-best]),nll=float(-law[c['target']].log()),
                             top=c['addresses'][s.argmax()] if s[0]!=s[1] else None,
                             target=c['target'],good=c['addresses'][best],weights=s.softmax(0).tolist()))
    return rows


def run(path):
    from .k15_phrase_qualification import backbone, ANCHOR_SHA
    from .k15_prefix_machine import PrefixK15
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.get_num_threads()==4
    torch.manual_seed(3411); stats=Counter()
    model=PrefixK15(backbone()).requires_grad_(False)
    head=PhraseQuery(model.write.out_features); value=head.value.weight.detach().clone()
    head.requires_grad_(False); head.value.weight.zero_()
    arms=[(cue,r) for r in (False,True) for cue in (250,251)]
    with torch.no_grad():
        cases=[replay(model,head,*arm,stats) for arm in arms]
        head.center.copy_((cases[0]['h']+cases[1]['h'])/2)
        head.scale.copy_((cases[0]['h']-head.center).square().mean().sqrt().clamp_min(1e-8))
        before=decisions(head,cases)
    head.key.requires_grad_(True); head.query.requires_grad_(True)
    opt=torch.optim.Adam([head.key.weight,head.query.weight],lr=.02); gradients=[]
    for step in range(80):
        loss=0
        for c in cases[:2]:
            s=c['scores']+head.relative(c['h'],c['state'])
            law=second_law(s,c['posterior'],c['keep'],c['lit'],c['second'])
            loss=loss-law[c['target']].log()/2
        opt.zero_grad(); loss.backward()
        if step<2: gradients.append([float(p.grad.norm()) for p in (head.query.weight,head.key.weight)])
        opt.step()
    with torch.no_grad():
        after=decisions(head,cases); traces=[]
        for arm,c in zip(arms,cases):
            live=replay(model,head,*arm,stats); s=c['scores']+head.relative(c['h'],c['state'])
            ref=second_law(s,c['posterior'],c['keep'],c['lit'],c['second'])
            delta=float((ref-live['later']).abs().max()); assert delta<1e-5
            traces.append(dict(arm=arm,rows=c['spans'],ids=c['addresses'],depth=c['depth'],keep=c['keep'].tolist(),
                               start=float(c['start']),posterior=float(c['posterior']),cache_error=delta,
                               live_nll=float(-live['later'][c['target']].log())))
    torch.save(head.state_dict(),path)
    head.value.requires_grad_(True); head.value.weight.data.copy_(value)
    witnesses=[]; literals=[]; first=[]
    for force,clear in ((0,False),(1,False),(0,True)):
        c=replay(model,head,250,False,stats,force,clear); lit=c['lit']; first.append(c['first'].detach())
        grad,=torch.autograd.grad(lit[120].log(),c['selected']); literals.append(lit.detach())
        witnesses.append(dict(forced=c['addresses'][force],clear=clear,grad_norm=float(grad.norm()),observed=85))
    assert all(torch.equal(first[0],p) for p in first)
    deltas=[float((p-literals[0]).abs().max()) for p in literals[1:]]
    assert min(deltas)>0 and witnesses[0]['grad_norm']>0 and witnesses[2]['grad_norm']==0
    return dict(anchor=ANCHOR_SHA,before=before,after=after,traces=traces,gradient_norms=gradients,
                witnesses=witnesses,next_literal_linf=deltas,native=stats)


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
