import itertools
import math
import random
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from .k15_bookmarks import BookmarkSession
from .k15_phrase import Controller, Factor, Filter, decode, greedy, phrase_source


@contextmanager
def machine(raw=b'', cap=8):
    s = BookmarkSession(Path.home()/'sxgc/xsa/target/debug/xsa')
    try:
        yield Filter(s, Controller().double(), raw, cap)
    finally:
        s.close()


def test_greedy_against_periodic_expansion():
    for size in range(8):
        for seq in itertools.product((0,255), repeat=size):
            text = bytes(seq)
            for cap in (1,2,5):
                i = 0; expected = []
                while i < size:
                    choices = [(k,-d) for k in range(2,cap+1) for d in range(1,i+1)
                               if (text[i-d:i]*cap)[:k] == text[i:i+k] and i+k<=size]
                    if choices:
                        k, d = max(choices); expected.append(Factor(-d,k)); i += k
                    else:
                        expected.append(Factor(text[i])); i += 1
                assert list(greedy(text,cap)) == expected
                assert decode(expected,cap)[0] == text
    text = bytes(range(256))+random.Random(19).randbytes(511)+b'\xff'*25
    assert decode(greedy(text))[0] == text
    for invalid in ([Factor(-1)], [Factor(256)], [Factor(1,1)], [Factor(2), Factor(1,9)]):
        with pytest.raises(ValueError): decode(invalid)


def test_committed_content_alias_and_phrase_alignment():
    head = [Factor(65),Factor(66),Factor(2,2)]
    a, ar = decode(head+[Factor(2,2)])
    b, br = decode(head+[Factor(4,2)])
    assert a == b and ar == br and ar[-1].content == b'AB'
    # Same bigram of phrase IDs recurs; following row starts on a raw boundary.
    _, rows = decode(head+[Factor(88)]+head)
    assert phrase_source(rows).start == 4
    assert phrase_source(rows).payload == 88/255


def replay(text, seal):
    out = []; logits = torch.linspace(-10,10,256,dtype=torch.double)
    with machine(cap=2) as m:
        for y in text:
            n = len(m.tape.raw)
            if n and seal: assert m.tape.session.rpc('seal')['sealed'] == n
            p = m.predict(logits); assert p.min()>0 and abs(float(p.detach().sum())-1)<1e-12
            out.append(p.detach()); m.observe(y)
            assert abs(sum(float(w.detach()) for _, w in m.paths)-1)<1e-12
    return out


def test_growth_seal_and_pre_observation():
    a = bytes([0,255,30,0,255,30,0,255]); b = a[:-1]+b'x'
    for x,y,z in zip(replay(a,False),replay(a,True),replay(b,False)):
        assert torch.equal(x,y) and torch.equal(x,z)


def reference(raw, future, delta, cap=8):
    """Enumerate successful parses as segment lengths (not Filter states).

    Return unnormalized prefix likelihood, including open final COPY. Literal
    law uniform in this fixture. Delta changes ONLY initial address and op logits.
    """
    sigmoid = lambda x: 1/(1+math.exp(-x))
    def run(text, phrases, remaining):
        if not remaining: return 1.
        n = len(text); k = len(remaining); first = n==len(raw)
        positions = [i+2 for i in range(n-2) if text[i:i+2]==text[-2:]]
        r = None
        for depth in (2,1):
            matches = [j for j in range(depth,len(phrases)) if phrases[j-depth:j]==phrases[-depth:]]
            if matches:
                j = matches[-1]; r = sum(map(len,phrases[:j])); payload=sum(phrases[j])/(255*len(phrases[j])); break
        aa = sorted({n-1,*positions[-2:]} | ({r} if r is not None else set()))
        scores = [math.exp(.5*(a-n)/n+(.5+.25*payload if a==r else 0)+(delta[i] if first else 0)) for i,a in enumerate(aa)]
        start = sigmoid(.25+.5*math.log1p(len(aa))-.5/256+(delta[-1] if first else 0))
        total = (1-start)/256*run(text+remaining[:1],phrases+[remaining[:1]],remaining[1:])
        for a,w in zip(aa,scores):
            prob = start*w/sum(scores)
            for l in range(1,min(k,cap)+1):
                if remaining[l-1] != (text[a:]*cap)[l-1]: break
                keep = sigmoid(1-.5*l/cap) if l<cap else 0.
                total += prob*(1 if l==k else 1-keep)*run(text+remaining[:l],phrases+[remaining[:l]],remaining[l:])
                prob *= keep
        return total
    return run(raw,[bytes([b]) for b in raw],future)


def test_later_only_credit_to_first_choice():
    raw = b'xyAB1xyAB2xy'; y = b'AB2'; delta = torch.zeros(4,dtype=torch.double,requires_grad=True)
    with machine(raw) as m:
        a = m.controller.address.register_forward_hook(lambda _, args, out: out+delta[:3,None])
        op = m.controller.phase.register_forward_hook(lambda _, args, out: out+torch.stack((delta[3],delta[3]*0)))
        for i,b in enumerate(y):
            p = m.predict(torch.zeros(256,dtype=torch.double)); m.observe(b)
            if i==0: a.remove(); op.remove()
            expected = reference(raw,y[:i+1],[0]*4)/reference(raw,y[:i],[0]*4)
            assert float(p[b].detach()) == pytest.approx(expected,abs=1e-12)
        grad, = torch.autograd.grad(-p[y[-1]].log(),delta)
    def loss(v): return -math.log(reference(raw,y,v)/reference(raw,y[:-1],v))
    for i,g in enumerate(grad):
        h = 1e-5; v = [0]*4; v[i]=h; hi=loss(v); v[i]=-h
        assert float(g) == pytest.approx((hi-loss(v))/(2*h),abs=2e-9)
    assert grad[0]>0>grad[1] and abs(grad[-1])>1e-8
    print('later-only gradient [source2,source7,last,start]',grad.tolist())


def test_all_bounded_binary_paths_and_cap_exit():
    raw = b'01010'; delta = [0]*5
    for future in itertools.product(b'012', repeat=3):
        with machine(raw,cap=2) as m, torch.no_grad():
            likelihood = 1.
            for y in future:
                p = m.predict(torch.zeros(256,dtype=torch.double))
                likelihood *= float(p[y]); m.observe(y)
                for s,_ in m.paths:
                    assert not s.distance or 1 <= len(m.tape.raw)-s.start <= 2
            assert likelihood == pytest.approx(reference(raw,bytes(future),delta,2),rel=1e-12,abs=0)


def test_forced_phrase_operand_outside_raw_budget_after_growth():
    from .k15_phrase import State
    # Three equal expanded AB strings; middle two ABs are single COPY
    # phrases, not the two-phrase suffix [A,B]. Raw hit cap excludes G.
    factors = [Factor(0x1e)]*129+[Factor(65),Factor(66),Factor(71),Factor(3,2),
                                     Factor(49),Factor(3,2),Factor(50),Factor(65),Factor(66)]
    text, rows = decode(factors); source = phrase_source(rows)
    assert source.start == 131 and source.content == b'G' and len(rows)>128
    for seal in (False,True):
        with machine(text) as m:
            if seal: assert m.tape.session.rpc('seal')['sealed']==len(text)
            assert source.start not in m.tape.sources()
            for a in (-1,len(text)):
                with pytest.raises(ValueError): m.tape.byte(a)
            m.paths = [(State(rows),m.paths[0][1])]  # forced past parse, not inferred
            m.predict(torch.zeros(256,dtype=torch.double)); m.observe(71)
            assert any(s.distance==len(text)-source.start and float(w.detach())>0 for s,w in m.paths)
