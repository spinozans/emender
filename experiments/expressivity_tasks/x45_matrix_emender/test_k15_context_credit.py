import numpy as np
from .k15_context_credit import OutcomeRows, evidence, mass


def trace(raw):
    rows=OutcomeRows(); p=np.full(256,1/256); h,f=evidence(p,{97:1}); out=[]
    for t in range(2,len(raw)):
        key=raw[t-2:t]; c,hit=rows.read(key); out.append((c,hit,mass(f,c,'writable',[-1,0,0,0,0,0,2])))
        rows.observe(key,p,h,raw[t]);
    return out


def test_shift_revisit_and_future():
    a=trace(b'aaaabaa'); b=trace(b'aaaaXYZ'); assert a[:3]==b[:3]
    assert a[0][:2]==(0,False); assert a[1][0]==(255/257)/3; assert a[2][0]==(1.9*(255/257))/3.9
    assert a[3][1] is False


def test_absent_hist_never_writes():
    r=OutcomeRows(); p=np.full(256,1/256); h,f=evidence(p,{}); r.observe(b'ab',p,h,42); assert not r.rows
    assert mass(f,0.,'global',[3])==0


def test_production_replay_future_perturbation(monkeypatch):
    # Actual resident suffix service; substitute deterministic local probabilities
    # so this test cannot silently require a GPU or an active broker lease.
    from experiments.expressivity_tasks.x45_matrix_emender import k15_context_run as run
    def local(_, prefixes):
        out=np.ones((len(prefixes),256)); out[np.arange(len(prefixes)),[p[-1] for p in prefixes]]=2
        return out/out.sum(axis=1,keepdims=True)
    monkeypatch.setattr(run,'local',local)
    raw=b'abcd'*1024; future=3853; other=raw[:future]+b'x'*(len(raw)-future)
    weights=[-1,.1,.3,0,0,.4,2]; first,_=run.replay(None,raw[:3871],'writable',weights,2)
    altered,_=run.replay(None,other[:3871],'writable',weights,2)
    np.testing.assert_array_equal(first[:future-run.START+1],altered[:future-run.START+1]); assert not np.array_equal(first,altered)
    x,dense=run.extract(None,raw,True); a=mass(x[:32,:5],x[:32,6],'writable',weights)[:,None]
    np.testing.assert_array_equal(first,(1-a)*dense[:,0]+a*dense[:,1]); np.testing.assert_allclose(first.sum(1),1)
