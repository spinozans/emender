import numpy as np
from experiments.expressivity_tasks.x45_matrix_emender import k15_singleton_credit as o


def test_shifted_dense_singleton_replay(monkeypatch):
    class State:
        def __init__(self,raw): self.raw=raw[:o.START]; self.session=self
        def close(self): pass
        def append(self,b): self.raw+=b
        def read(self): return {self.raw[-1]:3,1:2}
    monkeypatch.setattr(o,'prefill',State)
    def local(net,prefixes):
        assert len(prefixes)==1 and len(prefixes[0])==128
        p=np.ones((1,256)); p[0,prefixes[0][-1]]=10; return p/p.sum()
    monkeypatch.setattr(o,'local',local); net=type('Net',(),{'training':False})(); raw=(b'AAAB'*1024); x,dense=o.extract(net,raw)
    assert x[0,5]==x[0,6]==0 and np.any(x[:,5]!=x[:,6]); assert not np.array_equal(x[:,5],np.roll(x[:,5],1))
    for arm in o.ARMS:
        w=[.2]*(1 if arm=='global' else 2 if arm in ('support','online') else 7); a=o.mass(x[:32,:5],o.credit(x[:32],arm),arm,w)[:,None]; expected=(1-a)*dense[:,0]+a*dense[:,1]
        np.testing.assert_allclose(o.replay(net,raw[:o.START+31],arm,w),expected)
