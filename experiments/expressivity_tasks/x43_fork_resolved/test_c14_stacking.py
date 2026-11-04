"""Numerical and foreign-array edge cases for the frozen C13 recipe."""
import numpy as np
from .c14_stacking_score import objective, fuse, masks_for, add, table


def test_foreign_forced_contradiction_stays_infinite():
    p=np.array([.2,.3,.4,.5]); q=np.array([1.,0.,0.,0.]); support=np.array([1,1,1,0],bool)
    fork=np.array([0,0,1,0],bool); g=np.full(4,.5)
    actual=fuse(p,q,support,fork,g,True)
    np.testing.assert_allclose(actual,[1.,0.,.2,.5])
    with np.errstate(divide='ignore'): loss=-np.log(actual)
    sums={}; add(sums,'x',loss); row=table(sums)['x']
    assert row['targets']==4 and row['infinite_losses']==1 and row['nll']=='Infinity'
    assert np.isfinite(-np.log(fuse(p,q,support,fork,g,False))).all()


def test_log_metadata_bin_edges_and_crossed_masks():
    a=np.zeros((9,10),np.float32)
    a[:,5]=1; a[:,6]=np.log1p([1,3,4,12,13,64,65,128,0]); a[-1,5]=0
    a[:,4]=[1,0,1,0,1,0,1,0,0]; a[:,7]=[1,0,1,0,1,0,1,0,0]
    masks=masks_for(a)
    for label in ['1-3','4-12','13-64','65+']:
        assert masks['length_'+label].sum()==2
        assert masks['length_'+label+'/supported/fork'].sum()==1
        assert masks['length_'+label+'/zero/nonfork'].sum()==1
    assert masks['forced_zero_target'].sum()==4


def test_gate_gradient_including_clipped_logit():
    x=np.array([[1.,.1,.4,-.7],[1.,-.3,.8,.2],[1.,.6,-.9,.1]])
    p=np.array([.1,.2,.4]);q=np.array([.9,0,.3]);w=np.array([.2,-.1,.3,.1])
    loss,grad=objective(w,x,p,q)
    for i in range(4):
        delta=np.eye(4)[i]*1e-5
        numerical=(objective(w+delta,x,p,q)[0]-objective(w-delta,x,p,q)[0])/2e-5
        np.testing.assert_allclose(grad[i],numerical,rtol=1e-6,atol=1e-8)
    assert np.isfinite(loss)
    assert np.array_equal(objective(np.array([100.,0,0,0]),x,p,q)[1],np.zeros(4))
