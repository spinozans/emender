"""Coverage partitions and paired denominator regression tests."""
import numpy as np
from experiments.expressivity_tasks.x43_fork_resolved.c14_paired_eval import strata, add_sums


def test_support_partitions_include_zero_targets_and_mask_padding():
    f=np.zeros((2,128,260),np.float32)
    y=np.zeros((2,128),np.int64)
    lengths=np.array([3,18])
    f[0,0,256]=1; f[0,0,0]=.3; f[0,0,1]=.7
    f[0,1,256]=1; f[0,1,1]=1; f[0,1,258]=1
    f[1,127,256]=1  # Padding must never enter coverage.
    masks=strata(f,y,lengths)
    assert masks['total'].sum()==21
    assert masks['wikipedia_support'].sum()==2
    assert masks['empty_support'].sum()==19
    assert masks['supported_target'].sum()==1
    assert masks['nonempty_zero_target'].sum()==1
    assert masks['zero_target'].sum()==20
    assert masks['agreement'].sum()==1
    assert masks['argmax_agreement'].sum()==0
    assert masks['fork'].sum()==1
    assert masks['window_reset_1_16'].sum()==19
    assert masks['window_interior_17_128'].sum()==2
    np.testing.assert_array_equal(masks['wikipedia_support'] | masks['empty_support'],masks['total'])
    np.testing.assert_array_equal(masks['supported_target'] | masks['zero_target'],masks['total'])
    assert not (masks['supported_target'] & masks['zero_target']).any()
    sums={}; losses=np.ones((2,128),np.float32)
    add_sums(sums,'v2_on','final_test',losses,masks,np.array([False,True]))
    assert sums['v2_on/final_test/total']==[18.,18]
    assert sums['v2_on/final_test/wikipedia_support']==[0.,0]
    add_sums(sums,'v2_on','final_test',2*losses,masks,np.array([True,False]))
    assert sums['v2_on/final_test/total']==[24.,21]
