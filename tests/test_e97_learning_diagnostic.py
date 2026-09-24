import numpy as np
from scripts.freeze_e97_learning_diagnostic import first_target


def test_only_first_supervised_run_is_scored():
    result=first_target(np.arange(10),np.array([0,0,1,1,0,1,1,1,1,1]))
    assert result==([0,1],[2,3],2)


def test_complete_prefix_and_bounded_target():
    result=first_target(np.arange(100),np.array([0]*10+[1]*90),max_target=32)
    assert result==(list(range(10)),list(range(10,42)),10)


def test_no_prefix_truncation_or_unsupervised_samples():
    tokens=np.arange(10)
    assert first_target(tokens,np.zeros(10)) is None
    assert first_target(tokens,np.ones(10)) is None
    assert first_target(tokens,np.array([0]*9+[1]),max_prefix=8) is None
