import pytest
from experiments.expressivity_tasks.x40_emender_lm.final_depth_eval import validate_binding
from experiments.expressivity_tasks.x42_pile_scale.checkpoint import identity


def test_current_registration_needs_no_amendment():
    registration={'seed':0}
    assert validate_binding(dict(version=1,identity=identity(registration),step=30000),'hash',registration)=='current'


def test_amendment_covers_only_exact_checkpoint_and_registration():
    old={'seed':0};new={'seed':0,'evaluation':'certified reads'}
    state=dict(version=1,identity=identity(old),step=30000)
    amendment=dict(checkpoint_sha256='hash',recorded_identity=identity(old),
                   rebound_identity=identity(new),checkpoint_update=30000,cause_chain=['evaluation guard fix'])
    assert validate_binding(state,'hash',new,amendment)=='amendment-covered'
    with pytest.raises(ValueError,match='mismatch'):validate_binding(state,'hash',new)
    for key,value in [('checkpoint_sha256','wrong'),('recorded_identity','wrong'),
                      ('rebound_identity','wrong'),('checkpoint_update',29999),('cause_chain',[])]:
        with pytest.raises(ValueError,match='mismatch'):
            validate_binding(state,'hash',new,dict(amendment,**{key:value}))
    with pytest.raises(ValueError,match='mismatch'):
        validate_binding(state,'hash',{'seed':1},amendment)
