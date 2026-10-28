import pytest
from experiments.expressivity_tasks.x40_emender_lm.model import LM


def test_uncertified_store_precision_fails_closed():
    with pytest.raises(ValueError,match='parity'):
        LM(32,store_dtype='bf16')
