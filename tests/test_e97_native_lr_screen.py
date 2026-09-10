from pathlib import Path
import pytest
from scripts.prepare_e97_native_lr_screen import render_trial,replace_once

TEMPLATE="""ROOT=/baseline
eval "$(bash scripts/gpu_lease.sh acquire 8 --no-wait)"
--lr 5e-5 --warmup-steps 0
 assert checkpoint['sft_precision']['optimizer']=='bf16-sr-candidate'
 updates=8,world_size=8,assistant_target_tokens=123
"""


def test_trial_changes_only_declared_launcher_anchors():
    rendered=render_trial(TEMPLATE,Path('/baseline'),Path('/trial'),.0002)
    assert 'ROOT=/trial\n' in rendered
    assert '--lr 0.0002 --warmup-steps 0' in rendered
    assert 'LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)\neval "$LEASE"' in rendered
    assert "assert checkpoint['sft_precision']['learning_rate']==recipe['lr']" in rendered
    assert 'updates=8,world_size=8,assistant_target_tokens=123' in rendered


@pytest.mark.parametrize('rate',[0,-1,float('inf'),float('nan')])
def test_invalid_rate_fails(rate):
    with pytest.raises(ValueError):render_trial(TEMPLATE,Path('/baseline'),Path('/trial'),rate)


def test_ambiguous_or_missing_anchor_fails():
    with pytest.raises(ValueError):replace_once('same same','same','new')
    with pytest.raises(ValueError):replace_once('old','missing','new')
