import pytest
import torch
from experiments.expressivity_tasks.x40_emender_lm.scale_surface import SHAPES,SurfaceLM,counts


@pytest.mark.parametrize('scale',tuple(SHAPES))
def test_full_memory_stack_and_total_capacity_control(scale):
    with torch.device('meta'):
        pure=SurfaceLM(scale,'pure')
        actual=counts(pure)
        control=SurfaceLM(scale,'transformer',actual['total'])
    assert len(pure.blocks)==SHAPES[scale][1]
    assert all(len(b.cell.configs)==4 and not hasattr(b,'qkv') for b in pure.blocks)
    assert all(hasattr(b,'qkv') for b in control.blocks)
    assert abs(actual['trainable']/SHAPES[scale][2]-1)<.001
    assert abs(counts(control)['total']/actual['total']-1)<.001
