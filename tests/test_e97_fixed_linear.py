import pytest
import torch
from ndm.triton.fixed_fp32_linear import fixed_fp32_linear,_fixed_linear


def test_runtime_height_is_not_specialized_or_alignment_specialized():
    parameter=next(p for p in _fixed_linear.params if p.name=='M')
    assert parameter.do_not_specialize and parameter.do_not_specialize_on_alignment
    assert not hasattr(_fixed_linear,'configs')


def test_fixed_linear_rejects_cpu_without_launching():
    with pytest.raises(ValueError,match='CUDA'):
        fixed_fp32_linear(torch.zeros(1,3),torch.zeros(2,3))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification required')
def test_fixed_linear_bias_tail_and_row_independence():
    from scripts.qualify_e97_fixed_linear import small_cuda_checks
    torch.set_float32_matmul_precision('highest')
    assert small_cuda_checks(torch.device('cuda',torch.cuda.current_device()))
