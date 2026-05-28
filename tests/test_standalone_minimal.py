"""CPU-only smoke test for standalone ndm package import and forward pass.

Run with:
    python -m pytest tests/test_standalone_minimal.py -m 'not gpu'
"""

import pytest
import torch


def test_version_exported():
    import ndm
    assert hasattr(ndm, "__version__"), "ndm.__version__ not exported"
    assert isinstance(ndm.__version__, str)


def test_public_exports():
    from ndm import (
        StockElman, StockElmanCell,
        LadderLM, create_ladder_model,
        get_available_levels, get_ladder_level,
    )
    assert StockElman is not None
    assert LadderLM is not None


def test_e88_fused_importable():
    from ndm.models.e88_fused import E88FusedLM
    assert E88FusedLM.__name__ == "E88FusedLM"


def test_ladderlm_cpu_forward():
    """LadderLM (level=0 StockElman) forward pass on CPU with small config.

    Skipped when mamba_ssm is installed but CUDA is unavailable: mamba_ssm's
    Triton fused-norm kernel requires CUDA. Without mamba_ssm (the default for
    a clean install), LadderLM falls back to nn.RMSNorm and runs on CPU fine.
    """
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm  # noqa: F401
        fused_norm_available = True
    except ImportError:
        fused_norm_available = False

    if fused_norm_available:
        pytest.skip(
            "mamba_ssm Triton fused-norm is installed; LadderLM uses it for all "
            "forward passes and it requires CUDA tensors. CPU testing is only "
            "valid when mamba_ssm is absent (e.g., clean pip install). Known "
            "issue: mamba_ssm is not in ndm's dependencies, so a clean install "
            "falls back to nn.RMSNorm and runs on CPU without issues."
        )

    from ndm import LadderLM

    model = LadderLM(vocab_size=256, dim=64, depth=2, level=0)
    model.eval()

    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape == (2, 16, 256), f"Unexpected shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"


def test_e88_fused_cpu_forward():
    """E88FusedLM forward pass on CPU with small config (PyTorch fallback)."""
    from ndm.models.e88_fused import E88FusedLM

    # dim=64, n_heads=4, n_state=16: key_dim = 4*16 = 64 (divides dim)
    model = E88FusedLM(vocab_size=256, dim=64, depth=2, n_heads=4, n_state=16)
    model.eval()

    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape == (1, 8, 256), f"Unexpected shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"


def test_e97_reference_cpu_forward():
    """E97 split-edit reference path forward pass on CPU."""
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm  # noqa: F401
        fused_norm_available = True
    except ImportError:
        fused_norm_available = False

    if fused_norm_available:
        pytest.skip("mamba_ssm fused norm requires CUDA tensors")

    from ndm import LadderLM

    model = LadderLM(vocab_size=256, dim=64, depth=1, level="E97", n_heads=4, n_state=16)
    model.eval()

    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape == (1, 8, 256), f"Unexpected shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"


@pytest.mark.gpu
def test_ladderlm_gpu_forward():
    """LadderLM forward pass on GPU (skipped if CUDA unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from ndm import LadderLM

    model = LadderLM(vocab_size=256, dim=64, depth=2, level=0).cuda()
    model.eval()

    x = torch.randint(0, 256, (1, 16)).cuda()
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape == (1, 16, 256)
