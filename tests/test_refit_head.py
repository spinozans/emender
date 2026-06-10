"""Wiring tests for the `refit` (inner-optimization / TTT) head-type.

Verifies the fused kernel is wired as a candidate Emender head-type:
  * allocate_types grows to 9 types WITHOUT breaking the legacy 5/6/8 contract;
  * RefitHead (dim->dim) is differentiable end-to-end, fp32 + autocast-bf16,
    momentum ON (refit) and OFF (gated-delta special case);
  * TypedHeadMixtureLayer builds and runs refit heads alongside the bulk;
  * LadderLM (typed-gdn2-lm) trains a forward/backward end-to-end with refit heads.
Correctness of the kernel math itself lives in tests/test_refit_chunked.py.
"""
import math

import pytest
import torch

from ndm.models.typed_head_mixture import (
    TypedHeadMixtureLayer, allocate_types, TYPE_NAMES,
)

CUDA = torch.cuda.is_available()


def test_allocate_types_legacy_contract_preserved():
    """9th type `refit` added; 5/6/8-length legacy logit vectors still allocate it
    to ZERO heads (right-padded off), reproducing the prior allocation exactly."""
    assert TYPE_NAMES[8] == 'refit'
    for legacy in ([0.5, 0.2, 0.1, 0.1, 0.1],
                   [0.3, 0.0, 0.0, 0.0, 0.0, 0.3],
                   [0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.6]):
        a = allocate_types(48, legacy)
        assert a['n_refit'] == 0, f"legacy len {len(legacy)} allocated refit heads"
        assert sum(a['counts'].values()) == 48
    # explicit 9-vector with mass on refit allocates refit heads
    a9 = allocate_types(16, [-30] * 8 + [1.0])
    assert a9['n_refit'] == 16


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("has_mom", [True, False])
def test_refit_head_module_differentiable(has_mom):
    from ndm.models.refit_head import RefitHead
    dev = 'cuda'
    head = RefitHead(dim=256, n_heads=4, n_state=32, has_mom=has_mom).to(dev)
    x = torch.randn(2, 192, 256, device=dev, requires_grad=True)
    out = head(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.float().pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_typed_head_mixture_with_refit():
    dev = 'cuda'
    # gdn2_recall + refit (avoid the bf16-only e97 heads in this fp32 isolation)
    logits = [math.log(0.5)] + [-30] * 7 + [math.log(0.5)]
    layer = TypedHeadMixtureLayer(dim=256, n_state=32, n_heads=8,
                                  head_type_logits=logits).to(dev)
    assert layer.refit is not None and layer.alloc['counts']['refit'] > 0
    for dt in (None, torch.bfloat16):
        x = torch.randn(2, 128, 256, device=dev, requires_grad=True)
        if dt is None:
            out = layer(x)
        else:
            with torch.autocast('cuda', dtype=dt):
                out = layer(x)
        assert torch.isfinite(out).all()
        out.float().pow(2).sum().backward()
        assert all(torch.isfinite(p.grad).all()
                   for p in layer.refit.parameters() if p.grad is not None)
        layer.zero_grad()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_ladder_lm_refit_end_to_end():
    from ndm.models.ladder_lm import LadderLM
    dev = 'cuda'
    logits = [math.log(0.5)] + [-30] * 7 + [math.log(0.5)]   # gdn2_recall + refit
    model = LadderLM(vocab_size=256, dim=256, depth=2, n_heads=8, n_state=32,
                     level='typed-gdn2-lm',
                     layer_kwargs={'head_type_logits': logits}).to(dev)
    present = any(getattr(m, 'refit', None) is not None
                  for m in model.modules()
                  if m.__class__.__name__ == 'TypedHeadMixtureLayer')
    assert present, "refit head not constructed inside LadderLM"
    ids = torch.randint(0, 256, (2, 128), device=dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(ids)
    logits_out = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(logits_out).all()
    loss = torch.nn.functional.cross_entropy(
        logits_out.float().reshape(-1, 256), ids.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


if __name__ == '__main__':
    test_allocate_types_legacy_contract_preserved()
    print("allocate_types legacy contract preserved OK")
    if CUDA:
        for hm in (True, False):
            test_refit_head_module_differentiable(hm)
        print("RefitHead differentiable (mom on/off) OK")
        test_typed_head_mixture_with_refit()
        print("TypedHeadMixtureLayer + refit OK")
        test_ladder_lm_refit_end_to_end()
        print("LadderLM refit end-to-end OK")
        print("ALL REFIT WIRING OK")
