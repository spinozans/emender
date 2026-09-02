import copy

import pytest
import torch

from ndm.models.ladder_lm import LadderLM


def _model():
    torch.manual_seed(974120)
    return LadderLM(
        vocab_size=64,
        dim=16,
        depth=2,
        level="E97",
        expansion=1.0,
        n_state=4,
        n_heads=4,
        use_gate=True,
        gate_activation="sigmoid",
        linear_state=False,
        use_triton=False,
        mlp_ratio=2.0,
        mlp_multiple=4,
    )


def _packed_controls(device="cpu"):
    doc1 = torch.tensor([2, 3, 4, 5, 6], device=device)
    doc2 = torch.tensor([11, 12, 13, 14, 15, 16], device=device)
    # Seventeen tokens produce sixteen recurrence inputs, matching the fused
    # kernel's sparse-checkpoint interval during training.
    tokens = torch.cat([doc1, doc2, torch.zeros(6, dtype=torch.long, device=device)])
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    valid[: len(doc1) + len(doc2)] = True
    reset = torch.zeros_like(valid)
    reset[0] = True
    reset[len(doc1)] = True
    loss = valid[:-1] & valid[1:] & ~reset[1:]
    return doc1, doc2, tokens.unsqueeze(0), loss.unsqueeze(0), valid.unsqueeze(0), reset.unsqueeze(0)


def test_packed_e97_forward_equals_separate_documents_and_padding_is_invariant():
    model = _model().eval()
    doc1, doc2, packed, _loss, valid, reset = _packed_controls()
    with torch.no_grad():
        packed_logits = model(packed, reset_before=reset, valid_mask=valid)
        logits1 = model(doc1.unsqueeze(0))
        logits2 = model(doc2.unsqueeze(0))
    torch.testing.assert_close(packed_logits[:, : len(doc1)], logits1, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        packed_logits[:, len(doc1): len(doc1) + len(doc2)], logits2,
        rtol=2e-5, atol=2e-6)

    changed = packed.clone()
    changed[:, len(doc1) + len(doc2):] = 63
    with torch.no_grad():
        changed_logits = model(changed, reset_before=reset, valid_mask=valid)
    torch.testing.assert_close(
        changed_logits[:, : len(doc1) + len(doc2)],
        packed_logits[:, : len(doc1) + len(doc2)], rtol=0, atol=0)


def test_packed_e97_gradient_equals_sum_of_separate_document_gradients():
    packed_model = _model().train()
    packed_model.gradient_checkpointing = True
    packed_model.gradient_checkpoint_group_size = 2
    separate_model = copy.deepcopy(packed_model).train()
    separate_model.gradient_checkpointing = False
    doc1, doc2, packed, loss_mask, valid, reset = _packed_controls()

    packed_loss = packed_model(
        packed,
        return_loss=True,
        loss_mask=loss_mask,
        valid_mask=valid,
        reset_before=reset,
        loss_reduction="sum",
    )
    packed_loss.backward()

    separate_loss = separate_model(
        doc1.unsqueeze(0), return_loss=True,
        loss_mask=torch.ones((1, len(doc1) - 1), dtype=torch.bool),
        loss_reduction="sum",
    ) + separate_model(
        doc2.unsqueeze(0), return_loss=True,
        loss_mask=torch.ones((1, len(doc2) - 1), dtype=torch.bool),
        loss_reduction="sum",
    )
    separate_loss.backward()

    torch.testing.assert_close(packed_loss, separate_loss, rtol=2e-5, atol=2e-5)
    for (name_packed, parameter_packed), (name_separate, parameter_separate) in zip(
            packed_model.named_parameters(), separate_model.named_parameters()):
        assert name_packed == name_separate
        torch.testing.assert_close(
            parameter_packed.grad, parameter_separate.grad,
            rtol=2e-4, atol=2e-5, msg=lambda message: f"{name_packed}: {message}")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton integration")
def test_packed_e97_triton_model_forward_backward_smoke():
    torch.manual_seed(974121)
    model = LadderLM(
        vocab_size=64, dim=16, depth=1, level="E97", expansion=1.0,
        n_state=4, n_heads=4, use_gate=True, gate_activation="silu",
        linear_state=False, use_triton=True, mlp_ratio=2.0,
        mlp_multiple=4).cuda().to(torch.bfloat16).train()
    _doc1, _doc2, packed, loss_mask, valid, reset = _packed_controls("cuda")
    loss = model(
        packed, return_loss=True, loss_mask=loss_mask,
        valid_mask=valid, reset_before=reset, loss_reduction="sum")
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_packed_e97_rejects_cross_document_or_padding_targets():
    model = _model()
    _doc1, _doc2, packed, loss_mask, valid, reset = _packed_controls()
    cross_boundary = loss_mask.clone()
    cross_boundary[:, 4] = True
    with pytest.raises(ValueError, match="cross-document"):
        model(
            packed, return_loss=True, loss_mask=cross_boundary,
            valid_mask=valid, reset_before=reset)

    padding_target = loss_mask.clone()
    padding_target[:, 10] = True
    with pytest.raises(ValueError, match="invalid/padding"):
        model(
            packed, return_loss=True, loss_mask=padding_target,
            valid_mask=valid, reset_before=reset)
