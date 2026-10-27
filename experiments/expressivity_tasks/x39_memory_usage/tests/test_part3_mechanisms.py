import torch

from experiments.expressivity_tasks.x39_memory_usage.part3_mechanisms import (
    poverty_receipt, semantic_receipt,
)


def test_capacity_loss_signal_uses_real_pressure_and_differentiable_reads():
    torch.set_num_threads(1)
    receipt = poverty_receipt()
    poor, medium, rich = receipt['rows']
    assert poor['recall_accuracy'] < medium['recall_accuracy'] < rich['recall_accuracy']
    assert rich['recall_accuracy'] == 1.
    assert rich['pressure_fraction'] == 0.
    assert all(row['value_gradient_norm'] > 0 for row in receipt['rows'])


def test_semantic_retrieval_does_not_certify_repeat_gate_closure():
    torch.set_num_threads(1)
    receipt = semantic_receipt()
    assert receipt['status'] == 'failed'
    assert receipt['gate'] == 'STOP before GPU'
    assert all(row['repeat_write_enabled_fraction'] == 1. for row in receipt['rows'])
    assert all(row['repeat_allocation_fraction'] == 1. for row in receipt['rows'])
