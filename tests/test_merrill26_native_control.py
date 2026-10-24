"""X26b source-named checks for the native IMM control and launch contract."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_merrill26_native_control as native
from experiments.expressivity_tasks.run_merrill26_replication import auxiliary_targets, CurriculumSampler
from experiments.expressivity_tasks.merrill26_replication_models import upstream


def item(mats):
    module = native.source('rnn')
    mats = np.asarray(mats, dtype=np.int64).reshape(-1, 9)
    p = np.eye(3, dtype=object)
    for m in mats.reshape(-1, 3, 3):
        p = p @ m
    return {'T': len(mats), 'mats_pm1': torch.from_numpy(mats).short(),
            'state_prev_phi': torch.from_numpy(module.signed_log1p_np(module.compute_states_prev_nomod(mats))),
            'y': int(p[0, 0] == 0)}


def test_native_collate_bos_matrix_vectors_no_state_and_zero_labels():
    module = native.source('rnn')
    mats = [np.eye(3, dtype=np.int64), np.zeros((3, 3), dtype=np.int64), -np.eye(3, dtype=np.int64)]
    row = item(mats)
    batch = module.collate_batch([row, item(mats[:1])], teacher_force_state=False)
    assert batch.lengths.tolist() == [4, 2]
    assert batch.tok_type[0].tolist() == [module.TOK_BOS] + [module.TOK_MAT] * 3
    assert not (batch.tok_type == module.TOK_STATE).any()
    torch.testing.assert_close(batch.tok_val_phi[0, 1:], torch.tensor(np.asarray(mats).reshape(3, 9)).float())
    assert batch.y.tolist() == [1, 0]
    assert batch.aux_mask[0].tolist() == [False, True, True, False]
    assert batch.aux_label[0, 1:3].tolist() == [0., 1.]
    assert auxiliary_targets((np.asarray(mats).reshape(-1) + 1).astype(np.uint8)) == [1, 0]
    # Prefix states must never enter the default token inputs.
    row['state_prev_phi'].add_(123)
    changed = module.collate_batch([row], teacher_force_state=False)
    torch.testing.assert_close(changed.tok_val_phi, batch.tok_val_phi[:1])


@pytest.mark.parametrize('arm', ['rnn', 'delta'])
def test_native_models_keep_the_exact_upstream_encoder_and_dimensions(arm):
    model = native.build_model(arm)
    module = native.source(arm)
    assert isinstance(model.enc, module.TokenEncoder)
    assert model.enc.val_proj.in_features == 9
    assert model.enc.val_proj.out_features == 256
    assert model.enc.max_len == 301
    if arm == 'rnn':
        assert model.rnn.num_layers == 2 and model.rnn.hidden_size == 256
        assert model.rnn.dropout == .3
    else:
        assert len(model.blocks) == 1 and model.blocks[0].num_heads == 4
        assert not model.blocks[0].allow_neg_eigval


def test_native_auxiliary_loss_is_source_function_not_scalar_adapter():
    a, b = native.source('rnn'), native.source('delta')
    batch = a.collate_batch([item([np.eye(3)] * 3), item([np.eye(3)])])
    logits = torch.randn(2, 4, requires_grad=True)
    final = a.loss_final_bce(logits, batch.y, batch.y_mask)
    aux = a.loss_aux_bce(logits, batch.aux_label, batch.aux_mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[batch.y_mask], batch.y.float())
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(aux, b.loss_aux_bce(logits, batch.aux_label, batch.aux_mask))
    assert 'module.loss_final_bce' in inspect.getsource(native.forward)
    assert 'module.loss_aux_bce' in inspect.getsource(native.forward)


def test_native_control_fixed_cards_and_reconciled_protocol():
    assert native.CARDS == {'rnn_integer_seed0': 4, 'delta_integer_seed0': 5}
    assert native.BATCH == 128 and native.LR == native.WEIGHT_DECAY == 1e-4
    assert native.base.CEILING == 60000 and native.base.EVAL_EVERY == 500
    class Pool:
        lengths = np.tile(np.arange(1, 101), 7)
    sampler = CurriculumSampler(Pool(), np.random.default_rng(1), True)
    for _ in range(30):
        indices = sampler.draw(128)
        assert len(indices) == 128
        assert max(Pool.lengths[indices]) <= sampler.max_t


def test_native_rnn_and_delta_collate_identical():
    rows = [item([np.eye(3), -np.eye(3)]), item([np.zeros((3, 3))])]
    a = native.source('rnn').collate_batch(rows)
    b = native.source('delta').collate_batch(rows)
    for field in native.fields(a):
        torch.testing.assert_close(getattr(a, field.name), getattr(b, field.name))
