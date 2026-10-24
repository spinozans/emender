"""X26c keeps the shipped loop immutable and reports imbalance-proof metrics externally."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.expressivity_tasks import run_merrill26_raw_defaults as raw
from experiments.expressivity_tasks.merrill26_replication_models import SOURCES
from experiments.expressivity_tasks.run_merrill26_separators import digest


def test_prior_tracking_cannot_pass_balanced_accuracy():
    result = raw.confusion_metrics([[0, 10], [0, 90]])
    assert result['accuracy'] == result['majority_accuracy'] == .9
    assert result['balanced_accuracy'] == .5
    assert result['class_recall'] == [0., 1.]


def test_missing_class_never_fabricates_balanced_accuracy():
    result = raw.confusion_metrics([[0, 0], [0, 90]])
    assert result['balanced_accuracy'] is None
    assert result['class_recall'] == [None, 1.]


def test_metrics_weight_examples_not_partial_batches():
    result = raw.confusion_metrics([[8, 2], [30, 60]])
    assert result['count'] == 100
    assert result['accuracy'] == .68
    assert result['balanced_accuracy'] == pytest.approx((.8 + 2/3) / 2)


@pytest.mark.parametrize('arm', ['rnn', 'delta'])
def test_checkpoint_model_strict_round_trip_without_optimizer_updates(arm, monkeypatch):
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    model = raw.native.build_model(arm)
    args = {'teacher_force_state': False, 'batch_size': 256, 'max_steps': 30000,
        'max_len': 301, 'd_model': 256, 'hidden': 256, 'layers': 2 if arm == 'rnn' else 1,
        'dropout': .3 if arm == 'rnn' else .2, 'rnn_type': 'gru', 'heads': 4,
        'deltanet_mode': 'chunk', 'allow_neg_eigval': False, 'chunk_size': 64}
    restored = raw.checkpoint_model(arm, {'args': args, 'model': model.state_dict()})
    assert not restored.training
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    args['teacher_force_state'] = True
    with pytest.raises(ValueError, match='not registered'):
        raw.checkpoint_model(arm, {'args': args, 'model': model.state_dict()})


def test_vendored_training_pipeline_remains_byte_identical():
    manifest = json.loads((SOURCES.parent / 'manifest.json').read_text())
    assert manifest['commit'] == 'dbfa365ad680f425dfb79a9e78b5e0d7915757bb'
    for path, sha in manifest['selected_files_sha256'].items():
        assert digest(SOURCES / path) == sha


def test_launcher_only_overrides_environment_and_output_paths():
    script = Path('scripts/launch_merrill26_raw_defaults.sh').read_text()
    assert 'env -i HOME=' in script
    for name in ('BATCH_SIZE=', 'MAX_STEPS=', 'LR=', 'WD=', 'PATIENCE=', 'INSTALL_FLA='):
        assert name not in script
    assert 'bash "$source_dir/$arm.sh"' in script
    assert 'wait "$train_pid"' in script
    assert script.index('wait "$train_pid"; rc=') < script.index('--arm "$arm"')


def test_evaluate_split_counts_final_token_only_and_per_length(monkeypatch):
    @dataclass
    class Batch:
        tok_type: torch.Tensor
        tok_val_phi: torch.Tensor
        lengths: torch.Tensor
        y: torch.Tensor
        y_mask: torch.Tensor

    batch = Batch(torch.zeros(3, 4), torch.zeros(3, 4, 9), torch.tensor([2, 3, 4]),
                  torch.tensor([0, 1, 1]), torch.tensor([[False, True, False, False],
                    [False, False, True, False], [False, False, False, True]]))
    module = SimpleNamespace(PreloadedNoModBinaryDataset=lambda *a, **kw: object(),
        make_loader=lambda *a, **kw: [batch], pick_autocast_dtype=lambda *a: torch.float32)
    monkeypatch.setattr(raw.native, 'source', lambda arm: module)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    logits = torch.tensor([[99., -1., 99., 99.], [99., 99., 1., 99.], [99., 99., 99., -1.]])
    result = raw.evaluate_split(lambda *a: logits, 'rnn', 'test_bin0',
        {'batch_size': 256, 'bucket_size': 4096, 'seed': 0, 'amp': False, 'amp_dtype': 'bf16'})
    assert result['count'] == 3
    assert result['confusion_true_by_pred'] == [[1, 0], [1, 1]]
    assert result['class_recall'] == [1., .5]
    assert result['balanced_accuracy'] == .75
    assert set(result['by_matrix_length']) == {'1', '2', '3'}
    assert sum(v['count'] for v in result['by_matrix_length'].values()) == 3
