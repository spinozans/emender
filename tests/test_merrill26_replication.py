"""X26 upstream identity, scalar adapter, auxiliary output and launch gates."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.expressivity_tasks import run_merrill26_replication as run
from experiments.expressivity_tasks.merrill26_replication_models import build_model, upstream
from experiments.expressivity_tasks.tasks.merrill26_separators import Dataset, matrix_product
from scripts import watch_merrill26_replication as watcher


def test_reconciled_protocol_and_authorized_rungs():
    assert (run.BATCH, run.LR, run.WEIGHT_DECAY) == (128, 1e-4, 1e-4)
    assert (run.base.CEILING, run.base.EVAL_EVERY) == (60000, 500)
    assert run.CARDS == {'rnn_graph_seed0': 2, 'delta_graph_seed0': 3,
                         'rnn_integer_seed0': 4, 'delta_integer_seed0': 5}
    with pytest.raises(ValueError, match='card binding'):
        watcher.spawn('rnn_graph_seed0', 7)
    for pair in [('e97', 'graph'), ('rnn', 'mod97')]:
        with pytest.raises(ValueError):
            build_model(*pair)


def test_graph_rnn_is_byte_snapshot_architecture_not_stock_hybrid():
    torch.manual_seed(17)
    model = build_model('rnn', 'graph')
    torch.manual_seed(17)
    expected = upstream('dgc', 'rnn').StableFastReLURNNClassifier(8, 0, 128, 256, 1, .1,
                                                              act_clip=6., hh_shrink=.3)
    assert all(torch.equal(v, expected.state_dict()[k]) for k, v in model.state_dict().items())
    assert model.rnn.layers == 1 and model.emb.embedding_dim == 128
    model.eval()
    x = torch.tensor([[7, 2, 3, 4, 1, 5, 1, 6], [7, 2, 5, 1, 6, 0, 0, 0]])
    lengths = torch.tensor([8, 5])
    assert torch.allclose(model(x, lengths)[1], model(x[1:2, :5], lengths[1:])[0], atol=1e-6)
    loss, _, target = run.forward_rows(model, [(x[0].numpy(), 1)], 'graph')
    logits = model(x[:1], lengths[:1])
    assert torch.allclose(loss, F.cross_entropy(logits, target, reduction='sum'), atol=1e-6)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_integer_gru_preserves_exact_scalar_tokens_and_causal_prefixes():
    model = build_model('rnn', 'integer').eval()
    assert model.core.rnn.num_layers == 2 and model.core.rnn.hidden_size == 256
    assert model.core.enc.embedding.num_embeddings == 4
    assert model.core.rnn.dropout == .3
    x = torch.tensor([[0, 1, 2] * 6, [2, 0, 1] * 6])
    lengths = torch.tensor([18, 9])
    with torch.no_grad():
        logits = model(x, lengths)
        alone = model(x[1:2, :9], lengths[1:])
    assert logits.shape == (2, 18)
    assert torch.allclose(logits[1, :9], alone[0], atol=1e-6)
    changed = x.clone(); changed[0, 9:] = 2
    assert torch.allclose(model(changed, lengths)[0, :9], logits[0, :9], atol=1e-6)


def test_deltanet_is_real_delta_rule_and_repo_task_specific_configuration():
    graph, integer = build_model('delta', 'graph'), build_model('delta', 'integer')
    g, i = graph.backbone.layers[0], integer.core.blocks[0]
    assert len(graph.backbone.layers) == len(integer.core.blocks) == 1
    assert type(g).__name__ == type(i).__name__ == 'DeltaNet'
    for layer in (g, i):
        assert layer.hidden_size == 256 and layer.num_heads == 4
        assert layer.use_beta and not layer.allow_neg_eigval
    assert g.use_gate and not g.use_short_conv
    assert not i.use_gate and i.use_short_conv


def test_auxiliary_labels_are_exact_positivity_not_zero_or_int64_overflow():
    a = np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
    assert run.auxiliary_targets((a + 1).reshape(-1)) == [0, 1]
    long = np.ones((100, 3, 3), dtype=np.int64)
    assert matrix_product(long)[0, 0] > np.iinfo(np.int64).max
    assert run.auxiliary_targets((long + 1).reshape(-1)) == [1] * 99


def test_auxiliary_loss_normalization_is_independent_of_microbatch_partition():
    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bias = torch.nn.Parameter(torch.tensor(.4))
        def forward(self, x, lengths):
            return self.bias.expand(x.shape)
    model = Fixed()
    rows = [(np.full(18, 2), 1), (np.full(27, 2), 0)]
    full = run.forward_rows(model, rows, 'integer', True, 3)[0]
    parts = sum(run.forward_rows(model, [row], 'integer', True, 3)[0] for row in rows)
    assert torch.allclose(full, parts)
    no_aux = run.forward_rows(model, rows, 'integer', False)[0]
    expected = no_aux + .3 * run.BATCH * F.binary_cross_entropy_with_logits(
        torch.full((3,), .4), torch.ones(3))
    assert torch.allclose(full, expected)


def test_curriculum_uses_eligible_pool_passes_and_keeps_effective_batch():
    dataset = SimpleNamespace(lengths=np.arange(1, 101))
    sampler = run.CurriculumSampler(dataset, np.random.default_rng(0), True)
    first = sampler.draw(5)
    assert set(first) == set(range(5)) and sampler.epoch == 0
    assert len(sampler.draw(128)) == 128
    assert sampler.epoch > 0 and sampler.max_t < 100
    sampler.epoch, sampler.order, sampler.position = 149, np.array([]), 0
    assert len(sampler.draw(128)) == 128 and sampler.max_t == 100
    ordinary = run.CurriculumSampler(dataset, np.random.default_rng(0), False)
    assert set(ordinary.draw(100)) == set(range(100))


def test_actual_shared_pool_link_and_disk_guard(tmp_path, monkeypatch):
    root, comparison = tmp_path / 'x26', tmp_path / 'x23e'
    root.mkdir(); (comparison / 'datasets').mkdir(parents=True)
    monkeypatch.setattr(run, 'ROOT', root)
    monkeypatch.setattr(run, 'COMPARISON_ROOT', comparison)
    run.link_datasets()
    assert (root / 'datasets').resolve() == (comparison / 'datasets').resolve()
    (root / 'datasets').unlink(); (root / 'datasets').mkdir()
    with pytest.raises(ValueError, match='actual X23E'):
        run.link_datasets()
    monkeypatch.setattr(watcher, 'ROOT', root)
    monkeypatch.setattr(watcher, 'configure_watcher', lambda: None)
    monkeypatch.setattr(watcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(watcher.protocol, 'launch', lambda: pytest.fail('launch mutation before guard'))
    with pytest.raises(OSError, match='disk guard'):
        watcher.watch(initial=True)


def test_terminal_hash_and_marker_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher.protocol, 'ROOT', tmp_path)
    name = 'rnn_graph_seed0'
    watcher.protocol.terminal(name, 143)
    assert watcher.protocol.rung_status(name) == 'censored'
    (tmp_path / name / 'result.json').write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        watcher.protocol.rung_status(name)
