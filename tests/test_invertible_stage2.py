"""Source-named CPU tests for released-pair Stage-2 protocol plumbing."""
import copy

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.run_invertible_stage2 import (
    FLOOR, finish_metrics, frozen_pools, metric_counts, plateau,
    remove_state_tanh, selected,
)


def test_plateau_requires_eight_complete_intervals():
    history = [{"step": 250 * i, "best_accuracy": 0.2} for i in range(9)]
    assert not plateau(history[:8])
    assert plateau(history)
    history[-1]["best_accuracy"] = 0.201
    assert not plateau(history)  # Exactly 0.1pp is an improvement.
    history[-1]["best_accuracy"] = 0.20099
    assert plateau(history)
    history[-1]["step"] += 1
    with pytest.raises(ValueError):
        plateau(history)


def test_best_validation_selection_has_symmetric_tie_breaks():
    best = {"accuracy": 0.2, "loss": 3., "step": 250}
    assert selected(best, None)
    assert selected({"accuracy": 0.21, "loss": 4., "step": 500}, best)
    assert selected({"accuracy": 0.2, "loss": 2., "step": 500}, best)
    assert not selected({**best, "step": 500}, best)
    assert not selected({"accuracy": 0.19, "loss": 1., "step": 500}, best)


def test_full_vocab_metrics_reject_invalid_ids_and_mask_prefix():
    target = np.array([[0, 0, 62, 63, 64]])
    pred = np.array([[0, 0, 62, 124, 3721 + 3]])
    mask = np.array([[False, False, True, True, True]])
    counts = metric_counts(pred, target, mask)
    assert counts["count"] == 3
    assert counts["correct"] == 1
    assert counts["x_correct"] == 1
    assert counts["y_correct"] == 2
    assert counts["out_of_range"] == 1
    out = finish_metrics(counts, 6.)
    assert out["loss"] == 2.
    assert out["accuracy"] == 1 / 3
    assert out["suffix_floor"] == FLOOR
    assert out["post_suffix7_accuracy"] is None


def test_suffix7_burn_in_starts_at_ninth_driver():
    y = np.zeros((2, 130), dtype=np.int64)
    m = np.ones_like(y, dtype=bool)
    m[:, :2] = False
    c = metric_counts(y.copy(), y, m)
    assert c["post_suffix7_count"] == 2 * 120


def test_frozen_one_step_pools_are_reproducible_and_disjoint():
    a = frozen_pools(0, 1)
    b = frozen_pools(0, 1)
    seen = set()
    for name in a:
        assert np.array_equal(a[name][0], b[name][0])
        rows = {tuple(row) for row in a[name][0]}
        assert not rows & seen
        assert len(rows) == len(a[name][0])
        seen.update(rows)
        assert not a[name][2][:, :2].any()
        assert a[name][2][:, 2:].all()
    assert len(a["train"][0]) == 128
    assert len(a["val"][0]) == len(a["test_1x"][0]) == 512
    assert not np.array_equal(a["train"][0], frozen_pools(1, 1)["train"][0])


def test_identity_ablation_changes_only_activation_attributes():
    from ndm.models import E97SplitEditLayer
    layer = E97SplitEditLayer(dim=16, n_heads=2, n_state=4, use_triton=False)
    before = copy.deepcopy(layer.__dict__)
    weights = {k: v.clone() for k, v in layer.state_dict().items()}
    assert remove_state_tanh(layer) == 1
    for key in before:
        if key in ("linear_state", "state_activation", "_parameters", "_modules", "_buffers"):
            continue
        assert getattr(layer, key) == before[key]
    assert layer.linear_state and layer.state_activation == "identity"
    assert weights.keys() == layer.state_dict().keys()
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        remove_state_tanh(layer)
