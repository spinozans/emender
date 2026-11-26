"""Invented receipt math; does not open the running study."""

import numpy as np
import pytest

from .receipt_audit import time_contrast, primary, packed_result


def test_equal_clock_paired_denominator_and_seed_signs():
    # This is intentionally NOT a mean over the documents' mean losses.
    n = np.array([1, 9])
    base = np.tile([2.0, 18.0], (3, 1))
    diff = np.array([[1.0, 0.0], [0.5, 0.45], [-0.4, 0.27]])
    mult = np.array([[2, 0], [0, 2], [1, 1]], float)
    c = time_contrast(base - diff, base, n, mult)
    means = diff.mean(0)
    raw = [means[0], means[1] / 9, means.sum() / 10]
    lo, hi = np.quantile(raw, [0.025, 0.975])
    assert (
        np.allclose(c["ci"], [lo, hi])
        and np.isclose(c["gain"], diff.sum() / 30)
        and np.allclose(c["seed_gains"], diff.sum(1) / 10)
    )
    assert lo > 0 and not c["positive"] and not packed_result(c)[-1]
    with pytest.raises(RuntimeError):
        time_contrast(base[:1], base, n, mult)


def test_primary_is_conjunction_not_a_best_arm_rule():
    a = {
        f"gdn2/g-effect/full.intact:{x}": {"all": {"positive": True}}
        for x in (
            "g-state/full.intact",
            "g-neutral/full.intact",
            "g-effect/full.wrong_effect",
        )
    }
    assert primary(a)["established"]
    for k in a:
        wrong = dict(a)
        wrong[k] = {"all": {"positive": False}}
        assert not primary(wrong)["established"]
