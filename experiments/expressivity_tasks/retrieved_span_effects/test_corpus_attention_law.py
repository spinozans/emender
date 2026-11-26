"""Executable examples for CORPUS_ATTENTION.md; invented rows, no corpus or device."""

import numpy as np
import pytest


def test_joint_softmax_keeps_source_mass_not_just_the_best_four():
    # 100 corpus occurrences: 70 predict token 0 and 30 predict token 1.
    # One local occurrence predicts token 2. Its logit is ln(100).
    corpus = np.eye(3)[[0] * 70 + [1] * 30]
    local = np.array([0.0, 0.0, 1.0])
    scores = np.r_[np.zeros(100), np.log(100.0)]
    w = np.exp(scores - scores.max())
    w /= w.sum()
    exhaustive = w @ np.vstack((corpus, local))
    means = np.stack((corpus.mean(0), local))
    z = np.array([len(corpus), 100.0])
    np.testing.assert_allclose((z / z.sum()) @ means, exhaustive)
    np.testing.assert_allclose(exhaustive, [0.35, 0.15, 0.50])
    assert 4 / (100 + 4) < 0.04
    keep = np.r_[np.arange(4), 100]
    truncated = np.zeros(101)
    truncated[keep] = w[keep] / w[keep].sum()
    np.testing.assert_allclose(0.5 * np.abs(w - truncated).sum(), 1 - w[keep].sum())


def test_suffix_interval_sums_equal_exhaustive_occurrence_weights():
    docs = [[1, 4, 2, 5, 3], [1, 4, 2, 6], [2, 7], [9, 9, 8]]
    prefix = [0, 1, 4, 2]
    rows = [(d[: j + 1], d[j + 1]) for d in docs for j in range(len(d) - 1)]
    b = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    ids = np.array([y for _, y in rows])
    v = np.eye(10)[ids]
    z = b[0] * len(rows)
    num = b[0] * v.sum(0)
    for length in range(1, len(prefix) + 1):
        keep = [i for i, (k, _) in enumerate(rows) if k[-length:] == prefix[-length:]]
        z += b[length] * len(keep)
        num += b[length] * v[keep].sum(0)
    weights = []
    for k, _ in rows:
        longest = 0
        for a, c in zip(reversed(k), reversed(prefix)):
            if a != c:
                break
            longest += 1
        weights.append(b[: longest + 1].sum())
    w = np.asarray(weights)
    np.testing.assert_allclose(z, w.sum())
    np.testing.assert_allclose(num / z, w @ v / w.sum())
    assert num[8] > 0  # no lexical match, still has background mass


@pytest.mark.parametrize("refine", ["none", "some", "all"])
def test_background_plus_exact_corrections_preserves_the_entire_tail(refine):
    rng = np.random.default_rng(5604)
    q = rng.normal(size=(5, 8))
    k = rng.normal(size=(31, 8))
    v = rng.normal(size=(31, 7))
    base = rng.uniform(0.2, 1.6, len(k))
    truth = np.exp(q @ k.T / np.sqrt(k.shape[1])) * base
    # Positive features for an identity check, NOT fitted exponential-kernel features.
    phi = rng.uniform(0.1, 0.8, (len(q), 3))
    psi = rng.uniform(0.1, 0.8, (len(k), 3))
    weighted = psi * base[:, None]
    mass = phi @ weighted.sum(0)
    moment = phi @ (weighted.T @ v)
    h = np.unique(
        {"none": [], "some": [1, 3, 5, 1, 7], "all": np.arange(len(k))}[refine]
    ).astype(int)
    delta = truth[:, h] - phi @ weighted[h].T
    mass += delta.sum(1)
    moment += delta @ v[h]
    w = phi @ weighted.T
    w[:, h] = truth[:, h]
    np.testing.assert_allclose(mass, w.sum(1))
    np.testing.assert_allclose(
        moment / mass[:, None], (w @ v) / w.sum(1, keepdims=True)
    )
    if refine == "all":
        np.testing.assert_allclose(
            moment / mass[:, None], (truth @ v) / truth.sum(1, keepdims=True)
        )
