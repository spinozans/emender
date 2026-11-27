"""Brute-force oracles for the count skeleton and order-2 moment closure.

Everything here runs on invented bytes with CUDA hidden. These gates must
pass and be published before any new natural membership is opened.
"""

import numpy as np
import torch

from .ladder import (
    CountIndex,
    MomentTree,
    SkeletonSource,
    cover,
    dense,
    mix_sources,
)


def brute_depths(archive, x, t, depth):
    """Suffix-match depth of every eligible row against a live context."""
    n, w = archive.shape
    depths = np.zeros(n * (w - 1), np.int64)
    for r in range(n):
        for pos in range(1, w):
            cap = min(depth, pos, t)
            ell = 0
            for d in range(1, cap + 1):
                if np.array_equal(archive[r, pos - d : pos], x[t - d : t]):
                    ell = d
            depths[r * (w - 1) + pos - 1] = ell
    return depths


def test_bind_stats_matches_reference_stats():
    rng = np.random.default_rng(5773)
    archive = rng.integers(0, 3, (4, 21), dtype=np.uint8)
    x = rng.integers(0, 3, (3, 12), dtype=np.uint8)
    idx = CountIndex(archive, 6)
    c1, h1, m1, e1 = idx.stats(x)
    c2, h2, m2, e2 = idx.bind_stats(x)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_allclose(h1, h2, rtol=0, atol=0)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_allclose(e1, e2, rtol=0, atol=1e-5)
    assert (e2.sum(-1) == m2).all()


def test_skeleton_is_exact_depth_scored_attention():
    rng = np.random.default_rng(5765)
    archive = rng.integers(0, 4, (5, 33), dtype=np.uint8)
    x = rng.integers(0, 4, (1, 33), dtype=np.uint8)
    depth = 6
    idx = CountIndex(archive, depth)
    _, _, m, h = idx.stats(x)
    src = SkeletonSource(idx.rows, depth).double()
    with torch.no_grad():
        src.log_pi.fill_(np.log(0.37))
        src.g.copy_(torch.linspace(-1.5, 2.5, depth + 1, dtype=torch.float64))
    g = src.g.detach().numpy()
    succ = archive[:, 1:].ravel()
    log_b = np.log(0.37) - np.log(idx.rows)
    for t in range(1, 33):
        counts = brute_depths(archive, x[0], t, depth)
        mass = np.exp(g[counts] + log_b)
        want_z = float(np.log(mass.sum()))
        want_p = np.zeros(256)
        np.add.at(want_p, succ, mass)
        want_p /= mass.sum()
        log_z, p = src(
            torch.tensor(m[0, t - 1], dtype=torch.float64).unsqueeze(0),
            torch.tensor(h[0, t - 1], dtype=torch.float64).unsqueeze(0),
        )
        torch.testing.assert_close(
            log_z[0], torch.tensor(want_z, dtype=torch.float64), rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(p[0], torch.tensor(want_p), rtol=1e-9, atol=1e-14)


def test_exact_depth_classes_telescope_to_all_rows():
    rng = np.random.default_rng(5766)
    archive = rng.integers(0, 3, (4, 21), dtype=np.uint8)
    x = rng.integers(0, 3, (3, 21), dtype=np.uint8)
    idx = CountIndex(archive, 8)
    counts, hists, m, h = idx.stats(x)
    assert counts.shape == (3, 20, 9) and h.shape == (3, 20, 9, 256)
    assert (h.sum(-1) == m).all() and (m.sum(-1) == idx.rows).all()
    assert (m >= 0).all() and (h >= 0).all()


def test_order2_beats_order1_under_key_dispersion():
    torch.manual_seed(5767)
    rng = np.random.default_rng(5767)
    rows, leaf, d, dv = 256, 128, 8, 5
    tree = MomentTree(rows, leaf, d, dv).double()
    with torch.no_grad():
        tree.log_pi.fill_(np.log(0.4))
    centres = rng.normal(size=(rows // leaf, d)) * 2
    k_np = np.repeat(centres, leaf, axis=0) + rng.normal(size=(rows, d)) * 0.5
    A = rng.normal(size=(dv, d))
    v_np = (
        np.repeat(rng.normal(size=(rows // leaf, dv)), leaf, axis=0)
        + k_np @ A.T
        + rng.normal(size=(rows, dv)) * 0.02
    )
    k = torch.tensor(k_np, dtype=torch.float64)
    v = torch.tensor(v_np, dtype=torch.float64)
    log_b = torch.full((rows,), np.log(0.4) - np.log(rows), dtype=torch.float64)
    levels = tree.statistics(k, v)
    err1, err2 = [], []
    for _ in range(12):
        q = torch.randn(d, dtype=torch.float64)
        want = dense(q, k, v, log_b)[1]
        _, got2 = tree.read(q, levels, [(0, 0), (0, 1)])
        va = tree._node(levels, (0, 0))[2]
        vb = tree._node(levels, (0, 1))[2]
        na = tree._node(levels, (0, 0))[0]
        nb = tree._node(levels, (0, 1))[0]
        mass = torch.stack([na, nb]) * torch.exp(tree.log_pi) / rows
        got1 = (mass[:, None] * torch.stack([va, vb])).sum(0) / mass.sum()
        err1.append((got1 - want).norm().item())
        err2.append((got2 - want).norm().item())
    assert np.mean(err2) < np.mean(err1)


def test_zero_dispersion_order2_matches_dense_and_gradients():
    torch.manual_seed(5768)
    rows, leaf, d, dv = 128, 64, 6, 4
    tree = MomentTree(rows, leaf, d, dv).double()
    with torch.no_grad():
        tree.log_pi.fill_(np.log(0.25))
    k0 = torch.randn(d, dtype=torch.float64)
    v0 = torch.randn(dv, dtype=torch.float64)
    k = k0.repeat(rows, 1).requires_grad_(True)
    v = v0.repeat(rows, 1).requires_grad_(True)
    q = torch.randn(d, dtype=torch.float64)
    levels = tree.statistics(k, v)
    got = tree.read(q, levels, [(1, 0)])
    want = dense(
        q, k, v, torch.full((rows,), np.log(0.25) - np.log(rows), dtype=torch.float64)
    )
    torch.testing.assert_close(got[0], want[0], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(got[1], want[1], rtol=1e-9, atol=1e-12)
    up = torch.randn(dv, dtype=torch.float64)
    (got[1] @ up - got[0]).backward()
    gk, gv = k.grad.clone(), v.grad.clone()
    k.grad = v.grad = None
    (want[1] @ up - want[0]).backward()
    torch.testing.assert_close(gk, k.grad, rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(gv, v.grad, rtol=1e-8, atol=1e-12)


def test_radius_certificate_holds_and_merges_conservatively():
    torch.manual_seed(5769)
    rows, leaf, d, dv = 512, 128, 8, 4
    tree = MomentTree(rows, leaf, d, dv).double()
    with torch.no_grad():
        tree.log_pi.fill_(np.log(0.4))
    k = (
        torch.randn(rows, d, dtype=torch.float64)
        * torch.linspace(0.1, 1.5, rows, dtype=torch.float64)[:, None]
    )
    v = torch.randn(rows, dv, dtype=torch.float64)
    levels = tree.statistics(k, v)
    for lv, i in [(0, 0), (0, 1), (1, 0), (2, 0)]:
        sel = torch.arange(i * tree.leaf * 2**lv, (i + 1) * tree.leaf * 2**lv)
        q = torch.randn(d, dtype=torch.float64)
        t = q / d**0.5
        z = (torch.exp(k[sel] @ t).sum().log() + np.log(0.4) - np.log(rows)).item()
        lo, hi = tree.bounds(q, levels, (lv, i))
        assert lo.item() <= z <= hi.item()
        if lv == 0:
            direct = (k[sel] - k[sel].mean(0)).norm(dim=1).max()
            torch.testing.assert_close(levels[0]["R"][i], direct)


def test_moment_merges_match_direct_statistics():
    torch.manual_seed(5770)
    rows, leaf, d, dv = 512, 128, 8, 4
    tree = MomentTree(rows, leaf, d, dv).double()
    k = torch.randn(rows, d, dtype=torch.float64)
    v = torch.randn(rows, dv, dtype=torch.float64)
    levels = tree.statistics(k, v)
    for lv in range(3):
        sel = torch.arange(0, tree.leaf * 2**lv)
        got = tree._node(levels, (lv, 0))
        kc, vc = k[sel], v[sel]
        dev = kc - kc.mean(0, keepdim=True)
        devv = vc - vc.mean(0, keepdim=True)
        torch.testing.assert_close(
            got[0], torch.tensor(float(len(sel)), dtype=torch.float64)
        )
        torch.testing.assert_close(got[1], kc.mean(0), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(got[2], vc.mean(0), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            got[3], dev.T @ dev / len(sel), rtol=1e-9, atol=1e-11
        )
        torch.testing.assert_close(
            got[4], devv.T @ dev / len(sel), rtol=1e-9, atol=1e-11
        )


def test_complement_covers_and_joint_source_mix():
    leaves = 16
    lo, hi = 4, 10
    total = 0
    for seg_lo, seg_hi in ((0, lo), (hi, leaves)):
        pos = seg_lo
        for lv, i in cover(seg_lo, seg_hi, leaves):
            size = 2**lv
            assert i * size == pos
            pos += size
            total += size
        assert pos == seg_hi
    assert total + (hi - lo) == leaves
    parts = [
        (
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        ),
        (
            torch.tensor(-0.5, dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ),
    ]
    log_z, read = mix_sources(parts)
    z = np.exp(-1.0) + np.exp(-0.5)
    np.testing.assert_allclose(log_z.item(), np.log(z), rtol=1e-12)
    np.testing.assert_allclose(
        read.numpy(), [np.exp(-1.0) / z, np.exp(-0.5) / z], rtol=1e-12
    )


def test_skeleton_gradients_flow_to_prior_and_depth_weights():
    rng = np.random.default_rng(5772)
    archive = rng.integers(0, 4, (3, 17), dtype=np.uint8)
    idx = CountIndex(archive, 4)
    x = rng.integers(0, 4, (1, 17), dtype=np.uint8)
    _, _, m, h = idx.stats(x)
    src = SkeletonSource(idx.rows, 4).double()
    t = 9
    log_z, p = src(
        torch.tensor(m[0, t - 1], dtype=torch.float64).unsqueeze(0),
        torch.tensor(h[0, t - 1], dtype=torch.float64).unsqueeze(0),
    )
    target = int(x[0, t])
    assert p[0, target] > 0
    # The prior cancels inside p = U/Z; it only reaches training through the
    # source mass, so the loss must include log Z as the joint softmax does.
    (log_z[0] - p[0, target]).backward()
    assert src.g.grad is not None and src.g.grad.abs().sum() > 0
    assert src.log_pi.grad is not None and src.log_pi.grad.abs() > 0
    assert torch.isfinite(src.g.grad).all()
