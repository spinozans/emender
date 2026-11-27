"""Invented-data oracles for the SAL-1 model at small geometry.

The count skeleton must reproduce a brute per-row depth-scored corpus source
exactly; the order-2 node partials must match the reference MomentTree read;
cores and covers must partition the archive; causality must hold at window
boundaries; and every arm must train its own active parameters.
"""

import math

import numpy as np
import torch

from .ladder import MAX_DEPTH, moment_statistics
from .model import LEAF, SalLM, SalPack
from .settings import ARMS


def fixture(seed=5775, docs=1, width=129, batch=2):
    rng = np.random.default_rng(seed)
    build = rng.integers(0, 5, (docs, width), dtype=np.uint8)
    m = SalLM(seed)
    pack = SalPack(build, m, batch=batch)
    live = rng.integers(0, 5, (batch, 65), dtype=np.uint8)
    return m, pack, build, live


def test_skeleton_source_matches_brute_force():
    m, pack, build, live = fixture()
    batch = pack.bind(live)
    loss = m(batch, "skeleton")
    assert loss.shape == (pack.batch * pack.time,)
    # Brute: corpus mass and value from per-row suffix depths.
    w = torch.exp(m.g).detach()
    e_vec = m.value(m.embed.weight).detach()
    ids = batch["ids"].reshape(pack.batch, pack.time)
    log_b = m.log_pi.detach() - math.log(pack.rows)
    for q in range(batch["m"].shape[0]):
        b, tq = q // pack.time, q % pack.time
        x = ids[b]
        masses = []
        values = []
        for r in range(build.shape[0]):
            for pos in range(1, build.shape[1]):
                cap = min(MAX_DEPTH, pos, tq + 1)
                ell = 0
                for d in range(1, cap + 1):
                    if (
                        np.array_equal(
                            build[r, pos - d : pos],
                            x[max(0, tq + 1 - d) : tq + 1].cpu().numpy(),
                        )
                        and d <= tq + 1
                    ):
                        ell = d
                succ = int(build[r, pos])
                mass = float(w[ell]) * math.exp(log_b)
                masses.append(mass)
                values.append(e_vec[succ] * mass)
        z = sum(masses)
        want_val = sum(values) / z
        want_logm = math.log(z)
        # The model's skeleton partial must equal this exactly.
        got_logm = torch.log(
            torch.exp(m.log_pi.detach() - math.log(pack.rows))
            * (w[None, :] * batch["m"][q]).sum()
        )
        got_val = (
            torch.einsum("d,dv->v", w, batch["h"][q] @ e_vec)
            / (w * batch["m"][q]).sum()
        )
        torch.testing.assert_close(
            got_logm, torch.tensor(want_logm), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(got_val, want_val, rtol=1e-4, atol=1e-5)


def test_order2_node_partials_match_reference_read():
    m, pack, build, live = fixture()
    batch = pack.bind(live)
    k_bank, v_bank = m.bank(pack.archive)
    levels = moment_statistics(k_bank, v_bank, LEAF)
    t = m.query(m.live(batch["ids"], batch["lengths"])) * m.width**-0.5
    logm, val, own = m.node_partials(
        levels,
        t,
        batch["cover_owner"],
        batch["cover_lv"],
        batch["cover_idx"],
        math.log(pack.rows),
    )
    for i in range(0, len(logm), max(1, len(logm) // 12)):
        qi = int(own[i])
        lv = int(batch["cover_lv"][i])
        idx = int(batch["cover_idx"][i])
        st_n, st_kbar, st_cov = (
            levels[lv]["n"][idx],
            levels[lv]["kbar"][idx],
            levels[lv]["S"][idx] / levels[lv]["n"][idx],
        )
        tt = t[qi]
        want = (
            float(m.log_pi.detach())
            - math.log(pack.rows)
            + math.log(float(st_n))
            + float((tt * st_kbar).sum())
            + 0.5 * float(torch.einsum("d,de,e->", tt, st_cov, tt))
        )
        torch.testing.assert_close(logm[i], torch.tensor(want), rtol=1e-4, atol=1e-5)


def test_core_and_cover_partition_the_archive():
    m, pack, build, live = fixture()
    batch = pack.bind(live)
    q = 0
    covered = 0
    for i in range(len(batch["cover_owner"])):
        if int(batch["cover_owner"][i]) == q:
            covered += int(2 ** int(batch["cover_lv"][i])) * LEAF
    core = batch["core"][q]
    core_rows = int((core >= 0).sum()) if bool(batch["core_valid"][q]) else 0
    assert covered + core_rows == pack.rows


def test_future_bytes_never_leak_into_earlier_predictions():
    m, pack, build, live = fixture()
    batch = pack.bind(live)
    out1 = m(batch, "skeleton").detach().clone()
    poisoned = live.copy()
    poisoned[:, 31:] = (poisoned[:, 31:] + 3) % 5
    batch2 = pack.bind(poisoned)
    out2 = m(batch2, "skeleton").detach()
    torch.testing.assert_close(
        out1.reshape(pack.batch, pack.time)[:, :30],
        out2.reshape(pack.batch, pack.time)[:, :30],
        rtol=0,
        atol=0,
    )


def test_all_arms_train_their_active_parameters():
    for arm in ARMS:
        m, pack, build, live = fixture()
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        b = pack.bind(live)
        opt.zero_grad()
        m(b, arm, archive=pack.archive if arm == "order2" else None).mean().backward()
        opt.step()
        moved = [k for k, v in m.named_parameters() if not torch.equal(v, before[k])]
        assert moved, arm
        if arm == "skeleton":
            assert any("g" in k or "log_pi" in k for k in moved)
        assert all(torch.isfinite(v).all() for v in m.parameters())


def test_paired_starts_across_arms():
    m1, _, _, _ = fixture(seed=5751)
    m2, _, _, _ = fixture(seed=5751)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
