"""PT-only remainder census, paired statistics and frozen reader tests. All membership and bytes are invented."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import sha, write_json
from experiments.expressivity_tasks.x49_live_trace_memory import (
    correction_transfer_data as data,
    correction_transfer_panel as p,
    correction_transfer_run as run,
    correction_transfer_stats as stats,
    correction_transfer_analysis as close,
    correction_transfer_check as checks,
    test_correction_transfer_score as base,
    literal_mark as ls,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, digest
from experiments.expressivity_tasks.x49_live_trace_memory.test_transfer_data import (
    record,
)


def install_pt_parent(mp, d, parent, files):
    # For the shared full short-lifecycle fixture, including a real (invented) FIT seal.
    earlier = {}
    for a, b in base.BETA.items():
        if a in p.reader.panel.FIXED:
            k = p.reader.panel.FIXED[a]
            earlier[k] = {s: [v + int(s) * 1e-6 for v in b] for s in files}
    meta = dict(fit_identity_sha="fixture-fit-only")
    parent.update(
        **deepcopy(earlier),
        data=meta,
        ref="PC-fixture",
        fit_dimensions=dict(p.DIMS),
        swap_fit="PAIR",
    )
    mp.setattr(p.reader, "parent", lambda: parent)

    def fixed(s):
        return files[str(s)], {
            a: earlier[k][str(s)] for a, k in p.reader.panel.FIXED.items()
        }

    mp.setattr(p.reader.panel, "fixed", fixed)
    write_json(
        d / "frozen.json",
        dict(
            spec=data.pc.SPEC,
            ref=parent["ref"],
            endpoints={k: dict(sha256=v) for k, v in files.items()},
            data=meta,
            fit_dimensions=p.DIMS,
            swap_fit="PAIR",
            **earlier,
        ),
    )
    parent["freeze_sha"] = sha(d / "frozen.json")
    return data.pc.SPEC, dict(fit_identity_sha=meta["fit_identity_sha"], fit_units=768)


def test_whole_remainder_no_truncation_top_up_or_separated_prior_block():
    raw = [record(i, 5000 + 6000 * i, 2 + i * 3) for i in range(23)]
    raw.extend(
        [
            dict(raw[11], byte_offset=144000, document=144000),
            record(41, 125, 12),
            record(42, 155000, 40, "final_test"),
            record(43, 156000, 4097),
            record(44, 165000, 123, "train"),
        ]
    )
    size = 180000
    lo = 4000
    ex = {raw[2]["sha256"], raw[-1]["sha256"]}
    clean = sorted(
        [r for i, r in enumerate(raw[:23]) if i != 2],
        key=lambda r: digest(b"x49-jt-short-cal-v1:" + r["sha256"].encode()),
    )
    a = data.jt.choose(raw, ex, size, 132, lo)[0]
    ex |= {r["sha256"] for r in a}
    b = data.jt.choose(raw, ex, size, 200, lo)[0]
    ex |= {r["sha256"] for r in b}
    rest = clean[len(a + b) :]
    n = len(rest)
    nt = sum(r["byte_length"] - 1 for r in rest)
    result = data.remainder(raw[::-1], ex, a, b, size, n, nt, lo)
    assert [r["offset"] for r in result] == [r["byte_offset"] for r in rest] and all(
        r["start"] == 0 for r in result
    )
    for nn, dose in [
        (n - 1, nt - rest[-1]["byte_length"] + 1),
        (n + 1, nt),
        (n, nt + 1),
    ]:
        with pytest.raises(ValueError):
            data.remainder(raw, ex, a, b, size, nn, dose, lo)
    with pytest.raises(ValueError, match="contiguous"):
        data.remainder(raw, ex, b, a, size, n, nt, lo)
    with pytest.raises(ValueError, match="missing spent"):
        data.remainder(raw, ex - {b[0]["sha256"]}, a, b, size, n, nt, lo)


def test_both_pc_roles_and_actual_publication_before_even_metadata(
    tmp_path, monkeypatch
):
    def blocked():
        raise ValueError("no PT public pass")

    monkeypatch.setattr(run, "checked_fixture_ref", blocked)
    monkeypatch.setattr(data, "BASE", tmp_path / "PT")
    for call in (data.selection, data.reserve):
        with pytest.raises(ValueError, match="no PT public"):
            call()
    assert not data.BASE.exists()
    rows = [
        dict(
            doc=8192 * i,
            offset=8192 * i,
            length=4097,
            start=0,
            sha256=f"{i * 100 + 2:064x}",
            split="train",
        )
        for i in range(4)
    ]
    rows.sort(key=lambda r: digest(b"x49-live-v1:" + r["sha256"].encode()))
    book = dict(
        spec=data.pc.SPEC,
        schema="x49-pc-native-train",
        source=dict(size=40000),
        roles=dict(fit=rows[:2], probe=rows[2:]),
    )
    f = tmp_path / "reservation.json"
    write_json(f, book)
    m = dict(rows_sha=sha(f))
    pre = {"already-spent"}
    monkeypatch.setattr(data.pc, "BASE", tmp_path)
    monkeypatch.setattr(data.pc, "N", dict(fit=2, probe=2))
    monkeypatch.setattr(data.pc, "MIN_OFFSET", 0)
    monkeypatch.setattr(data.pc, "metadata", lambda: m)
    monkeypatch.setattr(data.pc, "selection_context", lambda: (book, [], pre))
    assert data.spent_context(m) == (book, [], pre | {r["sha256"] for r in rows})
    pre.add(rows[-1]["sha256"])
    with pytest.raises(ValueError, match="overlapping spent PC"):
        data.spent_context(m)
    with pytest.raises(ValueError, match="changed spent PC"):
        data.spent_context(dict(rows_sha="wrong"))


def test_weighted_pair_prefix_inference_actual_path_costs_and_independent_gates():
    torch.manual_seed(4918)
    host = base.orig.QueryHost().eval()
    head = base.orig.trained_base().eval()
    words = [b"\0\xff", base.orig.word() + b"\x00", base.planted()[0] + b"\xff"]
    hs = list(map(digest, words))
    sz = [len(w) - 1 for w in words]
    logs = {
        s: [
            dict(
                document_sha=h,
                **base.pt.document(
                    host, head, w, base.BETA, s, i, ls.text_tape, width=8
                ),
            )
            for i, (h, w) in enumerate(zip(hs, words, strict=True))
        ]
        for s in SEEDS
    }
    z = stats.summarize(logs, hs, sz)
    draw = np.random.default_rng(4918).integers(0, len(words), size=(10000, len(words)))
    d = []
    for s in SEEDS:
        k = str(s)
        close.audit(logs[s], z, k)
        assert (
            z["documents_with_support_by_seed"][k]["prefix_different_law"] == 1
            and z["conditional_tv_sum_by_seed"][k] > 0
        )
        diff = np.array(
            [r["nll_sums"]["PREFIX"] - r["nll_sums"]["PAIR"] for r in logs[s]]
        )
        d.append(diff[draw].sum(1) / np.array(sz)[draw].sum(1))
        assert z["contrasts"]["PAIR:PREFIX"]["by_seed"][
            SEEDS.index(s)
        ] == pytest.approx(sum(diff) / sum(sz), abs=1e-15)
    np.testing.assert_allclose(
        z["contrasts"]["PAIR:PREFIX"]["ci95"],
        np.percentile(np.mean(d, axis=0), [2.5, 97.5]),
        rtol=0,
        atol=1e-15,
    )
    bad = deepcopy(z)
    bad["branch_cost_by_seed"]["4901"]["PAIR_over_PREFIX"] += 0.03
    with pytest.raises(ValueError, match="price"):
        close.audit(logs[4901], bad, "4901")
    for path in [
        ("chosen_work", "PREFIX", "pair_peak_pair_cells"),
        ("work", "prefix_latest_mistake_discounts"),
        ("seconds", "PREFIX_branch_s"),
    ]:
        r = deepcopy(logs[4901][-1])
        p0 = r
        for key in path[:-1]:
            p0 = p0[key]
        p0[path[-1]] = 0 if path[0] == "seconds" else 10**9
        with pytest.raises(ValueError):
            stats.checked_row("probe", r)
    cs = {
        k: dict(gain=0.3, by_seed=[0.2, 0.3, 0.4], ci95=[0.1, 0.5])
        for k in z["contrasts"]
    }
    assert len(cs) == 14

    def dec():
        return stats.decision(dict(contrasts=cs))

    assert (
        dec()["replicated_queried_pair_table"]
        and dec()["replicated_marginal_trigger"]
        and dec()["prefix_better_than_marg_and_error"]
        and dec()["pair_vs_prefix"] == "PAIR"
    )
    cs["PAIR:SWAP"]["by_seed"][1] = -0.1
    assert (
        not dec()["replicated_queried_pair_table"]
        and dec()["replicated_marginal_trigger"]
    )
    cs["PREFIX:ERROR"]["ci95"][0] = 0
    assert not dec()["prefix_better_than_marg_and_error"]
    cs["PAIR:PREFIX"] = dict(gain=-0.3, by_seed=[-0.2, -0.3, -0.4], ci95=[-0.5, -0.1])
    assert dec()["pair_vs_prefix"] == "PREFIX"
    cs["PAIR:PREFIX"]["ci95"][1] = 0
    assert dec()["pair_vs_prefix"] == "unresolved"


def test_device_bisect_oracle_matches_rational_enumeration_when_the_laws_differ():
    w = base.planted()[0]
    _, _, _, z, p = base.orig.fixture(w, ls.text_tape)
    _, x, tv = base.direct(w, base.orig.queries(len(w)))
    assert tv.sum() > 1
    raw = p.base
    logits = raw + torch.from_numpy(x) @ torch.tensor(
        base.BETA["PAIR"], dtype=torch.float64
    )
    ref = logits - (logits.logsumexp(-1) - raw.logsumexp(-1))[:, None]
    got = checks.prefix_reference(w, z, p, base.BETA["PAIR"])
    torch.testing.assert_close(got, ref, atol=2e-14, rtol=0)
