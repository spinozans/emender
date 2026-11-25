"""Locked JT predictors and natural-denominator arithmetic on invented, short documents only."""

from copy import deepcopy
import json
from math import isclose
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    joint_reader as joint,
    literal_mark as ls,
    local_u,
    surprise as u,
    transfer_score as score,
    transfer_stats as stats,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory import (
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)


binary = (
    native_fixture.binary
)  # reuse the actual retained-or-temporary native pytest fixture


def beta(seed):
    return dict(
        U=list(local_u.contract(seed)[1]),
        H=json.loads((joint.RECEIPTS / "literal_surprise_result.json").read_bytes())[
            "beta"
        ][str(seed)]["H"],
        J=list(joint.contract(seed)[1]),
    )


def test_short_device_shape_reference_parity_and_no_control_search(binary, monkeypatch):
    _, _, writer = binary
    torch.manual_seed(4912)
    host = Tiny().eval()
    head = trained_base().eval()
    w = b"egg: \0\xff\x1e abc ababa " * 300
    s = 4901
    b = beta(s)
    for i, n in enumerate((1, 16, 64, 65, 4095)):
        obs = w[:n]
        h, z, _ = frozen_prefill(host, obs, 8)
        p = ls.fields(head, obs, h, z)
        ref = {"B": p.base.double(), **{a: ls.response(p, a, b[a]) for a in "UHJ"}}

        def refuse(*args, **kw):
            raise AssertionError("wrong-record access in a chosen branch")

        with monkeypatch.context() as mp:
            mp.setattr(u, "_rotation", refuse)
            out, packet, row = score.answers(host, head, obs, b, s, i, writer, width=8)
        assert (
            row["first_branch"] == ("J" if i % 2 else "U")
            and row["host_prefills"] == 1
            and row["targets"] == n
            and set(row["seconds"]) == stats.CLOCKS
            and set(packet.phi) == {"U"}
            and all(
                torch.equal(out[a], ref[a]) and not out[a].requires_grad for a in ref
            )
        )
    for bad in (b"", bytearray(b"a"), b"q" * 4096):
        with pytest.raises(ValueError, match="short observed"):
            score.answers(host, head, bad, b, s, 0, writer, width=8)
    with pytest.raises(ValueError, match="fixed readouts"):
        score.answers(host, head, b"a", dict(b, T=[]), s, 0, writer, width=8)
    for pair in [(4900, 0), (4901, -1), (4901, True)]:
        with pytest.raises(ValueError, match="document rank"):
            score.first(*pair)


def test_first_branch_order_and_labels_only_after_answers(binary, monkeypatch):
    _, _, writer = binary
    host = Tiny().eval()
    head = trained_base().eval()
    word = b"\x00\xffAabc " * 16
    b = beta(4902)
    events = []
    j = joint.branch
    resp = u.response
    labels = score._labels
    orig = score.frozen_prefill

    def mock_j(*a):
        events.append("J")
        return j(*a)

    def mock_resp(packet, arm, vec):
        events.append("H" if vec == b["H"] else "U")
        return resp(packet, arm, vec)

    def mock_prefill(h, obs, width):
        assert obs == word[:-1]
        events.append("prefill")
        return orig(h, obs, width)

    def mock_y(p, y):
        events.append("label")
        return labels(p, y)

    monkeypatch.setattr(joint, "branch", mock_j)
    monkeypatch.setattr(u, "response", mock_resp)
    monkeypatch.setattr(score, "frozen_prefill", mock_prefill)
    monkeypatch.setattr(score, "_labels", mock_y)
    a = score.document(host, head, word, b, 4902, 0, writer, width=8)
    assert events == ["prefill", "J", "U", "H", "label"]
    events.clear()
    c = score.document(host, head, word, b, 4902, 1, writer, width=8)
    assert (
        events == ["prefill", "U", "J", "H", "label"]
        and c["nll_sums"] == a["nll_sums"]
        and c["work"] == a["work"]
    )
    events.clear()
    d = score.document(host, head, word[:-1] + b"\x02", b, 4902, 0, writer, width=8)
    assert a["work"] == d["work"] and d["nll_sums"] != a["nll_sums"]


def test_variable_length_paired_cost_decision_and_conservation(binary):
    _, _, writer = binary
    words = [b"\x00\xff", b"A\xff\x00A" * 19, b"xx xx xy xy " * 29]
    hs = ["first", "second", "third"]
    sizes = [len(w) - 1 for w in words]
    head = trained_base().eval()
    host = Tiny().eval()
    logs = {
        s: [
            dict(
                document_sha=h,
                **score.document(host, head, w, beta(s), s, i, writer, width=8),
            )
            for i, (w, h) in enumerate(zip(words, hs, strict=True))
        ]
        for s in SEEDS
    }
    from experiments.expressivity_tasks.x49_live_trace_memory.transfer_analysis import (
        audit_cost,
    )

    r = stats.summarize(logs, hs, sizes)
    check = r["branch_cost_by_seed"]["4901"]
    audit_cost(logs[4901], check)
    with pytest.raises(ValueError, match="shared-path time"):
        audit_cost(
            logs[4901], dict(check, shared_plus_J_s=check["shared_plus_J_s"] * 2)
        )
    ix = np.random.default_rng(4912).integers(0, 3, (10000, 3))
    den = np.array(sizes)[ix].sum(1)
    ds = []
    for s in SEEDS:
        a = np.array([x["nll_sums"]["U"] - x["nll_sums"]["J"] for x in logs[s]])
        ds.append(a[ix].sum(1) / den)
        ct = r["branch_cost_by_seed"][str(s)]
        j = sum(x["seconds"]["J_branch_s"] for x in logs[s])
        u_s = sum(x["seconds"]["U_branch_s"] for x in logs[s])
        shared = sum(x["seconds"]["shared_s"] for x in logs[s])
        assert isclose(
            ct["shared_path_J_over_U"], (shared + j) / (shared + u_s)
        ) and isclose(
            ct["J_minus_U_s_per_million_targets"], (j - u_s) * 1e6 / sum(sizes)
        )
    np.testing.assert_allclose(
        r["contrasts"]["J:U"]["ci95"],
        np.percentile(np.mean(ds, axis=0), [2.5, 97.5]),
        rtol=0,
        atol=1e-15,
    )
    pos = {"gain": 1.0, "ci95": [0.2, 2.0], "by_seed": [0.2, 0.5, 2.3]}
    results = {"contrasts": {"J:U": deepcopy(pos), "J:H": deepcopy(pos)}}
    assert stats.decision(results)["replicated_fixed_joint_candidate"]
    results["contrasts"]["J:H"]["by_seed"][1] = -1.0
    assert not stats.decision(results)["replicated_fixed_joint_candidate"]
    results["contrasts"]["J:H"] = deepcopy(pos)
    results["contrasts"]["J:U"]["ci95"][0] = 0.0
    assert not stats.decision(results)["replicated_fixed_joint_candidate"]
    bad = deepcopy(logs)
    bad[4903][1]["first_branch"] = score.first(4903, 0)
    with pytest.raises(ValueError, match="order"):
        stats.summarize(bad, hs, sizes)
    bad = deepcopy(logs)
    bad[4901][1]["work"]["S_mark_reads"] = 0
    with pytest.raises(ValueError, match="JT marks"):
        stats.summarize(bad, hs, sizes)
