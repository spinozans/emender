"""ET-only data and reduction oracles. All descriptors/words are invented; no natural calibration access."""

from copy import deepcopy
import numpy as np
import pytest
from experiments.expressivity_tasks.x48_muon_continuation.data import write_json, sha
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_transfer_data as data,
    error_transfer_stats as stats,
    error_transfer_analysis as audit,
    error_transfer_score as score,
    literal_mark as ls,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, digest
from experiments.expressivity_tasks.x49_live_trace_memory.test_transfer_data import (
    record,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_transfer_score import (
    setup,
    beta,
    word,
)


def test_et_suffix_contiguous_after_jt_and_minimal_crossing():
    raw = [record(i, 5000 + 8192 * i, 30 + i) for i in range(23)]
    sz = 250000
    lo = 4000
    raw.extend(
        [
            dict(raw[6], byte_offset=200000, document=200000),
            record(25, 210000, 70, "final_test"),
            record(26, 216000, 4097),
            record(27, 225000, 40, "train"),
        ]
    )
    exclude = {raw[2]["sha256"]}
    spent, _, total = data.jt.choose(raw, exclude, sz, 170, lo)
    ex = exclude | {v["sha256"] for v in spent}
    rows, n, nt = data.next_short(raw[::-1], ex, spent, sz, 165, lo)
    order = sorted(
        [v for v in raw[:23] if v["sha256"] not in exclude],
        key=lambda r: digest(b"x49-jt-short-cal-v1:" + r["sha256"].encode()),
    )
    assert (
        [r["offset"] for r in spent + rows]
        == [r["byte_offset"] for r in order[: len(spent + rows)]]
        and nt == total - sum(r["length"] - 1 for r in spent)
        and n == 22 - len(spent)
    )
    assert (
        sum(r["length"] - 1 for r in rows)
        >= 165
        > sum(r["length"] - 1 for r in rows[:-1])
    )
    bad = deepcopy(spent)
    bad[-1]["offset"] += 1
    with pytest.raises(ValueError, match="contiguous"):
        data.next_short(raw, ex, bad, sz, 165, lo)
    with pytest.raises(ValueError, match="missing spent"):
        data.next_short(raw, exclude, spent, sz, 165, lo)
    with pytest.raises(ValueError, match="insufficient"):
        data.next_short(raw, ex, spent, sz, nt + 1, lo)


def test_spent_eh_receipt_and_both_roles_are_required(monkeypatch, tmp_path):
    prior = {"c" * 64}
    used = [
        dict(
            doc=5000 + i * 8192,
            offset=5000 + i * 8192,
            length=4097,
            start=0,
            split="train",
            sha256=f"{102 + i * 100:064x}",
        )
        for i in range(4)
    ]
    used.sort(key=lambda r: digest(b"x49-live-v1:" + r["sha256"].encode()))
    b = dict(
        spec=data.eh.SPEC,
        schema="x49-eh-native-train",
        source=dict(size=80000),
        roles=dict(fit=used[:2], probe=used[2:]),
    )
    f = tmp_path / "reservation.json"
    write_json(f, b)
    m = dict(rows_sha=sha(f))
    monkeypatch.setattr(data.eh, "metadata", lambda: m)
    monkeypatch.setattr(data.eh, "MIN_OFFSET", 0)
    monkeypatch.setattr(data.eh, "BASE", tmp_path)
    monkeypatch.setattr(data.eh, "N", dict(fit=2, probe=2))
    monkeypatch.setattr(data.eh, "selection_context", lambda: (b, ["manifest"], prior))
    assert data.spent_context(m) == (
        b,
        ["manifest"],
        prior | {v["sha256"] for v in used},
    )
    with pytest.raises(ValueError, match="changed spent"):
        data.spent_context(dict(rows_sha="wrong"))
    b["roles"]["probe"][0]["start"] = 4096
    with pytest.raises(ValueError, match="changed spent"):
        data.spent_context(m)
    write_json(f, b)
    m["rows_sha"] = sha(f)
    with pytest.raises(ValueError, match="overlapping FIT/PROBE"):
        data.spent_context(m)


def test_weighted_et_reduction_cost_support_and_two_decisions():
    host, head, _ = setup(4901)
    words = [b"\0\xff", b"abc abd " * 15, word() * 4]
    hs = ["doc-0", "doc-1", "doc-2"]
    lens = [len(w) - 1 for w in words]
    logs = {
        s: [
            dict(
                document_sha=h,
                **score.document(host, head, w, beta(s), s, i, ls.text_tape, width=8),
            )
            for i, (w, h) in enumerate(zip(words, hs, strict=True))
        ]
        for s in SEEDS
    }
    out = stats.summarize(logs, hs, lens)
    ix = np.random.default_rng(4916).integers(0, 3, size=(10000, 3))
    den = np.array(lens)[ix].sum(1)
    draw = []
    for s in SEEDS:
        k = str(s)
        audit.audit(logs[s], out, k)
        diff = np.asarray(
            [r["nll_sums"]["HISTORY"] - r["nll_sums"]["ERROR"] for r in logs[s]]
        )
        draw.append(diff[ix].sum(1) / den)
        sec = out["seconds_by_seed"][k]
        measured = out["branch_cost_by_seed"][k]
        assert measured["shared_path_ERROR_over_HISTORY"] == pytest.approx(
            (sec["shared_J_s"] + sec["ERROR_branch_s"])
            / (sec["shared_J_s"] + sec["HISTORY_branch_s"])
        )
    np.testing.assert_allclose(
        out["contrasts"]["ERROR:HISTORY"]["ci95"],
        np.percentile(np.mean(draw, 0), [2.5, 97.5]),
        rtol=0,
        atol=1e-15,
    )
    bad = deepcopy(out)
    bad["branch_cost_by_seed"]["4902"]["shared_path_ERROR_over_HISTORY"] *= 2
    with pytest.raises(ValueError, match="shared-path cost"):
        audit.audit(logs[4902], bad, "4902")
    bad = deepcopy(logs)
    bad[4901][-1]["chosen_work"]["HISTORY"]["packed_bytes"] += 1
    with pytest.raises(ValueError, match="chosen-only"):
        stats.summarize(bad, hs, lens)
    cs = {k: dict(gain=1, ci95=[0.1, 2], by_seed=[1, 1, 1]) for k in out["contrasts"]}
    a = dict(contrasts=cs)
    assert (
        stats.decision(a)["replicated_bound_wrong_guesses"]
        and stats.decision(a)["replicated_pooled_scalar_history"]
    )
    cs["ERROR:SWAP"]["ci95"][0] = 0
    assert (
        not stats.decision(a)["replicated_bound_wrong_guesses"]
        and stats.decision(a)["replicated_pooled_scalar_history"]
    )
    cs["HISTORY:J"]["by_seed"][0] = -0.1
    assert not stats.decision(a)["replicated_pooled_scalar_history"]
