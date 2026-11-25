"""LS-specific membership, funding, predictor identity and reporting checks; no natural LS rows."""

from copy import deepcopy
import pytest
import numpy as np
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MIN_OFFSET,
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    literal_mark_data as data,
    literal_mark_panel as p,
    literal_mark_run as run,
    literal_mark_analysis as ana,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import select
from experiments.expressivity_tasks.x49_live_trace_memory.test_alternative_pipeline import (
    fake,
)


def test_next_block_starts_after_both_spent_v_roles(tmp_path, monkeypatch):
    records = []
    for i in range(24):
        k = i if i != 10 else 5
        off = MIN_OFFSET + 8192 * i
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=4097,
                split="train",
                sha256=f"{k * 100 + 2:064x}",
            )
        )
    end = off + 4097
    pre = {records[1]["sha256"]}
    order = select(records, pre, {"train": 16}, end)[0]["train"]
    v = data.v
    monkeypatch.setattr(v, "N", dict(fit=4, probe=3))
    book = dict(
        schema="x49-v-native-train",
        spec=v.SPEC,
        source=dict(size=end),
        roles=dict(fit=order[:4], probe=order[4:7]),
    )
    new, excluded, supply = data.next_block(book, records, pre, 9)
    assert (
        new == order[7:]
        and excluded == pre | {x["sha256"] for x in order[:7]}
        and supply == 15
    )
    bad = deepcopy(book)
    bad["roles"]["probe"][0] = order[3]
    with pytest.raises(ValueError, match="overlapping"):
        data.next_block(bad, records, pre, 9)
    bad["roles"]["probe"] = order[5:8]
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(bad, records, pre, 9)
    monkeypatch.setattr(v, "BASE", tmp_path)
    file = tmp_path / "reservation.json"
    write_json(file, book)
    meta = dict(rows_sha=sha(file))
    monkeypatch.setattr(data, "checked_spec", lambda: dict(data=meta))
    monkeypatch.setattr(v, "metadata", lambda: meta)
    monkeypatch.setattr(v, "selection_context", lambda: (bad, records, pre))
    with pytest.raises(ValueError, match="spent V reservation"):
        data.selection()


def test_ls_freeze_requires_its_own_budget_model_and_incumbent(tmp_path, monkeypatch):
    _, calls = fake(tmp_path, monkeypatch, "LS")
    data.prepare()
    a = dict(p.expected("fixture"), operator_note="fixture")
    p.validate(a, "fixture")
    assert (
        a["role_documents"] == data.N
        and a["max_acquisitions"] == 1
        and a["total_card_seconds"] == 1200
        and a["v_result_sha"] == data.V_RESULT
        and a["fitted_scalars_per_seed"] == dict(N=3, T=4, H=4, J=5, S=5)
    )
    for key, val in [
        ("scope", p.previous.SCOPE),
        ("u_refits", 1),
        ("v_result_sha", "0" * 64),
        ("remote", True),
        ("job_caps", {"v-fit-probe": 1200}),
        ("data", {}),
        ("fitted_scalars_per_seed", dict(N=3, T=4, H=4, J=5, S=5, U=4)),
    ]:
        with pytest.raises(ValueError):
            p.validate(dict(a, **{key: val}), "fixture")
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    (tmp_path / "4902").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    (tmp_path / "4902").write_bytes(b"4902")
    dig = p.freeze(p.JOB, "fixture")
    initial = load(d / "frozen.json")
    assert (
        sha(d / "frozen.json") == dig
        and initial["v_result_sha"] == a["v_result_sha"]
        and initial["u_refits"] == initial["neural_updates"] == 0
    )
    bad = deepcopy(initial)
    bad["u_beta"]["4901"] = bad["u_beta"]["4903"]
    with pytest.raises(ValueError, match="foreign seed"):
        run.document_options(bad, 4901)
    with pytest.raises(ValueError, match="repeated LS"):
        p.freeze(p.JOB, "fixture")


def test_independent_solve_rejects_tampering_in_correlated_columns():
    from experiments.expressivity_tasks.x49_live_trace_memory.surprise_report import (
        checked_ridge_solutions,
    )

    m = np.ones((4, 4)) + p.RIDGE * np.eye(4)
    truth = np.array([1.0, 0.4, 0.08, 0.08])
    r = m @ truth
    orig = np.linalg.solve(m, r)
    chol = np.linalg.cholesky(m)
    audit = np.linalg.solve(chol.T, np.linalg.solve(chol, r))
    checked_ridge_solutions(m, r, orig, audit)
    # Move in a weak direction. Checking only the dominant direction or printed coefficient
    # digits would miss this corruption. Test both stored and independent solutions.
    altered = orig + np.array([0.0, 0.0, 1e-7, -1e-7])
    for a, b in ((altered, audit), (orig, altered)):
        with pytest.raises(ValueError, match="inaccurate independent"):
            checked_ridge_solutions(m, r, a, b)
    with pytest.raises(ValueError, match="missing positive ridge"):
        checked_ridge_solutions(np.ones((4, 4)), r, orig, audit)


def test_correlation_zero_variance_and_primary_vs_predictor_decisions():
    def calc(x, y):
        x, y = np.array(x), np.array(y)
        return dict(
            literal_hits=len(x),
            text_code_sum=int(x.sum()),
            neural_code_sum=int(y.sum()),
            text_code_square_sum=int(x @ x),
            neural_code_square_sum=int(y @ y),
            text_neural_code_product_sum=int(x @ y),
        )

    x = [1, 1, 3, 255, 92]
    y = [14, 123, 26, 77, 18]
    assert ana.correlation(calc(x, y)) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert (
        ana.correlation(calc(x, [2] * 5)) is None
        and ana.correlation(calc([], [])) is None
    )
    with pytest.raises(ValueError, match="invalid paired"):
        ana.correlation(dict(calc(x, y), text_neural_code_product_sum=100000000))
    cs = {
        a + ":" + b: dict(gain=0.1, ci95=[0.01, 0.2], by_seed=[0.09, 0.1, 0.11])
        for a, b in ana.PAIRS
    }
    r = ana.decision(cs)
    assert (
        r["residual_neural_information"]
        and r["consider_joint_confirmation"]
        and not r["consider_literal_rival"]
    )
    cs["J:H"]["by_seed"][0] = -0.01
    assert (
        ana.decision(cs)["residual_neural_information"]
        and not ana.decision(cs)["consider_joint_confirmation"]
    )
    cs["J:S"]["ci95"][0] = 0.0
    assert not ana.decision(cs)["residual_neural_information"]
    cs["H:T"] = dict(gain=-0.1, ci95=[-0.2, -0.01], by_seed=[-0.09, -0.1, -0.11])
    assert ana.decision(cs)["consider_literal_rival"]
    cs["T:U"]["ci95"][0] = 0
    assert not ana.decision(cs)["consider_literal_rival"]
