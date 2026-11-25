"""EH-specific refusal and inference rules, with invented descriptors and FIT/PROBE rows only."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import MIN_OFFSET
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_history_data as data,
    error_history_panel as p,
    error_history_run as run,
    error_history_stats as stats,
    error_history_fit as fit,
    joint_reader as joint,
    literal_mark as ls,
    local_u,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import select, T
from experiments.expressivity_tasks.x49_live_trace_memory.test_alternative_pipeline import (
    fake,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_error_history import (
    Host,
    word,
    trained_base,
)


def test_eh_rank_strictly_after_both_rb_roles_not_different_offsets(monkeypatch):
    records = []
    for i in range(18):
        off = MIN_OFFSET + 16384 * i
        v = i if i != 17 else 4
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=12289,
                split="train",
                sha256=f"{v * 100 + 17:064x}",
            )
        )
    size = off + 12289
    pre = {records[4]["sha256"]}
    rank = select(records, pre, {"train": 13}, size)[0]["train"]
    monkeypatch.setattr(data.rb, "N", dict(fit=4, probe=2))
    b = dict(
        spec=data.rb.SPEC,
        schema="x49-rb-native-train",
        source=dict(size=size),
        roles=dict(fit=rank[:4], probe=rank[4:6]),
    )
    new, ex, supply = data.next_block(b, records, pre, 7)
    assert (
        new == rank[6:] and ex == pre | {r["sha256"] for r in rank[:6]} and supply == 10
    )
    for k, v in [("split", "calibration"), ("start", 4096)]:
        bad = deepcopy(b)
        bad["roles"]["probe"][0][k] = v
        with pytest.raises(ValueError, match="overlapping"):
            data.next_block(bad, records, pre, 7)
    bad = deepcopy(b)
    bad["roles"]["probe"] = rank[5:7]
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(bad, records, pre, 7)


def test_eh_requires_original_anchors_own_scope_and_no_swap_fit(tmp_path, monkeypatch):
    fake(tmp_path, monkeypatch, "EH")
    data.prepare()
    cfg = dict(p.expected("fixture"), operator_note="EH test")
    p.validate(cfg, "fixture")
    z = p.plan("fixture", hash_weights=False)
    for key, v in [
        ("scope", p.dr.SCOPE),
        ("rb_result_sha", "wrong"),
        ("qualification_sha", "different"),
        ("role_documents", dict(fit=5, probe=0)),
        ("weight_and_readouts_id", "foreign"),
        ("j_refits", 1),
        ("swap_fit", "SHIFT"),
        ("fit_scalars_per_seed", dict(p.DIMS, SWAP=6)),
    ]:
        with pytest.raises(ValueError, match="authority"):
            p.validate(dict(cfg, **{key: v}), "fixture")
    for k in ("endpoints", "u_beta", "j_beta"):
        bad = deepcopy(z)
        bad[k]["4901"] = bad[k]["4902"]
        with pytest.raises(ValueError, match="foreign seed"):
            run.document_options(bad, 4901)
    with pytest.raises(ValueError, match="foreign seed"):
        run.document_options(dict(z, writer_binary_sha="wrong"), 4901)
    with pytest.raises(ValueError, match="swap rule"):
        run.document_options(dict(z, swap_fit="SHIFT"), 4901)
    (tmp_path / "4901").write_bytes(b"bad")
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    cs = {
        a + ":" + b: dict(gain=0.02, ci95=[0.01, 0.03], by_seed=[0.02] * 3)
        for a, b in stats.PAIRS
    }
    assert stats.decision(cs)["added_bound_wrong_guess_identities"]
    cs["ERROR:SWAP"]["ci95"][0] = 0
    assert (
        not stats.decision(cs)["added_bound_wrong_guess_identities"]
        and stats.decision(cs)["pooled_scalar_history"]
    )
    cs["HISTORY:COUNT"]["by_seed"][0] = 0
    assert not stats.decision(cs)["pooled_scalar_history"]


def test_eh_work_phase_histogram_and_wrong_common_column_refusal():
    torch.manual_seed(4915)
    word4097 = (word() * 32)[: T + 1]
    beta = {k: [0.1] * d for k, d in p.DIMS.items()}
    row = fit.document(
        Host().eval(),
        trained_base().eval(),
        word4097,
        ub=local_u.contract(4902)[1],
        jb=joint.contract(4902)[1],
        writer=ls.text_tape,
        beta=beta,
        width=8,
    )
    stats.checked_row("probe", row)
    for key in (
        "pool_key_admissions",
        "shift_peak_donor_keys",
        "pool_packed_bytes",
        "fp64_feature_bytes",
        "shift_older_mistake_uses",
        "changed_histogram_rows",
    ):
        bad = deepcopy(row)
        bad["work"][key] += 2 * T * T
        with pytest.raises(ValueError):
            stats.checked_row("probe", bad)
    with pytest.raises(ValueError, match="phase"):
        stats.checked_row("fit", row)
    m = fit.Moments()
    m.targets = T
    m.documents = 1
    for k, d in p.DIMS.items():
        m.gram[k][:] = np.eye(d)
        m.rhs[k][:] = 1.0
    m.freeze()
    m.rhs["ERROR"][4] += 1
    with pytest.raises(ValueError, match="different HISTORY"):
        m.freeze()
