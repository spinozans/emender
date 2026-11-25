"""DR-specific identity, independent table reconstruction and decision refusals. Invented only."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import MIN_OFFSET, sha
from experiments.expressivity_tasks.x49_live_trace_memory import (
    decode_data as data,
    decode_fit as fit,
    decode_panel as p,
    decode_run as run,
    decode_store as store,
    decode_stats as stats,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_alternative_pipeline import (
    fake,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import select


def test_dr_native_rank_after_both_ls_roles_and_spent_jt():
    records = []
    for i in range(40):
        off = MIN_OFFSET + 16384 * i
        h = f"{100 * (i if i < 39 else 4) + 3:064x}"
        records.append(
            dict(
                document=off, byte_offset=off, byte_length=8193, split="train", sha256=h
            )
        )
    cal = dict(split="calibration", sha256=f"{500:064x}")
    size = off + 8193
    pre = {records[7]["sha256"]}
    rank = select(records, pre, {"train": 23}, size)[0]["train"]
    roles = dict(fit=rank[:6], probe=rank[6:10])
    rows, ex, supply = data.next_block(records, pre, roles, [cal], 13, size)
    assert (
        rows == rank[10:]
        and ex == pre | {r["sha256"] for r in rank[:10] + [cal]}
        and supply == 28
    )
    with pytest.raises(ValueError, match="wrong spent JT split"):
        data.next_block(records, pre, roles, [dict(cal, split="final_test")], 13, size)
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(records, pre, dict(roles, fit=rank[1:6]), [cal], 13, size)
    with pytest.raises(ValueError, match="overlapping"):
        data.next_block(records, pre | {cal["sha256"]}, roles, [cal], 13, size)


def test_dr_frozen_coefficients_weights_binary_and_decision_do_not_transfer(
    tmp_path, monkeypatch
):
    fake(tmp_path, monkeypatch, "DR")
    data.prepare()
    cfg = dict(p.expected("fixture"), operator_note="fixture")
    p.validate(cfg, "fixture")
    for k, v in [
        ("scope", "x49-ls-bound-text-and-neural-surprise-v1"),
        ("fit_writes_per_seed", 8192),
        ("j_refits", 1),
        ("weight_and_readouts_id", "different"),
        ("remote", True),
        ("writer_binary_sha", "0" * 64),
    ]:
        with pytest.raises(ValueError, match="authority"):
            p.validate(dict(cfg, **{k: v}), "fixture")
    initial = p.plan("fixture", hash_weights=False)
    for key in ("j_beta", "u_beta", "endpoints"):
        z = deepcopy(initial)
        z[key]["4901"] = z[key]["4902"]
        with pytest.raises(ValueError, match="foreign seed"):
            run.document_options(z, 4901)
    with pytest.raises(ValueError, match="foreign seed"):
        run.document_options(dict(initial, writer_binary_sha="wrong"), 4901)
    (tmp_path / "4903").write_bytes(b"wrong")
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    cs = {
        a + ":" + b: dict(gain=0.01, ci95=[0.001, 0.02], by_seed=[0.01] * 3)
        for a, b in stats.PAIRS
    }
    assert stats.decision(cs)["consider_lower_payload_evaluation"]
    cs["D:U"]["ci95"][0] = 0
    assert (
        stats.decision(cs)["prediction_relevant_reconstruction"]
        and not stats.decision(cs)["consider_lower_payload_evaluation"]
    )
    cs["D:S"]["by_seed"][1] = -0.0001
    assert not stats.decision(cs)["prediction_relevant_reconstruction"]


def test_private_npz_shapes_hashes_ridge_and_independent_fp32_check_reject_tamper(
    tmp_path,
):
    m = fit.Moments()
    n = 145
    torch.manual_seed(4913)
    text = torch.randint(256, (n,), dtype=torch.uint8)
    neural = torch.randint(256, (n,), dtype=torch.uint8)
    text[0] = neural[0] = 0
    m.add(b"\xff" + b"\0a" * 72, text, neural, torch.randn(n, 32).tanh())
    dec = m.solve()
    z = store.store(tmp_path, 4901, m, dec)
    seal = store.check_state(tmp_path, 4901, z, n - 1)
    assert (
        seal["absent_cells"] == 254
        and store.independent(tmp_path, 4901, seal)["audited_cells"] == 256
    )
    with pytest.raises(ValueError, match="write denominator"):
        store.check_state(tmp_path, 4901, z, n)
    with pytest.raises(FileExistsError):
        store.store(tmp_path, 4901, m, dec)
    f = tmp_path / "dr-tables-4901.npz"
    a = dec.arrays()
    a["k"] = a["k"].copy()
    a["k"][0, 1, 7] += 0.01
    with f.open("wb") as out:
        np.savez(out, **a)
    with pytest.raises(ValueError, match="changed DR array bytes"):
        store.tables(tmp_path, 4901, seal)
    bad = dict(seal, tables_sha=sha(f))
    with pytest.raises(ValueError, match="reproduce FIT"):
        store.check_state(tmp_path, 4901, bad, n - 1)
    with pytest.raises(ValueError, match="independent ridge"):
        store.independent(tmp_path, 4901, bad)
    with f.open("wb") as out:
        np.savez(out, **dict(a, m=a["m"].astype(np.float64)))
    with pytest.raises(ValueError, match="shape/dtype"):
        store.tables(tmp_path, 4901, dict(seal, tables_sha=sha(f)))
