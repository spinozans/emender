"""PC-specific population, pre-access, marginal-control, clock and seed bindings using invented data only."""

from copy import deepcopy
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import MIN_OFFSET
from experiments.expressivity_tasks.x49_live_trace_memory import (
    paired_correction_data as data,
    paired_correction_panel as p,
    paired_correction_run as run,
    paired_correction_stats as stats,
    paired_correction_report as report,
    paired_correction_fit as fit,
    test_joint_reader as writer_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, T, select
from experiments.expressivity_tasks.x49_live_trace_memory.test_alternative_pipeline import (
    fake,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_paired_correction import (
    QueryHost,
    word,
    trained_base,
    UB,
    JB,
    HB,
    EB,
    COEF,
)

binary = writer_fixture.binary


def test_pc_rank_after_eh_fit_probe_and_whole_et_calibration_prefix(monkeypatch):
    records = []
    for i in range(22):
        off = MIN_OFFSET + 16384 * i
        short = i in (18, 19, 20)
        v = i if i != 21 else 3
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=23 if short else 8193,
                split="calibration" if short else "train",
                sha256=f"{v * 100 + (0 if short else 3):064x}",
            )
        )
    size = off + 8193
    source = dict(size=size)
    pre = {records[2]["sha256"]}
    rank = select(records, pre, {"train": 15}, size)[0]["train"]
    monkeypatch.setattr(data.et.eh, "N", dict(fit=4, probe=3))
    monkeypatch.setattr(data.et, "GOAL", 34)
    eb = dict(
        spec=data.et.eh.SPEC,
        schema="x49-eh-native-train",
        source=source,
        roles=dict(fit=rank[:4], probe=rank[4:7]),
    )
    after = pre | {x["sha256"] for x in rank[:7]}
    rows, n, nt = data.et.jt.choose(records, after, size, 34, MIN_OFFSET)
    sb = dict(
        spec=data.et.SPEC,
        source=source,
        rows=rows,
        eligible_documents=n,
        eligible_targets=nt,
    )
    pc, ex, supply = data.next_block(eb, sb, records, after, 8)
    assert (
        pc == rank[7:15]
        and len(rows) == 2
        and ex == after | {x["sha256"] for x in rows}
        and supply == 10
    )
    with pytest.raises(ValueError, match="missing EH"):
        data.next_block(eb, sb, records, pre, 8)
    with pytest.raises(ValueError, match="overlapping ET"):
        data.next_block(eb, sb, records, after | {rows[0]["sha256"]}, 8)
    broken = deepcopy(eb)
    broken["roles"]["probe"] = rank[5:8]
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(
            broken, sb, records, pre | {x["sha256"] for x in rank[:4] + rank[5:8]}, 8
        )
    broken = deepcopy(sb)
    broken["rows"][0]["start"] = 1
    with pytest.raises(ValueError, match="whole short"):
        data.next_block(eb, broken, records, after, 8)


def test_pc_preparation_needs_public_pass_and_own_authority_and_original_error_readouts(
    tmp_path, monkeypatch
):
    fake(tmp_path, monkeypatch, "PC")
    monkeypatch.setattr(run.joint, "_writer", lambda: object())
    with monkeypatch.context() as mp:

        def refused():
            raise ValueError("missing public qualification")

        mp.setattr(run, "checked_fixture_ref", refused)
        with pytest.raises(ValueError, match="missing public"):
            data.prepare()
    assert not data.BASE.exists()
    data.prepare()
    cfg = dict(p.expected("fixture"), operator_note="fixture PC")
    z = p.plan("fixture", hash_weights=False)
    p.validate(cfg, "fixture")
    for k, v in [
        ("scope", "old EH job"),
        ("et_result_sha", "old"),
        ("qualification_sha", "bad"),
        ("weight_and_readouts_id", "bad"),
        ("swap_fit", "SHUF"),
        ("error_refits", 1),
        ("retries", 1),
        ("fit_scalars_per_seed", dict(p.DIMS, SWAP=2)),
    ]:
        with pytest.raises(ValueError, match="authority"):
            p.validate(dict(cfg, **{k: v}), "fixture")
    run.document_options(z, 4901)
    for k in ("endpoints", *p.FIXED.values()):
        bad = deepcopy(z)
        bad[k]["4901"] = bad[k]["4902"]
        with pytest.raises(ValueError, match="foreign seed"):
            run.document_options(bad, 4901)
    for k, v in [
        ("eh_fit_freeze_sha", "wrong"),
        ("swap_fit", "SHUF"),
        ("fit_dimensions", dict(p.DIMS, PAIR=3)),
    ]:
        with pytest.raises(ValueError, match="rules"):
            run.document_options(dict(z, **{k: v}), 4901)
    f = p.ROOT / p.JOB.name
    f.mkdir(parents=True)
    (tmp_path / "4902").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "fixture")
    assert not (f / "frozen.json").exists()


def test_pc_support_changed_law_vs_exchange_peaks_prices_and_conjunctive_decision(
    binary,
):
    torch.manual_seed(4917)
    w = (word() * 17)[: T + 1]
    r = fit.document(
        QueryHost().eval(),
        trained_base().eval(),
        w,
        ub=UB,
        jb=JB,
        hb=HB,
        eb=EB,
        writer=binary[-1],
        beta=COEF,
        width=8,
    )
    stats.checked_row("probe", r)
    for k in [
        "matched_query_rows",
        "pair_peak_pair_cells",
        "shuf_mistake_commits",
        "changed_conditional_uses",
        "error_fp64_feature_bytes",
        "panel_feature_bytes",
    ]:
        bad = deepcopy(r)
        bad["work"][k] += 2 * T * T
        with pytest.raises(ValueError):
            stats.checked_row("probe", bad)
    with pytest.raises(ValueError, match="phase"):
        stats.checked_row("fit", r)
    bad = deepcopy(r)
    bad["seconds"]["shared_posthost_ERROR_s"] = 0
    with pytest.raises(ValueError, match="clocks"):
        stats.checked_row("probe", bad)
    z = stats.summarize(
        "probe", {s: [dict(r, document_sha="invented")] for s in SEEDS}, ["invented"]
    )
    report.audit("probe", [r], z, "4902")
    z["probe_path_estimates_by_seed"]["4902"]["PAIR_extra_s"] += 1
    with pytest.raises(ValueError, match="clock"):
        report.audit("probe", [r], z, "4902")
    cs = {
        a + ":" + b: dict(gain=0.02, ci95=[0.01, 0.03], by_seed=[0.02] * 3)
        for a, b in stats.PAIRS
    }
    assert stats.decision(cs)["consider_locked_pair_transfer"]
    cs["PAIR:SWAP"]["ci95"][0] = 0
    d = stats.decision(cs)
    assert (
        d["useful_marginal_trigger"]
        and not d["useful_queried_pair_table"]
        and not d["consider_locked_pair_transfer"]
    )
    cs["MARG:ERROR"]["by_seed"][1] = -0.01
    assert not stats.decision(cs)["useful_marginal_trigger"]
