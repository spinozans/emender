"""Synthetic-only tests for the natural schedule, endpoint clock, all-byte strata and final decision."""

from copy import deepcopy
from hashlib import sha256
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import panel as p
from experiments.expressivity_tasks.x49_live_trace_memory import panel_eval as ev
from experiments.expressivity_tasks.x49_live_trace_memory.panel_spec import (
    ARMS,
    SEEDS,
    JOBS,
    STEPS,
    T,
    TOTAL,
    SHARED_S,
    training,
    choose,
    common_wall,
    validate_authorization,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_analysis import (
    contrasts,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_freeze import freeze
from experiments.expressivity_tasks.x49_live_trace_memory.train_stream import (
    V2_IDENTITY,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import fresh
from experiments.expressivity_tasks.x49_live_trace_memory.meter_run import fixture
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import score_window


def test_one_bounded_schedule_cannot_inherit_meter_scope(tmp_path, monkeypatch):
    assert (
        len(JOBS) == 15
        and [j.cap for j in JOBS.values()]
        == [1380] * 6 + [900] + [1380] * 6 + [1200, 4200]
        and sum(j.cap for j in JOBS.values()) == TOTAL == 22860
    )
    assert list(JOBS)[:4] == ["A-4901", "B-4901", "A-4902", "B-4902"]
    # Only an in-memory schema fixture; no authorization is created on the production path.
    ref = "f" * 40
    a = dict(
        scope="x49-registered-natural-panel-v2",
        qualified_ref=ref,
        protocol_sha256=p.SPEC,
        total_card_seconds=TOTAL,
        job_caps={k: v.cap for k, v in JOBS.items()},
        gpus=1,
        max_acquisitions=15,
        retries=0,
        remote=False,
        seeds=list(SEEDS),
        train_identity=V2_IDENTITY,
        single_frozen_final=True,
        operator_note="synthetic unit test",
    )
    validate_authorization(a, ref, p.SPEC)
    for k, v in [
        ("total_card_seconds", 22861),
        ("gpus", True),
        ("remote", True),
        ("retries", 1),
        ("qualified_ref", "meter"),
        ("job_caps", {}),
        ("single_frozen_final", False),
    ]:
        with pytest.raises(ValueError):
            validate_authorization(dict(a, **{k: v}), ref, p.SPEC)
    monkeypatch.setattr(p, "OUT", tmp_path)
    with pytest.raises(ValueError, match="no separate whole-panel budget"):
        p.authority()
    with pytest.raises(ValueError, match="no separate whole-panel budget"):
        ev._documents(JOBS["final"], {})
    assert not list(tmp_path.iterdir())


def test_heldout_shift_checks_entire_document_and_refuses_train():
    raw = fixture(1) + fixture(2)
    r = dict(
        sha256=sha256(raw).hexdigest(),
        doc=55,
        start=T,
        length=len(raw),
        split="final_test",
    )
    assert ev.checked_eval_window(raw, r, "final_test") == raw[T : T + T + 1]
    with pytest.raises(ValueError, match="document"):
        ev.checked_eval_window(bytes([raw[0] ^ 1]) + raw[1:], r, "final_test")
    for split in ("train", "calibration"):
        with pytest.raises(ValueError):
            ev.checked_eval_window(raw, r, split)


def histories():
    runs = {}
    for k in ARMS:
        for s in SEEDS:
            ms = [
                dict(step=i, sha256=f"{i:064x}", wall_s=i * (1.25 if k == "C" else 1.0))
                for i in range(32, 1025, 32)
            ]
            runs[training(k, s)] = dict(milestones=ms, wall_s=ms[-1]["wall_s"] + 1.0)
    return runs


def test_wall_selection_is_floor_and_full_geometry():
    r = histories()
    a = r["A-4901"]["milestones"]
    assert choose(a, 32 + SHARED_S)["step"] == 32 and choose(a, 32) is None
    with pytest.raises(ValueError):
        choose(a[1:], 2000)
    w = common_wall(r)
    picks = w["C:A"]["4901"]
    assert (
        picks["selected"]["A"]["step"] == 1024
        and picks["selected"]["C"]["step"] == 800
        and picks["conclusive"]
    )
    assert picks["selected"]["C"]["targets"] == 800 * T
    bad = deepcopy(a)
    bad[-1]["wall_s"] = bad[-2]["wall_s"]
    with pytest.raises(ValueError):
        choose(bad, 3000)
    r["A-4901"]["wall_s"] = 1
    with pytest.raises(ValueError):
        common_wall(r)


def test_document_bootstrap_gates_and_inconclusive_wall():
    w = common_wall(histories())
    table = {}
    base = 2.0 + np.arange(128) / 70
    for k in ARMS:
        for seed in SEEDS:
            for step in {800, STEPS}:
                v = {}
                for h in (128, 1024, T):
                    v[h, "none"] = base - (
                        {"A": 0, "B": 0.001, "F": 0.002, "C": 0.012}[k]
                    ) * (1 if h == T else 0.25)
                if k != "A":
                    v[T, "literal"] = v[T, "none"] + 0.01
                if k in "FC":
                    v[T, "trace"] = v[T, "none"] + 0.02
                table[f"{training(k, seed)}@{step}"] = v
    verdict = contrasts(table, w)
    assert (
        verdict["simplest_nominee"] == "C" and not verdict["decisions"]["B"]["passes"]
    )
    x = verdict["contrasts"]["C:F"]["by_horizon"]["4096"]
    assert x["gain"] == pytest.approx(0.01) and x["ci95"] == pytest.approx([0.01, 0.01])
    w["C:A"]["4902"]["conclusive"] = False
    v = contrasts(table, w)
    assert v["simplest_nominee"] is None and not v["decisions"]["C"]["gates"]["wall"]
    w["C:A"]["4902"]["conclusive"] = True
    table["C-4903@1024"][T, "none"] = base + 0.0001
    v = contrasts(table, w)
    assert (
        v["contrasts"]["C:A"]["by_horizon"]["4096"]["gain"] > 0.005
        and not v["decisions"]["C"]["gates"]["exposure"]
    )


def test_same_scores_and_only_coarse_age_partition_on_frozen_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4901)
        host, head, _ = fresh("C")
        host.eval()
        head.eval()
        kw = dict(arm="C", seed=4901, document=22, width=8)
        w = fixture(4)
        a = score_window(host, head, w, **kw)
        b = score_window(host, head, w, **kw, stratify=True)
        for x, y in zip(a["cells"], b["cells"], strict=True):
            assert (
                x["nll_sum"] == y["nll_sum"]
                and x["work"] == y["work"]
                and not x["strata"]
            )
            for source, groups in y["strata"].items():
                assert sum(z["targets"] for z in groups.values()) == T and sum(
                    z["nll_sum"] for z in groups.values()
                ) == pytest.approx(y["nll_sum"])
                assert (
                    groups["over_1024"]["targets"] == y["work"][source + "_age_gt1024"]
                )


def test_freeze_refuses_without_full_predecessor(tmp_path):
    (tmp_path / "final").mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        freeze(JOBS["final"], "f" * 40, tmp_path)
    assert not (tmp_path / "final/frozen.json").exists()
