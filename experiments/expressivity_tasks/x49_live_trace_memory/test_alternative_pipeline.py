"""V one-shot lifecycle and negative gates. Invented temporary raw file; no natural text or CUDA."""

from copy import deepcopy
from hashlib import sha256
import json
import os
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MIN_OFFSET,
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    alternative_data as data,
    alternative_panel as p,
    alternative_run as run,
    alternative_analysis as ana,
    alternative_report as report,
    alternative_fit as fit,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, T, select
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise_pipeline import (
    invented,
    fake_endpoints,
    Tiny,
    trained_base,
)


def fake(tmp, mp):
    mp.setenv("OPENBLAS_NUM_THREADS", "1")
    book = invented(tmp, mp, data=data, p=p, run=run, schema="x49-v-native-train")
    calls = fake_endpoints(tmp, mp)
    original = p.local_u.contract
    beta = {s: original(s)[1] for s in SEEDS}
    mp.setattr(
        p.local_u, "contract", lambda s: (sha256(str(s).encode()).hexdigest(), beta[s])
    )
    return book, calls


def test_next_full_train_block_excludes_both_u_roles_and_short_final():
    records = []
    for i in range(19):
        off = MIN_OFFSET + i * 8192
        k = i if i != 10 else 3
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=4097,
                sha256=f"{k * 100 + 3:064x}",
                split="train",
            )
        )
    end = off + 4097
    pre = {records[1]["sha256"]}
    short = [dict(sha256=f"{201:064x}")]
    rank = select(records, pre, {"train": 14}, end)[0]["train"]
    roles = dict(fit=rank[:5], probe=rank[5:8])
    rows, ex, supply = data.next_block(records, pre, roles, short, 6, end)
    assert (
        rows == rank[8:14]
        and ex == pre | {r["sha256"] for r in rank[:8] + short}
        and supply == 9
    )
    with pytest.raises(ValueError, match="overlapping"):
        data.next_block(records, pre, dict(roles, fit=rank[:6]), short, 6, end)
    with pytest.raises(ValueError, match="overlapping"):
        data.next_block(records, pre, roles, short + roles["fit"][:1], 6, end)
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(records, pre, dict(roles, probe=rank[6:9]), short, 6, end)


def test_prepare_is_fit_only_reader_rejects_other_study_and_early_probe(
    tmp_path, monkeypatch
):
    book, _ = fake(tmp_path, monkeypatch)
    pre = os.pread
    reads = []

    def read(fd, n, off):
        reads.append(off)
        return pre(fd, n, off)

    monkeypatch.setattr(os, "pread", read)
    r = data.prepare()
    assert (
        r["fit_targets"] == 2 * T
        and data.metadata()["fit_identity_sha"] == r["fit_identity_sha"]
    )
    expected = [x["offset"] for x in book["roles"]["fit"]]
    assert reads == expected and r["probe_documents_opened"] == 0
    with pytest.raises(ValueError, match="no separate V"):
        data.extract("probe", job=p.JOB, fit_sha="0" * 64)
    with pytest.raises(ValueError, match="unlicensed PROBE"):
        data.extract("probe", job=p.old.JOB, fit_sha="0" * 64)
    with pytest.raises(ValueError, match="wrong V reader"):
        data.fit_windows(p.old.JOB)
    with pytest.raises(ValueError, match="prepare-only"):
        data.extract("fit")
    with pytest.raises(FileExistsError):
        data.prepare()
    with pytest.raises(ValueError, match="unknown two-phase study"):
        data.old.extract("probe", study="other")
    assert (
        reads == expected and not p.ROOT.exists() and not (data.BASE / "probe").exists()
    )


def test_exact_u_bindings_and_conjunctive_v_decision(tmp_path, monkeypatch):
    _, calls = fake(tmp_path, monkeypatch)
    data.prepare()
    a = dict(p.expected("fixture"), operator_note="fixture")
    p.validate(a, "fixture")
    assert (
        a["fitted_scalars_per_seed"] == {"D": 2, "V": 4, "K": 4}
        and a["total_card_seconds"] == 1200
        and a["max_acquisitions"] == 1
        and not a["u_refits"]
    )
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    for key, v in [
        ("scope", "x49-u-bound-arrival-mark-v1"),
        ("remote", True),
        ("u_refits", 1),
        ("data", {}),
        ("endpoints_id", "0" * 64),
    ]:
        with pytest.raises(ValueError):
            p.validate(dict(a, **{key: v}), "fixture")
    (tmp_path / "4902").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    (tmp_path / "4902").write_bytes(b"4902")
    calls.clear()
    h = p.freeze(p.JOB, "fixture")
    plan = load(d / "frozen.json")
    assert sha(d / "frozen.json") == h and calls == [(s, True) for s in SEEDS]
    bad = deepcopy(plan)
    bad["u_beta"]["4901"] = bad["u_beta"]["4902"]
    with pytest.raises(ValueError, match="foreign seed"):
        run.document_options(bad, 4901)
    cs = {
        k: dict(gain=1, ci95=[0.2, 2], by_seed=[1, 0.5, 1.5])
        for k in ("V:D", "V:K", "V:U")
    }
    assert ana.decision(cs)["warrants_separate_confirmation"]
    cs["V:K"]["by_seed"][0] = -0.1
    assert not ana.decision(cs)["warrants_separate_confirmation"]
    cs["V:K"]["by_seed"][0] = 1
    cs["V:U"]["ci95"][0] = 0
    assert not ana.decision(cs)["warrants_separate_confirmation"]


def test_all_seed_freeze_before_probe_and_multiplicity_cholesky_closure(
    tmp_path, monkeypatch
):
    book, calls = fake(tmp_path, monkeypatch)
    data.prepare()
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    h = p.freeze(p.JOB, "fixture")
    initial = load(d / "frozen.json")
    start = dict(
        name=p.JOB.name,
        ref="fixture",
        kind=p.JOB.kind,
        cap_s=p.TOTAL,
        t0=100,
        units=p.units(),
        freeze_sha=h,
    )
    write_json(d / "start.json", start)
    monkeypatch.setattr(run, "admit", lambda j: (d, "fixture", start))
    monkeypatch.setattr(run.panel, "licensed_admitted_reader", lambda *args: initial)
    pid = os.getpid()
    owner = dict(pid="44", starttime="9", gpu="0", created="100", adopted="1")
    ident = dict(
        job=p.JOB.name,
        ref="fixture",
        trainer=pid,
        trainer_start="12",
        owner=owner,
        device="fixture",
        torch="fixture",
        python="fixture",
    )
    write_json(d / "identity.json", ident)
    write_json(d / "launch.json", dict(pid=44, start="9"))
    models = []

    def model(seed, e):
        assert e == initial["endpoints"][str(seed)]
        models.append(seed)
        torch.manual_seed(seed)
        return Tiny().eval(), trained_base().eval()

    def prog(path, n, total, root):
        assert path == d and root == p.ROOT and total == n * T
        write_json(d / "progress.json", dict(pid=pid, units=n, scored_targets=total))
        o = dict(
            ppid=1,
            pid=44,
            start="9",
            trainer=pid,
            trainer_start="12",
            heartbeat=str(100 + n),
            units=n,
        )
        with (d / "survival.jsonl").open("a") as f:
            f.write(json.dumps(o) + "\n")
        if n == 2:
            write_json(d / "live.json", o)

    monkeypatch.setattr(run, "model", model)
    monkeypatch.setattr(
        run, "document", lambda *args, **kw: fit.document(*args, **kw, width=8)
    )
    monkeypatch.setattr(run.panel, "progress", prog)
    pre = os.pread
    seen = []

    def raw(fd, n, off):
        assert (
            (d / "raw-probe-open-attempted").is_file()
            and load(d / "progress.json")["units"] == 6
            and load(d / "fit-frozen.json")["spec"] == data.SPEC
        )
        seen.append(off)
        return pre(fd, n, off)

    monkeypatch.setattr(os, "pread", raw)
    seal = p.freeze_readouts

    def audited(d, plan):
        dig = seal(d, plan)
        live = d / "live.json"
        tmp = live.with_suffix(".hold")
        live.rename(tmp)
        with pytest.raises(ValueError, match="lacks witnessed"):
            data.extract("probe", job=p.JOB, fit_sha=dig)
        tmp.rename(live)
        f = d / "fit-4902.json"
        orig = f.read_bytes()
        bad = json.loads(orig)
        bad["state"]["beta"]["K"][3] += 1
        write_json(f, bad)
        with pytest.raises(ValueError, match="inconsistent"):
            data.extract("probe", job=p.JOB, fit_sha=dig)
        f.write_bytes(orig)
        assert not seen and not (d / "raw-probe-open-attempted").exists()
        return dig

    monkeypatch.setattr(p, "freeze_readouts", audited)
    z = run.evaluate(p.JOB, dict(folder=d, ref="fixture"))
    assert (
        models == list(SEEDS) * 2
        and seen == [x["offset"] for x in book["roles"]["probe"]]
        and not (d / "probe/windows.u8").exists()
        and z["targets"] == 15 * T
        and z["neural_updates"] == 0
    )
    write_json(d / "complete.json", z)
    charge = dict(
        job=p.JOB.name,
        rc=0,
        upper_card_s=16,
        complete_sha=sha(d / "complete.json"),
        renewed=True,
        witnessed_units=15,
        inclusive_wall_s=15.0,
        owner_and_waited_children_cpu_s=5.0,
    )
    write_json(d / "charge.json", charge)
    write_json(p.ROOT / "charged.json", {p.JOB.name: charge})
    (d / "watch-ack").touch()
    for k, hb in [("initial", 100), ("after", 116), ("last", 116)]:
        (d / k).write_text(
            "".join(
                k + "=" + v + "\n" for k, v in dict(owner, heartbeat=str(hb)).items()
            )
        )
    (d / "released").write_text(
        "rc=0\nreleased_epoch=115\nend_monotonic_ns=15000000100\n"
    )
    p.finish(p.ROOT)
    write_json(
        p.ROOT / "done.json",
        dict(
            complete=True,
            charges_sha=sha(p.ROOT / "charged.json"),
            summary_sha=sha(p.ROOT / "summary.json"),
            driver_and_short_parents_cpu_s=1,
        ),
    )
    write_json(run.AUTH, dict(p.expected("fixture"), operator_note="fixture"))
    r = report.report()
    assert (
        r["cost"]["witnessed_fit_boundary"] == 6
        and r["verified"]["independent_paired_bootstrap"]
        and r["u_beta"] == initial["u_beta"]
        and len(r["verified"]["independent_cholesky_residuals"]) == 3
        and set(r["stages"]["probe"]["contrasts"])
        == {"V:D", "V:K", "V:U", "D:U", "K:D"}
    )
    assert r["stages"]["probe"]["real_targets_per_seed"] == 3 * T and r["u_refits"] == 0
    path = d / "probe-4901.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"seconds"', b'"seconds" ', 1))
    with pytest.raises(ValueError, match="foreign V probe journal"):
        report.report()
