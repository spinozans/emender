"""Single-use post-U lifecycles and negative gates on invented data only."""

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
    fit_phase,
    literal_mark_report,
    decode_report,
    residual_report,
    test_joint_reader as native_fixture,
    local_u,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, T, select
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise_pipeline import (
    invented,
    fake_endpoints,
    Tiny,
    trained_base,
)


binary = native_fixture.binary


def fake(tmp, mp, name="V"):
    data, p, run = fit_phase.incumbent(name)
    mp.setenv("OPENBLAS_NUM_THREADS", "1")
    book = invented(
        tmp, mp, data=data, p=p, run=run, schema=f"x49-{name.lower()}-native-train"
    )
    calls = fake_endpoints(tmp, mp)
    original = local_u.contract
    beta = {s: original(s)[1] for s in SEEDS}
    if name in ("DR", "RB"):
        drp = p if name == "DR" else p.dr
        jbeta = {s: run.joint.contract(s)[1] for s in SEEDS}

        def coeff(s):
            return dict(U=list(beta[s]), J=list(jbeta[s]), H=[0] * 4)

        def ck(s):
            return sha256(str(s).encode()).hexdigest()

        mp.setattr(run.joint, "contract", lambda s: (ck(s), jbeta[s]))
        mp.setattr(
            drp.data,
            "checked_spec",
            lambda: dict(
                predictors={
                    str(s): dict(checkpoint=ck(s), beta=coeff(s)) for s in SEEDS
                }
            ),
        )
        mp.setattr(
            drp.jt,
            "endpoints",
            lambda *, hash_weights: {
                str(s): dict(
                    weights=p.old.checkpoint(s, hash_weights=hash_weights)[1],
                    beta=coeff(s),
                )
                for s in SEEDS
            },
        )
        if name == "RB":
            from experiments.expressivity_tasks.x49_live_trace_memory.test_residual_pipeline import (
                fake_residual_parent,
            )

            fake_residual_parent(tmp, mp, ck, coeff)
    mp.setattr(
        local_u, "contract", lambda s: (sha256(str(s).encode()).hexdigest(), beta[s])
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


@pytest.mark.parametrize("name", ["V", "LS", "DR", "RB"])
def test_prepare_is_fit_only_reader_rejects_other_study_and_early_probe(
    tmp_path, monkeypatch, name
):
    data, p, _ = fit_phase.incumbent(name)
    book, _ = fake(tmp_path, monkeypatch, name)
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
    with pytest.raises(ValueError, match=f"no separate {name}"):
        data.extract("probe", job=p.JOB, fit_sha="0" * 64)
    with pytest.raises(ValueError, match="unlicensed PROBE"):
        data.extract("probe", job=p.old.JOB, fit_sha="0" * 64)
    with pytest.raises(ValueError, match=f"wrong {name} reader"):
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


@pytest.mark.parametrize("name", ["V", "LS", "DR", "RB"])
def test_all_seed_freeze_before_probe_and_multiplicity_cholesky_closure(
    tmp_path, monkeypatch, name, request
):
    data, p, run = fit_phase.incumbent(name)
    publish = {
        "V": report.report,
        "LS": literal_mark_report.report,
        "DR": decode_report.report,
        "RB": residual_report.report,
    }[name]
    if name in ("DR", "RB"):
        writer = request.getfixturevalue("binary")[-1]
        monkeypatch.setattr(run.joint, "_writer", lambda: writer)
    book, calls = fake(tmp_path, monkeypatch, name)
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
    score_document = run.document
    monkeypatch.setattr(
        run, "document", lambda *args, **kw: score_document(*args, **kw, width=8)
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
        table_job = name in ("DR", "RB")
        f = d / (f"{name.lower()}-tables-4902.npz" if table_job else "fit-4902.json")
        orig = f.read_bytes()
        if table_job:
            f.write_bytes((d / f"{name.lower()}-tables-4901.npz").read_bytes())
        else:
            bad = json.loads(orig)
            bad["state"]["beta"]["K" if name == "V" else "S"][3] += 1
            write_json(f, bad)
        with pytest.raises(
            ValueError,
            match=f"changed {name} array bytes" if table_job else "inconsistent",
        ):
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
        and z["fitted_readouts"] == {"V": 9, "LS": 15, "DR": 3, "RB": 3}[name]
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
    r = publish()
    assert (
        r["cost"]["witnessed_fit_boundary"] == 6
        and r["verified"]["independent_paired_bootstrap"]
        and r["u_beta"] == initial["u_beta"]
        and len(
            r["verified"][
                "independent_decoder_checks"
                if name in ("DR", "RB")
                else "independent_cholesky_residuals"
            ]
        )
        == 3
        and set(r["stages"]["probe"]["contrasts"])
        == {
            "V": {"V:D", "V:K", "V:U", "D:U", "K:D"},
            "LS": {"J:T", "J:S", "H:T", "T:N", "H:N", "S:T", "J:H", "T:U", "J:U"},
            "DR": {"D:P", "D:S", "J:P", "J:D", "D:U", "D:B", "J:U", "U:B", "S:P"},
            "RB": {
                "R:D",
                "R:G",
                "R:S",
                "J:R",
                "R:U",
                "R:B",
                "J:D",
                "D:U",
                "S:G",
                "G:D",
                "J:U",
            },
        }[name]
    )
    assert r["stages"]["probe"]["real_targets_per_seed"] == 3 * T and r["u_refits"] == 0
    path = d / "probe-4901.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"seconds"', b'"seconds" ', 1))
    if name == "DR":
        assert (
            r["verified"]["independent_MSE_denominator"]
            and r["j_beta"] == initial["j_beta"]
            and r["j_refits"] == 0
        )
        assert all(
            x["table_bytes"] == 100352
            and x["fit_writes"] == 2 * (T - 1)
            and x["synthetic"]["all_256_teacher_disabled_D_equal"]
            for x in r["decoders"].values()
        )
    if name == "RB":
        assert (
            r["verified"]["independent_occupancy_and_fallback"]
            and r["verified"]["independent_MSE_denominator"]
            and r["qualified_axes"] == initial["qualified_axes"]
            and r["d_refits"] == 0
        )
        assert all(
            x["residual_table_bytes"] == 32768
            and x["fit_writes"] == 2 * (T - 1)
            and x["synthetic"]["one_write_standalone_R_equal"]
            for x in r["decoders"].values()
        )
    if name == "LS":
        assert r["verified"]["independent_pooled_mark_correlations"] and all(
            len(row) == 5
            for row in r["verified"]["independent_cholesky_residuals"].values()
        )
    with pytest.raises(ValueError, match=f"foreign {name} probe journal"):
        publish()
