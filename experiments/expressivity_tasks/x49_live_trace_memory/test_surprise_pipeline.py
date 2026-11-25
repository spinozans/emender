"""U rank, raw-access boundary, one-shot fits, probe and inference on invented native TRAIN documents only."""

from copy import deepcopy
from hashlib import sha256
import json
import os
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MIN_OFFSET,
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    surprise_panel as p,
    surprise_data as data,
    surprise_run as run,
    surprise_fit as fit,
    surprise_analysis as analysis,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import (
    SEEDS,
    T,
    select,
    digest,
    fingerprint,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)


def test_rank_after_all_prior_blocks_and_whole_document_exclusion(
    tmp_path, monkeypatch
):
    q = data.q
    monkeypatch.setattr(q, "BASE", tmp_path)
    monkeypatch.setattr(q, "N", 3)
    monkeypatch.setattr(data, "N", dict(fit=3, probe=2))
    records = []
    for i in range(18):
        off = MIN_OFFSET + 8192 * i
        h = f"{100 * (i if i != 15 else 3) + 3:064x}"
        records.append(
            dict(
                document=off, byte_offset=off, byte_length=4097, sha256=h, split="train"
            )
        )
    size = off + 4097
    ex = {records[3]["sha256"]}
    w = dict(source=dict(size=size), panels=select(records, ex, {"train": 2}, size)[0])
    rows, pre_q, left = q.select_new(
        records, base_ex=ex, prior=w["panels"], quota=3, size=size
    )
    qr = dict(
        rows=rows,
        source=w["source"],
        spec=q.SPEC,
        parent_rows_sha=q.W_ROWS_SHA,
        eligible_train=left,
        excluded_sha=fingerprint(sorted(pre_q)),
        manifest_sha="fixture",
    )
    f = tmp_path / "reservation.json"
    write_json(f, qr)
    result = {"data": {"rows_sha": sha(f)}}
    monkeypatch.setattr(data, "checked_spec", lambda: result)
    monkeypatch.setattr(q, "parent_context", lambda: (records, ex, w))
    r = data.selection()
    got = data.checked_roles(r)
    rank = select(records, ex, {"train": 10}, size)[0]["train"]
    assert (
        got["fit"] == rank[5:8]
        and got["probe"] == rank[8:10]
        and r["eligible_train"] == 11
        and r["excluded_unique"] == 6
    )
    bad = deepcopy(r)
    bad["roles"]["probe"][1] = bad["roles"]["fit"][0]
    with pytest.raises(ValueError, match="overlapping"):
        data.checked_roles(bad)
    bad = deepcopy(r)
    bad["roles"]["fit"].reverse()
    with pytest.raises(ValueError, match="rank"):
        data.checked_roles(bad)
    write_json(f, dict(qr, rows=rows[::-1]))
    result["data"]["rows_sha"] = sha(f)
    with pytest.raises(ValueError, match="Q rank"):
        data.selection()


def invented(tmp, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GPU_LEASE_HELD", raising=False)
    monkeypatch.setattr(data, "N", dict(fit=2, probe=3))
    monkeypatch.setattr(data, "MIN_OFFSET", 0)
    monkeypatch.setattr(data, "BASE", tmp / "data")
    monkeypatch.setattr(p, "ROOT", tmp / "run")
    monkeypatch.setattr(run, "AUTH", tmp / "no-budget")
    src = tmp / "raw"
    rows = []
    i = 0
    with src.open("xb") as f:
        while len(rows) < 5:
            w = (
                (i.to_bytes(2, "little") + b"\0\xff\x1eMandelbrot" + bytes(range(16)))
                * 305
            )[:8193]
            i += 1
            h = digest(w)
            if int(h, 16) % 100 < 2:
                continue
            off = f.tell()
            f.write(w)
            rows.append(
                dict(
                    doc=off,
                    offset=off,
                    length=len(w),
                    start=T * (int(h[16:32], 16) % 2),
                    sha256=h,
                    split="train",
                )
            )
    rows.sort(key=lambda r: digest(b"x49-live-v1:" + r["sha256"].encode()))
    st = src.stat()
    book = dict(
        schema="x49-u-native-train",
        spec=data.SPEC,
        source=dict(
            path=str(src), size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino
        ),
        eligible_train=5,
        excluded_unique=0,
        roles={"fit": rows[:2], "probe": rows[2:]},
    )
    monkeypatch.setattr(data, "selection", lambda: deepcopy(book))
    monkeypatch.setattr(run, "checked_fixture_ref", lambda: "fixture")
    return book


def test_single_fit_preparation_and_early_or_foreign_probe_refusal(
    tmp_path, monkeypatch
):
    book = invented(tmp_path, monkeypatch)
    seen = []
    real = os.pread

    def pread(fd, n, off):
        seen.append(off)
        return real(fd, n, off)

    monkeypatch.setattr(data.os, "pread", pread)
    monkeypatch.setattr(
        run,
        "checked_fixture_ref",
        lambda: (_ for _ in ()).throw(ValueError("missing gate")),
    )
    with pytest.raises(ValueError, match="missing gate"):
        data.prepare()
    assert not data.BASE.exists() and not seen
    monkeypatch.setattr(run, "checked_fixture_ref", lambda: "fixture")
    r = data.prepare()
    assert (
        seen == [x["offset"] for x in book["roles"]["fit"]]
        and r["probe_documents_opened"] == 0
        and r["fit_targets"] == 2 * T
        and not (data.BASE / "probe").exists()
    )
    meta = data.metadata()
    assert (
        meta["rows_sha"] == r["rows_sha"]
        and meta["fit_identity_sha"] == r["fit_identity_sha"]
    )
    with pytest.raises(ValueError, match="unlicensed PROBE"):
        data.extract("probe", fit_sha="0" * 64)
    with pytest.raises(ValueError, match="no separate U"):
        data.extract("probe", job=p.JOB, fit_sha="0" * 64)
    with pytest.raises(ValueError, match="wrong U reader"):
        data.fit_windows(None)
    with pytest.raises(ValueError, match="FIT is prepare-only"):
        data.extract("fit")
    with pytest.raises(FileExistsError):
        data.prepare()
    d = tmp_path / "stage"
    d.mkdir()
    monkeypatch.setattr(data, "licensed", lambda job: (d, dict(data=meta)))
    with pytest.raises(ValueError, match="no completed U FIT"):
        data.extract("probe", job=p.JOB, fit_sha="0" * 64)
    assert (
        seen == [x["offset"] for x in book["roles"]["fit"]]
        and not (d / "raw-probe-open-attempted").exists()
    )
    path = data.BASE / "fit/windows.u8"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 255]))
    with pytest.raises(ValueError, match="foreign TRAIN stream"):
        data.fit_windows(p.JOB)


def fake_endpoints(tmp, monkeypatch):
    for s in SEEDS:
        (tmp / str(s)).write_bytes(str(s).encode())
    calls = []

    def checkpoint(s, *, hash_weights):
        path = tmp / str(s)
        v = sha256(str(s).encode()).hexdigest()
        calls.append((s, hash_weights))
        if hash_weights and sha(path) != v:
            raise ValueError("changed weight bytes")
        return path, dict(sha256=v)

    monkeypatch.setattr(p, "checkpoint", checkpoint)
    return calls


def test_u_budget_endpoint_freeze_and_solved_state_are_not_interchangeable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(
        data, "metadata", lambda: {"prepared_source": "f", "identity": "fixture"}
    )
    calls = fake_endpoints(tmp_path, monkeypatch)
    a = dict(p.expected("f"), operator_note="test only")
    assert (
        a["units"] == 1152
        and a["role_documents"] == {"fit": 256, "probe": 128}
        and a["max_acquisitions"] == 1
        and a["neural_updates"] == 0
        and a["total_card_seconds"] == 1800
    )
    p.validate(a, "f")
    for k, v in [
        ("data", {}),
        ("scope", "x49-q-one-read-frozen-train-v1"),
        ("remote", True),
        ("retries", 1),
        ("ridge", 0),
        ("fitted_scalars_per_seed", {"U": 4}),
    ]:
        with pytest.raises(ValueError):
            p.validate(dict(a, **{k: v}), "f")
    d = tmp_path / p.JOB.name
    d.mkdir()
    path = tmp_path / "4902"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="weight bytes"):
        p.freeze(p.JOB, "f")
    assert not (d / "frozen.json").exists()
    path.write_bytes(b"4902")
    calls.clear()
    h = p.freeze(p.JOB, "f")
    assert h == sha(d / "frozen.json") and calls == [(s, True) for s in SEEDS]
    with pytest.raises(ValueError, match="already frozen"):
        p.freeze(p.JOB, "f")
    m = fit.Moments()
    m.documents = data.N["fit"]
    m.targets = m.documents * T
    z = m.freeze()
    p.checked_fit(z)
    z["beta"]["U"][3] = 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        p.checked_fit(z)


def test_real_tiny_fit_probe_freeze_progress_and_terminal_audit(tmp_path, monkeypatch):
    book = invented(tmp_path, monkeypatch)
    calls = fake_endpoints(tmp_path, monkeypatch)
    data.prepare()
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    dig = p.freeze(p.JOB, "fixture")
    plan = load(d / "frozen.json")
    assert calls == [(s, True) for s in SEEDS]
    start = dict(
        name=p.JOB.name,
        kind=p.JOB.kind,
        ref="fixture",
        units=p.units(),
        cap_s=p.TOTAL,
        t0=10,
        freeze_sha=dig,
    )
    write_json(d / "start.json", start)
    monkeypatch.setattr(run, "admit", lambda job: (d, "fixture", start))
    monkeypatch.setattr(
        run.panel,
        "licensed_admitted_reader",
        lambda job, where, ref, s: load(where / "frozen.json"),
    )
    pid = os.getpid()
    owner = dict(pid="88", starttime="2", gpu="0", adopted="1", created="101")
    ident = dict(
        job=p.JOB.name,
        ref="fixture",
        trainer=pid,
        trainer_start="5",
        owner=owner,
        device="fixture",
        torch="fixture",
        python="fixture",
    )
    write_json(d / "identity.json", ident)
    write_json(d / "launch.json", dict(pid=88, start="2"))
    orders = []

    def model(seed, e):
        assert e == plan["endpoints"][str(seed)]
        orders.append(seed)
        torch.manual_seed(seed)
        return Tiny().eval(), trained_base().eval()

    def progress(path, n, t, root):
        assert path == d and root == p.ROOT and t == n * T
        write_json(d / "progress.json", dict(pid=pid, units=n, scored_targets=t))
        o = dict(
            ppid=1,
            pid=88,
            start="2",
            trainer=pid,
            trainer_start="5",
            heartbeat=str(101 + n),
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
    monkeypatch.setattr(run.panel, "progress", progress)
    seen = []
    real = os.pread

    def read_probe(fd, n, off):
        assert (
            load(d / "progress.json")["units"] == 6
            and (d / "raw-probe-open-attempted").is_file()
            and set(load(d / "fit-frozen.json")["seeds"]) == set(map(str, SEEDS))
        )
        seen.append(off)
        return real(fd, n, off)

    monkeypatch.setattr(data.os, "pread", read_probe)
    seal = p.freeze_readouts

    def audited_seal(folder, initial):
        actual = seal(folder, initial)
        with pytest.raises(ValueError, match="changed bound receipt"):
            p.checked_readouts(folder, initial, "0" * 64)
        partial = run.panel.partial(p.JOB, root=p.ROOT)
        assert (
            partial["fit_freeze_sha"] == actual
            and partial["fit_stream_open_attempted"]
            and not partial["probe_raw_open_attempted"]
            and not seen
        )
        return actual

    monkeypatch.setattr(p, "freeze_readouts", audited_seal)
    z = run.evaluate(p.JOB, dict(folder=d, ref="fixture"))
    assert (
        z["units"] == 15
        and z["targets"] == 15 * T
        and z["neural_updates"] == 0
        and orders == list(SEEDS) * 2
        and seen == [r["offset"] for r in book["roles"]["probe"]]
        and calls == [(s, True) for s in SEEDS] * 3
    )
    assert not (d / "probe/windows.u8").exists()
    write_json(d / "complete.json", z)
    charge = dict(
        job=p.JOB.name,
        complete_sha=sha(d / "complete.json"),
        upper_card_s=21,
        rc=0,
        witnessed_units=15,
        renewed=True,
        inclusive_wall_s=20.0,
        owner_and_waited_children_cpu_s=12.0,
    )
    write_json(d / "charge.json", charge)
    write_json(p.ROOT / "charged.json", {p.JOB.name: charge})
    (d / "watch-ack").touch()
    for key, hb in [("initial", 101), ("after", 116), ("last", 116)]:
        (d / key).write_text(
            "".join(
                k + "=" + v + "\n" for k, v in dict(owner, heartbeat=str(hb)).items()
            )
        )
    (d / "released").write_text(
        "rc=0\nreleased_epoch=121\nend_monotonic_ns=20000000010\n"
    )
    out = analysis.analyze(p.ROOT)
    assert (
        out["cost"]["witnessed_fit_boundary"] == 6
        and out["stages"]["probe"]["real_targets_per_seed"] == 3 * T
        and set(out["stages"]["probe"]["policies"]) == set("BNUS")
    )
    assert out["beta"]["4902"] == load(d / "fit-4902.json")["state"]["beta"]
    with pytest.raises(ValueError, match="repeated U transition"):
        data.extract("probe", job=p.JOB, fit_sha=z["fit_freeze_sha"])
    f = d / "fit-4902.json"
    bad = load(f)
    bad["state"]["rhs"]["S"][3] += 100
    write_json(f, bad)
    with pytest.raises(ValueError, match="inconsistent"):
        analysis.analyze(p.ROOT)


def test_document_bootstrap_signs_independent_seed_reduction_and_pair_refusal():
    n = 6
    hs = list(map(str, range(n)))
    logs = {s: [] for s in SEEDS}
    times = {
        k: 0.1
        for k in (
            "prefill_s",
            "base_B_s",
            "shared_current_s",
            "shared_US_mark_build_s",
            "U_mark_read_s",
            "S_rotation_and_mark_read_s",
            "response_N_s",
            "response_U_s",
            "response_S_s",
        )
    }
    for seed, items in logs.items():
        for i, h in enumerate(hs):
            hit = 100 + i
            w = dict(
                literal_hits=hit,
                literal_misses=T - hit,
                literal_queries=T,
                literal_value_writes=T,
                mark_storage_bytes=T,
                literal_value_reads=hit,
                U_mark_reads=hit,
                S_mark_reads=hit,
                arrivals_encoded=T - 1,
                mark_norms_from_current=hit,
                mark_norms_extra=T - 1 - hit,
                marks_saturated=3,
                B_candidate_code_saturated=1,
                now_candidate_code_saturated=1,
                U_read_code_saturated=0,
                S_read_code_saturated=0,
                S_slots_displaced=25,
                S_codes_changed=(i + 1 if i else 0),
            )
            loss = dict(
                B=T + i, N=T + i * i - 0.1, U=T - i / 2 + 0.3 * (seed - 4900), S=T + i
            )
            items.append(
                dict(
                    document_sha=h,
                    targets=T,
                    host_prefills=1,
                    candidate_correct=3,
                    nll_sums=loss,
                    work=w,
                    seconds=times,
                )
            )
    z = analysis.summarize("probe", logs, hs)
    ix = np.random.default_rng(4907).integers(0, n, (10000, n))
    assert z["documents_with_support_by_seed"]["4902"]["changed_code"] == 5
    for a, b in analysis.PAIRS:
        y = []
        means = []
        for s in SEEDS:
            v = np.array([x["nll_sums"][b] - x["nll_sums"][a] for x in logs[s]]) / T
            y.append(v[ix].mean(1))
            means.append(v.mean())
        k = z["contrasts"][a + ":" + b]
        np.testing.assert_allclose(k["by_seed"], means, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            k["ci95"],
            np.percentile(np.mean(y, axis=0), [2.5, 97.5]),
            rtol=0,
            atol=1e-14,
        )
        assert k["gain"] == pytest.approx(np.mean(means))
    with pytest.raises(ValueError, match="role or pairing"):
        analysis.summarize("probe", logs, hs[::-1])
    logs[4903][2]["work"]["S_codes_changed"] = 99
    with pytest.raises(ValueError, match="foreign mark work"):
        analysis.summarize("probe", logs, hs)
