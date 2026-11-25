"""No natural text, real coefficients or accelerator. Test the exact frozen short-final lifecycle."""

from copy import deepcopy
import json
import os
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    confirm_data as data,
    confirm_analysis as ana,
    surprise_fit as fit,
    short_phase,
    test_joint_reader as joint_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_eval import _documents
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, digest
from experiments.expressivity_tasks.x49_live_trace_memory.write_data import choose_short
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)


def test_short_rank_suffix_dedup_and_minimal_crossing():
    records = []
    start = 0
    for i in range(48):
        k = i if i != 30 else 8
        h = f"{100 * k + 1:064x}"
        n = 22 + k
        records.append(
            dict(
                split="final_test",
                byte_offset=start,
                document=start,
                byte_length=n,
                sha256=h,
            )
        )
        start += n
    ex = {records[7]["sha256"]}
    old, _, _ = choose_short(records, ex, 231, min_offset=0)
    spent = ex | {x["sha256"] for x in old}
    goal = 145
    rows, n, nt = data.next_short(records, spent, old, goal, lo=0)
    clean = {}
    for r in records:
        if r["sha256"] not in ex:
            clean.setdefault(r["sha256"], r)
    ordered = sorted(clean, key=lambda h: digest(b"x49-w-short-v1:" + h.encode()))
    b = len(old)
    assert (
        [x["sha256"] for x in rows] == ordered[b : b + len(rows)]
        and n == len(ordered) - b
        and nt == sum(clean[h]["byte_length"] - 1 for h in ordered[b:])
    )
    size = sum(x["length"] - 1 for x in rows)
    assert size >= goal > size - rows[-1]["length"] + 1
    with pytest.raises(ValueError, match="next segment"):
        data.next_short(records, spent, old[::-1], goal, lo=0)
    with pytest.raises(ValueError, match="missing W"):
        data.next_short(records, ex, old, goal, lo=0)
    with pytest.raises(ValueError, match="insufficient"):
        data.next_short(records, spent, old, nt + 1, lo=0)


binary = joint_fixture.binary


def fixture(tmp, mp, name="UC"):
    data, p, run, _ = short_phase.study(name)
    source_panel = p.prior if name == "UC" else p.ls
    mp.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.delenv("GPU_LEASE_HELD", raising=False)
    mp.setattr(data, "BASE", tmp / "reserved")
    mp.setattr(data, "MIN_OFFSET", 0)
    mp.setattr(p, "ROOT", tmp / "run")
    mp.setattr(source_panel, "ROOT", tmp / "prior")
    mp.setattr(run, "AUTH", tmp / "authority")
    mp.setattr(run, "checked_fixture_ref", lambda: "fixture")
    src = tmp / "raw"
    rows = []
    i = 0
    with src.open("xb") as f:
        for n in p.LENGTHS:
            while True:
                unit = i.to_bytes(2, "little") + b"abc\0abc\xff\x1e"
                i += 1
                word = (unit * (n // len(unit) + 2))[: n + 1]
                h = digest(word)
                if int(h, 16) % 100 == (1 if name == "UC" else 0):
                    break
            off = f.tell()
            f.write(word)
            rows.append(
                dict(
                    doc=off,
                    offset=off,
                    start=0,
                    split=p.JOB.kind,
                    length=n + 1,
                    sha256=h,
                )
            )
    st = src.stat()
    rows.sort(
        key=lambda r: digest(b"x49-w-short-v1:" + r["sha256"].encode())
        if name == "UC"
        else data.rank(r["sha256"])
    )
    mp.setattr(data, "GOAL", sum(p.LENGTHS))
    book = dict(
        spec=data.SPEC,
        source=dict(
            path=str(src), size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino
        ),
        excluded_unique=99,
        eligible_documents=7,
        eligible_targets=sum(p.LENGTHS) + 100,
        rows=rows,
    )
    mp.setattr(data, "selection", lambda: deepcopy(book))
    files = {}
    beta = {}
    calls = []
    for s in SEEDS:
        path = tmp / str(s)
        path.write_bytes(str(s).encode())
        k = str(s)
        files[k] = sha(path)
        beta[k] = {
            a: (np.arange(d) * 0.1 + s * 0.00001).tolist() for a, d in p.DIMS.items()
        }
    parent = dict(beta=beta, freeze_sha="fixture-base", checkpoints=files)
    if name == "JT":
        parent["u_beta"] = {k: [0.2, -0.4, 0.1, int(k) * 0.00002] for k in beta}
        mp.setattr(
            p.local_u,
            "contract",
            lambda s: (files[str(s)], tuple(parent["u_beta"][str(s)])),
        )
        mp.setattr(
            p.joint, "contract", lambda s: (files[str(s)], tuple(beta[str(s)]["J"]))
        )
    d = source_panel.ROOT / source_panel.JOB.name
    d.mkdir(parents=True)
    sealed = d / "fit-frozen.json"
    write_json(
        sealed,
        dict(
            plan_sha="fixture-base",
            spec=data.u.SPEC if name == "UC" else data.ls.SPEC,
            seeds={k: dict(beta=b, checkpoint_sha=files[k]) for k, b in beta.items()},
        ),
    )
    mp.setattr(data, "FIT_SHA", sha(sealed))
    mp.setattr(data, "parent", lambda: parent)

    def ck(s, *, hash_weights):
        k = str(s)
        path = tmp / k
        calls.append((s, hash_weights))
        if hash_weights and sha(path) != files[k]:
            raise ValueError("foreign weights")
        return path, dict(sha256=files[k])

    mp.setattr(p.prior, "checkpoint", ck)
    return book, parent, calls


@pytest.mark.parametrize("name", ["UC", "JT"])
def test_metadata_only_same_fits_exact_weights_and_no_early_open(
    tmp_path, monkeypatch, name
):
    data, p, run, _ = short_phase.study(name)
    book, parent, calls = fixture(tmp_path, monkeypatch, name)
    reads = []
    monkeypatch.setattr(os, "pread", lambda *args: reads.append(args))
    r = data.reserve()
    assert r["payload_documents_opened"] == 0 and not reads and r["documents"] == 5
    with pytest.raises(FileExistsError):
        data.reserve()
    with pytest.raises(
        ValueError, match="separate short" if name == "UC" else "separate JT"
    ):
        _documents(p.JOB, {})
    d = p.ROOT / p.JOB.name
    d.mkdir(parents=True)
    a = dict(p.expected("fixture"), operator_note="fixture only")
    p.validate(a, "fixture")
    assert (
        a["new_fits"] == a["neural_updates"] == 0
        and a["total_card_seconds"] == 1200
        and a["max_acquisitions"] == 1
        and a["units"] == 15
    )
    for key, v in [
        ("ref", "another"),
        ("fit_freeze_sha", "0" * 64),
        ("data", {}),
        ("new_fits", 1),
        ("remote", True),
    ]:
        with pytest.raises(ValueError):
            p.validate(dict(a, **{key: v}), "fixture")
    path = tmp_path / "4902"
    path.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="foreign weights"):
        p.freeze(p.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    path.write_bytes(b"4902")
    parent["beta"]["4902"]["S" if name == "UC" else "J"][3] += 0.01
    with pytest.raises(ValueError, match="coefficient mismatch"):
        p.freeze(p.JOB, "fixture")
    assert not reads and not (d / "source-open-attempted").exists()
    bad = deepcopy(book)
    bad["rows"][0]["start"] = 1
    with pytest.raises(
        ValueError,
        match="short final membership"
        if name == "UC"
        else "short calibration membership",
    ):
        data.checked_rows(bad)
    if name == "JT":
        bad = deepcopy(book)
        bad["rows"][0]["split"] = "final_test"
        with pytest.raises(ValueError, match="calibration membership"):
            data.checked_rows(bad)
        with pytest.raises(ValueError, match="wrong frozen short reader"):
            p.fresh_descriptor(p.old.JOB, {})


@pytest.mark.parametrize("name", ["UC", "JT"])
def test_full_invented_frozen_evaluation_shape_guard_real_target_witness_and_closure(
    tmp_path, monkeypatch, name, request
):
    data, p, run, ana = short_phase.study(name)
    book, parent, calls = fixture(tmp_path, monkeypatch, name)
    if name == "JT":
        _, _, writer = request.getfixturevalue("binary")
        monkeypatch.setattr(p.joint, "_writer", lambda: writer)
        monkeypatch.setattr(run, "WIDTH", 8)
    r = data.reserve()
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
        units=initial["units"],
        freeze_sha=h,
    )
    write_json(d / "start.json", start)
    monkeypatch.setattr(run, "admit", lambda j: (d, "fixture", start))
    monkeypatch.setattr(run.panel, "licensed_admitted_split", lambda *args: initial)
    calls.clear()
    seen = []
    read = os.pread

    def pread(fd, n, off):
        assert (d / "source-open-attempted").is_file() and load(d / "shape-check.json")[
            "plan_sha"
        ] == h
        seen.append(off)
        return read(fd, n, off)

    monkeypatch.setattr(os, "pread", pread)
    with pytest.raises(ValueError, match="missing short device check"):
        _documents(p.JOB, initial)
    assert not (d / "source-open-attempted").exists() and not seen
    expected = p.targets_for(p.JOB, d, start)
    lens = [v["length"] - 1 for v in book["rows"]]
    assert [expected(i) for i in range(16)] == [0, *map(int, np.cumsum(lens * 3))]
    pid = os.getpid()
    owner = dict(pid="80", starttime="10", gpu="0", created="1000", adopted="1")
    ident = dict(
        ref="fixture",
        job=p.JOB.name,
        trainer=pid,
        trainer_start="21",
        owner=owner,
        device="fixture",
        python="fixture",
        torch="fixture",
    )
    write_json(d / "identity.json", ident)
    write_json(d / "launch.json", dict(pid=80, start="10"))
    models = []

    def model(s, e):
        assert e == dict(sha256=parent["checkpoints"][str(s)])
        models.append(s)
        torch.manual_seed(s)
        return Tiny().eval(), trained_base().eval()

    def prog(where, n, nt, root):
        assert where == d and root == p.ROOT and nt == expected(n)
        write_json(d / "progress.json", dict(pid=pid, units=n, scored_targets=nt))
        o = dict(
            pid=80,
            ppid=1,
            start="10",
            trainer=pid,
            trainer_start="21",
            heartbeat=str(1000 + n),
            units=n,
        )
        with (d / "survival.jsonl").open("a") as f:
            f.write(json.dumps(o) + "\n")
        if n == 2:
            write_json(d / "live.json", o)

    # Both studies share the frozen host loader; only their no-fit readers differ.
    monkeypatch.setattr(short_phase.study("UC")[2].u, "model", model)
    if name == "UC":
        monkeypatch.setattr(
            run,
            "short_frozen_document",
            lambda *args, beta: fit.short_frozen_document(*args, beta=beta, width=8),
        )
    monkeypatch.setattr(run.panel, "progress", prog)
    z = run.evaluate(p.JOB, dict(ref="fixture", folder=d))
    assert (
        models == list(SEEDS)
        and len(seen) == 5
        and seen == [x["offset"] for x in book["rows"]]
        and z["units"] == 15
        and z["targets"] == 3 * sum(lens)
        and z["new_fits"] == 0
        and calls == [(s, True) for s in SEEDS] * 2
    )
    write_json(d / "complete.json", z)
    c = dict(
        job=p.JOB.name,
        rc=0,
        upper_card_s=30,
        complete_sha=sha(d / "complete.json"),
        renewed=True,
        witnessed_units=15,
        inclusive_wall_s=29.0,
        owner_and_waited_children_cpu_s=13.0,
    )
    write_json(d / "charge.json", c)
    write_json(p.ROOT / "charged.json", {p.JOB.name: c})
    (d / "watch-ack").touch()
    for key, hb in [("initial", 1000), ("after", 1016), ("last", 1016)]:
        (d / key).write_text(
            "".join(
                k + "=" + v + "\n" for k, v in dict(owner, heartbeat=str(hb)).items()
            )
        )
    (d / "released").write_text(
        "rc=0\nreleased_epoch=1029\nend_monotonic_ns=29000000100\n"
    )
    p.finish(p.ROOT)
    write_json(
        p.ROOT / "done.json",
        dict(
            complete=True,
            summary_sha=sha(p.ROOT / "summary.json"),
            charges_sha=sha(p.ROOT / "charged.json"),
            driver_and_short_parents_cpu_s=1.0,
        ),
    )
    write_json(run.AUTH, dict(p.expected("fixture"), operator_note="fixture"))
    out = ana.report()
    assert (
        out["analysis"]["real_targets_per_seed"] == r["targets"]
        and out["by_length_per_seed"]["1-15"]
        == dict(documents=1, real_targets=1, hits=0, misses=1)
        and out["verified"]["independent_paired_bootstrap"]
    )
    assert out["cost"]["witnessed_fit_boundary"] is None and out["predictors"]["4901"][
        "beta"
    ] == (
        parent["beta"]["4901"]
        if name == "UC"
        else dict(
            U=parent["u_beta"]["4901"],
            H=parent["beta"]["4901"]["H"],
            J=parent["beta"]["4901"]["J"],
        )
    )
    if name == "JT":
        assert (
            out["verified"]["independent_branch_cost_reduction"]
            and set(out["analysis"]["contrasts"]) == {"J:U", "J:H", "J:B", "U:B", "H:U"}
            and out["writer_cost_sha"] == data.COST_SHA
        )
    with pytest.raises(FileExistsError):
        _documents(p.JOB, initial)
    path = d / "4903.jsonl"
    path.write_bytes(
        path.read_bytes().replace(b'"document_sha"', b'"document_sha" ', 1)
    )
    with pytest.raises(ValueError, match="scored endpoint"):
        ana.report()


def test_variable_denominator_and_conjunctive_decision():
    counts = [3, 25, 190, 3500]
    hs = list(map(str, range(4)))
    logs = {s: [] for s in SEEDS}
    secs = {
        k: 0.01
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
    for s in SEEDS:
        for i, n in enumerate(counts):
            hit = int(n > 17)
            w = dict(
                literal_hits=hit,
                literal_misses=n - hit,
                literal_queries=n,
                literal_value_writes=n,
                literal_value_reads=hit,
                mark_storage_bytes=n,
                U_mark_reads=hit,
                S_mark_reads=hit,
                arrivals_encoded=n - 1,
                mark_norms_extra=n - 1 - hit,
                mark_norms_from_current=hit,
                marks_saturated=0,
                B_candidate_code_saturated=0,
                now_candidate_code_saturated=0,
                U_read_code_saturated=0,
                S_read_code_saturated=0,
                S_slots_displaced=0,
                S_codes_changed=0,
            )
            gain = [0, 3, 4, -10][i] * (s - 4900)
            base = n * 2
            logs[s].append(
                dict(
                    document_sha=hs[i],
                    host_prefills=1,
                    targets=n,
                    candidate_correct=0,
                    work=w,
                    seconds=secs,
                    nll_sums=dict(B=base, N=base, S=base, U=base - gain),
                )
            )
    z = ana.u.summarize("final_test", logs, hs, short_targets=counts)
    r = z["contrasts"]["U:N"]
    delta = np.array([0, 3, 4, -10]) * 2
    assert r["gain"] == pytest.approx(delta.sum() / sum(counts)) and r[
        "gain"
    ] < 0 < np.mean(delta / counts)
    ix = np.random.default_rng(4908).integers(0, 4, (10000, 4))
    np.testing.assert_allclose(
        r["ci95"],
        np.percentile(delta[ix].sum(1) / np.array(counts)[ix].sum(1), [2.5, 97.5]),
        rtol=0,
        atol=1e-15,
    )
    assert not ana.decision(z)["recommend_local_frozen_U"]
    good = {
        "contrasts": {
            k: dict(gain=1.0, ci95=[0.1, 2.0], by_seed=[1, 1, 1])
            for k in ("U:N", "U:S", "U:B")
        }
    }
    assert ana.decision(good)["recommend_local_frozen_U"]
    good["contrasts"]["U:S"]["by_seed"][2] = -0.01
    assert not ana.decision(good)["recommend_local_frozen_U"]
    good["contrasts"]["U:S"]["by_seed"][2] = 1
    good["contrasts"]["U:B"]["ci95"][0] = 0
    assert not ana.decision(good)["recommend_local_frozen_U"]
    with pytest.raises(ValueError):
        ana.u.summarize("final_test", logs, hs, short_targets=counts[::-1])
    with pytest.raises(ValueError):
        fit.short_frozen_document(None, None, b"\0" * 4097, beta={})
