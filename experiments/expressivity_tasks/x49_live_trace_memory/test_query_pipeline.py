"""Q integration: invented data and frozen predictors. Nothing here selects a real document or takes a lease."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
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
    query_data as data,
    query_panel as panel,
    query_run as run,
    query_analysis as analysis,
    query_score as qs,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import T, SEEDS, select
from experiments.expressivity_tasks.x49_live_trace_memory.test_query_depth import (
    example,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny


def test_selection_is_exact_unused_rank_suffix_and_never_reuses_a_fresh_offset():
    records = []
    for i in range(20):
        off = MIN_OFFSET + 8192 * i
        k = i if i != 17 else 4
        h = f"{100 * k + 3:064x}"
        records.append(
            dict(
                document=off, byte_offset=off, byte_length=4097, sha256=h, split="train"
            )
        )
    size = records[-1]["byte_offset"] + 4097
    ex = {records[4]["sha256"]}
    prior = {
        "train": select(records, ex, {"train": 4}, size)[0]["train"],
        "final_test": [],
    }
    out, all_ex, left = data.select_new(
        records, base_ex=ex, prior=prior, quota=5, size=size
    )
    assert left == 14 and len(all_ex) == 5 and len(out) == 5
    hs = sorted(
        {r["sha256"] for r in records} - all_ex,
        key=lambda h: sha256(b"x49-live-v1:" + h.encode()).digest(),
    )
    assert [x["sha256"] for x in out] == hs[:5]
    with pytest.raises(ValueError, match="suffix"):
        data.select_new(
            records, base_ex=ex, prior={"train": prior["train"][1:]}, quota=5, size=size
        )
    with pytest.raises(ValueError, match="exclusion union"):
        data.select_new(
            records,
            base_ex=ex | {prior["train"][0]["sha256"]},
            prior=prior,
            quota=5,
            size=size,
        )


def test_single_train_only_reservation_preparation_and_checked_shifts(
    tmp_path, monkeypatch
):
    dst = tmp_path / "diag"
    src = tmp_path / "raw"
    rows = []
    i = 0
    monkeypatch.setattr(data, "BASE", dst)
    monkeypatch.setattr(data, "N", 3)
    monkeypatch.setattr(data, "MIN_OFFSET", 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GPU_LEASE_HELD", raising=False)
    with src.open("wb") as f:
        while len(rows) < 3:
            raw = ((i.to_bytes(4, "little") + b"text\0ab\x1e\xff") * 700)[:8193]
            i += 1
            h = sha256(raw).hexdigest()
            if int(h, 16) % 100 < 2:
                continue
            off = f.tell()
            f.write(raw)
            rows.append(
                dict(
                    split="train",
                    doc=off,
                    offset=off,
                    length=len(raw),
                    start=T * (int(h[16:32], 16) % 2),
                    sha256=h,
                )
            )
    st = src.stat()
    m = dict(
        source=dict(
            path=str(src), size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino
        ),
        rows=rows,
        excluded_unique=23,
        eligible_train=8,
    )
    monkeypatch.setattr(data, "selection", lambda: m)
    monkeypatch.setattr(
        data,
        "fixture_ref",
        lambda: (_ for _ in ()).throw(ValueError("unqualified fixture")),
    )
    with pytest.raises(ValueError, match="unqualified"):
        data.prepare()
    assert not dst.exists()
    monkeypatch.setattr(data, "fixture_ref", lambda: "fixture")
    p = data.prepare()
    meta = data.metadata()
    assert (
        p["rows_sha"] == meta["rows_sha"] == sha(dst / "reservation.json")
        and p["source_train_bytes"] == st.st_size
        and p["targets"] == 3 * 4096
        and p["calibration_or_test_documents_opened"] == 0
    )
    d = dst / "stream"
    ident = load(d / "identity.json")
    hs = load(d / "shifts.json")
    payload = (d / "windows.u8").read_bytes()
    got = data.verify_payload(rows, ident, hs, payload)
    raw = src.read_bytes()
    for (h, w), r in zip(got, rows, strict=True):
        lo = r["offset"] + r["start"]
        assert (
            h == r["sha256"]
            and w[:-1] == raw[lo : lo + T]
            and w[1:] == raw[lo + 1 : lo + T + 1]
        )
    with pytest.raises(ValueError):
        data.verify_payload(rows[::-1], ident, hs, payload)
    with pytest.raises(ValueError):
        data.verify_payload(rows, ident, hs, payload[:-1])
    with pytest.raises(ValueError):
        data.verify_payload(
            rows, ident, [dict(hs[0], targets="foreign"), *hs[1:]], payload
        )
    with pytest.raises(ValueError):
        data.verify_payload(
            [dict(rows[0], split="final_test"), *rows[1:]], ident, hs, payload
        )
    with pytest.raises(ValueError):
        data.verify_payload([rows[0], rows[0], rows[2]], ident, hs, payload)
    probe_root = tmp_path / "stage"
    stage = probe_root / panel.JOB.name
    stage.mkdir(parents=True)
    frozen = {"data": meta}
    monkeypatch.setattr(panel, "ROOT", probe_root)
    monkeypatch.setattr(run, "admit", lambda j: (stage, "fixture", {}))
    monkeypatch.setattr(run.p, "licensed_admitted_reader", lambda *args: frozen)

    def freshly_bound(ref, *, hash_weights):
        assert hash_weights and ref == "fixture"
        return frozen

    monkeypatch.setattr(panel, "plan", freshly_bound)
    assert data.diagnostic_windows(panel.JOB) == got
    with pytest.raises(FileExistsError):
        data.diagnostic_windows(panel.JOB)
    with pytest.raises(FileExistsError):
        data.prepare()


def test_stream_refuses_foreign_or_unadmitted_reader_before_payload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run, "AUTH", tmp_path / "no-budget")
    monkeypatch.setattr(data, "BASE", tmp_path / "no-payload")
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda path: SimpleNamespace(free=34 << 30)
    )
    with pytest.raises(ValueError, match="wrong Q reader"):
        data.diagnostic_windows(None)
    with pytest.raises(ValueError, match="no separate Q"):
        data.diagnostic_windows(panel.JOB)
    d = tmp_path / "job"
    d.mkdir()
    monkeypatch.setattr(run, "admit", lambda j: (d, "fixture", {}))
    monkeypatch.setattr(
        run.p, "licensed_admitted_reader", lambda *args: {"foreign": "freeze"}
    )
    monkeypatch.setattr(panel, "plan", lambda ref, hash_weights: {"ref": ref})
    with pytest.raises(ValueError, match="unlicensed"):
        data.diagnostic_windows(panel.JOB)
    assert not list(d.iterdir()) and not data.BASE.exists()


def test_q_cap_and_three_frozen_checkpoint_hashes_before_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    monkeypatch.setattr(data, "metadata", lambda: {"identity_sha": "private-checked"})
    calls = []

    def ck(s, *, hash_weights):
        p = tmp_path / str(s)
        note = {"sha256": sha256(str(s).encode()).hexdigest()}
        calls.append((s, hash_weights))
        if hash_weights and sha(p) != note["sha256"]:
            raise ValueError("changed bytes")
        return p, note

    for s in SEEDS:
        (tmp_path / str(s)).write_bytes(str(s).encode())
    monkeypatch.setattr(panel, "checkpoint", ck)
    a = dict(panel.expected("fixture"), operator_note="synthetic only")
    assert (
        a["units"] == 384
        and a["total_card_seconds"] == 1800
        and a["max_acquisitions"] == 1
        and a["updates"] == 0
    )
    panel.validate(a, "fixture")
    for k, v in [
        ("data", {}),
        ("scope", "x49-read-at-writer-local-v1"),
        ("remote", True),
        ("updates", 1),
        ("split", "final_test"),
    ]:
        with pytest.raises(ValueError):
            panel.validate(dict(a, **{k: v}), "fixture")
    d = tmp_path / panel.JOB.name
    d.mkdir()
    p = tmp_path / "4903"
    p.write_bytes(b"not-an-endpoint")
    with pytest.raises(ValueError, match="changed bytes"):
        panel.freeze(panel.JOB, "fixture")
    assert not (d / "frozen.json").exists()
    p.write_bytes(b"4903")
    calls.clear()
    digest = panel.freeze(panel.JOB, "fixture")
    assert (
        digest == sha(d / "frozen.json")
        and calls == [(s, True) for s in SEEDS]
        and load(d / "frozen.json")["units"] == 384
    )
    with pytest.raises(ValueError, match="already frozen"):
        panel.freeze(panel.JOB, "fixture")


def test_full_synthetic_window_shared_prefill_same_shift_and_all_byte_strata():
    torch.manual_seed(702)
    m = Tiny().eval()
    b = trained_base().eval()
    s, _ = example()
    w = (s * 12)[:4097]
    assert len(w) == 4097
    v = qs.window_score(m, b, w, seed=4901, rank=0, width=8)
    v2 = qs.window_score(
        m, b, w[:-1] + bytes([255 - w[-1]]), seed=4901, rank=1, width=8
    )
    assert v["order"] == "JEM" and v2["order"] == "EMJ" and v["strata"] == v2["strata"]
    counts = [v["strata"][k]["targets"] for k in qs.STRATA]
    assert (
        counts[0] > 0
        and counts[3] > 0
        and sum(counts) == 4096
        and v["host_prefills"] == 1
    )
    with torch.no_grad():
        h, z = m(torch.tensor([list(w[:-1])]))[0].split((8, 256), -1)
        lp = b(w[:-1], h, z, mode="B")
        assert (
            -lp[torch.arange(4096), list(w[1:])].double().sum().item()
            == v["cells"]["J"]["nll_sum"]
        )
        assert v2["cells"]["J"]["nll_sum"] - v["cells"]["J"][
            "nll_sum"
        ] == pytest.approx(
            float(lp[-1, w[-1]].double() - lp[-1, 255 - w[-1]].double()), abs=1e-9
        )
    for a in qs.POLICIES:
        c = v["cells"][a]
        assert (
            c["nll_sum"]
            == pytest.approx(sum(c["stratum_nll_sums"].values()), abs=1e-10)
            and c["work"] == v2["cells"][a]["work"]
        )
        assert (
            c["work"]["literal_misses"] == counts[0]
            and c["work"]["literal_value_reads"] == 4096 - counts[0]
        )
    with pytest.raises(ValueError):
        qs.window_score(m, b, w[:-1], seed=4901, rank=0, width=8)


def synth(seed, i):
    counts = [97, 3100, 830 + i, 69 - i]
    st = {
        k: {"targets": n, "geometry": {}}
        for k, n in zip(qs.STRATA, counts, strict=True)
    }
    cells = {}
    for a in qs.POLICIES:
        parts = {k: float(n) for k, n in zip(qs.STRATA, counts, strict=True)}
        if a in "EM":
            parts["move_unmatched"] += i - 0.9
            parts["move_matched"] -= 0.03 * (seed - 4900)
        if a == "M":
            parts["move_matched"] += 0.9 * i - 1
        w = dict(
            literal_hits=T - counts[0],
            literal_misses=counts[0],
            literal_queries=T,
            literal_value_reads=T - counts[0],
            literal_value_writes=T,
        )
        if a != "J":
            w.update(
                extended_moves=sum(counts[2:]),
                matched_controls=counts[3] if a == "M" else 0,
            )
        cells[a] = dict(
            work=w,
            policy_s=1.5,
            extra_address_s=0.2 if a != "J" else 0,
            nll_sum=sum(parts.values()),
            stratum_nll_sums=parts,
        )
    return dict(
        document_sha=str(i),
        order=qs.cycle(seed, i),
        host_prefills=1,
        host_prefill_s=0.5,
        targets=T,
        cells=cells,
        strata=st,
    )


def test_doc_paired_bootstrap_sign_geometry_and_refusals():
    n = 7
    hashes = list(map(str, range(n)))
    logs = {s: [synth(s, i) for i in range(n)] for s in SEEDS}
    v = analysis.aggregate(logs, hashes)
    assert v["policy_order_counts"] == {k: [7, 7, 7] for k in "JEM"}
    draw = np.random.default_rng(4906).integers(0, n, size=(10000, n))
    for a, b in analysis.PAIRS:
        ds = np.asarray(
            [
                [x["cells"][b]["nll_sum"] - x["cells"][a]["nll_sum"] for x in logs[s]]
                for s in SEEDS
            ]
        )
        g = v["contrasts"][a + ":" + b]
        np.testing.assert_allclose(g["by_seed"], ds.mean(axis=1) / T)
        np.testing.assert_allclose(
            g["ci95"],
            np.percentile(
                np.mean([d[draw].mean(1) / T for d in ds], axis=0), [2.5, 97.5]
            ),
        )
        assert g["gain"] == pytest.approx(
            sum(i["contribution_to_all_byte_gain"] for i in g["strata"].values())
        )
    with pytest.raises(ValueError, match="document/order"):
        analysis.aggregate(logs, hashes[::-1])
    bad = deepcopy(logs)
    bad[4901][0]["cells"]["M"]["stratum_nll_sums"]["hit_stay"] += 1
    bad[4901][0]["cells"]["M"]["nll_sum"] += 1
    with pytest.raises(ValueError, match="without a distinct"):
        analysis.aggregate(bad, hashes)


def test_evaluate_frozen_injection_unit_witness_and_zero_updates(tmp_path, monkeypatch):
    n = 4
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    monkeypatch.setattr(data, "BASE", tmp_path / "data")
    data.BASE.mkdir()
    write_json(
        data.BASE / "reservation.json",
        dict(rows=[{"sha256": str(i)} for i in range(n)]),
    )
    monkeypatch.setattr(data, "N", n)
    d = tmp_path / panel.JOB.name
    d.mkdir()
    f = d / "frozen.json"
    plan = dict(
        ref="f",
        units=n * 3,
        horizon=T,
        updates=0,
        data={},
        endpoints={str(s): dict(sha256=str(s)) for s in SEEDS},
    )
    write_json(f, plan)
    write_json(d / "start.json", dict(freeze_sha=sha(f), units=n * 3))
    monkeypatch.setattr(panel, "plan", lambda ref, hash_weights: plan)
    seen = []
    wins = [(str(i), bytes(T + 1)) for i in range(n)]
    monkeypatch.setattr(data, "diagnostic_windows", lambda j: wins)
    models = []

    class Model:
        def to(self, dev):
            assert dev == "cuda"
            return self

    def module(seed, e):
        assert e == plan["endpoints"][str(seed)]
        models.append(seed)
        return Model(), None

    monkeypatch.setattr(panel, "module", module)

    def scored(m, c, word, *, seed, rank):
        x = synth(seed, rank)
        del x["document_sha"]
        return x

    monkeypatch.setattr(run, "window_score", scored)
    monkeypatch.setattr(
        run.p, "progress", lambda d, n, t, root: seen.append((n, t, root))
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    for k in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, k, lambda: 0)
    z = run.evaluate(panel.JOB, dict(folder=d, ref="f"))
    assert (
        models == list(SEEDS)
        and seen == [(i, i * T, panel.ROOT) for i in range(1, n * 3 + 1)]
        and z["targets"] == n * 3 * T
        and z["updates"] == 0
    )
    for s in SEEDS:
        path = d / f"{s}.jsonl"
        assert z["endpoints"][str(s)]["scores_sha"] == sha(path) and [
            json.loads(v)["document_sha"] for v in path.read_text().splitlines()
        ] == [k for k, _ in wins]
    write_json(d / "complete.json", z)
    one = dict(
        job=panel.JOB.name,
        rc=0,
        upper_card_s=27,
        renewed=True,
        complete_sha=sha(d / "complete.json"),
        witnessed_units=n * 3,
    )
    write_json(d / "charge.json", one)
    write_json(tmp_path / "charged.json", {panel.JOB.name: one})
    ini = "pid=5\nstarttime=20\ngpu=0\ncreated=101\nadopted=1\n"
    (d / "initial").write_text(ini + "heartbeat=101\n")
    (d / "after").write_text(ini + "heartbeat=123\n")
    (d / "released").write_text("rc=0\nreleased_epoch=127\n")
    (d / "watch-ack").touch()
    out = analysis.analyze(tmp_path)
    assert (
        out["analysis"]["real_targets_per_seed_and_cell"] == n * T
        and out["charge"] == one
    )
    (d / "4902.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="wrong Q scored endpoint"):
        analysis.analyze(tmp_path)
