"""W integration fixtures; no lease, natural corpus, historic checkpoints or recorded final scores."""

from functools import partial
from hashlib import sha256
import json
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    panel,
    panel_eval as ev,
    panel_train as tr,
    write_analysis as wa,
    write_panel as w,
    write_run as run,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import (
    milestone,
    score_window,
    short_cells,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_analysis import (
    read_stage,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)
from experiments.expressivity_tasks.x49_live_trace_memory.write_head import WriteHead


def test_w_binding_schedule_and_guarded_first_update(tmp_path, monkeypatch):
    ref = "fixture"
    a = dict(w.expected(ref), operator_note="fixture")
    w.validate(a, ref)
    assert list(w.JOBS) == [f"w{k}-{s}" for s in w.SEEDS for k in ("B", "P", "L")] + [
        "w-final"
    ]
    assert sum(a["job_caps"].values()) == 18900 and a["max_acquisitions"] == 10
    assert len({w.binding(ref, "B", s)["origin"] for s in w.SEEDS}) == 3
    for key, v in [
        ("scope", "x49-relative-credit-local-v1"),
        ("train_identity", "v2"),
        ("remote", True),
        ("final_targets_per_cell", 561152),
        ("qualified_ref", "before-W"),
    ]:
        with pytest.raises(ValueError):
            w.validate(dict(a, **{key: v}), ref)
    with pytest.raises(ValueError):
        w.resolve("P", 4901, 1023, ref)
    monkeypatch.setattr(w, "ROOT", tmp_path)
    d = tmp_path / "wB-4902"
    d.mkdir()
    r = dict(
        step=1,
        targets=4096,
        preupdate_nll=1.0,
        head_lr=0.1,
        host_lr=0.01,
        work={"literal_hits": 30},
    )
    (d / "steps.jsonl").write_text(json.dumps(r) + "\n")
    write_json(
        d / "complete.json",
        dict(binding=w.binding(ref, "B", 4902), steps_sha256=sha(d / "steps.jsonl")),
    )
    j = w.JOBS["wL-4902"]
    run.guard_first(j, ref, dict(r, work=dict(r["work"], writer_old_value_reads=30)))
    for rr in [dict(r, preupdate_nll=2.0), dict(r, work={"literal_hits": 29})]:
        with pytest.raises(ValueError, match="unmatched initial"):
            run.guard_first(j, ref, rr)


def binary_final(n, pos):
    # Invent binary documents with the real SHA residue; no path into a real corpus.
    for i in range(65536):
        b = (i.to_bytes(2, "little") + (b"a\xff:zzz\0\x1eb" * 460))[:n]
        h = sha256(b).hexdigest()
        if int(h, 16) % 100 == 1:
            return b, dict(
                doc=pos, offset=pos, start=0, length=n, split="final_test", sha256=h
            )
    raise AssertionError("no fixture")


def test_short_descriptor_reads_only_whole_hash_bound_final_once(tmp_path, monkeypatch):
    d = tmp_path / "stage"
    d.mkdir()
    file = tmp_path / "synthetic"
    raw = bytearray()
    rows = []
    words = []
    for n in (2, 20, 67, 4096):
        raw.extend(b"unreserved")
        b, r = binary_final(n, len(raw))
        raw.extend(b)
        rows.append(r)
        words.append(b)
    file.write_bytes(raw)
    st = file.stat()
    src = dict(path=file, inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns)
    j = w.JOBS[w.FINAL]
    f = {"job": w.FINAL}
    monkeypatch.setattr(
        w,
        "fresh_descriptor",
        lambda job, p: (d, src, rows)
        if job == j and p == f
        else pytest.fail("unexpected plan"),
    )
    got, read = ev._documents(j, f)
    assert (
        [z[2] for z in got] == words
        and read == sum(map(len, words))
        and read < len(raw)
    )
    with pytest.raises(FileExistsError):
        ev._documents(j, f)
    r = rows[-1]
    b = words[-1]
    for bad in [
        dict(r, split="train"),
        dict(r, start=1),
        dict(r, length=len(b) - 1),
        dict(r, doc=1),
        dict(r, sha256="0" * 64),
        dict(r, sha256="0" * 63 + "1"),
    ]:
        with pytest.raises(ValueError):
            ev.checked_short_document(b, bad)
    for v in (b + b"\x00", b[:-1], b[:1], b[::-1]):
        with pytest.raises(ValueError):
            ev.checked_short_document(v, r)


def test_short_mixed_endpoint_eval_and_closed_ingest_score_all_bytes(
    tmp_path, monkeypatch
):
    class Host(Tiny):
        def to(self, dev):
            assert dev == "cuda"
            return self  # only placement is stubbed; CPU host/head and NLL are real

    def origin(s, a):
        assert s == 4901
        torch.manual_seed(9700)
        b = trained_base()
        c = b if a == "B" else WriteHead(b, arm=a, seed=s)
        if a != "B":
            with torch.no_grad():
                c.out.weight.normal_(std=0.04)
        return Host(), c

    ref = "test"
    folder = tmp_path / w.FINAL
    folder.mkdir()
    windows = []
    rows = []
    for n in (2, 41, 175):
        b, r = binary_final(n, n + 5500)
        rows.append(r)
        windows.append((r["doc"], r["sha256"], b))
    counts = [len(b) - 1 for _, _, b in windows]
    bound = {}
    ends = {}
    for a, i in [("B", 1024), ("P", 1024), ("L", 1024), ("L", 992)]:
        path = tmp_path / f"w{a}"
        path.mkdir(exist_ok=True)
        host, head = origin(4901, a)
        bind = w.binding(ref, a, 4901)
        m = milestone(path, bind, i, host, head)
        ends[f"{a}-4901@{i}"] = dict(
            arm=a, seed=4901, step=i, binding=bind, sha256=m["sha256"], bytes=m["bytes"]
        )
        bound[a, i] = (path / m["file"], bind)
    plan = dict(
        job=w.FINAL,
        split="final_test",
        ref=ref,
        visit=0,
        horizons=[128, 1024, 4096],
        units=len(ends) * len(rows),
        documents=len(rows),
        targets_per_endpoint=sum(counts),
        endpoints=ends,
    )
    write_json(folder / "frozen.json", plan)
    write_json(folder / "start.json", dict(freeze_sha=sha(folder / "frozen.json")))
    calls = []
    seen = []
    monkeypatch.setattr(ev, "decode_probe", lambda *args: {"fixture": True})
    monkeypatch.setattr(ev, "score_window", partial(score_window, width=8))
    monkeypatch.setattr(ev, "cpu_origin", lambda *args: pytest.fail("wrong origin"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    for k in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, k, lambda: 0)

    def factory(s, a):
        calls.append((s, a))
        return origin(s, a)

    j = w.JOBS[w.FINAL]
    z = ev.evaluate_admitted(
        j,
        dict(folder=folder, ref=ref),
        lambda job, p: (windows, sum(counts) + len(counts)),
        lambda d, n, t: seen.append((n, t)),
        lambda a, s, i, ref: bound[a, i],
        origin=factory,
    )
    nt = z["targets"]
    assert (
        seen == [(i, w.count_cycle(counts)(i)) for i in range(1, plan["units"] + 1)]
        and nt == len(ends) * sum(counts)
        and len(calls) == 4
    )
    write_json(folder / "complete.json", z)
    write_json(
        folder / "charge.json",
        dict(
            rc=0,
            renewed=True,
            complete_sha=sha(folder / "complete.json"),
            witnessed_units=plan["units"],
        ),
    )
    table, stats, _ = read_stage(tmp_path, w.FINAL, rows, job=j)
    for key, e in ends.items():
        assert list(table[key]) == short_cells(e["arm"], e["step"] == 1024)
        cs = stats["endpoints"][key]["cells"]["4096:none"]
        assert (
            cs["targets"] == sum(counts)
            and cs["work"]["literal_misses"] > 0
            and sum(v["targets"] for v in cs["by_document_length"].values())
            == sum(counts)
        )
        ds = table[key][4096, "none"]
        assert np.isclose(cs["all_byte_nll"], sum(ds) / sum(counts)) and not np.isclose(
            cs["all_byte_nll"], np.mean(ds / counts)
        )
        host, c = origin(4901, e["arm"])
        host.eval()
        c.eval()
        for i, (_, _, word) in enumerate(windows):
            with torch.no_grad():
                h, logits = host(torch.tensor([list(word[:-1])]))[0].split((8, 256), -1)
                lp = c(word[:-1], h, logits, mode=e["arm"], horizon=4096)
                wanted = (
                    -lp[torch.arange(len(word) - 1), list(word[1:])]
                    .double()
                    .sum()
                    .item()
                )
                assert ds[i] == wanted
    with pytest.raises(ValueError, match="pairing"):
        read_stage(tmp_path, w.FINAL, rows[::-1], job=j)
    host, c = origin(4901, "P")
    with pytest.raises(ValueError):
        score_window(
            host.eval(),
            c.eval(),
            bytes(4097),
            arm="P",
            seed=4901,
            document=1,
            width=8,
            short_full_dose=False,
        )


def test_joint_ratio_bootstrap_and_mixed_wall_contrasts():
    ns = [1, 3, 2, 17, 27, 11, 93, 10]
    n = len(ns)
    x = np.asarray([np.arange(n) - 2.7, 2 * np.arange(n) + 1, np.arange(n)[::-1] - 1])
    fn = wa.document_bootstrap(ns)
    v = fn(x)
    draw = np.random.default_rng(4905).integers(0, n, (10000, n))
    den = np.asarray(ns)[draw].sum(1)
    obs = np.stack([xs[draw].sum(1) / den for xs in x]).mean(0)
    np.testing.assert_allclose(v["ci95"], np.percentile(obs, [2.5, 97.5]))
    np.testing.assert_allclose(v["by_seed"], x.sum(1) / sum(ns))
    assert v["gain"] == pytest.approx(x.sum() / (3 * sum(ns))) and not np.isclose(
        v["gain"], np.mean(x / ns)
    )
    with pytest.raises(ValueError):
        fn(x[:2])
    keys = {}
    wall = {}
    for z, s in enumerate(w.SEEDS):
        for a, loss in [("B", 2.0), ("P", 1.0), ("L", 0.75)]:
            tab = {
                cell: np.asarray(ns) * (loss + (0.5 if cell[1] != "none" else 0))
                for cell in short_cells(a, True)
            }
            keys[f"{a}-{s}@1024"] = tab
            keys[f"{a}-{s}@992"] = {
                (4096, "none"): tab[4096, "none"] + 0.1 * np.asarray(ns)
            }
        for a, b in w.PAIRS:
            wall.setdefault(a + ":" + b, {})[str(s)] = dict(
                selected={a: {"step": 992}, b: {"step": 1024}}, conclusive=z != 0
            )
    r = wa.compare(keys, ns, wall)
    assert (
        r["contrasts"]["L:P"]["full"]["4096"]["gain"] == 0.25
        and r["contrasts"]["L:B"]["common_wall"]["gain"] == pytest.approx(1.15)
        and not r["contrasts"]["P:B"]["wall_conclusive"]
        and r["binding"]["L"]["inner"]["gain"] == 0.5
    )


def test_frozen_mixed_clocks_bind_w_prep_endpoints_and_variable_work(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    monkeypatch.setattr(w, "prep", lambda: {"total_prep_wall_s": 18.0})
    rr = tmp_path / "rows.json"
    write_json(rr, {"panels": {"final_test": [{"length": 2}, {"length": 75}]}})
    monkeypatch.setattr(w, "ROWS", rr)
    monkeypatch.setattr(w, "ROW_SHA", sha(rr))
    monkeypatch.setattr(w, "N", 2)
    monkeypatch.setattr(w, "NT", 75)
    runs = {}
    ref = "fixture"
    for j in list(w.JOBS.values())[:-1]:
        d = tmp_path / j.name
        d.mkdir()
        ms = []
        a = j.arms
        for i in range(32, 1025, 32):
            p, bind = w.resolve(a, j.seed, i, ref)
            p.write_bytes(f"fixture:{j.name}:{i}".encode())
            h = sha(p)
            sz = p.stat().st_size
            write_json(
                p.with_suffix(".json"),
                dict(file=p.name, binding=bind, step=i, sha256=h, bytes=sz),
            )
            ms.append(
                dict(
                    step=i,
                    wall_s=i * {"B": 1.0, "L": 1.3, "P": 1.1}[a],
                    sha256=h,
                    bytes=sz,
                )
            )
        runs[j.name] = dict(
            milestones=ms, wall_s=ms[-1]["wall_s"] + 2, charge_sha="receipt"
        )
    monkeypatch.setattr(w, "histories", lambda r: runs)
    j = w.JOBS[w.FINAL]
    d = tmp_path / j.name
    d.mkdir()
    fp = d / "frozen.json"
    clock = w.clocks(runs)
    pair = clock["L:B"]["4901"]
    assert pair["budget_s"] == 1028.0 and pair["selected"]["B"]["wall_s"] == 1026.0
    m = pair["selected"]["L"]
    assert m["step"] == 768 and m["wall_s"] == 1.3 * 768 + 2.0
    p, _ = w.resolve("L", 4901, m["step"], ref)
    good = p.read_bytes()
    p.write_bytes(bytes(len(good)))
    with pytest.raises(ValueError, match="not bound"):
        w.freeze(j, ref)
    assert not fp.exists()
    p.write_bytes(good)
    digest = w.freeze(j, ref)
    f = load(fp)
    assert len(f["endpoints"]) == 18 and f["wall_pairs"] == clock and f["units"] == 36
    st = dict(freeze_sha=digest, units=36)
    seen = w.targets_for(j, d, st)
    assert [seen(i) for i in (1, 2, 3, 36)] == [1, 75, 76, 1350]
    with pytest.raises(ValueError, match="already frozen"):
        w.freeze(j, ref)
    with pytest.raises(ValueError, match="unbound witness"):
        w.targets_for(j, d, dict(st, freeze_sha="X"))


def test_partial_after_raw_attempt_does_not_claim_release_or_score(tmp_path):
    j = w.JOBS[w.FINAL]
    d = tmp_path / j.name
    d.mkdir()
    r = panel.partial(j, root=tmp_path)
    assert (
        not r["source_open_attempted"]
        and r["upper_card_s"] is None
        and r["last_progress"] is None
    )
    (d / "source-open-attempted").touch()
    write_json(d / "progress.json", dict(units=2, scored_targets=175))
    (d / "released").write_text("released_epoch=120\nrc=124\n")
    (d / "initial").write_text("created=100\n")
    r = panel.partial(j, root=tmp_path)
    assert (
        r["source_open_attempted"]
        and r["upper_card_s"] == 21
        and r["last_progress"]["scored_targets"] == 175
    )


def test_w_driver_injects_correct_stream_origin_clock_and_no_old_eval(
    monkeypatch, tmp_path
):
    seen = []
    ctx = dict(folder=tmp_path, ref="fixture")
    monkeypatch.setattr(run, "admit", lambda j: ())
    monkeypatch.setattr(panel, "enter_admitted", lambda j: ctx)
    monkeypatch.setattr(run.loop, "complete", lambda d, o: seen.append(o))
    monkeypatch.setattr(
        run,
        "short_device_check",
        lambda j, c: seen.append(("synthetic-before-source", c)),
    )

    def train(j, c, prog, bind, birth, first, *, stream_factory, origin):
        assert (
            c == ctx
            and origin is run.origin
            and stream_factory.func is run.WriteStream
            and stream_factory.args == (w.W_ID,)
            and bind is w.binding
            and prog.keywords["root"] == w.ROOT
            and birth.func is run.guard_birth
            and first.func is run.guard_first
        )
        return j.name

    def evaluate(j, c, docs, prog, resolve, *, origin):
        assert (
            seen[-1] == ("synthetic-before-source", ctx)
            and docs is ev._documents
            and resolve is w.resolve
            and origin is run.origin
        )
        return j.name

    monkeypatch.setattr(tr, "train_admitted", train)
    monkeypatch.setattr(ev, "evaluate_admitted", evaluate)
    for j in w.JOBS.values():
        run.inside(j)
    assert [k for k in seen if isinstance(k, str)] == list(w.JOBS)
