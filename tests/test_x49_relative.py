"""X49-R qualification, synthetic inputs only. No real leases, corpus, prior scores or checkpoints."""

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
from experiments.expressivity_tasks.x48_muon_continuation.train import restore, save
from experiments.expressivity_tasks.x49_live_trace_memory import (
    relative_data as data,
    relative_panel as rp,
    relative_run as run,
)
from experiments.expressivity_tasks.x49_live_trace_memory import panel_eval as ev
from experiments.expressivity_tasks.x49_live_trace_memory.coupled import step
from experiments.expressivity_tasks.x49_live_trace_memory.panels import select
from experiments.expressivity_tasks.x49_live_trace_memory.posthoc_credit_control import (
    relative_objective,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import (
    equal,
    fresh,
)


def test_relative_budget_and_checkpoint_namespaces_are_not_interchangeable(
    tmp_path, monkeypatch
):
    a = {**rp.expected("new", "fresh"), "operator_note": "synthetic only"}
    rp.validate(a, "new", "fresh")
    assert (
        list(rp.JOBS) == ["R-4901", "R-4902", "R-4903", "r-final"]
        and sum(a["job_caps"].values()) == 7740
    )
    for k, v in [
        ("scope", "x49-registered-natural-panel-v2"),
        ("gpus", True),
        ("remote", True),
        ("qualified_ref", rp.TRAIN_REF),
        ("reservation_sha", "v2"),
        ("parent_summary", "other"),
        ("retries", 1),
        ("job_caps", {"r-final": 7740}),
    ]:
        with pytest.raises(ValueError):
            rp.validate(dict(a, **{k: v}), "new", "fresh")
    c, old = rp.resolve("C", 4901, 1024, "new")
    r, new = rp.resolve("R", 4901, 1024, "new")
    assert (
        c != r
        and old["ref"] == rp.TRAIN_REF
        and new["ref"] == "new"
        and old["arm"] != new["arm"]
        and old["origin"] == new["origin"]
    )
    with pytest.raises(ValueError):
        rp.resolve("A", 4901, 1024, "new")
    monkeypatch.setattr(run, "AUTH", tmp_path / "never-approved")
    with pytest.raises(ValueError, match="separate R campaign budget"):
        ev._documents(rp.JOBS[rp.FINAL], {})
    with pytest.raises(ValueError, match="unlicensed"):
        rp.fresh_descriptor(rp.JOBS["R-4901"], {})
    assert not list(tmp_path.iterdir())


def test_remaining_uses_full_document_not_offsets_or_spent_test_windows():
    # Native final split, duplicate identical content at another offset plus historical exclusions.
    records = []
    for i, n in enumerate([1, 101, 201, 301, 401, 501, 101, 601]):
        off = MIN_OFFSET + i * 8192
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=8192,
                sha256=f"{n:064x}",
                split="final_test",
            )
        )
    excluded = {records[0]["sha256"]}
    size = records[-1]["byte_offset"] + 8192
    old, _ = select(records, excluded, {"final_test": 3}, size)
    fresh, _ = data.remaining(records, excluded, old, size, 3)
    assert len(fresh) == 3 and not set(r["sha256"] for r in fresh) & set(
        r["sha256"] for r in old["final_test"]
    )
    with pytest.raises(ValueError, match="unexpected number"):
        data.remaining(records, excluded, old, size, 2)
    with pytest.raises(ValueError, match="insufficient"):
        data.remaining(records, excluded, old, size, 4)
    with pytest.raises(ValueError, match="exact unspent suffix"):
        data.remaining(
            records,
            excluded,
            {"final_test": old["final_test"][1:] + fresh[:1]},
            size,
            3,
        )


def test_relative_returns_reach_all_eligible_files_even_without_hits_and_detach_control():
    n = 148
    h = 128
    y = torch.zeros(n, dtype=torch.long)
    lp = torch.zeros(n, 256)
    lp[:, 0] = torch.linspace(-1.2, -1.6, n)
    lp.requires_grad_()
    base = torch.zeros_like(lp)
    base[:, 0] = -1.38
    base.requires_grad_()
    file = torch.full((n - 1,), -0.1, requires_grad=True)
    query = torch.full((n,), -0.1, requires_grad=True)
    roll = SimpleNamespace(lp=lp, file_lp=file, query_lp=query, horizon=h)
    loss, nll = relative_objective(roll, base, y)
    g, ask, ctrl, ml = torch.autograd.grad(
        loss, (file, query, base, lp), allow_unused=True
    )
    (path,) = torch.autograd.grad(torch.nn.functional.nll_loss(lp, y), lp)
    adv = (base[:, 0].detach() - lp[:, 0].detach()).double()
    expected = torch.tensor(
        [float(adv[j + 16 : min(n, j + h)].sum()) / n for j in range(1, n)]
    )
    torch.testing.assert_close(g, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(ask, adv.float() / n)
    assert ctrl is None and torch.equal(path, ml) and torch.equal(loss, nll)
    with pytest.raises(ValueError):
        relative_objective(roll, base[:-1], y)


def test_full_R_update_matches_fixed_birth_then_learns_and_replays_both_optimizers(
    tmp_path,
    monkeypatch,
):
    was = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4901)
            m, h, opts = fresh("C")
            torch.manual_seed(4901)
            mf, hf, ofs = fresh("F")
            assert equal(m.state_dict(), mf.state_dict()) and equal(
                h.state_dict(), hf.state_dict()
            )
            w1 = (b"abc:\xffa:ab\0\x1e" * 500)[:4097]
            w2 = (b"abc=123:abc456" * 400)[:4097]
            kw = dict(seed=4901, width=8)
            clip = torch.nn.utils.clip_grad_norm_
            before = []

            def capture(params, *args, **kwargs):
                before.append(
                    [None if v.grad is None else v.grad.clone() for v in params]
                )
                return clip(params, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
                u = step(m, h, opts, w1, index=1, document=7, mode="R", **kw)
                f = step(mf, hf, ofs, w1, index=1, document=7, mode="F", **kw)
            # Only the four trainable policy tensors differ in gradient participation.
            # Every common pathwise gradient is exactly equal before joint clipping.
            policy = {id(p) for layer in (h.filing, h.ask) for p in layer.parameters()}
            for par, rg, fg in zip(
                [*m.parameters(), *h.parameters()], *before, strict=True
            ):
                if id(par) in policy:
                    assert (
                        fg is None and rg is not None and torch.count_nonzero(rg) == 0
                    )
                else:
                    assert equal(rg, fg)
            assert (
                u["preupdate_nll"] == f["preupdate_nll"]
                and u["work"] == f["work"]
                and u["preclip"] == pytest.approx(f["preclip"], rel=1e-6, abs=1e-6)
            )
            assert (
                torch.count_nonzero(h.filing.weight)
                == torch.count_nonzero(h.ask.weight)
                == 0
            )
            # Reduction over additional zero-norm tensors can round the clip scalar differently.
            eps = torch.finfo(torch.float32).eps
            for a, b in zip(
                [*m.parameters(), *h.parameters()],
                [*mf.parameters(), *hf.parameters()],
                strict=True,
            ):
                torch.testing.assert_close(a, b, rtol=eps, atol=eps)
            bind = rp.binding("fixture", "R", 4901)
            rec = save(tmp_path, bind, 1, m, h, opts)
            second = step(m, h, opts, w2, index=2, document=55, mode="R", **kw)
            assert (
                torch.count_nonzero(h.filing.weight) > 0
                and torch.count_nonzero(h.ask.weight) > 0
            )
            m2, h2, o2 = fresh("C")
            with pytest.raises(ValueError, match="foreign"):
                restore(tmp_path, rp.binding("fixture", "C", 4901), 1, m2, h2, o2)
            assert (
                restore(tmp_path, bind, 1, m2, h2, o2) == rec["sha256"]
                and step(m2, h2, o2, w2, index=2, document=55, mode="R", **kw) == second
            )
            for a, b in zip((m, h, *opts), (m2, h2, *o2), strict=True):
                assert equal(a.state_dict(), b.state_dict())
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn)
        torch.set_num_threads(threads)


def test_R_freeze_rehashes_mixed_source_endpoints_and_deduplicates(
    tmp_path, monkeypatch
):
    ref = "new"
    old = tmp_path / "v2"
    root = tmp_path / "vR"
    old.mkdir()
    root.mkdir()
    f = root / rp.FINAL
    f.mkdir()
    reserve = tmp_path / "res"
    reserve.write_bytes(b"synthetic membership")
    monkeypatch.setattr(rp, "OLD", old)
    monkeypatch.setattr(rp, "ROOT", root)
    monkeypatch.setattr(rp, "RESERVATION", reserve)
    runs = {}
    for s in rp.SEEDS:
        for a in "RBFC":
            k = rp.name(a, s)
            d = (root if a == "R" else old) / k
            d.mkdir()
            clock = []
            for i in range(32, 1025, 32):
                p, b = rp.resolve(a, s, i, ref)
                v = f"toy {k} {i}".encode()
                p.write_bytes(v)
                write_json(
                    p.with_suffix(".json"),
                    dict(binding=b, bytes=len(v), file=p.name, sha256=sha(p), step=i),
                )
                clock.append(
                    dict(
                        step=i,
                        wall_s=i * (1.25 if a == "R" else 1.0),
                        sha256=sha(p),
                        bytes=len(v),
                    )
                )
            runs[k] = dict(
                milestones=clock, wall_s=clock[-1]["wall_s"] + 1, charge_sha="toy"
            )
    monkeypatch.setattr(rp, "runs_for", lambda r: runs)
    bad = old / "C-4901/weights-1024.pt"
    v = bad.read_bytes()
    bad.write_bytes(bytes(len(v)))
    with pytest.raises(ValueError, match="unbound"):
        rp.freeze(rp.JOBS[rp.FINAL], ref)
    assert not (f / "frozen.json").exists()
    bad.write_bytes(v)
    digest = rp.freeze(rp.JOBS[rp.FINAL], ref)
    p = load(f / "frozen.json")
    assert (
        digest == sha(f / "frozen.json")
        and p["reservation_sha"] == sha(reserve)
        and len(p["endpoints"]) == 15
        and p["units"] == 15 * 137
        and all(p["endpoints"][f"R-{s}@800"]["binding"]["ref"] == ref for s in rp.SEEDS)
    )
    with pytest.raises(ValueError, match="already frozen"):
        rp.freeze(rp.JOBS[rp.FINAL], ref)


def test_R_eval_uses_own_weights_and_unchanged_learned_predictor(tmp_path, monkeypatch):
    k = "R-4901@1024"
    job = rp.JOBS[rp.FINAL]
    ref = "follow"
    b = rp.binding(ref, "R", 4901)
    m = torch.nn.Linear(1, 1)
    h = torch.nn.Linear(1, 1)
    p = tmp_path / "only-R.pt"
    torch.save(dict(host=m.state_dict(), head=h.state_dict(), binding=b, step=1024), p)
    e = dict(
        arm="R", seed=4901, step=1024, binding=b, bytes=p.stat().st_size, sha256=sha(p)
    )
    f = tmp_path / "frozen.json"
    write_json(
        f, dict(ref=ref, visit=0, horizons=[128, 1024, 4096], endpoints={k: e}, units=1)
    )
    write_json(tmp_path / "start.json", dict(freeze_sha=sha(f)))
    events = []

    class Fake(torch.nn.Linear):
        def to(self, dev):
            assert dev == "cuda"
            return self

    def origin(seed, arm):
        assert (seed, arm) == (4901, "R")
        return Fake(1, 1), Fake(1, 1)

    def score(host, head, w, *, arm, seed, document, stratify):
        assert (
            arm == "C"
            and not host.training
            and not head.training
            and equal(host.state_dict(), m.state_dict())
            and equal(head.state_dict(), h.state_dict())
            and stratify
        )
        return dict(synthetic=True)

    monkeypatch.setattr(ev, "cpu_origin", origin)
    monkeypatch.setattr(ev, "score_window", score)
    monkeypatch.setattr(
        ev, "decode_probe", lambda host, head, a, s: {"inference_arm": a}
    )
    monkeypatch.setattr(ev.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ev.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(ev.torch.cuda, "max_memory_reserved", lambda: 0)
    out = ev.evaluate_admitted(
        job,
        dict(folder=tmp_path, ref=ref),
        lambda j, x: ([(71, "fake", bytes(4097))], 4097),
        lambda d, i, t: events.append((i, t)),
        lambda a, s, i, r: (p, b),
    )
    assert (
        events == [(1, 4096)]
        and out["endpoints"][k]["checkpoint_sha"] == e["sha256"]
        and out["endpoints"][k]["decode_probe"]["inference_arm"] == "C"
    )


def test_R_analysis_pairs_all_fresh_docs_including_misses():
    seeds = rp.SEEDS
    n = data.N
    tab = {}
    w = {}
    base = np.linspace(1, 2, n)
    for a, off in [("R", 0.00), ("C", 0.24), ("F", 0.01), ("B", 0.02)]:
        for s in seeds:
            cells = {(H, "none"): base + off * (H / 4096) for H in (128, 1024, 4096)}
            cells[4096, "literal"] = base + off + 0.004
            if a != "B":
                cells[4096, "trace"] = base + off + 0.001
            tab[f"{a}-{s}@1024"] = cells
        if a != "R":
            w["R:" + a] = {
                str(s): dict(
                    selected={z: dict(step=1024) for z in ("R", a)}, conclusive=True
                )
                for s in seeds
            }
    o = rp.compare(tab, w)
    assert o["contrasts"]["R:F"]["full"]["4096"]["gain"] == pytest.approx(0.01)
    assert o["contrasts"]["R:C"]["common_wall"]["ci95"] == pytest.approx([0.24, 0.24])
    assert o["binding"]["R"]["trace"]["gain"] == pytest.approx(0.001)
    del w["R:B"][str(seeds[0])]["selected"]["R"]["step"]
    w["R:B"][str(seeds[0])]["selected"]["R"] = None
    o = rp.compare(tab, w)
    assert (
        o["contrasts"]["R:B"]["common_wall"] is None
        and not o["contrasts"]["R:B"]["wall_conclusive"]
    )
