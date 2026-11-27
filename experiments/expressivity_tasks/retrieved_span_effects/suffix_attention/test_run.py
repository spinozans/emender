"""Driver oracles on invented bytes: schedule, dose, freeze, probe, report."""

import math

import numpy as np
import pytest

import experiments.expressivity_tasks.retrieved_span_effects.suffix_attention.run as run
from .model import SalLM, SalPack


@pytest.fixture
def env(tmp_path, monkeypatch):
    job = tmp_path / "job"
    data = tmp_path / "data"
    job.mkdir()
    data.mkdir()
    rng = np.random.default_rng(5776)
    build = rng.integers(0, 5, (1, 129), dtype=np.uint8)
    fit = rng.integers(0, 5, (12, 129), dtype=np.uint8)
    probe = rng.integers(0, 5, (2, 129), dtype=np.uint8)
    np.save(data / "build.npy", build)
    np.save(data / "fit.npy", fit)
    np.save(data / "probe.npy", probe)
    (data / "rows.json").write_text("{}")
    monkeypatch.setattr(run, "JOB", job)
    monkeypatch.setattr(run, "DATA", data)
    monkeypatch.setattr(run, "N", {"build": 1, "fit": 12, "probe": 2})
    monkeypatch.setattr(run, "UPDATES", 6)
    monkeypatch.setattr(
        run, "await_ack", lambda folder, name=None: (job / name).touch()
    )
    monkeypatch.setattr(
        run, "load", lambda role: np.load(data / (role + ".npy"), allow_pickle=False)
    )
    return dict(job=job, data=data, build=build, fit=fit, probe=probe)


def test_schedule_and_lr_endpoints():
    assert run.lr_at(1) == run.LR0 / run.WARMUP
    assert run.lr_at(run.WARMUP) == run.LR0
    assert run.lr_at(run.UPDATES) == run.LR1
    assert np.array_equal(run.order(5751), run.order(5751))
    assert run.order_sha(5751) == run.order_sha(5751)
    assert run.order_sha(5751) != run.order_sha(5752)


def test_full_driver_cycle_on_invented_bytes(env):
    job = env["job"]
    pack = SalPack(env["build"], SalLM(5751), batch=2)
    fit_batches = pack.dataset(env["fit"], shuffle=True)
    assert len(fit_batches) == 12
    run.fits(fit_batches, pack.index, lambda s: None, "pubref")
    frozen = run.get(job / "frozen.json")
    assert (job / "freeze-ack.json").exists()
    for s in run.SEEDS:
        inits = set()
        for arm in run.ARMS:
            k = f"{s}.{arm}"
            meta = run.get(job / (k + ".json"))
            assert meta["updates"] == 6 and meta["targets"] == 768
            assert meta["order_sha"] == run.order_sha(s)
            assert meta["weights_sha"] == run.sha(job / (k + ".pt"))
            assert frozen["fits"][k] == run.sha(job / (k + ".json"))
            assert meta["fit_nats_per_byte"] > 0
            inits.add(meta["initial"])
        assert len(inits) == 1  # paired starts across arms within a seed
    # Witnessed freeze gate: scoring refuses without it.
    (job / "freeze-ack.json").unlink()
    probe_batches = pack.dataset(env["probe"], shuffle=False)
    with pytest.raises(RuntimeError):
        run.score(probe_batches, pack.index, lambda s: None)
    (job / "freeze-ack.json").write_text(
        '{"frozen_sha": "%s"}' % run.sha(job / "frozen.json")
    )
    losses, clocks = run.score(probe_batches, pack.index, lambda s: None)
    q = run.summarize(losses)
    assert q["raw_bytes"] == 2 * 4097 and q["shifted_targets"] == 2 * 4096
    for arm in run.ARMS:
        assert math.isfinite(q["mean_bpb"][arm]) and q["mean_bpb"][arm] > 0
        assert len(q["per_seed_bpb"][arm]) == 2
    assert q["primary_positive"] is False or q["primary_positive"] is True
    for key, c in q["contrasts"].items():
        assert c["ci"][0] <= c["gain"] <= c["ci"][1] and len(c["seeds"]) == 2
    # Bits/byte accounting: per-document nats reproduce the mean exactly.
    z = run.frames(env["probe"], env["build"])
    for v in z.values():
        assert (
            v["frame_bytes"] > 0
            and v["bpb"] == 8 * v["frame_bytes"] / env["probe"].nbytes
        )
    with np.load(job / "document-losses.npz", allow_pickle=False) as f:
        for s in run.SEEDS:
            for arm in run.ARMS:
                n = f[f"{s}.{arm}"]
                assert n.shape == (2,) and (n > 0).all()
                expect = (n / math.log(2) + 8.0) / 4097.0
                np.testing.assert_allclose(
                    expect.mean(),
                    q["per_seed_bpb"][arm][run.SEEDS.index(s)],
                    rtol=1e-12,
                )
