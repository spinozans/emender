"""Data/price/learning gates with invented bytes and temp receipts only."""

import numpy as np
import pytest
import torch

from . import data as d, study as s
from .program import ARMS, T
from .test_read_tape import Fixture


def test_paired_document_draws_independent_oracle():
    rng = np.random.default_rng(186)
    losses = {a: rng.uniform(1, 90, (3, 7)) for a in (*ARMS, "PRIOR")}
    policies, contrasts = s.compare(losses, draws=127)
    samples = np.random.default_rng(5000).integers(7, size=(127, 7))
    for k, v in contrasts.items():
        a, b = k.split(":")
        vals = []
        ds = losses[b] - losses[a]
        for draw in samples:
            vals.append(sum(ds[i, j] for i in range(3) for j in draw) / (3 * 7 * T))
        np.testing.assert_allclose(v["ci95"], np.percentile(vals, [2.5, 97.5]))
        np.testing.assert_allclose(v["by_seed"], ds.sum(1) / (7 * T))
        assert np.isclose(v["gain"], policies[b]["mean"] - policies[a]["mean"])


@pytest.mark.parametrize("arm", ARMS)
def test_epoch_dose_birth_parity_fit_change_and_reset(arm):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    raw = rng.choice(np.array([0, 30, 65, 66, 67, 255], np.uint8), (16, T + 1))
    raw.flags.writeable = False
    seed = d.SEEDS[0]
    model = s.model_at_birth(seed)
    before = model.emit.weight.detach().clone()
    first, _, base = s.phase(model, Fixture(), raw, seed, arm, False, lambda n: None)
    assert first["epochs"] == 1 and np.isclose(
        first["mean_nll_by_epoch"][0], base.mean() / T, atol=1e-5
    )
    trace = []
    fit, _, _ = s.phase(model, Fixture(), raw, seed, arm, True, trace.append)
    assert trace == [16 * T] * s.EPOCHS and not torch.equal(before, model.emit.weight)
    if arm in ("ADAPT", "BLIND"):
        assert torch.count_nonzero(model.act.weight) > 0
    assert fit["archive"]["parent_reads"] == len(trace) * 16 * T and fit["archive"][
        "child_reads"
    ] == (0 if arm == "ONE" else fit["targets"])
    ev, loss, prior = s.phase(model, Fixture(), raw, seed, arm, False, lambda n: None)
    assert ev["targets"] == 16 * T and np.isclose(
        loss.sum() / ev["targets"], ev["mean_nll_by_epoch"][0]
    )
    np.testing.assert_array_equal(prior, base)


def test_freeze_refuses_one_missing_model_or_changed_state(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RUN", tmp_path)
    monkeypatch.setattr(d, "OUT", tmp_path)
    for name in ("reservation.json", "fit.json", "qualified.json"):
        d.write_json(tmp_path / name, {"bound": name})
    states = {}
    for i, k in enumerate(sorted(d.state_keys())):
        p = tmp_path / (k + ".pt")
        p.write_bytes(bytes([i]))
        states[k] = d.sha(p)
    seal = dict(
        spec=d.SPEC,
        rows_sha=d.sha(tmp_path / "reservation.json"),
        fit_identity_sha=d.sha(tmp_path / "fit.json"),
        qualification_sha=d.sha(tmp_path / "qualified.json"),
        states=states,
    )
    d.write_json(tmp_path / "frozen.json", seal)
    assert d.frozen() == seal
    p.write_bytes(b"not the frozen model")
    with pytest.raises(ValueError, match="incomplete or changed"):
        d.frozen()
    states.pop(k)
    d.write_json(tmp_path / "frozen.json", seal)
    with pytest.raises(ValueError, match="incomplete or changed"):
        d.frozen()


def test_probe_access_stops_before_any_source_or_reservation_without_fit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(d, "RUN", tmp_path)
    monkeypatch.setattr(d, "qualified", lambda: {})
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        d.extract("probe")
    assert not tuple(tmp_path.iterdir())


def test_window_cut_checks_raw_digest_before_shifts():
    import hashlib
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import window

    doc = bytes(range(256)) * 33 + bytes([255])
    r = dict(
        split="train",
        doc=718,
        length=len(doc),
        start=4096,
        sha256=hashlib.sha256(doc).hexdigest(),
    )
    w = window(doc, r)[: T + 1]
    assert w[:-1] == doc[4096 : 4096 + T] and w[1:] == doc[4097 : 4097 + T]
    with pytest.raises(ValueError, match="raw document"):
        window(doc[:-1] + b"\x00", r)
    with pytest.raises(ValueError, match="registered full"):
        window(doc, dict(r, start=8192))
