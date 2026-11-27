import hashlib

import numpy as np
import pytest

from . import gate
from .run import frames


def test_expired_or_too_little_space_fails_closed():
    assert gate.capacity(gate.END - gate.CAP - 11, 20 * 1024**3) == (900, gate.END)
    for t, d in (
        (gate.END, 40 * 1024**3),
        (gate.END - 20, 40 * 1024**3),
        (0, 19 * 1024**3),
    ):
        with pytest.raises(RuntimeError):
            gate.capacity(t, d)


def test_all_eight_paired_final_states_must_exist_before_probe(tmp_path, monkeypatch):
    job = tmp_path / "job"
    data = tmp_path / "data"
    job.mkdir()
    data.mkdir()
    monkeypatch.setattr(gate, "JOB", job)
    monkeypatch.setattr(gate, "DATA", data)
    for p in ("rows.json", "build.npy", "fit.npy"):
        (data / p).write_bytes(b"fixture")
    h = hashlib.sha256(b"fixture").hexdigest()
    gate.put(job / "admission.json", dict(qualified="pub", rows_sha=h))
    f = dict(qualified="pub", rows_sha=h, build_sha=h, fit_sha=h, fits={})
    for seed in gate.SEEDS:
        for arm in gate.ARMS:
            k = f"{seed}.{arm}"
            (job / (k + ".pt")).write_bytes(b"fixture")
            p = job / (k + ".json")
            gate.put(
                p,
                dict(
                    seed=seed,
                    arm=arm,
                    weights_sha=h,
                    updates=1024,
                    targets=1048576,
                    order_sha=gate.order_sha(seed),
                    initial=str(seed),
                ),
            )
            f["fits"][k] = gate.sha(p)
    gate.put(job / "frozen.json", f)
    p = job / "5752.indexed.pt"
    p.write_bytes(b"bad")
    with pytest.raises(RuntimeError):
        gate.witness_freeze()
    assert not (job / "freeze-ack.json").exists()
    p.write_bytes(b"fixture")
    gate.witness_freeze()
    assert gate.get(job / "freeze-ack.json")["fits"] == 8


def test_zstd_baselines_code_all_raw_bytes():
    x = np.arange(128, dtype=np.uint8).reshape(4, -1)
    out = frames(x, x.copy())
    assert all(
        q["bpb"] == 8 * q["frame_bytes"] / 128 and q["frame_bytes"] > 0
        for q in out.values()
    )
