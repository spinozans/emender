"""Invented data only: exact native notes/work, causal reset and all-256 bound-only J parity."""

import json
from math import log
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    joint_reader as joint,
    native_text as nt,
    literal_mark as ls,
    surprise as u,
    query_panel,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.test_literal_mark import (
    dense_forecast,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise import (
    example,
    scalar_code,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    # In the priced gate, test the actual retained binary. Ordinary tests build in temp.
    arg = os.getenv("X49_JOINT_TEST_BUILD")
    if arg:
        folder = Path(arg)
        note = json.loads((folder / "build.json").read_bytes())
    else:
        folder = tmp_path_factory.mktemp("joint_writer") / "build"
        note = nt.build(folder)
    return folder, note, nt._load(folder, note)


def test_native_notes_work_prefixes_and_reset(binary):
    _, _, writer = binary
    rng = np.random.default_rng(41)
    random = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    periodic = (b"\0\0\x01\0\0\0\xff\x1e\xffababa" * 360)[:4096]
    sat = b"A" * 4095 + b"\0"
    full = [random, periodic, sat, b"\0" * 4096, bytes(range(256)) * 16]
    for w in full:
        tape, work = writer(w)
        ref, rwork = ls.text_tape(w)
        assert (
            torch.equal(tape, ref)
            and work == rwork
            and tape.dtype == torch.uint8
            and tape.nbytes == len(w)
        )
        for n in (1, 16, 64, 65, 270, 4095):
            small, sw = writer(w[:n])
            py, pw = ls.text_tape(w[:n])
            assert torch.equal(small, tape[:n]) and torch.equal(small, py) and sw == pw
        assert torch.equal(writer(w[:1000] + b"changed suffix")[0][:1000], ref[:1000])
    assert writer(sat)[0][-1] == 255 and writer(b"\0")[0].tolist() == [0]
    for bad in (b"", bytearray(b"a"), b"a" * 4097):
        with pytest.raises(ValueError, match="observed bytes"):
            writer(bad)


def test_native_dense_independent_growing_prefix_oracle(binary):
    _, _, writer = binary
    rng = np.random.default_rng(841)
    w = (b"\x00\xff\x00\x01\x00\x00\x1e" + b"aba" * 8) * 3 + rng.integers(
        0, 10, 128, dtype=np.uint8
    ).tobytes()
    tape, _ = writer(w)
    assert tape.tolist() == [
        scalar_code(-log(dense_forecast(w[:j])[v])) if j else 0 for j, v in enumerate(w)
    ]


def test_wrong_or_changed_library_refused(binary, tmp_path):
    folder, note, _ = binary
    with pytest.raises(ValueError, match="stale native"):
        nt._load(folder, dict(note, source_sha256="0" * 64))
    other = tmp_path / "different"
    other.mkdir()
    (other / "build.json").write_text(json.dumps(note))
    (other / nt.BINARY).write_bytes((folder / nt.BINARY).read_bytes() + b"changed")
    with pytest.raises(ValueError, match="stale native"):
        nt._load(other, note)
    with pytest.raises(ValueError, match="stale native"):
        nt._load(tmp_path, note)


def test_bound_only_candidate_equals_ls_all_256_without_sham(binary, monkeypatch):
    _, _, writer = binary
    head, w, _, _ = example()
    host = Tiny().eval()
    words = [w[:1], w, (w * 24)[:4096]]
    refs = []
    for obs, s in zip(words, (4901, 4902, 4903), strict=True):
        h, z, _ = frozen_prefill(host, obs, 8)
        p = ls.fields(head, obs, h, z)
        beta = joint.contract(s)[1]
        refs.append(ls.response(p, "J", beta))

    def forbidden(*a, **kw):
        raise AssertionError(
            "bound-only J must not score a control, rotate a binding or call the Python writer"
        )

    for module, name in (
        (u, "_rotation"),
        (ls, "fields"),
        (ls, "response"),
        (ls, "text_tape"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    calls = []
    hook = host.register_forward_pre_hook(
        lambda m, inputs: calls.append(inputs[0].clone())
    )
    try:
        for obs, ref, s in zip(words, refs, (4901, 4902, 4903), strict=True):
            out = joint._predict(host, head, joint.contract(s)[1], obs, 8, writer)
            assert (
                torch.equal(ref, out)
                and out.shape == (len(obs), 256)
                and not out.requires_grad
            )
    finally:
        hook.remove()
    assert len(calls) == len(words) and all(
        torch.equal(x, torch.tensor([list(obs)]))
        for x, obs in zip(calls, words, strict=True)
    )


def test_seed_coefficient_attachment_and_reader_admission(monkeypatch, tmp_path):
    from experiments.expressivity_tasks.x47_xsa_attention import workbench_run

    orig = joint.local_u.contract
    raw = (joint.RECEIPTS / "literal_surprise_result.json").read_bytes()
    z = json.loads(raw)
    actual = [joint.contract(s) for s in (4901, 4902, 4903)]
    assert len(set(actual)) == 3 and all(
        b == tuple(z["beta"][str(s)]["J"]) and ck == orig(s)[0]
        for (ck, b), s in zip(actual, (4901, 4902, 4903), strict=True)
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        joint.contract(True)
    with monkeypatch.context() as m:
        m.setattr(joint.local_u, "contract", lambda s: orig(4902))
        with pytest.raises(ValueError, match="confirmed base"):
            joint.contract(4901)
    with monkeypatch.context() as m:
        m.setattr(joint, "RECEIPTS", tmp_path)
        z["beta"]["4901"]["J"] = z["beta"]["4903"]["J"]
        (tmp_path / "literal_surprise_result.json").write_text(json.dumps(z))
        with pytest.raises(ValueError, match="changed frozen"):
            joint.contract(4901)
        with pytest.raises(ValueError, match="no published"):
            joint._writer()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GPU_LEASE_HELD", raising=False)
    with pytest.raises(AssertionError, match="need physical 0/1 lease"):
        joint.load_candidate(4901)
    monkeypatch.setattr(workbench_run, "check_gpu_owner", lambda: None)
    monkeypatch.setattr(
        query_panel, "checkpoint", lambda *a, **kw: (None, {"sha256": orig(4902)[0]})
    )
    with pytest.raises(ValueError, match="checkpoint differ"):
        joint.load_candidate(4901)
