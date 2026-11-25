"""Synthetic control-flow/cadence and score ingestion; these stubs do not run the GPU learner."""

import json
from time import monotonic_ns
from types import SimpleNamespace
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    panel as p,
    panel_train as tr,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_spec import T, JOBS
from experiments.expressivity_tasks.x49_live_trace_memory.panel_analysis import (
    read_stage,
)
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    load,
    sha,
    write_json,
)


def test_training_driver_dose_clock_and_checkpoint_cadence_without_real_lease(
    tmp_path, monkeypatch
):
    job = JOBS["A-4901"]
    d = tmp_path
    prog = []
    recovery = []
    weights = []
    monkeypatch.setattr(
        p, "enter", lambda j: dict(folder=d, ref="fixture", t0=monotonic_ns())
    )
    monkeypatch.setattr(p, "progress", lambda d, i, t: prog.append((i, t)))

    class Fake(torch.nn.Linear):
        def to(self, dev):
            assert dev == "cuda"
            return self

    host = Fake(1, 1)
    head = Fake(1, 1)
    monkeypatch.setattr(tr, "cpu_origin", lambda s, k: (host, head))
    monkeypatch.setattr(
        tr,
        "optimizers",
        lambda m, h: (
            [SimpleNamespace(state={}, aux=SimpleNamespace(state={}))] * 2,
            {},
        ),
    )
    w = bytes(T + 1)
    monkeypatch.setattr(
        tr, "TrainStream", lambda: SimpleNamespace(sample=lambda seed, i: (i + 5, w))
    )

    def stub_step(m, h, o, word, *, mode, index, seed, document):
        assert word is w and seed == 4901 and mode == "A" and document == index + 5
        return dict(
            targets=T, step=index, preupdate_nll=2.0, work=dict(literal_queries=T)
        )

    monkeypatch.setattr(tr, "step", stub_step)
    for key in ("synchronize", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, key, lambda: 0)

    def snapshot(lst):
        def put(folder, b, i, *args):
            assert folder == d and b["seed"] == 4901
            lst.append(i)
            return dict(step=i, bytes=1, sha256=str(i))

        return put

    monkeypatch.setattr(tr, "save", snapshot(recovery))
    monkeypatch.setattr(tr, "milestone", snapshot(weights))
    r = tr.train(job)
    assert (
        recovery == list(range(128, 1025, 128))
        and weights == list(range(32, 1025, 32))
        and prog == [(i, i * T) for i in range(1, 1025)]
        and r["preupdate_train_nll"] == 2.0
        and r["targets"] == 4194304
    )
    ms = [json.loads(line) for line in (d / "clock.jsonl").read_text().splitlines()]
    assert [x["step"] for x in ms] == weights and all(
        a["wall_s"] < b["wall_s"] for a, b in zip(ms, ms[1:])
    )
    assert load(d / "start-state.json")["binding"] == r["binding"]


def test_score_ingest_rejects_misaligned_documents_and_keeps_misses(tmp_path):
    d = tmp_path / "cal-AB"
    d.mkdir()
    key = "A-4901@1024"
    rows = [dict(sha256="1" * 64), dict(sha256="2" * 64)]
    write_json(
        d / "frozen.json",
        dict(
            job="cal-AB",
            split="calibration",
            documents=2,
            units=2,
            endpoints={key: dict(arm="A", sha256="bound")},
        ),
    )
    write_json(d / "stream.json", dict(documents=2))
    f = d / (key + ".jsonl")
    data = []
    for i, row in enumerate(rows, 1):
        groups = dict(
            miss=dict(targets=T, nll_sum=i * T),
            **{
                k: dict(targets=0, nll_sum=0)
                for k in ("16_to_128", "129_to_1024", "over_1024")
            },
        )
        cells = [
            dict(
                horizon=h,
                sham="none",
                nll_sum=i * T,
                targets=T,
                work=dict(literal_hits=0, literal_misses=T),
                strata=dict(literal=groups),
                head_score_s=0.1,
            )
            for h in (128, 1024, T)
        ]
        data.append(
            dict(
                document_sha=row["sha256"],
                host_prefills=1,
                host_tokens=T,
                host_prefill_s=0.1,
                cells=cells,
            )
        )
    f.write_text("".join(json.dumps(x) + "\n" for x in data))
    write_json(
        d / "complete.json",
        dict(
            kind="calibration",
            units=2,
            freeze_sha=sha(d / "frozen.json"),
            stream_sha=sha(d / "stream.json"),
            endpoints={
                key: dict(
                    windows=2,
                    scores_sha=sha(f),
                    checkpoint_sha="bound",
                    decode_probe={},
                )
            },
            cpu_core_s=[0, 0],
            gpu_peak_alloc=0,
            gpu_peak_reserved=0,
            process_RSS_peak_KiB=0,
        ),
    )
    write_json(
        d / "charge.json",
        dict(
            rc=0, renewed=True, complete_sha=sha(d / "complete.json"), witnessed_units=2
        ),
    )
    t, stats, _ = read_stage(tmp_path, "cal-AB", rows)
    assert t[key][T, "none"].tolist() == [1.0, 2.0]
    assert stats["endpoints"][key]["cells"]["4096:none"]["all_byte_nll"] == 1.5
    with pytest.raises(ValueError, match="pairing"):
        read_stage(tmp_path, "cal-AB", rows[::-1])
