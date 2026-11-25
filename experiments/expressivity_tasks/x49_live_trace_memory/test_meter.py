"""Authorization refusal and CPU-only exercises of the actual checkpoint/meter code. No launch."""

import json
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import meter
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import (
    state_digest,
    milestone,
)
from experiments.expressivity_tasks.x49_live_trace_memory.meter_run import (
    arm_probe,
    fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import fresh


def test_missing_authorization_refuses_before_broker_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(meter, "OUT", tmp_path)
    with pytest.raises(ValueError, match="no explicit separate"):
        meter.authorize()
    assert not list(tmp_path.iterdir())


def test_meter_approval_must_bind_exact_scope_and_cost():
    # Fixture only: validate schema in memory, never create a reservation/authority file.
    ref = "f" * 40
    stub = dict(
        scope="x49-synthetic-meter-v1",
        spec_sha256=meter.SPEC,
        qualified_ref=ref,
        card_seconds=900,
        gpus=1,
        attempts=1,
        max_physical_updates=16,
        natural_training=False,
        natural_evaluation=False,
        remote=False,
        operator_note="schema fixture only",
    )
    meter.validate_approval(stub, ref)
    for k, v in [
        ("qualified_ref", "a" * 40),
        ("gpus", True),
        ("attempts", 2),
        ("card_seconds", 901),
        ("max_physical_updates", 17),
        ("natural_training", True),
        ("natural_evaluation", True),
        ("remote", True),
        ("spec_sha256", "?"),
        ("operator_note", ""),
    ]:
        with pytest.raises(ValueError):
            meter.validate_approval(dict(stub, **{k: v}), ref)
    with pytest.raises(ValueError):
        meter.validate_approval(dict(stub, override=True), ref)


def test_logical_state_digest_binds_shape_precision_values_and_nested_optimizer_state():
    a = torch.arange(12, dtype=torch.float32)
    orig = {"m": a, 0: ("a", 1)}
    d = state_digest(orig)
    assert state_digest({0: ("a", 1), "m": a.clone()}) == d
    for x in (a.reshape(3, 4), a.to(torch.float64), a + 1):
        assert state_digest(dict(orig, m=x)) != d
    assert state_digest({0: ["a", 1], "m": a}) != d


def test_full_meter_sequence_tiny_cpu_host(tmp_path):
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4901)
            host, head, opts = fresh("C")
            progress = []
            b = dict(scope="test-only", arm="C", source="tiny")
            r = arm_probe(
                tmp_path,
                "C",
                host,
                head,
                opts,
                binding=b,
                report=progress.append,
                width=8,
            )
        assert (
            r["exact_successor"]
            and progress == [1, 2, 3, 4]
            and [s["logical"] for s in r["updates"]] == [1, 2, 3, 3]
        )
        assert sum(s["targets"] for s in r["updates"]) == 16384
        assert (
            list(map(json.loads, (tmp_path / "steps.jsonl").read_text().splitlines()))
            == r["updates"]
        )
        weights = torch.load(
            tmp_path / r["weights"]["file"], map_location="cpu", weights_only=True
        )
        assert (
            set(weights) == {"binding", "step", "host", "head"}
            and weights["step"] == 2
            and weights["binding"] == b
        )
        with pytest.raises(ValueError, match="already exists"):
            milestone(tmp_path, b, 2, host, head)
        ev = r["synthetic_inference"]
        assert (
            ev["host_prefills"] == 1
            and ev["host_tokens"] == 4096
            and len(ev["cells"]) == 5
        )
        assert [(c["horizon"], c["sham"]) for c in ev["cells"]] == [
            (128, "none"),
            (1024, "none"),
            (4096, "none"),
            (4096, "literal"),
            (4096, "trace"),
        ]
        for c in ev["cells"]:
            w = c["work"]
            assert (
                c["targets"] == w["literal_queries"] == w["private_queries"] == 4096
                and w["private_writes"] == 4095
                and c["nll_sum"] >= 0
            )
        original, lit, trace = [c["work"] for c in ev["cells"][2:]]
        for key, val in original.items():
            if not key.endswith("payloads_reassigned"):
                assert val == lit[key] == trace[key]
        assert (
            lit["literal_payloads_reassigned"] > 0 == lit["private_payloads_reassigned"]
            and trace["private_payloads_reassigned"]
            > 0
            == trace["literal_payloads_reassigned"]
        )
        assert len({fixture(i) for i in range(1, 5)}) == 4
        with pytest.raises(ValueError):
            fixture(5)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
