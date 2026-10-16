"""Amendment #3 source-named CPU tests; X2b protocol remains unchanged."""
import numpy as np
import pytest

from experiments.expressivity_tasks.run_invertible_stage2 import parse_args, pilot_gate


def cli(*extra):
    return parse_args(["--arm", "e97", "--seed", "0", "--drivers", "128", "--out", "unused", *extra])


def test_stage2c_legacy_cli_retains_x2b_recipe():
    args = cli()
    assert args.lr == 5e-5 and args.protocol_amendment == 2 and not args.pilot


def test_stage2c_cli_requires_explicit_amendment_for_lr_or_pilot():
    with pytest.raises(SystemExit):
        cli("--lr", "0.0002")
    with pytest.raises(SystemExit):
        cli("--pilot")
    args = cli("--protocol-amendment", "3", "--lr", "0.0002", "--pilot")
    assert args.lr == 2e-4 and args.pilot
    for bad in ["nan", "inf", "0", "-1"]:
        with pytest.raises(SystemExit):
            cli("--protocol-amendment", "3", "--lr", bad)


def test_stage2c_pilot_gate_rescues_decline_but_rejects_stall():
    falling = pilot_gate(np.linspace(10., 8., 3000).tolist())
    assert falling["passed"] and falling["ols_slope_nats_per_update"] < 0
    assert falling["relative_improvement"] > .01
    assert not pilot_gate([8.68] * 3000)["passed"]
    assert not pilot_gate(np.linspace(8., 10., 3000).tolist())["passed"]


def test_stage2c_pilot_gate_requires_both_window_and_negative_slope():
    # Better endpoints alone must not pass a rising OLS trend.
    losses = np.full(3000, 10.)
    losses[1000:1250] = 10.
    losses[1250:2000] = 1.
    losses[2000:2750] = 20.
    losses[2750:] = 9.
    gate = pilot_gate(losses.tolist())
    assert gate["relative_improvement"] > .01
    assert gate["ols_slope_nats_per_update"] > 0
    assert not gate["passed"]
    # Exactly1% is not >1%, despite negative slope.
    losses = np.linspace(10., 9.9, 3000)
    losses[1000:1250] = 10.
    losses[2750:] = 9.9
    assert not pilot_gate(losses.tolist())["passed"]


def test_stage2c_first_pass_binds_both_arms_and_decision_is_immutable(tmp_path, monkeypatch):
    import json
    from experiments.expressivity_tasks import invertible_stage2c as protocol
    calls = []

    def audit(root, arm, tag, lr):
        calls.append((arm, tag))
        return {"lr": lr, "gate": {"passed": arm == "e97"}, "data_receipt": {"same": True}}

    monkeypatch.setattr(protocol, "audit_pilot", audit)
    assert protocol.gate(tmp_path, "2e-4")
    d = json.loads((tmp_path / "lr_decision.json").read_text())
    assert d["lr"] == 2e-4 and d["status"] == "passed"
    assert calls == [("e97", "2e-4"), ("gdn2", "2e-4")]
    assert d["fairness_rule"] == protocol.FAIRNESS
    with pytest.raises(FileExistsError):
        protocol.gate(tmp_path, "1e-3")


def test_stage2c_failed_first_rate_allows_only_one_paired_escalation(tmp_path, monkeypatch):
    import json
    from experiments.expressivity_tasks import invertible_stage2c as protocol

    def audit(root, arm, tag, lr):
        return {"lr": lr, "gate": {"passed": tag == "1e-3"}, "data_receipt": {"same": True}}

    monkeypatch.setattr(protocol, "audit_pilot", audit)
    assert not protocol.gate(tmp_path, "2e-4")
    first_receipt = (tmp_path / "pilot_gate_2e-4.json").read_bytes()
    assert not (tmp_path / "lr_decision.json").exists()
    assert protocol.gate(tmp_path, "1e-3")
    d = json.loads((tmp_path / "lr_decision.json").read_text())
    assert d["lr"] == 1e-3 and len(d["trials"]) == 2
    assert (tmp_path / "pilot_gate_2e-4.json").read_bytes() == first_receipt


def test_stage2c_gate_fails_closed_when_both_rates_stall(tmp_path, monkeypatch):
    import json
    from experiments.expressivity_tasks import invertible_stage2c as protocol
    monkeypatch.setattr(protocol, "audit_pilot", lambda root, arm, tag, lr:
                        {"lr": lr, "gate": {"passed": False}, "data_receipt": {"same": True}})
    assert not protocol.gate(tmp_path, "2e-4")
    assert not protocol.gate(tmp_path, "1e-3")
    d = json.loads((tmp_path / "lr_decision.json").read_text())
    assert d["status"] == "failed_both_rates" and d["lr"] is None and len(d["trials"]) == 2


@pytest.mark.parametrize("losses", [[1.] * 2999, [1.] * 3001, [float("nan")] * 3000, [0.] * 3000])
def test_stage2c_pilot_gate_fails_closed_on_invalid_losses(losses):
    with pytest.raises(ValueError):
        pilot_gate(losses)
