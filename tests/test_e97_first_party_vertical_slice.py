"""First-commit first-party authority boundaries.

The checked registry intentionally has no first-party source revision yet.  A
second handoff, after the implementation/source-archive commit is pinned, owns
collection generation, sealed overlap execution, and admission coverage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ndm.e97_acquisition_controller import ToolExecution
from ndm.e97_first_party_read_observe import (
    CHECKED_IN_REGISTRY,
    _ReplayState,
    _validator_execution_argv,
    generate,
    replay_corrective_suffix_from_failure,
    validator_logical_argv,
)
from ndm.e97_onpolicy_records import NoProgressDetector
from ndm.e97_first_party_source_archive import extract_verified_source_member, verify_checkout_components
from ndm.e97_task_lake import validate_source_registry


def test_generator_manifest_rejects_any_extra_component(tmp_path):
    manifest = json.loads(Path("configs/pi/e97-firstparty-generator-manifest-v1.json").read_text())
    manifest["components"].append({"path": "ndm/e97_artifact_store.py", "sha256": "0" * 64})
    manifest["components"].sort(key=lambda item: item["path"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="closed generator authority"):
        verify_checkout_components(path, checkout_root=Path("."))


def test_artifact_backed_validator_identity_rejects_alternate_pass_emitter(tmp_path):
    manifest = Path("configs/pi/e97-firstparty-generator-manifest-v1.json")
    archive = Path("configs/pi/e97-firstparty-source-v1.tar")
    extracted = tmp_path / "validator.py"
    trusted = extract_verified_source_member(
        manifest, archive, "scripts/e97_first_party_validator.py", extracted,
    )
    bundle = {"validator": {
        "focused_argv": validator_logical_argv("focused"),
        "regression_argv": validator_logical_argv("regression"),
    }}
    import sys
    import ndm.e97_first_party_read_observe as first_party
    spec = {"program_sha256": trusted, "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    logical, execution = _validator_execution_argv(
        bundle, spec, mode="focused", validator_program=extracted, trusted_validator_sha256=trusted,
    )
    assert logical == validator_logical_argv("focused")
    assert execution[:2] == [sys.executable, str(extracted)]

    alternate = tmp_path / "alternate-pass-emitter.py"
    alternate.write_text("print('not authoritative')\n")
    with pytest.raises(ValueError, match="program/interpreter identity"):
        _validator_execution_argv(
            bundle,
            {**spec, "program_sha256": hashlib.sha256(alternate.read_bytes()).hexdigest()},
            mode="focused", validator_program=alternate, trusted_validator_sha256=trusted,
        )
    bundle["validator"]["focused_argv"] = ["@runtime-python", "alternate-pass-emitter.py", "--mode", "focused"]
    with pytest.raises(ValueError, match="sealed logical"):
        _validator_execution_argv(
            bundle, spec, mode="focused", validator_program=extracted, trusted_validator_sha256=trusted,
        )


def test_corrective_replay_requires_explicit_bounded_completion_usage(tmp_path, monkeypatch):
    class Executor:
        def execute(self, tool_name, arguments):
            return ToolExecution({"ok": True}, '{"ok":true}')

        def workspace_state(self):
            return {"tree": "synthetic"}

        def close(self):
            pass

    def fresh_state(*args):
        return _ReplayState(Executor(), NoProgressDetector(), {}, {})

    monkeypatch.setattr("ndm.e97_first_party_read_observe._replay_actions", fresh_state)
    limits = {"turns": 4, "seconds": 10, "completion_tokens": 32, "output_bytes": 1024,
              "disk_bytes": 1024, "processes": 1}
    failed = {
        "status": "no_progress_terminated", "elapsed_seconds": 0.0,
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "actions": [{"tool_call_id": "failed", "effective_observation": "stop"}],
        "metadata": {"completion_usage": [{"sequence": 0, "completion_tokens": 1}]},
    }
    proposed = [{"tool_name": "read", "arguments": {"path": "x", "offset": 1, "limit": 1},
                 "arguments_json": '{"limit":1,"offset":1,"path":"x"}', "completion_tokens": 2}]
    receipt = replay_corrective_suffix_from_failure(
        {"limits": limits}, failed, tmp_path, proposed, "Final: ok", 1,
    )
    assert receipt["actions"][-1]["completion_tokens"] == 2
    assert receipt["metadata"]["completion_usage"] == [
        {"sequence": 0, "completion_tokens": 1},
        {"sequence": 1, "completion_tokens": 2},
        {"sequence": 2, "completion_tokens": 1},
    ]
    with pytest.raises(ValueError, match="proposed corrective action fields"):
        replay_corrective_suffix_from_failure(
            {"limits": limits}, failed, tmp_path,
            [{key: value for key, value in proposed[0].items() if key != "completion_tokens"}], "Final: ok", 1,
        )
    with pytest.raises(ValueError, match="proposed arguments or completion_tokens"):
        replay_corrective_suffix_from_failure(
            {"limits": limits}, failed, tmp_path,
            [{**proposed[0], "completion_tokens": 0}], "Final: ok", 1,
        )
    with pytest.raises(ValueError, match="final completion_tokens"):
        replay_corrective_suffix_from_failure(
            {"limits": limits}, failed, tmp_path, proposed, "Final: ok", 0,
        )
    with pytest.raises(ValueError, match="final completion_tokens"):
        replay_corrective_suffix_from_failure(
            {"limits": limits}, failed, tmp_path, proposed, "Final: ok", 33,
        )


def test_checked_registry_is_non_generatable_until_revision_pin(tmp_path):
    registry = validate_source_registry(json.loads(CHECKED_IN_REGISTRY.read_text()))
    assert len(registry["sources"]) == 3
    assert all(source["status"] == "candidate" for source in registry["sources"])
    policy = Path("docs/EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md")
    with pytest.raises(ValueError, match="lacks pre-admitted first-party split sources"):
        generate(
            tmp_path / "quarantine",
            seed="first-commit-refusal",
            registry_path=CHECKED_IN_REGISTRY,
            registry_sha256=hashlib.sha256(CHECKED_IN_REGISTRY.read_bytes()).hexdigest(),
            policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        )
    assert not (tmp_path / "quarantine").exists()
