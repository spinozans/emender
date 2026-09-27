"""First-commit first-party authority boundaries.

The checked registry intentionally has no first-party source revision yet.  A
second handoff, after the implementation/source-archive commit is pinned, owns
collection generation, sealed overlap execution, and admission coverage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ndm.e97_acquisition_controller import ToolExecution
from ndm.e97_first_party_read_observe import (
    CHECKED_IN_REGISTRY,
    _ReplayState,
    _read_snapshot_file,
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
    assert execution[0].startswith("/proc/self/fd/")
    assert execution[1] == f"/proc/self/fd/{execution.program_fd}"
    execution.close()

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


def test_validator_executes_private_verified_bytes_when_inputs_swap_before_popen(tmp_path, monkeypatch):
    import sys
    import ndm.e97_first_party_read_observe as first_party

    program = tmp_path / "validator.py"
    program.write_text(
        "import json, os, sys\n"
        "spec = json.load(os.fdopen(os.dup(int(sys.argv[4]))))\n"
        "terminal = json.load(os.fdopen(os.dup(int(sys.argv[6]))))\n"
        "if spec != {'identity': 'sealed'} or terminal != {'actions': []}: raise SystemExit(99)\n"
        "print(json.dumps({'mode':'focused','status':'pass','action_count':0}, sort_keys=True, separators=(',', ':')))\n"
    )
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    bundle = {"validator": {"focused_argv": validator_logical_argv("focused")}}
    spec = {"program_sha256": digest, "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    logical, execution = _validator_execution_argv(
        bundle, spec, mode="focused", validator_program=program, trusted_validator_sha256=digest,
    )
    spec_path, terminal_path = tmp_path / "spec.json", tmp_path / "terminal.json"
    spec_path.write_text('{"identity":"sealed"}')
    terminal_path.write_text('{"actions":[]}')
    real_popen = first_party.subprocess.Popen

    def swap_then_spawn(*args, **kwargs):
        # Every child input is an inherited unlinked descriptor rather than a
        # source pathname that an exchange immediately before Popen can alter.
        assert args[0][1] == f"/proc/self/fd/{execution.program_fd}"
        assert args[0][4:8:2] == ["--spec-fd", "--terminal-fd"]
        assert args[0][5].isdigit() and args[0][7].isdigit()
        assert len(kwargs["pass_fds"]) == 4
        program.write_text("raise SystemExit(99)\n")
        spec_path.write_text('{"identity":"attacker"}')
        terminal_path.write_text('{"actions":["attacker"]}')
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(first_party.subprocess, "Popen", swap_then_spawn)
    result = first_party._run_validator(
        logical, execution, spec_path, terminal_path, mode="focused", limit=1024,
        timeout=5, processes=1, expected_action_count=0,
    )
    assert result["output"] == {"mode": "focused", "status": "pass", "action_count": 0}
    assert result["spec_payload_sha256"] == hashlib.sha256(b'{"identity":"sealed"}').hexdigest()
    assert result["terminal_payload_sha256"] == hashlib.sha256(b'{"actions":[]}').hexdigest()


def test_validate_replay_receipt_binds_exact_inherited_validator_payloads(tmp_path, monkeypatch):
    import sys
    import ndm.e97_first_party_read_observe as first_party

    program = tmp_path / "validator.py"
    program.write_text(
        "import json, sys\n"
        "print(json.dumps({'mode':sys.argv[2],'status':'pass','action_count':0}, sort_keys=True, separators=(',', ':')))\n"
    )
    program_digest = hashlib.sha256(program.read_bytes()).hexdigest()
    limits = {"disk_bytes": 1024, "output_bytes": 1024, "seconds": 5, "processes": 1}
    bundle = {
        "limits": limits,
        "task": {"fixture_tree_digest": "tree", "identity": "identity", "generator_source_digest": "a" * 64},
        "runtime": {"controller_digest": "controller", "tool_schema_digest": "tools", "system_prompt_sha256": "prompt", "schema_digest": "schema"},
        "validator": {"focused_argv": validator_logical_argv("focused"), "regression_argv": validator_logical_argv("regression")},
    }
    spec = {"task_identity": "identity", "program_sha256": program_digest,
            "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    terminal = {"actions": [], "metadata": {"controller_build_sha256": "controller", "tool_schema_sha256": "tools",
                "system_prompt_sha256": "prompt", "configured_limits": limits}}

    class Executor:
        def close(self):
            pass

    monkeypatch.setattr(first_party, "_tree_digest", lambda _root: "tree")
    monkeypatch.setattr(first_party, "_verified_loaded_controller_digest", lambda _archive: "controller")
    monkeypatch.setattr(first_party, "_replay_actions", lambda *_args: _ReplayState(Executor(), NoProgressDetector(), {}, {}))
    fixture = tmp_path / "fixture"; fixture.mkdir()
    receipt = first_party.validate_replay(
        bundle, spec, fixture, terminal, runtime_schema_digest="schema",
        validator_program=program, trusted_validator_sha256=program_digest,
    )
    spec_payload = (first_party.canonical_json(spec) + "\n").encode()
    terminal_payload = (first_party.canonical_json(terminal) + "\n").encode()
    assert receipt["validator_spec_payload_sha256"] == hashlib.sha256(spec_payload).hexdigest()
    assert receipt["validator_terminal_payload_sha256"] == hashlib.sha256(terminal_payload).hexdigest()
    assert {result["spec_payload_sha256"] for result in receipt["validators"].values()} == {
        receipt["validator_spec_payload_sha256"]}
    assert {result["terminal_payload_sha256"] for result in receipt["validators"].values()} == {
        receipt["validator_terminal_payload_sha256"]}


def test_validator_pinned_descriptors_close_when_spawn_fails(tmp_path, monkeypatch):
    import sys
    import ndm.e97_first_party_read_observe as first_party

    program = tmp_path / "validator.py"
    program.write_text("print('unreachable')\n")
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    bundle = {"validator": {"focused_argv": validator_logical_argv("focused")}}
    spec = {"program_sha256": digest, "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    logical, execution = _validator_execution_argv(
        bundle, spec, mode="focused", validator_program=program, trusted_validator_sha256=digest,
    )
    interpreter_fd, program_fd = execution.interpreter_fd, execution.program_fd
    spec_path, terminal_path = tmp_path / "spec.json", tmp_path / "terminal.json"
    spec_path.write_text("{}"); terminal_path.write_text("{}")

    def fail_spawn(*_args, **_kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(first_party.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="spawn"):
        first_party._run_validator(
            logical, execution, spec_path, terminal_path, mode="focused", limit=1024,
            timeout=5, processes=1, expected_action_count=0,
        )
    assert execution.interpreter_fd == execution.program_fd == -1
    for descriptor in (interpreter_fd, program_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_validate_replay_closes_all_pinned_descriptors_on_pre_run_failure(tmp_path, monkeypatch):
    import sys
    import ndm.e97_first_party_read_observe as first_party

    program = tmp_path / "validator.py"; program.write_text("print('unused')\n")
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    interpreter = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    bundle = {
        "limits": {"disk_bytes": 1, "output_bytes": 1024, "seconds": 5, "processes": 1},
        "task": {"fixture_tree_digest": "tree", "identity": "identity", "generator_source_digest": "a" * 64},
        "runtime": {"controller_digest": "controller", "tool_schema_digest": "tools", "system_prompt_sha256": "prompt", "schema_digest": "schema"},
        "validator": {"focused_argv": validator_logical_argv("focused"), "regression_argv": validator_logical_argv("regression")},
    }
    spec = {"task_identity": "identity", "program_sha256": digest, "interpreter_sha256": interpreter}
    terminal = {"metadata": {"controller_build_sha256": "controller", "tool_schema_sha256": "tools", "system_prompt_sha256": "prompt", "configured_limits": bundle["limits"]}}
    opened = []
    original = first_party._validator_execution_argv

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        opened.append(result[1])
        return result

    monkeypatch.setattr(first_party, "_validator_execution_argv", capture)
    monkeypatch.setattr(first_party, "_verified_loaded_controller_digest", lambda _archive: "controller")
    monkeypatch.setattr(first_party, "_tree_digest", lambda _root: "tree")
    monkeypatch.setattr(first_party, "_replay_actions", lambda *_args: (_ for _ in ()).throw(ValueError("replay failed")))
    fixture = tmp_path / "fixture"; fixture.mkdir()
    with pytest.raises(ValueError, match="replay failed"):
        first_party.validate_replay(
            bundle, spec, fixture, terminal, runtime_schema_digest="schema",
            validator_program=program, trusted_validator_sha256=digest,
        )
    assert len(opened) == 2
    assert all(item.interpreter_fd == item.program_fd == -1 for item in opened)


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


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_validator_cli_snapshot_rejects_link_or_fifo(tmp_path, kind):
    from scripts.e97_first_party_validator import _snapshot_json

    target = tmp_path / "spec.json"
    if kind == "symlink":
        outside = tmp_path / "outside.json"; outside.write_text("{}")
        target.symlink_to(outside)
    else:
        os.mkfifo(target)
    with pytest.raises(ValueError, match="validator spec failed"):
        _snapshot_json(target, name="spec", maximum=1024)


def test_validator_cli_accepts_safe_pathnames_and_inherited_regular_descriptors(tmp_path):
    spec = {
        "schema": "emender-e97-first-party-validator-v2",
        "task_identity": "a" * 64,
        "fixture_tree_digest": "b" * 64,
        "archive_sha256": "c" * 64,
        "expected_token": "private-token",
        "required_read_path": "facts/token.txt",
        "program_sha256": "d" * 64,
        "interpreter_sha256": "e" * 64,
        "minefield": {"allowed_tools": ["list_files", "read"], "forbidden_paths": ["/", ".."]},
    }
    terminal = {"actions": []}
    spec_path, terminal_path = tmp_path / "spec.json", tmp_path / "terminal.json"
    spec_path.write_text(json.dumps(spec))
    terminal_path.write_text(json.dumps(terminal))
    command = [sys.executable, "scripts/e97_first_party_validator.py", "--mode", "regression"]
    path_result = subprocess.run(
        [*command, "--spec", str(spec_path), "--terminal", str(terminal_path)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(path_result.stdout) == {"action_count": 0, "mode": "regression", "status": "pass"}

    spec_fd = os.open(spec_path, os.O_RDONLY | os.O_CLOEXEC)
    terminal_fd = os.open(terminal_path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fd_result = subprocess.run(
            [*command, "--spec-fd", str(spec_fd), "--terminal-fd", str(terminal_fd)],
            check=True, capture_output=True, text=True, pass_fds=(spec_fd, terminal_fd),
        )
    finally:
        os.close(spec_fd)
        os.close(terminal_fd)
    assert json.loads(fd_result.stdout) == {"action_count": 0, "mode": "regression", "status": "pass"}


def test_validator_descriptor_snapshot_rejects_closed_nonregular_and_ambiguous_inputs(tmp_path):
    from scripts.e97_first_party_validator import _input_json, _snapshot_json_fd

    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    closed = os.open(payload, os.O_RDONLY | os.O_CLOEXEC)
    os.close(closed)
    with pytest.raises(ValueError, match="cannot be duplicated"):
        _snapshot_json_fd(closed, name="spec", maximum=1024)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            _snapshot_json_fd(directory, name="spec", maximum=1024)
        with pytest.raises(ValueError, match="exactly one path or descriptor"):
            _input_json(payload, directory, name="spec", maximum=1024)
    finally:
        os.close(directory)


def test_quarantine_snapshot_reader_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "authority-state.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        _read_snapshot_file(fifo, name="quarantine authority state")


def test_fixture_archive_tree_binding_rejects_mutation_between_digest_and_archive(tmp_path):
    import ndm.e97_first_party_read_observe as first_party

    fixture = tmp_path / "fixture"; fixture.mkdir()
    payload = fixture / "token.txt"; payload.write_text("token=sealed\n")
    declared_tree = first_party._tree_digest(fixture)
    # This is the exact adversarial ordering generation must reject: the tree
    # was digested, then the mutable fixture changed before it was archived.
    payload.write_text("token=attacker\n")
    archive = tmp_path / "fixture.tar"
    first_party._archive(fixture, archive)
    with pytest.raises(ValueError, match="expanded tree does not match"):
        first_party._verified_fixture_archive_payload(
            fixture, archive, tree_digest=declared_tree, disk_limit=1024)


def test_fixture_archive_tree_binding_rejects_empty_directory_mutation(tmp_path):
    import ndm.e97_first_party_read_observe as first_party

    fixture = tmp_path / "fixture"; fixture.mkdir()
    (fixture / "token.txt").write_text("token=sealed\n")
    declared_tree = first_party._tree_digest(fixture)
    (fixture / "new-empty-directory").mkdir()
    archive = tmp_path / "fixture.tar"
    first_party._archive(fixture, archive)
    with pytest.raises(ValueError, match="expanded tree does not match"):
        first_party._verified_fixture_archive_payload(
            fixture, archive, tree_digest=declared_tree, disk_limit=1024)


def test_loaded_workspace_controller_code_must_match_verified_archive(tmp_path, monkeypatch):
    import ndm.e97_first_party_read_observe as first_party

    archive_sha256 = hashlib.sha256(first_party.GENERATOR_SOURCE_ARCHIVE.read_bytes()).hexdigest()

    def substituted_execute(self, tool_name, arguments):
        return ToolExecution({"changed": True}, '{"changed":true}')

    monkeypatch.setattr(first_party.WorkspaceToolExecutor, "execute", substituted_execute)
    with pytest.raises(ValueError, match="does not match verified archived source"):
        first_party._verified_loaded_controller_digest(archive_sha256)


def test_loaded_replay_actions_must_match_verified_archive(monkeypatch):
    """Patching replay itself, without touching the controller, fails closed."""

    import ndm.e97_first_party_read_observe as first_party

    archive_sha256 = hashlib.sha256(first_party.GENERATOR_SOURCE_ARCHIVE.read_bytes()).hexdigest()

    def substituted_replay(*_args, **_kwargs):
        raise AssertionError("unsealed replay implementation executed")

    monkeypatch.setattr(first_party, "_replay_actions", substituted_replay)
    with pytest.raises(ValueError, match="does not match verified archived source"):
        first_party._verified_loaded_controller_digest(archive_sha256)


def test_loaded_replay_global_alias_must_match_verified_archive(monkeypatch):
    """A first-party imported alias cannot be rebound after source verification."""

    import ndm.e97_first_party_read_observe as first_party

    archive_sha256 = hashlib.sha256(first_party.GENERATOR_SOURCE_ARCHIVE.read_bytes()).hexdigest()

    class SubstituteWorkspaceToolExecutor:
        pass

    monkeypatch.setattr(first_party, "WorkspaceToolExecutor", SubstituteWorkspaceToolExecutor)
    with pytest.raises(ValueError, match="does not match verified archived source"):
        first_party._verified_loaded_controller_digest(archive_sha256)


def test_generated_replay_state_initializer_must_match_verified_archive(monkeypatch):
    """Dataclass-generated descriptors are part of the loaded replay closure."""

    import ndm.e97_first_party_read_observe as first_party

    archive_sha256 = hashlib.sha256(first_party.GENERATOR_SOURCE_ARCHIVE.read_bytes()).hexdigest()

    def substituted_initializer(self, *_args, **_kwargs):
        self.executor = None

    monkeypatch.setattr(first_party._ReplayState, "__init__", substituted_initializer)
    with pytest.raises(ValueError, match="does not match verified archived source"):
        first_party._verified_loaded_controller_digest(archive_sha256)


def test_loaded_controller_kwdefault_must_match_verified_archive(monkeypatch):
    import ndm.e97_first_party_read_observe as first_party
    from ndm.e97_acquisition_controller import OpenAICompletionClient

    archive_sha256 = hashlib.sha256(first_party.GENERATOR_SOURCE_ARCHIVE.read_bytes()).hexdigest()
    initializer = OpenAICompletionClient.__init__
    monkeypatch.setattr(initializer, "__kwdefaults__", {
        **initializer.__kwdefaults__, "max_response_bytes": 7,
    })
    with pytest.raises(ValueError, match="does not match verified archived source"):
        first_party._verified_loaded_controller_digest(archive_sha256)


def test_positive_generation_publishes_exact_private_authority_without_fixture_scratch(tmp_path, monkeypatch):
    """Synthetic admission uses only a temporary injected registry/config root."""
    import ndm.e97_first_party_read_observe as first_party
    from ndm.e97_first_party_source_archive import build_source_archive

    private = tmp_path / "private-authority"; private.mkdir()
    manifest = private / "manifest.json"
    manifest.write_bytes(first_party.GENERATOR_MANIFEST.read_bytes())
    archive = private / "source.tar"
    build_source_archive(manifest, archive, checkout_root=Path("."))
    environment = private / "environment.json"
    overlap = private / "overlap.json"
    license_path = private / "license.json"
    environment.write_bytes(first_party.ENVIRONMENT_DESCRIPTOR.read_bytes())
    overlap.write_bytes(first_party.OVERLAP_AUDIT.read_bytes())
    license_path.write_bytes(first_party.AUTHORIZATION_LICENSE.read_bytes())
    policy = Path("docs/EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md")
    registry = json.loads(CHECKED_IN_REGISTRY.read_text())
    registry["policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
    receipts = {
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
        "overlap_sha256": hashlib.sha256(overlap.read_bytes()).hexdigest(),
    }
    registry["sources"] = [
        {
            "id": f"e97-firstparty-{split}", "kind": "first-party", "status": "admitted",
            "url": "https://example.invalid/e97-first-party", "revision": "a" * 40,
            "framework_license": "Proprietary-first-party",
            "underlying_repository_policy": "per-repository-audit-required", "task_count_claim": 2,
            "receipts": receipts, "notes": "private synthetic generation test authority",
        }
        for split in ("train", "development")
    ]
    registry_path = private / "registry.json"
    registry_path.write_text(json.dumps(registry, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(first_party, "CHECKED_IN_REGISTRY", registry_path)
    monkeypatch.setattr(first_party, "GENERATOR_MANIFEST", manifest)
    monkeypatch.setattr(first_party, "GENERATOR_SOURCE_ARCHIVE", archive)
    monkeypatch.setattr(first_party, "ENVIRONMENT_DESCRIPTOR", environment)
    monkeypatch.setattr(first_party, "OVERLAP_AUDIT", overlap)
    monkeypatch.setattr(first_party, "AUTHORIZATION_LICENSE", license_path)
    monkeypatch.setattr(first_party, "_source_revision", lambda: "a" * 40)
    # This synthetic generation test deliberately bypasses the loaded-code
    # attestation after replacing its source-revision helper above; replay
    # attestation itself is covered by the dedicated sealed-code tests.
    monkeypatch.setattr(first_party, "_loaded_replay_digest_from_archive", lambda *_args: "c" * 64)
    output = tmp_path / "published"
    first_party.generate(
        output, seed="private-synthetic-generation", registry_path=registry_path,
        registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        policy_sha256=registry["policy_sha256"],
    )
    validated = first_party.validate_generated_quarantine(output)
    published = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    assert published == dict(validated.payloads)
    assert {"source-registry.json", "generator-manifest.json", "source-archive.tar",
            "environment-descriptor.json", "overlap-firewall-audit.json",
            "authorization-license.json", "generation-receipt.json", "authority-state.json",
            "tasks.jsonl"}.issubset(published)
    assert "fixtures" not in {path.name for path in output.iterdir()}
    assert not any("fixtures" in path.parts for path in output.rglob("*"))


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
