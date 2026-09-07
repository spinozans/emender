from __future__ import annotations

from pathlib import Path
import json
import multiprocessing
import os
import stat
import time

import pytest

from ndm.e97_acquisition_controller import (
    SERVICE_ATTESTATION_SCHEMA,
    READ_OBSERVE_TOOLS,
    AcquisitionController,
    OpenAICompletionClient,
    ToolExecution,
    WorkspaceToolExecutor,
)
from ndm.e97_onpolicy_records import CANONICAL_NO_PROGRESS_OBSERVATION, sha256_json, sha256_text
from ndm.e97_agent_protocol import serialize_pi_messages


CHECKPOINT = "a" * 64
CONTROLLER_BUILD = "b" * 64
ATTESTATION = {
    "schema": SERVICE_ATTESTATION_SCHEMA, "checkpoint_path": "/fake/e97.pt",
    "checkpoint_sha256": CHECKPOINT, "args_json_sha256": "c" * 64,
    "config_sha256": "d" * 64, "weight_mode": "saved", "tokenizer": "fake-tokenizer",
    "model_id": "e97-dense-agent", "server_build_sha256": "e" * 64,
    "controller_build_sha256": CONTROLLER_BUILD, "device": "cpu", "dtype": "float32",
    "use_triton": False, "ingest_mode": "tokenwise", "runtime_image_path": "/fake/image", "runtime_image_sha256": "f" * 64,
    "tool_schema_sha256": "PLACEHOLDER", "system_prompt_override_sha256": sha256_text(""),
    "runtime_identity_schema": "emender-e97-runtime-identity-v1", "max_output_tokens": 512, "max_sessions": 8,
    "python_implementation": "CPython", "python_version": "3.12", "torch_version": "fake", "cuda_runtime": "",
    "cuda_available": False, "platform": "Linux", "machine": "x86_64",
}
ATTESTATION["tool_schema_sha256"] = sha256_json(READ_OBSERVE_TOOLS)


class FakeClient:
    def __init__(self, messages, *, attestation=ATTESTATION, advance_clock=None):
        self.messages = list(messages)
        self.attestation = attestation
        self.advance_clock = advance_clock
        self.requests = []
        self.timeouts = []

    def complete(self, request, *, deadline):
        self.requests.append(request)
        self.timeouts.append(deadline)
        if self.advance_clock is not None:
            self.advance_clock()
        response = {"model": "e97-dense-agent", "choices": [{"message": self.messages.pop(0)}],
                    "usage": {"completion_tokens": 1}}
        if self.attestation is not None:
            response["emender_service_attestation"] = self.attestation
            response["emender_request_identity"] = {
                "messages_sha256": sha256_text(json.dumps(request["messages"], sort_keys=True, separators=(",", ":"))),
                "system_prompt_sha256": sha256_text(request["messages"][0]["content"]),
                "tool_schema_sha256": sha256_json(request["tools"]),
                "request_sha256": sha256_json(request),
            }
        return response


def action(name="read", arguments='{"path":"note.txt","offset":1,"limit":1}', call_id="call-read"):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


class FakeExecutor:
    def __init__(self, results, states):
        self.results = list(results)
        self.states = list(states)
        self.calls = []

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.results.pop(0)

    def workspace_state(self):
        return self.states.pop(0)


def controller(client, executor, **kwargs):
    return AcquisitionController(
        client, executor, checkpoint_sha256=CHECKPOINT,
        expected_service_attestation=ATTESTATION,
        controller_build_sha256=CONTROLLER_BUILD,
        **{"max_turns": 8, "max_seconds": 30, **kwargs},
    )


def test_workspace_tools_contain_paths_symlinks_and_bound_output(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note.txt").write_text("first\n" + "x" * 100 + "\n" + "y" * 100 + "\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("private\n")
    (root / "escape.txt").symlink_to(outside)
    executor = WorkspaceToolExecutor(root, max_output_bytes=140, max_list_entries=1)

    escaped = executor.execute("read", {"path": "../outside.txt", "offset": 1, "limit": 1})
    linked = executor.execute("read", {"path": "escape.txt", "offset": 1, "limit": 1})
    listed = executor.execute("list_files", {"path": ".", "depth": 1, "limit": 1})
    bounded = executor.execute("read", {"path": "note.txt", "offset": 1, "limit": 2})

    assert escaped.is_error and linked.is_error
    assert "escapes workspace" in escaped.effective_observation
    assert "escape.txt" in listed.effective_observation
    assert listed.raw_observation["truncated"] is True
    assert bounded.raw_observation["ok"] is True
    assert len(bounded.raw_observation["lines"]) == 2
    assert '"truncated":true' in bounded.effective_observation
    assert len(bounded.effective_observation.encode()) <= 140


def test_changed_observation_or_workspace_allows_same_action_again():
    client = FakeClient([action(), action(call_id="call-read-2"), {"role": "assistant", "content": "Final: done"}])
    executor = FakeExecutor(
        [
            ToolExecution({"value": "first"}, '{"ok":true,"value":"first"}', source_ledger={"note.txt": "one"}),
            ToolExecution({"value": "second"}, '{"ok":true,"value":"second"}', source_ledger={"note.txt": "two"}),
        ],
        [{"tree_sha256": "one"}, {"tree_sha256": "two"}],
    )
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")

    assert result.status == "success"
    assert [receipt.decision for receipt in result.actions] == ["continue", "continue"]
    assert result.actions[0].action_fingerprint == result.actions[1].action_fingerprint
    assert result.actions[0].progress_fingerprint != result.actions[1].progress_fingerprint


def test_raw_result_is_retained_while_effective_recovery_replaces_next_context():
    client = FakeClient([action(), action(call_id="call-read-2"), {"role": "assistant", "content": "Final: grounded blocker"}])
    execution = ToolExecution({"private": "authentic"}, '{"ok":true,"value":"same"}', source_ledger={"note.txt": "same"})
    executor = FakeExecutor([execution, execution], [{"tree_sha256": "same"}, {"tree_sha256": "same"}])
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")

    assert result.status == "success"
    assert result.actions[1].raw_observation == {"private": "authentic"}
    assert result.actions[1].effective_observation == CANONICAL_NO_PROGRESS_OBSERVATION
    assert result.actions[1].observation_sha256 != result.actions[1].raw_observation_sha256
    assert result.messages[-2] == {
        "role": "tool", "tool_call_id": "call-read-2", "content": CANONICAL_NO_PROGRESS_OBSERVATION,
    }
    assert serialize_pi_messages(result.messages[:-1]).endswith("Assistant:\n")
    assert result.metadata["checkpoint_sha256"] == "a" * 64
    assert "serialized_messages_sha256" in result.metadata
    assert result.actions[0].completion_tokens == 1
    assert result.metadata["completion_usage"] == [
        {"sequence": 0, "completion_tokens": 1},
        {"sequence": 1, "completion_tokens": 1},
        {"sequence": 2, "completion_tokens": 1},
    ]
    assert result.metadata["completion_tokens"] == 3


def test_one_recovery_then_same_stale_pair_terminates_without_another_model_request():
    client = FakeClient([action(), action(call_id="call-read-2"), action(call_id="call-read-3")])
    execution = ToolExecution({"value": "same"}, '{"ok":true,"value":"same"}', source_ledger={"note.txt": "same"})
    executor = FakeExecutor([execution, execution, execution], [{"tree_sha256": "same"}] * 3)
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")

    assert result.status == "no_progress_terminated"
    assert [receipt.decision for receipt in result.actions] == ["continue", "inject_no_progress", "terminate"]
    assert result.actions[-1].raw_observation == {"value": "same"}
    assert len(result.actions) == 3


def test_mapping_function_arguments_are_rejected_at_controller_ingress():
    client = FakeClient([action(arguments={"path": "note.txt"})])
    executor = FakeExecutor([], [])
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"
    assert "function.arguments must be a string" in result.error
    assert executor.calls == []


def test_unknown_model_tool_is_rejected_before_executor_invocation():
    client = FakeClient([action(name="bash", arguments='{"command":"id"}', call_id="call-bash")])
    executor = FakeExecutor([], [])
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")

    assert result.status == "completion_error"
    assert executor.calls == []


def test_each_completion_requires_exact_startup_service_attestation():
    bad_attestation = {**ATTESTATION, "checkpoint_sha256": "c" * 64}
    client = FakeClient([action()], attestation=bad_attestation)
    executor = FakeExecutor([], [])
    result = controller(client, executor).run(system_prompt="system", user_prompt="read")

    assert result.status == "completion_error"
    assert executor.calls == []


def test_controller_rejects_model_and_request_binding_mismatch():
    class WrongBindingClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            response["model"] = "other"
            return response

    result = controller(WrongBindingClient([action()]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"

    class WrongRequestClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            response["emender_request_identity"]["system_prompt_sha256"] = "0" * 64
            return response

    result = controller(WrongRequestClient([action()]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"

    class ControlMutationClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            altered = dict(request)
            altered["temperature"] = 1
            response["emender_request_identity"]["request_sha256"] = sha256_json(altered)
            return response

    result = controller(ControlMutationClient([action()]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"

    class MaxTokenMutationClient(ControlMutationClient):
        def complete(self, request, *, deadline):
            response = FakeClient.complete(self, request, deadline=deadline)
            altered = dict(request)
            altered["max_completion_tokens"] = 1
            response["emender_request_identity"]["request_sha256"] = sha256_json(altered)
            return response

    result = controller(MaxTokenMutationClient([action()]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"


def test_record_separator_suffix_final_is_rejected_from_exact_transcript():
    client = FakeClient([{"role": "assistant", "content": "Final: safe" + "\x1e" + "untrusted"}])
    result = controller(client, FakeExecutor([], [])).run(system_prompt="system", user_prompt="read")

    assert result.status == "completion_error"
    assert not result.messages[-1].get("content", "").endswith("untrusted")


def test_completion_usage_and_byte_ceiling_fail_closed():
    class MissingUsageClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            del response["usage"]
            return response

    missing = controller(MissingUsageClient([{"role": "assistant", "content": "Final: done"}]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert missing.status == "completion_error"
    assert "usage.completion_tokens" in missing.error

    class ZeroUsageClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            response["usage"] = {"completion_tokens": 0}
            return response

    zero = controller(ZeroUsageClient([{"role": "assistant", "content": "Final: done"}]), FakeExecutor([], [])).run(
        system_prompt="system", user_prompt="read")
    assert zero.status == "completion_error"
    assert "usage.completion_tokens" in zero.error

    class ExcessUsageClient(FakeClient):
        def complete(self, request, *, deadline):
            response = super().complete(request, deadline=deadline)
            response["usage"] = {"completion_tokens": 18}
            return response

    excess = controller(ExcessUsageClient([{"role": "assistant", "content": "Final: done"}]), FakeExecutor([], []), max_completion_tokens=17).run(
        system_prompt="system", user_prompt="read")
    assert excess.status == "completion_error"

    oversized = controller(FakeClient([{"role": "assistant", "content": "Final: " + "é" * 69}]), FakeExecutor([], []), max_completion_tokens=17).run(
        system_prompt="system", user_prompt="read")
    assert oversized.status == "completion_error"
    assert "byte ceiling" in oversized.error


def test_deadline_rechecked_before_model_action_and_token_limit_is_controller_owned():
    client = FakeClient([{"role": "assistant", "content": "Final: done"}])
    result = controller(client, FakeExecutor([], []), max_completion_tokens=17).run(
        system_prompt="system", user_prompt="read")

    assert result.status == "success"
    assert result.metadata["model_id"] == "e97-dense-agent"


def test_openai_client_rejects_oversized_response_without_unbounded_read(monkeypatch):
    class Response:
        headers = {}

        def __init__(self):
            self.read_sizes = []

        def read(self, size):
            self.read_sizes.append(size)
            return b"x" * size

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    response = Response()
    monkeypatch.setattr("ndm.e97_acquisition_controller.urlopen", lambda *args, **kwargs: response)
    client = OpenAICompletionClient("http://example.invalid/v1/chat/completions", max_response_bytes=64, clock=lambda: 0)
    with pytest.raises(Exception, match="exceeds byte limit"):
        client.complete({}, deadline=1)
    assert response.read_sizes == [65]


def test_descriptor_open_rejects_deterministic_symlink_swap(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("safe\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("private\n")
    executor = WorkspaceToolExecutor(root)
    original_open = os.open

    def swapped_open(name, flags, *args, **kwargs):
        if name == "target.txt" and kwargs.get("dir_fd") is not None:
            target.unlink()
            target.symlink_to(outside)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr("ndm.e97_acquisition_controller.os.open", swapped_open)
    result = executor.execute("read", {"path": "target.txt", "offset": 1, "limit": 1})
    assert result.is_error
    assert "private" not in result.effective_observation


def test_descriptor_read_enforces_cumulative_growth_bound(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "many.txt").write_text("x\n" * 100)
    executor = WorkspaceToolExecutor(root, max_read_bytes=64)
    original_fstat = os.fstat

    def stale_small_stat(fd):
        actual = original_fstat(fd)
        if stat.S_ISREG(actual.st_mode):
            return type("Stat", (), {"st_mode": actual.st_mode, "st_size": 1})()
        return actual

    monkeypatch.setattr("ndm.e97_acquisition_controller.os.fstat", stale_small_stat)
    result = executor.execute("read", {"path": "many.txt", "offset": 1, "limit": 1})
    assert result.is_error
    assert '"code":"input_limit"' in result.effective_observation


def test_deadline_returns_for_truly_blocking_completion_and_tool_work():
    class BlockingClient:
        def complete(self, request, *, deadline):
            time.sleep(0.2)
            return {}

    started = time.monotonic()
    completion_result = controller(BlockingClient(), FakeExecutor([], []), max_seconds=0.03).run(
        system_prompt="system", user_prompt="read")
    assert completion_result.status == "time_limit"
    assert time.monotonic() - started < 0.15

    class BlockingExecutor:
        def execute(self, tool_name, arguments):
            time.sleep(0.2)
            return ToolExecution({}, "{}")

        def workspace_state(self):
            return {"tree": "late"}

    tool_result = controller(FakeClient([action()]), BlockingExecutor(), max_seconds=0.03).run(
        system_prompt="system", user_prompt="read")
    assert tool_result.status == "time_limit"
    assert not multiprocessing.active_children()

    for _ in range(3):
        result = controller(BlockingClient(), FakeExecutor([], []), max_seconds=0.02).run(
            system_prompt="system", user_prompt="read")
        assert result.status == "time_limit"
    assert not multiprocessing.active_children()


def test_json_ipc_bounds_oversized_results_and_exceptions_reap_workers():
    class OversizedClient:
        def complete(self, request, *, deadline):
            return {"model": "e97-dense-agent", "blob": "x" * ((1 << 20) + 1)}

    result = controller(OversizedClient(), FakeExecutor([], [])).run(system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"
    assert not multiprocessing.active_children()

    class HugeErrorClient:
        def complete(self, request, *, deadline):
            raise ValueError("x" * 100_000)

    result = controller(HugeErrorClient(), FakeExecutor([], [])).run(system_prompt="system", user_prompt="read")
    assert result.status == "completion_error"
    assert result.error is not None and len(result.error.encode("utf-8")) < 5000
    assert not multiprocessing.active_children()


def test_read_rejects_large_single_line_before_opening(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "large.txt").write_bytes(b"x" * 65)
    executor = WorkspaceToolExecutor(root, max_read_bytes=64)

    result = executor.execute("read", {"path": "large.txt", "offset": 1, "limit": 1})

    assert result.is_error
    assert '"code":"input_limit"' in result.effective_observation
