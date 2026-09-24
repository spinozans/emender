from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from http.client import HTTPConnection
from http.server import HTTPServer
from threading import Thread

import pytest

from ndm.e97_agent_protocol import AgentProtocolError, RS
from ndm.e97_onpolicy_records import sha256_json, sha256_text
from ndm.e97_acquisition_controller import READ_OBSERVE_TOOLS
from ndm.e97_agent_server import (
    AgentCompletionService,
    RecurrentSessionStore,
    chat_completion_sse,
    make_openai_handler,
)


@dataclass(frozen=True)
class FakeCache:
    token_ids: tuple[int, ...]

    @property
    def state_bytes(self):
        return 1234


class FakeEngine:
    checkpoint = str(Path(__file__))

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.advance_calls = []
        self.checkpoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.args_json_sha256 = self.checkpoint_sha256
        self.runtime_image_sha256 = self.checkpoint_sha256
        self.controller_build_sha256 = hashlib.sha256(
            (Path(__file__).parents[1] / "ndm" / "e97_acquisition_controller.py").read_bytes()).hexdigest()
        self.server_build_sha256 = hashlib.sha256(
            (Path(__file__).parents[1] / "ndm" / "e97_agent_server.py").read_bytes()).hexdigest()

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8")

    def advance(self, token_ids, cache=None):
        consumed = tuple(token_ids)
        self.advance_calls.append((None if cache is None else cache.token_ids, consumed))
        return FakeCache((() if cache is None else cache.token_ids) + consumed)

    def generate(self, cache, *, max_new_tokens, temperature, top_p):
        text = self.outputs.pop(0)
        tokens = tuple(self.encode(text))[:max_new_tokens]
        return list(tokens), FakeCache(cache.token_ids + tokens)


def external_attestation_kwargs():
    from ndm.e97_acquisition_controller import READ_OBSERVE_TOOLS
    args_path = Path(__file__)
    args_sha = hashlib.sha256(args_path.read_bytes()).hexdigest()
    return {
        "checkpoint_sha256": hashlib.sha256(args_path.read_bytes()).hexdigest(), "checkpoint_path": str(args_path),
        "args_json_path": str(args_path), "args_json_sha256": args_sha, "config_sha256": "d" * 64,
        "weight_mode": "saved", "tokenizer": "fake-tokenizer", "device": "cpu",
        "dtype": "float32", "use_triton": False, "ingest_mode": "tokenwise",
        "runtime_image_path": str(args_path), "runtime_image_sha256": args_sha,
        "tool_schema_sha256": sha256_text(json.dumps(READ_OBSERVE_TOOLS, sort_keys=True, separators=(",", ":"))),
        "controller_build_sha256":  hashlib.sha256((Path(__file__).parents[1] / "ndm" / "e97_acquisition_controller.py").read_bytes()).hexdigest(),
        "server_build_sha256": hashlib.sha256((Path(__file__).parents[1] / "ndm" / "e97_agent_server.py").read_bytes()).hexdigest(),
    }


def tool(name):
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {"type": "object"}},
    }


def test_session_store_hit_replay_rollback_commit_and_lru():
    store = RecurrentSessionStore(max_sessions=2)

    def advance(tokens, cache=None):
        return FakeCache((() if cache is None else cache.token_ids) + tuple(tokens))

    first = store.prepare("a", [1, 2], advance)
    assert first.cache_event == "miss"
    store.discard(first)
    assert len(store) == 0
    assert store.commit(first, FakeCache((1, 2, 3))) is True

    hit = store.prepare("a", [1, 2, 3, 4], advance)
    assert hit.cache_event == "hit"
    assert hit.suffix_tokens == 1
    assert hit.prompt_cache.token_ids == (1, 2, 3, 4)
    assert store.commit(hit, FakeCache((1, 2, 3, 4, 5))) is True

    replay = store.prepare("a", [1, 9], advance)
    assert replay.cache_event == "replay"
    assert replay.suffix_tokens == 2

    stale = store.prepare("a", [1, 2, 3, 4, 5, 6], advance)
    concurrent = store.prepare("a", [1, 2, 3, 4, 5, 7], advance)
    assert store.commit(concurrent, FakeCache((1, 2, 3, 4, 5, 7, 8))) is True
    assert store.commit(stale, FakeCache((1, 2, 3, 4, 5, 6, 8))) is False

    for name in ("b", "c"):
        prepared = store.prepare(name, [10], advance)
        assert store.commit(prepared, FakeCache((10, 11))) is True
    assert len(store) == 2


def test_system_override_removes_harness_runtime_suffix():
    engine = FakeEngine(["Final: done" + RS])
    service = AgentCompletionService(engine, system_prompt_override="canonical system")
    service.prepare_completion(
        {
            "messages": [
                {"role": "system", "content": "canonical system\nCurrent working directory: /tmp/task"},
                {"role": "user", "content": "Do it."},
            ]
        },
        session_id=None,
    )
    consumed = bytes(engine.advance_calls[0][1]).decode("utf-8")
    assert consumed.startswith("System:\ncanonical system\n\nUser:")
    assert "working directory" not in consumed


def test_completion_round_trip_uses_cached_suffix_and_structured_tool_call():
    engine = FakeEngine([
        'Action: calculator\nArguments: {"expression":"2 + 3"}',
        "Final: 2 + 3 = 5." + RS,
    ])
    service = AgentCompletionService(engine, max_sessions=2)
    tools = [tool("calculator")]
    first_request = {
        "model": "e97-dense-agent",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Calculate 2 + 3."},
        ],
        "tools": tools,
        "temperature": 0,
        "max_tokens": 64,
    }

    first = service.prepare_completion(first_request, session_id="pi-session")
    assert first.diagnostics["x-emender-cache"] == "miss"
    message = first.response["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression":"2 + 3"}',
    }
    assert first.response["choices"][0]["finish_reason"] == "tool_calls"
    assert len(service.sessions) == 0
    assert service.commit(first) is True

    second_request = {
        "model": "e97-dense-agent",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Calculate 2 + 3."},
            {"role": "assistant", "content": None, "tool_calls": message["tool_calls"]},
            {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": '{"value":"5"}'},
        ],
        "tools": tools,
    }
    second = service.prepare_completion(second_request, session_id="pi-session")
    assert second.diagnostics["x-emender-cache"] == "hit"
    assert 0 < second.session.suffix_tokens < second.response["usage"]["prompt_tokens"]
    assert second.response["choices"][0]["message"]["content"] == "Final: 2 + 3 = 5."
    assert second.response["choices"][0]["finish_reason"] == "stop"
    assert service.commit(second) is True


def test_service_rejects_engine_protocol_mode_mismatch():
    engine = FakeEngine([])
    engine.private_analysis = True
    with pytest.raises(ValueError, match="does not match loaded engine"):
        AgentCompletionService(engine, private_analysis=False)


def test_private_analysis_round_trips_through_tool_turn_and_stays_out_of_final_content():
    engine = FakeEngine([
        'Analysis: "Inspect the requested file first."\nAction: read\nArguments: {"path":"README.md"}',
        'Analysis: "The observation supports the answer."\nFinal: verified\n',
    ])
    service = AgentCompletionService(engine, max_sessions=2, private_analysis=True)
    tools = [tool("read")]
    first = service.prepare_completion({
        "messages": [{"role": "user", "content": "Read it."}],
        "tools": tools,
    }, session_id="analysis-session")
    first_message = first.response["choices"][0]["message"]
    assert first_message["content"] is None
    assert first_message["reasoning_content"] == "Inspect the requested file first."
    assert first_message["tool_calls"][0]["function"]["name"] == "read"
    assert service.commit(first) is True

    second = service.prepare_completion({
        "messages": [
            {"role": "user", "content": "Read it."},
            first_message,
            {"role": "tool", "tool_call_id": first_message["tool_calls"][0]["id"],
             "content": "verified contents"},
        ],
        "tools": tools,
    }, session_id="analysis-session")
    assert second.session.cache_event == "hit"
    second_message = second.response["choices"][0]["message"]
    assert second_message["reasoning_content"] == "The observation supports the answer."
    assert second_message["content"] == "Final: verified\n"
    assert "The observation" not in second_message["content"]
    events = b"".join(chat_completion_sse(second.response))
    assert b'"reasoning_content":"The observation supports the answer."' in events


def test_private_analysis_bytes_are_part_of_recurrent_cache_prefix_identity():
    engine = FakeEngine([
        'Analysis: "original rationale"\nAction: read\nArguments: {"path":"README.md"}',
        'Analysis: "new rationale"\nFinal: done\n',
    ])
    service = AgentCompletionService(engine, max_sessions=1, private_analysis=True)
    tools = [tool("read")]
    first = service.prepare_completion({
        "messages": [{"role": "user", "content": "Read it."}], "tools": tools,
    }, session_id="analysis-prefix")
    assert service.commit(first) is True
    changed = dict(first.response["choices"][0]["message"])
    changed["reasoning_content"] = "client changed rationale"
    second = service.prepare_completion({
        "messages": [
            {"role": "user", "content": "Read it."}, changed,
            {"role": "tool", "tool_call_id": changed["tool_calls"][0]["id"], "content": "contents"},
        ],
        "tools": tools,
    }, session_id="analysis-prefix")

    assert second.session.cache_event == "replay"
    replayed_prompt = bytes(engine.advance_calls[-1][1]).decode("utf-8")
    assert "client changed rationale" in replayed_prompt
    assert "original rationale" not in replayed_prompt


def test_v2_tool_only_mode_rejects_unstructured_final():
    service = AgentCompletionService(
        FakeEngine(["Final: unsupported" + RS]), require_tool_call=True
    )
    with pytest.raises(AgentProtocolError, match="requires a structured tool call"):
        service.prepare_completion(
            {"messages": [{"role": "user", "content": "Do it."}]},
            session_id="tool-only",
        )
    assert len(service.sessions) == 0


def test_generated_error_trace_is_explicit_opt_in():
    request = {"messages": [{"role": "user", "content": "Do it."}]}
    ordinary = AgentCompletionService(FakeEngine(["secret malformed output" + RS]))
    with pytest.raises(AgentProtocolError) as ordinary_error:
        ordinary.prepare_completion(request, session_id=None)
    assert "secret" not in str(ordinary_error.value)

    traced = AgentCompletionService(
        FakeEngine(["secret malformed output" + RS]), trace_generated_errors=True
    )
    with pytest.raises(AgentProtocolError, match="generated_prefix=secret"):
        traced.prepare_completion(request, session_id=None)


def test_repeated_tool_call_cycle_fails_before_generation():
    repeated = {
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression":"5"}'},
        }],
    }
    service = AgentCompletionService(FakeEngine(["Final: unreachable" + RS]))
    with pytest.raises(AgentProtocolError, match="repeated tool call cycle"):
        service.prepare_completion(
            {
                "messages": [
                    {"role": "user", "content": "calculate"},
                    repeated,
                    {"role": "tool", "content": "error"},
                    repeated,
                    {"role": "tool", "content": "error"},
                ]
            },
            session_id="cycle",
        )
    assert len(service.sessions) == 0
    assert service.engine.outputs == ["Final: unreachable" + RS]


def test_external_controller_mode_delegates_repeat_policy_without_request_switch():
    repeated = {
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "read", "arguments": '{"path":"x","offset":1,"limit":1}'},
        }],
    }
    service = AgentCompletionService(
        FakeEngine(["Final: recovery accepted.\n"]), external_controller=True,
        **external_attestation_kwargs(),
    )
    prepared = service.prepare_completion(
        {
            "tools": READ_OBSERVE_TOOLS,
            "messages": [
                {"role": "user", "content": "read"}, repeated,
                {"role": "tool", "content": "same"}, repeated,
                {"role": "tool", "content": "no_progress"},
            ]
        },
        session_id="trusted-controller",
    )
    assert prepared.response["choices"][0]["message"]["content"] == "Final: recovery accepted.\n"
    attestation = prepared.response["emender_service_attestation"]
    assert attestation["schema"] == "emender-e97-agent-service-attestation-v1"
    assert attestation["checkpoint_path"] == str(Path(__file__))
    assert attestation["checkpoint_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert attestation["model_id"] == "e97-dense-agent"
    assert attestation["tool_schema_sha256"] == external_attestation_kwargs()["tool_schema_sha256"]
    assert attestation["system_prompt_override_sha256"] == sha256_text("")


def test_external_private_analysis_attestation_uses_versioned_protocol_identity():
    service = AgentCompletionService(
        FakeEngine(['Analysis: "reason"\nFinal: done\n']),
        external_controller=True, private_analysis=True,
        **external_attestation_kwargs(),
    )
    attestation = json.loads(service._service_attestation_json)
    assert attestation["schema"] == "emender-e97-agent-service-attestation-v2"
    assert attestation["runtime_identity_schema"] == "emender-e97-runtime-identity-v2"
    assert attestation["agent_protocol"] == "e97-pi-agent-analysis-v1"
    assert attestation["private_analysis"] is True
    prepared = service.prepare_completion(
        {"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "finish"}],
         "tools": READ_OBSERVE_TOOLS},
        session_id=None,
    )
    message = prepared.response["choices"][0]["message"]
    assert prepared.response["emender_assistant_message_sha256"] == sha256_json(message)


def test_external_controller_does_not_reopen_checkpoint_identity_path():
    engine = FakeEngine([])
    engine.checkpoint = "/missing/checkpoint.pt"
    identity = external_attestation_kwargs()
    identity["checkpoint_path"] = engine.checkpoint
    service = AgentCompletionService(engine, external_controller=True, **identity)
    assert json.loads(service._service_attestation_json)["checkpoint_path"] == engine.checkpoint


def test_external_controller_rejects_checkpoint_identity_that_disagrees_with_engine_bytes(tmp_path):
    checkpoint = tmp_path / "engine.pt"
    checkpoint.write_bytes(b"actual-checkpoint")
    engine = FakeEngine([])
    engine.checkpoint = str(checkpoint)
    identity = external_attestation_kwargs()
    identity["checkpoint_path"] = str(checkpoint)
    identity["checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        AgentCompletionService(engine, external_controller=True, **identity)


def test_external_service_consumes_bound_checkpoint_identity_without_reopening_swapped_path(tmp_path):
    checkpoint = tmp_path / "tiny-checkpoint.pt"
    checkpoint.write_bytes(b"loaded-model-bytes")
    digest_value = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    engine = FakeEngine([])
    engine.checkpoint = str(checkpoint)
    engine.checkpoint_sha256 = digest_value
    identity = external_attestation_kwargs()
    identity.update({"checkpoint_path": str(checkpoint), "checkpoint_sha256": digest_value})
    # Simulate a post-load pathname replacement.  A service that hashes or
    # opens it again would reject or attest the substituted bytes.
    checkpoint.write_bytes(b"substituted-path-bytes")
    service = AgentCompletionService(engine, external_controller=True, **identity)
    assert json.loads(service._service_attestation_json)["checkpoint_sha256"] == digest_value


def test_external_request_identity_uses_canonical_unicode_json():
    service = AgentCompletionService(FakeEngine(["Final: done\n"]), external_controller=True, **external_attestation_kwargs())
    messages = [
        {"role": "system", "content": "系统 α"}, {"role": "user", "content": "café"},
    ]
    prepared = service.prepare_completion({"messages": messages, "tools": READ_OBSERVE_TOOLS}, session_id=None)
    identity = prepared.response["emender_request_identity"]
    assert identity["messages_sha256"] == sha256_json(messages)
    assert identity["request_sha256"] == sha256_json({"messages": messages, "tools": READ_OBSERVE_TOOLS})


def test_external_controller_requires_bound_identity_receipt_fields():
    missing = external_attestation_kwargs()
    del missing["args_json_path"]
    with pytest.raises(ValueError, match="args_json_path"):
        AgentCompletionService(FakeEngine([]), external_controller=True, **missing)
    bad_args = external_attestation_kwargs()
    bad_args["args_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="args_json_sha256"):
        AgentCompletionService(FakeEngine([]), external_controller=True, **bad_args)
    bad_controller = external_attestation_kwargs()
    bad_controller["controller_build_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="controller_build_sha256"):
        AgentCompletionService(FakeEngine([]), external_controller=True, **bad_controller)


def test_external_controller_requires_startup_attestation_and_no_system_override():
    missing_tokenizer = external_attestation_kwargs()
    del missing_tokenizer["tokenizer"]
    with pytest.raises(ValueError, match="requires tokenizer"):
        AgentCompletionService(FakeEngine([]), external_controller=True, **missing_tokenizer)
    with pytest.raises(ValueError, match="forbids"):
        AgentCompletionService(
            FakeEngine([]), external_controller=True, system_prompt_override="override",
            **external_attestation_kwargs(),
        )


def test_failed_check_can_be_rerun_after_intervening_repair_action():
    check = {
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
        }],
    }
    repair = {
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "edit",
                "arguments": '{"path":"x.py","oldText":"0","newText":"1"}',
            },
        }],
    }
    service = AgentCompletionService(FakeEngine(["Final: repaired and verified.\n"]))
    prepared = service.prepare_completion(
        {
            "messages": [
                {"role": "user", "content": "repair and verify"},
                check,
                {"role": "tool", "content": "failed"},
                repair,
                {"role": "tool", "content": "Successfully replaced 1 block."},
                check,
                {"role": "tool", "content": "(no tool output)"},
            ]
        },
        session_id="repair-check",
    )
    assert prepared.response["choices"][0]["message"]["content"] == (
        "Final: repaired and verified.\n")


def test_unknown_tool_and_malformed_generation_do_not_commit():
    for output, match in (
        ('Action: shell\nArguments: {"command":"id"}' + RS, "unknown tool"),
        ("I refuse." + RS, "must begin"),
    ):
        service = AgentCompletionService(FakeEngine([output]))
        with pytest.raises(AgentProtocolError, match=match):
            service.prepare_completion(
                {"messages": [{"role": "user", "content": "Do it."}], "tools": [tool("read")]},
                session_id="unsafe",
            )
        assert len(service.sessions) == 0


def test_sse_has_role_payload_finish_and_done_events():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS]))
    completion = service.prepare_completion(
        {"messages": [{"role": "user", "content": "Do it."}]},
        session_id=None,
    )
    events = chat_completion_sse(completion.response)
    assert b'"role":"assistant"' in events[0]
    assert b'"content":"Final: done"' in events[1]
    assert b'"finish_reason":"stop"' in events[2]
    assert events[-1] == b"data: [DONE]\n\n"


def run_test_server(service, *, api_key=None):
    server = HTTPServer(("127.0.0.1", 0), make_openai_handler(service, api_key=api_key))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_models_nonstream_stream_auth_and_transaction_commit():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS, "Final: again" + RS]))
    server, thread = run_test_server(service, api_key="secret")
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/v1/models")
        assert connection.getresponse().status == 401

        headers = {"Authorization": "Bearer secret"}
        connection.request("GET", "/v1/models", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        assert b"e97-dense-agent" in response.read()

        body = '{"model":"e97-dense-agent","messages":[{"role":"user","content":"Do it."}]}'
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={**headers, "Content-Type": "application/json", "x-session-id": "http-test"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("x-emender-cache") == "miss"
        assert b'"content":"Final: done"' in response.read()
        assert len(service.sessions) == 1

        stream_body = '{"model":"e97-dense-agent","stream":true,"messages":[{"role":"user","content":"Again."}]}'
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=stream_body,
            headers={**headers, "Content-Type": "application/json", "x-session-id": "http-test"},
        )
        response = connection.getresponse()
        payload = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert b'"content":"Final: again"' in payload
        assert payload.endswith(b"data: [DONE]\n\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_request_limits_fail_closed():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS]), max_output_tokens=16)
    base = {"messages": [{"role": "user", "content": "Do it."}]}
    for update in (
        {"max_tokens": 0},
        {"max_tokens": "10"},
        {"temperature": -1},
        {"top_p": 0},
        {"stream": "yes"},
        {"model": "other"},
    ):
        with pytest.raises(AgentProtocolError):
            service.prepare_completion({**base, **update}, session_id=None)
