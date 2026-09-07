import copy
import json
from pathlib import Path

import pytest

from ndm.e97_onpolicy_records import (
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    canonical_json,
    sha256_text,
    task_identity,
)
from ndm.e97_task_lake import SOURCE_REGISTRY_SCHEMA, TASK_BUNDLE_SCHEMA, canonical_intent_digest
from ndm.e97_phase_c_collector import (
    PiEventError,
    apply_no_progress,
    parse_pi_events,
)


# All values below are synthetic and unrelated to any evaluation task.
def d(label):
    return sha256_text("phase-c-fresh:" + label)


def registry():
    return {
        "schema": SOURCE_REGISTRY_SCHEMA,
        "created_at": "2026-09-06T00:00:00Z",
        "policy_sha256": d("policy"),
        "protected_evaluation": [
            {"name": "v3", "manifest_sha256": CONSUMED_V3_MANIFEST_SHA256,
             "repositories": [], "family_ids": [], "task_identities": [],
             "fixture_tree_digests": [], "intent_digests": [], "validator_spec_digests": []},
            {"name": "v4", "manifest_sha256": CONSUMED_V4_MANIFEST_SHA256,
             "repositories": [], "family_ids": [], "task_identities": [],
             "fixture_tree_digests": [], "intent_digests": [], "validator_spec_digests": []},
            {"name": "real-repo-holdout-v1", "manifest_sha256": "939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60",
             "repositories": [], "family_ids": [], "task_identities": [],
             "fixture_tree_digests": [], "intent_digests": [], "validator_spec_digests": []},
        ],
        "sources": [{
            "id": "fresh-synthetic", "kind": "first-party", "status": "admitted",
            "url": "https://example.invalid/fresh-synthetic", "revision": "a" * 40,
            "framework_license": "MIT", "underlying_repository_policy": "per-repository-audit-required",
            "task_count_claim": 1,
            "receipts": {"source_archive_sha256": d("archive"), "license_sha256": d("license"),
                         "environment_sha256": d("environment"), "overlap_sha256": d("overlap")},
            "notes": "fresh fixture",
        }],
    }


def bundle():
    prompt = "Read the fresh synthetic value."
    intent = canonical_intent_digest(prompt)
    tree = d("tree")
    generator = d("archive")
    namespace = "e97-train-fresh-synthetic"
    family = "fresh-synthetic-read-v1"
    identity = task_identity(namespace=namespace, family_id=family,
                             generator_source_digest=generator,
                             fixture_tree_digest=tree, intent_digest=intent)
    return {
        "schema": TASK_BUNDLE_SCHEMA, "split": "train",
        "task": {"namespace": namespace, "family_id": family, "identity": identity,
                 "generator_source_digest": generator, "fixture_tree_digest": tree,
                 "intent_digest": intent, "prompt": prompt, "difficulty": 1},
        "source": {"registry_id": "fresh-synthetic", "kind": "first-party",
                    "repository": "fresh/synthetic", "revision": "a" * 40,
                    "source_record_digest": d("record"), "license_receipt_digest": d("license")},
        "fixture": {"artifact_path": "fixtures/fresh.tar.zst", "artifact_bytes": 7,
                    "artifact_sha256": d("artifact"), "tree_digest": tree},
        "runtime": {"schema_digest": d("schema"), "tool_schema_digest": d("tools"),
                    "controller_digest": d("controller"), "sandbox_image_digest": d("image"),
                    "system_prompt_sha256": sha256_text("Fresh system\n")},
        "limits": {"turns": 4, "seconds": 10, "completion_tokens": 64, "output_bytes": 1024,
                    "disk_bytes": 1024, "processes": 2},
        "validator": {"spec_digest": d("validator"), "focused_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "focused"],
                       "regression_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "regression"], "milestone_digest": d("milestone"),
                       "minefield_digest": d("minefield")},
    }


def events(result="opal-731"):
    return [
        {"type": "session", "id": "fresh-session"},
        {"type": "agent_start"}, {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": "fresh"}},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "id": "call-read", "name": "read",
             "arguments": {"path": "fresh.txt", "offset": 1, "limit": 20}},
        ]}},
        {"type": "tool_execution_start", "toolCallId": "call-read", "toolName": "read",
         "args": {"path": "fresh.txt", "offset": 1, "limit": 20}},
        {"type": "tool_execution_end", "toolCallId": "call-read", "result": result, "isError": False},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "call-read",
                                                "content": [{"type": "text", "text": result}], "isError": False}},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Final: The fresh value is opal-731."},
        ]}},
        {"type": "turn_end"}, {"type": "agent_end"}, {"type": "agent_settled"},
    ]


def state(sequence=0, tree="same", observation="opal-731"):
    return {"sequence": sequence, "tool_call_id": "call-read", "tool_name": "read",
            "arguments": {"path": "fresh.txt", "offset": 1, "limit": 20},
            "observation_sha256": sha256_text(observation if observation else "(no tool output)"),
            "workspace_state": {"tree": d(tree)}, "source_ledger": {},
            "acquired_observations": [{"value": "opal-731"}]}


def test_authentic_events_produce_exact_messages_and_linked_observation():
    transcript = parse_pi_events(events(), system_prompt="Fresh system\n", user_prompt="fresh")
    assert [message["role"] for message in transcript.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert transcript.messages[2]["tool_call_id"] == transcript.messages[3]["tool_call_id"] == "call-read"
    assert transcript.messages[2]["text"] == 'Action: read\nArguments: {"limit":20,"offset":1,"path":"fresh.txt"}'
    assert transcript.actions[0].observation_text == "opal-731"
    assert transcript.terminal_status == "success"


def test_pi_tool_execution_result_retains_raw_details_but_canonicalizes_text():
    rows = events()
    rows[6] = {"type": "tool_execution_end", "toolCallId": "call-read",
               "result": {"content": [{"type": "text", "text": "opal-731"}],
                          "details": {"exitCode": 0}}, "isError": False}
    transcript = parse_pi_events(rows, system_prompt="Fresh system\n", user_prompt="fresh")
    assert transcript.actions[0].observation_raw["details"] == {"exitCode": 0}
    assert transcript.messages[3]["text"] == "opal-731"


def test_real_shaped_empty_tool_output_uses_canonical_context_placeholder():
    transcript = parse_pi_events(events(result=""), system_prompt="Fresh system\n", user_prompt="fresh")
    assert transcript.actions[0].observation_text == "(no tool output)"
    assert transcript.actions[0].observation_raw == ""
    assert transcript.messages[3]["text"] == "(no tool output)"


def error_events():
    return [
        {"type": "session", "id": "fresh-error-session"},
        {"type": "agent_start"}, {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": "fresh"}},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "id": "call-error", "name": "read", "arguments": {"path": "bad"}},
        ]}},
        {"type": "tool_execution_start", "toolCallId": "call-error", "toolName": "read", "args": {"path": "bad"}},
        {"type": "tool_execution_end", "toolCallId": "call-error", "toolName": "read", "result": "missing", "isError": True},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "call-error",
                                                "content": [{"type": "text", "text": "missing"}], "isError": True}},
        {"type": "message_end", "message": {"role": "assistant", "content": [],
                                                "stopReason": "error", "errorMessage": "protocol error"}},
        {"type": "agent_end"}, {"type": "agent_settled"},
    ]


def test_terminal_protocol_error_is_collectable_without_invented_assistant_message():
    transcript = parse_pi_events(error_events(), system_prompt="Fresh system\n", user_prompt="fresh")
    assert transcript.terminal_status == "error"
    assert transcript.terminal_error == "protocol error"
    assert [message["role"] for message in transcript.messages] == ["system", "user", "assistant", "tool"]


def test_missing_tool_result_and_observation_receipt_mismatch_fail_closed():
    missing = events()[:7] + events()[8:]
    with pytest.raises(PiEventError, match="every execution"):
        parse_pi_events(missing, system_prompt="Fresh system\n", user_prompt="fresh")
    transcript = parse_pi_events(events(), system_prompt="Fresh system\n", user_prompt="fresh")
    bad_receipt = state()
    bad_receipt["observation_sha256"] = d("wrong-observation")
    with pytest.raises(PiEventError, match="observation digest mismatch"):
        apply_no_progress(transcript, [bad_receipt])


def test_changed_state_repeated_action_is_allowed_and_stale_interleave_terminates():
    transcript = parse_pi_events(events(), system_prompt="Fresh system\n", user_prompt="fresh")
    assert apply_no_progress(transcript, [state(tree="changed")])[0]["kind"] == "continue"

    # The same action is valid again after a changed post-action state.
    from ndm.e97_onpolicy_records import ActionProgressReceipt, NoProgressDetector
    detector = NoProgressDetector()
    first = ActionProgressReceipt("read", {"path": "x"}, {"tree": "before"}, {}, ["old"])
    changed = ActionProgressReceipt("read", {"path": "x"}, {"tree": "after"}, {}, ["old", "new"])
    assert detector.observe(first).kind == "continue"
    assert detector.observe(changed).kind == "continue"

    # Exercise the detector through repeated receipts with interleaving action.
    detector = NoProgressDetector()
    stale = ActionProgressReceipt("read", {"path": "x"}, {"tree": "same"}, {}, ["old"])
    other = ActionProgressReceipt("read", {"path": "y"}, {"tree": "same"}, {}, ["old"])
    assert detector.observe(stale).kind == "continue"
    assert detector.observe(stale).kind == "inject_no_progress"
    assert detector.observe(other).kind == "continue"
    assert detector.observe(stale).kind == "terminate"


def test_malformed_tool_linkage_and_incomplete_trace_fail_closed():
    bad = events()
    bad[5] = dict(bad[5], toolCallId="different")
    with pytest.raises(PiEventError, match="linkage|match"):
        parse_pi_events(bad, system_prompt="Fresh system\n", user_prompt="fresh")
    incomplete = events()[:-1]
    with pytest.raises(PiEventError, match="agent_settled"):
        parse_pi_events(incomplete, system_prompt="Fresh system\n", user_prompt="fresh")
    without_start = [events()[0], *events()[2:]]
    with pytest.raises(PiEventError, match="before agent_start"):
        parse_pi_events(without_start, system_prompt="Fresh system\n", user_prompt="fresh")
    without_result = events()
    without_result[6] = {key: value for key, value in without_result[6].items()
                         if key != "result"}
    with pytest.raises(PiEventError, match="requires result"):
        parse_pi_events(without_result, system_prompt="Fresh system\n", user_prompt="fresh")
    without_tool_content = events()
    without_tool_content[7] = copy.deepcopy(without_tool_content[7])
    del without_tool_content[7]["message"]["content"]
    with pytest.raises(PiEventError, match="requires content"):
        parse_pi_events(without_tool_content, system_prompt="Fresh system\n", user_prompt="fresh")


def test_current_schema_collection_inputs_validate_successfully():
    from ndm.e97_phase_c_collector import validate_collection_inputs
    task = bundle(); reg = registry()
    validated_task, validated_registry = validate_collection_inputs(
        task_bundle=task,
        source_registry=reg,
        task_bundle_sha256=sha256_text(canonical_json(task)),
        source_registry_sha256=sha256_text(canonical_json(reg)),
        actual_task_bundle_sha256=sha256_text(canonical_json(task)),
        actual_source_registry_sha256=sha256_text(canonical_json(reg)),
    )
    assert validated_task == task
    assert len(validated_registry["protected_evaluation"]) == 3


def test_task_and_registry_input_sha_mismatch_is_checked_before_collection(tmp_path):
    from ndm.e97_phase_c_collector import validate_collection_inputs
    task = bundle(); reg = registry()
    with pytest.raises(PiEventError, match="task bundle SHA-256 mismatch"):
        validate_collection_inputs(task_bundle=task, source_registry=reg, task_bundle_sha256=d("wrong"),
                                   source_registry_sha256=d("registry"), actual_task_bundle_sha256=d("task"),
                                   actual_source_registry_sha256=d("registry"))


def test_atomic_interruption_leaves_no_partial_output(tmp_path, monkeypatch):
    from scripts.collect_e97_onpolicy_rollout import atomic_json_write
    output = tmp_path / "receipt.json"
    output.write_text("old\n")
    def interrupted(*args, **kwargs):
        raise OSError("simulated interruption")
    monkeypatch.setattr("scripts.collect_e97_onpolicy_rollout.os.link", interrupted)
    with pytest.raises(OSError, match="interruption"):
        atomic_json_write(output, {"status": "complete"})
    assert output.read_text() == "old\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_output_refuses_existing_receipt(tmp_path):
    from scripts.collect_e97_onpolicy_rollout import atomic_json_write
    output = tmp_path / "receipt.json"
    output.write_text("sealed\n")
    with pytest.raises(FileExistsError):
        atomic_json_write(output, {"status": "new"})
    assert output.read_text() == "sealed\n"


def test_raw_and_state_sha_pins_are_required_and_verified(tmp_path):
    from scripts.collect_e97_onpolicy_rollout import file_sha256, verify_file_pin
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"type":"session"}\n')
    pin = file_sha256(raw)
    assert verify_file_pin(raw, pin, "raw events") == pin
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        verify_file_pin(raw, d("wrong-pin"), "state receipts")
