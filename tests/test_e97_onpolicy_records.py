import copy

import pytest

from ndm.e97_onpolicy_records import (
    CANONICAL_NO_PROGRESS_OBSERVATION,
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    RECOVERY_RECORD_SCHEMA,
    ActionProgressReceipt,
    NoProgressDetector,
    action_fingerprint,
    canonical_action,
    canonical_json,
    completion_marker_relative_path,
    progress_fingerprint,
    recovery_record_fingerprint,
    sha256_json,
    sha256_text,
    task_identity,
    validate_recovery_record,
)


def digest(label: str) -> str:
    return sha256_text(label)


def action(sequence: int, call_id: str, *, decision: str, effective: str, raw: str) -> dict:
    arguments = {"path": "missing.txt", "offset": 1, "limit": 40}
    workspace = {"tree": digest("unchanged")}
    observations = [{"tool": "read", "raw_observation_sha256": sha256_json(raw)}]
    return {
        "sequence": sequence,
        "tool_call_id": call_id,
        "tool_name": "read",
        "arguments": arguments,
        "arguments_json": '{"limit":40,"offset":1,"path":"missing.txt"}',
        "raw_observation": raw,
        "raw_observation_sha256": sha256_json(raw),
        "effective_observation": effective,
        "observation_sha256": sha256_text(effective),
        "is_error": True,
        "workspace_state": workspace,
        "source_ledger": {},
        "acquired_observations": observations,
        "action_fingerprint": action_fingerprint("read", arguments),
        "progress_fingerprint": progress_fingerprint(
            workspace_state=workspace,
            source_ledger={},
            acquired_observations=observations,
        ),
        "completion_tokens": 1,
        "decision": decision,
    }


def valid_record(*, split: str = "train") -> dict:
    """A real continue/inject/terminate failed history plus a corrective action."""

    system = "Use bounded tools and ground the final in observed evidence."
    failure = "FileNotFoundError: missing.txt\nCommand exited with code 1"
    failed_actions = [
        action(0, "bad-0", decision="continue", effective=failure, raw=failure),
        action(1, "bad-1", decision="inject_no_progress", effective=CANONICAL_NO_PROGRESS_OBSERVATION, raw=failure),
        action(2, "bad-2", decision="terminate", effective=CANONICAL_NO_PROGRESS_OBSERVATION, raw=failure),
    ]
    task = {
        "namespace": f"e97-{'train' if split == 'train' else 'dev'}-opaque-read-v1",
        "family_id": "opaque-read-recovery-v1",
        "generator_source_digest": digest("generator-v1"),
        "fixture_tree_digest": digest(f"fixture-{split}"),
        "intent_digest": digest("read-the-declared-value"),
    }
    task["identity"] = task_identity(**task)
    runtime = {
        "schema_digest": digest("runtime-schema-v1"),
        "controller_digest": digest("controller-v1"),
        "system_prompt_sha256": sha256_text(system),
        "tool_schema_digest": sha256_json([]),
        "sandbox_image_digest": digest("sandbox"),
    }
    limits = {
        "turns": 12,
        "seconds": 60,
        "completion_tokens": 512,
        "output_bytes": 16384,
        "disk_bytes": 1 << 20,
        "processes": 1,
    }
    metadata = {
        "controller_build_sha256": runtime["controller_digest"],
        "tool_schema_sha256": runtime["tool_schema_digest"],
        "system_prompt_sha256": runtime["system_prompt_sha256"],
        "configured_limits": limits,
        "turn_count": 3,
        "action_count": 3,
        "elapsed_seconds": 1.0,
        "completion_usage": [
            {"sequence": 0, "completion_tokens": 1},
            {"sequence": 1, "completion_tokens": 1},
            {"sequence": 2, "completion_tokens": 1},
        ],
        "completion_tokens": 3,
    }

    def assistant(item: dict) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": item["tool_call_id"],
                "type": "function",
                "function": {"name": item["tool_name"], "arguments": item["arguments_json"]},
            }],
        }

    failed_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Read the declared value from the workspace."},
        assistant(failed_actions[0]),
        {"role": "tool", "tool_call_id": "bad-0", "content": failure},
        assistant(failed_actions[1]),
        {"role": "tool", "tool_call_id": "bad-1", "content": CANONICAL_NO_PROGRESS_OBSERVATION},
        assistant(failed_actions[2]),
    ]
    failed = {
        "status": "no_progress_terminated",
        "turns": 3,
        "elapsed_seconds": 1.0,
        "messages": failed_messages,
        "actions": failed_actions,
        "metadata": metadata,
        "error": None,
    }
    good_arguments = {"path": "facts/value.txt", "offset": 1, "limit": 40}
    good_raw = '{"limit":40,"offset":1,"path":"facts/value.txt"}'
    good_workspace = {"tree": digest("changed")}
    good = {
        "sequence": 3,
        "tool_call_id": "good",
        "tool_name": "read",
        "arguments": good_arguments,
        "arguments_json": good_raw,
        "raw_observation": "opal-731",
        "raw_observation_sha256": sha256_json("opal-731"),
        "effective_observation": "opal-731",
        "observation_sha256": sha256_text("opal-731"),
        "is_error": False,
        "workspace_state": good_workspace,
        "source_ledger": {},
        "acquired_observations": [{"tool": "read", "raw_observation_sha256": sha256_json("opal-731")}],
        "action_fingerprint": action_fingerprint("read", good_arguments),
        "progress_fingerprint": progress_fingerprint(
            workspace_state=good_workspace,
            source_ledger={},
            acquired_observations=[{"tool": "read", "raw_observation_sha256": sha256_json("opal-731")}],
        ),
        "completion_tokens": 1,
        "decision": "continue",
    }
    intervention = {
        "action_index": 2,
        "effective_observation": CANONICAL_NO_PROGRESS_OBSERVATION,
        "sent_to_model": False,
    }
    corrective_messages = failed_messages + [
        {"role": "tool", "tool_call_id": "bad-2", "content": CANONICAL_NO_PROGRESS_OBSERVATION, "controller_intervention": True},
        assistant(good),
        {"role": "tool", "tool_call_id": "good", "content": "opal-731"},
        {"role": "assistant", "content": "Final: facts/value.txt contains opal-731."},
    ]
    def attestation():
        return {
            "schema": "emender-e97-agent-service-attestation-v1", "checkpoint_path": "unit://checkpoint",
            "checkpoint_sha256": digest("checkpoint"), "args_json_sha256": digest("args"),
            "config_sha256": digest("config"), "weight_mode": "saved", "tokenizer": "unit",
            "model_id": "unit-model", "server_build_sha256": digest("server"),
            "controller_build_sha256": runtime["controller_digest"], "device": "cpu", "dtype": "unit",
            "use_triton": False, "ingest_mode": "tokenwise", "runtime_image_path": "unit://runtime",
            "runtime_image_sha256": runtime["sandbox_image_digest"], "tool_schema_sha256": runtime["tool_schema_digest"],
            "system_prompt_override_sha256": digest(""), "runtime_identity_schema": "emender-e97-runtime-identity-v1",
            "max_output_tokens": 512, "max_sessions": 1, "python_implementation": "CPython",
            "python_version": "unit", "torch_version": "unit", "cuda_runtime": "", "cuda_available": False,
            "platform": "unit", "machine": "unit",
        }

    def completion_receipts(messages):
        receipts = []
        for position, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            sequence = len(receipts)
            request = {
                "model": "unit-model", "messages": [dict(item) for item in messages[:position]],
                "tools": [], "temperature": 0, "max_completion_tokens": limits["completion_tokens"],
            }
            identity = {
                "messages_sha256": sha256_text(canonical_json(request["messages"])),
                "system_prompt_sha256": runtime["system_prompt_sha256"],
                "tool_schema_sha256": runtime["tool_schema_digest"],
                "request_sha256": sha256_json(request),
            }
            service = attestation()
            response = {
                "model": "unit-model", "choices": [{"message": dict(message)}],
                "usage": {"completion_tokens": 1}, "emender_request_identity": identity,
                "emender_service_attestation": service,
            }
            receipts.append({"sequence": sequence, "request": request, "request_identity": identity,
                             "request_sha256": identity["request_sha256"], "response": response,
                             "response_sha256": sha256_json(response), "service_attestation": service,
                             "service_attestation_sha256": sha256_json(service), "model_id": "unit-model",
                             "completion_tokens": 1, "assistant_message_sha256": sha256_json(message)})
        return receipts

    metadata.update({
        "model_id": "unit-model", "checkpoint_sha256": digest("checkpoint"),
        "service_attestation": attestation(),
        "completion_receipts": completion_receipts(failed_messages),
    })
    corrective_metadata = {
        **metadata,
        "turn_count": 5,
        "action_count": 4,
        "elapsed_seconds": 2.0,
        "completion_usage": [
            {"sequence": 0, "completion_tokens": 1},
            {"sequence": 1, "completion_tokens": 1},
            {"sequence": 2, "completion_tokens": 1},
            {"sequence": 3, "completion_tokens": 1},
            {"sequence": 4, "completion_tokens": 1},
        ],
        "completion_tokens": 5,
        "completion_receipts": completion_receipts(corrective_messages),
    }
    corrective = {
        "schema": "emender-e97-corrective-terminal-v2",
        "status": "success",
        "failed_terminal_sha256": sha256_json(failed),
        "correction_start_message_index": 8,
        "prefix_sha256": sha256_json(corrective_messages[:8]),
        "turns": 5,
        "elapsed_seconds": 2.0,
        "messages": corrective_messages,
        "actions": failed_actions + [good],
        "metadata": corrective_metadata,
        "error": None,
    }
    bundle = {
        "schema": "emender-e97-onpolicy-task-v1",
        "split": split,
        "task": {**task, "prompt": "Read the declared value.", "difficulty": 2},
        "source": {
            "registry_id": "test-source",
            "kind": "first-party",
            "repository": "example/test-source",
            "revision": "a" * 40,
            "source_record_digest": digest("source-record"),
            "license_receipt_digest": digest("license"),
        },
        "fixture": {
            "artifact_path": "archives/example.tar",
            "artifact_bytes": 10,
            "artifact_sha256": digest("archive"),
            "tree_digest": task["fixture_tree_digest"],
        },
        "runtime": runtime,
        "limits": limits,
        "validator": {
            "spec_digest": digest("validator-v1"),
            "focused_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "focused"],
            "regression_argv": ["@runtime-python", "@generator-source/scripts/e97_first_party_validator.py", "--mode", "regression"],
            "milestone_digest": digest("milestone"),
            "minefield_digest": digest("minefield"),
        },
    }
    validator_spec_payload_sha256 = sha256_text("validator-spec-payload")
    validator_terminal_payload_sha256 = sha256_text("validator-terminal-payload")
    validators = {}
    for mode in ("focused", "regression"):
        logical = bundle["validator"][f"{mode}_argv"] + [
            "--spec-fd", "<inherited-spec-fd>",
            "--terminal-fd", "<inherited-terminal-fd>",
        ]
        output = {"mode": mode, "status": "pass", "action_count": 4}
        bound = ["@verified-interpreter-fd", "@private-verified-validator", "--mode", mode,
                 "--spec-fd", "<inherited-spec-fd>",
                 "--terminal-fd", "<inherited-terminal-fd>"]
        validators[mode] = {
            "logical_argv": logical,
            "logical_argv_sha256": sha256_json(logical),
            "bound_argv": bound,
            "bound_argv_sha256": sha256_json(bound),
            "stdout_sha256": sha256_text(canonical_json(output) + "\n"),
            "stderr_sha256": sha256_text(""),
            "spec_payload_sha256": validator_spec_payload_sha256,
            "terminal_payload_sha256": validator_terminal_payload_sha256,
            "output": output,
        }
    execution = {
        "schema": "emender-e97-first-party-validator-receipt-v4",
        "status": "pass",
        "task_identity": task["identity"],
        "bundle_sha256": sha256_json(bundle),
        "fixture_tree_digest": task["fixture_tree_digest"],
        "archive_expanded_bytes": 10,
        "configured_limits": limits,
        "terminal_sha256": sha256_json(corrective),
        "validator_spec_digest": bundle["validator"]["spec_digest"],
        "validator_spec_payload_sha256": validator_spec_payload_sha256,
        "validator_terminal_payload_sha256": validator_terminal_payload_sha256,
        "runtime_schema_digest": runtime["schema_digest"],
        "validators": validators,
    }
    lease_base = {"task_id": task["identity"], "owner": "unit", "attempt": 1, "deadline_ns": 42}
    lease = {**lease_base, "identity": sha256_json(lease_base)}
    completion = {
        "schema": "emender-e97-task-completion-receipt-v3",
        "task_id": task["identity"],
        "lease": lease,
        "receipt": {
            "task_id": task["identity"],
            "bundle_sha256": sha256_json(bundle),
            "archive_sha256": digest("archive"),
            "failed_terminal_sha256": sha256_json(failed),
            "accepted_terminal_sha256": sha256_json(corrective),
            "validator_receipt_sha256": sha256_json(execution),
            "validator_status": "pass",
        },
    }
    direct = {
        "bundle": sha256_json(bundle),
        "archive": digest("archive"),
        "failed_terminal": sha256_json(failed),
        "corrective_terminal": sha256_json(corrective),
        "validator_execution": sha256_json(execution),
        "completion_receipt": sha256_json(completion),
    }
    artifacts = {
        "registry": {"sha256": digest("registry"), "path": "artifacts/aa/registry.json"},
        "generation_receipt": {"sha256": digest("generation"), "path": "artifacts/bb/generation.json"},
        "generator_manifest": {"sha256": digest("generator-manifest"), "path": "artifacts/bc/generator-manifest.json"},
        "source_archive": {"sha256": digest("source-archive"), "path": "artifacts/bd/source-archive.tar"},
        "environment_descriptor": {"sha256": digest("environment"), "path": "artifacts/be/environment.json"},
        "overlap_firewall_audit": {"sha256": digest("overlap-audit"), "path": "artifacts/bf/overlap-audit.json"},
        "authorization_license": {"sha256": digest("license"), "path": "artifacts/c0/license.json"},
        "overlap_receipt": {"sha256": digest("overlap"), "path": "artifacts/cc/overlap.json"},
        "admission_receipt": {"sha256": digest("admission"), "path": "artifacts/cd/admission.json"},
        "authority_state": {"sha256": digest("authority-state"), "path": "artifacts/ce/authority-state.json"},
        "collection_authorization_allowlist": {"sha256": digest("allowlist"), "path": "artifacts/cf/allowlist.json"},
        "tasks_collection": {"sha256": digest("tasks"), "path": "artifacts/dd/tasks.jsonl"},
        "private_spec": {"sha256": digest("spec"), "path": "artifacts/ee/spec.json"},
    }
    artifacts.update({
        name: {"sha256": value, "path": f"artifacts/{value[:2]}/{value}.json" if name != "archive" else f"artifacts/{value[:2]}/{value}.tar"}
        for name, value in direct.items()
    })
    artifacts["completion_receipt"]["path"] = completion_marker_relative_path(
        task["identity"], direct["completion_receipt"],
    )
    prefix = [
        {"role": "system", "text": system, "loss": 0},
        {"role": "user", "text": failed_messages[1]["content"], "loss": 0},
        {"role": "assistant", "text": "Action: read\nArguments: " + failed_actions[0]["arguments_json"], "loss": 0},
        {"role": "tool", "text": failure, "loss": 0},
        {"role": "assistant", "text": "Action: read\nArguments: " + failed_actions[1]["arguments_json"], "loss": 0},
        {"role": "tool", "text": CANONICAL_NO_PROGRESS_OBSERVATION, "loss": 0},
        {"role": "assistant", "text": "Action: read\nArguments: " + failed_actions[2]["arguments_json"], "loss": 0},
        {"role": "tool", "text": CANONICAL_NO_PROGRESS_OBSERVATION, "loss": 0},
    ]
    return {
        "schema": RECOVERY_RECORD_SCHEMA,
        "split": split,
        "task": task,
        "student": {"rollout_identity": sha256_json(failed), "checkpoint_sha256": digest("checkpoint"), "decode": {"temperature": 0}},
        "teacher": {"tier": "luna", "model_revision": "synthetic-unit-test", "evidence": "closed-completion-receipts-v1"},
        "provenance": {"scope": "teacher-evidenced", "training_eligible": True},
        "runtime": runtime,
        "first_divergence": {
            "message_index": 4,
            "target_start_message_index": 8,
            "class": "stale-action",
            "student_action_canonical": canonical_action("read", failed_actions[1]["arguments"]),
            "pre_state_digest": failed_actions[1]["progress_fingerprint"],
            "observation_digest": sha256_text(CANONICAL_NO_PROGRESS_OBSERVATION),
        },
        "messages": prefix + [
            {"role": "assistant", "text": "Action: read\nArguments: " + good_raw, "loss": 1},
            {"role": "tool", "text": "opal-731", "loss": 0},
            {"role": "assistant", "text": "Final: facts/value.txt contains opal-731.", "loss": 1},
        ],
        "validator_receipt": {
            "validator_digest": bundle["validator"]["spec_digest"],
            "input_digest": task["fixture_tree_digest"],
            "postcondition_digest": sha256_json(execution),
            "action_graph_digest": sha256_json(corrective["actions"]),
            "passed": True,
            "cycle_free": True,
        },
        "source_provenance": {
            "source_digests": [task["generator_source_digest"]],
            "forbidden_panel_digests_checked": [CONSUMED_V3_MANIFEST_SHA256, CONSUMED_V4_MANIFEST_SHA256],
        },
        "terminal_binding": {
            "artifacts": artifacts,
            "bundle": bundle,
            "bundle_sha256": sha256_json(bundle),
            "archive_sha256": digest("archive"),
            "failed_terminal": failed,
            "failed_terminal_sha256": sha256_json(failed),
            "corrective_terminal": corrective,
            "corrective_terminal_sha256": sha256_json(corrective),
            "validator_execution": execution,
            "validator_execution_sha256": sha256_json(execution),
            "completion_receipt": completion,
            "completion_receipt_sha256": sha256_json(completion),
            "lease_identity": lease["identity"],
            "prefix_sha256": corrective["prefix_sha256"],
            "model_prefix_sha256": sha256_json(prefix),
            "terminating_intervention": intervention,
        },
    }


def _reseal_terminal_graph(record: dict) -> None:
    """Recompute every enclosing receipt relation after a deliberate mutation."""

    binding = record["terminal_binding"]
    bundle = binding["bundle"]
    failed = binding["failed_terminal"]
    corrective = binding["corrective_terminal"]
    execution = binding["validator_execution"]
    completion = binding["completion_receipt"]

    for terminal in (failed, corrective):
        for receipt in terminal["metadata"]["completion_receipts"]:
            attestation = receipt["service_attestation"]
            receipt["service_attestation_sha256"] = sha256_json(attestation)
            receipt["response"]["emender_service_attestation"] = attestation
            receipt["response_sha256"] = sha256_json(receipt["response"])

    bundle_digest = sha256_json(bundle)
    failed_digest = sha256_json(failed)
    corrective["failed_terminal_sha256"] = failed_digest
    corrective_digest = sha256_json(corrective)
    execution["bundle_sha256"] = bundle_digest
    execution["terminal_sha256"] = corrective_digest
    execution_digest = sha256_json(execution)
    completion["receipt"].update({
        "bundle_sha256": bundle_digest,
        "failed_terminal_sha256": failed_digest,
        "accepted_terminal_sha256": corrective_digest,
        "validator_receipt_sha256": execution_digest,
    })
    completion_digest = sha256_json(completion)

    binding.update({
        "bundle_sha256": bundle_digest,
        "failed_terminal_sha256": failed_digest,
        "corrective_terminal_sha256": corrective_digest,
        "validator_execution_sha256": execution_digest,
        "completion_receipt_sha256": completion_digest,
    })
    record["student"]["rollout_identity"] = failed_digest
    record["validator_receipt"].update({
        "postcondition_digest": execution_digest,
        "action_graph_digest": sha256_json(corrective["actions"]),
    })
    for name, digest_value in {
        "bundle": bundle_digest,
        "failed_terminal": failed_digest,
        "corrective_terminal": corrective_digest,
        "validator_execution": execution_digest,
        "completion_receipt": completion_digest,
    }.items():
        binding["artifacts"][name]["sha256"] = digest_value
    binding["artifacts"]["completion_receipt"]["path"] = completion_marker_relative_path(
        record["task"]["identity"], completion_digest)


def _set_all_persisted_attestations(record: dict, field: str, value: object) -> None:
    """Mutate live-equivalent attestation data everywhere its receipt closes it."""

    for terminal in (
            record["terminal_binding"]["failed_terminal"],
            record["terminal_binding"]["corrective_terminal"],
    ):
        terminal["metadata"]["service_attestation"][field] = value
        for receipt in terminal["metadata"]["completion_receipts"]:
            receipt["service_attestation"][field] = value
            receipt["response"]["emender_service_attestation"][field] = value


def test_action_and_progress_fingerprints_are_canonical():
    assert action_fingerprint("read", {"path": "a.txt", "limit": 40}) == action_fingerprint(
        "read", {"limit": 40, "path": "a.txt"},
    )
    assert canonical_action("read", {"path": "a.txt", "limit": 40}) == 'Action: read\nArguments: {"limit":40,"path":"a.txt"}'


def test_no_progress_detector_injects_once_then_terminates():
    detector = NoProgressDetector()
    receipt = ActionProgressReceipt("read", {"path": "missing.txt"}, {"tree": digest("same")}, {}, ["missing"])
    assert detector.observe(receipt).kind == "continue"
    assert detector.observe(receipt).kind == "inject_no_progress"
    assert detector.observe(receipt).kind == "terminate"


def test_three_action_history_is_structurally_replayed():
    record = valid_record()
    assert [action["decision"] for action in record["terminal_binding"]["failed_terminal"]["actions"]] == [
        "continue", "inject_no_progress", "terminate",
    ]
    assert validate_recovery_record(record) == record


@pytest.mark.parametrize(("field", "value"), [
    ("runtime_identity_schema", "forged-runtime-schema"),
    ("use_triton", "false"),
    ("cuda_available", 1),
    ("weight_mode", "forged-weight-mode"),
    ("ingest_mode", "forged-ingest-mode"),
    ("max_output_tokens", True),
    ("max_sessions", 0),
])
def test_resealed_persisted_attestation_reaches_full_live_validation(field, value):
    record = valid_record()
    _set_all_persisted_attestations(record, field, value)
    _reseal_terminal_graph(record)
    with pytest.raises(ValueError, match="completion receipt service attestation is invalid"):
        validate_recovery_record(record)


def test_recovery_record_stale_digest_mutations_fail_closed():
    mutations = [
        lambda record: record["messages"].__setitem__(4, {"role": "assistant", "text": "forged", "loss": 0}),
        lambda record: record["terminal_binding"]["failed_terminal"]["actions"][2].__setitem__("decision", "continue"),
        lambda record: record["terminal_binding"]["corrective_terminal"]["actions"][3].__setitem__("decision", "terminate"),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["focused"]["output"].__setitem__("mode", "regression"),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["focused"]["logical_argv"].append("--forged"),
        lambda record: record["terminal_binding"]["bundle"]["validator"]["focused_argv"].__setitem__(1, "alternate-pass-emitter.py"),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["regression"].__setitem__("stdout_sha256", digest("forged-stdout")),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["regression"]["output"].__setitem__("action_count", 99),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["focused"].__setitem__("stderr_sha256", digest("forged-stderr")),
        lambda record: record["terminal_binding"]["validator_execution"].__setitem__("validator_spec_payload_sha256", digest("forged-spec-payload")),
        lambda record: record["terminal_binding"]["validator_execution"]["validators"]["focused"].__setitem__("terminal_payload_sha256", digest("forged-terminal-payload")),
        lambda record: record["terminal_binding"]["failed_terminal"]["actions"][0].__setitem__("completion_tokens", 0),
        lambda record: record["terminal_binding"]["corrective_terminal"]["metadata"]["completion_usage"][4].__setitem__("completion_tokens", 0),
        lambda record: record["terminal_binding"]["corrective_terminal"]["metadata"]["completion_usage"][4].__setitem__("completion_tokens", 513),
        lambda record: record["terminal_binding"]["corrective_terminal"]["metadata"]["completion_receipts"][4].__setitem__("assistant_message_sha256", digest("forged-assistant")),
        lambda record: record["terminal_binding"]["failed_terminal"]["metadata"]["completion_receipts"][0]["request"].__setitem__("model", "forged-model"),
        lambda record: record["terminal_binding"]["failed_terminal"]["metadata"]["completion_receipts"][0]["service_attestation"].__setitem__("runtime_image_sha256", digest("forged-runtime")),
        lambda record: record["terminal_binding"]["corrective_terminal"].__setitem__("schema", "emender-e97-corrective-terminal-v1"),
        lambda record: record["terminal_binding"]["completion_receipt"]["receipt"].__setitem__("accepted_terminal_sha256", digest("other")),
        lambda record: record["terminal_binding"]["artifacts"]["completion_receipt"].__setitem__("path", "receipts/orphan.json"),
        lambda record: record["source_provenance"].__setitem__("source_digests", [digest("not-generator")]),
    ]
    for mutate in mutations:
        record = valid_record()
        mutate(record)
        with pytest.raises(ValueError):
            validate_recovery_record(record)


@pytest.mark.parametrize("mutate", [
    lambda completion: completion.__setitem__("forged", True),
    lambda completion: completion["lease"].__setitem__("forged", True),
    lambda completion: completion["lease"].__setitem__("owner", 7),
    lambda completion: completion["lease"].__setitem__("attempt", True),
    lambda completion: completion["lease"].__setitem__("deadline_ns", "late"),
    lambda completion: completion["lease"].__setitem__("deadline_ns", 0),
])
def test_resealed_completion_and_lease_schemas_are_closed_and_typed(mutate):
    record = valid_record()
    mutate(record["terminal_binding"]["completion_receipt"])
    _reseal_terminal_graph(record)
    with pytest.raises(ValueError):
        validate_recovery_record(record)


def test_record_json_hash_is_stable():
    record = valid_record()
    assert recovery_record_fingerprint(record) == recovery_record_fingerprint(copy.deepcopy(record))
