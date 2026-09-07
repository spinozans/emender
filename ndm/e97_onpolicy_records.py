"""Canonical records and no-progress detection for E97 on-policy correction.

Model and teacher outputs are untrusted.  This module only describes immutable
execution receipts and validates already replayed, deterministically accepted
correction records.  Consumed V3/V4 evaluation identities are inadmissible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ndm.e97_agent_protocol import AgentProtocolError, parse_agent_turn, serialize_pi_messages


RECOVERY_RECORD_SCHEMA = "emender-e97-student-state-correction-v3"
TASK_IDENTITY_SCHEMA = "emender-e97-onpolicy-task-identity-v1"
NO_PROGRESS_SCHEMA = "emender-e97-no-progress-v1"
CONSUMED_V3_MANIFEST_SHA256 = (
    "ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09"
)
CONSUMED_V4_MANIFEST_SHA256 = (
    "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8"
)
CONSUMED_PANEL_MANIFEST_SHA256S = (
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
)
CONSUMED_ID_PREFIXES = ("pi-eval-v3-", "pi-eval-v4-")
CANONICAL_NO_PROGRESS_OBSERVATION = (
    "no_progress: this identical action produced no new workspace, source, or "
    "observation state; choose a different action or finish with a grounded blocker."
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALIDATOR_LOGICAL_RUNTIME = "@runtime-python"
_VALIDATOR_LOGICAL_PROGRAM = "@generator-source/scripts/e97_first_party_validator.py"


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON; reject NaN and non-JSON objects."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def completion_marker_relative_path(task_id: str, receipt_sha256: str) -> str:
    """Return the task-specific immutable completion-authority marker path."""

    return f"completed/{_require_digest(task_id, 'completion task identity')}/{_require_digest(receipt_sha256, 'completion receipt SHA-256')}.json"


def canonical_action(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Canonicalize a single bounded tool action for comparison and storage."""

    _require_name(tool_name, "tool_name")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    arguments_json = canonical_json(dict(arguments))
    # Reuse the live protocol parser as the final syntax authority.
    turn = parse_agent_turn(f"Action: {tool_name}\nArguments: {arguments_json}")
    if turn.kind != "tool_call":  # pragma: no cover - parser contract
        raise AssertionError("canonical action did not parse as a tool call")
    return turn.raw_text


def action_fingerprint(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Hash the tool name and normalized JSON arguments."""

    action = canonical_action(tool_name, arguments)
    return sha256_text(f"{NO_PROGRESS_SCHEMA}\0action\0{action}")


def _canonical_observation_state(value: Any) -> Any:
    """Treat acquired-observation sequences as a deduplicated information set."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        encoded = sorted({canonical_json(item) for item in value})
        return [json.loads(item) for item in encoded]
    raise ValueError("acquired_observations must be an object or sequence")


def task_identity(
    *,
    namespace: str,
    family_id: str,
    generator_source_digest: str,
    fixture_tree_digest: str,
    intent_digest: str,
) -> str:
    """Bind a task identity to its immutable generator, fixture, and intent."""

    _require_name(namespace, "task namespace")
    _require_name(family_id, "task family_id")
    values = {
        "schema": TASK_IDENTITY_SCHEMA,
        "namespace": namespace,
        "family_id": family_id,
        "generator_source_digest": _require_digest(
            generator_source_digest, "task generator_source_digest"),
        "fixture_tree_digest": _require_digest(
            fixture_tree_digest, "task fixture_tree_digest"),
        "intent_digest": _require_digest(intent_digest, "task intent_digest"),
    }
    return sha256_json(values)


def progress_fingerprint(
    *,
    workspace_state: Any,
    source_ledger: Any,
    acquired_observations: Any,
) -> str:
    """Hash the canonical task-visible world and acquired information state."""

    state = {
        "schema": NO_PROGRESS_SCHEMA,
        "workspace_state": workspace_state,
        "source_ledger": source_ledger,
        "acquired_observations": _canonical_observation_state(acquired_observations),
    }
    return sha256_json(state)


@dataclass(frozen=True)
class ActionProgressReceipt:
    """Post-execution state used to decide whether an action made progress."""

    tool_name: str
    arguments: Mapping[str, Any]
    workspace_state: Any
    source_ledger: Any
    acquired_observations: Any

    @property
    def action_fingerprint(self) -> str:
        return action_fingerprint(self.tool_name, self.arguments)

    @property
    def progress_fingerprint(self) -> str:
        return progress_fingerprint(
            workspace_state=self.workspace_state,
            source_ledger=self.source_ledger,
            acquired_observations=self.acquired_observations,
        )


@dataclass(frozen=True)
class NoProgressDecision:
    kind: str
    action_fingerprint: str
    progress_fingerprint: str
    observation: str | None = None


class NoProgressDetector:
    """Inject one recovery observation, then fail closed on another stale repeat.

    Receipts describe state *after* tool execution.  Consequently a duplicate
    read result does not expand the deduplicated observation set, while an edit,
    changed source ledger, or genuinely new observation changes the progress
    fingerprint and permits the same action again.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._recovery_injected: set[tuple[str, str]] = set()

    def observe(self, receipt: ActionProgressReceipt) -> NoProgressDecision:
        action = receipt.action_fingerprint
        progress = receipt.progress_fingerprint
        pair = (action, progress)
        if pair in self._seen:
            if pair in self._recovery_injected:
                return NoProgressDecision("terminate", action, progress)
            self._recovery_injected.add(pair)
            return NoProgressDecision(
                "inject_no_progress", action, progress,
                CANONICAL_NO_PROGRESS_OBSERVATION,
            )
        self._seen.add(pair)
        return NoProgressDecision("continue", action, progress)


def _require_fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} fields mismatch: {observed}")
    return value


def _validate_messages(
    messages: Any,
    *,
    divergence_index: int,
    target_start_index: int,
    observation_digest: str,
    system_prompt_sha256: str,
    max_messages: int,
    max_message_bytes: int,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not 5 <= len(messages) <= max_messages:
        raise ValueError("messages length is invalid")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        message = _require_fields(raw, {"role", "text", "loss"}, f"message {index}")
        role, text, loss = message["role"], message["text"], message["loss"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} role is invalid")
        if not isinstance(text, str) or len(text.encode("utf-8")) > max_message_bytes:
            raise ValueError(f"message {index} text is invalid or oversized")
        if isinstance(loss, bool) or loss not in {0, 1}:
            raise ValueError(f"message {index} loss must be zero or one")
        if role != "assistant" and loss:
            raise ValueError("only assistant messages may carry loss")
        normalized.append({"role": role, "text": text, "loss": int(loss)})
    if [normalized[0]["role"], normalized[1]["role"]] != ["system", "user"]:
        raise ValueError("records must begin with system then user")
    if sha256_text(normalized[0]["text"]) != system_prompt_sha256:
        raise ValueError("system prompt digest mismatch")
    if normalized[-1]["role"] != "assistant" or not normalized[-1]["text"].startswith("Final:"):
        raise ValueError("records require one terminal Final assistant turn")
    if normalized[-1]["loss"] != 1:
        raise ValueError("terminal Final must be a target")
    if not 2 <= divergence_index < len(normalized) - 2:
        raise ValueError("first divergence message index is invalid")
    if not divergence_index < target_start_index < len(normalized):
        raise ValueError("target start must follow the first divergence")
    if normalized[divergence_index]["role"] != "assistant" or normalized[divergence_index]["loss"]:
        raise ValueError("first divergent student action must be zero-loss assistant context")
    if normalized[divergence_index + 1]["role"] != "tool":
        raise ValueError("first divergent action must retain its authentic tool observation")
    if sha256_text(normalized[divergence_index + 1]["text"]) != observation_digest:
        raise ValueError("first divergence observation digest mismatch")
    if normalized[target_start_index]["role"] != "assistant":
        raise ValueError("target suffix must begin on an assistant message")
    for index, message in enumerate(normalized):
        if message["role"] == "assistant":
            expected_loss = int(index >= target_start_index)
            if message["loss"] != expected_loss:
                raise ValueError("assistant targets must be one contiguous corrective suffix")
    # After the initial user turn, assistant/tool messages must alternate until Final.
    expected = "assistant"
    for index, message in enumerate(normalized[2:], start=2):
        if message["role"] != expected:
            raise ValueError(f"message {index} violates assistant/tool alternation")
        expected = "tool" if expected == "assistant" else "assistant"
    return normalized


def _validate_action_progress_history(
    actions: Any,
    *,
    terminated: bool,
    permitted_termination_indexes: set[int] | None = None,
) -> list[Mapping[str, Any]]:
    """Recompute all no-progress decisions from stored action/progress pairs."""

    if not isinstance(actions, list) or not actions:
        raise ValueError("terminal actions are required")
    detector = NoProgressDetector()
    normalized: list[Mapping[str, Any]] = []
    fields = {
        "sequence", "tool_call_id", "tool_name", "arguments", "arguments_json", "raw_observation",
        "raw_observation_sha256", "effective_observation", "observation_sha256", "is_error",
        "workspace_state", "source_ledger", "acquired_observations", "action_fingerprint",
        "progress_fingerprint", "completion_tokens", "decision",
    }
    for index, action in enumerate(actions):
        action = _require_fields(action, fields, f"terminal action {index}")
        if action["sequence"] != index or not isinstance(action["tool_call_id"], str) or not action["tool_call_id"]:
            raise ValueError("terminal action sequence or call ID is invalid")
        if (isinstance(action["completion_tokens"], bool)
                or not isinstance(action["completion_tokens"], int)
                or action["completion_tokens"] < 1):
            raise ValueError("terminal action completion_tokens is invalid")
        if not isinstance(action["arguments"], Mapping) or not isinstance(action["arguments_json"], str):
            raise ValueError("terminal action arguments are invalid")
        try:
            parsed_arguments = json.loads(action["arguments_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("terminal action arguments JSON is invalid") from exc
        if parsed_arguments != action["arguments"]:
            raise ValueError("terminal action arguments JSON does not bind object")
        if (action["raw_observation_sha256"] != sha256_json(action["raw_observation"])
                or not isinstance(action["effective_observation"], str)
                or action["observation_sha256"] != sha256_text(action["effective_observation"])):
            raise ValueError("terminal action observation hashes are invalid")
        receipt = ActionProgressReceipt(
            action["tool_name"],
            action["arguments"],
            action["workspace_state"],
            action["source_ledger"],
            action["acquired_observations"],
        )
        decision = detector.observe(receipt)
        if (action["action_fingerprint"] != receipt.action_fingerprint
                or action["progress_fingerprint"] != receipt.progress_fingerprint
                or action["decision"] != decision.kind):
            raise ValueError("terminal action/progress decision is not structurally replayable")
        if decision.observation is not None and action["effective_observation"] != decision.observation:
            raise ValueError("terminal no-progress injection is not canonical")
        normalized.append(action)
    if terminated:
        if normalized[-1]["decision"] != "terminate":
            raise ValueError("terminated terminal must end in a structural no-progress termination")
    else:
        permitted = permitted_termination_indexes or set()
        if any(action["decision"] == "terminate" and index not in permitted
               for index, action in enumerate(normalized)):
            raise ValueError("successful terminal may not contain new no-progress termination")
    return normalized


def _validate_terminal_completion_usage(terminal: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    """Bind every accepted assistant turn to bounded service token accounting."""

    limit = limits.get("completion_tokens")
    metadata, messages, actions = terminal.get("metadata"), terminal.get("messages"), terminal.get("actions")
    if (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
            or not isinstance(metadata, Mapping) or not isinstance(messages, list)
            or not isinstance(actions, list)):
        raise ValueError("terminal completion token limits are invalid")
    usage = metadata.get("completion_usage")
    assistants = [message for message in messages if isinstance(message, Mapping) and message.get("role") == "assistant"]
    if not isinstance(usage, list) or len(usage) != len(assistants):
        raise ValueError("terminal completion usage does not cover every assistant turn")
    total = 0
    for index, (entry, message) in enumerate(zip(usage, assistants)):
        if (not isinstance(entry, Mapping) or set(entry) != {"sequence", "completion_tokens"}
                or entry.get("sequence") != index or isinstance(entry.get("completion_tokens"), bool)
                or not isinstance(entry.get("completion_tokens"), int)
                or not 1 <= entry["completion_tokens"] <= limit):
            raise ValueError("terminal completion usage receipt is invalid")
        try:
            body = serialize_pi_messages([message], append_assistant_header=False).removeprefix("Assistant:\n")
        except AgentProtocolError as exc:
            raise ValueError("terminal assistant completion serialization is invalid") from exc
        if len(body.encode("utf-8")) > limit * 8:
            raise ValueError("terminal assistant body exceeds deterministic token byte ceiling")
        total += entry["completion_tokens"]
    if metadata.get("completion_tokens") != total:
        raise ValueError("terminal completion token total is invalid")
    if len(actions) > len(usage):
        raise ValueError("terminal actions exceed completion usage")
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or action.get("completion_tokens") != usage[index]["completion_tokens"]:
            raise ValueError("terminal action completion usage does not bind its assistant turn")


def validate_recovery_record(
    value: Any,
    *,
    forbidden_source_digests: Sequence[str] = CONSUMED_PANEL_MANIFEST_SHA256S,
    forbidden_id_prefixes: Sequence[str] = CONSUMED_ID_PREFIXES,
    max_messages: int = 32,
    max_message_bytes: int = 64 << 10,
) -> dict[str, Any]:
    """Validate and normalize one deterministically accepted correction record."""

    record = _require_fields(value, {
        "schema", "split", "task", "student", "teacher", "runtime",
        "first_divergence", "messages", "validator_receipt", "source_provenance", "terminal_binding",
    }, "record")
    if record["schema"] != RECOVERY_RECORD_SCHEMA:
        raise ValueError("unsupported recovery record schema")
    if record["split"] not in {"train", "validation"}:
        raise ValueError("record split must be train or validation")

    task = _require_fields(record["task"], {
        "namespace", "family_id", "identity", "generator_source_digest",
        "fixture_tree_digest", "intent_digest",
    }, "task")
    namespace = _require_name(task["namespace"], "task namespace")
    family = _require_name(task["family_id"], "task family_id")
    expected_namespace = "e97-train-" if record["split"] == "train" else "e97-dev-"
    if not namespace.startswith(expected_namespace):
        raise ValueError("task namespace does not match record split")
    if any(namespace.startswith(prefix) or family.startswith(prefix) for prefix in forbidden_id_prefixes):
        raise ValueError("consumed evaluation identity is inadmissible")
    for key in ("identity", "generator_source_digest", "fixture_tree_digest", "intent_digest"):
        _require_digest(task[key], f"task {key}")
    derived_task_identity = task_identity(
        namespace=namespace,
        family_id=family,
        generator_source_digest=task["generator_source_digest"],
        fixture_tree_digest=task["fixture_tree_digest"],
        intent_digest=task["intent_digest"],
    )
    if task["identity"] != derived_task_identity:
        raise ValueError("task identity is not bound to its immutable inputs")

    student = _require_fields(record["student"], {
        "rollout_identity", "checkpoint_sha256", "decode",
    }, "student")
    _require_digest(student["rollout_identity"], "student rollout_identity")
    _require_digest(student["checkpoint_sha256"], "student checkpoint_sha256")
    if not isinstance(student["decode"], Mapping) or not student["decode"]:
        raise ValueError("student decode identity must be a non-empty object")
    canonical_json(student["decode"])

    teacher = _require_fields(record["teacher"], {"tier", "model_revision"}, "teacher")
    if teacher["tier"] not in {"luna", "terra", "sol", "human"}:
        raise ValueError("teacher tier is invalid")
    if not isinstance(teacher["model_revision"], str) or not teacher["model_revision"]:
        raise ValueError("teacher model revision is required")

    runtime = _require_fields(record["runtime"], {
        "schema_digest", "controller_digest", "system_prompt_sha256", "tool_schema_digest",
    }, "runtime")
    for key, item in runtime.items():
        _require_digest(item, f"runtime {key}")

    divergence = _require_fields(record["first_divergence"], {
        "message_index", "target_start_message_index", "class",
        "student_action_canonical", "pre_state_digest", "observation_digest",
    }, "first_divergence")
    if isinstance(divergence["message_index"], bool) or not isinstance(divergence["message_index"], int):
        raise ValueError("first divergence message_index must be an integer")
    if (isinstance(divergence["target_start_message_index"], bool)
            or not isinstance(divergence["target_start_message_index"], int)):
        raise ValueError("target start message index must be an integer")
    _require_name(divergence["class"], "first divergence class")
    _require_digest(divergence["pre_state_digest"], "first divergence pre_state_digest")
    observation_digest = _require_digest(
        divergence["observation_digest"], "first divergence observation_digest")

    messages = _validate_messages(
        record["messages"],
        divergence_index=divergence["message_index"],
        target_start_index=divergence["target_start_message_index"],
        observation_digest=observation_digest,
        system_prompt_sha256=runtime["system_prompt_sha256"],
        max_messages=max_messages,
        max_message_bytes=max_message_bytes,
    )
    binding = _require_fields(record["terminal_binding"], {
        "artifacts", "bundle", "bundle_sha256", "archive_sha256", "failed_terminal", "failed_terminal_sha256",
        "corrective_terminal", "corrective_terminal_sha256", "validator_execution",
        "validator_execution_sha256", "completion_receipt", "completion_receipt_sha256",
        "lease_identity", "prefix_sha256", "model_prefix_sha256", "terminating_intervention",
    }, "terminal binding")
    for key in ("bundle_sha256", "archive_sha256", "failed_terminal_sha256", "corrective_terminal_sha256",
                "validator_execution_sha256", "completion_receipt_sha256", "lease_identity", "prefix_sha256", "model_prefix_sha256"):
        _require_digest(binding[key], f"terminal binding {key}")
    artifacts = _require_fields(binding["artifacts"], {
        "registry", "generation_receipt", "generator_manifest", "source_archive", "overlap_receipt",
        "tasks_collection", "bundle", "archive", "private_spec", "failed_terminal",
        "corrective_terminal", "validator_execution", "completion_receipt",
    }, "terminal artifact references")
    for name, reference in artifacts.items():
        reference = _require_fields(reference, {"sha256", "path"}, f"terminal artifact {name}")
        _require_digest(reference["sha256"], f"terminal artifact {name} SHA-256")
        if (not isinstance(reference["path"], str) or not reference["path"]
                or reference["path"].startswith("/") or ".." in reference["path"].split("/")):
            raise ValueError(f"terminal artifact {name} path is invalid")
    for name, digest in {
        "bundle": binding["bundle_sha256"], "archive": binding["archive_sha256"],
        "failed_terminal": binding["failed_terminal_sha256"],
        "corrective_terminal": binding["corrective_terminal_sha256"],
        "validator_execution": binding["validator_execution_sha256"],
        "completion_receipt": binding["completion_receipt_sha256"],
    }.items():
        if artifacts[name]["sha256"] != digest:
            raise ValueError("terminal inline object digest does not match artifact reference")
    if artifacts["completion_receipt"]["path"] != completion_marker_relative_path(
            task["identity"], binding["completion_receipt_sha256"]):
        raise ValueError("completion receipt artifact must be the task-specific finalized marker")
    bundle = binding["bundle"]
    if not isinstance(bundle, Mapping) or sha256_json(bundle) != binding["bundle_sha256"]:
        raise ValueError("bundle bytes do not bind terminal relation")
    bundle_task, bundle_runtime = bundle.get("task"), bundle.get("runtime")
    if (not isinstance(bundle_task, Mapping) or not isinstance(bundle_runtime, Mapping)
            or any(bundle_task.get(key) != item for key, item in task.items())
            or any(bundle_runtime.get(key) != item for key, item in runtime.items())):
        raise ValueError("record task/runtime do not match sealed bundle")
    fixture = bundle.get("fixture")
    if not isinstance(fixture, Mapping) or fixture.get("artifact_sha256") != binding["archive_sha256"]:
        raise ValueError("archive does not bind sealed bundle fixture")

    failed = binding["failed_terminal"]
    if (not isinstance(failed, Mapping) or failed.get("status") != "no_progress_terminated"
            or sha256_json(failed) != binding["failed_terminal_sha256"]):
        raise ValueError("correction must derive from its sealed no_progress_terminated terminal")
    if student["rollout_identity"] != binding["failed_terminal_sha256"]:
        raise ValueError("student rollout identity does not bind failed terminal")
    failed_actions = failed.get("actions")
    failed_messages = failed.get("messages")
    if not isinstance(failed_actions, list) or not failed_actions or not isinstance(failed_messages, list):
        raise ValueError("failed terminal transcript/actions are required")
    _validate_action_progress_history(failed_actions, terminated=True)
    _validate_terminal_completion_usage(failed, bundle["limits"])
    if len(failed_messages) != 2 + len(failed_actions) * 2 - 1:
        raise ValueError("failed terminal must omit only its terminating observation")

    def assistant_text(message: Any, name: str) -> str:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise ValueError(f"{name} assistant message is invalid")
        try:
            encoded = serialize_pi_messages([message], append_assistant_header=False)
        except AgentProtocolError as exc:
            raise ValueError(f"{name} native assistant serialization is invalid") from exc
        if not encoded.startswith("Assistant:\n"):
            raise ValueError(f"{name} native assistant serialization is invalid")
        return encoded[len("Assistant:\n"):]

    derived_prefix = [
        {"role": "system", "text": failed_messages[0].get("content"), "loss": 0},
        {"role": "user", "text": failed_messages[1].get("content"), "loss": 0},
    ]
    if not isinstance(derived_prefix[0]["text"], str) or not isinstance(derived_prefix[1]["text"], str):
        raise ValueError("failed terminal system/user content is invalid")
    for index, action in enumerate(failed_actions):
        assistant_index = 2 + index * 2
        assistant = failed_messages[assistant_index] if assistant_index < len(failed_messages) else None
        text = assistant_text(assistant, f"failed action {index}")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
            raise ValueError("failed terminal tool call is invalid")
        call = calls[0]
        function = call.get("function")
        raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
        if (call.get("id") != action.get("tool_call_id") or function.get("name") != action.get("tool_name")
                or not isinstance(raw_arguments, str) or raw_arguments != action.get("arguments_json")
                or text != f"Action: {action.get('tool_name')}\nArguments: {raw_arguments}"
                or json.loads(raw_arguments) != action.get("arguments")):
            raise ValueError("failed terminal action/message/function.arguments linkage mismatch")
        if action.get("raw_observation_sha256") != sha256_json(action.get("raw_observation")):
            raise ValueError("failed terminal raw observation digest mismatch")
        if (not isinstance(action.get("effective_observation"), str)
                or action.get("observation_sha256") != sha256_text(action["effective_observation"])):
            raise ValueError("failed terminal effective observation digest mismatch")
        derived_prefix.append({"role": "assistant", "text": text, "loss": 0})
        if index < len(failed_actions) - 1:
            tool = failed_messages[assistant_index + 1]
            if (not isinstance(tool, Mapping) or tool.get("role") != "tool"
                    or tool.get("tool_call_id") != action["tool_call_id"]
                    or tool.get("content") != action["effective_observation"]):
                raise ValueError("failed terminal model-facing observation linkage mismatch")
            derived_prefix.append({"role": "tool", "text": action["effective_observation"], "loss": 0})
    intervention = binding["terminating_intervention"]
    if (not isinstance(intervention, Mapping) or set(intervention) != {"action_index", "effective_observation", "sent_to_model"}
            or intervention.get("action_index") != len(failed_actions) - 1
            or intervention.get("sent_to_model") is not False
            or intervention.get("effective_observation") != failed_actions[-1]["effective_observation"]):
        raise ValueError("terminating observation must be one explicit unsent controller intervention")
    derived_prefix.append({"role": "tool", "text": intervention["effective_observation"], "loss": 0})
    if messages[:divergence["target_start_message_index"]] != derived_prefix:
        raise ValueError("zero-loss prefix is not exactly the failed model-facing transcript plus intervention")
    if sha256_json(derived_prefix) != binding["model_prefix_sha256"]:
        raise ValueError("model correction prefix digest does not bind failed transcript")

    action_index = (divergence["message_index"] - 2) // 2
    if divergence["message_index"] != 2 + action_index * 2 or action_index >= len(failed_actions):
        raise ValueError("first divergence does not select a failed action")
    divergent = failed_actions[action_index]
    try:
        parsed = parse_agent_turn(messages[divergence["message_index"]]["text"])
    except AgentProtocolError as exc:
        raise ValueError("first divergent action is not protocol text") from exc
    if (parsed.kind != "tool_call" or parsed.tool_name != divergent["tool_name"]
            or parsed.arguments != divergent["arguments"]
            or divergence["student_action_canonical"] != canonical_action(parsed.tool_name, parsed.arguments)
            or divergence["pre_state_digest"] != divergent.get("progress_fingerprint")
            or divergence["observation_digest"] != sha256_text(divergent["effective_observation"])):
        raise ValueError("first divergence action/progress/observation binding mismatch")

    corrective = binding["corrective_terminal"]
    if (not isinstance(corrective, Mapping) or sha256_json(corrective) != binding["corrective_terminal_sha256"]
            or corrective.get("status") != "success" or corrective.get("failed_terminal_sha256") != binding["failed_terminal_sha256"]
            or corrective.get("prefix_sha256") != binding["prefix_sha256"]
            or corrective.get("correction_start_message_index") != divergence["target_start_message_index"]):
        raise ValueError("corrective terminal success/resume/prefix relation is invalid")
    corrective_messages, corrective_actions = corrective.get("messages"), corrective.get("actions")
    if (not isinstance(corrective_messages, list) or not isinstance(corrective_actions, list)
            or corrective_messages[:len(failed_messages)] != failed_messages
            or len(corrective_actions) < len(failed_actions)
            or corrective_actions[:len(failed_actions)] != failed_actions):
        raise ValueError("corrective terminal does not retain failed messages/actions")
    _validate_action_progress_history(
        corrective_actions,
        terminated=False,
        permitted_termination_indexes={len(failed_actions) - 1},
    )
    _validate_terminal_completion_usage(corrective, bundle["limits"])
    expected_intervention = {"role": "tool", "tool_call_id": failed_actions[-1]["tool_call_id"],
                             "content": intervention["effective_observation"], "controller_intervention": True}
    if corrective_messages[len(failed_messages):divergence["target_start_message_index"]] != [expected_intervention]:
        raise ValueError("corrective terminal intervention relation is invalid")
    suffix_messages: list[dict[str, Any]] = []
    cursor = divergence["target_start_message_index"]
    for index, action in enumerate(corrective_actions[len(failed_actions):], start=len(failed_actions)):
        assistant = corrective_messages[cursor] if cursor < len(corrective_messages) else None
        text = assistant_text(assistant, f"corrective action {index}")
        calls = assistant.get("tool_calls")
        call = calls[0] if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], Mapping) else None
        function = call.get("function") if isinstance(call, Mapping) else None
        raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
        tool = corrective_messages[cursor + 1] if cursor + 1 < len(corrective_messages) else None
        if (not isinstance(function, Mapping) or action.get("sequence") != index
                or call.get("id") != action.get("tool_call_id")
                or function.get("name") != action.get("tool_name") or raw_arguments != action.get("arguments_json")
                or not isinstance(raw_arguments, str) or json.loads(raw_arguments) != action.get("arguments")
                or action.get("raw_observation_sha256") != sha256_json(action.get("raw_observation"))
                or not isinstance(action.get("effective_observation"), str)
                or action.get("observation_sha256") != sha256_text(action["effective_observation"])
                or text != f"Action: {action.get('tool_name')}\nArguments: {raw_arguments}"
                or not isinstance(tool, Mapping) or tool.get("role") != "tool"
                or tool.get("tool_call_id") != action.get("tool_call_id")
                or tool.get("content") != action.get("effective_observation")):
            raise ValueError("corrective terminal message/action/function.arguments linkage mismatch")
        suffix_messages.extend(({"role": "assistant", "text": text, "loss": 1},
                                {"role": "tool", "text": action["effective_observation"], "loss": 0}))
        cursor += 2
    final = corrective_messages[cursor] if cursor < len(corrective_messages) else None
    if (cursor + 1 != len(corrective_messages) or not isinstance(final, Mapping)
            or final.get("role") != "assistant" or not isinstance(final.get("content"), str)
            or not final["content"].startswith("Final:")):
        raise ValueError("corrective terminal final message relation is invalid")
    suffix_messages.append({"role": "assistant", "text": final["content"], "loss": 1})
    if messages[divergence["target_start_message_index"]:] != suffix_messages:
        raise ValueError("corrective suffix messages are not mechanically derived")

    execution = binding["validator_execution"]
    if not isinstance(execution, Mapping) or sha256_json(execution) != binding["validator_execution_sha256"]:
        raise ValueError("validator execution bytes do not bind terminal relation")
    if (execution.get("schema") != "emender-e97-first-party-validator-receipt-v3" or execution.get("status") != "pass"
            or execution.get("task_identity") != task["identity"] or execution.get("bundle_sha256") != binding["bundle_sha256"]
            or execution.get("fixture_tree_digest") != task["fixture_tree_digest"]
            or execution.get("terminal_sha256") != binding["corrective_terminal_sha256"]
            or execution.get("validator_spec_digest") != bundle.get("validator", {}).get("spec_digest")
            or execution.get("runtime_schema_digest") != runtime["schema_digest"]):
        raise ValueError("validator execution task/bundle/terminal/spec/runtime relation is invalid")
    execution_fields = {
        "schema", "status", "task_identity", "bundle_sha256", "fixture_tree_digest",
        "archive_expanded_bytes", "configured_limits", "terminal_sha256", "validator_spec_digest",
        "runtime_schema_digest", "validators",
    }
    if set(execution) != execution_fields or execution.get("configured_limits") != bundle.get("limits"):
        raise ValueError("validator execution limits/schema are invalid")
    if (isinstance(execution.get("archive_expanded_bytes"), bool)
            or not isinstance(execution.get("archive_expanded_bytes"), int)
            or execution["archive_expanded_bytes"] < 0
            or execution["archive_expanded_bytes"] > bundle["limits"]["disk_bytes"]):
        raise ValueError("validator execution archive bounds are invalid")
    validators = _require_fields(execution.get("validators"), {"focused", "regression"}, "validator execution outputs")
    for mode in ("focused", "regression"):
        output = _require_fields(validators[mode], {
            "logical_argv", "logical_argv_sha256", "stdout_sha256", "stderr_sha256", "output",
        }, f"validator {mode} output")
        expected_argv = [
            _VALIDATOR_LOGICAL_RUNTIME, _VALIDATOR_LOGICAL_PROGRAM, "--mode", mode,
            "--spec", "<spec>", "--terminal", "<terminal>",
        ]
        if (bundle["validator"].get(f"{mode}_argv") != expected_argv[:4]
                or output["logical_argv"] != expected_argv
                or output["logical_argv_sha256"] != sha256_json(expected_argv)
                or output["stdout_sha256"] != sha256_text(canonical_json({
                    "action_count": len(corrective_actions), "mode": mode, "status": "pass",
                }) + "\n")
                or output["stderr_sha256"] != sha256_text("")
                or output["output"] != {
                    "mode": mode, "status": "pass", "action_count": len(corrective_actions),
                }):
            raise ValueError("validator output is not exact focused/regression evidence")
    receipt = _require_fields(record["validator_receipt"], {
        "validator_digest", "input_digest", "postcondition_digest", "action_graph_digest", "passed", "cycle_free",
    }, "validator_receipt")
    for key in ("validator_digest", "input_digest", "postcondition_digest", "action_graph_digest"):
        _require_digest(receipt[key], f"validator receipt {key}")
    if receipt["passed"] is not True or receipt["cycle_free"] is not True:
        raise ValueError("only passed, cycle-free validator receipts are trainable")
    if (receipt["validator_digest"] != execution["validator_spec_digest"]
            or receipt["input_digest"] != task["fixture_tree_digest"]
            or receipt["postcondition_digest"] != binding["validator_execution_sha256"]
            or receipt["action_graph_digest"] != sha256_json(corrective_actions)):
        raise ValueError("compact validator receipt does not bind full validator/action graph")

    completion = binding["completion_receipt"]
    if not isinstance(completion, Mapping) or sha256_json(completion) != binding["completion_receipt_sha256"]:
        raise ValueError("completion receipt bytes do not bind terminal relation")
    lease = completion.get("lease")
    completion_body = completion.get("receipt")
    if not isinstance(completion_body, Mapping):
        raise ValueError("completion receipt body is invalid")
    _require_fields(completion_body, {
        "task_id", "bundle_sha256", "archive_sha256", "failed_terminal_sha256",
        "accepted_terminal_sha256", "validator_receipt_sha256", "validator_status",
    }, "completion receipt body")
    if (completion.get("schema") != "emender-e97-task-completion-receipt-v3" or not isinstance(lease, Mapping)
            or completion.get("task_id") != task["identity"]
            or lease.get("task_id") != task["identity"] or lease.get("identity") != binding["lease_identity"]
            or sha256_json({"task_id": lease.get("task_id"), "owner": lease.get("owner"), "attempt": lease.get("attempt"), "deadline_ns": lease.get("deadline_ns")}) != binding["lease_identity"]
            or completion_body.get("task_id") != task["identity"]
            or completion_body.get("bundle_sha256") != binding["bundle_sha256"]
            or completion_body.get("archive_sha256") != binding["archive_sha256"]
            or completion_body.get("failed_terminal_sha256") != binding["failed_terminal_sha256"]
            or completion_body.get("accepted_terminal_sha256") != binding["corrective_terminal_sha256"]
            or completion_body.get("validator_receipt_sha256") != binding["validator_execution_sha256"]
            or completion_body.get("validator_status") != "pass"):
        raise ValueError("completion receipt lease/task/bundle/archive/terminal/validator relation is invalid")

    provenance = _require_fields(record["source_provenance"], {
        "source_digests", "forbidden_panel_digests_checked",
    }, "source_provenance")
    if not isinstance(provenance["source_digests"], list):
        raise ValueError("source provenance digests must be a list")
    sources = [_require_digest(item, "source provenance digest")
               for item in provenance["source_digests"]]
    if len(sources) != len(set(sources)):
        raise ValueError("source provenance digests must be unique")
    forbidden = {_require_digest(item, "forbidden source digest")
                 for item in forbidden_source_digests}
    if forbidden.intersection(sources):
        raise ValueError("consumed evaluation source is inadmissible")
    if task["generator_source_digest"] not in sources:
        raise ValueError("source provenance must include the exact bundle generator digest")
    checked = provenance["forbidden_panel_digests_checked"]
    if not isinstance(checked, list) or not forbidden.issubset(checked):
        raise ValueError("record lacks the required forbidden-panel check receipt")
    for item in checked:
        _require_digest(item, "checked forbidden panel digest")

    # Round-trip through canonical JSON to detach callers' mutable mappings.
    return json.loads(canonical_json(dict(record)))


def recovery_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the immutable identity of one validated correction record."""

    normalized = validate_recovery_record(record)
    return sha256_text(f"{RECOVERY_RECORD_SCHEMA}\0{canonical_json(normalized)}")
