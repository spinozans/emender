import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import tiktoken

from ndm.data.masked_sft_dataset import RECORD_INDEX, sha256
from ndm.e97_onpolicy_records import (
    CANONICAL_NO_PROGRESS_OBSERVATION,
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
    RECOVERY_RECORD_SCHEMA,
    ActionProgressReceipt,
    NoProgressDetector,
    action_fingerprint,
    canonical_action,
    progress_fingerprint,
    recovery_record_fingerprint,
    sha256_json,
    sha256_text,
    task_identity,
    validate_recovery_record,
)


def digest(label):
    return sha256_text(label)


def valid_record(*, split="train"):
    system = "Use bounded tools and ground the final in observed evidence."
    bad = canonical_action("read", {"path": "missing.txt", "offset": 1, "limit": 40})
    correction = canonical_action("read", {"path": "facts/value.txt", "offset": 1, "limit": 40})
    failure = "FileNotFoundError: missing.txt\nCommand exited with code 1"
    task = {
        "namespace": f"e97-{'train' if split == 'train' else 'dev'}-opaque-read-v1",
        "family_id": "opaque-read-recovery-v1",
        "generator_source_digest": digest("generator-v1"),
        "fixture_tree_digest": digest(f"fixture-{split}"),
        "intent_digest": digest("read-the-declared-value"),
    }
    task["identity"] = task_identity(**task)
    return {
        "schema": RECOVERY_RECORD_SCHEMA,
        "split": split,
        "task": task,
        "student": {
            "rollout_identity": digest(f"rollout-{split}"),
            "checkpoint_sha256": digest("checkpoint-aae654aa"),
            "decode": {"temperature": 0, "seed": 0, "max_output_tokens": 512},
        },
        "teacher": {"tier": "luna", "model_revision": "gpt-5.6-luna@test"},
        "runtime": {
            "schema_digest": digest("runtime-schema-v1"),
            "controller_digest": digest("controller-v1"),
            "system_prompt_sha256": sha256_text(system),
            "tool_schema_digest": digest("tool-schema-v1"),
        },
        "first_divergence": {
            "message_index": 2,
            "target_start_message_index": 4,
            "class": "wrong-path",
            "student_action_canonical": bad,
            "pre_state_digest": digest("pre-state"),
            "observation_digest": sha256_text(failure),
        },
        "messages": [
            {"role": "system", "text": system, "loss": 0},
            {"role": "user", "text": "Read the declared value from the workspace.", "loss": 0},
            {"role": "assistant", "text": bad, "loss": 0},
            {"role": "tool", "text": failure, "loss": 0},
            {"role": "assistant", "text": correction, "loss": 1},
            {"role": "tool", "text": "opal-731", "loss": 0},
            {"role": "assistant", "text": "Final: `facts/value.txt` contains `opal-731`.", "loss": 1},
        ],
        "validator_receipt": {
            "validator_digest": digest("validator-v1"),
            "input_digest": digest(f"validator-input-{split}"),
            "postcondition_digest": digest("postcondition-pass"),
            "action_graph_digest": digest("read-missing-then-declared"),
            "passed": True,
            "cycle_free": True,
        },
        "source_provenance": {
            "source_digests": [digest(f"fresh-source-{split}")],
            "forbidden_panel_digests_checked": [
                CONSUMED_V3_MANIFEST_SHA256, CONSUMED_V4_MANIFEST_SHA256,
            ],
        },
    }


def test_action_and_progress_fingerprints_are_canonical():
    first = action_fingerprint("read", {"limit": 40, "path": "a.txt", "offset": 1})
    reordered = action_fingerprint("read", {"offset": 1, "path": "a.txt", "limit": 40})
    changed = action_fingerprint("read", {"offset": 1, "path": "b.txt", "limit": 40})
    assert first == reordered and first != changed
    assert canonical_action("read", {"path": "a.txt", "limit": 40}) == (
        'Action: read\nArguments: {"limit":40,"path":"a.txt"}')

    state_a = progress_fingerprint(
        workspace_state={"tree": digest("tree")},
        source_ledger={"src-1": digest("page")},
        acquired_observations=[{"source": "src-1", "view": "a"},
                               {"source": "src-1", "view": "a"}],
    )
    state_b = progress_fingerprint(
        workspace_state={"tree": digest("tree")},
        source_ledger={"src-1": digest("page")},
        acquired_observations=[{"source": "src-1", "view": "a"}],
    )
    assert state_a == state_b


def test_no_progress_detector_injects_once_then_terminates():
    detector = NoProgressDetector()
    stale = ActionProgressReceipt(
        tool_name="read",
        arguments={"path": "missing.txt", "offset": 1, "limit": 40},
        workspace_state={"tree": digest("unchanged")},
        source_ledger={},
        acquired_observations=[{"error": "missing.txt"}],
    )
    assert detector.observe(stale).kind == "continue"
    injected = detector.observe(stale)
    assert injected.kind == "inject_no_progress"
    assert injected.observation == CANONICAL_NO_PROGRESS_OBSERVATION
    assert detector.observe(stale).kind == "terminate"


def test_repeated_action_after_changed_state_is_progress():
    detector = NoProgressDetector()
    before = ActionProgressReceipt(
        "bash", {"command": "pytest -q"}, {"tree": digest("broken")}, {}, ["failed"])
    after_edit = ActionProgressReceipt(
        "bash", {"command": "pytest -q"}, {"tree": digest("fixed")}, {}, ["failed", "passed"])
    assert detector.observe(before).kind == "continue"
    assert detector.observe(after_edit).kind == "continue"


def test_unrelated_action_cannot_reset_stale_pair_recovery_budget():
    detector = NoProgressDetector()
    stale = ActionProgressReceipt(
        "read", {"path": "missing.txt"}, {"tree": digest("same")}, {}, ["missing"])
    unrelated = ActionProgressReceipt(
        "read", {"path": "other-missing.txt"}, {"tree": digest("same")}, {}, ["missing"])
    assert detector.observe(stale).kind == "continue"
    assert detector.observe(stale).kind == "inject_no_progress"
    assert detector.observe(unrelated).kind == "continue"
    assert detector.observe(stale).kind == "terminate"


def test_recovery_record_requires_zero_loss_bad_prefix_and_fresh_sources():
    record = valid_record()
    normalized = validate_recovery_record(record)
    assert normalized == record
    assert recovery_record_fingerprint(record) == recovery_record_fingerprint(copy.deepcopy(record))

    bad_mask = copy.deepcopy(record)
    bad_mask["messages"][2]["loss"] = 1
    with pytest.raises(ValueError, match="zero-loss|contiguous"):
        validate_recovery_record(bad_mask)

    for consumed_digest in (CONSUMED_V3_MANIFEST_SHA256, CONSUMED_V4_MANIFEST_SHA256):
        consumed_source = copy.deepcopy(record)
        consumed_source["source_provenance"]["source_digests"] = [consumed_digest]
        with pytest.raises(ValueError, match="consumed evaluation source"):
            validate_recovery_record(consumed_source)

    v4_family = copy.deepcopy(record)
    v4_family["task"]["family_id"] = "pi-eval-v4-leak"
    with pytest.raises(ValueError, match="consumed evaluation identity"):
        validate_recovery_record(v4_family)

    relabeled = copy.deepcopy(record)
    relabeled["task"]["fixture_tree_digest"] = digest("different-fixture")
    with pytest.raises(ValueError, match="not bound"):
        validate_recovery_record(relabeled)

    failed_receipt = copy.deepcopy(record)
    failed_receipt["validator_receipt"]["passed"] = False
    with pytest.raises(ValueError, match="passed, cycle-free"):
        validate_recovery_record(failed_receipt)


def test_correction_builder_preserves_failure_as_zero_loss_context(tmp_path):
    source = tmp_path / "verified.jsonl"
    train = valid_record(split="train")
    validation = valid_record(split="validation")
    source.write_text(
        json.dumps(train, sort_keys=True) + "\n" +
        json.dumps(validation, sort_keys=True) + "\n"
    )
    authority = tmp_path / "authority"
    subprocess.run([
        sys.executable,
        "-m", "scripts.build_e97_onpolicy_correction_sft",
        "--source-jsonl", str(source),
        "--source-sha256", sha256(source),
        "--output-root", str(authority),
    ], check=True, capture_output=True, text=True)

    manifest = json.loads((authority / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts"]["records"] == 2
    assert manifest["counts"]["train_records"] == 1
    assert manifest["counts"]["validation_records"] == 1
    assert manifest["counts"]["assistant_target_tokens"] > 0
    assert manifest["counts"]["zero_loss_context_tokens"] > 0
    assert manifest["forbidden_evaluation"]["v3_manifest_sha256"] == CONSUMED_V3_MANIFEST_SHA256
    assert manifest["forbidden_evaluation"]["v4_manifest_sha256"] == CONSUMED_V4_MANIFEST_SHA256

    index = (authority / "records.idx").read_bytes()
    first_offset, first_tokens, first_targets, first_split = RECORD_INDEX.unpack_from(index, 0)
    assert first_offset == 0 and first_split == 0 and first_targets > 0
    tokens = np.fromfile(authority / "tokens.uint32.bin", dtype=np.uint32)[:first_tokens]
    masks = np.fromfile(authority / "assistant_mask.uint8.bin", dtype=np.uint8)[:first_tokens]
    encoding = tiktoken.get_encoding("p50k_base")
    targeted = b"".join(
        encoding.decode_single_token_bytes(int(token))
        for token, mask in zip(tokens, masks) if mask
    ).decode(errors="replace")
    context = b"".join(
        encoding.decode_single_token_bytes(int(token))
        for token, mask in zip(tokens, masks) if not mask
    ).decode(errors="replace")
    assert "missing.txt" not in targeted
    assert "facts/value.txt" in targeted and "Final:" in targeted
    assert "missing.txt" in context and "FileNotFoundError" in context
    for output in manifest["outputs"].values():
        assert sha256(authority / __import__("pathlib").Path(output["path"]).name) == output["sha256"]


def test_record_json_hash_is_stable():
    record = valid_record()
    assert sha256_json(record) == sha256_json(json.loads(json.dumps(record)))
