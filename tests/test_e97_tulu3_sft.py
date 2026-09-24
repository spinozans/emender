import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import tiktoken

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity, sha256
from scripts import build_e97_tulu3_sft as builder


def setup_module():
    builder._WORKER_ENCODING = tiktoken.get_encoding(builder.TOKENIZER)


def test_tulu_record_serialization_and_assistant_mask_are_exact():
    row = {
        "id": "example-1",
        "source": "fixture",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "Spell it."},
            {"role": "assistant", "content": "F-O-U-R"},
        ],
    }
    result = builder.serialize_row(row)
    assert "error" not in result
    encoding = builder._WORKER_ENCODING
    tokens = list(builder.struct.unpack(
        f"<{result['tokens']}I", result["token_bytes"]))
    assert encoding.decode(tokens) == (
        "System:\nBe concise.\n\nUser:\nWhat is two plus two?\n\n"
        "Assistant:\nFour.\n\nUser:\nSpell it.\n\nAssistant:\nF-O-U-R\x1e")
    assert len(result["mask_bytes"]) == len(tokens)
    targeted = b"".join(
        encoding.decode_single_token_bytes(token)
        for token, mask in zip(tokens, result["mask_bytes"]) if mask)
    assert targeted == b"Four.F-O-U-R\x1e"
    assert result["targets"] == sum(result["mask_bytes"])


def test_tulu_record_trims_role_boundary_whitespace_before_masking():
    result = builder.serialize_row({
        "id": "whitespace", "source": "fixture", "messages": [
            {"role": "user", "content": "  question  "},
            {"role": "assistant", "content": "\n answer \n"},
        ]})
    assert "error" not in result
    tokens = list(builder.struct.unpack(
        f"<{result['tokens']}I", result["token_bytes"]))
    assert builder._WORKER_ENCODING.decode(tokens) == (
        "User:\nquestion\n\nAssistant:\nanswer\x1e")
    assert result["boundary_whitespace_trimmed_characters"] == 8


def test_tulu_record_rejects_nonassistant_final_message_and_unknown_role():
    base = {"id": "bad", "source": "fixture"}
    result = builder.serialize_row({
        **base, "messages": [{"role": "user", "content": "unfinished"}]})
    assert result["error"] == "final_message_is_not_assistant"
    result = builder.serialize_row({
        **base, "messages": [
            {"role": "user", "content": "call"},
            {"role": "tool", "content": "result"},
        ]})
    assert result["error"] == "unsupported_role:tool"


def test_tulu_publication_descriptors_relocate_into_packs(tmp_path):
    """The Tulu producer emits root-relative payload descriptors for pack users."""

    root = tmp_path / "tulu-authority"
    root.mkdir()
    row = {
        "id": "relocation-fixture", "source": "fixture",
        "messages": [{"role": "user", "content": "Question"},
                     {"role": "assistant", "content": "Answer"}],
    }
    result = builder.serialize_row(row)
    assert "error" not in result
    # Pick a deterministic train-side identity so Dataset materialization has a pack.
    for suffix in range(100):
        candidate = f"relocation-fixture-{suffix}"
        if builder._split("fixture", candidate) == 0:
            row["id"] = candidate
            result = builder.serialize_row(row)
            break
    assert result["split"] == 0
    outputs = {
        "tokens": root / "tokens.uint32.bin", "mask": root / "assistant_mask.uint8.bin",
        "index": root / "records.idx", "metadata": root / "records.jsonl",
    }
    outputs["tokens"].write_bytes(result["token_bytes"])
    outputs["mask"].write_bytes(result["mask_bytes"])
    outputs["index"].write_bytes(builder.INDEX.pack(0, result["tokens"], result["targets"], 0))
    outputs["metadata"].write_text(json.dumps({"id": row["id"], "source": "fixture"}) + "\n")
    manifest = {
        "schema": builder.SCHEMA, "status": "complete", "training_eligible": True,
        "outputs": {name: builder.output_entry(path) for name, path in outputs.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    relocated = tmp_path / "tulu-relocated"
    root.rename(relocated)
    authority_sha = sha256(relocated / "manifest.json")
    packs = tmp_path / "tulu-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py", "--authority-root", str(relocated),
        "--output-root", str(packs), "--context-size", "128",
        "--authority-manifest-sha256", authority_sha,
    ], check=True, capture_output=True, text=True)
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=1,
        data_world_size=1, context_size=128,
    )
    dataset = MaskedSFTPackedDataset(relocated, packs, identity=identity, rank=0)
    try:
        assert dataset.get_batch(1)[0].shape == (1, 129)
    finally:
        dataset.close()


@pytest.mark.parametrize("training_eligible", (False, None))
def test_tulu_validator_rejects_false_or_missing_training_eligibility_subprocess(tmp_path, training_eligible):
    root = tmp_path / "authority"
    root.mkdir()
    manifest = {"schema": builder.SCHEMA, "status": "complete", "outputs": {}, "counts": {}}
    if training_eligible is not None:
        manifest["training_eligible"] = training_eligible
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, "scripts/validate_e97_tulu3_sft.py", "--root", str(root)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "explicitly training-eligible" in result.stderr


def test_tulu_validation_split_is_stable_and_bound_to_source_and_id():
    observed = builder._split("source-a", "record-1")
    assert observed == builder._split("source-a", "record-1")
    digest = hashlib.sha256(
        f"{builder.SCHEMA}\0source-a\0record-1".encode()).digest()
    assert observed == (1 if int.from_bytes(digest[:8], "little") % 100 == 0 else 0)
    values = {builder._split("source-a", f"record-{index}") for index in range(1000)}
    assert values == {0, 1}
