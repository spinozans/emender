import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

from ndm.data.e97_pack_inspector import inspect_nontrainable_pack
from ndm.data.masked_sft_dataset import (
    AUTHORITY_SCHEMA, RECORD_INDEX, MaskedSFTPackedDataset, SFTSamplerIdentity,
    restore_sft_checkpoint_metadata, sft_checkpoint_metadata, sha256,
)


def _authority(root: Path):
    root.mkdir()
    records = [
        ([10, 11, 218], [0, 1, 1], 0),
        ([20, 21], [0, 1], 0),
        ([30, 31, 32, 33, 34, 35], [0, 0, 1, 1, 1, 1], 0),
        ([40, 218], [0, 1], 1),
    ]
    token_path = root / "tokens.uint32.bin"
    mask_path = root / "assistant_mask.uint8.bin"
    index_path = root / "records.idx"
    metadata_path = root / "records.jsonl"
    offset = 0
    with token_path.open("wb") as tokens, mask_path.open("wb") as masks, index_path.open("wb") as index:
        for token_values, mask_values, split in records:
            tokens.write(struct.pack(f"<{len(token_values)}I", *token_values))
            masks.write(bytes(mask_values))
            index.write(RECORD_INDEX.pack(offset, len(token_values), sum(mask_values), split))
            offset += len(token_values)
    metadata_path.write_text("\n".join(json.dumps({"source": source}) for source in
                                        ("keep", "drop", "drop", "keep")) + "\n")
    outputs = {}
    for name, path in (("tokens", token_path), ("mask", mask_path),
                       ("index", index_path), ("metadata", metadata_path)):
        outputs[name] = {"path": path.name, "bytes": path.stat().st_size,
                         "sha256": sha256(path)}
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "training_eligible": True, "outputs": outputs,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return sha256(root / "manifest.json")


def _fixture(tmp_path: Path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "4", "--authority-manifest-sha256", authority_sha,
    ], check=True, capture_output=True, text=True)
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=42,
        data_world_size=2, context_size=4)
    return authority, packs, identity


def test_complete_record_packing_and_counter_sampling(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    manifest = json.loads((packs / "manifest.json").read_text())
    assert manifest["splits"]["train"] == {
        "records": 2, "tokens": 5, "assistant_target_tokens": 3, "packs": 1,
        "excluded_oversize_records": 1, "excluded_oversize_tokens": 6,
        "excluded_oversize_assistant_target_tokens": 4,
    }
    dataset = MaskedSFTPackedDataset(
        authority, packs, identity=identity, rank=1,
        initial_absolute_rank_sample_index=7, verify_payload_hashes=True)
    tokens, masks, lengths, targets = dataset.get_batch(2)
    assert tokens.tolist() == [[10, 11, 218, 20, 21], [10, 11, 218, 20, 21]]
    assert masks.tolist() == [[False, True, True, False, True]] * 2
    assert lengths.tolist() == [5, 5]
    assert targets.tolist() == [3, 3]
    assert dataset.next_absolute_rank_sample_index == 9
    assert len(dataset.last_batch_sample_ids) == 2

    reset_dataset = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=1)
    reset_batch = reset_dataset.get_batch_with_record_spans(1)
    assert reset_batch[4] == (((0, 3), (3, 5)),)
    assert reset_dataset.next_absolute_rank_sample_index == 1
    with pytest.raises(ValueError, match="batch_size=1"):
        reset_dataset.get_batch_with_record_spans(2)


def test_dataset_unconditionally_rejects_mechanical_authority_and_inspector_cannot_materialize(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    authority_path = authority / "manifest.json"
    authority_manifest = json.loads(authority_path.read_text())
    authority_manifest["training_eligible"] = False
    authority_path.write_text(json.dumps(authority_manifest, sort_keys=True) + "\n")
    authority_sha = sha256(authority_path)
    pack_path = packs / "manifest.json"
    pack_manifest = json.loads(pack_path.read_text())
    pack_manifest.update({"authority_manifest_sha256": authority_sha, "training_eligible": False})
    pack_path.write_text(json.dumps(pack_manifest, sort_keys=True) + "\n")
    pack_sha = sha256(pack_path)
    mechanical_identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha, pack_manifest_sha256=pack_sha,
        sampler_key=identity.sampler_key, data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    with pytest.raises(RuntimeError, match="non-trainable"):
        MaskedSFTPackedDataset(authority, packs, identity=mechanical_identity, rank=0)
    with pytest.raises(RuntimeError, match="not permitted"):
        MaskedSFTPackedDataset(
            authority, packs, identity=mechanical_identity, rank=0,
            diagnostic_cpu_system_gate=True,
        )
    inspection = inspect_nontrainable_pack(
        authority, packs, authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=pack_sha,
    )
    assert inspection["training_eligible"] is False
    assert "payloads" in inspection and not hasattr(inspection, "get_batch")


def test_nontrainable_authority_cannot_be_laundered_by_mix_or_rewrite_or_dataset(tmp_path):
    """Mechanical inputs stop at every retained transformer/training boundary."""

    authority = tmp_path / "mechanical"
    authority_sha = _authority(authority)
    manifest_path = authority / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_eligible"] = False
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    authority_sha = sha256(manifest_path)

    mixed = subprocess.run([
        sys.executable, "scripts/build_e97_masked_sft_mix.py", "--output-root", str(tmp_path / "mixed"),
        "--source", f"mechanical={authority},{authority_sha},1",
    ], capture_output=True, text=True)
    assert mixed.returncode != 0 and "non-trainable" in mixed.stderr
    repaired = subprocess.run([
        sys.executable, "scripts/build_e97_pi_finalization_repair_sft.py", "--source-root", str(authority),
        "--source-sha256", authority_sha, "--output-root", str(tmp_path / "repaired"),
    ], capture_output=True, text=True)
    assert repaired.returncode != 0 and "non-trainable" in repaired.stderr
    rewritten = subprocess.run([
        sys.executable, "scripts/rewrite_e97_sft_system_prompt.py", "--input-root", str(authority),
        "--input-manifest-sha256", authority_sha, "--output-root", str(tmp_path / "rewritten"),
    ], capture_output=True, text=True)
    assert rewritten.returncode != 0 and "non-trainable" in rewritten.stderr

    packs = tmp_path / "mechanical-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py", "--authority-root", str(authority),
        "--output-root", str(packs), "--context-size", "4",
        "--authority-manifest-sha256", authority_sha, "--diagnostic-cpu-system-gate",
    ], check=True, capture_output=True, text=True)
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=7,
        data_world_size=1, context_size=4,
    )
    with pytest.raises(RuntimeError, match="non-trainable"):
        MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0)


def test_deleted_training_eligibility_fails_closed_at_every_training_boundary(tmp_path):
    """Deleting false/true authority state cannot upgrade a retained authority."""

    authority, packs, identity = _fixture(tmp_path)
    manifest_path = authority / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # An attacker can remove this prior mechanical false state rather than
    # carrying it through a transformer; absence must not become trainable.
    manifest["training_eligible"] = False
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest = json.loads(manifest_path.read_text())
    del manifest["training_eligible"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    authority_sha = sha256(manifest_path)

    mixed = subprocess.run([
        sys.executable, "scripts/build_e97_masked_sft_mix.py", "--output-root", str(tmp_path / "mixed"),
        "--source", f"missing={authority},{authority_sha},1",
    ], capture_output=True, text=True)
    assert mixed.returncode != 0 and "eligibility" in mixed.stderr
    repaired = subprocess.run([
        sys.executable, "scripts/build_e97_pi_finalization_repair_sft.py", "--source-root", str(authority),
        "--source-sha256", authority_sha, "--output-root", str(tmp_path / "repaired"),
    ], capture_output=True, text=True)
    assert repaired.returncode != 0 and "eligibility" in repaired.stderr
    rewritten = subprocess.run([
        sys.executable, "scripts/rewrite_e97_sft_system_prompt.py", "--input-root", str(authority),
        "--input-manifest-sha256", authority_sha, "--output-root", str(tmp_path / "rewritten"),
    ], capture_output=True, text=True)
    assert rewritten.returncode != 0 and "eligibility" in rewritten.stderr
    packed = subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py", "--authority-root", str(authority),
        "--output-root", str(tmp_path / "missing-packs"), "--context-size", "4",
        "--authority-manifest-sha256", authority_sha,
    ], capture_output=True, text=True)
    assert packed.returncode != 0 and "explicit boolean" in packed.stderr

    # Rebinding an existing pack cannot launder the deleted authority field.
    pack_path = packs / "manifest.json"
    pack_manifest = json.loads(pack_path.read_text())
    pack_manifest["authority_manifest_sha256"] = authority_sha
    pack_path.write_text(json.dumps(pack_manifest, sort_keys=True) + "\n")
    pack_sha = sha256(pack_path)
    validated = subprocess.run([
        sys.executable, "scripts/validate_e97_sft_packs.py",
        "--authority-root", str(authority), "--pack-root", str(packs),
        "--authority-manifest-sha256", authority_sha, "--pack-manifest-sha256", pack_sha,
    ], capture_output=True, text=True)
    assert validated.returncode != 0 and "explicit" in validated.stderr
    deleted_identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha, pack_manifest_sha256=pack_sha,
        sampler_key=identity.sampler_key, data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    with pytest.raises(RuntimeError, match="explicit boolean"):
        MaskedSFTPackedDataset(authority, packs, identity=deleted_identity, rank=0)
    with pytest.raises(ValueError, match="explicitly non-trainable"):
        inspect_nontrainable_pack(
            authority, packs, authority_manifest_sha256=authority_sha,
            pack_manifest_sha256=sha256(pack_path),
        )


def test_dataset_requires_complete_authority_and_pack_manifests(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    authority_path, pack_path = authority / "manifest.json", packs / "manifest.json"
    authority_manifest = json.loads(authority_path.read_text()); authority_manifest["status"] = "partial"
    authority_path.write_text(json.dumps(authority_manifest, sort_keys=True) + "\n")
    authority_sha = sha256(authority_path)
    pack_manifest = json.loads(pack_path.read_text())
    pack_manifest.update({"authority_manifest_sha256": authority_sha, "status": "partial"})
    pack_path.write_text(json.dumps(pack_manifest, sort_keys=True) + "\n")
    incomplete = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha, pack_manifest_sha256=sha256(pack_path),
        sampler_key=identity.sampler_key, data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    with pytest.raises(RuntimeError, match="complete"):
        MaskedSFTPackedDataset(authority, packs, identity=incomplete, rank=0)
    validated = subprocess.run([
        sys.executable, "scripts/validate_e97_sft_packs.py",
        "--authority-root", str(authority), "--pack-root", str(packs),
        "--authority-manifest-sha256", authority_sha,
        "--pack-manifest-sha256", incomplete.pack_manifest_sha256,
    ], capture_output=True, text=True)
    assert validated.returncode != 0 and "authority manifest is not complete" in validated.stderr


def test_dataset_manifest_snapshot_prevents_path_substitution_of_nontrainable_state(tmp_path, monkeypatch):
    import ndm.data.masked_sft_dataset as dataset_module

    authority, packs, identity = _fixture(tmp_path)
    authority_path = authority / "manifest.json"
    authority_manifest = json.loads(authority_path.read_text())
    authority_manifest["training_eligible"] = False
    authority_path.write_text(json.dumps(authority_manifest, sort_keys=True) + "\n")
    authority_sha = sha256(authority_path)
    pack_path = packs / "manifest.json"
    pack_manifest = json.loads(pack_path.read_text())
    pack_manifest.update({"authority_manifest_sha256": authority_sha, "training_eligible": False})
    pack_path.write_text(json.dumps(pack_manifest, sort_keys=True) + "\n")
    sealed = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha, pack_manifest_sha256=sha256(pack_path),
        sampler_key=identity.sampler_key, data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    reader = dataset_module.read_regular_file_no_follow

    def replace_after_snapshot(path, *, maximum):
        payload = reader(path, maximum=maximum)
        if Path(path) == authority_path:
            replacement = json.loads(payload)
            replacement["training_eligible"] = True
            authority_path.write_text(json.dumps(replacement, sort_keys=True) + "\n")
        return payload

    monkeypatch.setattr(dataset_module, "read_regular_file_no_follow", replace_after_snapshot)
    with pytest.raises(RuntimeError, match="non-trainable"):
        MaskedSFTPackedDataset(authority, packs, identity=sealed, rank=0)
    assert json.loads(authority_path.read_text())["training_eligible"] is True


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_manifest_snapshot_rejects_linked_and_special_manifest_inputs(tmp_path, kind):
    from ndm.data.masked_sft_dataset import snapshot_manifest

    manifest = tmp_path / "manifest.json"
    if kind == "symlink":
        outside = tmp_path / "outside.json"; outside.write_text("{}")
        manifest.symlink_to(outside)
    else:
        os.mkfifo(manifest)
    with pytest.raises(RuntimeError, match="cannot be snapshotted safely"):
        snapshot_manifest(manifest, "0" * 64, name="authority")


def test_dataset_rejects_stale_absolute_manifest_payload_paths(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    manifest_path = authority / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["tokens"]["path"] = "/vanished/.stage/tokens.uint32.bin"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    authority_sha = sha256(manifest_path)
    pack_path = packs / "manifest.json"; pack_manifest = json.loads(pack_path.read_text())
    pack_manifest["authority_manifest_sha256"] = authority_sha
    pack_path.write_text(json.dumps(pack_manifest, sort_keys=True) + "\n")
    bad_identity = SFTSamplerIdentity(authority_manifest_sha256=authority_sha,
                                      pack_manifest_sha256=sha256(pack_path),
                                      sampler_key=identity.sampler_key, data_world_size=identity.data_world_size,
                                      context_size=identity.context_size)
    with pytest.raises(RuntimeError, match="publication-relative"):
        MaskedSFTPackedDataset(authority, packs, identity=bad_identity, rank=0)


def test_immutable_token_payload_is_verified_by_builder_validator_and_dataset(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    token_path = authority / "tokens.uint32.bin"
    payload = bytearray(token_path.read_bytes()); payload[0] ^= 1; token_path.write_bytes(payload)
    with pytest.raises(ValueError, match="cannot be disabled"):
        MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0, verify_payload_hashes=False)
    with pytest.raises(RuntimeError, match="payload digest"):
        MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0)
    failed = subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py", "--authority-root", str(authority),
        "--output-root", str(tmp_path / "tampered-packs"), "--context-size", "4",
        "--authority-manifest-sha256", identity.authority_manifest_sha256,
    ], capture_output=True, text=True)
    assert failed.returncode != 0 and "token payload integrity mismatch" in failed.stderr


def test_boundary_aware_pack_materializes_resets_validity_and_loss(tmp_path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "boundary-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "7", "--authority-manifest-sha256", authority_sha,
        "--boundary-aware",
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((packs / "manifest.json").read_text())
    assert manifest["schema"] == "emender-e97-sft-boundary-aware-packs-v2"
    assert manifest["boundary_semantics"]["cross_document_target"] == "masked"
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=42,
        data_world_size=1, context_size=7)
    dataset = MaskedSFTPackedDataset(
        authority, packs, identity=identity, rank=0, verify_payload_hashes=True)
    # The only train pack that combines records 0 and 1 is pack zero. Record 2
    # is retained separately because it does not fit the eight-token sequence.
    tokens, loss, valid, reset, length, targets, name = dataset.pack_at_with_boundaries(0)
    assert tokens.tolist() == [10, 11, 218, 20, 21, 0, 0, 0]
    assert valid.tolist() == [True, True, True, True, True, False, False, False]
    assert reset.tolist() == [True, False, False, True, False, False, False, False]
    assert loss.tolist() == [True, True, False, True, False, False, False]
    assert (length, targets, name) == (5, 3, "pack-00000000")

    batch = dataset.get_boundary_aware_batch(1)
    assert [tuple(value.shape) for value in batch] == [
        (1, 8), (1, 7), (1, 8), (1, 8), (1,), (1,)]
    subprocess.run([
        sys.executable, "scripts/validate_e97_sft_packs.py",
        "--authority-root", str(authority), "--pack-root", str(packs),
        "--authority-manifest-sha256", authority_sha,
        "--pack-manifest-sha256", sha256(packs / "manifest.json"),
    ], check=True, capture_output=True, text=True)


def test_pack_builder_can_limit_records_per_pack(tmp_path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "single-record-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "4", "--authority-manifest-sha256", authority_sha,
        "--max-records-per-pack", "1",
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((packs / "manifest.json").read_text())
    assert manifest["max_records_per_pack"] == 1
    assert manifest["splits"]["train"]["records"] == 2
    assert manifest["splits"]["train"]["packs"] == 2
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=42,
        data_world_size=2, context_size=4)
    dataset = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0)
    assert dataset.get_batch_with_record_spans(1)[4] in (
        (((0, 2),),), (((0, 3),),))


def test_pack_builder_exact_source_filter(tmp_path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "filtered-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "4", "--authority-manifest-sha256", authority_sha,
        "--include-source", "keep",
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((packs / "manifest.json").read_text())
    assert manifest["source_filter"] == {"include_exact": ["keep"]}
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["splits"]["validation"]["records"] == 1
    subprocess.run([
        sys.executable, "scripts/validate_e97_sft_packs.py",
        "--authority-root", str(authority), "--pack-root", str(packs),
        "--authority-manifest-sha256", authority_sha,
        "--pack-manifest-sha256", sha256(packs / "manifest.json"),
    ], check=True, capture_output=True, text=True)
    assert json.loads((packs / "validation.json").read_text())["status"] == "pass"


def test_exact_pack_access_bypasses_replacement_sampler(tmp_path):
    authority, packs, train_identity = _fixture(tmp_path)
    validation_identity = SFTSamplerIdentity(
        authority_manifest_sha256=train_identity.authority_manifest_sha256,
        pack_manifest_sha256=train_identity.pack_manifest_sha256,
        sampler_key=train_identity.sampler_key,
        data_world_size=train_identity.data_world_size,
        context_size=train_identity.context_size,
        split="validation")
    dataset = MaskedSFTPackedDataset(
        authority, packs, identity=validation_identity, rank=0)
    token, mask, length, targets, pack_id = dataset.pack_at(0)
    assert token.tolist() == [40, 218, 0, 0, 0]
    assert mask.tolist() == [False, True, False, False, False]
    assert (length, targets, pack_id) == (2, 1, "pack-00000000")
    with pytest.raises(IndexError, match="out of range"):
        dataset.pack_at(1)


def test_epoch_permutation_sampler_covers_each_pack_once_across_ranks(tmp_path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "permutation-packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "2", "--authority-manifest-sha256", authority_sha,
        "--sampler-mode", "epoch-permutation",
    ], check=True, capture_output=True, text=True)
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=43,
        data_world_size=3, context_size=2)
    datasets = [MaskedSFTPackedDataset(
        authority, packs, identity=identity, rank=rank,
        sampler_mode="epoch-permutation") for rank in range(3)]
    pack_count = len(datasets[0].packs)
    assert pack_count == 2
    global_pack_ids = [
        datasets[global_index % 3].pack_id_at(global_index // 3)
        for global_index in range(pack_count * 4)
    ]
    for start in range(0, len(global_pack_ids), pack_count):
        assert sorted(global_pack_ids[start:start + pack_count]) == list(range(pack_count))
    with pytest.raises(ValueError, match="sampler mode"):
        MaskedSFTPackedDataset(
            authority, packs, identity=identity, rank=0, sampler_mode="unknown")


def test_sft_checkpoint_clocks_fail_closed(tmp_path):
    _authority_root, _packs, identity = _fixture(tmp_path)
    parent = {"manifest_sha256": "a" * 64, "step": 10,
              "accepted_tokens": 1000, "generation": "/immutable/parent"}
    metadata = sft_checkpoint_metadata(
        identity, parent=parent, total_tokens=50, assistant_target_tokens=30,
        absolute_rank_sample_index=4)
    assert restore_sft_checkpoint_metadata(
        metadata, expected_identity=identity, expected_parent=parent,
        model_accepted_tokens=1050) == (50, 30, 4)
    with pytest.raises(ValueError, match="accepted tokens"):
        restore_sft_checkpoint_metadata(
            metadata, expected_identity=identity, expected_parent=parent,
            model_accepted_tokens=1051)


def test_counter_samples_depend_on_rank_and_absolute_cursor(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    rank0 = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0)
    rank1 = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=1)
    assert rank0.sample_id(0) != rank1.sample_id(0)
    assert rank0.sample_id(0) != rank0.sample_id(1)
