import hashlib
import json
import os
from pathlib import Path
import re

import pytest

import ndm.e97_protected_overlap as overlap
from tests.e97_protected_overlap_support import write_synthetic_real_schema_panels


def _synthetic_fixed_maps(panels):
    manifests = {}
    records = {}
    for manifest, record in panels:
        schema = json.loads(manifest.read_text())["schema"]
        manifests[schema] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        records[schema] = hashlib.sha256(record.read_bytes()).hexdigest()
    return manifests, records


def test_real_schema_adapters_accept_only_test_monkeypatched_fixed_pairs(tmp_path, monkeypatch):
    panels = write_synthetic_real_schema_panels(tmp_path / "panels")
    manifests, records = _synthetic_fixed_maps(panels)
    # Synthetic adapter coverage changes module constants only inside this test;
    # no public checker/admission API accepts caller-supplied sealed hashes.
    monkeypatch.setattr(overlap, "_EXPECTED_MANIFEST_SHA256S", manifests)
    monkeypatch.setattr(overlap, "_EXPECTED_RECORD_SHA256S", records)
    loaded = [overlap.load_protected_panel(manifest, record) for manifest, record in panels]
    assert len(loaded) == 3
    assert all(panel_records for _, _, panel_records in loaded)

    bad = tmp_path / "bad-records.jsonl"
    bad.write_bytes(panels[0][1].read_bytes() + b"\n")
    with pytest.raises(overlap.OverlapError, match="records SHA-256"):
        overlap.load_protected_panel(panels[0][0], bad)


def test_candidate_archive_snapshot_is_descriptor_bound_when_path_is_substituted(tmp_path, monkeypatch):
    root = tmp_path / "candidate"; archive = root / "archives" / "one.tar"; archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first-authority-bytes")
    replacement = tmp_path / "replacement.tar"; replacement.write_bytes(b"substituted-authority-bytes")
    task = {"fixture": {"artifact_path": "archives/one.tar"}}
    original_read = overlap.os.read; swapped = False

    def read_then_substitute(fd, count):
        nonlocal swapped
        payload = original_read(fd, count)
        if payload and not swapped:
            swapped = True; archive.unlink(); replacement.replace(archive)
        return payload

    monkeypatch.setattr(overlap.os, "read", read_then_substitute)
    snapshots = overlap._snapshot_candidate_archives([task], root)
    assert snapshots == {"archives/one.tar": b"first-authority-bytes"}
    assert overlap._archive_root_from_snapshots([task], snapshots) == overlap.sha256_text(overlap.canonical_json([
        {"path": "archives/one.tar", "bytes": len(b"first-authority-bytes"),
         "sha256": hashlib.sha256(b"first-authority-bytes").hexdigest()},
    ]))
    assert archive.read_bytes() == b"substituted-authority-bytes"


def test_private_allowlist_snapshot_is_retained_and_production_root_remains_empty(tmp_path, monkeypatch):
    fields = sorted(overlap._domains([]))
    receipt = {
        "schema": overlap.OVERLAP_RECEIPT_SCHEMA, "status": "pass",
        "candidate_collection_sha256": "a" * 64, "candidate_archive_root_sha256": "b" * 64,
        "checker_source_sha256": hashlib.sha256(Path(overlap.__file__).read_bytes()).hexdigest(),
        "protected_manifest_sha256s": sorted(overlap._EXPECTED_MANIFEST_SHA256S.values()),
        "protected_records_sha256s": sorted(overlap._EXPECTED_RECORD_SHA256S.values()),
        "field_domain_counts": {field: {"candidate": 0, "protected": 0} for field in fields},
        "collision_counts": {field: 0 for field in fields},
    }
    receipt_payload = overlap.canonical_json(receipt).encode()
    entry = {
        "scope": "non-production-cpu-system-gate", "registry_sha256": "c" * 64,
        "generation_receipt_sha256": "d" * 64, "protected_overlap_receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "tasks_sha256": "a" * 64, "archive_root_sha256": "b" * 64,
        "generator_manifest_sha256": "e" * 64, "source_archive_sha256": "f" * 64,
        "source_revision": "1" * 40, "controller_source_sha256": "2" * 64,
    }
    private_root = tmp_path / "private-allowlist.json"
    retained_payload = overlap.canonical_json({
        "schema": overlap.COLLECTION_AUTHORIZATION_SCHEMA, "status": "authorized", "authorizations": [entry],
    }).encode()
    private_root.write_bytes(retained_payload)
    production_before = overlap.COLLECTION_AUTHORIZATION_ALLOWLIST.read_bytes()
    monkeypatch.setattr(overlap, "COLLECTION_AUTHORIZATION_ALLOWLIST", private_root)
    authorized = overlap.validate_authorized_overlap_receipt(
        receipt, receipt_sha256=entry["protected_overlap_receipt_sha256"], registry_sha256="c" * 64,
        generation_receipt_sha256="d" * 64, tasks_sha256="a" * 64, archive_root_sha256="b" * 64,
        generator_manifest_sha256="e" * 64, source_archive_sha256="f" * 64,
        source_revision="1" * 40, controller_source_sha256="2" * 64, diagnostic_cpu_system_gate=True,
    )
    private_root.write_text("{}")
    assert authorized["allowlist_payload"] == retained_payload
    assert (Path("configs/pi/e97-firstparty-collection-authorizations-v1.json").read_bytes()
            == production_before)


def test_candidate_archive_snapshot_rejects_fifo_without_blocking(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    fifo = root / "archive.tar"
    os.mkfifo(fifo)
    with pytest.raises(overlap.OverlapError, match="bounded regular"):
        overlap._snapshot_candidate_archives(
            [{"fixture": {"artifact_path": "archive.tar"}}], root,
        )


def test_overlap_snapshot_reader_rejects_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "overlap.json"
    os.mkfifo(fifo)
    with pytest.raises(overlap.OverlapError, match="cannot be opened safely"):
        overlap._read_path_once(fifo, name="overlap input")


def test_candidate_root_rejects_a_symlinked_parent_component(tmp_path):
    real_parent = tmp_path / "real"; root = real_parent / "candidate"; root.mkdir(parents=True)
    (root / "archive.tar").write_bytes(b"candidate")
    link = tmp_path / "linked"; link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(overlap.OverlapError, match="candidate root cannot be opened safely"):
        overlap._snapshot_candidate_archives(
            [{"fixture": {"artifact_path": "archive.tar"}}], link / "candidate",
        )


def test_normalizers_replace_quoted_paths_and_values_without_prose_scalars():
    protected = "Read `document-01234567/token-99.txt` and answer token 12345."
    candidate = re.sub(r"`[^`]+`", "`other/path-999999.txt`", protected)
    assert overlap.normalize_template(candidate) == overlap.normalize_template(protected)
    assert "token" in overlap.normalize_template(protected)
    assert overlap.extract_exact_scalars(["ordinary prose fourword token"]) == set()
    assert overlap.extract_exact_scalars(['{"answer":"opal-731","n":7}', "target=value"]) == {
        "opal-731", "7", "value",
    }


def test_overlap_receipt_requires_the_fixed_manifest_and_record_pairs():
    fields = sorted(overlap._domains([]))
    receipt = {
        "schema": overlap.OVERLAP_RECEIPT_SCHEMA,
        "status": "pass",
        "candidate_collection_sha256": "a" * 64,
        "candidate_archive_root_sha256": "b" * 64,
        "checker_source_sha256": hashlib.sha256(overlap.__file__.encode()).hexdigest(),
        "protected_manifest_sha256s": sorted(overlap._EXPECTED_MANIFEST_SHA256S.values()),
        "protected_records_sha256s": sorted(overlap._EXPECTED_RECORD_SHA256S.values()),
        "field_domain_counts": {field: {"candidate": 0, "protected": 0} for field in fields},
        "collision_counts": {field: 0 for field in fields},
    }
    # Bind the current checker file after constructing a receipt without panel content.
    receipt["checker_source_sha256"] = hashlib.sha256(open(overlap.__file__, "rb").read()).hexdigest()
    # Structural parsing is intentionally not admission authority.  The
    # canonical checked-in allowlist is empty/candidate in this repair, so the
    # fully well-shaped forged pass must fail the production authorization step.
    assert overlap.validate_overlap_receipt(
        receipt,
        candidate_collection_sha256="a" * 64,
        candidate_archive_root_sha256="b" * 64,
    ) == receipt
    with pytest.raises(overlap.OverlapError, match="not operator-authorized"):
        overlap.validate_authorized_overlap_receipt(
            receipt, receipt_sha256=hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            registry_sha256="c" * 64, generation_receipt_sha256="d" * 64,
            tasks_sha256="a" * 64, archive_root_sha256="b" * 64,
            generator_manifest_sha256="e" * 64, source_archive_sha256="f" * 64,
            source_revision="1" * 40, controller_source_sha256="2" * 64,
        )
    receipt["protected_records_sha256s"][0] = "0" * 64
    with pytest.raises(overlap.OverlapError, match="fixed panel records"):
        overlap.validate_overlap_receipt(
            receipt,
            candidate_collection_sha256="a" * 64,
            candidate_archive_root_sha256="b" * 64,
        )
