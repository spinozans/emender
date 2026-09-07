import hashlib
import json
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
    assert overlap.validate_overlap_receipt(
        receipt,
        candidate_collection_sha256="a" * 64,
        candidate_archive_root_sha256="b" * 64,
    ) == receipt
    receipt["protected_records_sha256s"][0] = "0" * 64
    with pytest.raises(overlap.OverlapError, match="fixed panel records"):
        overlap.validate_overlap_receipt(
            receipt,
            candidate_collection_sha256="a" * 64,
            candidate_archive_root_sha256="b" * 64,
        )
