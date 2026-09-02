import gzip
import json
from pathlib import Path

from scripts import build_e97_commapile_document_sft as builder


def test_commapile_document_overlap_filter_catches_identity_and_content(tmp_path):
    content = "\n".join(f"unique holdout implementation line {index} with enough detail" for index in range(7))
    signatures = builder._shingles(content)
    assert signatures
    assert builder._overlap_reason(
        "dependency: MarkupSafe", signatures, ("markupsafe",)) == (
            "repository_identity:markupsafe")
    assert builder._overlap_reason(content, signatures, ("markupsafe",)) == (
        "repository_content_shingle")
    assert builder._overlap_reason(
        "unrelated document", signatures, ("markupsafe",)) is None


def test_commapile_source_iteration_is_deterministic_and_document_complete(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for shard_index in range(2):
        with gzip.open(source / f"part-{shard_index}.jsonl.gz", "wt") as stream:
            stream.write(json.dumps({"text": f"document-{shard_index}-a"}) + "\n")
            stream.write(json.dumps({"text": f"document-{shard_index}-b"}) + "\n")
    first = builder._iter_source_rows(tmp_path, "source", 17)
    second = builder._iter_source_rows(tmp_path, "source", 17)
    first_rows = [next(first)[:3] for _ in range(6)]
    second_rows = [next(second)[:3] for _ in range(6)]
    assert first_rows == second_rows
    assert {row[0] for row in first_rows} == {0, 1}


def test_commapile_builder_is_revision_bound_and_forbids_github_archive():
    text = Path("scripts/build_e97_commapile_document_sft.py").read_text()
    assert builder.DATASET_REVISION == "5afc546db324e7f39f297ba757c9a60547151e7c"
    assert 'if "github_archive" in sources' in text
    assert "repository_content_shingle" in text
    assert "max_record_tokens" in text
    assert "causal targets after whole-document eligibility" in text
    assert "Git LFS payload mismatch" in text
