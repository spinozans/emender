"""Read-only integrity inspection for non-trainable E97 pack authorities.

This module deliberately has no Dataset, mmap, NumPy, or Torch dependency.  It
can attest immutable descriptor bytes and accounting metadata for the CPU
system gate but cannot yield records, samples, tensors, or training masks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

# Keep this module independent from ``masked_sft_dataset``: importing the
# training consumer must not become an accidental diagnostic materialization
# path.
AUTHORITY_SCHEMA = "emender-e97-tulu3-masked-sft-v1"
PACK_SCHEMA = "emender-e97-sft-complete-record-packs-v1"
BOUNDARY_PACK_SCHEMA = "emender-e97-sft-boundary-aware-packs-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(root: Path, descriptor: Any, name: str) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} descriptor is invalid")
    relative = descriptor["path"]
    candidate = Path(relative) if isinstance(relative, str) else None
    if (candidate is None or candidate.is_absolute() or len(candidate.parts) != 1
            or candidate.name != relative or relative in {"", ".", ".."}):
        raise ValueError(f"{name} descriptor path is not publication-relative")
    path = root / candidate
    if not path.is_file() or path.stat().st_size != descriptor["bytes"] or sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{name} descriptor integrity mismatch")
    return {"path": relative, "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]}


def inspect_nontrainable_pack(
    authority_root: Path | str,
    pack_root: Path | str,
    *,
    authority_manifest_sha256: str,
    pack_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify mechanical pack bytes and return only non-materializing metadata."""

    authority_root, pack_root = Path(authority_root), Path(pack_root)
    authority_path, pack_path = authority_root / "manifest.json", pack_root / "manifest.json"
    if sha256(authority_path) != authority_manifest_sha256 or sha256(pack_path) != pack_manifest_sha256:
        raise ValueError("pack inspector manifest digest mismatch")
    authority, packs = json.loads(authority_path.read_text()), json.loads(pack_path.read_text())
    if authority.get("schema") != AUTHORITY_SCHEMA or authority.get("training_eligible") is not False:
        raise ValueError("pack inspector accepts only explicitly non-trainable authorities")
    if (packs.get("schema") not in {PACK_SCHEMA, BOUNDARY_PACK_SCHEMA}
            or packs.get("training_eligible") is not False
            or packs.get("authority_manifest_sha256") != authority_manifest_sha256):
        raise ValueError("pack inspector authority binding is invalid")
    authority_outputs, pack_outputs = authority.get("outputs"), packs.get("outputs")
    if not isinstance(authority_outputs, Mapping) or set(authority_outputs) != {"tokens", "mask", "index", "metadata"}:
        raise ValueError("pack inspector authority outputs are invalid")
    if not isinstance(pack_outputs, Mapping) or set(pack_outputs) != {"pack_records", "train_index", "validation_index"}:
        raise ValueError("pack inspector pack outputs are invalid")
    inspected = {
        f"authority_{name}": _payload(authority_root, descriptor, f"authority {name}")
        for name, descriptor in authority_outputs.items()
    }
    inspected.update({
        f"pack_{name}": _payload(pack_root, descriptor, f"pack {name}")
        for name, descriptor in pack_outputs.items()
    })
    splits = packs.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation"}:
        raise ValueError("pack inspector split accounting is invalid")
    return {
        "schema": "emender-e97-read-only-pack-inspection-v1",
        "training_eligible": False,
        "authority_manifest_sha256": authority_manifest_sha256,
        "pack_manifest_sha256": pack_manifest_sha256,
        "pack_schema": packs["schema"],
        "splits": {name: dict(value) for name, value in splits.items()},
        "payloads": inspected,
    }
