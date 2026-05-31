#!/usr/bin/env python3
"""Prepare local v0.3 release-candidate HF artifacts (matching the V3 paper).

Local-only helper for `v3-validate-current`. It selects the retained racer
checkpoints nearest the **committed V3 paper** endpoint steps (NOT the latest
training step), records checkpoint/stat/hash/metric evidence, converts
`model_state_dict` to `model.safetensors`, writes the custom Transformers loader
artifacts, and validates local
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` loading.

It performs NO Hugging Face upload / repo / branch / tag / visibility calls and
writes only under the requested workdir (default `/tmp/...`). Raw `.pt`
checkpoints are read in place and never copied into git.

Checkpoint selection comes from `scripts/select_v03_racer_checkpoints.py`
(canonical smooth.py trail_100k against the committed paper AS_OF steps). The
selected steps are pinned here so conversion is deterministic and auditable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hf_private_staging_upload import (  # noqa: E402
    CONFIGURATION_NDM,
    MODELING_NDM,
    ModelSpec,
    build_config,
    load_args,
    write_json,
    write_tokenizer_files,
)

# Pinned tokenizer constant (authoritative; matches V3_NUMBERS.md / AS_OF.md).
_TOK = json.loads((ROOT / "scripts/estimate_tokenizer_bytes_per_token.json").read_text())
BPB_PER_NAT = _TOK["bits_per_byte_per_nat_per_token"]  # 0.3681635882200934
DEFAULT_WORKDIR = Path(f"/tmp/release-v03-local-hf-candidates-{os.environ.get('USER', 'agent')}")
RELEASE_CANDIDATE = "v0.3-rc-local"


@dataclass(frozen=True)
class CandidateSpec:
    spec: ModelSpec
    paper_endpoint_step: int
    paper_endpoint_bpb: float
    ticket_as_of_step: int
    selection_note: str


# Selected checkpoints = retained .pt nearest the committed V3 paper endpoint
# (paper/main.typ §5, paper/results/figure_2/AS_OF.md, 2026-05-31T13:49:33Z).
# Verified by scripts/select_v03_racer_checkpoints.py.
SPECS: dict[str, CandidateSpec] = {
    "e88": CandidateSpec(
        spec=ModelSpec(
            key="e88",
            repo_id="poietic-pbc/emender-e88-1.3b",
            identity="Emender/E88",
            short_name="Emender E88 1.3B",
            checkpoint=Path(
                "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
                "levelE88_1270M_20260511_233832/checkpoint_step_1524000_loss_2.6143.pt"
            ),
            smoke_label="e88",
            smoke_new_token_ids=[218, 218],
            smoke_new_text_repr="'\\x1e\\x1e'",
            smoke_param_count=1_273_191_856,
        ),
        paper_endpoint_step=1_523_250,
        paper_endpoint_bpb=0.973765,
        ticket_as_of_step=1_405_450,
        selection_note=(
            "Retained checkpoint nearest the committed V3 paper E88 endpoint "
            "(step 1,523,250, 0.974 bpb); selected step 1,524,000 (Δ+750). "
            "trail_100k bpb at selected step 0.973780 -> rounds to 0.974, "
            "matching the paper. Ticket AS_OF 1,405,450 was overwritten."
        ),
    ),
    "gdn": CandidateSpec(
        spec=ModelSpec(
            key="gdn",
            repo_id="poietic-pbc/gdn-1.3b",
            identity="GDN",
            short_name="GDN 1.3B",
            checkpoint=Path(
                "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/"
                "levelfla-gdn_1270M_20260511_233832/checkpoint_step_1998000_loss_2.6148.pt"
            ),
            smoke_label="gdn",
            smoke_new_token_ids=[318, 318],
            smoke_new_text_repr="' is is'",
            smoke_param_count=1_352_352_498,
        ),
        paper_endpoint_step=1_999_300,
        paper_endpoint_bpb=0.976965,
        ticket_as_of_step=1_847_050,
        selection_note=(
            "Retained checkpoint nearest the committed V3 paper GDN endpoint "
            "(step 1,999,300, 0.977 bpb); selected step 1,998,000 (Δ-1,300). "
            "trail_100k bpb at selected step 0.977067 -> rounds to 0.977, "
            "matching the paper. Ticket AS_OF 1,847,050 was overwritten."
        ),
    ),
    "m2rnn": CandidateSpec(
        spec=ModelSpec(
            key="m2rnn",
            repo_id="poietic-pbc/m2rnn-cma-1.3b",
            identity="M2RNN-CMA",
            short_name="M2RNN-CMA 1.3B",
            checkpoint=Path(
                "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/"
                "levelm2rnn_1270M_20260511_175023/checkpoint_step_1467000_loss_2.6277.pt"
            ),
            smoke_label="m2rnn",
            smoke_new_token_ids=[2109, 34059],
            smoke_new_text_repr="'........Officers'",
            smoke_param_count=1_307_101_140,
        ),
        paper_endpoint_step=1_466_400,
        paper_endpoint_bpb=0.979845,
        ticket_as_of_step=1_343_050,
        selection_note=(
            "Retained checkpoint nearest the committed V3 paper M2RNN-CMA "
            "endpoint (step 1,466,400, 0.980 bpb); selected step 1,467,000 "
            "(Δ+600). trail_100k bpb at selected step 0.979832 -> rounds to "
            "0.980, matching the paper. Ticket AS_OF 1,343,050 was overwritten."
        ),
    ),
}

CHECKPOINT_RE = re.compile(r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>\d+(?:\.\d+)?)\.pt$")


def parse_checkpoint_name(path: Path) -> dict[str, Any]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        raise ValueError(f"checkpoint filename does not match expected pattern: {path}")
    return {"step": int(match.group("step")), "loss": float(match.group("loss"))}


def utc_mtime(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(128 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retained_steps(candidate: CandidateSpec) -> list[int]:
    out = []
    for ckpt in sorted(candidate.spec.checkpoint.parent.glob("checkpoint_step_*_loss_*.pt")):
        m = CHECKPOINT_RE.match(ckpt.name)
        if m:
            out.append(int(m.group("step")))
    return out


def write_local_model_card(out_dir: Path, spec: ModelSpec, checkpoint: dict[str, Any], weights: dict[str, Any]) -> None:
    text = f"""\
---
library_name: transformers
pipeline_tag: text-generation
tags:
- ndm
- emender
- local-validation
- v0.3-rc
---

# {spec.short_name} v0.3 local release candidate

Generated for local v0.3 release-candidate validation only. NOT a public
Hugging Face upload; does not modify any public tag; must not be published
without a separate human approval gate. The source checkpoint matches the
**committed V3 paper** endpoint (not the latest training step).

## Source Checkpoint

- Model identity: {spec.identity}
- Intended public repository after approval: `{spec.repo_id}`
- Local release-candidate marker: `{RELEASE_CANDIDATE}`
- Source checkpoint: `{checkpoint["path"]}`
- Source checkpoint SHA256: `{checkpoint["sha256"]}`
- Checkpoint step: `{checkpoint["step"]}`
- Raw checkpoint loss: `{checkpoint["raw_checkpoint_loss"]}`
- Converted safetensors size: `{weights["size"]}` bytes
- Safetensors keys: `{weights["keys"]}`

## Loading Notes

Use `trust_remote_code=True` with the matching `ndm` source package installed.
"""
    out_dir.joinpath("README.md").write_text(text)


def convert_weights_local(spec: ModelSpec, out_dir: Path, force: bool) -> dict[str, Any]:
    safetensors_path = out_dir / "model.safetensors"
    if safetensors_path.exists() and safetensors_path.stat().st_size > 0 and not force:
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
        return {
            "path": str(safetensors_path),
            "size": safetensors_path.stat().st_size,
            "keys": len(keys),
            "metadata": metadata,
            "skipped_existing": True,
        }

    print(f"[{spec.key}] converting checkpoint to safetensors: {spec.checkpoint}", flush=True)
    ckpt = torch.load(str(spec.checkpoint), map_location="cpu", mmap=True, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    tensors: dict[str, torch.Tensor] = {}
    seen_storage: dict[tuple[int, int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        prefixed = f"model.{name}"
        value = tensor.detach().cpu().contiguous()
        storage_key = (
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            value.numel(),
            tuple(value.size()),
            tuple(value.stride()),
        )
        if storage_key in seen_storage:
            value = value.clone()
        else:
            seen_storage[storage_key] = prefixed
        tensors[prefixed] = value

    metadata = {
        "format": "pt",
        "local_validation_only": "true",
        "release_candidate": RELEASE_CANDIDATE,
        "repo_id": spec.repo_id,
        "model_identity": spec.identity,
        "checkpoint_step": str(ckpt.get("step")),
        "checkpoint_loss": str(ckpt.get("loss")),
        "source_state_dict": "model_state_dict",
    }
    save_file(tensors, str(safetensors_path), metadata=metadata)
    del tensors, state_dict, ckpt
    gc.collect()

    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        saved_metadata = handle.metadata()
    return {
        "path": str(safetensors_path),
        "size": safetensors_path.stat().st_size,
        "keys": len(keys),
        "metadata": saved_metadata,
        "skipped_existing": False,
    }


def validate_basic_artifact(artifact_dir: Path) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(str(artifact_dir), trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), local_files_only=True)
    encoded = tokenizer.encode("The theorem states")
    expected = [464, 44728, 2585]
    if encoded != expected:
        raise RuntimeError(f"tokenizer mismatch for {artifact_dir}: {encoded} != {expected}")
    with safe_open(artifact_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if "model.embedding.weight" not in keys:
            raise RuntimeError(f"{artifact_dir} missing model.embedding.weight")
    return {
        "config_class": config.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "prompt_token_ids": encoded,
        "safetensors_keys": len(keys),
    }


def validate_automodel_load(artifact_dir: Path) -> dict[str, Any]:
    print(f"[{artifact_dir.name}] validating AutoModelForCausalLM local load", flush=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        str(artifact_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
        local_files_only=True,
    )
    missing = info.get("missing_keys") or []
    unexpected = info.get("unexpected_keys") or []
    mismatched = info.get("mismatched_keys") or []
    result = {
        "model_class": model.__class__.__name__,
        "model_class_module": model.__class__.__module__,
        "core_model_class": f"{model.model.__class__.__module__}.{model.model.__class__.__name__}",
        "param_count": int(sum(param.numel() for param in model.parameters())),
        "dtype_sample": str(next(model.parameters()).dtype),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "ok": not missing and not unexpected and not mismatched,
    }
    del model
    gc.collect()
    if not result["ok"]:
        raise RuntimeError(f"load key validation failed for {artifact_dir}: {result}")
    return result


def prepare_candidate(candidate: CandidateSpec, workdir: Path, force: bool) -> dict[str, Any]:
    spec = candidate.spec
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)

    parsed = parse_checkpoint_name(spec.checkpoint)
    stat = spec.checkpoint.stat()
    checkpoint = {
        "path": str(spec.checkpoint),
        "step": parsed["step"],
        "raw_checkpoint_loss": parsed["loss"],
        "raw_checkpoint_bpb": parsed["loss"] * BPB_PER_NAT,
        "mtime_utc": utc_mtime(spec.checkpoint),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(spec.checkpoint),
    }

    model_args = load_args(spec.checkpoint)
    out_dir = workdir / spec.key
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_meta = write_tokenizer_files(out_dir, model_args.get("tokenizer", "p50k_base"))
    config = build_config(spec, model_args, {"step": parsed["step"], "loss": parsed["loss"]}, tokenizer_meta)
    config.update(
        {
            "private_staging": False,
            "release_revision_name": "v0.3",
            "release_candidate": RELEASE_CANDIDATE,
            "local_validation_only": True,
            "source_checkpoint_path": str(spec.checkpoint),
            "source_checkpoint_sha256": checkpoint["sha256"],
            "paper_endpoint_step": candidate.paper_endpoint_step,
            "paper_endpoint_bpb": candidate.paper_endpoint_bpb,
            "selected_step_delta_vs_paper": parsed["step"] - candidate.paper_endpoint_step,
        }
    )
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "generation_config.json",
        {
            "bos_token_id": tokenizer_meta["eot_token"],
            "eos_token_id": tokenizer_meta["eot_token"],
            "pad_token_id": tokenizer_meta["eot_token"],
            "do_sample": False,
            "max_new_tokens": 32,
            "transformers_version": "4.57.3",
        },
    )
    out_dir.joinpath("configuration_ndm.py").write_text(CONFIGURATION_NDM)
    out_dir.joinpath("modeling_ndm.py").write_text(MODELING_NDM)
    out_dir.joinpath("requirements.txt").write_text(
        "\n".join(
            [
                "torch>=2.9",
                "transformers>=4.57",
                "safetensors>=0.7",
                "tiktoken>=0.12",
                "# Runtime must also install the matching ndm source checkout.",
                "",
            ]
        )
    )

    weights = convert_weights_local(spec, out_dir, force=force)
    if weights["metadata"].get("checkpoint_step") != str(parsed["step"]):
        raise RuntimeError(f"{spec.key} safetensors checkpoint step mismatch: {weights['metadata']}")
    write_local_model_card(out_dir, spec, checkpoint, weights)

    basic_validation = validate_basic_artifact(out_dir)
    automodel_validation = validate_automodel_load(out_dir)
    files = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())

    return {
        "key": spec.key,
        "identity": spec.identity,
        "repo_id": spec.repo_id,
        "artifact_dir": str(out_dir),
        "selection_note": candidate.selection_note,
        "paper_endpoint_step": candidate.paper_endpoint_step,
        "paper_endpoint_bpb": candidate.paper_endpoint_bpb,
        "ticket_as_of_step": candidate.ticket_as_of_step,
        "ticket_as_of_available_on_disk": candidate.ticket_as_of_step in retained_steps(candidate),
        "selected_step_delta_vs_paper": parsed["step"] - candidate.paper_endpoint_step,
        "selected_step_delta_vs_ticket_as_of": parsed["step"] - candidate.ticket_as_of_step,
        "retained_steps_on_disk": retained_steps(candidate),
        "checkpoint": checkpoint,
        "weights": weights,
        "local_validation": {
            "basic_artifact": basic_validation,
            "automodel_load": automodel_validation,
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--models", nargs="*", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "started_at_unix": time.time(),
        "workdir": str(args.workdir),
        "release_candidate": RELEASE_CANDIDATE,
        "bpb_per_nat": BPB_PER_NAT,
        "selection_rule": (
            "For each architecture, choose the retained on-disk checkpoint .pt "
            "nearest the committed V3 paper endpoint step (trail_100k convention, "
            "paper/main.typ + AS_OF.md, 2026-05-31T13:49:33Z). The selected "
            "checkpoint's trail_100k smoothed BPB rounds to the paper label. The "
            "ticket AS_OF steps are stale pre-recompute draft values and are not "
            "retained on disk; they are recorded for transparency only."
        ),
        "models": [],
    }

    for key in args.models:
        candidate = SPECS[key]
        print(f"[{key}] preparing local v0.3 artifact", flush=True)
        manifest["models"].append(prepare_candidate(candidate, args.workdir, force=args.force))

    manifest["finished_at_unix"] = time.time()
    manifest_path = args.workdir / "validation_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
