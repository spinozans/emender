#!/usr/bin/env python3
"""Prepare local v0.2 release-candidate HF artifacts.

This helper is intentionally local-only. It selects retained racer checkpoints,
records checkpoint/stat/hash/metric evidence, converts `model_state_dict` to
`model.safetensors`, writes the custom Transformers loader artifacts, and then
validates local `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
loading. It does not call Hugging Face upload, repo, branch, tag, or visibility
APIs.
"""

from __future__ import annotations

import argparse
import csv
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

BPB_PER_NAT = math.log2(math.e) / 3.918625
DEFAULT_WORKDIR = Path(f"/tmp/release-v02-local-hf-candidates-{os.environ.get('USER', 'agent')}")
RELEASE_CANDIDATE = "v0.2-rc-local"


@dataclass(frozen=True)
class CandidateSpec:
    spec: ModelSpec
    figure2_csv: Path
    selection_note: str


SPECS: dict[str, CandidateSpec] = {
    "e88": CandidateSpec(
        spec=ModelSpec(
            key="e88",
            repo_id="poietic-pbc/emender-e88-1.3b",
            identity="Emender/E88",
            short_name="Emender E88 1.3B",
            checkpoint=Path(
                "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
                "levelE88_1270M_20260511_233832/checkpoint_step_1395000_loss_2.6663.pt"
            ),
            smoke_label="e88",
            smoke_new_token_ids=[218, 218],
            smoke_new_text_repr="'\\x1e\\x1e'",
            smoke_param_count=1_273_191_856,
        ),
        figure2_csv=ROOT / "paper/results/figure_2/E88_NDM.csv",
        selection_note=(
            "Best exact 10K trailing BPB among retained E88 checkpoint files "
            "covered by the refreshed Figure 2 CSV."
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
                "levelfla-gdn_1270M_20260511_233832/checkpoint_step_1845000_loss_2.7198.pt"
            ),
            smoke_label="gdn",
            smoke_new_token_ids=[318, 318],
            smoke_new_text_repr="' is is'",
            smoke_param_count=1_352_352_498,
        ),
        figure2_csv=ROOT / "paper/results/figure_2/FLA_GDN.csv",
        selection_note=(
            "Best exact 10K trailing BPB among retained GDN checkpoint files "
            "covered by the refreshed Figure 2 CSV."
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
                "levelm2rnn_1270M_20260511_175023/checkpoint_step_1332000_loss_2.6762.pt"
            ),
            smoke_label="m2rnn",
            smoke_new_token_ids=[2109, 34059],
            smoke_new_text_repr="'........Officers'",
            smoke_param_count=1_307_101_140,
        ),
        figure2_csv=ROOT / "paper/results/figure_2/M2RNN_CMA.csv",
        selection_note=(
            "Best exact 10K trailing BPB among retained M2RNN-CMA checkpoint "
            "files covered by the refreshed Figure 2 CSV."
        ),
    ),
}


PUBLIC_V01: dict[str, dict[str, Any]] = {
    "e88": {
        "step": 1_281_000,
        "raw_loss": 2.6850,
        "raw_bpb": 0.988519,
        "paper_snapshot_bpb": 0.979277,
        "path": (
            "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
            "levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt"
        ),
        "sha256": "2ccb8851c798c5aa72ff0d6d45318496b6fbbc952c8379c7d56b23281ecedcfb",
    },
    "gdn": {
        "step": 1_686_000,
        "raw_loss": 2.6105,
        "raw_bpb": 0.961091,
        "paper_snapshot_bpb": 0.974841,
        "path": (
            "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/"
            "levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt"
        ),
        "sha256": "9e2b8baad914d9b7ab28f411fc65d37875fa269db3fc2f6e37503fd4c1730148",
    },
    "m2rnn": {
        "step": 1_212_000,
        "raw_loss": 2.6870,
        "raw_bpb": 0.989256,
        "paper_snapshot_bpb": 0.984356,
        "path": (
            "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/"
            "levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt"
        ),
        "sha256": "2ce9ada25d374c0bab7f20017d8ff5324a8583b8dc46bcd6180aefe923866197",
    },
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


def read_figure2_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nearest_metric(rows: list[dict[str, str]], step: int) -> dict[str, Any]:
    row = min(rows, key=lambda item: abs(int(item["step"]) - step))
    metric = {
        "figure2_step": int(row["step"]),
        "step_delta": int(row["step"]) - step,
        "raw_log_loss_at_figure2_step": float(row["loss"]),
        "trail_10k_loss": float(row["trail_10k"]),
        "trail_10k_bpb": float(row["trail_10k"]) * BPB_PER_NAT,
        "trail_50k_loss": float(row["trail_50k"]),
        "trail_50k_bpb": float(row["trail_50k"]) * BPB_PER_NAT,
        "trail_100k_loss": float(row["trail_100k"]),
        "trail_100k_bpb": float(row["trail_100k"]) * BPB_PER_NAT,
        "wallclock_h": float(row["wallclock_h"]),
    }
    return metric


def retained_checkpoint_metrics(candidate: CandidateSpec) -> list[dict[str, Any]]:
    rows = read_figure2_rows(candidate.figure2_csv)
    records = []
    for checkpoint in sorted(candidate.spec.checkpoint.parent.glob("checkpoint_step_*_loss_*.pt")):
        parsed = parse_checkpoint_name(checkpoint)
        metric = nearest_metric(rows, parsed["step"])
        exact = metric["step_delta"] == 0
        records.append(
            {
                "path": str(checkpoint),
                "step": parsed["step"],
                "raw_checkpoint_loss": parsed["loss"],
                "raw_checkpoint_bpb": parsed["loss"] * BPB_PER_NAT,
                "mtime_utc": utc_mtime(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "figure2_nearest": metric,
                "exact_figure2_step": exact,
            }
        )
    exact_records = [record for record in records if record["exact_figure2_step"]]
    ranked = sorted(exact_records, key=lambda record: record["figure2_nearest"]["trail_10k_bpb"])
    for rank, record in enumerate(ranked, start=1):
        record["exact_trail_10k_rank"] = rank
    return records


def write_local_model_card(out_dir: Path, spec: ModelSpec, config: dict[str, Any], checkpoint: dict[str, Any], weights: dict[str, Any]) -> None:
    text = f"""\
---
library_name: transformers
pipeline_tag: text-generation
tags:
- ndm
- emender
- local-validation
- v0.2-rc
---

# {spec.short_name} v0.2 local release candidate

This artifact directory was generated for local v0.2 release-candidate
validation only. It is not a public Hugging Face upload, does not modify `v0.1`,
and must not be published without a separate human approval gate.

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
The validation harness checks local `AutoModelForCausalLM.from_pretrained`
loading and Docker CPU/GPU generation before any public publish decision.
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
    tokenizer_meta = None
    out_dir = workdir / spec.key
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_meta = write_tokenizer_files(out_dir, model_args.get("tokenizer", "p50k_base"))
    config = build_config(spec, model_args, {"step": parsed["step"], "loss": parsed["loss"]}, tokenizer_meta)
    config.update(
        {
            "private_staging": False,
            "release_revision_name": "v0.2",
            "release_candidate": RELEASE_CANDIDATE,
            "local_validation_only": True,
            "source_checkpoint_path": str(spec.checkpoint),
            "source_checkpoint_sha256": checkpoint["sha256"],
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
    write_local_model_card(out_dir, spec, config, checkpoint, weights)

    figure2 = nearest_metric(read_figure2_rows(candidate.figure2_csv), parsed["step"])
    basic_validation = validate_basic_artifact(out_dir)
    automodel_validation = validate_automodel_load(out_dir)
    files = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())

    return {
        "key": spec.key,
        "identity": spec.identity,
        "repo_id": spec.repo_id,
        "artifact_dir": str(out_dir),
        "selection_note": candidate.selection_note,
        "checkpoint": checkpoint,
        "figure2_nearest": figure2,
        "public_v0_1": PUBLIC_V01[spec.key],
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
        "bpB_per_nat": BPB_PER_NAT,
        "selection_rule": (
            "For each model, choose the retained checkpoint with the best exact "
            "10K trailing BPB in the refreshed Figure 2 CSV, excluding checkpoint "
            "files whose steps are newer than the refreshed CSV tail to avoid "
            "extrapolating smoothed metrics."
        ),
        "models": [],
        "retained_candidates": {},
    }

    for key in args.models:
        candidate = SPECS[key]
        manifest["retained_candidates"][key] = retained_checkpoint_metrics(candidate)
        print(f"[{key}] preparing local v0.2 artifact", flush=True)
        manifest["models"].append(prepare_candidate(candidate, args.workdir, force=args.force))

    manifest["finished_at_unix"] = time.time()
    manifest_path = args.workdir / "validation_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
